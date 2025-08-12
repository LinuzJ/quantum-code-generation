import logging
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

# --- Logging / warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_info()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen3-8B"
    block_size: int = 12288
    wandb_project: Optional[str] = "quantum-circuit-generation"
    train_file_path: str = "linuzj/graph-data-quantum-tokenized_sft"

    def __post_init__(self):
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project


def _maybe_get_resume_ckpt(output_dir: str) -> Optional[str]:
    """Return last checkpoint path if it exists; otherwise None."""
    if not output_dir or not os.path.isdir(output_dir):
        return None
    try:
        return get_last_checkpoint(output_dir)
    except Exception:
        return None


def train():
    parser = transformers.HfArgumentParser((TrainingConfig, SFTConfig))
    config, args = parser.parse_args_into_dataclasses()

    # Explicit reporting to silence v5 warning (and make intent clear)
    args.report_to = ["wandb"]
    args.dataset_text_field = "text"
    args.max_seq_length = config.block_size

    # Require absolute, shared output_dir (e.g., $SCRATCH/...)
    if not args.output_dir or not os.path.isabs(args.output_dir):
        raise ValueError("--output_dir must be an absolute, shared path (e.g., $SCRATCH/...).")

    # DDP/accelerate state for rank-safe mkdir + barriers
    try:
        from accelerate import PartialState
        state = PartialState()
    except Exception:
        state = None

    logging.info("Parsed config: %s", {**asdict(config), **asdict(args)})

    # Create output dir on rank 0, then barrier
    if state is None or state.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    if state is not None:
        state.wait_for_everyone()

    # ----- Data & tokenizer -----
    dataset = load_dataset(config.train_file_path)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    # Use EOS as pad if missing (avoids embedding resize warnings)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|im_start|>user",
        response_template="<|im_start|>assistant",
        tokenizer=tokenizer,
        mlm=False,
    )

    # ----- Load model (robust across transformers versions) -----
    common_kwargs = dict(
        pretrained_model_name_or_path=config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,  # Qwen models often require this
    )

    try:
        # Newer Transformers accept attn_implementation here
        model = AutoModelForCausalLM.from_pretrained(
            attn_implementation="flash_attention_2",
            **common_kwargs,
        )
    except TypeError:
        # Older Transformers: fall back without the kwarg
        model = AutoModelForCausalLM.from_pretrained(**common_kwargs)
        # Best-effort set on config if present
        for key in ("attn_implementation", "_attn_implementation"):
            if hasattr(model.config, key):
                setattr(model.config, key, "flash_attention_2")
                break

    # Avoid KV cache during training (mandatory with gradient ckpt)
    model.config.use_cache = False

    # Prefer non-reentrant checkpointing to avoid metadata mismatches
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            # older transformers: no kwargs support
            model.gradient_checkpointing_enable()

    # ----- Trainer -----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset else dataset["train"],
        data_collator=collator,
        args=args,
    )

    # ----- Safe resume -----
    resume_ckpt = _maybe_get_resume_ckpt(args.output_dir)
    if state is None or state.is_main_process:
        logging.info("Resume checkpoint: %s", resume_ckpt if resume_ckpt else "None (fresh run)")
    if state is not None:
        state.wait_for_everyone()

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.accelerator.wait_for_everyone()

    # Ensure full-state saves under FSDP
    if getattr(trainer, "is_fsdp_enabled", False):
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    train()
