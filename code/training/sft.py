import logging
import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import transformers
from datasets import DatasetDict, concatenate_datasets, load_dataset
from trl import DataCollatorForCompletionOnlyLM, ModelConfig, SFTConfig, SFTTrainer

# ----- Config Logging and Warnings -----
warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_info()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen3-4B-Instruct-2507")
    block_size: int = field(default=10000)
    wandb_project: Optional[str] = field(default="sft-QASM-QWEN-4B")
    train_file_path: Optional[str] = field(
        default="Benyucong/graph-data-quantum-tokenized-4B_sft"
    )
    dagger: bool = field(default=False)

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project


def train():
    parser = transformers.HfArgumentParser((TrainingConfig, SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    args.report_to = ["wandb"]

    log_config = {**asdict(config), **asdict(args)}
    logging.info("Training config: %s", log_config)

    # ----- Load Model, Data and Tokenizer -----
    model_config = ModelConfig(
        model_name_or_path=config.model_name,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
    )
    dataset = load_dataset(config.train_file_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, use_fast=True
    )

    # ----- Setup Instruction Templates -----
    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant"
    special_tokens_dict = {"pad_token": "<|fim_pad_token|>"}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    args.dataset_text_field = "text"
    args.max_seq_length = config.block_size

    # ----- Setup Trainer -----
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset else dataset["train"],
        args=args,
        data_collator=collator,
    )

    # ----- Train Model -----
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
            "FULL_STATE_DICT"
        )

    trainer.save_model(output_dir=args.output_dir)
    


if __name__ == "__main__":
    train()
