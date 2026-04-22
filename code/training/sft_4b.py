"""
This training script modifies the original TRL `SFTTrainer` example to avoid the
`CheckpointError: A different number of tensors was saved during the original
forward and recomputation` that occurs when using FSDP activation checkpointing.

The key changes are:

* **Disable KV caching:** Large language models from Hugging Face keep a
  `past_key_values` cache to speed up inference.  When activation
  checkpointing is enabled (either through `gradient_checkpointing` or
  FSDP activation checkpointing), this cache updates after the first forward
  call but is not recreated during recomputation.  This causes the number of
  saved tensors to differ between the original forward pass and the backward
  recomputation.  We load the model with `use_cache=False` to prevent this.

* **Use re‑entrant gradient checkpointing:** PyTorch offers two
  implementations of activation checkpointing controlled via the
  ``use_reentrant`` flag.  Empirically, enabling the re‑entrant variant
  resolves the tensor‑mismatch error when combined with FSDP.  We explicitly
  enable gradient checkpointing on the model with
  `use_reentrant=True`.

* **Load the model manually:** Instead of passing a string identifier into
  `SFTTrainer`, we load the `AutoModelForCausalLM` ourselves with the
  appropriate kwargs (dtype, attention implementation, and `use_cache=False`).
  This lets us configure gradient checkpointing and resize token embeddings
  after adding special tokens.

You can adjust the `fsdp_config.yaml` separately (for example, set
`fsdp_activation_checkpointing: true`) and run this script with the same
command‑line arguments as the original.  See the associated GitHub issue for
context on the checkpointing error.
"""

import logging
import warnings
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import transformers
import torch
from datasets import load_dataset
from trl import (
    DataCollatorForCompletionOnlyLM,
    ModelConfig,
    SFTConfig,
    SFTTrainer,
)

# Suppress noisy warnings and set up logging
warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_info()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class TrainingConfig:
    """Minimal configuration for the training script."""

    # Name of the model to fine tune
    model_name: str = field(default="Qwen/Qwen3-4B-Instruct-2507")
    # Maximum sequence length (block size)
    block_size: int = field(default=10000)
    wandb_project: Optional[str] = field(default="sft-quantum")
    # Path on the Hugging Face Hub or local filesystem to a dataset in JSON format
    train_file_path: Optional[str] = field(default="linuzj/graph-data-quantum-tokenized_sft")
    # A flag that isn't used in this script but can be extended for DAGGER training
    dagger: bool = field(default=False)
    
    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project


def train() -> None:
    """Load model, data and tokenizer, then train using SFTTrainer."""
    parser = transformers.HfArgumentParser((TrainingConfig, SFTConfig))
    # Parse command‑line arguments into two dataclasses
    config, args = parser.parse_args_into_dataclasses()
    # Report training progress to TensorBoard
    args.report_to = ["wandb"]
    args.logging_dir = "logs"

    # Log the merged configuration (TrainingConfig + SFTConfig)
    log_config = {**asdict(config), **asdict(args)}
    logging.info("Training config: %s", log_config)

    # ----- Load Model, Data and Tokenizer -----
    # Use TRL's ModelConfig for meta information; dtype and attention implementation
    model_config = ModelConfig(
        model_name_or_path=config.model_name,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
    )

    # Load the dataset from the specified path; returns a DatasetDict
    dataset = load_dataset(config.train_file_path)

    # Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    # ----- Setup Instruction Templates -----
    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant"

    # Add a pad token if it doesn't exist
    special_tokens_dict = {"pad_token": "<|fim_pad_token|>"}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Only compute loss over assistant responses
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    # Set dataset text field and maximum sequence length from config
    args.dataset_text_field = "text"
    args.max_seq_length = config.block_size

    # ----- Load and configure the model manually -----
    # Loading the model here lets us disable use_cache and enable re‑entrant checkpointing
    # Convert the string dtype ("bfloat16") to the actual torch dtype.  We import
    # torch directly instead of using transformers.utils.str_to_torch_dtype to
    # maintain compatibility with older versions of transformers.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype="bfloat16",
        attn_implementation=model_config.attn_implementation,
        use_cache=False,  # Disable KV caching during training (important for checkpointing)
    )
    # Resize embeddings to accommodate the newly added pad token
    model.resize_token_embeddings(len(tokenizer))
    # Enable gradient checkpointing with re‑entrant variant to avoid tensor mismatch
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    # ----- Setup Trainer -----
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset else dataset["train"],
        args=args,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # ----- Train Model -----
    trainer.train()
    trainer.accelerator.wait_for_everyone()

    # If FSDP is enabled, ensure we save full state dict
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
            "FULL_STATE_DICT",
            offload_to_cpu=True,
            rank0_only=True,
        )


    # Save the fine‑tuned model
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    train()