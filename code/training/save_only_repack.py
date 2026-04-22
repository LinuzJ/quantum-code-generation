# repack_fsdp_to_hf.py
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

modelpath="Qwen/Qwen3-4B-Instruct-2507"

model = AutoModelForCausalLM.from_pretrained(
    modelpath,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(modelpath)

import torch.distributed._shard.checkpoint as dist_cp
state_dict = {
        "model": model.state_dict()
    }

distcp_checkpoint_path = "/scratch/cs/adis/yuc10/quantum-code-generation/code/training/Benyucong/quantum-circuit-qubo-4B/checkpoint-26100/pytorch_model_fsdp_0"

dist_cp.load(state_dict=state_dict, storage_reader=dist_cp.FileSystemReader(distcp_checkpoint_path), no_dist=True)

model.load_state_dict(state_dict["model"])
model.save_pretrained("/scratch/cs/adis/yuc10/quantum-code-generation/code/training/Benyucong/quantum-circuit-qubo-4B/saved_model")