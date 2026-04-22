from transformers import AutoModelForCausalLM
import torch

# Load your existing 4-shard model folder (clone/download first if needed)
m = AutoModelForCausalLM.from_pretrained("Benyucong/sft_quantum_circuit_gen_3B", device_map="cpu")

m.to(torch.bfloat16)            # optional: or .half() for fp16
m.config.torch_dtype = "bfloat16"

# Repack to a new folder; this writes a fresh, correct index
m.save_pretrained(
    "repacked",
    safe_serialization=True,
    max_shard_size="5GB",   # or 2GB/8GB as you like
)
