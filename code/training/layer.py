# save as probe_layer.py and run: python probe_layer.py
from transformers import AutoModelForCausalLM
import inspect

MODEL_ID = "Qwen/Qwen3-8B"

m = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

# Try common module paths
candidates = [
    "model.layers",             # Qwen2/3 style
    "model.decoder.layers",     # some decoder-based models
    "transformer.h",            # GPT-NeoX style
    "model.layer",              # BLOOM style
]

layer_type = None
for path in candidates:
    try:
        obj = m
        for part in path.split("."):
            obj = getattr(obj, part)
        first = obj[0]
        layer_type = type(first).__name__
        print(f"Found layers at {path} -> {layer_type}")
        break
    except Exception:
        pass

if not layer_type:
    # Last resort: scan attributes for a list of modules
    for name, val in m.__dict__.items():
        if hasattr(val, "__getitem__") and len(val) > 0:
            try:
                tname = type(val[0]).__name__
                if "Layer" in tname or "Block" in tname or "Decoder" in tname:
                    print(f"Guessed via scan: {name}[0] -> {tname}")
                    layer_type = tname
                    break
            except Exception:
                pass

print("Layer class:", layer_type)
