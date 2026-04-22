from huggingface_hub import HfApi
import os, pathlib

REPO_ID = "Benyucong/sft_quantum_circuit_gen_3B"
FOLDER = "repacked"  # the folder you created with save_pretrained(...)

api = HfApi(token=os.getenv("HF_TOKEN"))

# 1) (Optional but wise) remove old shards & the old index so there’s no mix
#    List existing files and delete the model shards/index only.
existing = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
to_delete = [
    f for f in existing
    if f.startswith("model-") or
       f.startswith("pytorch_model-") or
       f in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
]
if to_delete:
    api.delete_file(
        path_in_repo=to_delete[0],
        repo_id=REPO_ID, repo_type="model",
        commit_message="Clean old shards/index before repack upload"
    )
    for f in to_delete[1:]:
        api.delete_file(
            path_in_repo=f, repo_id=REPO_ID, repo_type="model",
            commit_message="Clean old shards/index before repack upload"
        )

# 2) Upload the freshly repacked folder
api.upload_folder(
    folder_path=FOLDER,
    repo_id=REPO_ID,
    repo_type="model",
    allow_patterns=["*.json", "*.safetensors", "*.md"],
    commit_message="Repacked weights: regenerate index to fix total_parameters",
)
