from huggingface_hub import create_repo, upload_folder, login
import os

login(token=os.environ["HF_TOKEN"])

repo_id = "Benyucong/rl_quantum_circuit_gen_8B"   # change if you want a new name
create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

upload_folder(
    folder_path="/scratch/cs/adis/rl_quantum/global_step_210/huggingface",
    repo_id=repo_id,
    commit_message="update",
)
