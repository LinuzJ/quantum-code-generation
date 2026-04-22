#!/bin/bash
#SBATCH --job-name=sft_qcg_multigpu
#SBATCH --time=2-00:00:00
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-141g-ellis

module purge
module load scicomp-python-env/2024-01
module load scicomp-llm-env

source .venv/bin/activate

export BASE_DIR="$(pwd)"
export WANDB_API_KEY="$(cat .wandb_api_key)"
export HF_TOKEN="${HF_TOKEN:-$(cat .hf_token 2>/dev/null || true)}"
export HF_HOME="${BASE_DIR}/hf"

python save_only_repack.py