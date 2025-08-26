#!/bin/bash
#SBATCH --job-name=run_quantum_circuit_gen_singlegpu
#SBATCH --time=00:30:00
#SBATCH --output=../../logs/run_%A_%a.out
#SBATCH --error=../../logs/run_%A_%a.err
#SBATCH --cpus-per-task=3
#SBATCH --mem=20GB
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=linus.jern@aalto.fi
##SBATCH --partition=gpu-debug

module purge
module load gcc cuda cmake openmpi
module load scicomp-python-env/2024-01
module load scicomp-llm-env

export HF_HOME=$(pwd)/hf_cache

source .venv/bin/activate

pip install -r requirements.txt

model="20250218_214344"

uid="$(date +%Y%m%d_%H%M%S)"
model_path="../sft/data/checkpoints/${model}"

python3 run_model.py \
    --uid=${uid} \
    --model_path=${model_path}

