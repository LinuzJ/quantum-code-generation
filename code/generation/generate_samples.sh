#!/bin/bash
#SBATCH --job-name=generate_samples_quantum_circuit_gen_singlegpu
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=20GB
#SBATCH --gpus=1
##SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --partition=gpu-h200-141g-ellis,gpu-h200-141g-short,gpu-a100-80g,gpu-h100-80g

module purge
module load gcc cuda cmake openmpi
module load scicomp-python-env/2024-01
module load scicomp-llm-env

source .venv/bin/activate

export HF_HOME="$PWD/hf_cache"

uid="$(date +%Y%m%d_%H%M%S)"

n_samples=580

# model_path="Benyucong/rl_quantum_4b"
# model_path="linuzj/quantum-circuit-qubo-3B"
model_path="Benyucong/sft_quantum_circuit_gen_4B"
# dataset="linuzj/graph-data-quantum-tokenized_sft"
dataset="Benyucong/graph-data-quantum-tokenized-4B_sft"

python3 -u generate_samples_pass@k.py \
    --uid=${uid} \
    --model_path=${model_path} \
    --n_samples=${n_samples} \
    --dataset=${dataset}

