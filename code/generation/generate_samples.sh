#!/bin/bash
#SBATCH --job-name=generate_samples_quantum_circuit_gen_singlegpu
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=20GB
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-141g-short

module purge
module load gcc cuda cmake openmpi
module load scicomp-python-env/2024-01
module load scicomp-llm-env

source .venv/bin/activate

pip install --upgrade -r requirements.txt

uid="$(date +%Y%m%d_%H%M%S)"

n_samples=580

model_path="Benyucong/quantum_3b"
dataset="linuzj/graph-data-quantum-tokenized_sft"  

python3 -u generate_samples_vllm.py \
    --uid=${uid} \
    --model_path=${model_path} \
    --n_samples=${n_samples} \
    --dataset=${dataset}

