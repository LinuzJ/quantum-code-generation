#!/bin/bash
#SBATCH --job-name=generate_samples_quantum_circuit_gen_singlegpu
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=20GB
#SBATCH --gpus=4
##SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --partition=gpu-h200-141g-ellis,gpu-h200-141g-short,gpu-a100-80g,gpu-h100-80g

module purge
module load gcc cuda cmake openmpi
module load scicomp-python-env/2024-01
module load scicomp-llm-env

source .venv/bin/activate


uid="$(date +%Y%m%d_%H%M%S)"
 

python3 generate_samples_pass@k.py \
  --uid run_p10_tp4 \
  --model_path Benyucong/sft_quantum_circuit_gen_8B \
  --dataset Benyucong/graph-data-quantum-tokenized_sft \
  --split test \
  --n_per_prompt 10 \
  --temperature 0.7 \
  --top_p 0.95 \
  --tensor_parallel_size 4 \
  --gpu_mem_util 0.92

