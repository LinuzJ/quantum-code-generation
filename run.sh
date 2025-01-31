#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --output=logs/part_%A_%a.out
#SBATCH --error=logs/part_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-12
#SBATCH --partition=gpu-h100-80g

module purge
module load gcc cuda cmake openmpi
module load scicomp-python-env/2024-01
module load scicomp-llm-env

source /scratch/work/jernl1/quantum-code-generation/.venv/bin/activate

mkdir -p logs

python3 -u src/synthetic_data_generation.py \
    --total-parts 10 \
    --current-part $SLURM_ARRAY_TASK_ID
