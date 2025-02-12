#!/bin/bash
#SBATCH --job-name=run_quantum_circuit_gen_singlegpu
#SBATCH --time=00:30:00
#SBATCH --output=../logs/run_%A_%a.out
#SBATCH --error=../logs/run_%A_%a.err
#SBATCH --cpus-per-task=3
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu-debug

module purge
module load gcc cuda cmake openmpi
module load scicomp-python-env/2024-01
module load scicomp-llm-env

source ../.venv/bin/activate

pip install -r ../requirements.txt

uid="$(date +%Y%m%d_%H%M%S)"

python3 -u run_model.py
