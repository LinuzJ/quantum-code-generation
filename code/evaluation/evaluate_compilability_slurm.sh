#!/bin/bash
#SBATCH --job-name=evaluate_compilability
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40GB

module purge
module load scicomp-python-env/2024-01

# Activate local venv if present
if [ -d .venv ]; then
  source .venv/bin/activate
fi

# Ensure deps are present

mkdir -p ./out_HQ 

# Run the evaluator (no args required)
python3 -u src/evaluate_compilability.py
