#!/bin/bash
#SBATCH --job-name=evaluate_samples_quantum_circuit
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --mem=50GB

module purge
module load scicomp-python-env/2024-01

source .venv/bin/activate


uid="$(date +%Y%m%d_%H%M%S)"


path="../generation/out/quantum_circuits_output_20250826_212403_sft_quantum_circuit_gen_4B.json"
out_path="./out"

filename=$(basename "$path")
base="${filename%.json}"
base="${base#quantum_circuits_output_}"
model=$(echo "$base" | cut -d'_' -f3-)

echo "Processing $filename with model: $model"

python3 -u src/evaluate_samples.py $path $out_path $model

