#!/bin/bash
#SBATCH --job-name=evaluate_samples_quantum_circuit
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --mem=50GB

module purge
module load scicomp-python-env/2024-01

source .venv/bin/activate


uid="$(date +%Y%m%d_%H%M%S)"


# ---- Inputs ----
# FIXED: added missing '/' after '..'
path="../generation/out/quantum_circuits_output_20250916_130538_sft_quantum_circuit_gen_8B_n10.json"
out_path="./out_pass_at_k/${uid}"

mkdir -p "$out_path" logs

filename="$(basename "$path")"
base="${filename%.json}"
# Strip leading prefix to get a compact base
base="${base#quantum_circuits_output_}"
# Model tag from the 3rd field onward (e.g., sft_quantum_circuit_gen_4B_n10)
model="$(echo "$base" | cut -d'_' -f3-)"

echo "[$(date)] Processing: $filename"
echo "[$(date)] Model tag : $model"
echo "[$(date)] Output dir: $out_path"

# Run evaluation with pass@10 (k=10)
python3 -u src/evaluate_pass@k.py \
  "$path" \
  "$out_path" \
  "$model" \
  --k 10

echo "[$(date)] Done."

