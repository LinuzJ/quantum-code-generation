uid="$(date +%Y%m%d_%H%M%S)"

path="../generation/out/quantum_circuits_output_20250404_104520_Qwen2.5-Coder-3B-Instruct_few_shot.json"
out="./out/quantum_ciruits_validated_${uid}.json"
summary="./out/quantum_ciruits_summary_${uid}.json"

python3 src/evaluate_samples.py $path $out $summary