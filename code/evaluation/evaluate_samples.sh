uid="$(date +%Y%m%d_%H%M%S)"

path="../generation/out/quantum_circuits_output_20250807_104506_quantum_3b.json"
model="quantum_circuit"
out_path="./out"

python3 src/evaluate_samples.py $path $out_path $model