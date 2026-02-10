import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_qasm3_import import parse
from qiskit import QuantumCircuit

from util import construct_qiskit_hamiltonian


def load_ground_truth(ground_truth_json: str) -> Tuple[Dict[int, Dict[str, str]], List[Dict]]:
	"""Load ground truth circuits and Hamiltonians indexed by problem index.

	Returns tuple (ground_truth_map, full_entries) where ground_truth_map maps index->info
	and full_entries retains original entries for optional response parsing.
	"""
	with open(ground_truth_json, "r", encoding="utf-8") as f:
		data = json.load(f)

	gt: Dict[int, Dict[str, str]] = {}
	for entry in data:
		extra = entry.get("extra_info", {})
		idx = extra.get("index")
		if idx is None:
			continue
		hamiltonian = extra.get("cost_hamiltonian")
		ground_truth_qasm = entry.get("ground_truth")
		if hamiltonian and ground_truth_qasm:
			gt[idx] = {
				"hamiltonian": hamiltonian,
				"ground_truth_qasm": ground_truth_qasm,
			}
	return gt, data


def compute_expectation_value(qasm_code: str, hamiltonian_expr: str, simulator: AerSimulator) -> float:
	"""Parse QASM 3.0 code, simulate statevector, compute expectation value for Hamiltonian."""
	circuit: QuantumCircuit = parse(qasm_code)
	sim_circuit = circuit.remove_final_measurements(inplace=False)
	sim_circuit.save_statevector()
	result = simulator.run(sim_circuit).result()
	statevector = result.get_statevector(experiment=sim_circuit)
	oper = construct_qiskit_hamiltonian(hamiltonian_expr)
	value = statevector.expectation_value(oper)
	return float(np.real(value))


def find_generated_qasm(model_dir: str, index: int) -> Tuple[str, str]:
	"""Return path and content of generated circuit for given index if exists.

	For deepseek we expect pattern sample_{i}_gen1.qasm, otherwise sample_{i}.qasm.
	"""
	path_plain = os.path.join(model_dir, f"sample_{index}.qasm")
	path_gen1 = os.path.join(model_dir, f"sample_{index}_gen1.qasm")
	if os.path.isfile(path_plain):
		with open(path_plain, "r", encoding="utf-8") as f:
			return path_plain, f.read()
	if os.path.isfile(path_gen1):
		with open(path_gen1, "r", encoding="utf-8") as f:
			return path_gen1, f.read()
	return "", ""


def evaluate_models(generated_root: str, ground_truth: Dict[int, Dict[str, str]]):
	simulator = AerSimulator(method="statevector")

	# Precompute ground truth expectation values
	gt_expectations: Dict[int, float] = {}
	for idx, info in ground_truth.items():
		try:
			gt_expectations[idx] = compute_expectation_value(
				info["ground_truth_qasm"], info["hamiltonian"], simulator
			)
		except Exception:
			# Skip indices that fail to parse/simulate
			continue

	model_dirs = [d for d in os.listdir(generated_root) if os.path.isdir(os.path.join(generated_root, d))]
	results: List[Dict[str, object]] = []

	for model in sorted(model_dirs):
		model_path = os.path.join(generated_root, model)
		for idx in ground_truth.keys():
			file_path, gen_qasm = find_generated_qasm(model_path, idx)
			if not gen_qasm:
				continue  # no generated circuit for this index
			if idx not in gt_expectations:
				continue  # ground truth expectation unavailable
			h_expr = ground_truth[idx]["hamiltonian"]
			try:
				gen_exp = compute_expectation_value(gen_qasm, h_expr, simulator)
				sol_exp = gt_expectations[idx]
				delta = gen_exp - sol_exp
				abs_delta = abs(delta)
				results.append(
					{
						"model": model,
						"index": idx,
						"generated_expectation_value": gen_exp,
						"solution_expectation_value": sol_exp,
						"delta_expectation_value": delta,
						"abs_delta_expectation_value": abs_delta,
						"file_path": file_path,
					}
				)
			except Exception:
				continue
	return results


QASM_BLOCK_RE = re.compile(
	r"OPENQASM\s+3.0;.*?(?:\n|\r)(?:.*?)(c\[[0-9]+\]\s*=\s*measure.*?;.*?)+",
	re.DOTALL,
)


def extract_first_qasm_block(text: str) -> Optional[str]:
	"""Attempt to extract the first plausible QASM 3.0 block from a free-form response string.

	Strategy:
	- Find 'OPENQASM 3.0;' substring
	- Truncate at first triple backticks ``` if present (common Markdown fence)
	- Ensure at least one measurement line 'measure q['
	- Return cleaned block or None
	"""
	if "OPENQASM" not in text:
		return None
	# Remove trailing code fences or output markers
	truncated = text.split("```", 1)[0]
	# Heuristic: start at first occurrence of OPENQASM 3.0;
	idx = truncated.find("OPENQASM 3.0;")
	if idx == -1:
		return None
	candidate = truncated[idx:]
	# Stop at double newline followed by non-QASM chatter maybe — keep simple
	# Ensure measurement appears
	if "measure" not in candidate:
		return None
	# Basic sanity: include stdgates or bit/ qubit declarations
	if "qubit" not in candidate:
		return None
	return candidate.strip()


def evaluate_json_responses(label: str, entries: List[Dict], ground_truth: Dict[int, Dict[str, str]], gt_expectations: Dict[int, float], simulator: AerSimulator):
	results: List[Dict[str, object]] = []
	for entry in entries:
		extra = entry.get("extra_info", {})
		idx = extra.get("index")
		if idx is None or idx not in ground_truth or idx not in gt_expectations:
			continue
		response_text = entry.get("response") or ""
		qasm_block = extract_first_qasm_block(response_text)
		if not qasm_block:
			continue
		h_expr = ground_truth[idx]["hamiltonian"]
		try:
			gen_exp = compute_expectation_value(qasm_block, h_expr, simulator)
			sol_exp = gt_expectations[idx]
			delta = gen_exp - sol_exp
			results.append(
				{
					"model": label,
					"index": idx,
					"generated_expectation_value": gen_exp,
					"solution_expectation_value": sol_exp,
					"delta_expectation_value": delta,
					"abs_delta_expectation_value": abs(delta),
					"file_path": f"{label}:response_index_{idx}",
				}
			)
		except Exception:
			continue
	return results


def summarize(results: List[Dict[str, object]]):
	summary: Dict[str, Dict[str, float]] = {}
	by_model: Dict[str, List[float]] = {}
	for r in results:
		by_model.setdefault(r["model"], []).append(r["abs_delta_expectation_value"])
	for model, vals in by_model.items():
		arr = np.array(vals)
		summary[model] = {
			"count": int(arr.size),
			"mean_abs_delta": float(arr.mean()),
			"median_abs_delta": float(np.median(arr)),
			"std_abs_delta": float(arr.std()),
			"min_abs_delta": float(arr.min()),
			"max_abs_delta": float(arr.max()),
		}
	return summary


def save_csv(results: List[Dict[str, object]], out_csv: str):
	import csv

	headers = [
		"model",
		"index",
		"generated_expectation_value",
		"solution_expectation_value",
		"delta_expectation_value",
		"abs_delta_expectation_value",
		"file_path",
	]
	with open(out_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=headers)
		writer.writeheader()
		for row in results:
			writer.writerow(row)


def save_csv_abs_only(results: List[Dict[str, object]], out_csv: str):
	import csv
	headers = ["model", "index", "abs_delta_expectation_value"]
	with open(out_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=headers)
		writer.writeheader()
		for row in results:
			writer.writerow({
				"model": row["model"],
				"index": row["index"],
				"abs_delta_expectation_value": row["abs_delta_expectation_value"],
			})


def save_summary(summary: Dict[str, Dict[str, float]], out_json: str):
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)


def main():
	parser = argparse.ArgumentParser(description="Compute absolute expectation value differences for generated circuits.")
	parser.add_argument("--generated-root", required=True, help="Root directory containing model subdirectories with generated QASM files.")
	parser.add_argument("--ground-truth-json", required=True, help="Path to ground truth JSON (e.g., quasar.json).")
	parser.add_argument("--additional-json", nargs="*", default=[], help="Optional extra JSON files whose response fields should be parsed as candidate circuits.")
	parser.add_argument("--include-json-responses", action="store_true", help="If set, parse QASM from response fields of the provided JSON(s) and include as additional models.")
	parser.add_argument("--json-only", action="store_true", help="Only compute abs delta for JSON response circuits (no directory scans) and output minimal CSV.")
	parser.add_argument("--out-dir", required=True, help="Directory to write CSV and summary JSON.")
	parser.add_argument("--csv-name", default="delta_expectation_values.csv", help="Output CSV filename.")
	parser.add_argument("--summary-name", default="delta_expectation_values_summary.json", help="Output summary JSON filename.")
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)

	ground_truth, primary_entries = load_ground_truth(args.ground_truth_json)
	if args.json_only:
		results: List[Dict[str, object]] = []
	else:
		results = evaluate_models(args.generated_root, ground_truth)

	# Build ground truth expectation cache once for reuse (reuse logic from evaluate_models simplified here)
	simulator = AerSimulator(method="statevector")
	gt_expectations: Dict[int, float] = {}
	for idx, info in ground_truth.items():
		try:
			gt_expectations[idx] = compute_expectation_value(info["ground_truth_qasm"], info["hamiltonian"], simulator)
		except Exception:
			continue

	# In json-only mode we force include JSON responses
	if args.include_json_responses or args.json_only:
		# Include primary ground truth JSON responses as one model label
		json_results = evaluate_json_responses(label=os.path.splitext(os.path.basename(args.ground_truth_json))[0] + "_response", entries=primary_entries, ground_truth=ground_truth, gt_expectations=gt_expectations, simulator=simulator)
		results.extend(json_results)
		# Additional JSONs
		for add_path in args.additional_json:
			if not os.path.isfile(add_path):
				continue
			gt_map, entries = load_ground_truth(add_path)
			# Must share indices with original ground truth to compare expectation values; we reuse existing gt_expectations
			add_label = os.path.splitext(os.path.basename(add_path))[0] + "_response"
			add_results = evaluate_json_responses(label=add_label, entries=entries, ground_truth=ground_truth, gt_expectations=gt_expectations, simulator=simulator)
			results.extend(add_results)

	summary = summarize(results)

	out_csv = os.path.join(args.out_dir, args.csv_name)
	out_json = os.path.join(args.out_dir, args.summary_name)
	if args.json_only:
		save_csv_abs_only(results, out_csv)
	else:
		save_csv(results, out_csv)
	save_summary(summary, out_json)

	print(f"Wrote per-sample delta expectation values to: {out_csv}")
	print(f"Wrote summary statistics to: {out_json}")
	print("Summary:")
	for model, stats in summary.items():
		print(model, stats)


if __name__ == "__main__":
	main()

