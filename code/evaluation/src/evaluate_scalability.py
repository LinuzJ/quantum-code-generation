import os
import json
import csv
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_qasm3_import import parse

from computations import compute_relative_entropy


# Parsing helpers (mirrors logic from evaluate_samples.parse_qasm_from_str)
ASSISTANT_START_STRING = "<|im_start|>assistant"
ASSISTANS_END_STRING = "<|im_end|>"


def parse_qasm_from_str(qasm_str: str) -> QuantumCircuit:
	potential_code_split = qasm_str.split("```")
	potential_code = potential_code_split[0].strip()

	if "OPENQASM 3.0" in potential_code:
		qasm_str = potential_code

	qasm_str = re.sub("`", "", qasm_str)

	if qasm_str.startswith(ASSISTANT_START_STRING):
		qasm_str = qasm_str[
			len(ASSISTANT_START_STRING) : len(qasm_str) - len(ASSISTANS_END_STRING)
		].strip()

	if "OPENQASM 3.0" not in qasm_str:
		raise ValueError("QASM code does not appear to be QASM 3.0.")

	try:
		circuit = parse(qasm_str)
	except Exception as e:
		raise ValueError(
			f"Failed to parse QASM code. It might be invalid QASM 3.0 code: {e}"
		) from e

	return circuit


def get_probabilities(circuit: QuantumCircuit, simulator: AerSimulator) -> List[float]:
	sim_circuit = circuit.remove_final_measurements(inplace=False)
	sim_circuit.save_statevector()
	result = simulator.run(sim_circuit).result()
	statevector = result.get_statevector(experiment=sim_circuit)
	return statevector.probabilities().tolist()


def extract_openqasm_from_response(text: str, remove_newlines: bool = True) -> str:
	"""Extract the OPENQASM 3.0 program from a free-form response.

	- Finds the first occurrence of "OPENQASM 3.0" and takes content until the next
	  triple backtick fence if present; otherwise to the end of the string.
	- Removes all backtick characters and optionally newline characters.
	"""
	if text is None:
		raise ValueError("Empty response text")

	start = text.find("OPENQASM 3.0")
	if start == -1:
		raise ValueError("OPENQASM 3.0 not found in response")
	# Cut from OPENQASM to the end or to the next code fence
	remainder = text[start:]
	fence_pos = remainder.find("```")
	if fence_pos != -1:
		qasm = remainder[:fence_pos]
	else:
		qasm = remainder

	qasm = qasm.replace("`", "")
	if remove_newlines:
		qasm = qasm.replace("\r", "").replace("\n", "")
	return qasm.strip()


def load_quasar_ground_truth(quasar_path: str, simulator: AerSimulator) -> Tuple[List[Optional[List[float]]], List[Optional[int]]]:
	"""Load ground truth circuits and return their probabilities and qubit counts."""
	with open(quasar_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	solution_probs: List[Optional[List[float]]] = []
	solution_qubits: List[Optional[int]] = []
	solution_depths: List[Optional[int]] = []
	solution_gates: List[Optional[int]] = []

	for idx, item in enumerate(data):
		gt_qasm = item.get("ground_truth", "")
		if not gt_qasm:
			solution_probs.append(None)
			solution_qubits.append(None)
			solution_depths.append(None)
			solution_gates.append(None)
			continue
		try:
			gt_circ = parse_qasm_from_str(gt_qasm)
			probs = get_probabilities(gt_circ, simulator)
			solution_probs.append(probs)
			solution_qubits.append(gt_circ.num_qubits)
			solution_depths.append(gt_circ.depth())
			solution_gates.append(gt_circ.size())
		except Exception:
			solution_probs.append(None)
			solution_qubits.append(None)
			solution_depths.append(None)
			solution_gates.append(None)

	return solution_probs, solution_qubits, solution_depths, solution_gates


def compare_circuit_to_solution(
	gen_qasm: str, solution_probs: List[float], simulator: AerSimulator
	) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
		"""Compare generated circuit to solution and return KL divergence, qubit count, depth, and gate count."""
		try:
			circ = parse_qasm_from_str(gen_qasm)
			gen_probs = get_probabilities(circ, simulator)
			kl = compute_relative_entropy(gen_probs, solution_probs)
			return float(kl), circ.num_qubits, circ.depth(), circ.size()
		except Exception:
			return None, None, None, None


def read_text_file(path: str) -> Optional[str]:
	try:
		with open(path, "r", encoding="utf-8") as f:
			return f.read()
	except Exception:
		return None


def main():
	# Threshold fixed at 0.1
	threshold = 0.8
	denominator = 580  # fixed per request

	# Resolve paths relative to this file
	root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	gen_dir = os.path.join(root_dir, "generated_circuits")
	quasar_path = os.path.join(gen_dir, "quasar.json")
	sft_path = os.path.join(gen_dir, "sft.json")

	# Output
	out_dir = os.path.join(root_dir, "out_HQ")
	os.makedirs(out_dir, exist_ok=True)
	out_csv = os.path.join(out_dir, "scalability_table.csv")

	simulator = AerSimulator(method="statevector")

	# Load ground truth once (per index) from quasar.json
	solution_probs_list, solution_qubits_list, solution_depths_list, solution_gates_list = load_quasar_ground_truth(quasar_path, simulator)

	# Prepare models and file patterns
	models = {
		"deepseek": {
			"kind": "folder",
			"path": os.path.join(gen_dir, "deepseek"),
			"pattern": "sample_{i}_gen1.qasm",
		},
		"gpt4o": {
			"kind": "folder",
			"path": os.path.join(gen_dir, "gpt4o"),
			"pattern": "sample_{i}.qasm",
		},
		# Label requested as gpt5o, folder is gpt5
		"gpt5o": {
			"kind": "folder",
			"path": os.path.join(gen_dir, "gpt5"),
			"pattern": "sample_{i}.qasm",
		},
		"quasar.json": {"kind": "quasar", "path": quasar_path},
		"sft.json": {"kind": "sft", "path": sft_path},
	}

	# Initialize counts: model -> metric -> (success_count, total_count)
	counts_qubits: Dict[str, Dict[int, Tuple[int, int]]] = {
		label: defaultdict(lambda: [0, 0]) for label in models.keys()
	}
	counts_depths: Dict[str, Dict[int, Tuple[int, int]]] = {
		label: defaultdict(lambda: [0, 0]) for label in models.keys()
	}
	counts_gates: Dict[str, Dict[int, Tuple[int, int]]] = {
		label: defaultdict(lambda: [0, 0]) for label in models.keys()
	}

	# Track all metric values seen
	all_qubits = set()
	all_depths = set()
	all_gates = set()

	# Iterate over indices and compute KL
	for i in range(denominator):
		solution_probs = solution_probs_list[i] if i < len(solution_probs_list) else None
		solution_qubits = solution_qubits_list[i] if i < len(solution_qubits_list) else None
		solution_depth = solution_depths_list[i] if i < len(solution_depths_list) else None
		solution_gates = solution_gates_list[i] if i < len(solution_gates_list) else None
		# If solution unavailable, skip comparisons for this index
		if solution_probs is None or solution_qubits is None or solution_depth is None or solution_gates is None:
			continue

		all_qubits.add(solution_qubits)
		all_depths.add(solution_depth)
		all_gates.add(solution_gates)

		# Folder-based models
		for label, cfg in models.items():
			kind = cfg["kind"]
			if kind == "folder":
				fpath = os.path.join(cfg["path"], cfg["pattern"].format(i=i))
				content = read_text_file(fpath)
				if not content:
					continue
				kl, num_qubits, num_depth, num_gates = compare_circuit_to_solution(content, solution_probs, simulator)
				if kl is None or num_qubits is None or num_depth is None or num_gates is None:
					continue
				# Track by the ground truth metrics (not the generated circuit's metrics)
				counts_qubits[label][solution_qubits][1] += 1  # total count
				counts_depths[label][solution_depth][1] += 1
				counts_gates[label][solution_gates][1] += 1
				if kl < threshold:
					counts_qubits[label][solution_qubits][0] += 1  # success count
					counts_depths[label][solution_depth][0] += 1
					counts_gates[label][solution_gates][0] += 1

	# Handle quasar.json entries: response vs ground_truth per index
	try:
		with open(quasar_path, "r", encoding="utf-8") as f:
			quasar_data = json.load(f)
	except Exception:
		quasar_data = []

	for i in range(min(denominator, len(quasar_data))):
		item = quasar_data[i]
		solution_probs = solution_probs_list[i]
		solution_qubits = solution_qubits_list[i]
		solution_depth = solution_depths_list[i]
		solution_gates = solution_gates_list[i]
		if solution_probs is None or solution_qubits is None or solution_depth is None or solution_gates is None:
			continue
		raw_response = item.get("response", "")
		if not raw_response:
			continue
		# Extract and sanitize response OPENQASM; remove newlines per request
		try:
			response_qasm = extract_openqasm_from_response(raw_response, remove_newlines=True)
		except Exception:
			continue
		kl, num_qubits, num_depth, num_gates = compare_circuit_to_solution(response_qasm, solution_probs, simulator)
		if kl is None or num_qubits is None or num_depth is None or num_gates is None:
			continue
		counts_qubits["quasar.json"][solution_qubits][1] += 1
		counts_depths["quasar.json"][solution_depth][1] += 1
		counts_gates["quasar.json"][solution_gates][1] += 1
		if kl < threshold:
			counts_qubits["quasar.json"][solution_qubits][0] += 1
			counts_depths["quasar.json"][solution_depth][0] += 1
			counts_gates["quasar.json"][solution_gates][0] += 1

	# Handle sft.json entries: response vs its own ground_truth per index
	try:
		with open(sft_path, "r", encoding="utf-8") as f:
			sft_data = json.load(f)
	except Exception:
		sft_data = []

	for i in range(min(denominator, len(sft_data))):
		item = sft_data[i]
		gt_qasm = item.get("ground_truth", "")
		if not gt_qasm:
			continue
		try:
			gt_circ = parse_qasm_from_str(gt_qasm)
			solution_probs = get_probabilities(gt_circ, simulator)
			solution_qubits = gt_circ.num_qubits
			solution_depth = gt_circ.depth()
			solution_gates = gt_circ.size()
		except Exception:
			continue
		all_qubits.add(solution_qubits)
		all_depths.add(solution_depth)
		all_gates.add(solution_gates)
		raw_response = item.get("response", "")
		if not raw_response:
			continue
		try:
			response_qasm = extract_openqasm_from_response(raw_response, remove_newlines=True)
		except Exception:
			continue
		kl, num_qubits, num_depth, num_gates = compare_circuit_to_solution(response_qasm, solution_probs, simulator)
		if kl is None or num_qubits is None or num_depth is None or num_gates is None:
			continue
		counts_qubits["sft.json"][solution_qubits][1] += 1
		counts_depths["sft.json"][solution_depth][1] += 1
		counts_gates["sft.json"][solution_gates][1] += 1
		if kl < threshold:
			counts_qubits["sft.json"][solution_qubits][0] += 1
			counts_depths["sft.json"][solution_depth][0] += 1
			counts_gates["sft.json"][solution_gates][0] += 1

	# Write CSVs: rows=models, columns=metric values
	ordered_labels = ["deepseek", "gpt4o", "gpt5o", "quasar.json", "sft.json"]

	# Qubit count CSV
	sorted_qubits = sorted(all_qubits)
	header_qubits = ["model"] + [str(q) for q in sorted_qubits]
	with open(out_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(header_qubits)
		for label in ordered_labels:
			row = [label]
			for q in sorted_qubits:
				success, total = counts_qubits[label][q]
				ratio = success / total if total > 0 else 0.0
				row.append(ratio)
			writer.writerow(row)

	# Depth CSV
	out_csv_depth = os.path.join(out_dir, "scalability_table_depth.csv")
	sorted_depths = sorted(all_depths)
	header_depths = ["model"] + [str(d) for d in sorted_depths]
	with open(out_csv_depth, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(header_depths)
		for label in ordered_labels:
			row = [label]
			for d in sorted_depths:
				success, total = counts_depths[label][d]
				ratio = success / total if total > 0 else 0.0
				row.append(ratio)
			writer.writerow(row)

	# Gate count CSV
	out_csv_gates = os.path.join(out_dir, "scalability_table_gates.csv")
	sorted_gates = sorted(all_gates)
	header_gates = ["model"] + [str(g) for g in sorted_gates]
	with open(out_csv_gates, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(header_gates)
		for label in ordered_labels:
			row = [label]
			for g in sorted_gates:
				success, total = counts_gates[label][g]
				ratio = success / total if total > 0 else 0.0
				row.append(ratio)
			writer.writerow(row)

	# Also print a quick summary
	print(f"Saved scalability table to {out_csv}")
	print(f"Saved depth table to {out_csv_depth}")
	print(f"Saved gate count table to {out_csv_gates}")
	print(f"Threshold: {threshold}")
	print(f"Qubit counts found: {sorted_qubits}")
	print(f"Depths found: {sorted_depths}")
	print(f"Gate counts found: {sorted_gates}")
	for label in ordered_labels:
		ratios_qubits = ", ".join(
			f"{q}:{counts_qubits[label][q][0]}/{counts_qubits[label][q][1]}" 
			for q in sorted_qubits
		)
		ratios_depths = ", ".join(
			f"{d}:{counts_depths[label][d][0]}/{counts_depths[label][d][1]}"
			for d in sorted_depths
		)
		ratios_gates = ", ".join(
			f"{g}:{counts_gates[label][g][0]}/{counts_gates[label][g][1]}"
			for g in sorted_gates
		)
		print(f"{label} (qubits): {ratios_qubits}")
		print(f"{label} (depth): {ratios_depths}")
		print(f"{label} (gates): {ratios_gates}")


if __name__ == "__main__":
	main()
