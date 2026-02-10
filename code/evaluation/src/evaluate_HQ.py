import os
import json
import csv
import re
from typing import Dict, List, Optional, Tuple

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


def load_quasar_ground_truth(quasar_path: str, simulator: AerSimulator) -> Tuple[List[Optional[List[float]]], List[Optional[QuantumCircuit]]]:
	with open(quasar_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	solution_probs: List[Optional[List[float]]] = []
	solution_circs: List[Optional[QuantumCircuit]] = []

	for idx, item in enumerate(data):
		gt_qasm = item.get("ground_truth", "")
		if not gt_qasm:
			solution_probs.append(None)
			solution_circs.append(None)
			continue
		try:
			gt_circ = parse_qasm_from_str(gt_qasm)
			probs = get_probabilities(gt_circ, simulator)
			solution_circs.append(gt_circ)
			solution_probs.append(probs)
		except Exception:
			solution_probs.append(None)
			solution_circs.append(None)

	return solution_probs, solution_circs


def compare_circuit_to_solution(
	gen_qasm: str, solution_probs: List[float], simulator: AerSimulator
) -> Optional[float]:
	try:
		circ = parse_qasm_from_str(gen_qasm)
		gen_probs = get_probabilities(circ, simulator)
		kl = compute_relative_entropy(gen_probs, solution_probs)
		return float(kl)
	except Exception:
		return None


def read_text_file(path: str) -> Optional[str]:
	try:
		with open(path, "r", encoding="utf-8") as f:
			return f.read()
	except Exception:
		return None


def main():
	# Thresholds and constants
	thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
	denominator = 580  # fixed per request

	# Resolve paths relative to this file
	root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	gen_dir = os.path.join(root_dir, "generated_circuits")
	quasar_path = os.path.join(gen_dir, "quasar.json")
	sft_path = os.path.join(gen_dir, "sft.json")

	# Output
	out_dir = os.path.join(root_dir, "out_HQ")
	os.makedirs(out_dir, exist_ok=True)
	out_csv = os.path.join(out_dir, "relative_entropy_table.csv")

	simulator = AerSimulator(method="statevector")

	# Load ground truth once (per index) from quasar.json
	solution_probs_list, _ = load_quasar_ground_truth(quasar_path, simulator)

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

	# Initialize counts
	counts: Dict[str, Dict[float, int]] = {
		label: {th: 0 for th in thresholds} for label in models.keys()
	}

	# Iterate over indices and compute KL
	for i in range(denominator):
		solution_probs = solution_probs_list[i] if i < len(solution_probs_list) else None
		# If solution unavailable, skip comparisons for this index
		if solution_probs is None:
			continue

		# Folder-based models
		for label, cfg in models.items():
			kind = cfg["kind"]
			if kind == "folder":
				fpath = os.path.join(cfg["path"], cfg["pattern"].format(i=i))
				content = read_text_file(fpath)
				if not content:
					continue
				kl = compare_circuit_to_solution(content, solution_probs, simulator)
				if kl is None:
					continue
				for th in thresholds:
					if kl < th:
						counts[label][th] += 1
			elif kind == "quasar":
				# Will handle below after loop since it needs the JSON item
				pass

	# Handle quasar.json entries: response vs ground_truth per index
	try:
		with open(quasar_path, "r", encoding="utf-8") as f:
			quasar_data = json.load(f)
	except Exception:
		quasar_data = []

	for i in range(min(denominator, len(quasar_data))):
		item = quasar_data[i]
		solution_probs = solution_probs_list[i]
		if solution_probs is None:
			continue
		raw_response = item.get("response", "")
		if not raw_response:
			continue
		# Extract and sanitize response OPENQASM; remove newlines per request
		try:
			response_qasm = extract_openqasm_from_response(raw_response, remove_newlines=True)
		except Exception:
			continue
		kl = compare_circuit_to_solution(response_qasm, solution_probs, simulator)
		if kl is None:
			continue
		for th in thresholds:
			if kl < th:
				counts["quasar.json"][th] += 1

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
		except Exception:
			continue
		raw_response = item.get("response", "")
		if not raw_response:
			continue
		try:
			response_qasm = extract_openqasm_from_response(raw_response, remove_newlines=True)
		except Exception:
			continue
		kl = compare_circuit_to_solution(response_qasm, solution_probs, simulator)
		if kl is None:
			continue
		for th in thresholds:
			if kl < th:
				counts["sft.json"][th] += 1

	# Write CSV: rows deepseek, gpt4o, gpt5o, quasar.json; columns thresholds
	ordered_labels = ["deepseek", "gpt4o", "gpt5o", "quasar.json", "sft.json"]
	header = ["model"] + [str(th) for th in thresholds]

	with open(out_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(header)
		for label in ordered_labels:
			row = [label]
			for th in thresholds:
				ratio = counts[label][th] / denominator
				row.append(ratio)
			writer.writerow(row)

	# Also print a quick summary
	print(f"Saved relative entropy table to {out_csv}")
	for label in ordered_labels:
		ratios = ", ".join(f"{th}:{counts[label][th]}/{denominator}" for th in thresholds)
		print(f"{label}: {ratios}")


if __name__ == "__main__":
	main()

