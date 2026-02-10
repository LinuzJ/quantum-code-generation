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


def load_quasar_ground_truth_qubits(quasar_path: str) -> List[Optional[int]]:
	"""Load ground truth circuits and return their qubit counts."""
	with open(quasar_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	solution_qubits: List[Optional[int]] = []

	for idx, item in enumerate(data):
		gt_qasm = item.get("ground_truth", "")
		if not gt_qasm:
			solution_qubits.append(None)
			continue
		try:
			gt_circ = parse_qasm_from_str(gt_qasm)
			solution_qubits.append(gt_circ.num_qubits)
		except Exception:
			solution_qubits.append(None)

	return solution_qubits


def check_circuit_compiles(gen_qasm: str) -> bool:
	"""Check if a circuit can be compiled (parsed successfully)."""
	try:
		circ = parse_qasm_from_str(gen_qasm)
		return True
	except Exception:
		return False


def read_text_file(path: str) -> Optional[str]:
	try:
		with open(path, "r", encoding="utf-8") as f:
			return f.read()
	except Exception:
		return None


def main():
	denominator = 580  # fixed per request

	# Resolve paths relative to this file
	root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	gen_dir = os.path.join(root_dir, "generated_circuits")
	quasar_path = os.path.join(gen_dir, "quasar.json")
	sft_path = os.path.join(gen_dir, "sft.json")

	# Output
	out_dir = os.path.join(root_dir, "out_HQ")
	os.makedirs(out_dir, exist_ok=True)
	out_csv = os.path.join(out_dir, "compilability_table.csv")

	# Load ground truth qubit counts from quasar.json
	solution_qubits_list = load_quasar_ground_truth_qubits(quasar_path)

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

	# Initialize counts: model -> qubit_count -> (compiled_count, total_count)
	counts: Dict[str, Dict[int, Tuple[int, int]]] = {
		label: defaultdict(lambda: [0, 0]) for label in models.keys()
	}

	# Track all qubit numbers seen
	all_qubits = set()

	# Iterate over indices and check compilation
	for i in range(denominator):
		solution_qubits = solution_qubits_list[i] if i < len(solution_qubits_list) else None
		# If solution unavailable, skip comparisons for this index
		if solution_qubits is None:
			continue

		all_qubits.add(solution_qubits)

		# Folder-based models
		for label, cfg in models.items():
			kind = cfg["kind"]
			if kind == "folder":
				fpath = os.path.join(cfg["path"], cfg["pattern"].format(i=i))
				content = read_text_file(fpath)
				if not content:
					continue
				can_compile = check_circuit_compiles(content)
				# Track by the ground truth qubit count
				counts[label][solution_qubits][1] += 1  # total count
				if can_compile:
					counts[label][solution_qubits][0] += 1  # compiled count

	# Handle quasar.json entries: check if response compiles
	try:
		with open(quasar_path, "r", encoding="utf-8") as f:
			quasar_data = json.load(f)
	except Exception:
		quasar_data = []

	for i in range(min(denominator, len(quasar_data))):
		item = quasar_data[i]
		solution_qubits = solution_qubits_list[i]
		if solution_qubits is None:
			continue
		raw_response = item.get("response", "")
		if not raw_response:
			continue
		# Extract and sanitize response OPENQASM; remove newlines per request
		try:
			response_qasm = extract_openqasm_from_response(raw_response, remove_newlines=True)
		except Exception:
			continue
		can_compile = check_circuit_compiles(response_qasm)
		counts["quasar.json"][solution_qubits][1] += 1
		if can_compile:
			counts["quasar.json"][solution_qubits][0] += 1

	# Handle sft.json entries: check if response compiles
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
			solution_qubits = gt_circ.num_qubits
		except Exception:
			continue
		
		all_qubits.add(solution_qubits)
		
		raw_response = item.get("response", "")
		if not raw_response:
			continue
		try:
			response_qasm = extract_openqasm_from_response(raw_response, remove_newlines=True)
		except Exception:
			continue
		can_compile = check_circuit_compiles(response_qasm)
		counts["sft.json"][solution_qubits][1] += 1
		if can_compile:
			counts["sft.json"][solution_qubits][0] += 1

	# Write CSV: rows=models, columns=qubit numbers
	ordered_labels = ["deepseek", "gpt4o", "gpt5o", "quasar.json", "sft.json"]
	sorted_qubits = sorted(all_qubits)
	header = ["model"] + [str(q) for q in sorted_qubits]

	with open(out_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(header)
		for label in ordered_labels:
			row = [label]
			for q in sorted_qubits:
				compiled, total = counts[label][q]
				if total > 0:
					ratio = compiled / total
				else:
					ratio = 0.0
				row.append(ratio)
			writer.writerow(row)

	# Also print a quick summary
	print(f"Saved compilability table to {out_csv}")
	print(f"Qubit counts found: {sorted_qubits}")
	for label in ordered_labels:
		ratios = ", ".join(
			f"{q}:{counts[label][q][0]}/{counts[label][q][1]}" 
			for q in sorted_qubits
		)
		print(f"{label}: {ratios}")


if __name__ == "__main__":
	main()
