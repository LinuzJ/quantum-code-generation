#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
import time
import os
from math import comb
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_qasm3_import import parse

from computations import compute_relative_entropy
from util import construct_qiskit_hamiltonian

# ---------------------- Constants / thresholds ----------------------
ASSISTANT_START_STRING = "<|im_start|>assistant"
ASSISTANT_END_STRING = "<|im_end|>"

RANDOM_SAMPLING_AMOUNT = 1000
EXPECTATION_VALUE_DIFFERENCE_THRESHOLD = 0.2


# ----------------------------- Utils ------------------------------
def int_to_bitstring(i: int, n_qubits: int) -> str:
    return format(i, "0" + str(n_qubits) + "b")


def get_probability_distribution_and_expectation_value(
    circuit: QuantumCircuit, simulator: AerSimulator, hamiltonian: str
) -> Tuple[List[float], complex]:
    """Return (probabilities, expectation_value) for a circuit and Hamiltonian."""
    sim_circuit = circuit.remove_final_measurements(inplace=False)
    sim_circuit.save_statevector()
    result = simulator.run(sim_circuit).result()
    statevector: Statevector = result.get_statevector(experiment=sim_circuit)
    probs = statevector.probabilities().tolist()
    oper = construct_qiskit_hamiltonian(hamiltonian)
    expectation_value = statevector.expectation_value(oper)
    return probs, expectation_value


def evaluate_qiskit_circuit(
    circuit: QuantumCircuit, hamiltonian: str, simulator: AerSimulator
) -> Tuple[List[float], complex, str]:
    probs, expectation_value = get_probability_distribution_and_expectation_value(
        circuit, simulator, hamiltonian
    )
    most_probable_state_index = int(np.argmax(probs))
    bitstring = int_to_bitstring(most_probable_state_index, circuit.num_qubits)
    return probs, expectation_value, bitstring


def is_most_probable_state_correct(sample: Dict[str, Any], most_probable_state: str) -> bool:
    bitstrings = sample["dataset_metrics"]["solution"]["bitstrings"]
    return most_probable_state in bitstrings


def parse_qasm_from_str(qasm_str: str) -> QuantumCircuit:
    """Robustly parse a QASM 3.0 string, trimming chat wrappers / code fences."""
    # If triple fences exist, prefer the first fenced chunk that contains OPENQASM
    if "```" in qasm_str:
        parts = qasm_str.split("```")
        # try the first chunk that looks like QASM
        for chunk in parts:
            if "OPENQASM 3.0" in chunk:
                qasm_str = chunk
                break

    # Remove stray backticks
    qasm_str = re.sub(r"`", "", qasm_str).strip()

    # Trim assistant role wrappers if present
    if qasm_str.startswith(ASSISTANT_START_STRING) and qasm_str.endswith(ASSISTANT_END_STRING):
        qasm_str = qasm_str[len(ASSISTANT_START_STRING) : len(qasm_str) - len(ASSISTANT_END_STRING)].strip()

    if "OPENQASM 3.0" not in qasm_str:
        raise ValueError("QASM code does not appear to be QASM 3.0.")

    try:
        circuit = parse(qasm_str)
    except Exception as e:
        raise ValueError(f"Failed to parse QASM 3.0: {e}") from e
    return circuit


def randomize_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Create a randomized-parameter variant of a circuit (structure preserved)."""
    new_circ = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    for instr in circuit.data:
        op = instr.operation
        qubits = instr.qubits
        clbits = instr.clbits
        # Clone to mutable if available
        mutable_op = getattr(op, "to_mutable", lambda: op)()
        if getattr(mutable_op, "params", None):
            mutable_op.params = [np.random.uniform(0, 2 * np.pi) for _ in mutable_op.params]
        new_circ.append(mutable_op, qubits, clbits)
    return new_circ


def compare_solution(
    sim_probs: List[float],
    solution_probs: List[float],
    expectation_value: complex,
    solution_expectation_value: complex,
    hamiltonian: str,
    circuit: QuantumCircuit,
    simulator: AerSimulator,
) -> Dict[str, float]:
    """Compare generated vs solution: relative entropy + EV metrics (+ randomized baseline)."""
    relative_entropy = compute_relative_entropy(sim_probs, solution_probs)
    ev_diff_signed = float(np.real(expectation_value - solution_expectation_value))
    ev_diff_abs = float(abs(ev_diff_signed))

    cumulative_random_entropy = 0.0
    cumulative_expectation_value = 0.0
    if RANDOM_SAMPLING_AMOUNT > 0:
        for _ in range(RANDOM_SAMPLING_AMOUNT):
            randomized_circuit = randomize_circuit(circuit)
            randomized_probs, randomized_expectation_value = get_probability_distribution_and_expectation_value(
                randomized_circuit, simulator, hamiltonian
            )
            cumulative_random_entropy += compute_relative_entropy(randomized_probs, solution_probs)
            cumulative_expectation_value += randomized_expectation_value

        cumulative_expectation_value /= RANDOM_SAMPLING_AMOUNT
        random_relative_entropy = cumulative_random_entropy / RANDOM_SAMPLING_AMOUNT
    else:
        cumulative_expectation_value = float("nan")
        random_relative_entropy = float("nan")

    return {
        "relative_entropy": float(relative_entropy),
        "random_relative_entropy": float(random_relative_entropy),
        "solution_expectation_value": float(np.real(solution_expectation_value)),
        "generated_expectation_value": float(np.real(expectation_value)),
        "randomized_expectation_value": float(np.real(cumulative_expectation_value)),
        "expectation_value_difference": ev_diff_signed,      # keep signed for diagnostics
        "expectation_value_difference_abs": ev_diff_abs,     # used for thresholding
    }


def pass_at_k(c: int, n: int, k: int) -> float:
    """Unbiased HumanEval estimator: 1 - C(n-c, k) / C(n, k)."""
    if n <= 0 or k <= 0 or c <= 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - (comb(n - c, k) / comb(n, k))


# ------------------------ Core evaluation -------------------------
def _load_results(json_file: str) -> List[Dict[str, Any]]:
    """Accept either:
      - a list[ sample ], or
      - {"meta": {...}, "results": [ sample, ... ]}
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized input JSON structure.")


def _get_generations(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of generation dicts with 'generated_circuit' & 'raw_text'.
       Backward-compat: wrap single 'generated_circuit' into a list.
    """
    gens = sample.get("generations")
    if isinstance(gens, list) and len(gens) > 0:
        # Normalize keys
        norm = []
        for i, g in enumerate(gens):
            norm.append(
                {
                    "candidate_index": g.get("candidate_index", i),
                    "generated_circuit": g.get("generated_circuit", "") or "",
                    "raw_text": g.get("raw_text", "") or "",
                }
            )
        return norm

    # Fallback (old schema)
    gc = sample.get("generated_circuit", "") or ""
    return [{"candidate_index": 0, "generated_circuit": gc, "raw_text": sample.get("raw_text", "") or ""}]


def process_circuits(
    json_file: str,
    k: int = 1,
    relative_entropy_threshold: float = 0.1,
    ev_diff_threshold: float = EXPECTATION_VALUE_DIFFERENCE_THRESHOLD,
) -> Tuple[Dict[str, Any], Dict[str, List[float]], List[Dict[str, Any]]]:
    """Evaluate all samples and aggregate pass@k, EV, and RE metrics.

    Returns:
        summary_stats, raw_data_arrays, per_sample_summaries
    """
    results_in = _load_results(json_file)
    simulator = AerSimulator(method="statevector")

    # For global summaries (based on best-by-relative-entropy candidate per sample)
    all_rel_entropies: List[float] = []
    all_expectation_value_differences: List[float] = []
    all_expectation_values: List[float] = []
    all_solution_expectation_values: List[float] = []
    all_random_expectation_values: List[float] = []

    # For aggregated pass@k summaries (per-sample then averaged)
    passk_syntactic: List[float] = []
    passk_correct_state: List[float] = []
    passk_relent: List[float] = []
    passk_evd: List[float] = []

    # Also track pass@1 alongside the chosen k
    pass1_syntactic: List[float] = []
    pass1_correct_state: List[float] = []
    pass1_relent: List[float] = []
    pass1_evd: List[float] = []

    processed_samples: List[Dict[str, Any]] = []

    for idx, sample in enumerate(results_in):
        # Ensure solution JSON is parsed
        if isinstance(sample["dataset_metrics"].get("solution"), str):
            sample["dataset_metrics"]["solution"] = json.loads(sample["dataset_metrics"]["solution"])

        hamiltonian = sample["dataset_metrics"]["cost_hamiltonian"]
        generations = _get_generations(sample)
        n = len(generations)

        # Precompute solution circuit and stats once per sample
        solution_probs: Optional[List[float]] = None
        sol_expectation_value: Optional[complex] = None
        try:
            solution_circuit = parse_qasm_from_str(sample["dataset_metrics"]["optimal_circuit"])
            solution_probs, sol_expectation_value = get_probability_distribution_and_expectation_value(
                solution_circuit, simulator, hamiltonian
            )
        except Exception as err:
            # If the solution fails to parse/simulate, we can still compute syntax & most-probable-state metrics
            solution_parse_or_sim_error = str(err)
            solution_probs = None
            sol_expectation_value = None
        else:
            solution_parse_or_sim_error = None

        per_cand = []
        c_syntactic = 0
        c_correct_state = 0
        c_relent = 0
        c_evd = 0

        # Evaluate all candidates for this prompt
        for g in generations:
            cand_res: Dict[str, Any] = {
                "candidate_index": g["candidate_index"],
                "qasm_valid": False,
                "parse_error": None,
                "simulation_error": None,
                "most_probable_state_generated": None,
                "is_most_probable_state_correct": None,
                "comparison": None,
            }

            qasm_text = g["generated_circuit"]

            # 1) Parse
            try:
                circuit = parse_qasm_from_str(qasm_text)
                cand_res["qasm_valid"] = True
                c_syntactic += 1
            except Exception as err:
                cand_res["parse_error"] = str(err)
                per_cand.append(cand_res)
                continue

            # 2) Simulate + correctness
            try:
                probs, expectation_value, bitstring = evaluate_qiskit_circuit(circuit, hamiltonian, simulator)
                cand_res["most_probable_state_generated"] = bitstring
                is_correct = is_most_probable_state_correct(sample, bitstring)
                cand_res["is_most_probable_state_correct"] = bool(is_correct)
                if is_correct:
                    c_correct_state += 1
            except Exception as err:
                cand_res["simulation_error"] = str(err)
                per_cand.append(cand_res)
                continue

            # 3) Compare with solution (relative entropy + EV diff), if solution is available
            if solution_probs is not None and sol_expectation_value is not None:
                try:
                    comp = compare_solution(
                        probs,
                        solution_probs,
                        expectation_value,
                        sol_expectation_value,
                        hamiltonian,
                        circuit,
                        simulator,
                    )
                    cand_res["comparison"] = comp

                    if comp["relative_entropy"] < relative_entropy_threshold:
                        c_relent += 1
                    if comp["expectation_value_difference_abs"] <= ev_diff_threshold:
                        c_evd += 1

                except Exception as err:
                    cand_res["simulation_error"] = f"Solution comparison failed: {err}"
            else:
                cand_res["simulation_error"] = (
                    f"Solution unavailable for comparison: {solution_parse_or_sim_error}"
                )

            per_cand.append(cand_res)

        # Choose a representative candidate for global raw arrays: best by relative entropy
        best_cand = None
        best_relent = float("inf")
        for cr in per_cand:
            comp = cr.get("comparison")
            if comp is not None and comp.get("relative_entropy") is not None:
                if comp["relative_entropy"] < best_relent:
                    best_relent = comp["relative_entropy"]
                    best_cand = cr

        if best_cand and best_cand.get("comparison"):
            comp = best_cand["comparison"]
            all_rel_entropies.append(comp["relative_entropy"])
            all_expectation_value_differences.append(comp["expectation_value_difference"])
            all_expectation_values.append(comp["generated_expectation_value"])
            all_solution_expectation_values.append(comp["solution_expectation_value"])
            all_random_expectation_values.append(comp["randomized_expectation_value"])

        # Per-sample pass@k (using counts c over n)
        k_eff = min(k, n) if n > 0 else 0
        passk_syntactic.append(pass_at_k(c_syntactic, n, k_eff))
        passk_correct_state.append(pass_at_k(c_correct_state, n, k_eff))
        passk_relent.append(pass_at_k(c_relent, n, k_eff))
        passk_evd.append(pass_at_k(c_evd, n, k_eff))

        # pass@1 per-sample
        k1 = 1 if n > 0 else 0
        pass1_syntactic.append(pass_at_k(c_syntactic, n, k1))
        pass1_correct_state.append(pass_at_k(c_correct_state, n, k1))
        pass1_relent.append(pass_at_k(c_relent, n, k1))
        pass1_evd.append(pass_at_k(c_evd, n, k1))

        # Store processed sample summary
        processed_samples.append(
            {
                "sample_index": sample.get("sample_index", idx),
                "signature": sample.get("signature"),
                "n_candidates": n,
                "counts": {
                    "syntactic": c_syntactic,
                    "correct_state": c_correct_state,
                    "relative_entropy_ok": c_relent,
                    "ev_diff_ok": c_evd,
                },
                "pass_at_k": {
                    "syntactic": passk_syntactic[-1],
                    "correct_state": passk_correct_state[-1],
                    "relative_entropy": passk_relent[-1],
                    "expectation_value_diff": passk_evd[-1],
                },
                "pass_at_1": {
                    "syntactic": pass1_syntactic[-1],
                    "correct_state": pass1_correct_state[-1],
                    "relative_entropy": pass1_relent[-1],
                    "expectation_value_diff": pass1_evd[-1],
                },
                "notes": {
                    "solution_error": solution_parse_or_sim_error,
                },
            }
        )

    # ---- Summary across samples ----
    total_samples = len(processed_samples)

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    summary_stats: Dict[str, Any] = {
        "total_samples": total_samples,
        "k": k,
        "mean_pass_at_k": {
            "syntactic": _mean(passk_syntactic),
            "correct_state": _mean(passk_correct_state),
            "relative_entropy": _mean(passk_relent),
            "expectation_value_diff": _mean(passk_evd),
        },
        "mean_pass_at_1": {
            "syntactic": _mean(pass1_syntactic),
            "correct_state": _mean(pass1_correct_state),
            "relative_entropy": _mean(pass1_relent),
            "expectation_value_diff": _mean(pass1_evd),
        },
        # Legacy-style aggregates (based on best-by-relative-entropy candidate per problem)
        "average_relative_entropy": float(np.mean(all_rel_entropies)) if all_rel_entropies else None,
        "median_relative_entropy": float(np.median(all_rel_entropies)) if all_rel_entropies else None,
        "std_relative_entropy": float(np.std(all_rel_entropies)) if all_rel_entropies else None,
        "average_expectation_value": float(np.mean(all_expectation_values)) if all_expectation_values else None,
        "median_expectation_value": float(np.median(all_expectation_values)) if all_expectation_values else None,
        "std_expectation_value": float(np.std(all_expectation_values)) if all_expectation_values else None,
        "average_solution_expectation_value": float(np.mean(all_solution_expectation_values))
        if all_solution_expectation_values
        else None,
        "average_random_expectation_value": float(np.mean(all_random_expectation_values))
        if all_random_expectation_values
        else None,
    }

    raw_data = {
        "all_rel_entropies": all_rel_entropies,
        "all_expectation_value_differences": all_expectation_value_differences,
        "all_expectation_values": all_expectation_values,
        "all_solution_expectation_values": all_solution_expectation_values,
        "all_random_expectation_values": all_random_expectation_values,
    }

    return summary_stats, raw_data, processed_samples


def write_raw_data_to_csv(raw_data: Dict[str, List[float]], csv_file: str) -> None:
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Relative Entropy",
                "Expectation Value Difference",
                "Generated Expectation Value",
                "Solution Expectation Value",
                "Random Expectation Value",
            ]
        )
        n = len(raw_data["all_rel_entropies"])
        for i in range(n):
            writer.writerow(
                [
                    raw_data["all_rel_entropies"][i],
                    raw_data["all_expectation_value_differences"][i],
                    raw_data["all_expectation_values"][i],
                    raw_data["all_solution_expectation_values"][i],
                    raw_data["all_random_expectation_values"][i],
                ]
            )
    print(f"Raw data saved to {csv_file}")


# ----------------------------- CLI ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate QASM generations with pass@k aggregation.")
    # Keep backward-compatible positionals: input_file out_dir model
    ap.add_argument("input_file", type=str, help="Path to generation JSON")
    ap.add_argument("out_dir", type=str, help="Directory to write outputs")
    ap.add_argument("model", type=str, help="Model name tag for filenames")
    # New options
    ap.add_argument("--k", type=int, default=10, help="k for pass@k aggregation (per-sample)")
    ap.add_argument("--rel_entropy_threshold", type=float, default=0.1, help="Threshold for relative entropy success")
    ap.add_argument(
        "--ev_diff_threshold",
        type=float,
        default=EXPECTATION_VALUE_DIFFERENCE_THRESHOLD,
        help="Threshold for |EV_generated - EV_solution| success",
    )
    args = ap.parse_args()

    t0 = time.time()
    summary_data, raw_data, processed_samples = process_circuits(
        json_file=args.input_file,
        k=args.k,
        relative_entropy_threshold=args.rel_entropy_threshold,
        ev_diff_threshold=args.ev_diff_threshold,
    )
    dt = time.time() - t0

    # Prepare outputs
    os.makedirs(args.out_dir, exist_ok=True)
    csv_output_file_raw_data = f"{args.out_dir}/summary_{args.model}_raw_data.csv"
    summary_file = f"{args.out_dir}/summary_stats_{args.model}.json"
    per_sample_file = f"{args.out_dir}/per_sample_{args.model}.json"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary statistics saved to {summary_file}")

    with open(per_sample_file, "w", encoding="utf-8") as f:
        json.dump(processed_samples, f, indent=2)
    print(f"Per-sample pass@k details saved to {per_sample_file}")

    if raw_data and raw_data.get("all_rel_entropies"):
        write_raw_data_to_csv(raw_data, csv_output_file_raw_data)
    else:
        print("No raw data generated to save to CSV.")

    print(f"Done in {dt:.2f}s")


if __name__ == "__main__":
    main()
