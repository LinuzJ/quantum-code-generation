import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import SparsePauliOp
from qiskit_qasm3_import import parse
from qiskit.transpiler import CouplingMap

try:
	from .util import construct_qiskit_hamiltonian
except ImportError:  # pragma: no cover
	from util import construct_qiskit_hamiltonian


ASSISTANT_START_STRING = "<|im_start|>assistant"
ASSISTANS_END_STRING = "<|im_end|>"


def extract_openqasm_from_response(text: str, remove_newlines: bool = False) -> str:
	"""Extract the first OPENQASM 3.0 program from a free-form response."""
	if text is None:
		raise ValueError("Empty response text")

	start = text.find("OPENQASM 3.0")
	if start == -1:
		raise ValueError("OPENQASM 3.0 not found in response")

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
		return parse(qasm_str)
	except Exception as e:
		raise ValueError(f"Failed to parse QASM code: {e}") from e


def load_json(path: str) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def get_fake_backend(fake_backend_name: str):
	"""Return an IBM Fake backend instance from qiskit-ibm-runtime (preferred).

	The runtime package is the supported source for fake backends.
	"""
	try:
		from qiskit_ibm_runtime import fake_provider as fp
	except Exception as e:
		raise RuntimeError(
			"qiskit_ibm_runtime.fake_provider not available; install qiskit-ibm-runtime"
		) from e

	try:
		backend_cls = getattr(fp, fake_backend_name)
	except AttributeError as e:
		raise ValueError(
			f"Unknown fake backend '{fake_backend_name}'. "
			"Try e.g. FakeTorino, FakeKyoto, FakeSherbrooke, FakeWashingtonV2, FakeBrisbane."
		) from e

	return backend_cls()


def compile_to_backend(
	circuit: QuantumCircuit,
	*,
	basis_gates: Optional[Sequence[str]],
	coupling_map: Optional[CouplingMap],
	seed_transpiler: int,
	optimization_level: int,
	layout_method: str,
	routing_method: str,
	translation_method: Optional[str] = None,
	scheduling_method: Optional[str] = None,
	approximation_degree: float = 1.0,
) -> QuantumCircuit:
	"""Compile using a fixed, shared Qiskit pipeline.

	We intentionally route+layout with the same methods+seed for fair comparison.
	"""
	return transpile(
		circuit,
		basis_gates=list(basis_gates) if basis_gates is not None else None,
		coupling_map=coupling_map,
		seed_transpiler=seed_transpiler,
		optimization_level=optimization_level,
		layout_method=layout_method,
		routing_method=routing_method,
		translation_method=translation_method,
		scheduling_method=scheduling_method,
		approximation_degree=approximation_degree,
	)


def compile_to_runtime_backend(
	circuit: QuantumCircuit,
	*,
	backend,
	seed_transpiler: int,
	optimization_level: int,
	layout_method: str,
	routing_method: str,
	translation_method: Optional[str] = None,
	scheduling_method: Optional[str] = None,
	approximation_degree: float = 1.0,
) -> QuantumCircuit:
	return transpile(
		circuit,
		backend=backend,
		seed_transpiler=seed_transpiler,
		optimization_level=optimization_level,
		layout_method=layout_method,
		routing_method=routing_method,
		translation_method=translation_method,
		scheduling_method=scheduling_method,
		approximation_degree=approximation_degree,
	)


def fake_subdevice_coupling_map(backend, n_qubits: int) -> CouplingMap:
	"""Induce a subdevice on physical qubits [0..n_qubits-1].

	Important: transpile(backend=FakeKyoto) expands circuits to backend.num_qubits,
	which makes statevector-based evaluation infeasible. Using a coupling_map keeps
	the output circuit width equal to the input circuit width.
	"""
	if n_qubits <= 1:
		return CouplingMap([])
	try:
		edges = [e for e in backend.coupling_map.get_edges() if max(e) < n_qubits]
	except Exception as e:
		raise RuntimeError("Backend does not expose coupling_map") from e
	return CouplingMap(edges)


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
	count_ops = circuit.count_ops()
	# Conservative: count operations acting on exactly 2 qubits
	count = 0
	for instr, qargs, _ in circuit.data:
		try:
			nq = len(qargs)
		except Exception:
			nq = 0
		if nq == 2:
			count += 1
	# As a sanity check fallback, if circuit.data is empty, use common 2q names.
	if count == 0 and len(circuit.data) > 0:
		pass
	if count == 0 and len(circuit.data) == 0:
		return 0
	return count



def _duration_proxy_from_target(circuit: QuantumCircuit, backend) -> Optional[float]:
	"""Sum per-instruction durations from backend.target when available.

	Returns a number in backend dt units when possible, else None.
	"""
	target = getattr(backend, "target", None)
	if target is None:
		return None

	qubit_index = {q: i for i, q in enumerate(circuit.qubits)}
	total = 0.0
	for instr, qargs, _ in circuit.data:
		name = getattr(instr, "name", None)
		if not name:
			continue
		qubits = tuple(qubit_index[q] for q in qargs)
		try:
			props = target.instruction_properties(name, qubits)
			if props is None or props.duration is None:
				continue
			total += float(props.duration)
		except Exception:
			continue
	return total


def _scheduled_duration_dt(
	circuit: QuantumCircuit,
	backend,
	*,
	method: str = "alap",
) -> Optional[float]:
	"""Estimate scheduled circuit duration (in dt) from backend instruction durations.

	This is a *physical gate execution time proxy* that accounts for parallelism
	(via ASAP/ALAP scheduling) and backend-calibrated gate durations.

	Returns None if scheduling information is unavailable.
	"""
	try:
		from qiskit.transpiler import InstructionDurations, PassManager
		from qiskit.transpiler.passes import ALAPScheduleAnalysis, ASAPScheduleAnalysis, PadDelay
	except Exception:
		return None

	try:
		durations = InstructionDurations.from_backend(backend)
	except Exception:
		return None

	analysis = (
		ALAPScheduleAnalysis(durations)
		if str(method).lower() == "alap"
		else ASAPScheduleAnalysis(durations)
	)
	pm = PassManager([analysis, PadDelay(durations)])
	try:
		out = pm.run(circuit.copy())
	except Exception:
		return None

	dur = getattr(out, "duration", None)
	unit = getattr(out, "unit", None)
	if dur is None:
		return None
	# Qiskit durations are in dt when using backend durations.
	# Some versions store unit as "dt".
	if unit is not None and str(unit) not in ("dt", "s", "ns", "us", "ms"):
		# Unexpected unit; still return numeric.
		pass
	try:
		return float(dur)
	except Exception:
		return None


def hardware_metrics(circuit: QuantumCircuit, *, backend=None) -> Dict[str, Any]:
	count_ops = {str(k): int(v) for k, v in circuit.count_ops().items()}
	metrics: Dict[str, Any] = {
		"num_qubits": int(circuit.num_qubits),
		"depth": int(circuit.depth()),
		"size": int(circuit.size()),
		"count_ops": count_ops,
		"two_qubit_gates": int(count_two_qubit_gates(circuit)),
	}

	# Physical gate execution time proxy (scheduled) when backend durations exist.
	if backend is not None:
		dt_seconds = getattr(backend, "dt", None)
		try:
			dt_seconds_f = float(dt_seconds) if dt_seconds is not None else None
		except Exception:
			dt_seconds_f = None
		sched_dt = _scheduled_duration_dt(circuit, backend, method="alap")
		metrics["scheduled_duration_dt"] = float(sched_dt) if sched_dt is not None else None
		metrics["dt_seconds"] = dt_seconds_f
		metrics["scheduled_duration_s"] = (
			float(sched_dt) * float(dt_seconds_f)
			if (sched_dt is not None and dt_seconds_f is not None)
			else None
		)
	else:
		metrics["scheduled_duration_dt"] = None
		metrics["scheduled_duration_s"] = None
		metrics["dt_seconds"] = None

	# Runtime proxy: prefer duration already present, else estimate from backend target.
	if hasattr(circuit, "duration") and getattr(circuit, "duration") is not None:
		metrics["runtime_proxy"] = float(getattr(circuit, "duration"))
		metrics["runtime_proxy_unit"] = str(getattr(circuit, "unit", "dt"))
	elif backend is not None:
		dur = _duration_proxy_from_target(circuit, backend)
		metrics["runtime_proxy"] = float(dur) if dur is not None else None
		metrics["runtime_proxy_unit"] = "dt" if dur is not None else None
		# dt_seconds already set above.
	else:
		metrics["runtime_proxy"] = None
		metrics["runtime_proxy_unit"] = None
	return metrics


def get_statevector_probabilities(circuit: QuantumCircuit) -> List[float]:
	simulator = AerSimulator(method="statevector")
	sim_circuit = circuit.remove_final_measurements(inplace=False)
	sim_circuit.save_statevector()
	result = simulator.run(sim_circuit).result()
	statevector = result.get_statevector(experiment=sim_circuit)
	return statevector.probabilities().tolist()


def get_statevector(circuit: QuantumCircuit):
	simulator = AerSimulator(method="statevector")
	sim_circuit = circuit.remove_final_measurements(inplace=False)
	sim_circuit.save_statevector()
	result = simulator.run(sim_circuit).result()
	return result.get_statevector(experiment=sim_circuit)


def _ensure_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
	# Ensure there is at least one measurement so counts exist.
	# (Some QASM may be missing measurement or have final measurements removed.)
	if any(instr.operation.name == "measure" for instr in circuit.data):
		return circuit
	qc = circuit.copy()
	qc.measure_all()
	return qc


def _prepare_for_sampling_before_transpile(circuit: QuantumCircuit) -> QuantumCircuit:
	"""Ensure measurement ops exist prior to transpilation.

	This is important: if we add measurements after transpilation, the classical bits no
	longer track the *logical* qubit order, which can corrupt energy evaluation.
	"""
	return _ensure_measurements(circuit)


def _counts_to_probabilities(counts: Dict[str, int], n_qubits: int) -> Dict[int, float]:
	"""Convert Qiskit counts keys to integer bitstrings.

	Qiskit count keys are formatted as c[n-1]..c[0]. If the circuit measures q[i] -> c[i],
	then integer bit i corresponds to the rightmost character in the string.
	"""
	total = sum(counts.values())
	if total <= 0:
		return {}
	probs: Dict[int, float] = {}
	for bitstr, c in counts.items():
		# Remove spaces if multiple classical registers were present
		bitstr = bitstr.replace(" ", "")
		if len(bitstr) != n_qubits:
			# If sizes mismatch, skip (better than silently corrupting)
			continue
		bit_int = 0
		# Rightmost char is c0
		for i in range(n_qubits):
			if bitstr[-1 - i] == "1":
				bit_int |= (1 << i)
		probs[bit_int] = probs.get(bit_int, 0.0) + (float(c) / float(total))
	return probs


def simulate_counts_noisy(
	circuit: QuantumCircuit,
	*,
	backend,
	basis_gates: Optional[Sequence[str]],
	coupling_map: Optional[CouplingMap],
	shots: int,
	seed_simulator: int,
) -> Dict[str, int]:
	"""Run a noisy shot-based simulation using a noise model from the fake backend."""
	noise_model = NoiseModel.from_backend(backend)
	sim = AerSimulator(
		noise_model=noise_model,
		basis_gates=list(basis_gates) if basis_gates is not None else None,
		coupling_map=coupling_map,
	)
	qc = _ensure_measurements(circuit)
	result = sim.run(qc, shots=int(shots), seed_simulator=int(seed_simulator)).result()
	return result.get_counts(qc)


def _get_runtime_service(*, channel: str, instance: Optional[str]) :
	try:
		from qiskit_ibm_runtime import QiskitRuntimeService
	except Exception as e:
		raise RuntimeError("qiskit-ibm-runtime is required for sim_mode=runtime") from e

	# Prefer explicit env token if present, else rely on saved account.
	token = (
		os.environ.get("IBM_QUANTUM_TOKEN")
		or os.environ.get("QISKIT_IBM_TOKEN")
		or os.environ.get("QISKIT_IBM_RUNTIME_TOKEN")
	)
	try:
		if token:
			return QiskitRuntimeService(channel=channel, token=token, instance=instance)
		return QiskitRuntimeService(channel=channel, instance=instance)
	except Exception as e:
		raise RuntimeError(
			"Failed to initialize QiskitRuntimeService. "
			"Set IBM_QUANTUM_TOKEN (or save_account) and ensure network access."
		) from e


def _auto_pick_runtime_backend_name(service, *, min_qubits: int) -> str:
	# Pick an operational, non-simulator backend with enough qubits.
	try:
		candidates = service.backends(simulator=False, operational=True, min_num_qubits=int(min_qubits))
	except Exception as e:
		raise RuntimeError("Unable to list backends from IBM Runtime service") from e

	if not candidates:
		raise RuntimeError(f"No operational IBM backends found with >= {min_qubits} qubits")

	best_name = None
	best_pending = None
	best_nq = None
	for b in candidates:
		name = getattr(b, "name", None) or getattr(b, "backend_name", None) or str(b)
		try:
			st = b.status()
			pending = int(getattr(st, "pending_jobs", 10**9))
		except Exception:
			pending = 10**9
		try:
			nq = int(getattr(b, "num_qubits", 10**9))
		except Exception:
			nq = 10**9

		if best_pending is None or pending < best_pending or (pending == best_pending and nq < (best_nq or 10**9)):
			best_name = name
			best_pending = pending
			best_nq = nq

	if not best_name:
		raise RuntimeError("Failed to auto-pick a runtime backend")
	return str(best_name)


def _quasi_to_probs_dict(quasi: Any) -> Dict[int, float]:
	# QuasiDistribution typically behaves like a dict[int,float].
	if quasi is None:
		return {}
	if isinstance(quasi, dict):
		items = quasi.items()
	else:
		try:
			items = list(quasi.items())
		except Exception:
			return {}

	probs: Dict[int, float] = {}
	for k, v in items:
		if v is None:
			continue
		try:
			p = float(v)
		except Exception:
			continue
		# Keys are usually ints; if strings appear, parse as bitstring.
		if isinstance(k, int):
			probs[int(k)] = probs.get(int(k), 0.0) + p
		else:
			ks = str(k).replace(" ", "")
			if ks.startswith("0b"):
				ks = ks[2:]
			if ks and all(ch in "01" for ch in ks):
				bit_int = 0
				for i in range(len(ks)):
					if ks[-1 - i] == "1":
						bit_int |= (1 << i)
				probs[int(bit_int)] = probs.get(int(bit_int), 0.0) + p
	return probs


def algorithmic_metrics_runtime(
	circuit: QuantumCircuit,
	*,
	hamiltonian: Optional[str],
	service,
	backend_name: str,
	shots: int,
) -> Dict[str, Any]:
	if not hamiltonian:
		return {"energy": None, "solution_probability": None, "solution_probability_mode": None}
	try:
		from qiskit_ibm_runtime import SamplerV2
	except Exception as e:
		raise RuntimeError("qiskit-ibm-runtime is required for sim_mode=runtime") from e

	# Open-plan accounts may not be authorized to create Runtime Sessions.
	# Use "job mode" directly on a backend instead.
	backend = service.backend(backend_name)
	sampler = SamplerV2(mode=backend)
	job = sampler.run([circuit], shots=int(shots))
	res = job.result()
	job_id = getattr(job, "job_id", None)
	job_id = job_id() if callable(job_id) else job_id

	# Extract quasi distribution
	quasis = getattr(res, "quasi_dists", None)
	if quasis is None:
		quasis = getattr(res, "quasi_distributions", None)
	if quasis is None and isinstance(res, list):
		quasis = res
	if not quasis:
		return {"energy": None, "solution_probability": None, "solution_probability_mode": None, "shots": int(shots), "job_id": job_id}

	probs = _quasi_to_probs_dict(quasis[0])
	metrics = distribution_metrics_from_probs_dict(probs, hamiltonian)
	metrics["shots"] = int(shots)
	metrics["job_id"] = job_id
	return metrics


def apply_layout_to_operator(op: SparsePauliOp, circuit: QuantumCircuit) -> SparsePauliOp:
	layout = getattr(circuit, "layout", None)
	if layout is None:
		return op
	try:
		return op.apply_layout(layout)
	except Exception:
		return op


_COEFF_RE = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"


def _parse_ising_terms(expression: str) -> Tuple[List[Tuple[float, Tuple[int, ...]]], int]:
	"""Parse an Ising-style expression containing only Z(i) and Z(i)@Z(j).

	Returns (terms, n_qubits) where terms are (coeff, (qubits...)).
	"""
	if not expression:
		return [], 0

	# Example terms:
	# 0.5 * (Z(1) @ Z(3))
	# 2.0 * Z(3)
	zz_pat = re.compile(_COEFF_RE + r"\s*\*\s*\(\s*Z\((\d+)\)\s*@\s*Z\((\d+)\)\s*\)")
	z_pat = re.compile(_COEFF_RE + r"\s*\*\s*Z\((\d+)\)")

	terms: List[Tuple[float, Tuple[int, ...]]] = []
	max_q = -1

	for m in zz_pat.finditer(expression):
		c = float(m.group(1))
		q1 = int(m.group(2))
		q2 = int(m.group(3))
		max_q = max(max_q, q1, q2)
		terms.append((c, (q1, q2)))

	# Avoid double-counting ZZ terms when scanning Z terms
	# We'll remove matched ZZ substrings before searching for Z.
	expr_wo_zz = zz_pat.sub("", expression)
	for m in z_pat.finditer(expr_wo_zz):
		c = float(m.group(1))
		q = int(m.group(2))
		max_q = max(max_q, q)
		terms.append((c, (q,)))

	return terms, max_q + 1


def _z_eig(bit_int: int, qubit: int) -> int:
	# Z|0> = +1, Z|1> = -1
	return 1 if ((bit_int >> qubit) & 1) == 0 else -1


def classical_energy(expression: str, bit_int: int) -> float:
	terms, _ = _parse_ising_terms(expression)
	energy = 0.0
	for coeff, qs in terms:
		val = 1
		for q in qs:
			val *= _z_eig(bit_int, q)
		energy += coeff * val
	return float(energy)


def min_energy_states(expression: str, n_qubits: int, *, max_exact_qubits: int = 20) -> Dict[str, Any]:
	"""Compute (exact or approximate) minimum energy and an estimate of solution probability.

	- For n_qubits <= max_exact_qubits: exact enumeration over all 2^n bitstrings.
	- Otherwise: caller should use an approximate solution set.
	"""
	if n_qubits <= 0:
		return {"mode": "none", "min_energy": None}
	if n_qubits > max_exact_qubits:
		return {"mode": "approx", "min_energy": None, "note": "n_qubits too large for exact enumeration"}

	min_e = None
	for bit_int in range(1 << n_qubits):
		e = classical_energy(expression, bit_int)
		if min_e is None or e < min_e:
			min_e = e

	return {"mode": "exact", "min_energy": float(min_e)}


def solution_probability(
	probabilities: Sequence[float],
	expression: str,
	*,
	max_exact_qubits: int = 20,
	topk_if_large: int = 4096,
) -> Dict[str, Any]:
	n_qubits = int(round(np.log2(len(probabilities)))) if len(probabilities) else 0
	if n_qubits <= 0 or (1 << n_qubits) != len(probabilities):
		return {"mode": "none", "solution_probability": None, "min_energy": None}

	min_info = min_energy_states(expression, n_qubits, max_exact_qubits=max_exact_qubits)
	if min_info.get("mode") == "exact":
		min_e = float(min_info["min_energy"])
		sol_p = 0.0
		for bit_int in range(1 << n_qubits):
			if classical_energy(expression, bit_int) == min_e:
				sol_p += float(probabilities[bit_int])
		return {"mode": "exact", "solution_probability": float(sol_p), "min_energy": min_e}

	# Approximate: evaluate energy only on the most likely outcomes
	idx = np.argsort(np.asarray(probabilities))[-topk_if_large:]
	best = None
	for bit_int in idx:
		e = classical_energy(expression, int(bit_int))
		if best is None or e < best:
			best = e
	if best is None:
		return {"mode": "approx", "solution_probability": None, "min_energy": None}
	best = float(best)
	sol_p = 0.0
	for bit_int in idx:
		if classical_energy(expression, int(bit_int)) == best:
			sol_p += float(probabilities[int(bit_int)])
	return {
		"mode": "approx_topk",
		"topk": int(topk_if_large),
		"solution_probability": float(sol_p),
		"min_energy": best,
	}


def distribution_metrics_from_probs_dict(
	probs: Dict[int, float],
	hamiltonian: str,
	*,
	max_exact_qubits: int = 20,
) -> Dict[str, Any]:
	if not probs:
		return {"energy": None, "solution_probability": None, "solution_probability_mode": None}

	n_qubits = max((k.bit_length() for k in probs.keys()), default=0)
	# Qubits might be exactly n even if max bit_length smaller
	# (e.g., only low-index bits observed). We'll trust caller for n.

	# Energy as average classical energy under the measurement distribution.
	energy = 0.0
	for bit_int, p in probs.items():
		energy += float(p) * classical_energy(hamiltonian, int(bit_int))

	# Solution probability against the min-energy of the Hamiltonian.
	# Exact enumeration if small enough; else approximate using observed outcomes.
	# Determine n from Hamiltonian parse when possible.
	terms, inferred_n = _parse_ising_terms(hamiltonian)
	if inferred_n > 0:
		n = inferred_n
	else:
		n = n_qubits

	if n <= max_exact_qubits and n > 0:
		min_info = min_energy_states(hamiltonian, n, max_exact_qubits=max_exact_qubits)
		min_e = float(min_info["min_energy"]) if min_info.get("min_energy") is not None else None
		if min_e is None:
			sol_p = None
			mode = "none"
		else:
			sol_p = 0.0
			for bit_int, p in probs.items():
				if classical_energy(hamiltonian, int(bit_int)) == min_e:
					sol_p += float(p)
			mode = "exact"
		return {
			"energy": float(energy),
			"solution_probability": float(sol_p) if sol_p is not None else None,
			"solution_probability_mode": mode,
			"min_energy": min_e,
		}

	# Approximate: take best energy among observed outcomes
	best = None
	for bit_int in probs.keys():
		e = classical_energy(hamiltonian, int(bit_int))
		if best is None or e < best:
			best = e
	if best is None:
		return {"energy": float(energy), "solution_probability": None, "solution_probability_mode": "approx"}
	best = float(best)
	sol_p = 0.0
	for bit_int, p in probs.items():
		if classical_energy(hamiltonian, int(bit_int)) == best:
			sol_p += float(p)
	return {
		"energy": float(energy),
		"solution_probability": float(sol_p),
		"solution_probability_mode": "approx_observed",
		"min_energy": best,
	}


def algorithmic_metrics(circuit: QuantumCircuit, hamiltonian: Optional[str]) -> Dict[str, Any]:
	if not hamiltonian:
		return {"energy": None, "solution_probability": None, "solution_probability_mode": None}

	statevector = get_statevector(circuit)
	probs = statevector.probabilities().tolist()
	oper = construct_qiskit_hamiltonian(hamiltonian)
	oper_mapped = apply_layout_to_operator(oper, circuit)
	energy = statevector.expectation_value(oper_mapped)
	energy_val = float(np.real(energy))

	sol = solution_probability(probs, hamiltonian)
	return {
		"energy": energy_val,
		"solution_probability": sol.get("solution_probability"),
		"solution_probability_mode": sol.get("mode"),
		"min_energy": sol.get("min_energy"),
	}


def algorithmic_metrics_noisy(
	circuit: QuantumCircuit,
	*,
	hamiltonian: Optional[str],
	backend,
	basis_gates: Optional[Sequence[str]],
	coupling_map: Optional[CouplingMap],
	shots: int,
	seed_simulator: int,
) -> Dict[str, Any]:
	if not hamiltonian:
		return {"energy": None, "solution_probability": None, "solution_probability_mode": None}
	counts = simulate_counts_noisy(
		circuit,
		backend=backend,
		basis_gates=basis_gates,
		coupling_map=coupling_map,
		shots=shots,
		seed_simulator=seed_simulator,
	)
	# Use the circuit width for parsing count strings
	probs = _counts_to_probabilities(counts, int(circuit.num_qubits))
	metrics = distribution_metrics_from_probs_dict(probs, hamiltonian)
	metrics["shots"] = int(shots)
	metrics["seed_simulator"] = int(seed_simulator)
	return metrics


@dataclass
class EvalConfig:
	quasar_json: str
	index: int
	fake_backend: str
	seed_transpiler: int = 0
	optimization_level: int = 3
	layout_method: str = "sabre"
	routing_method: str = "sabre"
	translation_method: Optional[str] = None
	scheduling_method: Optional[str] = None
	approximation_degree: float = 1.0
	sim_mode: str = "ideal"  # 'ideal' (statevector), 'noisy' (Aer+noise), 'runtime' (real hardware)
	shots: int = 2000
	seed_simulator: int = 0
	runtime_channel: str = os.environ.get("RUNTIME_CHANNEL", "ibm_cloud")
	runtime_instance: Optional[str] = None
	runtime_backend: str = "ibm_torino"  # backend name or 'auto'
	out_dir: str = "out_instance_eval"


def evaluate_instance(cfg: EvalConfig) -> Dict[str, Any]:
	backend = None
	service = None
	backend_name = None
	if cfg.sim_mode == "runtime":
		service = _get_runtime_service(channel=cfg.runtime_channel, instance=cfg.runtime_instance)
		backend_name = str(cfg.runtime_backend)
	else:
		backend = get_fake_backend(cfg.fake_backend)
		backend_name = cfg.fake_backend

	data = load_json(cfg.quasar_json)
	if cfg.index < 0 or cfg.index >= len(data):
		raise IndexError(f"index {cfg.index} out of range (0..{len(data)-1})")
	item = data[cfg.index]

	ground_truth_qasm = item.get("ground_truth", "")
	response = item.get("response", "")
	if not ground_truth_qasm:
		raise ValueError("Missing ground_truth QASM")
	if not response:
		raise ValueError("Missing response QASM")

	quasar_qasm = extract_openqasm_from_response(response, remove_newlines=False)

	gt_circ = parse_qasm_from_str(ground_truth_qasm)
	q_circ = parse_qasm_from_str(quasar_qasm)

	n_qubits = max(int(gt_circ.num_qubits), int(q_circ.num_qubits))
	if cfg.sim_mode == "runtime":
		# Resolve backend after we know how many qubits are required.
		if str(backend_name).lower() == "auto":
			backend_name = _auto_pick_runtime_backend_name(service, min_qubits=n_qubits)
		try:
			backend = service.backend(backend_name)
		except Exception as e:
			raise ValueError(
				f"Runtime backend '{backend_name}' not available. "
				"Try --runtime-backend auto or a backend name returned by service.backends()."
			) from e
		if n_qubits > int(getattr(backend, "num_qubits", n_qubits)):
			raise ValueError(
				f"Instance requires {n_qubits} qubits, but runtime backend {backend_name} has only {backend.num_qubits}"
			)
	else:
		if n_qubits > int(getattr(backend, "num_qubits", n_qubits)):
			raise ValueError(
				f"Instance requires {n_qubits} qubits, but backend {backend_name} has only {backend.num_qubits}"
			)

	# For sampling-based modes, add measurements before transpiling so classical bits track logical qubits.
	if cfg.sim_mode in ("noisy", "runtime"):
		gt_circ = _prepare_for_sampling_before_transpile(gt_circ)
		q_circ = _prepare_for_sampling_before_transpile(q_circ)

	# Fake backend compilation uses an induced subdevice coupling map to avoid width inflation.
	cmap = None
	basis_gates = None
	if cfg.sim_mode != "runtime":
		cmap = fake_subdevice_coupling_map(backend, n_qubits)
		basis_gates = list(getattr(backend, "operation_names", [])) or None

	# Hamiltonian is stored in extra_info for quasar.json items
	hamiltonian = (item.get("extra_info") or {}).get("cost_hamiltonian")

	if cfg.sim_mode == "runtime":
		gt_compiled = compile_to_runtime_backend(
			gt_circ,
			backend=backend,
			seed_transpiler=cfg.seed_transpiler,
			optimization_level=cfg.optimization_level,
			layout_method=cfg.layout_method,
			routing_method=cfg.routing_method,
			translation_method=cfg.translation_method,
			scheduling_method=cfg.scheduling_method,
			approximation_degree=cfg.approximation_degree,
		)
		q_compiled = compile_to_runtime_backend(
			q_circ,
			backend=backend,
			seed_transpiler=cfg.seed_transpiler,
			optimization_level=cfg.optimization_level,
			layout_method=cfg.layout_method,
			routing_method=cfg.routing_method,
			translation_method=cfg.translation_method,
			scheduling_method=cfg.scheduling_method,
			approximation_degree=cfg.approximation_degree,
		)
	else:
		gt_compiled = compile_to_backend(
			gt_circ,
			basis_gates=basis_gates,
			coupling_map=cmap,
			seed_transpiler=cfg.seed_transpiler,
			optimization_level=cfg.optimization_level,
			layout_method=cfg.layout_method,
			routing_method=cfg.routing_method,
			translation_method=cfg.translation_method,
			scheduling_method=cfg.scheduling_method,
			approximation_degree=cfg.approximation_degree,
		)
		q_compiled = compile_to_backend(
			q_circ,
			basis_gates=basis_gates,
			coupling_map=cmap,
			seed_transpiler=cfg.seed_transpiler,
			optimization_level=cfg.optimization_level,
			layout_method=cfg.layout_method,
			routing_method=cfg.routing_method,
			translation_method=cfg.translation_method,
			scheduling_method=cfg.scheduling_method,
			approximation_degree=cfg.approximation_degree,
		)

	if cfg.sim_mode == "noisy":
		gt_algo = algorithmic_metrics_noisy(
			gt_compiled,
			hamiltonian=hamiltonian,
			backend=backend,
			basis_gates=basis_gates,
			coupling_map=cmap,
			shots=cfg.shots,
			seed_simulator=cfg.seed_simulator,
		)
		q_algo = algorithmic_metrics_noisy(
			q_compiled,
			hamiltonian=hamiltonian,
			backend=backend,
			basis_gates=basis_gates,
			coupling_map=cmap,
			shots=cfg.shots,
			seed_simulator=cfg.seed_simulator,
		)
	elif cfg.sim_mode == "runtime":
		gt_algo = algorithmic_metrics_runtime(
			gt_compiled,
			hamiltonian=hamiltonian,
			service=service,
			backend_name=backend_name,
			shots=cfg.shots,
		)
		q_algo = algorithmic_metrics_runtime(
			q_compiled,
			hamiltonian=hamiltonian,
			service=service,
			backend_name=backend_name,
			shots=cfg.shots,
		)
	else:
		gt_algo = algorithmic_metrics(gt_compiled, hamiltonian)
		q_algo = algorithmic_metrics(q_compiled, hamiltonian)

	result = {
		"index": int(cfg.index),
		"fake_backend": cfg.fake_backend if cfg.sim_mode != "runtime" else None,
		"real_backend": backend_name if cfg.sim_mode == "runtime" else None,
		"subdevice": {"physical_qubits": list(range(n_qubits))} if cfg.sim_mode != "runtime" else None,
		"compile": {
			"seed_transpiler": int(cfg.seed_transpiler),
			"optimization_level": int(cfg.optimization_level),
			"layout_method": cfg.layout_method,
			"routing_method": cfg.routing_method,
			"translation_method": cfg.translation_method,
			"scheduling_method": cfg.scheduling_method,
			"approximation_degree": float(cfg.approximation_degree),
		},
		"simulation": {
			"mode": cfg.sim_mode,
			"shots": int(cfg.shots) if cfg.sim_mode in ("noisy", "runtime") else None,
			"seed_simulator": int(cfg.seed_simulator) if cfg.sim_mode == "noisy" else None,
			"runtime_channel": cfg.runtime_channel if cfg.sim_mode == "runtime" else None,
			"runtime_instance": cfg.runtime_instance if cfg.sim_mode == "runtime" else None,
		},
		"hamiltonian_present": bool(hamiltonian),
		"reference": {
			"algorithmic": gt_algo,
			"hardware": hardware_metrics(gt_compiled, backend=backend),
		},
		"quasar": {
			"algorithmic": q_algo,
			"hardware": hardware_metrics(q_compiled, backend=backend),
		},
	}

	os.makedirs(cfg.out_dir, exist_ok=True)
	out_path = os.path.join(cfg.out_dir, f"instance_{cfg.index:04d}.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(result, f, indent=2)

	return result


def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Evaluate one Quasar instance vs reference on an IBM Fake backend")
	p.add_argument("--quasar-json", required=True)
	p.add_argument("--index", type=int, required=True)
	p.add_argument("--fake-backend", default="FakeTorino")
	p.add_argument("--seed-transpiler", type=int, default=0)
	p.add_argument("--optimization-level", type=int, default=3)
	p.add_argument("--layout-method", default="sabre")
	p.add_argument("--routing-method", default="sabre")
	p.add_argument("--translation-method", default=None)
	p.add_argument("--scheduling-method", default=None)
	p.add_argument("--approximation-degree", type=float, default=1.0)
	p.add_argument("--sim-mode", choices=["ideal", "noisy", "runtime"], default="ideal")
	p.add_argument("--shots", type=int, default=2000)
	p.add_argument("--seed-simulator", type=int, default=0)
	p.add_argument("--runtime-channel", default=os.environ.get("RUNTIME_CHANNEL", "ibm_cloud"))
	p.add_argument("--runtime-instance", default=None)
	p.add_argument("--runtime-backend", default="ibm_torino", help="Backend name or 'auto' (sim_mode=runtime)")
	p.add_argument("--out-dir", default="out_instance_eval")
	return p


def main():
	args = build_arg_parser().parse_args()
	cfg = EvalConfig(
		quasar_json=args.quasar_json,
		index=args.index,
		fake_backend=args.fake_backend,
		seed_transpiler=args.seed_transpiler,
		optimization_level=args.optimization_level,
		layout_method=args.layout_method,
		routing_method=args.routing_method,
		translation_method=args.translation_method,
		scheduling_method=args.scheduling_method,
		approximation_degree=args.approximation_degree,
		sim_mode=args.sim_mode,
		shots=args.shots,
		seed_simulator=args.seed_simulator,
		runtime_channel=args.runtime_channel,
		runtime_instance=args.runtime_instance,
		runtime_backend=args.runtime_backend,
		out_dir=args.out_dir,
	)
	res = evaluate_instance(cfg)
	print(json.dumps(res, indent=2))


if __name__ == "__main__":
	main()
