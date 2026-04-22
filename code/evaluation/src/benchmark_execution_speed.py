import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
	from .instance_eval_common import (
		EvalConfig,
		_auto_pick_runtime_backend_name,
		_get_runtime_service,
		_prepare_for_sampling_before_transpile,
		algorithmic_metrics_noisy,
		compile_to_backend,
		compile_to_runtime_backend,
		extract_openqasm_from_response,
		fake_subdevice_coupling_map,
		get_fake_backend,
		hardware_metrics,
		load_json,
		parse_qasm_from_str,
	)
except ImportError:  # pragma: no cover
	from instance_eval_common import (
		EvalConfig,
		_auto_pick_runtime_backend_name,
		_get_runtime_service,
		_prepare_for_sampling_before_transpile,
		algorithmic_metrics_noisy,
		compile_to_backend,
		compile_to_runtime_backend,
		extract_openqasm_from_response,
		fake_subdevice_coupling_map,
		get_fake_backend,
		hardware_metrics,
		load_json,
		parse_qasm_from_str,
	)


def _utc_ts() -> str:
	return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _best_effort_runtime_timing(metrics: Any) -> Optional[Dict[str, Any]]:
	"""Extract a small, stable subset of runtime timing/accounting metrics.

	The exact schema can differ across runtime versions; keep this defensive.
	"""
	if metrics is None:
		return None
	if not isinstance(metrics, dict):
		return {"raw": str(metrics)[:400]}

	usage = metrics.get("usage") if isinstance(metrics.get("usage"), dict) else {}
	out: Dict[str, Any] = {}
	if usage:
		for k in ("quantum_seconds", "qpu_seconds", "qpu_time", "seconds"):
			if k in usage:
				out[k] = usage.get(k)
	# Sometimes flattened
	for k in ("quantum_seconds", "qpu_seconds"):
		if k in metrics and k not in out:
			out[k] = metrics.get(k)
	return out or None


def _parse_indices_arg(indices: Optional[str], indices_file: Optional[str]) -> List[int]:
	if indices_file:
		out: List[int] = []
		with open(indices_file, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line or line.startswith("#"):
					continue
				out.append(int(line))
		return out
	if not indices:
		return []
	parts = [p.strip() for p in str(indices).split(",") if p.strip()]
	return [int(p) for p in parts]


def _runtime_compile(
	*,
	circuit,
	backend,
	seed_transpiler: int,
	optimization_level: int,
	layout_method: str,
	routing_method: str,
	translation_method: Optional[str],
	scheduling_method: Optional[str],
	approximation_degree: float,
):
	circ2 = _prepare_for_sampling_before_transpile(circuit)
	t0 = time.perf_counter()
	compiled = compile_to_runtime_backend(
		circ2,
		backend=backend,
		seed_transpiler=seed_transpiler,
		optimization_level=optimization_level,
		layout_method=layout_method,
		routing_method=routing_method,
		translation_method=translation_method,
		scheduling_method=scheduling_method,
		approximation_degree=approximation_degree,
	)
	return float(time.perf_counter() - t0), compiled


def _local_compile_and_simulate_noisy(
	*,
	circuit,
	hamiltonian: Optional[str],
	backend,
	n_qubits: int,
	seed_transpiler: int,
	optimization_level: int,
	layout_method: str,
	routing_method: str,
	translation_method: Optional[str],
	scheduling_method: Optional[str],
	approximation_degree: float,
	shots: int,
	seed_simulator: int,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
	cmap = fake_subdevice_coupling_map(backend, int(n_qubits))
	basis_gates = list(getattr(backend, "operation_names", [])) or None
	circ2 = _prepare_for_sampling_before_transpile(circuit)

	t0 = time.perf_counter()
	compiled = compile_to_backend(
		circ2,
		basis_gates=basis_gates,
		coupling_map=cmap,
		seed_transpiler=seed_transpiler,
		optimization_level=optimization_level,
		layout_method=layout_method,
		routing_method=routing_method,
		translation_method=translation_method,
		scheduling_method=None,
		approximation_degree=approximation_degree,
	)
	t_compile = float(time.perf_counter() - t0)

	hw = hardware_metrics(compiled, backend=backend)

	t1 = time.perf_counter()
	algo = algorithmic_metrics_noisy(
		compiled,
		hamiltonian=hamiltonian,
		backend=backend,
		basis_gates=basis_gates,
		coupling_map=cmap,
		shots=int(shots),
		seed_simulator=int(seed_simulator),
	)
	t_sim = float(time.perf_counter() - t1)

	return {"compile_s": t_compile, "simulate_s": t_sim}, {"hardware": hw, "algorithmic": algo}


def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description=(
			"Benchmark execution speed: local Fake backend compilation (and optional noisy Aer simulation) "
			"vs IBM Runtime (real backend calibration). Writes JSON+CSV."
		)
	)
	p.add_argument("--quasar-json", default="generated_circuits/quasar.json")
	p.add_argument("--indices", default=None, help="Comma-separated instance indices")
	p.add_argument("--indices-file", default=None, help="File with one index per line")
	p.add_argument("--out-dir", default="out_speed")

	# Local
	p.add_argument("--fake-backend", default="FakeTorino")
	p.add_argument("--local-no-sim", action="store_true", help="Skip Aer; compile only and record physical duration")

	# Shared compile knobs
	p.add_argument("--seed-transpiler", type=int, default=0)
	p.add_argument("--optimization-level", type=int, default=3)
	p.add_argument("--layout-method", default="sabre")
	p.add_argument("--routing-method", default="sabre")
	p.add_argument("--translation-method", default=None)
	p.add_argument("--scheduling-method", default=None)
	p.add_argument("--approximation-degree", type=float, default=1.0)

	# Sampling
	p.add_argument("--shots", type=int, default=200)
	p.add_argument("--seed-simulator", type=int, default=0)

	# Runtime
	p.add_argument("--skip-runtime", action="store_true")
	p.add_argument(
		"--runtime-no-submit",
		action="store_true",
		help="Do not submit jobs; only compile against the runtime backend and compute scheduled-duration metrics.",
	)
	p.add_argument("--runtime-channel", default=os.environ.get("RUNTIME_CHANNEL", "ibm_cloud"))
	p.add_argument("--runtime-instance", default=os.environ.get("RUNTIME_INSTANCE", None))
	p.add_argument("--runtime-backend", default=os.environ.get("RUNTIME_BACKEND", "ibm_torino"))

	return p


def main() -> None:
	args = build_arg_parser().parse_args()
	indices = _parse_indices_arg(args.indices, args.indices_file)
	if not indices:
		raise ValueError("No indices selected")

	data = load_json(args.quasar_json)
	for idx in indices:
		if idx < 0 or idx >= len(data):
			raise IndexError(f"index {idx} out of range (0..{len(data)-1})")

	instances: List[Dict[str, Any]] = []
	skipped: List[Dict[str, Any]] = []
	max_nq = 0
	for idx in indices:
		item = data[idx]
		ground_truth_qasm = item.get("ground_truth", "")
		response = item.get("response", "")
		if not ground_truth_qasm or not response:
			skipped.append({"index": int(idx), "reason": "missing_ground_truth_or_response"})
			continue
		try:
			vista_qasm = extract_openqasm_from_response(response, remove_newlines=False)
			gt_circ = parse_qasm_from_str(ground_truth_qasm)
			vista_circ = parse_qasm_from_str(vista_qasm)
		except Exception as e:
			skipped.append({"index": int(idx), "reason": "qasm_parse_failed", "error": str(e)[:400]})
			continue
		ham = (item.get("extra_info") or {}).get("cost_hamiltonian")
		nq = max(int(gt_circ.num_qubits), int(vista_circ.num_qubits))
		max_nq = max(max_nq, nq)
		instances.append(
			{
				"index": int(idx),
				"n_qubits": int(nq),
				"hamiltonian": ham,
				"ground_truth_circuit": gt_circ,
				"vista_circuit": vista_circ,
			}
		)

	if not instances:
		raise ValueError("No valid instances after parsing; all selected indices were skipped")

	compile_cfg = EvalConfig(
		quasar_json=args.quasar_json,
		index=int(instances[0]["index"]),
		fake_backend=args.fake_backend,
		seed_transpiler=int(args.seed_transpiler),
		optimization_level=int(args.optimization_level),
		layout_method=str(args.layout_method),
		routing_method=str(args.routing_method),
		translation_method=args.translation_method,
		scheduling_method=args.scheduling_method,
		approximation_degree=float(args.approximation_degree),
		sim_mode="noisy",
		shots=int(args.shots),
		seed_simulator=int(args.seed_simulator),
		runtime_channel=str(args.runtime_channel),
		runtime_instance=args.runtime_instance,
		runtime_backend=str(args.runtime_backend),
		out_dir=str(args.out_dir),
	)

	rows: List[Dict[str, Any]] = []

	# Local
	fake_backend = get_fake_backend(args.fake_backend)
	if max_nq > int(getattr(fake_backend, "num_qubits", max_nq)):
		raise ValueError(
			f"Need {max_nq} qubits but fake backend {args.fake_backend} has {fake_backend.num_qubits}"
		)
	for inst in instances:
		idx = int(inst["index"])
		nq = int(inst["n_qubits"])
		ham = inst["hamiltonian"]
		for variant, circ in (
			("ground_truth", inst["ground_truth_circuit"]),
			("vista", inst["vista_circuit"]),
		):
			try:
				if args.local_no_sim:
					cmap = fake_subdevice_coupling_map(fake_backend, nq)
					basis_gates = list(getattr(fake_backend, "operation_names", [])) or None
					circ2 = _prepare_for_sampling_before_transpile(circ)
					t0 = time.perf_counter()
					compiled = compile_to_backend(
						circ2,
						basis_gates=basis_gates,
						coupling_map=cmap,
						seed_transpiler=compile_cfg.seed_transpiler,
						optimization_level=compile_cfg.optimization_level,
						layout_method=compile_cfg.layout_method,
						routing_method=compile_cfg.routing_method,
						translation_method=compile_cfg.translation_method,
						scheduling_method=None,
						approximation_degree=compile_cfg.approximation_degree,
					)
					t_compile = float(time.perf_counter() - t0)
					hw = hardware_metrics(compiled, backend=fake_backend)
					rows.append(
						{
							"index": idx,
							"variant": variant,
							"n_qubits": nq,
							"backend": args.fake_backend,
							"mode": "local_physical",
							"compile_s": t_compile,
							"run_s": None,
							"depth": hw.get("depth"),
							"two_qubit_gates": hw.get("two_qubit_gates"),
							"scheduled_duration_dt": hw.get("scheduled_duration_dt"),
							"scheduled_duration_s": hw.get("scheduled_duration_s"),
						}
					)
				else:
					timing, metrics = _local_compile_and_simulate_noisy(
						circuit=circ,
						hamiltonian=ham,
						backend=fake_backend,
						n_qubits=nq,
						seed_transpiler=compile_cfg.seed_transpiler,
						optimization_level=compile_cfg.optimization_level,
						layout_method=compile_cfg.layout_method,
						routing_method=compile_cfg.routing_method,
						translation_method=compile_cfg.translation_method,
						scheduling_method=compile_cfg.scheduling_method,
						approximation_degree=compile_cfg.approximation_degree,
						shots=compile_cfg.shots,
						seed_simulator=compile_cfg.seed_simulator,
					)
					hw = metrics["hardware"]
					rows.append(
						{
							"index": idx,
							"variant": variant,
							"n_qubits": nq,
							"backend": args.fake_backend,
							"mode": "local_noisy",
							"compile_s": float(timing["compile_s"]),
							"run_s": float(timing["simulate_s"]),
							"depth": hw.get("depth"),
							"two_qubit_gates": hw.get("two_qubit_gates"),
							"scheduled_duration_dt": hw.get("scheduled_duration_dt"),
							"scheduled_duration_s": hw.get("scheduled_duration_s"),
						}
					)
			except Exception as e:
				skipped.append(
					{
						"index": idx,
						"variant": variant,
						"stage": "local",
						"reason": "compile_or_sim_failed",
						"error": str(e)[:400],
					}
				)
				continue

	# Runtime
	runtime_summary: Optional[Dict[str, Any]] = None
	if not args.skip_runtime:
		try:
			from qiskit_ibm_runtime import SamplerV2
		except Exception as e:
			raise RuntimeError("qiskit-ibm-runtime is required unless --skip-runtime is set") from e

		service = _get_runtime_service(channel=args.runtime_channel, instance=args.runtime_instance)
		backend_name = (
			_auto_pick_runtime_backend_name(service, min_qubits=max_nq)
			if str(args.runtime_backend).lower() == "auto"
			else str(args.runtime_backend)
		)
		rt_backend = service.backend(backend_name)

		pubs: List[Any] = []
		pub_meta: List[Tuple[int, str, int, float, Optional[int], Optional[int], Optional[float], Optional[float]]] = []
		for inst in instances:
			idx = int(inst["index"])
			nq = int(inst["n_qubits"])
			for variant, circ in (
				("ground_truth", inst["ground_truth_circuit"]),
				("vista", inst["vista_circuit"]),
			):
				try:
					t_compile, compiled = _runtime_compile(
						circuit=circ,
						backend=rt_backend,
						seed_transpiler=compile_cfg.seed_transpiler,
						optimization_level=compile_cfg.optimization_level,
						layout_method=compile_cfg.layout_method,
						routing_method=compile_cfg.routing_method,
						translation_method=compile_cfg.translation_method,
						scheduling_method=compile_cfg.scheduling_method,
						approximation_degree=compile_cfg.approximation_degree,
					)
					hw = hardware_metrics(compiled, backend=rt_backend)
				except Exception as e:
					skipped.append(
						{
							"index": idx,
							"variant": variant,
							"stage": "runtime_compile",
							"reason": "compile_failed",
							"error": str(e)[:400],
						}
					)
					continue
				pubs.append(compiled)
				pub_meta.append(
					(
						idx,
						variant,
						nq,
						float(t_compile),
						hw.get("depth"),
						hw.get("two_qubit_gates"),
						hw.get("scheduled_duration_dt"),
						hw.get("scheduled_duration_s"),
					)
				)

		if args.runtime_no_submit:
			runtime_summary = {
				"runtime_channel": args.runtime_channel,
				"runtime_instance": args.runtime_instance,
				"runtime_backend": backend_name,
				"shots": int(args.shots),
				"num_circuits": int(len(pubs)),
				"submitted": False,
			}
			job_id = None
			t_total = None
		else:
			sampler = SamplerV2(mode=rt_backend)
			t_submit0 = time.perf_counter()
			job = sampler.run(pubs, shots=int(args.shots))
			_ = job.result()
			t_total = float(time.perf_counter() - t_submit0)
			job_id_attr = getattr(job, "job_id", None)
			job_id = job_id_attr() if callable(job_id_attr) else job_id_attr

			metrics_raw = None
			metrics_extract = None
			try:
				metrics_raw = job.metrics()
				metrics_extract = _best_effort_runtime_timing(metrics_raw)
			except Exception:
				metrics_raw = None
				metrics_extract = None

			runtime_summary = {
				"runtime_channel": args.runtime_channel,
				"runtime_instance": args.runtime_instance,
				"runtime_backend": backend_name,
				"shots": int(args.shots),
				"num_circuits": int(len(pubs)),
				"submitted": True,
				"job_id": job_id,
				"submit_to_result_s": float(t_total),
				"per_circuit_submit_to_result_s": float(t_total) / float(len(pubs)) if len(pubs) else None,
				"metrics": metrics_raw,
				"metrics_extract": metrics_extract,
			}

		for (idx, variant, nq, t_compile, depth, twoq, sched_dt, sched_s) in pub_meta:
			rows.append(
				{
					"index": int(idx),
					"variant": str(variant),
					"n_qubits": int(nq),
					"backend": backend_name,
					"mode": "runtime" if not args.runtime_no_submit else "runtime_physical",
					"compile_s": float(t_compile),
					"run_s": None,
					"depth": depth,
					"two_qubit_gates": twoq,
					"scheduled_duration_dt": sched_dt,
					"scheduled_duration_s": sched_s,
					"job_id": job_id if not args.runtime_no_submit else None,
					"submit_to_result_s": float(t_total) if t_total is not None else None,
				}
			)

	os.makedirs(args.out_dir, exist_ok=True)
	stamp = _utc_ts()
	json_path = os.path.join(args.out_dir, f"speed_benchmark_{stamp}.json")
	csv_path = os.path.join(args.out_dir, f"speed_benchmark_{stamp}.csv")

	payload = {
		"created_utc": stamp,
		"quasar_json": args.quasar_json,
		"indices": indices,
		"skipped": skipped,
		"compile_config": asdict(compile_cfg),
		"local": {
			"fake_backend": args.fake_backend,
			"local_no_sim": bool(args.local_no_sim),
			"shots": int(args.shots),
			"seed_simulator": int(args.seed_simulator),
		},
		"runtime": runtime_summary,
		"rows": rows,
	}
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, default=str)

	fieldnames: List[str] = []
	for r in rows:
		for k in r.keys():
			if k not in fieldnames:
				fieldnames.append(k)
	with open(csv_path, "w", encoding="utf-8", newline="") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for r in rows:
			w.writerow(r)

	print(json.dumps({"json": json_path, "csv": csv_path, "skipped": len(skipped)}, indent=2))


if __name__ == "__main__":
	main()
