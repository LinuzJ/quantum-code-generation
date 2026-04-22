import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: str) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _find_inputs(inputs: List[str]) -> List[str]:
	paths: List[str] = []
	for p in inputs:
		if os.path.isdir(p):
			paths.extend(sorted(glob.glob(os.path.join(p, "speed_benchmark_*.json"))))
		elif any(ch in p for ch in "*?["):
			paths.extend(sorted(glob.glob(p)))
		else:
			paths.append(p)
	# de-dup while preserving order
	seen = set()
	out = []
	for p in paths:
		if p not in seen:
			seen.add(p)
			out.append(p)
	return out


@dataclass(frozen=True)
class RowKey:
	index: int
	variant: str  # ground_truth | vista


@dataclass
class TimeRow:
	key: RowKey
	backend: str
	mode: str
	scheduled_s: float


@dataclass(frozen=True)
class SpeedupKey:
	index: int


@dataclass
class SpeedupRow:
	key: SpeedupKey
	backend: str
	gt_s: float
	vista_s: float
	ratio_gt_over_vista: float


def _iter_time_rows(payload: Dict[str, Any]) -> Iterable[TimeRow]:
	rows = payload.get("rows") or []
	for r in rows:
		try:
			mode = str(r.get("mode"))
			backend = str(r.get("backend"))
			variant = str(r.get("variant"))
			idx = int(r.get("index"))
			s = r.get("scheduled_duration_s")
			if s is None:
				continue
			s = float(s)
			if not np.isfinite(s) or s < 0:
				continue
		except Exception:
			continue

		if mode not in ("local_noisy", "runtime", "runtime_physical"):
			continue
		if variant not in ("ground_truth", "vista"):
			continue
		yield TimeRow(key=RowKey(index=idx, variant=variant), backend=backend, mode=mode, scheduled_s=s)


def _pair_fake_vs_real(
	* ,
	time_rows: Iterable[TimeRow],
	fake_backend: str,
	real_backend: str,
) -> List[Tuple[RowKey, float, float]]:
	"""Return pairs (key, fake_s, real_s) for keys seen on both backends."""
	by_key_backend: Dict[Tuple[RowKey, str], float] = {}
	for tr in time_rows:
		by_key_backend[(tr.key, tr.backend)] = tr.scheduled_s

	pairs: List[Tuple[RowKey, float, float]] = []
	keys = {k for (k, b) in by_key_backend.keys() if b in (fake_backend, real_backend)}
	for k in sorted(keys, key=lambda kk: (kk.index, kk.variant)):
		f = by_key_backend.get((k, fake_backend))
		r = by_key_backend.get((k, real_backend))
		if f is None or r is None:
			continue
		pairs.append((k, float(f), float(r)))
	return pairs


def _compute_speedups(*, time_rows: Iterable[TimeRow], backend: str) -> List[SpeedupRow]:
	"""Compute per-instance speedup using scheduled durations.

	Speedup is defined as GT/VISTA (ratio > 1 => VISTA faster).
	"""
	by_index_variant: Dict[Tuple[int, str], float] = {}
	for tr in time_rows:
		if tr.backend != backend:
			continue
		by_index_variant[(tr.key.index, tr.key.variant)] = float(tr.scheduled_s)

	out: List[SpeedupRow] = []
	indices = sorted({i for (i, v) in by_index_variant.keys() if v in ("ground_truth", "vista")})
	for idx in indices:
		gt = by_index_variant.get((idx, "ground_truth"))
		vs = by_index_variant.get((idx, "vista"))
		if gt is None or vs is None:
			continue
		if gt <= 0 or vs <= 0:
			continue
		ratio = float(gt) / float(vs)
		if not np.isfinite(ratio) or ratio <= 0:
			continue
		out.append(
			SpeedupRow(
				key=SpeedupKey(index=int(idx)),
				backend=str(backend),
				gt_s=float(gt),
				vista_s=float(vs),
				ratio_gt_over_vista=float(ratio),
			)
		)
	return out


def _configure_matplotlib() -> None:
	plt.rcParams["pdf.fonttype"] = 42
	plt.rcParams["ps.fonttype"] = 42
	plt.rcParams["font.size"] = 10
	plt.rcParams["axes.titlesize"] = 10
	plt.rcParams["axes.labelsize"] = 10
	plt.rcParams["legend.fontsize"] = 9
	plt.rcParams["xtick.labelsize"] = 9
	plt.rcParams["ytick.labelsize"] = 9


def plot_scatter_fake_vs_real(
	*,
	pairs: List[Tuple[RowKey, float, float]],
	fake_backend: str,
	real_backend: str,
	out_path: str,
	log_scale: bool,
) -> None:
	"""Scatter plot of physical duration: fake (x) vs real (y)."""
	_configure_matplotlib()
	fig, ax = plt.subplots(figsize=(3.6, 3.2))

	def _to_us(x: float) -> float:
		return x * 1e6

	x_gt, y_gt, x_vs, y_vs = [], [], [], []
	for k, f, r in pairs:
		if k.variant == "ground_truth":
			x_gt.append(_to_us(f))
			y_gt.append(_to_us(r))
		else:
			x_vs.append(_to_us(f))
			y_vs.append(_to_us(r))

	if x_gt:
		ax.scatter(x_gt, y_gt, s=14, marker="o", label="Ground-truth", alpha=0.8)
	if x_vs:
		ax.scatter(x_vs, y_vs, s=14, marker="^", label="VISTA", alpha=0.8)

	all_x = np.asarray(x_gt + x_vs, dtype=float)
	all_y = np.asarray(y_gt + y_vs, dtype=float)
	if all_x.size and all_y.size:
		lo = float(min(all_x.min(), all_y.min()))
		hi = float(max(all_x.max(), all_y.max()))
		if log_scale:
			# avoid non-positive limits
			lo = max(lo, 1e-3)
			ax.set_xscale("log")
			ax.set_yscale("log")
		# Diagonal reference
		ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="0.5")
		ax.set_xlim(lo, hi)
		ax.set_ylim(lo, hi)

	ax.set_xlabel(f"{fake_backend} scheduled duration (µs)")
	ax.set_xlabel(f"{fake_backend} execution time (µs)")
	ax.set_ylabel(f"{real_backend} execution time (µs)")
	ax.legend(frameon=False, loc="best")
	ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def plot_boxplot_by_backend(
	*,
	time_rows: List[TimeRow],
	fake_backend: str,
	real_backend: str,
	out_path: str,
) -> None:
	"""Boxplot of scheduled durations by backend and variant."""
	_configure_matplotlib()
	fig, ax = plt.subplots(figsize=(4.0, 3.0))

	def _collect(backend: str, variant: str) -> List[float]:
		out: List[float] = []
		for tr in time_rows:
			if tr.backend != backend:
				continue
			if tr.key.variant != variant:
				continue
			out.append(tr.scheduled_s * 1e6)
		return out

	data = [
		_collect(fake_backend, "ground_truth"),
		_collect(fake_backend, "vista"),
		_collect(real_backend, "ground_truth"),
		_collect(real_backend, "vista"),
	]
	labels = [
		f"{fake_backend}\nGround-truth",
		f"{fake_backend}\nVISTA",
		f"{real_backend}\nGround-truth",
		f"{real_backend}\nVISTA",
	]

	ax.boxplot(
		data,
		tick_labels=labels,
		showfliers=False,
		medianprops={"color": "black", "linewidth": 1.0},
		boxprops={"linewidth": 1.0},
		whiskerprops={"linewidth": 1.0},
		capprops={"linewidth": 1.0},
	)
	ax.set_ylabel("Execution time (µs)", fontsize=12)
	ax.tick_params(axis="x", which="major", labelsize=10)
	ax.tick_params(axis="y", which="major", labelsize=10)
	ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def plot_speedup_boxplot(
	*,
	fake_speedups: List[SpeedupRow],
	real_speedups: List[SpeedupRow],
	fake_backend: str,
	real_backend: str,
	out_path: str,
) -> None:
	"""Boxplot of per-instance speedup ratio GT/VISTA for each backend."""
	_configure_matplotlib()
	fig, ax = plt.subplots(figsize=(3.9, 3.0))

	data = [
		[sr.ratio_gt_over_vista for sr in fake_speedups],
		[sr.ratio_gt_over_vista for sr in real_speedups],
	]
	labels = [
		f"{fake_backend}\nGT/VISTA",
		f"{real_backend}\nGT/VISTA",
	]

	ax.boxplot(
		data,
		tick_labels=labels,
		showfliers=False,
		medianprops={"color": "black", "linewidth": 1.0},
		boxprops={"linewidth": 1.0},
		whiskerprops={"linewidth": 1.0},
		capprops={"linewidth": 1.0},
	)
	ax.axhline(1.0, linestyle="--", linewidth=1, color="0.5")
	ax.set_ylabel("Speedup ratio (Ground-truth / VISTA)")
	ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def plot_speedup_scatter_fake_vs_real(
	*,
	fake_speedups: List[SpeedupRow],
	real_speedups: List[SpeedupRow],
	fake_backend: str,
	real_backend: str,
	out_path: str,
) -> None:
	"""Scatter of per-instance speedup ratio: fake (x) vs real (y)."""
	_configure_matplotlib()
	fig, ax = plt.subplots(figsize=(3.6, 3.2))

	by_idx_fake = {sr.key.index: sr.ratio_gt_over_vista for sr in fake_speedups}
	by_idx_real = {sr.key.index: sr.ratio_gt_over_vista for sr in real_speedups}
	idxs = sorted(set(by_idx_fake.keys()) & set(by_idx_real.keys()))
	x = np.asarray([by_idx_fake[i] for i in idxs], dtype=float)
	y = np.asarray([by_idx_real[i] for i in idxs], dtype=float)

	ax.scatter(x, y, s=14, marker="o", alpha=0.85)
	if x.size and y.size:
		lo = float(min(x.min(), y.min(), 1.0))
		hi = float(max(x.max(), y.max(), 1.0))
		ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="0.5")
		ax.set_xlim(lo, hi)
		ax.set_ylim(lo, hi)

	ax.set_xlabel(f"Speedup on {fake_backend} (GT/VISTA)")
	ax.set_ylabel(f"Speedup on {real_backend} (GT/VISTA)")
	ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Plot physical execution-time comparison from speed benchmark JSON outputs")
	p.add_argument(
		"--inputs",
		nargs="+",
		required=True,
		help="One or more JSON files, a directory containing speed_benchmark_*.json, or a glob.",
	)
	p.add_argument("--out-dir", default="out_speed/plots_time")
	p.add_argument("--fake-backend", default="FakeTorino")
	p.add_argument("--real-backend", default="ibm_torino")
	p.add_argument("--log-scale", action="store_true", help="Use log-log axes for scatter")
	return p


def main() -> None:
	args = build_arg_parser().parse_args()
	paths = _find_inputs(list(args.inputs))
	if not paths:
		raise SystemExit("No inputs found")

	all_rows: List[TimeRow] = []
	for p in paths:
		payload = _load_json(p)
		all_rows.extend(list(_iter_time_rows(payload)))

	if not all_rows:
		raise SystemExit("No scheduled_duration_s rows found in inputs")

	pairs = _pair_fake_vs_real(
		time_rows=all_rows,
		fake_backend=str(args.fake_backend),
		real_backend=str(args.real_backend),
	)
	if not pairs:
		raise SystemExit(
			"No paired rows found across both backends. "
			"Ensure your benchmark output includes both local_noisy (FakeTorino) and runtime/runtime_physical (ibm_torino) rows."
		)

	os.makedirs(args.out_dir, exist_ok=True)
	scatter_path = os.path.join(args.out_dir, "scheduled_duration_scatter.pdf")
	box_path = os.path.join(args.out_dir, "scheduled_duration_boxplot.pdf")
	speedup_box_path = os.path.join(args.out_dir, "scheduled_duration_speedup_boxplot.pdf")
	speedup_scatter_path = os.path.join(args.out_dir, "scheduled_duration_speedup_scatter.pdf")

	plot_scatter_fake_vs_real(
		pairs=pairs,
		fake_backend=str(args.fake_backend),
		real_backend=str(args.real_backend),
		out_path=scatter_path,
		log_scale=bool(args.log_scale),
	)
	plot_boxplot_by_backend(
		time_rows=all_rows,
		fake_backend=str(args.fake_backend),
		real_backend=str(args.real_backend),
		out_path=box_path,
	)

	fake_speedups = _compute_speedups(time_rows=all_rows, backend=str(args.fake_backend))
	real_speedups = _compute_speedups(time_rows=all_rows, backend=str(args.real_backend))
	if fake_speedups and real_speedups:
		plot_speedup_boxplot(
			fake_speedups=fake_speedups,
			real_speedups=real_speedups,
			fake_backend=str(args.fake_backend),
			real_backend=str(args.real_backend),
			out_path=speedup_box_path,
		)
		plot_speedup_scatter_fake_vs_real(
			fake_speedups=fake_speedups,
			real_speedups=real_speedups,
			fake_backend=str(args.fake_backend),
			real_backend=str(args.real_backend),
			out_path=speedup_scatter_path,
		)

	print(
		json.dumps(
			{
				"scatter": scatter_path,
				"boxplot": box_path,
				"speedup_boxplot": speedup_box_path,
				"speedup_scatter": speedup_scatter_path,
				"n_pairs": len(pairs),
				"n_speedup_fake": len(fake_speedups),
				"n_speedup_real": len(real_speedups),
			},
			indent=2,
		)
	)


if __name__ == "__main__":
	main()
