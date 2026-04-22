import argparse
import csv
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _safe_get(d: Dict[str, Any], path: Iterable[str], default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


@dataclass
class Row:
    index: int
    fake_backend: str
    sim_mode: str
    shots: Optional[int]
    seed_simulator: Optional[int]

    ref_energy: Optional[float]
    ref_sol_p: Optional[float]
    ref_depth: Optional[float]
    ref_twoq: Optional[float]
    ref_size: Optional[float]

    q_energy: Optional[float]
    q_sol_p: Optional[float]
    q_depth: Optional[float]
    q_twoq: Optional[float]
    q_size: Optional[float]

    delta_energy_q_minus_ref: Optional[float]
    delta_sol_p_q_minus_ref: Optional[float]
    delta_depth_q_minus_ref: Optional[float]
    delta_twoq_q_minus_ref: Optional[float]
    delta_size_q_minus_ref: Optional[float]

    # Closeness to Ground-truth (absolute differences)
    min_energy: Optional[float]
    gt_gap: Optional[float]
    vista_gap: Optional[float]
    abs_energy_diff: Optional[float]
    abs_gap_diff: Optional[float]
    abs_sol_p_diff: Optional[float]


def _load_rows(in_dir: str) -> List[Row]:
    paths = sorted(glob.glob(os.path.join(in_dir, "instance_[0-9]*.json")))
    rows: List[Row] = []

    skipped = 0

    for p in paths:
        try:
            data = _read_json(p)
        except Exception as e:
            skipped += 1
            print(f"[WARN] Skipping unreadable JSON: {p} ({e})", file=sys.stderr)
            continue

        idx_val = data.get("index", None)
        if idx_val is None:
            m = re.search(r"instance_(\d+)\.json$", os.path.basename(p))
            if m:
                idx_val = int(m.group(1))
        try:
            idx = int(idx_val)
        except Exception:
            skipped += 1
            print(f"[WARN] Skipping file with missing index: {p}", file=sys.stderr)
            continue
        backend = data.get("fake_backend")
        if backend is None:
            backend = data.get("real_backend")
        backend = str(backend)
        sim_mode = str(_safe_get(data, ["simulation", "mode"], "unknown"))
        shots = _safe_get(data, ["simulation", "shots"], None)
        seed_sim = _safe_get(data, ["simulation", "seed_simulator"], None)

        ref_energy = _as_float(_safe_get(data, ["reference", "algorithmic", "energy"], None))
        ref_sol_p = _as_float(_safe_get(data, ["reference", "algorithmic", "solution_probability"], None))
        ref_min_e = _as_float(_safe_get(data, ["reference", "algorithmic", "min_energy"], None))
        ref_depth = _as_float(_safe_get(data, ["reference", "hardware", "depth"], None))
        ref_twoq = _as_float(_safe_get(data, ["reference", "hardware", "two_qubit_gates"], None))
        ref_size = _as_float(_safe_get(data, ["reference", "hardware", "size"], None))

        q_energy = _as_float(_safe_get(data, ["quasar", "algorithmic", "energy"], None))
        q_sol_p = _as_float(_safe_get(data, ["quasar", "algorithmic", "solution_probability"], None))
        q_min_e = _as_float(_safe_get(data, ["quasar", "algorithmic", "min_energy"], None))
        q_depth = _as_float(_safe_get(data, ["quasar", "hardware", "depth"], None))
        q_twoq = _as_float(_safe_get(data, ["quasar", "hardware", "two_qubit_gates"], None))
        q_size = _as_float(_safe_get(data, ["quasar", "hardware", "size"], None))

        # Use whichever min_energy is present (they should match)
        min_energy = ref_min_e if ref_min_e is not None else q_min_e

        def gap(e: Optional[float], e_star: Optional[float]) -> Optional[float]:
            if e is None or e_star is None:
                return None
            return float(e - e_star)

        gt_gap = gap(ref_energy, min_energy)
        vista_gap = gap(q_energy, min_energy)

        abs_energy_diff = abs(q_energy - ref_energy) if (q_energy is not None and ref_energy is not None) else None
        abs_gap_diff = abs(vista_gap - gt_gap) if (vista_gap is not None and gt_gap is not None) else None
        abs_sol_p_diff = abs(q_sol_p - ref_sol_p) if (q_sol_p is not None and ref_sol_p is not None) else None

        def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
            # a - b
            if a is None or b is None:
                return None
            return float(a - b)

        rows.append(
            Row(
                index=idx,
                fake_backend=backend,
                sim_mode=sim_mode,
                shots=int(shots) if shots is not None else None,
                seed_simulator=int(seed_sim) if seed_sim is not None else None,
                ref_energy=ref_energy,
                ref_sol_p=ref_sol_p,
                ref_depth=ref_depth,
                ref_twoq=ref_twoq,
                ref_size=ref_size,
                q_energy=q_energy,
                q_sol_p=q_sol_p,
                q_depth=q_depth,
                q_twoq=q_twoq,
                q_size=q_size,
                delta_energy_q_minus_ref=delta(q_energy, ref_energy),
                delta_sol_p_q_minus_ref=delta(q_sol_p, ref_sol_p),
                delta_depth_q_minus_ref=delta(q_depth, ref_depth),
                delta_twoq_q_minus_ref=delta(q_twoq, ref_twoq),
                delta_size_q_minus_ref=delta(q_size, ref_size),

                min_energy=min_energy,
                gt_gap=gt_gap,
                vista_gap=vista_gap,
                abs_energy_diff=abs_energy_diff,
                abs_gap_diff=abs_gap_diff,
                abs_sol_p_diff=abs_sol_p_diff,
            )
        )

    if skipped:
        print(f"[INFO] Loaded {len(rows)} rows; skipped {skipped} files", file=sys.stderr)
    return rows


def _write_csv(rows: List[Row], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = [f.name for f in Row.__dataclass_fields__.values()]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


def _finite(values: Iterable[Optional[float]]) -> np.ndarray:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return arr


def _summary_stats(rows: List[Row]) -> Dict[str, Any]:
    def summarize(arr: np.ndarray) -> Dict[str, Any]:
        if arr.size == 0:
            return {"n": 0}
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0,
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }

    dE = _finite(r.delta_energy_q_minus_ref for r in rows)
    dP = _finite(r.delta_sol_p_q_minus_ref for r in rows)
    dD = _finite(r.delta_depth_q_minus_ref for r in rows)
    d2 = _finite(r.delta_twoq_q_minus_ref for r in rows)

    abs_dE = _finite(r.abs_energy_diff for r in rows)
    abs_dGap = _finite(r.abs_gap_diff for r in rows)
    abs_dP = _finite(r.abs_sol_p_diff for r in rows)

    # Directional “wins” (lower energy is better; higher solution prob better; lower depth/2Q better)
    wins = {
        "energy_quasar_better_frac": float(np.mean(dE < 0)) if dE.size else None,
        "sol_p_quasar_better_frac": float(np.mean(dP > 0)) if dP.size else None,
        "depth_quasar_better_frac": float(np.mean(dD < 0)) if dD.size else None,
        "twoq_quasar_better_frac": float(np.mean(d2 < 0)) if d2.size else None,
    }

    by_mode: Dict[str, Any] = {}
    modes = sorted({r.sim_mode for r in rows})
    for m in modes:
        sub = [r for r in rows if r.sim_mode == m]
        by_mode[m] = {
            "n_instances": int(len(sub)),
            "delta_energy_q_minus_ref": summarize(_finite(r.delta_energy_q_minus_ref for r in sub)),
            "delta_sol_p_q_minus_ref": summarize(_finite(r.delta_sol_p_q_minus_ref for r in sub)),
            "delta_depth_q_minus_ref": summarize(_finite(r.delta_depth_q_minus_ref for r in sub)),
            "delta_twoq_q_minus_ref": summarize(_finite(r.delta_twoq_q_minus_ref for r in sub)),
        }

    return {
        "n_instances": int(len(rows)),
        "unique_backends": sorted({r.fake_backend for r in rows}),
        "unique_sim_modes": modes,
        "delta_energy_q_minus_ref": summarize(dE),
        "delta_sol_p_q_minus_ref": summarize(dP),
        "delta_depth_q_minus_ref": summarize(dD),
        "delta_twoq_q_minus_ref": summarize(d2),
        "abs_energy_diff": summarize(abs_dE),
        "abs_gap_diff": summarize(abs_dGap),
        "abs_sol_p_diff": summarize(abs_dP),
        "wins": wins,
        "by_mode": by_mode,
    }


def _write_json(obj: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _scatter_with_diag(
    plt,
    x: np.ndarray,
    y: np.ndarray,
    title: Optional[str],
    xlabel: str,
    ylabel: str,
    out_path: str,
    *,
    show_grid: bool = False,
    scatter_color: Optional[str] = None,
    diag_color: Optional[str] = None,
    label_fontsize: Optional[float] = None,
    tick_fontsize: Optional[float] = None,
    title_fontsize: Optional[float] = None,
):
    fig = plt.figure(figsize=(3.35, 3.35))
    ax = fig.add_subplot(1, 1, 1)
    scatter_kwargs = {"s": 7, "alpha": 0.65, "linewidths": 0}
    if scatter_color is not None:
        scatter_kwargs["color"] = scatter_color
    ax.scatter(x, y, **scatter_kwargs)
    mn = float(np.min([np.min(x), np.min(y)]))
    mx = float(np.max([np.max(x), np.max(y)]))
    diag_kwargs = {"linestyle": "--", "linewidth": 0.8}
    if diag_color is not None:
        diag_kwargs["color"] = diag_color
    ax.plot([mn, mx], [mn, mx], **diag_kwargs)
    if title:
        if title_fontsize is None:
            ax.set_title(title)
        else:
            ax.set_title(title, fontsize=title_fontsize)

    if label_fontsize is None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)

    if tick_fontsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    if show_grid:
        ax.grid(True, linewidth=0.3, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _hist(plt, vals: np.ndarray, title: str, xlabel: str, out_path: str):
    fig = plt.figure(figsize=(4.2, 3.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(vals, bins=40, alpha=0.85)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot(rows: List[Row], plots_dir: str, fmt: str) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        raise RuntimeError(
            "matplotlib is not installed. Install it (e.g., pip install matplotlib) or run only --aggregate."
        )

    os.makedirs(plots_dir, exist_ok=True)

    # Paper-friendly defaults (safe fallbacks if fonts unavailable).
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    def paired(a: Iterable[Optional[float]], b: Iterable[Optional[float]]) -> Tuple[np.ndarray, np.ndarray]:
        ax: List[float] = []
        bx: List[float] = []
        for x, y in zip(a, b):
            if x is None or y is None:
                continue
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax.append(float(x))
            bx.append(float(y))
        return np.array(ax, dtype=float), np.array(bx, dtype=float)

    gt_label = "Ground-truth"
    vista_label = "VISTA"

    # Scatter: Ground-truth vs VISTA (algorithmic)
    refE, qE = paired((r.ref_energy for r in rows), (r.q_energy for r in rows))
    if refE.size:
        _scatter_with_diag(
            plt,
            refE,
            qE,
            f"Energy: {gt_label} vs {vista_label}",
            f"{gt_label} energy",
            f"{vista_label} energy",
            os.path.join(plots_dir, f"energy_scatter.{fmt}"),
        )
        _hist(
            plt,
            qE - refE,
            f"Δ energy ({vista_label} − {gt_label})",
            "delta energy",
            os.path.join(plots_dir, f"delta_energy_hist.{fmt}"),
        )

    refP, qP = paired((r.ref_sol_p for r in rows), (r.q_sol_p for r in rows))
    if refP.size:
        _scatter_with_diag(
            plt,
            refP,
            qP,
            f"Solution prob: {gt_label} vs {vista_label}",
            f"{gt_label} solution prob",
            f"{vista_label} solution prob",
            os.path.join(plots_dir, f"solution_prob_scatter.{fmt}"),
        )
        _hist(
            plt,
            qP - refP,
            f"Δ solution prob ({vista_label} − {gt_label})",
            "delta solution prob",
            os.path.join(plots_dir, f"delta_solution_prob_hist.{fmt}"),
        )

    # Scatter: hardware
    refD, qD = paired((r.ref_depth for r in rows), (r.q_depth for r in rows))
    if refD.size:
        _scatter_with_diag(
            plt,
            refD,
            qD,
            f"Depth: {gt_label} vs {vista_label}",
            f"{gt_label} depth",
            f"{vista_label} depth",
            os.path.join(plots_dir, f"depth_scatter.{fmt}"),
        )
        _hist(
            plt,
            qD - refD,
            f"Δ depth ({vista_label} − {gt_label})",
            "delta depth",
            os.path.join(plots_dir, f"delta_depth_hist.{fmt}"),
        )

    ref2, q2 = paired((r.ref_twoq for r in rows), (r.q_twoq for r in rows))
    if ref2.size:
        _scatter_with_diag(
            plt,
            ref2,
            q2,
            f"2Q gates: {gt_label} vs {vista_label}",
            f"{gt_label} 2Q",
            f"{vista_label} 2Q",
            os.path.join(plots_dir, f"twoq_scatter.{fmt}"),
        )
        _hist(
            plt,
            q2 - ref2,
            f"Δ 2Q ({vista_label} − {gt_label})",
            "delta 2Q gates",
            os.path.join(plots_dir, f"delta_twoq_hist.{fmt}"),
        )

    # Tradeoff: algorithmic vs hardware (VISTA)
    if qE.size and q2.size:
        # Align by rows again to avoid mismatch
        xs: List[float] = []
        ys: List[float] = []
        for r in rows:
            if r.q_energy is None or r.q_twoq is None:
                continue
            if not (np.isfinite(r.q_energy) and np.isfinite(r.q_twoq)):
                continue
            xs.append(float(r.q_twoq))
            ys.append(float(r.q_energy))
        xarr = np.array(xs, dtype=float)
        yarr = np.array(ys, dtype=float)
        if xarr.size:
            fig = plt.figure(figsize=(4.2, 3.2))
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(xarr, yarr, s=8, alpha=0.6)
            ax.set_title(f"{vista_label}: energy vs 2Q")
            ax.set_xlabel(f"2Q gates ({vista_label})")
            ax.set_ylabel(f"energy ({vista_label})")
            ax.grid(True, linewidth=0.3, alpha=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"quasar_energy_vs_twoq.{fmt}"))
            plt.close(fig)

    # Closeness plots (behavioral similarity)
    gt_gap, vista_gap = paired((r.gt_gap for r in rows), (r.vista_gap for r in rows))
    if gt_gap.size:
        _scatter_with_diag(
            plt,
            gt_gap,
            vista_gap,
            None,
            f"{gt_label} gap $E - E^*$",
            f"{vista_label} gap $E - E^*$",
            os.path.join(plots_dir, f"gap_scatter.{fmt}"),
            scatter_color="black",
            diag_color="black",
            label_fontsize=12,
            tick_fontsize=11,
        )

    abs_gap = _finite(r.abs_gap_diff for r in rows)
    if abs_gap.size:
        _hist(
            plt,
            abs_gap,
            f"|Δ gap| = |({vista_label}−E*)−({gt_label}−E*)|",
            "absolute gap difference",
            os.path.join(plots_dir, f"abs_gap_diff_hist.{fmt}"),
        )

    abs_dp = _finite(r.abs_sol_p_diff for r in rows)
    if abs_dp.size:
        _hist(
            plt,
            abs_dp,
            f"|Δ solution prob| = |{vista_label}−{gt_label}|",
            "absolute solution probability difference",
            os.path.join(plots_dir, f"abs_solution_prob_diff_hist.{fmt}"),
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate and plot per-instance evaluation results (out_instance_eval/instance_*.json)."
    )
    p.add_argument("--in-dir", default="out_instance_eval")
    p.add_argument("--out-csv", default="out_instance_eval/instance_eval_aggregate.csv")
    p.add_argument("--out-stats", default="out_instance_eval/instance_eval_stats.json")
    p.add_argument("--plots-dir", default="out_instance_eval/plots")
    p.add_argument("--plot-format", choices=["pdf", "png"], default="pdf")
    p.add_argument("--no-plots", action="store_true", help="Only aggregate + stats; skip plotting")
    return p


def main():
    args = build_parser().parse_args()
    rows = _load_rows(args.in_dir)
    if not rows:
        raise SystemExit(f"No instance_*.json found in {args.in_dir}")

    _write_csv(rows, args.out_csv)
    stats = _summary_stats(rows)
    _write_json(stats, args.out_stats)

    if not args.no_plots:
        _plot(rows, args.plots_dir, args.plot_format)

    print(json.dumps({"n_instances": len(rows), "out_csv": args.out_csv, "out_stats": args.out_stats, "plots_dir": args.plots_dir}, indent=2))


if __name__ == "__main__":
    main()
