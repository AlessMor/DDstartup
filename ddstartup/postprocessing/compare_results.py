"""
Lightweight multi-run comparison utility.

Usage:
    python -m ddstartup.postprocessing.compare_results \
        --files outputs/runA outputs/runB \
        --targets t_startup unrealized_profits \
        --inputs T_i tau_p_T n_tot \
        --out outputs/compare_runA_runB

Generates per-target overlay PDFs and quartile-probability plots across runs, plus CSV stats.
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

import h5py
from ddstartup.postprocessing.plot_utils_functions import ensure_registry
from ddstartup.postprocessing.plot_quartile_probability_functions import quartile_probability_plot
from ddstartup.utils.io_functions import h5_to_df_core, resolve_h5_inputs


def _shared_dataset_names(paths: list[Path]) -> set[str]:
    """Return intersection of dataset names across provided H5 files."""
    shared: set[str] | None = None
    for p in paths:
        with h5py.File(p, "r") as f:
            names = {k for k, v in f.items() if isinstance(v, h5py.Dataset) and v.ndim >= 1}
        shared = names if shared is None else shared & names
    return shared or set()


def _load_runs(files: Sequence[str | Path], columns: list[str], root: Path, resolved_paths: list[Path] | None = None) -> list[tuple[str, pd.DataFrame]]:
    """Load requested columns from each H5 and tag with run_id; skip missing targets gracefully."""
    runs: list[tuple[str, pd.DataFrame]] = []
    if resolved_paths is None:
        resolved, _ = resolve_h5_inputs(files, root=root)
    else:
        resolved = resolved_paths
    for p in resolved:
        run_id = p.stem
        df = h5_to_df_core(p, columns=columns)
        df["run_id"] = run_id
        runs.append((run_id, df))
    return runs


def _per_target_summary(df_all: pd.DataFrame, target: str) -> pd.DataFrame:
    """Compute simple stats per run for one target."""
    summaries = (
        df_all.groupby("run_id")[target]
        .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
        .reset_index()
    )
    return summaries


def _plot_pdf(df_all: pd.DataFrame, target: str, registry, outdir: Path, show_titles: bool) -> Path | None:
    plt.figure(figsize=(8, 5))
    plotted = False
    for run, sub in df_all.groupby("run_id"):
        vals = pd.to_numeric(sub[target], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        counts, edges = np.histogram(vals, bins="fd", density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.step(centers, counts, where="mid", label=run, alpha=0.9)
        plt.fill_between(centers, counts, step="mid", alpha=0.2)
        plotted = True
    if not plotted:
        plt.close()
        return None
    tlabel = registry.get_param_label(target) if registry else target
    if show_titles:
        plt.title(f"PDF of {tlabel} across runs")
    plt.xlabel(tlabel)
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()
    out = outdir / f"compare_pdf_{target}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _usable_inputs(df_all: pd.DataFrame, targets: list[str]) -> list[str]:
    """Return scalar, non-empty columns excluding targets/run_id/sol_success."""
    inner = (getattr(df_all, "attrs", {}) or {}).get("_inner_dims", {})
    usable: list[str] = []
    for col in df_all.columns:
        if col in targets or col in {"run_id", "sol_success"}:
            continue
        if int(inner.get(col, 1)) != 1:
            continue
        vals = pd.to_numeric(df_all[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() == 0 and df_all[col].dtype == object:
            # skip obvious vector/object columns even if inner_dims was lost
            sample = df_all[col].dropna().head(1)
            if any(isinstance(v, (list, np.ndarray)) for v in sample):
                continue
        if vals.notna().any():
            usable.append(col)
    return usable


def compare_runs(
    *,
    files: Sequence[str | Path],
    targets: list[str],
    inputs: list[str] | None,
    output_dir: Path,
    show_titles: bool = True,
    root: Path | None = None,
) -> None:
    registry = ensure_registry(None)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved, _ = resolve_h5_inputs(files, root=root or Path.cwd())
    shared_names = _shared_dataset_names(resolved)
    registry_inputs = set(registry.get_all_field_names())
    requested_inputs = set(inputs or []) or (shared_names & registry_inputs)
    columns = sorted(set(targets) | requested_inputs | {"sol_success"})

    runs = _load_runs(files, columns, root=root or Path.cwd(), resolved_paths=resolved)
    if not runs:
        raise FileNotFoundError("No H5 files resolved from --files")

    # concatenate
    df_all = pd.concat([df for _, df in runs], ignore_index=True)
    inner_dims = {}
    for _, df in runs:
        inner_dims.update((getattr(df, "attrs", {}) or {}).get("_inner_dims", {}))
    df_all.attrs["_inner_dims"] = inner_dims

    # Track which targets are present per run
    present_by_run: dict[str, set[str]] = {r: set(df.columns) for r, df in runs}
    candidate_inputs = _usable_inputs(df_all, targets)

    for target in targets:
        have = {r for r, cols in present_by_run.items() if target in cols}
        if len(have) < 2:
            print(f"⚠️  Skipping target '{target}': present in {len(have)} run(s) ({', '.join(sorted(have)) or 'none'}).")
            continue
        if target not in df_all.columns:
            print(f"⚠️  Target '{target}' missing in concatenated data; skipping.")
            continue
        # plots
        _plot_pdf(df_all, target, registry, output_dir, show_titles)
        # quartile-probability plots need some usable inputs
        inputs_for_qp = [c for c in candidate_inputs if c in df_all.columns and c not in targets]
        if inputs_for_qp:
            quartile_probability_plot(
                df=df_all,
                target=target,
                inputs=inputs_for_qp,
                target_unit=registry.get_unit(target) if registry else None,
                output_dir=output_dir,
                plot_name_prefix=f"compare_{target}",
                registry=registry,
                COUNT_FAILED=False,
                PLOT_STYLE="line",
                show_titles=show_titles,
            )
        else:
            print(f"⚠️  No usable inputs for quartile-probability plot of '{target}'.")
        # summary CSV
        summary = _per_target_summary(df_all, target)
        summary.to_csv(output_dir / f"compare_stats_{target}.csv", index=False)
        print(f"✅ Compared {target}: stats + plots saved.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare target distributions across multiple H5 runs.")
    p.add_argument("--files", "-f", nargs="+", required=True, help="H5 files or folders (accepts 'latest').")
    p.add_argument("--targets", "-t", nargs="+", required=True, help="Target variables to compare.")
    p.add_argument("--inputs", "-i", nargs="+", help="Optional input columns to load (for future extensions).")
    p.add_argument("--out", "-o", type=Path, default=Path("outputs") / "compare_runs", help="Output directory for comparison artifacts.")
    p.add_argument("--no-titles", action="store_true", help="Omit titles on plots.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    compare_runs(
        files=args.files,
        targets=args.targets,
        inputs=args.inputs,
        output_dir=args.out,
        show_titles=not args.no_titles,
        root=Path.cwd(),
    )


if __name__ == "__main__":
    main()
