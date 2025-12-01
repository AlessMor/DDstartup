"""
Lightweight multi-run comparison utility.

Usage:
    python -m ddstartup.postprocessing.compare_results \
        --files outputs/runA outputs/runB \
        --targets t_startup unrealized_profits \
        --inputs T_i tau_p_T n_tot \
        --out outputs/compare_runA_runB

Generates per-target overlay KDE/box plots and CSV summaries across runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ddstartup.postprocessing.plot_utils_functions import ensure_registry
from ddstartup.utils.io_functions import h5_to_df_core, resolve_h5_inputs


def _load_runs(files: Sequence[str | Path], inputs: list[str] | None, targets: list[str], root: Path) -> list[tuple[str, pd.DataFrame]]:
    """Load requested columns from each H5 and tag with run_id; skip missing targets gracefully."""
    runs: list[tuple[str, pd.DataFrame]] = []
    resolved, _ = resolve_h5_inputs(files, root=root)
    for p in resolved:
        run_id = p.stem
        cols = list(set((inputs or []) + targets))
        df = h5_to_df_core(p, columns=cols)
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


def _plot_kde(df_all: pd.DataFrame, target: str, registry, outdir: Path, show_titles: bool) -> Path:
    plt.figure(figsize=(8, 5))
    for run, sub in df_all.groupby("run_id"):
        vals = pd.to_numeric(sub[target], errors="coerce").dropna()
        if vals.empty:
            continue
        sns.kdeplot(vals, fill=True, alpha=0.3, label=run)
    tlabel = registry.get_param_label(target) if registry else target
    if show_titles:
        plt.title(f"KDE of {tlabel} across runs")
    plt.xlabel(tlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    out = outdir / f"compare_kde_{target}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _plot_box(df_all: pd.DataFrame, target: str, registry, outdir: Path, show_titles: bool) -> Path:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df_all, x="run_id", y=target)
    tlabel = registry.get_param_label(target) if registry else target
    plt.xlabel("Run")
    plt.ylabel(tlabel)
    if show_titles:
        plt.title(f"Distribution of {tlabel} per run")
    plt.tight_layout()
    out = outdir / f"compare_box_{target}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


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
    runs = _load_runs(files, inputs, targets, root=root or Path.cwd())
    if not runs:
        raise FileNotFoundError("No H5 files resolved from --files")

    # concatenate
    df_all = pd.concat([df for _, df in runs], ignore_index=True)

    # Track which targets are present per run
    present_by_run: dict[str, set[str]] = {r: set(df.columns) for r, df in runs}

    for target in targets:
        have = {r for r, cols in present_by_run.items() if target in cols}
        if len(have) < 2:
            print(f"⚠️  Skipping target '{target}': present in {len(have)} run(s) ({', '.join(sorted(have)) or 'none'}).")
            continue
        if target not in df_all.columns:
            print(f"⚠️  Target '{target}' missing in concatenated data; skipping.")
            continue
        # plots
        _plot_kde(df_all, target, registry, output_dir, show_titles)
        _plot_box(df_all, target, registry, output_dir, show_titles)
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
