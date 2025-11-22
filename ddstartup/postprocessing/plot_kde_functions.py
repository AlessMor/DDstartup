"""
KDE (Kernel Density Estimation) Plot Functions

Generates KDE plots split by quartiles of a scalar target variable.
- Works with the new DF format: scalar columns are numeric; vector columns are object
  series containing per-row 1D numpy arrays (skipped here).
- Uses df.attrs["_inner_dims"] to detect scalar vs vector columns.
- Backward-compatible argument names (df/df_filtered, inputs/input_parameters,
  output_dir/outputs_dir, plot_name_prefix/plot_name).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ddstartup.postprocessing.postprocess_functions import get_discrete_colorscale


def _resolve_args(
    *,
    df=None,
    df_filtered=None,
    inputs=None,
    input_parameters=None,
    output_dir=None,
    outputs_dir=None,
    plot_name_prefix=None,
    plot_name=None,
    **_
):
    """Normalize old/new argument names to a single set."""
    _df = df_filtered if df is None else df
    _inputs = input_parameters if inputs is None else inputs
    _outdir = outputs_dir if output_dir is None else output_dir

    # Build filename stem once. Prefer explicit plot_name_prefix over legacy plot_name.
    if plot_name_prefix:
        stem = f"{plot_name_prefix}_kde_by_quartile"
    elif plot_name:
        # strip extension if a file-like name was passed
        stem = Path(plot_name).stem
    else:
        stem = "kde_by_quartile"

    return _df, _inputs, Path(_outdir), stem


def _scalar_input_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Pick scalar columns among candidates (inner_dim==1 and numeric dtype)."""
    inner = (df.attrs or {}).get("_inner_dims", {})
    out = []
    for c in candidates:
        if c not in df.columns:
            continue
        if int(inner.get(c, 1)) != 1:
            # vector-valued input; this plot handles only scalars
            continue
        # keep numeric columns only
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def _varying_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Remove near-constant columns to avoid degenerate KDEs."""
    keep = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        # robust constant check
        try:
            std = float(s.std())
            mean = abs(float(s.mean()))
        except Exception:
            continue
        if std > 1e-10 and (mean == 0 or std / max(mean, 1e-30) > 1e-6):
            keep.append(c)
        else:
            # silently skip constants; the caller already logs per-target summaries
            pass
    return keep


def _quartile_bins(series: pd.Series) -> tuple[pd.Series, list[str]]:
    """Return (categorical bins, ordered label list) for quartiles of a numeric series."""
    y = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if y.empty:
        # produce empty bins with default labels (the caller will bail out upstream)
        labels = ["Q1", "Q2", "Q3", "Q4"]
        return pd.Categorical([np.nan] * len(series), categories=labels), labels
    qs = series.quantile([0, 0.25, 0.5, 0.75, 1.0])
    labels = [
        f"Q1: {qs.iloc[0]:.2e}–{qs.iloc[1]:.2e}",
        f"Q2: {qs.iloc[1]:.2e}–{qs.iloc[2]:.2e}",
        f"Q3: {qs.iloc[2]:.2e}–{qs.iloc[3]:.2e}",
        f"Q4: {qs.iloc[3]:.2e}–{qs.iloc[4]:.2e}",
    ]
    bins = pd.qcut(series, q=4, labels=labels, duplicates="drop")
    # If duplicates collapsed (<4 bins), regenerate labels to the actual count
    if bins.dtype == "category" and len(bins.cat.categories) != 4:
        cats = list(bins.cat.categories)
        return bins, cats
    return bins, labels


def _save_quartile_extremes_to_csv(
    df: pd.DataFrame,
    target: str,
    inputs: list[str],
    bins: pd.Series,
    bin_labels: list[str],
    outdir: Path,
    stem: str,
):
    """Write one CSV with lowest and middle rows per quartile, keeping inputs (+ optional t_startup)."""
    rows = []
    # Align bins to df index
    bcol = pd.Series(bins.values, index=df.index, name="_bin")
    for label in bin_labels:
        sel = df.index[bcol == label]
        if sel.size == 0:
            continue
        sub = df.loc[sel].sort_values(by=target)
        lo = sub.iloc[0]
        mid = sub.iloc[len(sub) // 2]
        for which, row in (("lowest", lo), ("middle", mid)):
            rec = {"quartile": label, "which": which, target: row[target]}
            for p in inputs:
                rec[p] = row.get(p, np.nan)
            if "t_startup" in df.columns:
                rec["t_startup"] = row.get("t_startup", np.nan)
            rows.append(rec)
    if not rows:
        return
    csv_df = pd.DataFrame(rows)
    csv_path = outdir / f"{stem}__quartile_values__{target}.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"   Quartile values saved: {csv_path.name}")


def kde_quartile_plot(
    *,
    df=None,
    df_filtered=None,
    target: str,
    inputs=None,
    input_parameters=None,
    target_unit: str | None = None,
    output_dir=None,
    outputs_dir=None,
    file_type: str = "",
    plot_name_prefix: str | None = None,
    plot_name: str | None = None,
    registry=None,
    **_,
):
    """
    Create KDE plots for scalar inputs, split by quartiles of the scalar target.

    Expected modern call (from orchestrator):
        kde_quartile_plot(
            df=..., target=..., inputs=..., target_unit=..., output_dir=...,
            file_type=..., plot_name_prefix=..., registry=...
        )

    Legacy names (df_filtered, input_parameters, outputs_dir, plot_name) are also accepted.
    """
    # Registry
    if registry is None:
        from ddstartup.utils.parameter_registry import get_registry as _get_registry
        registry = _get_registry()

    # Normalize incoming args
    df, inputs, outdir, stem = _resolve_args(
        df=df,
        df_filtered=df_filtered,
        inputs=inputs,
        input_parameters=input_parameters,
        output_dir=output_dir,
        outputs_dir=outputs_dir,
        plot_name_prefix=plot_name_prefix,
        plot_name=plot_name,
    )

    if df is None or len(df) == 0:
        print("   No data provided to KDE plot. Skipping.")
        return

    # Select scalar, numeric inputs only; drop near-constants
    scalar_inputs = _scalar_input_columns(df, list(inputs or []))
    if not scalar_inputs:
        print("   No scalar numeric inputs available. Skipping KDE plot.")
        return
    varying_inputs = _varying_columns(df, scalar_inputs)
    if not varying_inputs:
        print("   No varying scalar inputs to plot. Skipping KDE plot.")
        return

    # Quartile binning (do not mutate df)
    bins, labels = _quartile_bins(df[target])
    if isinstance(bins, pd.Series):
        valid_mask = bins.notna()
    else:
        valid_mask = pd.Series([False] * len(df))

    if not valid_mask.any():
        print(f"   Target '{target}' has no valid finite values for quartiles. Skipping KDE plot.")
        return

    # Colors per quartile
    colorscale = get_discrete_colorscale(4 if len(labels) >= 4 else len(labels))
    quartile_colors = [colorscale[i * 2][1] for i in range(len(labels))]
    color_map = {labels[i]: quartile_colors[i] for i in range(len(labels))}

    # Grid size
    n_inputs = len(varying_inputs)
    ncols = min(3, n_inputs)
    nrows = int(np.ceil(n_inputs / ncols))

    # Figure / gridspec
    from matplotlib import gridspec
    height_ratios = [0.07] + [1] * nrows + [0.45]
    fig = plt.figure(figsize=(4 * ncols, 3 * nrows + 2))
    gs = gridspec.GridSpec(
        nrows=nrows + 2,
        ncols=ncols,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.4,
        wspace=0.3,
    )

    # Create axes for data plots (skip first and last rows)
    axes = []
    for r in range(1, nrows + 1):
        for c in range(ncols):
            axes.append(fig.add_subplot(gs[r, c]))

    # Plot KDEs per input by quartile
    # If a bin collapses to a single value for an input, draw a dashed line at y=1 as a visual placeholder.
    for i, param in enumerate(varying_inputs):
        ax = axes[i]
        for label in labels:
            sel = (bins == label)
            if sel.sum() == 0:
                continue
            data = pd.to_numeric(df.loc[sel, param], errors="coerce").dropna()
            if data.empty:
                continue
            if np.var(data) == 0:
                ax.axhline(1.0, linestyle="--", label=str(label), color=color_map[label])
            else:
                sns.kdeplot(data, fill=True, alpha=0.3, ax=ax, label=str(label), color=color_map[label])

        # Titles with label + unit
        p_label = getattr(registry, "get_param_label", lambda n, **k: n)(param)
        p_unit = getattr(registry, "get_param_unit", lambda n, **k: None)(param)
        ax.set_title(p_label if not p_unit else f"{p_label} [{p_unit}]", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Density")
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for j in range(n_inputs, len(axes)):
        axes[j].set_visible(False)

    # Figure title
    t_label = getattr(registry, "get_param_label", lambda n, **k: n)(target)
    t_unit = target_unit or getattr(registry, "get_param_unit", lambda n, **k: None)(target) or ""
    sup_title = f"KDE of Inputs by {t_label if not t_unit else f'{t_label} [{t_unit}]'} quartile"
    if file_type:
        sup_title += f" for {file_type}"
    fig.suptitle(sup_title, fontsize=14, y=0.97)

    # Legend (use first visible axis that has handles)
    handles, labels_seen = None, None
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels_seen = h, l
            break
    if handles:
        fig.legend(
            handles,
            labels_seen,
            title=f"{t_label if not t_unit else f'{t_label} [{t_unit}]'} quartile",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=2,
            fontsize=12,
            title_fontsize=14,
        )

    # Save plot and CSV with extremes
    out_png = outdir / f"{stem}_{target}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    _save_quartile_extremes_to_csv(df, target, varying_inputs, bins, list(labels), outdir, stem)
