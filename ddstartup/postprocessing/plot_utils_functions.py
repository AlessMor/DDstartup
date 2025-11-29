"""
Shared utilities for plotting helpers.

These functions centralize common tasks used across plot modules:
  - output directory/stem resolution
  - registry retrieval
  - scalar numeric column selection (respecting df.attrs["_inner_dims"])
  - near-constant column filtering
  - quartile binning and consistent quartile colors
  - basic label/unit formatting helpers
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def ensure_registry(registry=None):
    """Return a registry instance, importing lazily when not provided."""
    if registry is not None:
        return registry
    from ddstartup.utils.parameter_registry import get_registry as _get_registry

    return _get_registry()


def resolve_outdir_and_stem(
    output_dir=None,
    outputs_dir=None,
    plot_name_prefix=None,
    plot_name=None,
    default_stem: str = "plot",
    suffix: str | None = None,
) -> tuple[Path, str]:
    """
    Resolve output directory and filename stem in a consistent way.

    - output_dir overrides outputs_dir; both default to "."
    - plot_name_prefix is preferred; plot_name stem is the next fallback; otherwise use default_stem
    - optional suffix is appended when using plot_name_prefix or default_stem
    """
    outdir = Path(output_dir if output_dir is not None else (outputs_dir if outputs_dir is not None else "."))
    outdir.mkdir(parents=True, exist_ok=True)

    if plot_name_prefix:
        stem = f"{plot_name_prefix}{suffix or ''}"
    elif plot_name:
        stem = Path(plot_name).stem
    else:
        stem = f"{default_stem}{suffix or ''}"
    return outdir, stem


def select_scalar_numeric(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    """Keep scalar, numeric columns only (uses df.attrs['_inner_dims'] when present)."""
    inner = (getattr(df, "attrs", {}) or {}).get("_inner_dims", {})
    usable: list[str] = []
    for c in cols or []:
        if c not in df.columns:
            continue
        if int(inner.get(c, 1)) != 1:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        usable.append(c)
    return usable


def drop_near_constant(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    std_tol: float = 1e-10,
    rel_tol: float = 1e-6,
) -> list[str]:
    """Remove near-constant numeric columns (robust to zeros)."""
    keep: list[str] = []
    for c in cols or []:
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        try:
            std = float(s.std())
            mean = abs(float(s.mean()))
        except Exception:
            continue
        if std > std_tol and (mean == 0 or std / max(mean, 1e-30) > rel_tol):
            keep.append(c)
    return keep


def quartile_bins(series: pd.Series, q: int = 4) -> Tuple[pd.Series, list[str]]:
    """Compute quantile bins and readable labels; gracefully handle duplicates/NaNs."""
    y = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    labels = [f"Q{i+1}" for i in range(q)]
    if y.empty:
        return pd.Categorical([np.nan] * len(series), categories=labels), labels
    qs = series.quantile(np.linspace(0, 1, q + 1))
    labels = [f"Q{i+1}: {qs.iloc[i]:.2e}–{qs.iloc[i+1]:.2e}" for i in range(q)]
    bins = pd.qcut(series, q=q, labels=labels, duplicates="drop")
    if getattr(bins, "dtype", None) == "category" and len(bins.cat.categories) != q:
        cats = list(bins.cat.categories)
        return bins, cats
    return bins, labels


def quartile_colors(k: int) -> list[str]:
    """Return a list of colors for k quartiles using the shared discrete colorscale."""
    from ddstartup.postprocessing.postprocess_functions import get_discrete_colorscale

    colorscale = get_discrete_colorscale(k if k > 0 else 1)
    return [colorscale[i * 2][1] for i in range(k)]


def param_label_and_unit(registry, name: str, unit_override: str | None = None) -> tuple[str, str]:
    """Fetch parameter label and unit from registry, applying an optional override."""
    label = getattr(registry, "get_param_label", lambda n, **k: n)(name)
    unit = unit_override if unit_override is not None else getattr(registry, "get_param_unit", lambda n, **k: None)(name) or ""
    return label, unit
