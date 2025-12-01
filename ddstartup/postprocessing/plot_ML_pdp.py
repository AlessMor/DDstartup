# -*- coding: utf-8 -*-
"""
Train an MLP on a pandas DataFrame and generate ML PDP plots for all feature pairs.

Entry point: generate_ml_pairwise_plots(...)
 - Cleans the provided DataFrame (numeric, finite, non-constant columns).
 - Trains an MLP with log-target handling and early stopping.
 - Saves diagnostics/model/scalers (optional) and diagnostic plot if verbose.
 - Generates pairwise PDP contour plots for all feature combinations.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ddstartup.postprocessing.fit_ML_method import (
    clean_dataframe,
    density_2d,
    pairwise_pdp_grid,
    plot_overfitting_diagnostics,
    train_model,
)


def generate_ml_pairwise_plots(
    *,
    df: pd.DataFrame,
    target: str,
    inputs: List[str],
    target_unit: str,
    output_dir: Path,
    file_type: str,
    plot_name_prefix: str,
    registry=None,
    ml_pairwise_settings: Dict[str, object] | None = None,
    show_titles: bool = True,
    **_,
) -> None:
    """
    Clean DataFrame -> train MLP -> optional diagnostics -> pairwise PDP plots.
    """
    cfg = ml_pairwise_settings or {}
    verbose = bool(cfg.get("verbose", False))
    grid_size = int(cfg.get("grid_size", 80))
    bg_samples = int(cfg.get("bg_samples", 256))
    hidden = tuple(cfg.get("hidden", (128, 64, 32)))
    dropout = float(cfg.get("dropout", 0.5))
    lr = float(cfg.get("lr", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-2))
    batch_size = int(cfg.get("batch_size", 16384))
    max_epochs = int(cfg.get("max_epochs", 100))
    patience = int(cfg.get("patience", 10))
    use_augmentation = bool(cfg.get("use_augmentation", True))
    min_rows_val = cfg.get("min_rows", 200)
    min_rows = int(min_rows_val) if min_rows_val not in (None, 0) else 0
    max_train_val = cfg.get("max_train_samples", 200_000)
    # If set to 0/None, do not subsample
    max_train_samples = None if max_train_val in (None, 0) else int(max_train_val)
    save_artifacts = bool(cfg.get("save_artifacts", True))
    plot_diag = verbose and bool(cfg.get("plot_diagnostics", True))
    qrange = tuple(cfg.get("qrange", (0.05, 0.95)))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean DataFrame -> X, y
    try:
        X, y, usable = clean_dataframe(df, target, feature_cols=inputs, min_rows=min_rows, verbose=verbose)
    except Exception as e:
        print(f"   Error during ML PDP cleaning: {e}")
        return

    if max_train_samples is not None and len(X) > max_train_samples:
        idx = np.random.default_rng(0).choice(len(X), size=max_train_samples, replace=False)
        X = X[idx]
        y = y[idx]
        if verbose:
            print(f"   Subsampled training set to {max_train_samples:,} rows for ML PDP")

    if len(usable) < 2:
        print(f"   Warning: need at least 2 usable inputs (found {len(usable)}). Skipping ML PDP.")
        return

    # Train model
    try:
        (
            model,
            x_scaler,
            y_scaler,
            r2,
            mae,
            rmse,
            nrmse,
            mape,
            device,
            diagnostics,
            y_shift,
            y_is_log,
        ) = train_model(
            X,
            y,
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            use_augmentation=use_augmentation,
        )
    except Exception as e:
        print(f"   Error training ML PDP model: {e}")
        return

    if verbose:
        print(
            f"   ML PDP model metrics (test split): R^2={r2:.3f}, RMSE={rmse:.4g}, "
            f"NRMSE={nrmse:.4g}, MAPE={mape:.3f}"
        )

    prefix = f"{plot_name_prefix}_{target}"
    if save_artifacts:
        model_path = out_dir / f"{prefix}.pt"
        scalers_path = out_dir / f"{prefix}_scalers.pkl"
        diagnostics_path = out_dir / f"{prefix}_diagnostics.pkl"
        torch.save(model.state_dict(), model_path)
        with open(scalers_path, "wb") as f:
            import pickle

            pickle.dump({"x": x_scaler, "y": y_scaler, "y_shift": y_shift, "y_is_log": y_is_log}, f)
        with open(diagnostics_path, "wb") as f:
            import pickle

            pickle.dump(diagnostics, f)
        if verbose:
            print(f"   Saved ML PDP artifacts to {out_dir.name}: {model_path.name}, {scalers_path.name}, {diagnostics_path.name}")

    if plot_diag:
        diag_path = out_dir / f"{prefix}_overfitting.png"
        plot_overfitting_diagnostics(diagnostics, target_name=target, save_path=diag_path)

    # Generate pairwise PDPs for all feature combinations
    pairs = list(itertools.combinations(range(len(usable)), 2))
    print(f"   Generating {len(pairs)} pairwise PDP plots...")

    target_label = registry.get_param_label(target) if registry is not None else target
    for pidx, (i, j) in enumerate(pairs, 1):
        pi, pj = usable[i], usable[j]
        xi = registry.get_param_label(pi) if registry is not None else pi
        xj = registry.get_param_label(pj) if registry is not None else pj
        try:
            gi, gj, Z = pairwise_pdp_grid(
                model,
                x_scaler,
                y_scaler,
                X,
                i,
                j,
                grid_size=grid_size,
                bg_samples=bg_samples,
                qrange=qrange,
                device=device,
                y_shift=y_shift,
                y_is_log=y_is_log,
            )

            II, JJ = np.meshgrid(gi, gj, indexing="ij")
            fig, ax = plt.subplots(figsize=(7.5, 6.0))
            cs = ax.contourf(II, JJ, Z, levels=40, alpha=0.9, cmap="viridis")
            cbar = fig.colorbar(cs, ax=ax)
            if target_unit:
                cbar.set_label(f"{target_label} [{target_unit}]")
            else:
                cbar.set_label(target_label)

            H, extent = density_2d(X, i, j, gi, gj, bins=100)
            ax.imshow(H, extent=extent, origin="lower", alpha=0.25, aspect="auto", cmap="gray")

            ax.set_xlabel(xi, fontsize=11)
            ax.set_ylabel(xj, fontsize=11)
            if show_titles:
                ax.set_title(f"ML PDP: {target_label} ({file_type})", fontsize=12, fontweight="bold")

            plt.tight_layout()
            fname = f"{prefix}_{pi}x{pj}.png"
            plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

            if verbose and pidx == 1:
                print(f"      Saved: {fname}")
        except Exception as e:
            print(f"      Error plotting {pi} vs {pj}: {e}")
            continue

    if len(pairs) > 1:
        print(f"      ... and {len(pairs) - 1} more pairwise plots")
    print("   ML PDP plots complete")
