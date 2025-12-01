# -*- coding: utf-8 -*-
# pdp_analysis.py
"""
Combined script for ML model diagnostics and Partial Dependence Plot (PDP) analysis.
Loads a trained PyTorch model and generates:
1. Overfitting diagnostic plots (from diagnostics.pkl)
2. 2D PDP plots for all feature pairs
3. 1D PDP plots for all features
4. ICE (Individual Conditional Expectation) plots
"""

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import h5py
import hdf5plugin
from pathlib import Path
from itertools import combinations
from sklearn.preprocessing import StandardScaler


# --------------------------
# Model Architecture
# --------------------------
class MLPRegressor(nn.Module):
    def __init__(self, n_in=13, hidden=(128, 64, 32), dropout=0.5, act=nn.SiLU):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), act(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------
# Prediction Helper
# --------------------------
def predict_numpy(
    model, x_scaler, y_scaler, X_input, device="cpu", y_shift=0, y_is_log=False
):
    """Predict using the trained model with inverse transform handling."""
    Xs = x_scaler.transform(np.asarray(X_input, dtype=np.float32))
    xb = torch.from_numpy(Xs.astype(np.float32)).to(device)
    with torch.no_grad():
        yhat = model(xb).cpu().numpy()

    # Inverse transform
    y_pred = y_scaler.inverse_transform(yhat).ravel()

    # If log-transformed, apply exp and subtract shift
    if y_is_log:
        y_pred = np.exp(y_pred)
        if y_shift != 0:
            y_pred = y_pred - y_shift

    return y_pred


# --------------------------
# PDP Functions
# --------------------------
def pairwise_pdp_grid(
    model,
    x_scaler,
    y_scaler,
    X_raw,
    i,
    j,
    grid_size=80,
    bg_samples=256,
    qrange=(0.05, 0.95),
    device="cpu",
    y_shift=0,
    y_is_log=False,
):
    """
    Compute 2D PDP surface Z for features i, j.
    Returns grid_i, grid_j, Z (PDP surface).
    """
    X_raw = np.asarray(X_raw, dtype=np.float32)
    n, d = X_raw.shape

    # Define grid ranges
    def _safe_quantile_range(col, qrange):
        lo, hi = np.quantile(col, qrange)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            m = np.nanmedian(col)
            span = max(1e-6, 0.01 * max(1.0, abs(m)))
            lo, hi = m - span, m + span
        return float(lo), float(hi)

    gi_lo, gi_hi = _safe_quantile_range(X_raw[:, i], qrange)
    gj_lo, gj_hi = _safe_quantile_range(X_raw[:, j], qrange)
    grid_i = np.linspace(gi_lo, gi_hi, grid_size, dtype=np.float32)
    grid_j = np.linspace(gj_lo, gj_hi, grid_size, dtype=np.float32)

    # Sample background data
    idx = np.random.choice(n, size=min(bg_samples, n), replace=False)
    BG = X_raw[idx].copy()
    B = BG.shape[0]

    # Create grid
    G = grid_size * grid_size
    BG_rep = np.tile(BG, (G, 1))
    ii, jj = np.meshgrid(grid_i, grid_j, indexing="ij")
    g_i = np.repeat(ii.ravel(), B)
    g_j = np.repeat(jj.ravel(), B)
    BG_rep[:, i] = g_i
    BG_rep[:, j] = g_j

    # Predict and average
    y_pred = predict_numpy(
        model,
        x_scaler,
        y_scaler,
        BG_rep,
        device=device,
        y_shift=y_shift,
        y_is_log=y_is_log,
    )
    Z = y_pred.reshape(grid_size, grid_size, B).mean(axis=2)

    return grid_i, grid_j, Z


def density_2d(X_raw, i, j, grid_i, grid_j, bins=100):
    """Compute 2D histogram for data density."""
    xi = np.clip(X_raw[:, i], grid_i.min(), grid_i.max())
    xj = np.clip(X_raw[:, j], grid_j.min(), grid_j.max())
    H, _, _ = np.histogram2d(
        xi,
        xj,
        bins=bins,
        range=[[grid_i.min(), grid_i.max()], [grid_j.min(), grid_j.max()]],
    )
    H = H.T / (H.max() + 1e-9)
    extent = (grid_i.min(), grid_i.max(), grid_j.min(), grid_j.max())
    return H, extent


def compute_1d_pdp(
    model,
    x_scaler,
    y_scaler,
    X_raw,
    feature_idx,
    grid_size=100,
    bg_samples=500,
    qrange=(0.05, 0.95),
    y_shift=0,
    y_is_log=False,
    device="cpu",
):
    """Compute 1D PDP for a single feature."""
    X_raw = np.asarray(X_raw, dtype=np.float32)
    n = X_raw.shape[0]

    # Define grid
    lo, hi = np.quantile(X_raw[:, feature_idx], qrange)
    grid = np.linspace(lo, hi, grid_size, dtype=np.float32)

    # Sample background
    idx = np.random.choice(n, size=min(bg_samples, n), replace=False)
    BG = X_raw[idx].copy()

    # Create inputs
    BG_rep = np.tile(BG, (grid_size, 1))
    grid_rep = np.repeat(grid, len(BG))
    BG_rep[:, feature_idx] = grid_rep

    # Predict
    y_pred = predict_numpy(
        model,
        x_scaler,
        y_scaler,
        BG_rep,
        device=device,
        y_shift=y_shift,
        y_is_log=y_is_log,
    )
    pdp = y_pred.reshape(grid_size, len(BG)).mean(axis=1)
    pdp_std = y_pred.reshape(grid_size, len(BG)).std(axis=1)

    return grid, pdp, pdp_std


def compute_ice_curves(
    model,
    x_scaler,
    y_scaler,
    X_raw,
    feature_idx,
    grid_size=50,
    n_samples=100,
    qrange=(0.05, 0.95),
    y_shift=0,
    y_is_log=False,
    device="cpu",
):
    """Compute ICE curves for selected samples."""
    X_raw = np.asarray(X_raw, dtype=np.float32)

    # Define grid
    lo, hi = np.quantile(X_raw[:, feature_idx], qrange)
    grid = np.linspace(lo, hi, grid_size, dtype=np.float32)

    # Sample instances
    idx = np.random.choice(len(X_raw), size=min(n_samples, len(X_raw)), replace=False)
    X_sample = X_raw[idx].copy()

    # Compute predictions for each sample at each grid point
    ice_curves = np.zeros((n_samples, grid_size))

    for i, grid_val in enumerate(grid):
        X_temp = X_sample.copy()
        X_temp[:, feature_idx] = grid_val
        ice_curves[:, i] = predict_numpy(
            model,
            x_scaler,
            y_scaler,
            X_temp,
            device=device,
            y_shift=y_shift,
            y_is_log=y_is_log,
        )

    return grid, ice_curves


# --------------------------
# Overfitting Diagnostics
# --------------------------
def plot_overfitting_diagnostics(
    diagnostics, target_name="Target", save_path="overfitting_diagnostics.png"
):
    """Create comprehensive overfitting diagnostic plots."""

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Training history (loss curves)
    ax1 = plt.subplot(3, 3, 1)
    epochs = diagnostics["history"]["epoch"]
    train_loss = diagnostics["history"]["train_loss"]
    val_loss = diagnostics["history"]["val_loss"]
    ax1.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    ax1.axvline(
        diagnostics["best_epoch"],
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Best Epoch",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE, scaled)")
    ax1.set_title("Learning Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Calculate overfitting indicators
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    gap = final_val_loss - final_train_loss
    gap_pct = (gap / final_train_loss) * 100 if final_train_loss > 0 else 0

    # 2. Loss gap over time
    ax2 = plt.subplot(3, 3, 2)
    loss_gap = np.array(val_loss) - np.array(train_loss)
    ax2.plot(epochs, loss_gap, linewidth=2, color="darkred")
    ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Loss - Train Loss")
    ax2.set_title(f"Overfitting Gap (final: {gap:.6f}, {gap_pct:.1f}%)")
    ax2.grid(True, alpha=0.3)

    # 3. Learning rate schedule
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(epochs, diagnostics["history"]["lr"], linewidth=2, color="green")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # 4-6. Predictions vs Actuals for Train/Val/Test
    for idx, split in enumerate(["train", "val", "test"]):
        ax = plt.subplot(3, 3, 4 + idx)
        y_true = diagnostics[split]["y_true"]
        y_pred = diagnostics[split]["y_pred"]
        r2 = diagnostics[split]["r2"]
        mae = diagnostics[split]["mae"]

        # Scatter plot with density
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, edgecolors="none")

        # Perfect prediction line
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        ]
        ax.plot(lims, lims, "r--", alpha=0.75, linewidth=2, label="Perfect Pred")

        ax.set_xlabel(f"True {target_name}")
        ax.set_ylabel(f"Predicted {target_name}")
        ax.set_title(f"{split.upper()}: R²={r2:.4f}, MAE={mae:.4g}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    # 7-9. Residual plots for Train/Val/Test
    for idx, split in enumerate(["train", "val", "test"]):
        ax = plt.subplot(3, 3, 7 + idx)
        y_true = diagnostics[split]["y_true"]
        y_pred = diagnostics[split]["y_pred"]
        residuals = y_true - y_pred
        rmse = diagnostics[split]["rmse"]

        ax.scatter(y_pred, residuals, alpha=0.3, s=10, edgecolors="none")
        ax.axhline(0, color="red", linestyle="--", linewidth=2)
        ax.axhline(rmse, color="orange", linestyle=":", linewidth=1.5, label=f"±RMSE")
        ax.axhline(-rmse, color="orange", linestyle=":", linewidth=1.5)

        ax.set_xlabel(f"Predicted {target_name}")
        ax.set_ylabel("Residuals")
        ax.set_title(f"{split.upper()} Residuals (RMSE={rmse:.4g})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()

    # Print summary
    print("\n" + "=" * 70)
    print("OVERFITTING ANALYSIS SUMMARY")
    print("=" * 70)
    print(
        f"\nTraining stopped at epoch {len(epochs)} (best: {diagnostics['best_epoch']})"
    )
    print(f"Final train loss: {final_train_loss:.6f}")
    print(f"Final val loss:   {final_val_loss:.6f}")
    print(f"Loss gap:         {gap:.6f} ({gap_pct:+.1f}%)")
    print("\nMetrics Comparison:")
    print(f"{'Metric':<10} {'Train':>12} {'Val':>12} {'Test':>12} {'Val/Train':>12}")
    print("-" * 70)
    for metric in ["r2", "mae", "rmse", "nrmse", "mape"]:
        tr_val = diagnostics["train"][metric]
        va_val = diagnostics["val"][metric]
        te_val = diagnostics["test"][metric]
        ratio = va_val / tr_val if tr_val != 0 else float("nan")
        print(
            f"{metric.upper():<10} {tr_val:>12.6f} {va_val:>12.6f} {te_val:>12.6f} {ratio:>12.3f}"
        )

    # Overfitting assessment
    print("\n" + "=" * 70)
    print("OVERFITTING ASSESSMENT:")
    print("-" * 70)

    r2_gap = diagnostics["train"]["r2"] - diagnostics["val"]["r2"]
    rmse_ratio = diagnostics["val"]["rmse"] / diagnostics["train"]["rmse"]

    warnings = []
    if gap_pct > 20:
        warnings.append(f"⚠️  Large loss gap: {gap_pct:.1f}% (train vs val)")
    if r2_gap > 0.05:
        warnings.append(f"⚠️  R² gap: {r2_gap:.4f} (train better than val)")
    if rmse_ratio > 1.2:
        warnings.append(f"⚠️  Val RMSE is {rmse_ratio:.2f}x train RMSE")
    if diagnostics["val"]["r2"] < diagnostics["test"]["r2"] - 0.05:
        warnings.append("⚠️  Test performs better than validation (unusual)")
    if diagnostics["train"]["r2"] > 0.999:
        warnings.append("⚠️  Near-perfect train R² (0.999+) suggests overfitting")

    if warnings:
        print("Potential overfitting detected:")
        for w in warnings:
            print(f"  {w}")
        print("\nRecommendations:")
        print("  • Increase dropout rate")
        print("  • Increase weight_decay (L2 regularization)")
        print("  • Reduce model complexity (fewer/smaller hidden layers)")
        print("  • Get more training data")
        print("  • Add data augmentation")
    else:
        print("✓ No significant overfitting detected")
        print("  Train/val/test metrics are reasonably consistent")

    print("=" * 70 + "\n")


# --------------------------
# Data Loading
# --------------------------
def load_data_from_h5(h5_path, features, target):
    """Load features and target from HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        X_list = []
        for feat in features:
            arr = np.asarray(f[feat][:], dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=tuple(range(1, arr.ndim)))
            X_list.append(arr.reshape(-1))
        X = np.column_stack(X_list).astype(np.float32)
        y = np.asarray(f[target][:], dtype=np.float32).reshape(-1)

    print(f"Data loaded (raw): X.shape={X.shape}, y.shape={y.shape}")
    print(f"Target range (raw): [{np.nanmin(y):.2e}, {np.nanmax(y):.2e}]")

    # Check for NaN/Inf values
    n_nan_X = np.isnan(X).any(axis=1).sum()
    n_inf_X = np.isinf(X).any(axis=1).sum()
    n_nan_y = np.isnan(y).sum()
    n_inf_y = np.isinf(y).sum()

    print(f"\nData quality check:")
    print(f"  Rows with NaN in X: {n_nan_X:,} ({n_nan_X/len(X)*100:.2f}%)")
    print(f"  Rows with Inf in X: {n_inf_X:,} ({n_inf_X/len(X)*100:.2f}%)")
    print(f"  NaN values in y: {n_nan_y:,} ({n_nan_y/len(y)*100:.2f}%)")
    print(f"  Inf values in y: {n_inf_y:,} ({n_inf_y/len(y)*100:.2f}%)")

    # Filter out invalid data
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    n_removed = len(X) - len(X_clean)
    print(f"\n✓ Removed {n_removed:,} invalid samples ({n_removed/len(X)*100:.2f}%)")
    print(f"✓ Clean data: X.shape={X_clean.shape}, y.shape={y_clean.shape}")
    print(f"✓ Target range (clean): [{y_clean.min():.2e}, {y_clean.max():.2e}]")

    return X_clean, y_clean


# --------------------------
# Main Analysis Function
# --------------------------
def run_pdp_analysis(
    h5_path,
    features,
    target,
    model_path="mlp_13d.pt",
    scalers_path="scalers.pkl",
    diagnostics_path="diagnostics.pkl",
    output_dir="pdp_analysis_output",
    generate_2d_pdp=True,
    generate_1d_pdp=True,
    generate_ice=True,
    pdp_grid_size=80,
    pdp_bg_samples=256,
    ice_grid_size=50,
    ice_n_samples=200,
):
    """
    Run complete PDP analysis including diagnostics and plots.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 file with data
    features : list
        List of feature names
    target : str
        Target variable name
    model_path : str
        Path to saved model weights
    scalers_path : str
        Path to saved scalers
    diagnostics_path : str
        Path to saved diagnostics
    output_dir : str
        Directory to save output plots
    generate_2d_pdp : bool
        Whether to generate 2D PDP plots
    generate_1d_pdp : bool
        Whether to generate 1D PDP plots
    generate_ice : bool
        Whether to generate ICE plots
    pdp_grid_size : int
        Grid size for PDP plots
    pdp_bg_samples : int
        Background samples for PDP
    ice_grid_size : int
        Grid size for ICE plots
    ice_n_samples : int
        Number of samples for ICE plots
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path.absolute()}/")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load model
    print("\n" + "=" * 70)
    print("LOADING MODEL AND SCALERS")
    print("=" * 70)

    model = MLPRegressor(n_in=len(features), hidden=(128, 64, 32), dropout=0.5).to(
        device
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Model loaded from {model_path}")

    # Load scalers
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
        x_scaler = scalers["x"]
        y_scaler = scalers["y"]
        y_shift = scalers.get("y_shift", 0)
        y_is_log = scalers.get("y_is_log", False)
    print(f"✓ Scalers loaded from {scalers_path}")
    print(f"  Log-transform: {y_is_log}, Shift: {y_shift:.4e}")

    # Load diagnostics
    with open(diagnostics_path, "rb") as f:
        diagnostics = pickle.load(f)
    print(f"✓ Diagnostics loaded from {diagnostics_path}")

    # Plot overfitting diagnostics
    print("\n" + "=" * 70)
    print("GENERATING OVERFITTING DIAGNOSTICS")
    print("=" * 70)
    plot_overfitting_diagnostics(
        diagnostics,
        target_name=target,
        save_path=str(output_path / "overfitting_diagnostics.png"),
    )

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    X, y = load_data_from_h5(h5_path, features, target)

    # Generate 2D PDP plots
    if generate_2d_pdp:
        print("\n" + "=" * 70)
        print("GENERATING 2D PDP PLOTS")
        print("=" * 70)

        feature_pairs = list(combinations(range(len(features)), 2))
        print(f"Total number of feature pairs: {len(feature_pairs)}")
        print(f"Generating {len(feature_pairs)} individual 2D PDP plots...")

        start_time = time.time()

        for idx, (i, j) in enumerate(feature_pairs):
            print(
                f"  [{idx+1}/{len(feature_pairs)}] {features[i]} vs {features[j]}... ",
                end="",
                flush=True,
            )

            # Compute PDP
            grid_i, grid_j, Z = pairwise_pdp_grid(
                model,
                x_scaler,
                y_scaler,
                X,
                i,
                j,
                grid_size=pdp_grid_size,
                bg_samples=pdp_bg_samples,
                qrange=(0.05, 0.95),
                device=device,
                y_shift=y_shift,
                y_is_log=y_is_log,
            )

            # Create plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ci, cj = np.meshgrid(grid_i, grid_j, indexing="ij")

            # Plot PDP contours
            cs = ax.contourf(ci, cj, Z, levels=40, cmap="viridis", alpha=0.85)

            # Add contour lines
            ax.contour(ci, cj, Z, levels=10, colors="white", alpha=0.3, linewidths=0.5)

            # Overlay data density
            H, extent = density_2d(X, i, j, grid_i, grid_j, bins=80)
            ax.imshow(
                H,
                extent=extent,
                origin="lower",
                alpha=0.2,
                cmap="gray_r",
                aspect="auto",
            )

            ax.set_xlabel(features[i], fontsize=13, fontweight="bold")
            ax.set_ylabel(features[j], fontsize=13, fontweight="bold")
            ax.set_title(
                f"2D PDP: {features[i]} vs {features[j]}",
                fontsize=14,
                fontweight="bold",
                pad=15,
            )
            ax.tick_params(labelsize=11)

            # Add colorbar
            cbar = plt.colorbar(cs, ax=ax, label=target)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label(target, fontsize=12, fontweight="bold")

            # Save plot
            filename = output_path / f"pdp_2d_{features[i]}_vs_{features[j]}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)

            print(f"✓")

        elapsed_time = time.time() - start_time
        print(
            f"\n✓ Generated all {len(feature_pairs)} 2D PDP plots in {elapsed_time:.1f}s"
        )
        print(f"  Average time per plot: {elapsed_time/len(feature_pairs):.2f}s")

    # Generate 1D PDP plots
    if generate_1d_pdp:
        print("\n" + "=" * 70)
        print("GENERATING 1D PDP PLOTS")
        print("=" * 70)

        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.ravel()

        for feat_idx in range(len(features)):
            print(f"  Computing 1D PDP for {features[feat_idx]}...")
            grid, pdp, pdp_std = compute_1d_pdp(
                model,
                x_scaler,
                y_scaler,
                X,
                feat_idx,
                grid_size=100,
                bg_samples=500,
                y_shift=y_shift,
                y_is_log=y_is_log,
                device=device,
            )

            ax = axes[feat_idx]
            ax.plot(grid, pdp, linewidth=2.5, color="darkblue", label="PDP")
            ax.fill_between(
                grid,
                pdp - pdp_std,
                pdp + pdp_std,
                alpha=0.3,
                color="lightblue",
                label="±1 std",
            )

            # Add rug plot
            sample_idx = np.random.choice(len(X), size=min(1000, len(X)), replace=False)
            ax.scatter(
                X[sample_idx, feat_idx],
                [pdp.min()] * len(sample_idx),
                alpha=0.05,
                color="black",
                s=1,
            )

            ax.set_xlabel(features[feat_idx], fontsize=11, fontweight="bold")
            ax.set_ylabel(target, fontsize=10)
            ax.set_title(
                f"1D PDP: {features[feat_idx]}", fontsize=12, fontweight="bold"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        # Remove extra subplot if needed
        if len(features) < len(axes):
            fig.delaxes(axes[-1])

        plt.tight_layout()
        save_path = output_path / "pdp_1d_all_features.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Saved: {save_path}")
        plt.close()

    # Generate ICE plots
    if generate_ice:
        print("\n" + "=" * 70)
        print("GENERATING ICE PLOTS")
        print("=" * 70)

        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.ravel()

        for feat_idx in range(len(features)):
            print(f"  Computing ICE for {features[feat_idx]}...")
            grid, ice_curves = compute_ice_curves(
                model,
                x_scaler,
                y_scaler,
                X,
                feat_idx,
                grid_size=ice_grid_size,
                n_samples=ice_n_samples,
                y_shift=y_shift,
                y_is_log=y_is_log,
                device=device,
            )

            # Compute PDP (average of ICE)
            pdp = ice_curves.mean(axis=0)
            pdp_std = ice_curves.std(axis=0)

            ax = axes[feat_idx]

            # Plot individual ICE curves (transparent)
            for curve in ice_curves:
                ax.plot(grid, curve, alpha=0.05, color="gray", linewidth=0.5)

            # Plot PDP (thick line)
            ax.plot(
                grid, pdp, linewidth=3, color="darkblue", label="PDP (mean)", zorder=100
            )

            # Add std bands
            ax.fill_between(
                grid,
                pdp - pdp_std,
                pdp + pdp_std,
                alpha=0.25,
                color="blue",
                label="±1 std",
                zorder=99,
            )

            ax.set_xlabel(features[feat_idx], fontsize=11, fontweight="bold")
            ax.set_ylabel(target, fontsize=10)
            ax.set_title(f"ICE: {features[feat_idx]}", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        # Remove extra subplot if needed
        if len(features) < len(axes):
            fig.delaxes(axes[-1])

        plt.tight_layout()
        save_path = output_path / "ice_plots_all.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Saved: {save_path}")
        plt.close()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"All outputs saved to: {output_path.absolute()}/")


# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    # Configuration
    H5_PATH = "outputs/20251113_002935_parametric_T_seeded/ddstartup_20251113_002935_parametric_T_seeded.h5"
    FEATURES = [
        "V_plasma",  # 0
        "T_i",  # 1
        "n_tot",  # 2
        "tau_p_T",  # 3
        "P_aux",  # 4
        "P_aux_DT_eq",  # 5
        "TBR_DT",  # 6
        "TBR_DDn",  # 7
        "tau_ifc",  # 8
        "tau_ofc",  # 9
        "eta_th",  # 10
        "capacity_factor",  # 11
        "price_of_electricity",  # 12
    ]
    TARGET = "unrealized_profits"

    # Run analysis
    run_pdp_analysis(
        h5_path=H5_PATH,
        features=FEATURES,
        target=TARGET,
        model_path="mlp_13d.pt",
        scalers_path="scalers.pkl",
        diagnostics_path="diagnostics.pkl",
        output_dir="pdp_analysis_output",
        generate_2d_pdp=True,
        generate_1d_pdp=True,
        generate_ice=True,
        pdp_grid_size=80,
        pdp_bg_samples=256,
        ice_grid_size=50,
        ice_n_samples=200,
    )
