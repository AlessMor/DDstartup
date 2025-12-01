# -*- coding: utf-8 -*-
# mlp_pdp_13d.py
import os, math, pickle, numpy as np
import torch
import torch.nn as nn
import h5py
import hdf5plugin  # Required for reading compressed HDF5 files
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# --------------------------
# Data utilities
# --------------------------
def load_from_h5(h5_path, x_key, y_key):
    """Load X and y from an HDF5 file where each dataset is 1D/2D."""
    print(f"Loading data from: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        X_list = []
        for key in x_key:
            arr = f[key][:]
            X_list.append(arr.reshape(-1, 1))
        X = np.hstack(X_list)
        y = f[y_key][:]
        y = y.reshape(-1)
    print(f"Loaded shapes: X={X.shape}, y={y.shape}")
    return X, y


# --------------------------
# Dataset
# --------------------------
class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --------------------------
# Model
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
# Training
# --------------------------
def train_model(
    X,
    y,
    hidden=(128, 64, 32),
    dropout=0.5,
    lr=3e-4,
    weight_decay=1e-2,
    batch_size=16384,
    max_epochs=100,
    patience=10,
    num_workers=4,
    pin_memory=True,
    use_augmentation=True,
):
    """
    Train an MLP regressor with standardization, log-target, early stopping,
    and optional data augmentation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)

    # ---- Handle non-positive targets for log-transform ----
    y_shift = 0.0
    y_min = float(np.min(y))
    if y_min <= 0:
        y_shift = abs(y_min) + 1.0
        print(f"Applying positive shift to target: {y_shift:.4e}")
        y_shifted = y + y_shift
    else:
        y_shifted = y.copy()

    y_log = np.log(y_shifted)
    print(
        f"Target log stats: mean={np.mean(y_log):.4f}, std={np.std(y_log):.4f}, "
        f"min={np.min(y_log):.4f}, max={np.max(y_log):.4f}"
    )

    # ---- Split first (on raw data) to avoid leakage ----
    X_tr_raw, X_tmp_raw, y_tr_raw, y_tmp_raw = train_test_split(
        X, y_log, test_size=0.30, random_state=42
    )
    X_va_raw, X_te_raw, y_va_raw, y_te_raw = train_test_split(
        X_tmp_raw, y_tmp_raw, test_size=0.50, random_state=42
    )

    # ---- Fit scalers on training only, then transform all splits ----
    x_scaler = StandardScaler().fit(X_tr_raw)
    y_scaler = StandardScaler().fit(y_tr_raw.reshape(-1, 1))

    X_tr = x_scaler.transform(X_tr_raw)
    X_va = x_scaler.transform(X_va_raw)
    X_te = x_scaler.transform(X_te_raw)

    y_tr = y_scaler.transform(y_tr_raw.reshape(-1, 1)).ravel()
    y_va = y_scaler.transform(y_va_raw.reshape(-1, 1)).ravel()
    y_te = y_scaler.transform(y_te_raw.reshape(-1, 1)).ravel()

    # ---- Data augmentation: add small Gaussian noise to training data (optional) ----
    if use_augmentation:
        print(f"\nApplying data augmentation to training set...")
        X_tr_aug = X_tr + np.random.normal(0, 0.02 * np.std(X_tr, axis=0), X_tr.shape)
        y_tr_aug = y_tr.copy()

        # Combine original + augmented
        X_tr = np.vstack([X_tr, X_tr_aug]).astype(np.float32)
        y_tr = np.concatenate([y_tr, y_tr_aug]).astype(np.float32)
        print(f"✓ Augmented training set: {X_tr.shape[0]:,} samples (2x original)")
    else:
        X_tr = X_tr.astype(np.float32)
        X_va = X_va.astype(np.float32)
        X_te = X_te.astype(np.float32)
        y_tr = y_tr.astype(np.float32)
        y_va = y_va.astype(np.float32)
        y_te = y_te.astype(np.float32)

    # ---- Datasets / loaders ----
    ds_tr = ArrayDataset(X_tr, y_tr)
    ds_va = ArrayDataset(X_va, y_va)
    ds_te = ArrayDataset(X_te, y_te)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and (device == "cuda"),
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and (device == "cuda"),
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and (device == "cuda"),
    )

    # ---- Model / optimizer / scheduler ----
    model = MLPRegressor(
        n_in=X.shape[1],
        hidden=hidden,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.3, patience=3, verbose=True
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    use_amp = device == "cuda"

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience_left = patience

    # ---- Training loop ----
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / n

        # Validate
        model.eval()
        running = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device, non_blocking=True), yb.to(
                    device, non_blocking=True
                )
                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred = model(xb)
                        loss = loss_fn(pred, yb)
                else:
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                running += loss.item() * xb.size(0)
                n += xb.size(0)
        val_loss = running / n
        scheduler.step(val_loss)

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(opt.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | lr={opt.param_groups[0]['lr']:.2e}"
        )

        # Early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("Warning: no best_state saved, using final model parameters.")

    # ---- Evaluate on train/val/test in original target space ----
    def evaluate(dl, split_name):
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                preds.append(pred.cpu().numpy())
                trues.append(yb.cpu().numpy())
        preds = np.concatenate(preds, axis=0).reshape(-1, 1)
        trues = np.concatenate(trues, axis=0).reshape(-1, 1)

        # Inverse-transform
        y_true_log = y_scaler.inverse_transform(trues).ravel()
        y_pred_log = y_scaler.inverse_transform(preds).ravel()

        y_true = np.exp(y_true_log)
        y_pred = np.exp(y_pred_log)

        # Undo shift
        if y_shift != 0:
            y_true -= y_shift
            y_pred -= y_shift

        mse = np.mean((y_true - y_pred) ** 2)
        rmse = math.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        var = np.var(y_true)
        r2 = 1 - mse / var if var > 0 else float("nan")
        nrmse = rmse / (np.max(y_true) - np.min(y_true) + 1e-12)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12))) * 100.0

        print(
            f"{split_name:>5} | R^2={r2:.4f} | RMSE={rmse:.4f} | "
            f"MAE={mae:.4f} | NRMSE={nrmse:.4f} | MAPE={mape:.2f}%"
        )
        return {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "nrmse": nrmse,
            "mape": mape,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    print("\nFinal evaluation on each split (original target space):")
    train_metrics = evaluate(dl_tr, "Train")
    val_metrics = evaluate(dl_va, "Val")
    test_metrics = evaluate(dl_te, "Test")

    diagnostics = {
        "history": history,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }

    return (
        model,
        x_scaler,
        y_scaler,
        test_metrics["r2"],
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["nrmse"],
        test_metrics["mape"],
        device,
        diagnostics,
        y_shift,
    )


# --------------------------
# Inference helpers
# --------------------------
def predict_numpy(
    model, x_scaler, y_scaler, X, device="cpu", y_shift=0.0, y_is_log=True
):
    """
    Convenience wrapper: scale X, run model, inverse-transform y.
    """
    X = np.asarray(X, dtype=np.float32)
    Xs = x_scaler.transform(X)
    xb = torch.from_numpy(Xs).to(device)
    with torch.no_grad():
        ys = model(xb).cpu().numpy()
    y_log = y_scaler.inverse_transform(ys).ravel()
    if y_is_log:
        y = np.exp(y_log)
        if y_shift != 0:
            y = y - y_shift
    else:
        y = y_log
    return y


# --------------------------
# PDP utilities
# --------------------------
def pairwise_pdp_grid(
    model,
    x_scaler,
    y_scaler,
    X_raw,
    i,
    j,
    grid_size=60,
    bg_samples=256,
    qrange=(0.05, 0.95),
    device="cpu",
    y_shift=0.0,
    y_is_log=True,
    mc_samples=0,
):
    """
    Compute a 2D partial dependence surface for features i and j.

    model, x_scaler, y_scaler: trained model + scalers
    X_raw: original (unscaled) data, shape (N, D)
    i, j: feature indices
    """
    X_raw = np.asarray(X_raw, dtype=np.float32)
    N, D = X_raw.shape

    # Limits based on quantiles to avoid extreme tails
    xi = X_raw[:, i]
    xj = X_raw[:, j]

    lo_i, hi_i = np.quantile(xi, qrange)
    lo_j, hi_j = np.quantile(xj, qrange)

    grid_i = np.linspace(lo_i, hi_i, grid_size)
    grid_j = np.linspace(lo_j, hi_j, grid_size)

    # Sample background points
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(bg_samples, N), replace=False)
    X_bg = X_raw[idx]

    model.eval()
    model.to(device)

    Z = np.zeros((grid_size, grid_size), dtype=np.float32)
    Z_std = np.zeros_like(Z)

    with torch.no_grad():
        for a, xa in enumerate(grid_i):
            for b, xb in enumerate(grid_j):
                X_mod = X_bg.copy()
                X_mod[:, i] = xa
                X_mod[:, j] = xb
                preds = []

                if mc_samples > 0:
                    # Monte Carlo dropout / noise
                    for _ in range(mc_samples):
                        yhat = predict_numpy(
                            model,
                            x_scaler,
                            y_scaler,
                            X_mod,
                            device=device,
                            y_shift=y_shift,
                            y_is_log=y_is_log,
                        )
                        preds.append(yhat.mean())
                    preds = np.array(preds)
                else:
                    yhat = predict_numpy(
                        model,
                        x_scaler,
                        y_scaler,
                        X_mod,
                        device=device,
                        y_shift=y_shift,
                        y_is_log=y_is_log,
                    )
                    preds = yhat

                Z[a, b] = preds.mean()
                Z_std[a, b] = preds.std() if preds.size > 1 else 0.0

    return grid_i, grid_j, Z, Z_std


def density_2d(X_raw, i, j, grid_i, grid_j, bins=100):
    """
    Compute a 2D normalized histogram for features i and j to show data density.
    """
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


# --------------------------
# Overfitting diagnostics plot
# --------------------------
def plot_overfitting_diagnostics(diagnostics, target_name="target", save_prefix="mlp"):
    """
    Plot learning curves, overfitting gap, LR schedule, and residuals for
    train/val/test splits. Uses the diagnostics dict returned by train_model.
    """
    hist = diagnostics["history"]
    train = diagnostics["train"]
    val = diagnostics["val"]
    test = diagnostics["test"]

    epochs = hist["epoch"]
    train_loss = np.array(hist["train_loss"])
    val_loss = np.array(hist["val_loss"])
    lr = np.array(hist["lr"])

    # Find best epoch (min val loss)
    best_idx = np.argmin(val_loss)
    best_epoch = epochs[best_idx]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3)

    # Learning curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, label="Train Loss")
    ax1.plot(epochs, val_loss, label="Val Loss")
    ax1.set_yscale("log")
    ax1.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE, scaled)")
    ax1.set_title("Learning Curves")
    ax1.legend()

    # Overfitting gap
    ax2 = fig.add_subplot(gs[0, 1])
    gap = val_loss - train_loss
    ax2.plot(epochs, gap, color="maroon")
    ax2.axhline(0, color="k", linestyle="--")
    ax2.axvline(best_epoch, color="r", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Loss - Train Loss")
    ax2.set_title(
        f"Overfitting Gap (final: {gap[-1]:.6f}, {gap[-1]/train_loss[-1]*100:.1f}%)"
    )

    # LR schedule
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, lr, color="green")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")

    # Scatter: true vs pred for each split
    def scatter_true_pred(ax, split_data, name):
        y_true = split_data["y_true"]
        y_pred = split_data["y_pred"]
        r2 = split_data["r2"]
        mae = split_data["mae"]
        ax.scatter(
            y_true,
            y_pred,
            s=3,
            alpha=0.3,
            edgecolors="none",
        )
        lo = np.min([y_true.min(), y_pred.min()])
        hi = np.max([y_true.max(), y_pred.max()])
        ax.plot([lo, hi], [lo, hi], "r--", label="Perfect Pred")
        ax.set_xlabel(f"True {target_name}")
        ax.set_ylabel(f"Predicted {target_name}")
        ax.set_title(f"{name}: R²={r2:.4f}, MAE={mae:.5f}")
        ax.legend()

    ax4 = fig.add_subplot(gs[1, 0])
    scatter_true_pred(ax4, train, "TRAIN")

    ax5 = fig.add_subplot(gs[1, 1])
    scatter_true_pred(ax5, val, "VAL")

    ax6 = fig.add_subplot(gs[1, 2])
    scatter_true_pred(ax6, test, "TEST")

    # Residual plots
    def residual_plot(ax, split_data, name):
        y_true = split_data["y_true"]
        y_pred = split_data["y_pred"]
        resid = y_pred - y_true
        rmse = split_data["rmse"]
        ax.scatter(y_pred, resid, s=3, alpha=0.3, edgecolors="none")
        ax.axhline(0, color="k", linestyle="--")
        ax.axhline(rmse, color="orange", linestyle="--", label="±RMSE")
        ax.axhline(-rmse, color="orange", linestyle="--")
        ax.set_xlabel(f"Predicted {target_name}")
        ax.set_ylabel("Residuals")
        ax.set_title(f"{name} Residuals (RMSE={rmse:.4f})")
        ax.legend()

    ax7 = fig.add_subplot(gs[2, 0])
    residual_plot(ax7, train, "TRAIN")

    ax8 = fig.add_subplot(gs[2, 1])
    residual_plot(ax8, val, "VAL")

    ax9 = fig.add_subplot(gs[2, 2])
    residual_plot(ax9, test, "TEST")

    plt.tight_layout()
    out_path = f"{save_prefix}_overfitting_diagnostics.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved overfitting diagnostics figure to: {out_path}")
    plt.close(fig)


# --------------------------
# Main: TRAIN ONLY, no PDP/plots
# --------------------------
if __name__ == "__main__":
    # ---- Load your data here ----
    H5_PATH = "/Users/gabrieleiob/Library/CloudStorage/OneDrive-...etric_T_seeded/ddstartup_20251113_002935_parametric_T_seeded.h5"
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
    TARGET = "unrealized_profits"  # or "t_startup"

    # ---- Load data ----
    X, y = load_from_h5(H5_PATH, x_key=FEATURES, y_key=TARGET)
    print(f"Loaded data (raw) from {H5_PATH}: X.shape={X.shape}, y.shape={y.shape}")

    # ---- Clean data (remove NaN/Inf) ----
    print("\n" + "=" * 70)
    print("DATA CLEANING")
    print("=" * 70)

    n_nan_X = np.isnan(X).any(axis=1).sum()
    n_inf_X = np.isinf(X).any(axis=1).sum()
    n_nan_y = np.isnan(y).sum()
    n_inf_y = np.isinf(y).sum()

    print("Raw data quality:")
    print(f"  Rows with NaN in features: {n_nan_X:,} ({n_nan_X/len(X)*100:.2f}%)")
    print(f"  Rows with Inf in features: {n_inf_X:,} ({n_inf_X/len(X)*100:.2f}%)")
    print(f"  NaN in target: {n_nan_y:,} ({n_nan_y/len(y)*100:.2f}%)")
    print(f"  Inf in target: {n_inf_y:,} ({n_inf_y/len(y)*100:.2f}%)")
    print(f"  Target range (with NaN): [{np.nanmin(y):.2e}, {np.nanmax(y):.2e}]")

    # Filter out invalid samples
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    n_removed = len(X) - len(X_clean)
    print(f"\n✓ Removed {n_removed:,} invalid samples ({n_removed/len(X)*100:.2f}%)")
    print(f"✓ Clean data: X.shape={X_clean.shape}, y.shape={y_clean.shape}")
    print(f"✓ Target range (clean): [{y_clean.min():.2e}, {y_clean.max():.2e}]")
    print(f"✓ Target mean: {y_clean.mean():.2e}, std: {y_clean.std():.2e}")

    X, y = X_clean, y_clean
    if len(X) == 0:
        raise ValueError("No valid samples remaining after cleaning! Check your data.")

    # ---- Train model ----
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    model, xsc, ysc, r2, mae, rmse, nrmse, mape, device, diagnostics, y_shift = (
        train_model(
            X,
            y,
            hidden=(128, 64, 32),  # architecture
            dropout=0.5,
            lr=3e-4,
            weight_decay=1e-2,
            batch_size=16384,
            max_epochs=100,
            patience=10,
            num_workers=4,
            pin_memory=True,
            use_augmentation=True,
        )
    )

    print(
        f"\nTest R^2: {r2:.4f} | MAE: {mae:.4g} | RMSE: {rmse:.4g} | "
        f"NRMSE: {nrmse:.4g} | MAPE: {mape:.4g}"
    )

    # ---- Save artifacts ----
    with open("scalers.pkl", "wb") as f:
        pickle.dump({"x": xsc, "y": ysc, "y_shift": y_shift, "y_is_log": True}, f)
    torch.save(model.state_dict(), f"mlp_{X.shape[1]}d.pt")
    with open("diagnostics.pkl", "wb") as f:
        pickle.dump(diagnostics, f)
    print(f"\nSaved: mlp_{X.shape[1]}d.pt, scalers.pkl, diagnostics.pkl")
    print("\n✓ Training complete! Use pdp_analysis.ipynb for detailed PDP analysis.")
