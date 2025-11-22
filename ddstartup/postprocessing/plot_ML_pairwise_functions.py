"""
ML Pairwise Plot Functions (refactored for the new DF + dispatcher)

Expected call (kwargs filtered by your dispatcher):
generate_ml_pairwise_plots(
    df=..., target=..., inputs=..., target_unit=..., output_dir=...,
    file_type=..., plot_name_prefix=..., registry=..., ml_pairwise_settings={...},
    # extra kwargs tolerated via **_
)
"""

from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional deps
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available - skipping ML pairwise plots")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available - skipping ML pairwise plots")


# --------------------------
# Data/Model utilities
# --------------------------
if TORCH_AVAILABLE:
    class ArrayDataset(Dataset):
        def __init__(self, X, y):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
        def __len__(self): return self.X.shape[0]
        def __getitem__(self, i): return self.X[i], self.y[i]

if TORCH_AVAILABLE:
    class MLPRegressor(nn.Module):
        def __init__(self, n_in, hidden=(256, 256, 128), dropout=0.05, act=nn.SiLU):
            super().__init__()
            layers, prev = [], n_in
            for h in hidden:
                layers += [nn.Linear(prev, h), act(), nn.Dropout(dropout)]
                prev = h
            layers += [nn.Linear(prev, 1)]
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)


def _train_model(X, y, *, hidden=(256,256,128), dropout=0.05, lr=1e-3, weight_decay=1e-4,
                 batch_size=4096, max_epochs=200, patience=20, device=None, verbose=False):
    if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
        raise RuntimeError("PyTorch and scikit-learn are required for ML pairwise plots")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0], f"Bad shapes X{X.shape}, y{y.shape}"

    # Split BEFORE scaling
    X_tr_raw, X_tmp_raw, y_tr_raw, y_tmp_raw = train_test_split(X, y, test_size=0.30, random_state=42)
    X_va_raw, X_te_raw, y_va_raw, y_te_raw = train_test_split(X_tmp_raw, y_tmp_raw, test_size=0.50, random_state=42)

    x_scaler = StandardScaler().fit(X_tr_raw)
    y_scaler = StandardScaler().fit(y_tr_raw.reshape(-1, 1))

    X_tr = x_scaler.transform(X_tr_raw); X_va = x_scaler.transform(X_va_raw); X_te = x_scaler.transform(X_te_raw)
    y_tr = y_scaler.transform(y_tr_raw.reshape(-1, 1)).ravel()
    y_va = y_scaler.transform(y_va_raw.reshape(-1, 1)).ravel()
    y_te = y_scaler.transform(y_te_raw.reshape(-1, 1)).ravel()

    # guard for small datasets
    bs = max(32, min(batch_size, len(X_tr)))
    dl_tr = DataLoader(ArrayDataset(X_tr, y_tr), batch_size=bs, shuffle=True, drop_last=False)
    dl_va = DataLoader(ArrayDataset(X_va, y_va), batch_size=bs, shuffle=False)
    dl_te = DataLoader(ArrayDataset(X_te, y_te), batch_size=bs, shuffle=False)

    model = MLPRegressor(n_in=X.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

    best_val = float("inf"); best_state = None; patience_left = patience
    for epoch in range(1, max_epochs + 1):
        # train
        model.train(); run = 0.0; n = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb); loss = loss_fn(pred, yb)
            loss.backward(); opt.step()
            run += loss.item() * xb.size(0); n += xb.size(0)
        train_loss = run / max(n, 1)

        # val
        model.eval(); run = 0.0; n = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb); loss = loss_fn(pred, yb)
                run += loss.item() * xb.size(0); n += xb.size(0)
        val_loss = run / max(n, 1)
        scheduler.step(val_loss)

        if verbose and epoch % 10 == 0:
            print(f"      Epoch {epoch}/{max_epochs}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose: print(f"      Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # metrics on test (inverse-transform)
    def _metrics(loader):
        y_true_s, y_pred_s = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                y_pred_s.append(model(xb).cpu().numpy().ravel())
                y_true_s.append(yb.numpy().ravel())
        y_true_s = np.concatenate(y_true_s); y_pred_s = np.concatenate(y_pred_s)
        y_true = y_scaler.inverse_transform(y_true_s.reshape(-1, 1)).ravel()
        y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
        err = y_true - y_pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        y_std = float(np.std(y_true, ddof=0))
        r2 = 1.0 - (rmse**2) / (y_std**2 + 1e-12)
        nrmse = rmse / (y_std + 1e-12)
        mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-12)))
        return r2, mae, rmse, nrmse, mape

    r2, mae, rmse, nrmse, mape = _metrics(dl_te)
    return model, x_scaler, y_scaler, r2, mae, rmse, nrmse, mape, device


def _predict_numpy(model, x_scaler, y_scaler, X, device="cpu"):
    Xs = x_scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
    xb = torch.from_numpy(Xs).to(device)
    with torch.no_grad():
        yhat = model(xb).cpu().numpy()
        return y_scaler.inverse_transform(yhat).ravel()


def _safe_quantile_range(col, qrange):
    lo, hi = np.quantile(col, qrange)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        m = np.nanmedian(col); span = max(1e-6, 0.01 * max(1.0, abs(m)))
        lo, hi = m - span, m + span
    return float(lo), float(hi)


def _pairwise_pdp_grid(model, x_scaler, y_scaler, X_raw, i, j,
                       grid_size=60, bg_samples=128, qrange=(0.05, 0.95),
                       device="cpu"):
    X_raw = np.asarray(X_raw, dtype=np.float32)
    n, d = X_raw.shape
    assert d > max(i, j)

    gi_lo, gi_hi = _safe_quantile_range(X_raw[:, i], qrange)
    gj_lo, gj_hi = _safe_quantile_range(X_raw[:, j], qrange)
    grid_i = np.linspace(gi_lo, gi_hi, grid_size, dtype=np.float32)
    grid_j = np.linspace(gj_lo, gj_hi, grid_size, dtype=np.float32)

    # background subset
    idx = np.random.choice(n, size=min(bg_samples, n), replace=False)
    BG = X_raw[idx].copy()
    B = BG.shape[0]

    # expand to grid
    G = grid_size * grid_size
    BG_rep = np.tile(BG, (G, 1))
    ii, jj = np.meshgrid(grid_i, grid_j, indexing="ij")
    g_i = np.repeat(ii.ravel(), B); g_j = np.repeat(jj.ravel(), B)
    BG_rep[:, i] = g_i; BG_rep[:, j] = g_j

    y_pred = _predict_numpy(model, x_scaler, y_scaler, BG_rep, device=device)
    Z = y_pred.reshape(grid_size, grid_size, B).mean(axis=2)
    return grid_i, grid_j, Z


def _density_2d(X_raw, i, j, grid_i, grid_j, bins=100):
    xi = np.clip(X_raw[:, i], grid_i.min(), grid_i.max())
    xj = np.clip(X_raw[:, j], grid_j.min(), grid_j.max())
    H, xedges, yedges = np.histogram2d(
        xi, xj, bins=bins,
        range=[[grid_i.min(), grid_i.max()],
               [grid_j.min(), grid_j.max()]]
    )
    H = (H.T) / (H.max() + 1e-9)
    x0, x1 = float(grid_i.min()), float(grid_i.max())
    y0, y1 = float(grid_j.min()), float(grid_j.max())
    if x0 == x1: x0, x1 = x0 - 1e-6, x1 + 1e-6
    if y0 == y1: y0, y1 = y0 - 1e-6, y1 + 1e-6
    return H, (x0, x1, y0, y1)


# --------------------------
# Public entry point
# --------------------------
def generate_ml_pairwise_plots(
    *,
    df: pd.DataFrame,
    target: str,
    inputs: list[str],
    target_unit: str,
    output_dir: Path,
    file_type: str,
    plot_name_prefix: str,
    registry=None,
    ml_pairwise_settings: dict | None = None,
    **_
) -> None:
    """
    New signature aligned with your dispatcher.

    df: scalar columns are numeric; vector columns are object (arrays) -> ignored for ML.
    inputs: all candidate inputs (will be filtered to usable scalar, varying, finite).
    """
    if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
        print("   ⚠️  PyTorch and/or scikit-learn not available - skipping ML pairwise plots")
        return

    mlcfg = ml_pairwise_settings or {}
    pairs_mode     = mlcfg.get("pairs", "auto")              # 'auto' | 'all' | list[(i,j)]
    grid_size      = int(mlcfg.get("grid_size", 60))
    max_train      = int(mlcfg.get("max_train_samples", 100_000))
    hidden         = tuple(mlcfg.get("hidden", (256, 256, 128)))
    verbose        = bool(mlcfg.get("verbose", False))

    # 1) Select usable scalar inputs
    usable = []
    for col in inputs:
        if col == target: 
            continue
        s = df[col]
        if s.dtype == object:
            # vector/array column -> skip
            continue
        x = pd.to_numeric(s, errors="coerce").values
        finite = np.isfinite(x)
        if finite.mean() < 0.99:
            # too many NaNs/Infs -> skip as feature
            continue
        if np.nanstd(x) <= 1e-12:
            # constant feature -> skip
            continue
        usable.append(col)

    if len(usable) < 2:
        print(f"   ⚠️  Need at least 2 usable scalar inputs (found {len(usable)}). Skipping ML pairwise.")
        return

    # 2) Build clean matrix (drop rows with NaN in inputs/target)
    cols = usable + [target]
    M = df[cols].apply(pd.to_numeric, errors="coerce")
    M = M.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if M.empty or len(M) < 200:
        print("   ⚠️  Not enough clean rows after NaN filtering. Skipping ML pairwise.")
        return

    X = M[usable].values.astype(np.float32)
    y = M[target].values.astype(np.float32)

    # 3) Subsample for memory/time
    if len(X) > max_train:
        if verbose:
            print(f"      Subsampling from {len(X):,} to {max_train:,} samples for training")
        idx = np.random.choice(len(X), size=max_train, replace=False)
        X = X[idx]; y = y[idx]

    print("   Training ML surrogate model...")
    try:
        model, xsc, ysc, r2, mae, rmse, nrmse, mape, device = _train_model(
            X, y,
            hidden=hidden, dropout=0.10,
            lr=1e-3, weight_decay=1e-4,
            batch_size=max(32, min(4096, len(X)//10)),
            max_epochs=200, patience=20,
            device=None, verbose=verbose
        )
        print(f"      Model R²={r2:.3f}, RMSE={rmse:.4g}, NRMSE={nrmse:.3f}")
        if r2 < 0.7:
            print("      ⚠️  Low R²: pairwise surfaces may be unreliable.")
    except Exception as e:
        print(f"   ❌ Error training ML model: {e}")
        return

    # 4) Choose pairs
    if pairs_mode == "all":
        pair_list = [(i, j) for i in range(len(usable)) for j in range(i+1, len(usable))]
    elif pairs_mode == "auto":
        # take top-k by absolute Pearson corr with y
        k = min(5, len(usable))
        cors = []
        for i, name in enumerate(usable):
            try:
                c = np.corrcoef(X[:, i], y)[0, 1]
            except Exception:
                c = 0.0
            if not np.isfinite(c):
                c = 0.0
            cors.append((abs(float(c)), i))
        cors.sort(reverse=True)
        idxs = [i for _, i in cors[:k]]
        pair_list = [(i, j) for i in idxs for j in idxs if i < j]
        if verbose:
            picked = ", ".join(usable[i] for i in idxs)
            print(f"      Auto pairs among top features: {picked} -> {len(pair_list)} pairs")
    else:
        # assume explicit list of (i,j) in feature indices or names
        if isinstance(pairs_mode, (list, tuple)) and len(pairs_mode) and isinstance(pairs_mode[0], (list, tuple)):
            pair_list = [(int(i), int(j)) for (i, j) in pairs_mode]
        else:
            print("      ⚠️  'pairs' misconfigured; using 'auto'")
            k = min(5, len(usable))
            idxs = list(range(k))
            pair_list = [(i, j) for i in idxs for j in idxs if i < j]

    if not pair_list:
        print("   ⚠️  No pairs to plot. Skipping.")
        return

    # 5) Plot pairs
    print(f"   Generating {len(pair_list)} pairwise PDP plots...")
    for pidx, (i, j) in enumerate(pair_list, 1):
        pi, pj = usable[i], usable[j]
        try:
            gi, gj, Z = _pairwise_pdp_grid(model, xsc, ysc, X, i, j,
                                           grid_size=grid_size, bg_samples=128,
                                           qrange=(0.05, 0.95), device=device)

            # base fig
            fig, ax = plt.subplots(figsize=(7.5, 6.0))
            II, JJ = np.meshgrid(gi, gj, indexing="ij")
            cs = ax.contourf(II, JJ, Z, levels=40, alpha=0.9, cmap="viridis")
            cbar = fig.colorbar(cs, ax=ax)

            # target label + unit
            if registry is not None:
                tlabel = registry.get_param_label(target)
            else:
                tlabel = target
            if target_unit:
                cbar.set_label(f"{tlabel} [{target_unit}]", fontsize=11)
            else:
                cbar.set_label(tlabel, fontsize=11)

            # data density overlay
            H, extent = _density_2d(X, i, j, gi, gj, bins=100)
            ax.imshow(H, extent=extent, origin="lower", alpha=0.25, aspect="auto", cmap="gray")

            # axis labels
            xi = registry.get_param_label(pi) if registry is not None else pi
            xj = registry.get_param_label(pj) if registry is not None else pj
            ax.set_xlabel(xi, fontsize=11)
            ax.set_ylabel(xj, fontsize=11)

            # title
            tshort = registry.get_param_label(target) if registry is not None else target
            ax.set_title(f"ML Pairwise PDP: {tshort} ({file_type})", fontsize=12, fontweight="bold")

            plt.tight_layout()

            fname = f"{plot_name_prefix}_{target}_{pi}x{pj}.png"
            plt.savefig(Path(output_dir) / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

            if pidx == 1:
                print(f"      Saved: {fname}")
        except Exception as e:
            print(f"      ❌ Error plotting {pi} vs {pj}: {e}")
            continue

    if len(pair_list) > 1:
        print(f"      ... and {len(pair_list) - 1} more pairwise plots")
    print("   ✅ ML pairwise plots complete")
