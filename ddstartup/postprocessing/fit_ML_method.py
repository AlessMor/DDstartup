# -*- coding: utf-8 -*-
# mlp_pdp_13d.py
import os, math, pickle, numpy as np
import torch
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --------------------------
# Data utilities
# --------------------------
class ArrayDataset(Dataset):
    def __init__(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# --------------------------
# Model
# --------------------------
class MLPRegressor(nn.Module):
    def __init__(self, n_in=13, hidden=(256, 256, 128), dropout=0.05, act=nn.SiLU):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), act(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# --------------------------
# Training
# --------------------------
def train_model(X, y, hidden=(256,256,128), dropout=0.05, lr=1e-3, weight_decay=1e-4,
                batch_size=4096, max_epochs=200, patience=20, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sanity checks
    assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0], \
        f"Inconsistent shapes: X{X.shape}, y{y.shape}"

    # ---- Split first (on raw data) to avoid leakage ----
    X_tr_raw, X_tmp_raw, y_tr_raw, y_tmp_raw = train_test_split(X, y, test_size=0.30, random_state=42)
    X_va_raw, X_te_raw, y_va_raw, y_te_raw = train_test_split(X_tmp_raw, y_tmp_raw, test_size=0.50, random_state=42)

    # ---- Fit scalers on training only, then transform all splits ----
    x_scaler = StandardScaler().fit(X_tr_raw)
    y_scaler = StandardScaler().fit(y_tr_raw.reshape(-1, 1))

    X_tr = x_scaler.transform(X_tr_raw)
    X_va = x_scaler.transform(X_va_raw)
    X_te = x_scaler.transform(X_te_raw)

    y_tr = y_scaler.transform(y_tr_raw.reshape(-1, 1)).ravel()
    y_va = y_scaler.transform(y_va_raw.reshape(-1, 1)).ravel()
    y_te = y_scaler.transform(y_te_raw.reshape(-1, 1)).ravel()

    dl_tr = DataLoader(ArrayDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ArrayDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    dl_te = DataLoader(ArrayDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    model = MLPRegressor(n_in=X.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

    best_val = float('inf'); best_state = None; patience_left = patience
    for epoch in range(1, max_epochs+1):
        # Train
        model.train()
        running = 0.0; n = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0); n += xb.size(0)
        train_loss = running / n

        # Validate
        model.eval()
        running = 0.0; n = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                running += loss.item() * xb.size(0); n += xb.size(0)
        val_loss = running / n
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ---- Test metrics: R², MAE, RMSE (in original target units) ----
    def _eval_metrics(dl):
        y_true_s, y_pred_s = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device)
                y_pred_s.append(model(xb).cpu().numpy().ravel())
                y_true_s.append(yb.numpy().ravel())
        y_true_s = np.concatenate(y_true_s)
        y_pred_s = np.concatenate(y_pred_s)
        # inverse-transform to original units
        y_true = y_scaler.inverse_transform(y_true_s.reshape(-1,1)).ravel()
        y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1,1)).ravel()
        err = y_true - y_pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        y_std = float(np.std(y_true, ddof=0))
        r2 = 1.0 - (rmse**2) / (y_std**2 + 1e-12)
        nrmse = rmse / (y_std + 1e-12)
        mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-12)))
        return r2, mae, rmse, nrmse, mape

    r2, mae, rmse, nrmse, mape = _eval_metrics(dl_te)
    return model, x_scaler, y_scaler, r2, mae, rmse, nrmse, mape, device

# --------------------------
# Inference helpers
# --------------------------
def predict_numpy(model, x_scaler, y_scaler, X, device="cpu", mc_samples=0):
    """Returns y_pred in original scale. If mc_samples>0, returns (mean, std)."""
    Xs = x_scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)  # ensure float32
    xb = torch.from_numpy(Xs).to(device)  # now float32, matches model dtype
    with torch.no_grad():
        if mc_samples and any(isinstance(m, nn.Dropout) for m in model.modules()):
            preds = []
            model.train()  # enable dropout
            for _ in range(mc_samples):
                preds.append(model(xb).cpu().numpy())
            model.eval()
            yhat_s = np.stack(preds, axis=0).squeeze(-1)  # [T,N]
            yhat_mean = yhat_s.mean(axis=0, keepdims=True).T  # [N,1]
            yhat_std = yhat_s.std(axis=0, ddof=1, keepdims=True).T
            y_mean = y_scaler.inverse_transform(yhat_mean).ravel()
            y_std = yhat_std * float(y_scaler.scale_)  # std rescales by scaler.scale_
            return y_mean, y_std.ravel()
        else:
            yhat = model(xb).cpu().numpy()
            return y_scaler.inverse_transform(yhat).ravel()

def pairwise_pdp_grid(model, x_scaler, y_scaler, X_raw, i, j,
                      grid_size=60, bg_samples=128, qrange=(0.05, 0.95),
                      device="cpu", mc_samples=0):
    """
    Compute 2D PDP surface Z for features i, j:
      Z[u,v] = E_model[ y | x_i=grid_i[u], x_j=grid_j[v] ]
    Estimated by averaging predictions over bg_samples rows from X_raw (other features).
    """
    X_raw = np.asarray(X_raw, dtype=np.float32)
    n, d = X_raw.shape
    assert d > max(i, j)

    def _safe_quantile_range(col, qrange):
        lo, hi = np.quantile(col, qrange)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            m = np.nanmedian(col)
            span = max(1e-6, 0.01 * max(1.0, abs(m)))
            lo, hi = m - span, m + span
        return float(lo), float(hi)

    # grid ranges (restrict to quantiles; pad if degenerate)
    gi_lo, gi_hi = _safe_quantile_range(X_raw[:, i], qrange)
    gj_lo, gj_hi = _safe_quantile_range(X_raw[:, j], qrange)
    grid_i = np.linspace(gi_lo, gi_hi, grid_size, dtype=np.float32)
    grid_j = np.linspace(gj_lo, gj_hi, grid_size, dtype=np.float32)

    # background sample
    idx = np.random.choice(n, size=min(bg_samples, n), replace=False)
    BG = X_raw[idx].copy()  # [B, d]
    B = BG.shape[0]

    # Vectorized expansion: G = grid_size^2
    G = grid_size * grid_size
    BG_rep = np.tile(BG, (G, 1))                      # [G*B, d]
    ii, jj = np.meshgrid(grid_i, grid_j, indexing='ij')
    g_i = np.repeat(ii.ravel(), B)
    g_j = np.repeat(jj.ravel(), B)
    BG_rep[:, i] = g_i
    BG_rep[:, j] = g_j

    # Predict and average back to grid
    if mc_samples and any(isinstance(m, nn.Dropout) for m in model.modules()):
        y_mean, y_std = predict_numpy(model, x_scaler, y_scaler, BG_rep, device=device, mc_samples=mc_samples)
        Z = y_mean.reshape(grid_size, grid_size, B).mean(axis=2)
        Z_std = y_std.reshape(grid_size, grid_size, B).mean(axis=2)  # rough; you can refine if needed
        return grid_i, grid_j, Z, Z_std
    else:
        y_pred = predict_numpy(model, x_scaler, y_scaler, BG_rep, device=device)  # [G*B]
        Z = y_pred.reshape(grid_size, grid_size, B).mean(axis=2)                  # [gi, gj]
        return grid_i, grid_j, Z, None

def density_2d(X_raw, i, j, grid_i, grid_j, qrange=(0.05,0.95), bins=100):
    """Simple 2D histogram to visualize data support. Returns (H, extent)."""
    xi = np.clip(X_raw[:, i], grid_i.min(), grid_i.max())
    xj = np.clip(X_raw[:, j], grid_j.min(), grid_j.max())
    H, xedges, yedges = np.histogram2d(xi, xj, bins=bins,
                                       range=[[grid_i.min(), grid_i.max()],
                                              [grid_j.min(), grid_j.max()]])
    H = H.T
    H = H / (H.max() + 1e-9)
    # Safe extent (avoid identical limits)
    x0, x1 = float(grid_i.min()), float(grid_i.max())
    y0, y1 = float(grid_j.min()), float(grid_j.max())
    if x0 == x1: x0, x1 = x0 - 1e-6, x1 + 1e-6
    if y0 == y1: y0, y1 = y0 - 1e-6, y1 + 1e-6
    return H, (x0, x1, y0, y1)

def load_from_h5(path, x_key="X", y_key="y"):
    """Load features X and target y from an HDF5 file.
       x_key can be a string (single dataset) or a list/tuple of dataset names."""
    def _list_datasets(f):
        keys = []
        f.visititems(lambda name, obj: keys.append(name) if isinstance(obj, h5py.Dataset) else None)
        return keys

    def _resolve_key(f, name, all_ds):
        # If it's an exact dataset, accept it
        if name in f and isinstance(f[name], h5py.Dataset):
            return name
        # If it's a group, try to find a dataset inside it
        if name in f and isinstance(f[name], h5py.Group):
            grp = f[name]
            ds_in_group = []
            grp.visititems(lambda n, o: ds_in_group.append(f"{name}/{n}") if isinstance(o, h5py.Dataset) else None)
            # Prefer a dataset with the same basename
            same_basename = [ds for ds in ds_in_group if ds.split("/")[-1] == name]
            if len(same_basename) == 1:
                return same_basename[0]
            if len(ds_in_group) == 1:
                return ds_in_group[0]
            raise KeyError(f"'{name}' is a group, not a dataset. Datasets inside: {ds_in_group}")
        # Match by basename anywhere in the file
        candidates = [k for k in all_ds if k.endswith("/" + name) or k == name]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise KeyError(f"Ambiguous dataset base name '{name}'. Matches: {candidates}")
        raise KeyError(f"Could not find dataset '{name}'. Available datasets: {all_ds}")

    with h5py.File(path, "r") as f:
        all_ds = _list_datasets(f)

        # Build X
        if isinstance(x_key, (list, tuple)):
            cols = []
            for name in x_key:
                k = _resolve_key(f, name, all_ds)
                try:
                    arr = np.asarray(f[k][:], dtype=np.float32).reshape(-1)
                except Exception as e:
                    raise RuntimeError(f"Failed to read dataset '{k}' for feature '{name}': {e}")
                cols.append(arr)
            lengths = {len(c) for c in cols}
            if len(lengths) != 1:
                raise ValueError(f"Feature arrays have different lengths: {lengths}")
            X = np.column_stack(cols).astype(np.float32)
        else:
            kx = _resolve_key(f, x_key, all_ds)
            try:
                X = np.asarray(f[kx][:], dtype=np.float32)
            except Exception as e:
                raise RuntimeError(f"Failed to read dataset '{kx}' for X: {e}")
            if X.ndim == 1:
                X = X.reshape(-1, 1)

        # Build y
        ky = _resolve_key(f, y_key, all_ds)
        try:
            y = np.asarray(f[ky][:], dtype=np.float32).reshape(-1)
        except Exception as e:
            raise RuntimeError(f"Failed to read dataset '{ky}' for y: {e}")

    return X, y

# --------------------------
# Example usage
# --------------------------

if __name__ == "__main__":
    # ---- Load your data here ----
    H5_PATH = "C:\\Users\\gabriele.iob\\Desktop\\dd_startup\\outputs\\20251022_211344_parametric_T_seeded\\ddstartup_20251022_211344_parametric_T_seeded.h5"
    FEATURES = ["V_plasma", # 0
                "T_i", # 1
                "n_tot", # 2
                "tau_p_T", # 3
                "P_aux", # 4
                "P_aux_DT_eq", # 5
                "TBR_DT", # 6
                "TBR_DDn", # 7
                "tau_ifc", # 8
                "tau_ofc", # 9
                "eta_th", # 10
                "capacity_factor", # 11
                "cost_of_electricity", # 12
               ]
    TARGET = "unrealized_profits" # "unrealized_profits" # "t_startup"
    X, y = load_from_h5(H5_PATH, x_key=FEATURES, y_key=TARGET)
    print(f"Loaded data from {H5_PATH}: X.shape={X.shape}, y.shape={y.shape}")

    # ---- Train model ----
    model, xsc, ysc, r2, mae, rmse, nrmse, mape, device = train_model(
        X, y, hidden=(256,256,64), dropout=0.1,
        lr=1e-3, weight_decay=1e-4,
        batch_size=32, max_epochs=200, patience=20
    )
    print(f"Test R^2: {r2:.4f} | MAE: {mae:.4g} | RMSE: {rmse:.4g} | NRMSE: {nrmse:.4g} | MAPE: {mape:.4g}")

    # ---- Save artifacts ----
    with open("scalers.pkl", "wb") as f:
        pickle.dump({"x": xsc, "y": ysc}, f)
    torch.save(model.state_dict(), f"mlp_{X.shape[1]}d.pt")
    print(f"Saved: mlp_{X.shape[1]}d.pt, scalers.pkl")

    # ---- Make a pairwise PDP (features i,j) ----
    i, j = 0, 10  # choose the pair you want
    grid_i, grid_j, Z, Z_std = pairwise_pdp_grid(model, xsc, ysc, X, i, j,
                                                 grid_size=60, bg_samples=128,
                                                 qrange=(0.05,0.95), device=device, mc_samples=0)

    # ---- Plot: contour + data density overlay ----
    fig, ax = plt.subplots(figsize=(6,5))
    ci, cj = np.meshgrid(grid_i, grid_j, indexing='ij')
    cs = ax.contourf(ci, cj, Z, levels=40, alpha=0.85)
    cbar = fig.colorbar(cs, ax=ax, label=TARGET)

    H, extent = density_2d(X, i, j, grid_i, grid_j, bins=100)
    ax.imshow(H, extent=extent, origin='lower', alpha=0.25, aspect='auto')  # data support heat

    ax.set_xlabel(FEATURES[i])
    ax.set_ylabel(FEATURES[j])
    plt.tight_layout()
    plt.savefig(f"pdp_{FEATURES[i]}_{FEATURES[j]}.png", dpi=150)
    print(f"Saved: pdp_{FEATURES[i]}_{FEATURES[j]}.png")
