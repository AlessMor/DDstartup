# -*- coding: utf-8 -*-
# mlp_grid_plot_13d.py
import pickle
import numpy as np
import torch
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
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

# --------------------------
# Grid evaluation 
# --------------------------
def baseline_from_data(X_raw, strategy="median"):
    """Return a baseline feature vector to hold constant (median or mean)."""
    if strategy == "median":
        return np.median(X_raw, axis=0).astype(np.float32)
    elif strategy == "mean":
        return np.mean(X_raw, axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy}")

def feature_range(X_raw, idx, name=None, explicit=None, qrange=(0.05, 0.95)):
    """Return (lo, hi) for feature idx. Prefer explicit[name] if provided, else use quantiles."""
    if explicit and name in explicit:
        lo, hi = explicit[name]
        return float(lo), float(hi)
    col = X_raw[:, idx]
    lo, hi = np.quantile(col, qrange)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        m = float(np.nanmedian(col))
        span = max(1e-6, 0.01 * max(1.0, abs(m)))
        lo, hi = m - span, m + span
    return float(lo), float(hi)

def grid_output_two_features(model, x_scaler, y_scaler, baseline_x, i, j,
                             range_i, range_j, grid_size=60, device="cpu", mc_samples=0):
    """Fix all features to baseline_x, vary i and j, predict.
       Returns (grid_i, grid_j, Z) if mc_samples==0, else (grid_i, grid_j, Z_mean, Z_std)."""
    grid_i = np.linspace(range_i[0], range_i[1], grid_size, dtype=np.float32)
    grid_j = np.linspace(range_j[0], range_j[1], grid_size, dtype=np.float32)
    ii, jj = np.meshgrid(grid_i, grid_j, indexing='ij')
    G = grid_size * grid_size
    Xg = np.repeat(baseline_x.reshape(1, -1), G, axis=0)
    Xg[:, i] = ii.ravel()
    Xg[:, j] = jj.ravel()

    if mc_samples and any(isinstance(m, nn.Dropout) for m in model.modules()):
        y_mean, y_std = predict_numpy(model, x_scaler, y_scaler, Xg, device=device, mc_samples=mc_samples)
        Z_mean = y_mean.reshape(grid_size, grid_size)
        Z_std = y_std.reshape(grid_size, grid_size)
        return grid_i, grid_j, Z_mean, Z_std
    else:
        y_pred = predict_numpy(model, x_scaler, y_scaler, Xg, device=device)
        Z = y_pred.reshape(grid_size, grid_size)
        return grid_i, grid_j, Z

# --------------------------
# HDF5 loader
# --------------------------
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

    def _safe_read_dataset(f, key, name):
        """Try multiple methods to read a dataset."""
        ds = f[key]
        print(f"  Reading '{name}' from '{key}' | shape={ds.shape} | dtype={ds.dtype}")
        
        # Try direct slicing first
        try:
            data = ds[:]
            print(f"  ✓ Direct read successful")
            return data
        except Exception as e1:
            print(f"  ✗ Direct read failed: {e1}")
            
            # Try reading in chunks
            try:
                print(f"  Attempting chunked read...")
                chunk_size = 100000
                chunks = []
                for i in range(0, ds.shape[0], chunk_size):
                    end = min(i + chunk_size, ds.shape[0])
                    chunks.append(ds[i:end])
                data = np.concatenate(chunks)
                print(f"  ✓ Chunked read successful")
                return data
            except Exception as e2:
                print(f"  ✗ Chunked read failed: {e2}")
                
                # Try forcing load into memory
                try:
                    print(f"  Attempting forced memory load...")
                    data = np.array(ds)
                    print(f"  ✓ Forced memory load successful")
                    return data
                except Exception as e3:
                    raise RuntimeError(f"All read methods failed for '{key}': Direct={e1}, Chunked={e2}, Forced={e3}")

    with h5py.File(path, "r") as f:
        all_ds = _list_datasets(f)

        # Build X
        print("\nLoading features...")
        if isinstance(x_key, (list, tuple)):
            cols = []
            for name in x_key:
                k = _resolve_key(f, name, all_ds)
                try:
                    arr = _safe_read_dataset(f, k, name)
                    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
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
                X = _safe_read_dataset(f, kx, x_key)
                X = np.asarray(X, dtype=np.float32)
            except Exception as e:
                raise RuntimeError(f"Failed to read dataset '{kx}' for X: {e}")
            if X.ndim == 1:
                X = X.reshape(-1, 1)

        # Build y
        print(f"\nLoading target '{y_key}'...")
        ky = _resolve_key(f, y_key, all_ds)
        try:
            y = _safe_read_dataset(f, ky, y_key)
            y = np.asarray(y, dtype=np.float32).reshape(-1)
        except Exception as e:
            raise RuntimeError(f"Failed to read dataset '{ky}' for y: {e}")
        
        print(f"\n✓ Successfully loaded all data")

    return X, y

# --------------------------
# HDF5 utilities
# --------------------------
def inspect_h5_structure(path, max_items=50):
    """Print the structure of an HDF5 file to help debug loading issues."""
    print(f"\n{'='*60}")
    print(f"Inspecting HDF5 file: {path}")
    print(f"{'='*60}")
    
    with h5py.File(path, "r") as f:
        def print_structure(name, obj, indent=0):
            prefix = "  " * indent
            if isinstance(obj, h5py.Group):
                print(f"{prefix}📁 Group: {name}/")
            elif isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = obj.dtype
                print(f"{prefix}📄 Dataset: {name} | shape={shape} | dtype={dtype}")
        
        print("\nFile structure:")
        f.visititems(lambda n, o: print_structure(n, o, n.count('/')))
        
        print("\n" + "="*60)
        print("All dataset paths (use these for x_key/y_key):")
        print("="*60)
        datasets = []
        f.visititems(lambda n, o: datasets.append(n) if isinstance(o, h5py.Dataset) else None)
        for ds in datasets[:max_items]:
            print(f"  • {ds}")
        if len(datasets) > max_items:
            print(f"  ... and {len(datasets) - max_items} more")
        print()

# --------------------------
# K-Fold Cross Validation
# --------------------------
def kfold_cross_validation(X, y, n_splits=5, hidden=(256,256,128), dropout=0.05, 
                          lr=1e-3, weight_decay=1e-4, batch_size=4096, 
                          max_epochs=200, patience=20, device=None):
    """Perform k-fold cross-validation and return metrics + loss histories."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    all_histories = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n=== Fold {fold_idx}/{n_splits} ===")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Fit scalers on training data only
        x_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        
        X_train_scaled = x_scaler.transform(X_train)
        X_val_scaled = x_scaler.transform(X_val)
        y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        
        # Create dataloaders
        dl_train = DataLoader(ArrayDataset(X_train_scaled, y_train_scaled), 
                             batch_size=batch_size, shuffle=True, drop_last=False)
        dl_val = DataLoader(ArrayDataset(X_val_scaled, y_val_scaled), 
                           batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = MLPRegressor(n_in=X.shape[1], hidden=hidden, dropout=dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
        
        # Training loop with history tracking
        train_losses = []
        val_losses = []
        best_val = float('inf')
        best_state = None
        patience_left = patience
        
        for epoch in range(1, max_epochs+1):
            # Train
            model.train()
            running = 0.0
            n = 0
            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
                n += xb.size(0)
            train_loss = running / n
            train_losses.append(train_loss)
            
            # Validate
            model.eval()
            running = 0.0
            n = 0
            with torch.no_grad():
                for xb, yb in dl_val:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    running += loss.item() * xb.size(0)
                    n += xb.size(0)
            val_loss = running / n
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        
        # Evaluate on validation set
        y_true_list, y_pred_list = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                y_pred_list.append(model(xb).cpu().numpy().ravel())
                y_true_list.append(yb.numpy().ravel())
        
        y_true_scaled = np.concatenate(y_true_list)
        y_pred_scaled = np.concatenate(y_pred_list)
        
        # Inverse transform to original scale
        y_true = y_scaler.inverse_transform(y_true_scaled.reshape(-1,1)).ravel()
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
        
        # Calculate metrics
        err = y_true - y_pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        y_std = float(np.std(y_true, ddof=0))
        r2 = 1.0 - (rmse**2) / (y_std**2 + 1e-12)
        nrmse = rmse / (y_std + 1e-12)
        mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-12)))
        
        fold_results.append({
            'fold': fold_idx,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'nrmse': nrmse,
            'mape': mape
        })
        
        all_histories.append({
            'fold': fold_idx,
            'train_losses': train_losses,
            'val_losses': val_losses
        })
        
        print(f"Fold {fold_idx} - R²: {r2:.4f} | MAE: {mae:.4g} | RMSE: {rmse:.4g} | NRMSE: {nrmse:.4g} | MAPE: {mape:.4g}")
    
    return fold_results, all_histories

def plot_kfold_results(fold_results, all_histories, save_path="kfold_results.png"):
    """Plot k-fold cross-validation results."""
    n_folds = len(fold_results)
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Loss curves for each fold
    ax1 = fig.add_subplot(gs[0, :])
    for hist in all_histories:
        fold = hist['fold']
        epochs = range(1, len(hist['train_losses']) + 1)
        ax1.plot(epochs, hist['train_losses'], alpha=0.6, label=f'Fold {fold} Train')
        ax1.plot(epochs, hist['val_losses'], alpha=0.6, linestyle='--', label=f'Fold {fold} Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss Curves - All Folds')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Average loss curves
    ax2 = fig.add_subplot(gs[1, 0])
    max_epochs = max(len(h['train_losses']) for h in all_histories)
    avg_train = np.zeros(max_epochs)
    avg_val = np.zeros(max_epochs)
    counts = np.zeros(max_epochs)
    
    for hist in all_histories:
        n_ep = len(hist['train_losses'])
        avg_train[:n_ep] += hist['train_losses']
        avg_val[:n_ep] += hist['val_losses']
        counts[:n_ep] += 1
    
    avg_train /= np.maximum(counts, 1)
    avg_val /= np.maximum(counts, 1)
    
    epochs = range(1, max_epochs + 1)
    ax2.plot(epochs, avg_train, 'b-', linewidth=2, label='Avg Train Loss')
    ax2.plot(epochs, avg_val, 'r--', linewidth=2, label='Avg Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('Average Loss Curves Across Folds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Metrics comparison
    ax3 = fig.add_subplot(gs[1, 1])
    metrics_names = ['r2', 'mae', 'rmse', 'nrmse', 'mape']
    x_pos = np.arange(len(metrics_names))
    
    means = [np.mean([f[m] for f in fold_results]) for m in metrics_names]
    stds = [np.std([f[m] for f in fold_results]) for m in metrics_names]
    
    bars = ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics_names)
    ax3.set_ylabel('Value')
    ax3.set_title('Mean Metrics Across Folds (±std)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Metrics by fold
    ax4 = fig.add_subplot(gs[2, :])
    folds = [f['fold'] for f in fold_results]
    for metric in ['r2', 'mae', 'rmse']:
        values = [f[metric] for f in fold_results]
        ax4.plot(folds, values, marker='o', label=metric.upper())
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Metric Value')
    ax4.set_title('Metrics by Fold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(folds)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved k-fold results plot: {save_path}")
    
    # Print summary statistics
    print("\n=== Cross-Validation Summary ===")
    for metric in metrics_names:
        values = [f[metric] for f in fold_results]
        print(f"{metric.upper():8s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

# --------------------------
# Example usage
# --------------------------

if __name__ == "__main__":
    # ---- Load your data here ----
    H5_PATH = "C:\\Users\\gabriele.iob\\Desktop\\dd_startup\\outputs\\20251113_002935_parametric_T_seeded\\ddstartup_20251113_002935_parametric_T_seeded.h5"
    
    inspect_h5_structure(H5_PATH)
    
    FEATURES = ["parameter_fields/V_plasma_values",
                "parameter_fields/T_i_values",
                "parameter_fields/n_tot_values", 
                "parameter_fields/tau_p_T_values", 
                "parameter_fields/P_aux_values",
                "parameter_fields/P_aux_DT_eq_values",
                "parameter_fields/TBR_DT_values",
                "parameter_fields/TBR_DDn_values",
               ]
    TARGET = "t_startup"  # "unrealized_profits" "t_startup"
    X, y = load_from_h5(H5_PATH, x_key=FEATURES, y_key=TARGET)
    print(f"Loaded data from {H5_PATH}: X.shape={X.shape}, y.shape={y.shape}")

    # ---- Perform K-Fold Cross Validation ----
    print("\n" + "="*60)
    print("Starting 5-Fold Cross Validation")
    print("="*60)

    hidden = (32, 12)
    dropout = 0.15
    lr = 1e-3
    weight_decay = 1e-3
    batch_size = 64
    max_epochs = 40
    patience = 20
    
    fold_results, all_histories = kfold_cross_validation(
        X, y, n_splits=5, 
        hidden=hidden, dropout=dropout,
        lr=lr, weight_decay=weight_decay,
        batch_size=batch_size, max_epochs=max_epochs, patience=patience
    )
    
    # Plot results
    plot_kfold_results(fold_results, all_histories, save_path="kfold_validation_results.png")
    
    # ---- Train final model on full dataset ----
    print("\n" + "="*60)
    print("Training final model on full dataset")
    print("="*60)

    # ---- Train model ----
    model, xsc, ysc, r2, mae, rmse, nrmse, mape, device = train_model(
        X, y, hidden=hidden, dropout=dropout,
        lr=lr, weight_decay=weight_decay,
        batch_size=batch_size, max_epochs=max_epochs, patience=patience
    )
    print(f"Test R^2: {r2:.4f} | MAE: {mae:.4g} | RMSE: {rmse:.4g} | NRMSE: {nrmse:.4g} | MAPE: {mape:.4g}")

    # ---- Save artifacts ----
    with open("scalers.pkl", "wb") as f:
        pickle.dump({"x": xsc, "y": ysc}, f)
    torch.save(model.state_dict(), f"mlp_{X.shape[1]}d.pt")
    print(f"Saved: mlp_{X.shape[1]}d.pt, scalers.pkl")

    # ---- Choose the pair of features to vary ----
    i = FEATURES.index("V_plasma")
    j = FEATURES.index("cost_of_electricity")

    # Optional explicit ranges for features by name; if not set, uses quantiles
    EXPLICIT_RANGES = {
        # "V_plasma": (lo_value, hi_value),
        # "cost_of_electricity": (lo_value, hi_value),
    }
    range_i = feature_range(X, i, FEATURES[i], explicit=EXPLICIT_RANGES, qrange=(0.05, 0.95))
    range_j = feature_range(X, j, FEATURES[j], explicit=EXPLICIT_RANGES, qrange=(0.05, 0.95))

    # ---- Build baseline (fixed) vector for other inputs ----
    x_base = baseline_from_data(X, strategy="median")

    # ---- Evaluate model on the 2D grid (no PDP averaging) ----
    grid_size = 80
    MC_SAMPLES = 0  # e.g., 50 for MC dropout averaging
    out = grid_output_two_features(
        model, xsc, ysc, x_base, i, j, range_i, range_j,
        grid_size=grid_size, device=device, mc_samples=MC_SAMPLES
    )

    fig, ax = plt.subplots(figsize=(6,5))
    if MC_SAMPLES > 0 and len(out) == 4:
        grid_i, grid_j, Z_mean, Z_std = out
        ci, cj = np.meshgrid(grid_i, grid_j, indexing='ij')
        cs = ax.contourf(ci, cj, Z_mean, levels=40, alpha=0.95)
        fig.colorbar(cs, ax=ax, label=f"{TARGET} (mean)")
        # Optional: visualize uncertainty
        ax.contour(ci, cj, Z_std, levels=6, colors='k', linewidths=0.5)
    else:
        grid_i, grid_j, Z = out
        ci, cj = np.meshgrid(grid_i, grid_j, indexing='ij')
        cs = ax.contourf(ci, cj, Z, levels=40, alpha=0.95)
        fig.colorbar(cs, ax=ax, label=TARGET)
    ax.set_xlabel(FEATURES[i])
    ax.set_ylabel(FEATURES[j])
    plt.tight_layout()
    out_path = f"model_output_{FEATURES[i]}_{FEATURES[j]}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")