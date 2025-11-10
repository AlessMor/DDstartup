"""
ML Pairwise Plot Functions

This module contains functions for generating 2D pairwise partial dependence plots
using machine learning surrogate models (MLP Neural Networks). These plots show
how pairs of input parameters jointly affect target variables.

The ML model learns the complex nonlinear mapping from input parameters to targets,
then generates smooth 2D surfaces showing the predicted response across parameter space.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available - ML pairwise plots will be disabled")

# Check for scikit-learn availability  
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available - ML pairwise plots will be disabled")


# --------------------------
# Data utilities
# --------------------------
if TORCH_AVAILABLE:
    class ArrayDataset(Dataset):
        """PyTorch Dataset for numpy arrays."""
        def __init__(self, X, y):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
        
        def __len__(self):
            return self.X.shape[0]
        
        def __getitem__(self, i):
            return self.X[i], self.y[i]


# --------------------------
# Model
# --------------------------
if TORCH_AVAILABLE:
    class MLPRegressor(nn.Module):
        """Multi-layer perceptron regressor with configurable architecture."""
        def __init__(self, n_in=13, hidden=(256, 256, 128), dropout=0.05, act=nn.SiLU):
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
def train_model(X, y, hidden=(256, 256, 128), dropout=0.05, lr=1e-3, weight_decay=1e-4,
                batch_size=4096, max_epochs=200, patience=20, device=None, verbose=False):
    """
    Train an MLP regressor on the data.
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        hidden: Tuple of hidden layer sizes
        dropout: Dropout rate for regularization
        lr: Learning rate
        weight_decay: L2 regularization strength
        batch_size: Batch size for training
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience
        device: Device to train on ('cuda' or 'cpu')
        verbose: Whether to print training progress
        
    Returns:
        model: Trained model
        x_scaler: StandardScaler for inputs
        y_scaler: StandardScaler for outputs
        r2: R² score on test set
        mae: Mean absolute error on test set
        rmse: Root mean squared error on test set
        nrmse: Normalized RMSE
        mape: Mean absolute percentage error
        device: Device used for training
    """
    if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
        raise RuntimeError("PyTorch and scikit-learn are required for ML pairwise plots")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sanity checks
    assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0], \
        f"Inconsistent shapes: X{X.shape}, y{y.shape}"

    # Split first (on raw data) to avoid leakage
    X_tr_raw, X_tmp_raw, y_tr_raw, y_tmp_raw = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_va_raw, X_te_raw, y_va_raw, y_te_raw = train_test_split(
        X_tmp_raw, y_tmp_raw, test_size=0.50, random_state=42
    )

    # Fit scalers on training only, then transform all splits
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

    best_val = float('inf')
    best_state = None
    patience_left = patience
    
    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        running = 0.0
        n = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
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
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                running += loss.item() * xb.size(0)
                n += xb.size(0)
        val_loss = running / n
        scheduler.step(val_loss)

        if verbose and epoch % 10 == 0:
            print(f"      Epoch {epoch}/{max_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"      Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Test metrics: R², MAE, RMSE (in original target units)
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

    r2, mae, rmse, nrmse, mape = _eval_metrics(dl_te)
    return model, x_scaler, y_scaler, r2, mae, rmse, nrmse, mape, device


# --------------------------
# Inference helpers
# --------------------------
def predict_numpy(model, x_scaler, y_scaler, X, device="cpu", mc_samples=0):
    """
    Predict using trained model on numpy array inputs.
    
    Args:
        model: Trained PyTorch model
        x_scaler: StandardScaler for inputs
        y_scaler: StandardScaler for outputs
        X: Input features (n_samples, n_features)
        device: Device to run inference on
        mc_samples: Number of Monte Carlo samples for uncertainty (if >0, uses dropout)
        
    Returns:
        If mc_samples=0: y_pred in original scale
        If mc_samples>0: (y_mean, y_std) in original scale
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for ML pairwise plots")
    
    Xs = x_scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
    xb = torch.from_numpy(Xs).to(device)
    
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
            y_std = yhat_std * float(y_scaler.scale_)
            return y_mean, y_std.ravel()
        else:
            yhat = model(xb).cpu().numpy()
            return y_scaler.inverse_transform(yhat).ravel()


def pairwise_pdp_grid(model, x_scaler, y_scaler, X_raw, i, j,
                      grid_size=60, bg_samples=128, qrange=(0.05, 0.95),
                      device="cpu", mc_samples=0):
    """
    Compute 2D Partial Dependence Plot surface Z for features i, j.
    
    Z[u,v] = E_model[ y | x_i=grid_i[u], x_j=grid_j[v] ]
    
    Estimated by averaging predictions over bg_samples rows from X_raw (other features).
    
    Args:
        model: Trained PyTorch model
        x_scaler: StandardScaler for inputs
        y_scaler: StandardScaler for outputs
        X_raw: Raw input data (n_samples, n_features)
        i: Index of first feature
        j: Index of second feature
        grid_size: Number of grid points per dimension
        bg_samples: Number of background samples to average over
        qrange: Quantile range for grid bounds (e.g., (0.05, 0.95))
        device: Device to run inference on
        mc_samples: Number of MC samples for uncertainty estimation
        
    Returns:
        grid_i: 1D array of values for feature i
        grid_j: 1D array of values for feature j
        Z: 2D array of predicted values (grid_size x grid_size)
        Z_std: 2D array of prediction std (if mc_samples>0), else None
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for ML pairwise plots")
    
    X_raw = np.asarray(X_raw, dtype=np.float32)
    n, d = X_raw.shape
    assert d > max(i, j), f"Feature indices {i}, {j} out of bounds for {d} features"

    def _safe_quantile_range(col, qrange):
        lo, hi = np.quantile(col, qrange)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            m = np.nanmedian(col)
            span = max(1e-6, 0.01 * max(1.0, abs(m)))
            lo, hi = m - span, m + span
        return float(lo), float(hi)

    # Grid ranges (restrict to quantiles; pad if degenerate)
    gi_lo, gi_hi = _safe_quantile_range(X_raw[:, i], qrange)
    gj_lo, gj_hi = _safe_quantile_range(X_raw[:, j], qrange)
    grid_i = np.linspace(gi_lo, gi_hi, grid_size, dtype=np.float32)
    grid_j = np.linspace(gj_lo, gj_hi, grid_size, dtype=np.float32)

    # Background sample
    idx = np.random.choice(n, size=min(bg_samples, n), replace=False)
    BG = X_raw[idx].copy()  # [B, d]
    B = BG.shape[0]

    # Vectorized expansion: G = grid_size^2
    G = grid_size * grid_size
    BG_rep = np.tile(BG, (G, 1))  # [G*B, d]
    ii, jj = np.meshgrid(grid_i, grid_j, indexing='ij')
    g_i = np.repeat(ii.ravel(), B)
    g_j = np.repeat(jj.ravel(), B)
    BG_rep[:, i] = g_i
    BG_rep[:, j] = g_j

    # Predict and average back to grid
    if mc_samples and any(isinstance(m, nn.Dropout) for m in model.modules()):
        y_mean, y_std = predict_numpy(model, x_scaler, y_scaler, BG_rep, device=device, mc_samples=mc_samples)
        Z = y_mean.reshape(grid_size, grid_size, B).mean(axis=2)
        Z_std = y_std.reshape(grid_size, grid_size, B).mean(axis=2)
        return grid_i, grid_j, Z, Z_std
    else:
        y_pred = predict_numpy(model, x_scaler, y_scaler, BG_rep, device=device)
        Z = y_pred.reshape(grid_size, grid_size, B).mean(axis=2)
        return grid_i, grid_j, Z, None


def density_2d(X_raw, i, j, grid_i, grid_j, qrange=(0.05, 0.95), bins=100):
    """
    Simple 2D histogram to visualize data support overlay.
    
    Args:
        X_raw: Raw input data (n_samples, n_features)
        i: Index of first feature
        j: Index of second feature
        grid_i: 1D array of grid values for feature i
        grid_j: 1D array of grid values for feature j
        qrange: Quantile range (not used, kept for API compatibility)
        bins: Number of bins for histogram
        
    Returns:
        H: 2D histogram (normalized to [0, 1])
        extent: Tuple of (x_min, x_max, y_min, y_max) for imshow
    """
    xi = np.clip(X_raw[:, i], grid_i.min(), grid_i.max())
    xj = np.clip(X_raw[:, j], grid_j.min(), grid_j.max())
    
    H, xedges, yedges = np.histogram2d(
        xi, xj, bins=bins,
        range=[[grid_i.min(), grid_i.max()],
               [grid_j.min(), grid_j.max()]]
    )
    H = H.T
    H = H / (H.max() + 1e-9)
    
    # Safe extent (avoid identical limits)
    x0, x1 = float(grid_i.min()), float(grid_i.max())
    y0, y1 = float(grid_j.min()), float(grid_j.max())
    if x0 == x1:
        x0, x1 = x0 - 1e-6, x1 + 1e-6
    if y0 == y1:
        y0, y1 = y0 - 1e-6, y1 + 1e-6
    
    return H, (x0, x1, y0, y1)


# --------------------------
# Main plotting function
# --------------------------
def generate_ml_pairwise_plots(df_filtered, target, input_parameters, target_unit, 
                              outputs_dir, file_type, plot_name_prefix, registry=None,
                              pairs='auto', grid_size=60, max_train_samples=100000,
                              hidden=(256, 256, 128), verbose=False):
    """
    Generate ML-based pairwise partial dependence plots for all pairs of input parameters.
    
    This function:
    1. Trains an MLP neural network to predict the target from input parameters
    2. Generates 2D partial dependence plots showing how pairs of parameters affect the target
    3. Overlays data density to show which regions are well-supported by data
    
    Args:
        df_filtered: Filtered DataFrame with data
        target: Target variable name
        input_parameters: List of input parameter names
        target_unit: Unit string for target variable
        outputs_dir: Directory to save plots
        file_type: Type of file (for plot title)
        plot_name_prefix: Prefix for saved plot files
        registry: ParameterRegistry instance (optional)
        pairs: 'auto' (top 5x5 pairs), 'all', or list of (i,j) tuples
        grid_size: Number of grid points per dimension for PDP surface
        max_train_samples: Maximum number of samples to use for training
        hidden: Tuple of hidden layer sizes for MLP
        verbose: Whether to print detailed progress
    """
    if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
        print(f"   ⚠️  PyTorch and/or scikit-learn not available - skipping ML pairwise plots")
        return
    
    # Get registry if not provided
    if registry is None:
        from ddstartup.utils.parameter_registry import get_registry
        registry = get_registry()
    
    n_inputs = len(input_parameters)
    if n_inputs < 2:
        print(f"   ⚠️  Need at least 2 input parameters for pairwise plots (found {n_inputs}). Skipping.")
        return
    
    print(f"   Training ML surrogate model...")
    
    # Prepare data
    X = df_filtered[input_parameters].values.astype(np.float32)
    y = df_filtered[target].values.astype(np.float32)
    
    # Subsample if needed
    if len(X) > max_train_samples:
        if verbose:
            print(f"      Subsampling from {len(X):,} to {max_train_samples:,} samples for training")
        idx = np.random.choice(len(X), size=max_train_samples, replace=False)
        X = X[idx]
        y = y[idx]
    
    # Train model
    try:
        model, x_scaler, y_scaler, r2, mae, rmse, nrmse, mape, device = train_model(
            X, y, hidden=hidden, dropout=0.1,
            lr=1e-3, weight_decay=1e-4,
            batch_size=min(4096, len(X) // 10),
            max_epochs=200, patience=20,
            device=None, verbose=verbose
        )
        print(f"      Model trained: R²={r2:.4f}, RMSE={rmse:.4g}, NRMSE={nrmse:.4f}")
        
        if r2 < 0.7:
            print(f"      ⚠️  Warning: Low R² score ({r2:.3f}) - predictions may be unreliable")
    
    except Exception as e:
        print(f"   ❌ Error training ML model: {e}")
        return
    
    # Determine which pairs to plot
    if pairs == 'auto':
        # Plot top 5 most important parameters (pairwise)
        n_top = min(5, n_inputs)
        param_indices = list(range(n_top))
        pair_list = [(i, j) for i in param_indices for j in param_indices if i < j]
        if verbose:
            print(f"      Auto mode: plotting top {n_top} parameters ({len(pair_list)} pairs)")
    elif pairs == 'all':
        # Plot all pairs
        pair_list = [(i, j) for i in range(n_inputs) for j in range(n_inputs) if i < j]
        if verbose:
            print(f"      All pairs mode: plotting {len(pair_list)} pairs")
    else:
        # User-provided list of pairs
        pair_list = pairs
        if verbose:
            print(f"      Custom pairs: plotting {len(pair_list)} pairs")
    
    # Generate plots for each pair
    print(f"   Generating {len(pair_list)} pairwise PDP plots...")
    
    for pair_idx, (i, j) in enumerate(pair_list):
        param_i = input_parameters[i]
        param_j = input_parameters[j]
        
        if verbose:
            print(f"      [{pair_idx+1}/{len(pair_list)}] {param_i} vs {param_j}")
        
        try:
            # Compute PDP grid
            grid_i, grid_j, Z, Z_std = pairwise_pdp_grid(
                model, x_scaler, y_scaler, X, i, j,
                grid_size=grid_size, bg_samples=128,
                qrange=(0.05, 0.95), device=device, mc_samples=0
            )
            
            # Create plot
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot contour surface
            ci, cj = np.meshgrid(grid_i, grid_j, indexing='ij')
            cs = ax.contourf(ci, cj, Z, levels=40, alpha=0.85, cmap='viridis')
            cbar = fig.colorbar(cs, ax=ax)
            
            # Get target label with unit
            target_label = registry.get_param_label(target, unit=target_unit)
            cbar.set_label(target_label, fontsize=11)
            
            # Overlay data density
            H, extent = density_2d(X, i, j, grid_i, grid_j, bins=100)
            ax.imshow(H, extent=extent, origin='lower', alpha=0.25, 
                     aspect='auto', cmap='gray')
            
            # Get parameter labels with units
            label_i = registry.get_param_label(param_i)
            label_j = registry.get_param_label(param_j)
            
            ax.set_xlabel(label_i, fontsize=11)
            ax.set_ylabel(label_j, fontsize=11)
            
            # Get title label (without unit for title)
            target_title = registry.get_param_label(target, unit=None)
            ax.set_title(f'ML Pairwise PDP: {target_title} ({file_type})', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{plot_name_prefix}_{param_i}_{param_j}.png"
            plt.savefig(outputs_dir / plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            if pair_idx == 0 or verbose:
                print(f"      Saved: {plot_filename}")
        
        except Exception as e:
            print(f"      ❌ Error plotting {param_i} vs {param_j}: {e}")
            continue
    
    if not verbose and len(pair_list) > 1:
        print(f"      ... and {len(pair_list)-1} more pairwise plots")
    
    print(f"   ✅ ML pairwise plots complete")
