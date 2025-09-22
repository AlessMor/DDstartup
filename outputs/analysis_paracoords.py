# =============================================================================
# DD Startup Analysis - Parallel Coordinates Plot (Standalone Script)
# =============================================================================

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION OPTIONS - Set these before running the analysis
# =============================================================================

# File selection - Choose which HDF5 file to analyze
SELECTED_FILE = "dd_startup_results_20250919_152037.h5"  # Set to None for automatic selection, or specify filename

# Target variable selection - Choose which output metric to visualize
TARGET_VARIABLE = 't_startup'    

# Filter (sets the maximum value for color scaling)
FILTER = None        # Set to threshold value or None

# Color and styling options
N_COLOR_CHUNKS = 6           # Number of discrete color levels (None sets to gradient)

# =============================================================================
# 1. File Selection
# =============================================================================
outputs_dir = Path(__file__).parent
h5_files = list(outputs_dir.glob("*.h5"))

if SELECTED_FILE is not None:
    selected_file_path = outputs_dir / SELECTED_FILE
    if selected_file_path.exists():
        selected_file = selected_file_path
    else:
        raise FileNotFoundError(f"File '{SELECTED_FILE}' not found in outputs directory")
else:
    if h5_files:
        selected_file = max(h5_files, key=lambda f: f.stat().st_mtime)
    else:
        raise FileNotFoundError("No HDF5 files found in outputs directory")

print(f"\n✅ Selected file: {selected_file.name}")

# =============================================================================
# 2. Read HDF5 Data (independent of analysis mode)
# =============================================================================
data = {}
expected_length = None
with h5py.File(selected_file, 'r') as f:
    # Load all 1D datasets
    for key in f.keys():
        if isinstance(f[key], h5py.Dataset):
            arr = f[key][:]
            if arr.ndim == 1:
                if expected_length is None:
                    expected_length = len(arr)
                if len(arr) == expected_length:
                    data[key] = arr
    # Load parameter_fields group if present
    if 'parameter_fields' in f:
        param_group = f['parameter_fields']
        for subkey in param_group.keys():
            arr = param_group[subkey][:]
            if arr.ndim == 1 and len(arr) == expected_length:
                data[subkey] = arr

if not data:
    raise ValueError("No valid 1D datasets found in the HDF5 file.")

# =============================================================================
# 3. DataFrame Creation and Filtering
# =============================================================================
df = pd.DataFrame(data)

# Filter out infinite/NaN target values
finite_mask = np.isfinite(df[TARGET_VARIABLE])
df_filtered = df[finite_mask].copy()

# Apply filter threshold if specified
if FILTER is not None:
    df_filtered = df_filtered[df_filtered[TARGET_VARIABLE] <= FILTER]

# =============================================================================
# 4. Identify Input Parameters
# =============================================================================
# Use all columns except known outputs as input axes
output_like = [
    'linear_index', 'injection_rate_max', 'sigmav_DT', 'sigmav_DD_p', 'sigmav_DD_n',
    'P_DT', 'P_DDn', 'P_DDp', 'P_DT_full', 'P_fusion_DD_avg', 'P_e_net_DD_avg',
    'P_e_net_DT_full_avg', 'Q_DD_total', 'Q_DT_full_total', 'E_fusion_total_DD',
    'E_fusion_DT_full', 'E_e_net_DD', 'E_e_net_DT_full', 'E_lost', 'Dollar_Lost',
    'n_T_final', 'sol_success', 't_startup', 'tau_ifc', 'tau_ofc'
]
input_parameters = [col for col in df_filtered.columns if col not in output_like and col != TARGET_VARIABLE]

# =============================================================================
# 5. Color Mapping (Green to Red, Discrete Chunks)
# =============================================================================
def get_discrete_colorscale(n_chunks):
    # Green (low) to Red (high)
    base_colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF8C00', '#FF4500', '#8B0000']
    if n_chunks > len(base_colors):
        # Interpolate if more chunks needed
        from matplotlib import cm
        cmap = cm.get_cmap('RdYlGn_r', n_chunks)
        return [[i/(n_chunks-1), f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'] for i, (r,g,b,_) in enumerate(cmap(np.linspace(0,1,n_chunks)))]
    else:
        return [[i/(n_chunks-1), base_colors[i]] for i in range(n_chunks)]

if N_COLOR_CHUNKS:
    # Discretize target variable
    target_values = df_filtered[TARGET_VARIABLE]
    quantiles = np.linspace(0, 1, N_COLOR_CHUNKS+1)
    chunk_bounds = target_values.quantile(quantiles).values
    color_indices = np.zeros(len(target_values), dtype=int)
    for i in range(N_COLOR_CHUNKS):
        if i == N_COLOR_CHUNKS-1:
            mask = target_values > chunk_bounds[i]
        else:
            mask = (target_values > chunk_bounds[i]) & (target_values <= chunk_bounds[i+1])
        color_indices[mask] = i
    color_data = color_indices
    colorscale = get_discrete_colorscale(N_COLOR_CHUNKS)
    cmin, cmax = 0, N_COLOR_CHUNKS-1
else:
    color_data = df_filtered[TARGET_VARIABLE]
    colorscale = 'RdYlGn_r'
    cmin, cmax = color_data.min(), color_data.max()

# =============================================================================
# 6. Parallel Coordinates Plot
# =============================================================================
dimensions = []
for param in input_parameters:
    values = df_filtered[param]
    unique_vals = np.sort(np.unique(values))
    dim = dict(
        label=param,
        values=values,
        range=[values.min(), values.max()]
    )
    if len(unique_vals) <= 20:
        dim['tickvals'] = unique_vals.tolist()
    dimensions.append(dim)
# Add target variable as last axis
values = df_filtered[TARGET_VARIABLE]
dimensions.append(dict(
    label=TARGET_VARIABLE,
    values=values,
    range=[values.min(), values.max()]
))

fig = go.Figure(data=go.Parcoords(
    line=dict(
        color=color_data,
        colorscale=colorscale,
        showscale=True,
        cmin=cmin,
        cmax=cmax,
        colorbar=dict(
            title=TARGET_VARIABLE,
            thickness=20,
            len=0.8
        )
    ),
    dimensions=dimensions
))

fig.update_layout(
    title=f"DD Startup Parallel Coordinates Plot ({selected_file.name})",
    font=dict(size=12),
    width=1400,
    height=700,
    margin=dict(l=100, r=120, t=120, b=100),
    paper_bgcolor='white',
    plot_bgcolor='white'
)

fig.write_html(outputs_dir / f"paracoords_plot_{TARGET_VARIABLE}_{selected_file.name.strip('dd_startup_')}.html")
print(f"Plot saved as {outputs_dir / f'paracoords_plot_{TARGET_VARIABLE}_{selected_file.name.strip('dd_startup_')}.html'}")