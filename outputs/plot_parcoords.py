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
SELECTED_FILE = "dd_startup_20250923_112837_parametric_T_seeded.h5"  # Set to None for automatic selection, or specify filename

# Target variable selection - Choose which output metric to visualize
TARGET_VARIABLE = 'Dollar_Lost'  # Example: 'Dollar_Lost', 't_startup', 'P_e_net_DD_avg', etc.

# Filter (sets the maximum value for color scaling)
FILTER = 500e6      # Set to threshold value or None

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

# print minimum and maximum of target variable
if TARGET_VARIABLE in data:
    target_values = data[TARGET_VARIABLE]
    finite_target_values = target_values[np.isfinite(target_values)]
    if finite_target_values.size > 0:
        print(f"Target variable '{TARGET_VARIABLE}' stats: min={finite_target_values.min()}, max={finite_target_values.max()}")
    else:
        print(f"Target variable '{TARGET_VARIABLE}' contains no finite values.")
else:
    raise KeyError(f"Target variable '{TARGET_VARIABLE}' not found in the data.")

# =============================================================================
# 3. DataFrame Creation and Filtering
# =============================================================================
df = pd.DataFrame(data)
# Convert Cost_per_kWh from 1/J to 1/kWh if present
if 'Cost_per_kWh' in df.columns:
    df['Cost_per_kWh'] = df['Cost_per_kWh'] * 3.6e6

# Filter out infinite/NaN target values
finite_mask = np.isfinite(df[TARGET_VARIABLE])
df_filtered = df[finite_mask].copy()

# Apply filter threshold if specified
if FILTER is not None:
    df_filtered = df_filtered[df_filtered[TARGET_VARIABLE] <= FILTER]
        
        # Randomly sample up to 1e6 rows for plotting
max_plot_rows = int(1e6)
if len(df_filtered) > max_plot_rows:
    df_filtered = df_filtered.sample(n=max_plot_rows, random_state=42)
    print(f"Sampled {max_plot_rows} rows for plotting (out of {len(df[finite_mask])} finite rows)")
else:
    print(f"Plotting all {len(df_filtered)} finite rows")

# =============================================================================
# 4. Identify Input Parameters
# =============================================================================
# Use all columns except known outputs as input axes
output_like = [
    'linear_index', 'injection_rate_max', 'sigmav_DT', 'sigmav_DD_p', 'sigmav_DD_n',
    'P_DT', 'P_DDn', 'P_DDp', 'P_DT_full', 'P_fusion_DD_avg', 'P_e_net_DD_avg',
    'P_e_net_DT_full_avg', 'Q_DD_total', 'Q_DT_full_total', 'E_fusion_total_DD',
    'E_fusion_DT_full', 'E_e_net_DD', 'E_e_net_DT_full', 'E_lost', 'Dollar_Lost',
    'n_T_final', 'sol_success', 't_startup'
]

# Determine input axes based on file name
selected_file_str = str(selected_file.name)
base_inputs = [col for col in df_filtered.columns if col not in output_like and col != TARGET_VARIABLE]
DESIRED_ORDER = [
    'V_plasma','n_tot', 'T_i', 'tau_p_T','tau_p_He3','P_aux', 'P_aux_all_DT', 'tau_ifc', 'tau_ofc', 'TBR_DT', 'TBR_DDn', 'eta_th', 'plant_avail', 'Cost_per_kWh',  'I_target',
]
input_parameters = [p for p in DESIRED_ORDER if p in base_inputs] + [p for p in base_inputs if p not in DESIRED_ORDER]

REMOVE_PARAMS = []
if 'T_seeded' in selected_file_str:
    REMOVE_PARAMS = ['tau_p_He3', 'I_target']
elif 'lump' in selected_file_str:
    REMOVE_PARAMS = ['tau_ifc', 'tau_ofc']
# Additional removal for t_startup target
if TARGET_VARIABLE == 't_startup':
    REMOVE_PARAMS += ['eta_th', 'plant_avail', 'Cost_per_kWh','P_aux', 'P_aux_all_DT']

input_parameters = [p for p in input_parameters if p not in REMOVE_PARAMS]

# Define units for parameters (add as needed)
PARAM_UNITS = {
    'V_plasma': 'm³',
    'n_tot': 'm⁻³',
    'T_i': 'keV',
    'P_aux': 'W',
    'P_aux_all_DT': 'W',
    'tau_p_T': 's',
    'tau_p_He3': 's',
    'I_target': 'A',
    'Cost_per_kWh': '1/kWh',
    'Dollar_Lost': '$',
    't_startup': 's',
    'P_e_net_DD_avg': 'W',
    'P_e_net_DT_full_avg': 'W',
    'P_DT': 'W',
    'P_DDn': 'W',
    'P_DDp': 'W',
    'P_DT_full': 'W',
    'P_fusion_DD_avg': 'W',
    'Q_DD_total': 'J',
    'Q_DT_full_total': 'J',
    'E_fusion_total_DD': 'J',
    'E_fusion_DT_full': 'J',
    'E_e_net_DD': 'J',
    'E_e_net_DT_full': 'J',
    'E_lost': 'J',
    'n_T_final': 'm⁻³',
    'tau_ifc': 's',
    'tau_ofc': 's',
    'sigmav_DT': 'm³/s',
    'sigmav_DD_p': 'm³/s',
    'sigmav_DD_n': 'm³/s',
    'injection_rate_max': 's⁻¹',
    # Add more as needed
}

if TARGET_VARIABLE == 't_startup':
    max_val = df_filtered[TARGET_VARIABLE].max()
    scale = 1
    unit = "s"
    if max_val > 2*365*24*3600:
        scale = 1/(365*24*3600)
        unit = "years"
    elif max_val > 365*24*3600:
        scale = 1/(30*24*3600)
        unit = "months"
    elif max_val > 30*24*3600:
        scale = 1/(24*3600)
        unit = "days"
    elif max_val > 24*3600:
        scale = 1/3600
        unit = "hours"
    # Apply scaling
    df_filtered[TARGET_VARIABLE] = df_filtered[TARGET_VARIABLE] * scale
    PARAM_UNITS[TARGET_VARIABLE] = unit


# =============================================================================
# 5. Color Mapping (Green to Red, Discrete Chunks)
# =============================================================================
def get_discrete_colorscale(n_chunks):
    # Green (low) to Red (high)
    # Vibrant green to red (no dull/dark colors)
    base_colors = [
        '#00FF00',  # bright green
        '#7FFF00',  # chartreuse
        '#FFFF00',  # yellow
        '#FFD700',  # gold
        '#FF9900',  # orange
        '#FF4500',  # orange-red
        '#FF0000',  # bright red
    ]
    if n_chunks > len(base_colors):
        from matplotlib import cm
        cmap = cm.get_cmap('RdYlGn_r', n_chunks)
        color_list = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r,g,b,_ in cmap(np.linspace(0,1,n_chunks))]
    else:
        color_list = base_colors[:n_chunks]
    # Repeat each color for its full interval
    colorscale = []
    for i, color in enumerate(color_list):
        frac0 = i / n_chunks
        frac1 = (i + 1) / n_chunks
        colorscale.append([frac0, color])
        colorscale.append([frac1, color])
    return colorscale

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

    # Prepare colorbar tickvals and ticktext
    tickvals = list(range(N_COLOR_CHUNKS))
    # Show intervals as [low, high)
    ticktext = [f"{chunk_bounds[i]:.2e} – {chunk_bounds[i+1]:.2e}" for i in range(N_COLOR_CHUNKS)]
else:
    color_data = df_filtered[TARGET_VARIABLE]
    colorscale = 'RdYlGn_r'
    cmin, cmax = color_data.min(), color_data.max()
    tickvals = None
    ticktext = None

# =============================================================================
# 6. Parallel Coordinates Plot
# =============================================================================
dimensions = []
for param in input_parameters:
    values = df_filtered[param]
    unique_vals = np.sort(np.unique(values))
    # Split label on two lines if units exist
    if param in PARAM_UNITS:
        label = f"{param}<br>[{PARAM_UNITS[param]}]"
    else:
        label = param
    dim = dict(
        label=label,
        values=values,
        range=[values.min(), values.max()]
    )
    if len(unique_vals) <= 20:
        dim['tickvals'] = unique_vals.tolist()
    dimensions.append(dim)
# Add target variable as last axis
values = df_filtered[TARGET_VARIABLE]
if TARGET_VARIABLE in PARAM_UNITS:
    target_label = f"{TARGET_VARIABLE}<br>[{PARAM_UNITS[TARGET_VARIABLE]}]"
else:
    target_label = TARGET_VARIABLE
dimensions.append(dict(
    label=target_label,
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
            title=f"{TARGET_VARIABLE} [{PARAM_UNITS[TARGET_VARIABLE]}]" if TARGET_VARIABLE in PARAM_UNITS else TARGET_VARIABLE,
            thickness=20,
            len=0.8,
            tickvals=tickvals,
            ticktext=ticktext,
            tickmode='array',
            dtick=1
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
if FILTER is not None:
    filter_str = f"{FILTER:.2e}"
    plot_name = f"paracoords_plot_{TARGET_VARIABLE}_{selected_file.name.strip('dd_startup_')}_FILTER={filter_str}.html"
else:
    plot_name = f"paracoords_plot_{selected_file.name.strip('dd_startup_')}_{TARGET_VARIABLE}.html"
# Save plot as HTML
fig.write_html(outputs_dir / plot_name)
print(f"Plot saved as {outputs_dir / plot_name}")
