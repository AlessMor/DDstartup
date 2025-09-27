import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys
# --- Custom modules ---
sys.path.append("..")
from utils.postprocess import load_h5_to_dataframe, filter_finite, get_input_parameters, scale_target


# ======================== SETUP ========================
# File selection - Choose one or more HDF5 file to analyze
SELECTED_FILES = ["dd_startup_20250927_182615_parametric_T_seeded.h5"]  # Set to None for automatic selection, or specify filename

# Target variable selection - Choose which output metrics to visualize
TARGET_VARIABLES = ['unrealized_gains', 't_startup']  

# Optional: set min/max filters for each variable (None means no filter)
FILTERS = {
    't_startup': {'min': 1e6, 'max': None},  # 3 years in seconds
    'unrealized_gains': {'min': None, 'max': 2e9},
}


# ======================== CREATE df FROM h5m ========================
dataframes = {}
for selected_file in SELECTED_FILES:
    file_data = {}
    df = load_h5_to_dataframe(selected_file)
    for target in TARGET_VARIABLES:
        filter_dict = FILTERS.get(target, {})
        df_filtered = filter_finite(df, target, filter_dict)
        file_data[target] = df_filtered
    dataframes[selected_file] = file_data




# Define which columns are outputs and which are inputs
output_like = [
    'linear_index', 'injection_rate_max', 'sigmav_DT', 'sigmav_DD_p', 'sigmav_DD_n',
    'P_DT', 'P_DDn', 'P_DDp', 'P_DT_full', 'P_fusion_DD_avg', 'P_e_net_DD_avg',
    'P_e_net_DT_full_avg', 'Q_DD_total', 'Q_DT_full_total', 'E_fusion_total_DD',
    'E_fusion_DT_full', 'E_e_net_DD', 'E_e_net_DT_full', 'E_lost', 'Dollar_Lost',
    'n_T_final', 'sol_success', 't_startup'
]
input_parameters = [col for col in df.columns if col not in output_like]

# For each output, find the best row (min for cost/time, max for power/efficiency)
def find_best(df, output):
    finite_mask = np.isfinite(df[output])
    df_filtered = df[finite_mask]
    if df_filtered.empty:
        return None, None, None
    if output in ['Dollar_Lost', 't_startup', 'E_lost']:
        idx = df_filtered[output].idxmin()
        best_val = df_filtered[output].min()
        direction = "minimized"
    else:
        idx = df_filtered[output].idxmax()
        best_val = df_filtered[output].max()
        direction = "maximized"
    best_row = df_filtered.loc[idx]
    return best_row, best_val, direction

# Plot t_startup vs Dollar_Lost for best 10% Dollar_Lost

# 3D subplots for all pairs of input parameters
from mpl_toolkits.mplot3d import Axes3D
def plot_3d_subplots(df, outputs_dir, max_dollar_lost, input_parameters, z_param='t_startup', color_param='Dollar_Lost'):
    # Print stats for finite values before filtering
    finite_mask = np.isfinite(df[color_param]) & np.isfinite(df[z_param])
    finite_df = df[finite_mask]
    print(f"Finite {color_param}: count={finite_df[color_param].count()}, min={finite_df[color_param].min():.2e}, max={finite_df[color_param].max():.2e}")
    print(f"Finite {z_param}: count={finite_df[z_param].count()}, min={finite_df[z_param].min():.2e}, max={finite_df[z_param].max():.2e}")

    # Print rows with Dollar_Lost < max_dollar_lost before further filtering
    prefilter = df[(df[color_param] < max_dollar_lost)]
    print(f"Rows with {color_param} < {max_dollar_lost:.2e}: {len(prefilter)}")
    print(prefilter[[color_param, z_param] + input_parameters].head(10))  # Print first 10 rows for inspection

    # Remove tau_p_He3 from input_parameters if any inf in prefilter
    if 'tau_p_He3' in input_parameters:
        if np.isinf(prefilter['tau_p_He3']).any():
            print("Removing tau_p_He3 from input_parameters due to inf values.")
            input_parameters = [p for p in input_parameters if p != 'tau_p_He3']

    # Filter finite values and by max_dollar_lost
    mask = np.isfinite(df[color_param]) & np.isfinite(df[z_param]) & (df[color_param] <= max_dollar_lost)
    for param in input_parameters:
        mask = mask & np.isfinite(df[param])
    df_filtered = df[mask].copy()
    print(f"Rows after filtering by MAX_DOLLAR_LOST: {len(df_filtered)}")
    if df_filtered.empty:
        print(f"No finite values for {color_param} and {z_param} under the specified threshold, skipping 3D plots.")
        return
    # User-specified parameter pairs for x-y axes
    user_pairs = [
        ('V_plasma', 'T_i'),
        ('tau_ifc', 'tau_ofc'),
        ('eta_th', 'plant_avail'),
        ('tau_p_T', 'tau_p_He3'),
    ]
    # Only keep pairs where both parameters are present in df_filtered
    pairs = [(x, y) for x, y in user_pairs if x in df_filtered.columns and y in df_filtered.columns]
    n_plots = len(pairs)
    ncols = 3
    nrows = int(np.ceil(n_plots / ncols))
    fig = plt.figure(figsize=(7*ncols, 6*nrows))
    for idx, (x_param, y_param) in enumerate(pairs):
        ax = fig.add_subplot(nrows, ncols, idx+1, projection='3d')
        x = df_filtered[x_param]
        y = df_filtered[y_param]
        z = df_filtered[z_param]
        c = df_filtered[color_param]
        p = ax.scatter(x, y, z, c=c, cmap='viridis', alpha=0.7)
        ax.set_xlabel(f'{x_param}')
        ax.set_ylabel(f'{y_param}')
        ax.set_zlabel(f'{z_param}')
        ax.set_title(f'{x_param} vs {y_param} vs {z_param}')
        fig.colorbar(p, ax=ax, shrink=0.5, aspect=10, pad=0.1, label=f'{color_param}')
    plt.tight_layout()
    plot_path = outputs_dir / f'3d_subplots_selected_{z_param}_color_{color_param}_max{int(max_dollar_lost):d}.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"3D subplots saved as {plot_path}")

print(f"Analyzing file: {h5_path.name}\n")
for output in output_like:
    if output in df.columns:
        best_row, best_val, direction = find_best(df, output)
        if best_row is None:
            print(f"No finite values for {output}, skipping.")
            continue
        print(f"Best combination for {output} ({direction}, value={best_val:.4e}):")
        for param in input_parameters:
            print(f"  {param}: {best_row[param]}")
        print("-" * 40)

# Generate the requested plots
if 'Dollar_Lost' in df.columns and 't_startup' in df.columns:
    #plot_t_startup_vs_dollar_lost(df, outputs_dir, MAX_DOLLAR_LOST)
    plot_3d_subplots(df, outputs_dir, MAX_DOLLAR_LOST, input_parameters, z_param='t_startup', color_param='Dollar_Lost')