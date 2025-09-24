PARAM_UNITS = {
    'V_plasma': 'm³',
    'n_tot': 'm⁻³',
    'T_i': 'keV',
    'P_aux': 'W',
    'P_aux_all_DT': 'W',
    'P_lost_rad': 'W',
    'P_lost_rad_all_DT': 'W',
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
    'TBR_DT': '-',
    'TBR_DDn': '-',
    # Add more as needed
}
# =============================================================================
# DD Startup Analysis - Kernel Density Estimation Plots for Parametric Analysis
# =============================================================================

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# CONFIGURATION OPTIONS
# =============================================================================
SELECTED_FILE = ["dd_startup_20250924_142215_parametric_lump.h5"] # Can be a string or list of filenames, or None for auto-selection
TARGET_VARIABLES = ['t_startup']
outputs_dir = Path(__file__).parent

# =============================================================================
# Helper Functions
# =============================================================================
def get_h5_files():
    """Return list of selected HDF5 files."""
    if SELECTED_FILE is None:
        files = list(outputs_dir.glob("*.h5"))
        if not files:
            raise FileNotFoundError("No HDF5 files found in outputs directory")
        return [max(files, key=lambda f: f.stat().st_mtime)]
    if isinstance(SELECTED_FILE, str):
        SELECTED_FILE_LIST = [SELECTED_FILE]
    else:
        SELECTED_FILE_LIST = SELECTED_FILE
    files = []
    for fname in SELECTED_FILE_LIST:
        fpath = outputs_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"File '{fname}' not found in outputs directory")
        files.append(fpath)
    return files

def load_data(h5_path):
    """Load 1D datasets and parameter_fields from HDF5 file."""
    data = {}
    expected_length = None
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                arr = f[key][:]
                if arr.ndim == 1:
                    if expected_length is None:
                        expected_length = len(arr)
                    if len(arr) == expected_length:
                        data[key] = arr
        if 'parameter_fields' in f:
            param_group = f['parameter_fields']
            for subkey in param_group.keys():
                arr = param_group[subkey][:]
                if arr.ndim == 1 and len(arr) == expected_length:
                    data[subkey] = arr
    return pd.DataFrame(data)

def get_input_parameters(df, filename, target):
    """Determine input parameters for the given file and target variable."""
    output_like = [
        'linear_index', 'injection_rate_max', 'sigmav_DT', 'sigmav_DD_p', 'sigmav_DD_n',
        'P_DT', 'P_DDn', 'P_DDp', 'P_DT_full', 'P_fusion_DD_avg', 'P_e_net_DD_avg',
        'P_e_net_DT_full_avg', 'Q_DD_total', 'Q_DT_full_total', 'E_fusion_total_DD',
        'E_fusion_DT_full', 'E_e_net_DD', 'E_e_net_DT_full', 'E_lost', 'Dollar_Lost',
        'n_T_final', 'sol_success', 't_startup'
    ]
    DESIRED_ORDER = [
        'V_plasma','n_tot', 'T_i', 'tau_p_T','tau_p_He3','P_aux', 'P_aux_all_DT', 'P_lost_rad', 'P_lost_rad_all_DT', 'tau_ifc', 'tau_ofc', 'TBR_DT', 'TBR_DDn', 'eta_th', 'plant_avail', 'Cost_per_kWh',  'I_target',
    ]
    base_inputs = [col for col in df.columns if col not in output_like and col != target]
    input_parameters = [p for p in DESIRED_ORDER if p in base_inputs] + [p for p in base_inputs if p not in DESIRED_ORDER]
    REMOVE_PARAMS = []
    if 'T_seeded' in filename:
        REMOVE_PARAMS = ['tau_p_He3', 'I_target']
    elif 'lump' in filename:
        REMOVE_PARAMS = ['tau_ifc', 'tau_ofc']
    if target == 't_startup':
        REMOVE_PARAMS += ['eta_th', 'plant_avail', 'Cost_per_kWh','P_aux', 'P_aux_all_DT', 'P_lost_rad', 'P_lost_rad_all_DT']
    input_parameters = [p for p in input_parameters if p not in REMOVE_PARAMS]
    return input_parameters

def scale_target(df, target):
    """Scale t_startup to appropriate units and return unit label for legend."""
    if target == 'Dollar_Lost':
        # Use Mega dollars for legend
        df[target] = df[target] / 1e6
        return df, 'M$'
    if target != 't_startup':
        return df, 's'
    max_val = df[target].max()
    scale = 1
    unit = "s"
    if max_val > 30*24*3600:
        scale = 1/(24*3600)
        unit = "days"
    elif max_val > 24*3600:
        scale = 1/3600
        unit = "hours"
    df[target] = df[target] * scale
    return df, unit

# =============================================================================
# Main Plotting Routine
# =============================================================================
for h5_path in get_h5_files():
    print(f"\n✅ Selected file: {h5_path.name}")
    df = load_data(h5_path)
    if 'Cost_per_kWh' in df.columns:
        df['Cost_per_kWh'] = df['Cost_per_kWh'] * 3.6e6
    for target in TARGET_VARIABLES:
        finite_mask = np.isfinite(df[target])
        df_filtered = df[finite_mask].copy()
        df_filtered, target_unit = scale_target(df_filtered, target)
        input_parameters = get_input_parameters(df_filtered, h5_path.name, target)
        n_inputs = len(input_parameters)
        ncols = min(4, n_inputs)
        nrows = int(np.ceil(n_inputs / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows), sharey=False)
        axes = axes.flatten()
        # Quartile binning
        bin_edges = df_filtered[target].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
        bin_labels = [
            f"Q1: {bin_edges[0]:.2e}–{bin_edges[1]:.2e}",
            f"Q2: {bin_edges[1]:.2e}–{bin_edges[2]:.2e}",
            f"Q3: {bin_edges[2]:.2e}–{bin_edges[3]:.2e}",
            f"Q4: {bin_edges[3]:.2e}–{bin_edges[4]:.2e}"
        ]
        df_filtered[f"{target}_bin"] = pd.qcut(df_filtered[target], q=4, labels=bin_labels)
        for i, param in enumerate(input_parameters):
            ax = axes[i]
            for bin_label in df_filtered[f"{target}_bin"].cat.categories:
                data = df_filtered.loc[df_filtered[f"{target}_bin"] == bin_label, param].dropna()
                if len(data) == 0:
                    continue
                if np.var(data) == 0:
                    ax.axhline(1, color='gray', linestyle='--', label=f'Flat ({bin_label})')
                else:
                    sns.kdeplot(data, fill=True, alpha=0.5, ax=ax, label=str(bin_label))
            # Add unit to title if available
            if param in PARAM_UNITS:
                param_label = f"{param} [{PARAM_UNITS[param]}]"
            else:
                param_label = param
            ax.set_title(param_label, fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("Density")
            ax.tick_params(labelsize=8)
        # Hide unused subplots
        for j in range(n_inputs, len(axes)):
            axes[j].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        legend_title = f"{target} quartile [{target_unit}]"
        fig.legend(handles, labels,
                   title=legend_title,
                   loc='center right',
                   fontsize=9, title_fontsize=10)
        plt.subplots_adjust(
            left=0.05,
            right=0.85,
            top=0.95,
            bottom=0.05,
            wspace=0.3,
            hspace=0.4
        )
        if "lump" in h5_path.name:
            file_type = "lump"
        elif "T_seeded" in h5_path.name or "Tseeded" in h5_path.name:
            file_type = "Tseeded"
        else:
            file_type = ""
        fig.suptitle(f"KDE of Inputs by {target} quartile for {file_type}", fontsize=14)
        plot_name = f"kde_quartiles_{h5_path.stem}_{target}.png"
        plt.savefig(outputs_dir / plot_name, dpi=150)
        plt.close(fig)
        print(f"Plot saved as {outputs_dir / plot_name}")
