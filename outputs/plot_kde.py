
# ======================== IMPORTS ========================
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# --- Custom modules ---
sys.path.append("..")
from utils.postprocess import load_h5_to_dataframe, filter_finite, get_input_parameters, scale_target
from utils.tools import PARAM_UNITS


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
        if 'cost_of_electricity' in df_filtered.columns:
            df_filtered['cost_of_electricity'] = df_filtered['cost_of_electricity'] * 3.6e6
            PARAM_UNITS['cost_of_electricity'] = '1/kWh'
        file_data[target] = df_filtered
    dataframes[selected_file] = file_data
    
    
# ======================== FUNCTION DEFINITIONS ========================
def prepare_dataframes(selected_files, target_variables, filters):
    """
    Load and filter dataframes for each selected file and target variable.
    Returns a nested dict: {filename: {target: filtered_df}}
    """
    dataframes = {}
    for selected_file in selected_files:
        file_data = {}
        df = load_h5_to_dataframe(selected_file)
        for target in target_variables:
            filter_dict = filters.get(target, {})
            df_filtered = filter_finite(df, target, filter_dict)
            # Convert units if needed
            if 'cost_of_electricity' in df_filtered.columns:
                df_filtered['cost_of_electricity'] = df_filtered['cost_of_electricity'] * 3.6e6
                PARAM_UNITS['cost_of_electricity'] = '1/kWh'
            file_data[target] = df_filtered
        dataframes[selected_file] = file_data
    return dataframes

def kde_quartile_plot(df_filtered, target, input_parameters, target_unit, outputs_dir, file_type, plot_name):
    """
    Create KDE plots for each input parameter, split by quartiles of the target variable.
    """
    n_inputs = len(input_parameters)
    ncols = min(3, n_inputs)
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
    fig.suptitle(f"KDE of Inputs by {target} quartile for {file_type}", fontsize=14)
    plt.savefig(outputs_dir / plot_name, dpi=150)
    plt.close(fig)
    print(f"Plot saved as {outputs_dir / plot_name}")


# ======================== LOAD AND FILTER DATA ========================
# Prepare filtered dataframes for each file and target variable
dataframes = prepare_dataframes(SELECTED_FILES, TARGET_VARIABLES, FILTERS)

# Output directory for saving plots
outputs_dir = Path(__file__).parent

# ======================== MAIN PLOTTING ROUTINE ========================
for selected_file in SELECTED_FILES:
    print(f"\n✅ Selected file: {selected_file}")
    for target in TARGET_VARIABLES:
        df_filtered = dataframes[selected_file][target]
        # Scale target variable if needed
        df_filtered, target_unit = scale_target(df_filtered, target)
        PARAM_UNITS[target] = target_unit
        # Identify input parameters
        input_parameters = get_input_parameters(df_filtered, target, filename=selected_file)
        # Determine file type for plot title
        if "lump" in selected_file:
            file_type = "lump"
        elif "T_seeded" in selected_file or "Tseeded" in selected_file:
            file_type = "Tseeded"
        else:
            file_type = ""
        plot_name = f"kde_quartiles_{Path(selected_file).stem}_{target}.png"
        # Create KDE quartile plot
        kde_quartile_plot(df_filtered, target, input_parameters, target_unit, outputs_dir, file_type, plot_name)



# ======================== CREATE df FROM h5m ========================
outputs_dir = Path(__file__).parent


for selected_file in (SELECTED_FILES if isinstance(SELECTED_FILES, list) else [SELECTED_FILES]):
    print(f"\n✅ Selected file: {selected_file}")
    df = load_h5_to_dataframe(selected_file)
    if 'Cost_per_kWh' in df.columns:
        df['Cost_per_kWh'] = df['Cost_per_kWh'] * 3.6e6
    for target in TARGET_VARIABLES:
        df_filtered = filter_finite(df, target)
        df_filtered, target_unit = scale_target(df_filtered, target)
        input_parameters = get_input_parameters(df_filtered, [
            'linear_index', 'injection_rate_max', 'sigmav_DT', 'sigmav_DD_p', 'sigmav_DD_n',
            'P_DT', 'P_DDn', 'P_DDp', 'P_DT_full', 'P_fusion_DD_avg', 'P_e_net_DD_avg',
            'P_e_net_DT_full_avg', 'Q_DD_total', 'Q_DT_full_total', 'E_fusion_total_DD',
            'E_fusion_DT_full', 'E_e_net_DD', 'E_e_net_DT_full', 'E_lost', 'Dollar_Lost',
            'n_T_final', 'sol_success', 't_startup'
        ], target, filename=selected_file)
        n_inputs = len(input_parameters)
        ncols = min(3, n_inputs)
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
        if "lump" in selected_file:
            file_type = "lump"
        elif "T_seeded" in selected_file or "Tseeded" in selected_file:
            file_type = "Tseeded"
        else:
            file_type = ""
        fig.suptitle(f"KDE of Inputs by {target} quartile for {file_type}", fontsize=14)
        plot_name = f"kde_quartiles_{Path(selected_file).stem}_{target}.png"
        plt.savefig(outputs_dir / plot_name, dpi=150)
        plt.close(fig)
        print(f"Plot saved as {outputs_dir / plot_name}")