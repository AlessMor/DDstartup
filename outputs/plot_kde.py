
# ======================== IMPORTS ========================
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# --- Custom modules ---
sys.path.append("..")
from utils.postprocess import get_input_parameters, scale_target, prepare_dataframes, get_discrete_colorscale
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
    
# ======================== FUNCTION DEFINITIONS ========================
def save_quartile_extremes_to_csv(df_filtered, target, input_parameters, bin_labels, outputs_dir, plot_name):
    """
    Save the lowest and middle value for each quartile of the target variable to a CSV,
    including the input parameter combination for each row.
    """
    quartile_csv_rows = []
    for bin_label in bin_labels:
        bin_df = df_filtered[df_filtered[f"{target}_bin"] == bin_label]
        if bin_df.empty:
            continue
        # Find lowest and middle value of target variable in this quartile
        sorted_bin = bin_df.sort_values(by=target)
        lowest_row = sorted_bin.iloc[0]
        middle_row = sorted_bin.iloc[len(sorted_bin)//2]
        for row, which in zip([lowest_row, middle_row], ["lowest", "middle"]):
            csv_row = {"quartile": bin_label, "which": which, target: row[target]}
            for param in input_parameters:
                csv_row[param] = row[param]
            quartile_csv_rows.append(csv_row)
    csv_df = pd.DataFrame(quartile_csv_rows)
    csv_name = f"quartile_{target}_values_{Path(plot_name).stem}.csv"
    csv_df.to_csv(outputs_dir / csv_name, index=False)
    print(f"Quartile values saved as {outputs_dir / csv_name}")

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
    # Save quartile extremes to CSV using the new function (after bin_labels is defined)
    save_quartile_extremes_to_csv(df_filtered, target, input_parameters, bin_labels, outputs_dir, plot_name)
    colorscale = get_discrete_colorscale(4)
    # Extract just the color hex codes in order
    quartile_colors = [colorscale[i*2][1] for i in range(4)]
    color_map = {label: quartile_colors[i] for i, label in enumerate(bin_labels)}

    for i, param in enumerate(input_parameters):
        ax = axes[i]
        for bin_label in df_filtered[f"{target}_bin"].cat.categories:
            data = df_filtered.loc[df_filtered[f"{target}_bin"] == bin_label, param].dropna()
            if len(data) == 0:
                continue
            color = color_map[bin_label]
            if np.var(data) == 0:
                ax.axhline(1, color=color, linestyle='--', label=str(bin_label))
            else:
                sns.kdeplot(data, fill=True, alpha=0.3, ax=ax, label=str(bin_label), color=color)
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
        kde_quartile_plot(df_filtered, target, input_parameters, target_unit, outputs_dir, file_type, plot_name)