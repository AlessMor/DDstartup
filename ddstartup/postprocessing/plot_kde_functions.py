"""
KDE (Kernel Density Estimation) Plot Functions

This module contains functions for generating KDE plots split by quartiles
of target variables.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from ddstartup.postprocessing.postprocess_functions import get_discrete_colorscale
from ddstartup.utils.tools import PARAM_UNITS
from ddstartup.utils.parameter_symbols import get_param_label, get_param_symbol


def save_quartile_extremes_to_csv(df_filtered, target, input_parameters, bin_labels, outputs_dir, plot_name):
    """
    Save the lowest and middle value for each quartile of the target variable to a CSV,
    including the input parameter combination for each row.
    
    Args:
        df_filtered: Filtered DataFrame with data
        target: Target variable name
        input_parameters: List of input parameter names
        bin_labels: List of quartile bin labels
        outputs_dir: Directory to save CSV file
        plot_name: Name of the plot (used for CSV filename)
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
            # Force t_startup column to be present for unrealized_profits
            if "t_startup" in df_filtered.columns:
                csv_row["t_startup"] = row.get("t_startup", np.nan)
            quartile_csv_rows.append(csv_row)
    
    csv_df = pd.DataFrame(quartile_csv_rows)
    csv_name = f"quartile_{target}_values_{Path(plot_name).stem}.csv"
    csv_df.to_csv(outputs_dir / csv_name, index=False)
    print(f"   Quartile values saved: {csv_name}")


def kde_quartile_plot(df_filtered, target, input_parameters, target_unit, outputs_dir, file_type, plot_name):
    """
    Create KDE plots for each input parameter, split by quartiles of the target variable.
    
    Args:
        df_filtered: Filtered DataFrame with data
        target: Target variable name
        input_parameters: List of input parameter names
        target_unit: Unit string for target variable
        outputs_dir: Directory to save plot
        file_type: Type of file (for plot title)
        plot_name: Name for saved plot file
    """
    n_inputs = len(input_parameters)
    if n_inputs == 0:
        print(f"   No input parameters found for target '{target}'. Skipping KDE plot.")
        return
    
    # Set up subplot grid
    ncols = min(3, n_inputs)
    nrows = int(np.ceil(n_inputs / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows), sharey=False)
    axes = axes.flatten() if n_inputs > 1 else [axes]
    
    # Quartile binning
    bin_edges = df_filtered[target].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    bin_labels = [
        f"Q1: {bin_edges[0]:.2e}–{bin_edges[1]:.2e}",
        f"Q2: {bin_edges[1]:.2e}–{bin_edges[2]:.2e}",
        f"Q3: {bin_edges[2]:.2e}–{bin_edges[3]:.2e}",
        f"Q4: {bin_edges[3]:.2e}–{bin_edges[4]:.2e}"
    ]
    df_filtered[f"{target}_bin"] = pd.qcut(df_filtered[target], q=4, labels=bin_labels)
    
    # Save quartile extremes to CSV
    save_quartile_extremes_to_csv(df_filtered, target, input_parameters, bin_labels, outputs_dir, plot_name)
    
    # Get colors for quartiles
    colorscale = get_discrete_colorscale(4)
    quartile_colors = [colorscale[i*2][1] for i in range(4)]
    color_map = {label: quartile_colors[i] for i, label in enumerate(bin_labels)}
    
    # Plot KDE for each input parameter
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
        
        # Add unit to title with symbol
        param_label = get_param_label(param, PARAM_UNITS.get(param))
        ax.set_title(param_label, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Density")
        ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for j in range(n_inputs, len(axes)):
        axes[j].set_visible(False)
    
    # Add legend with symbol
    handles, labels = axes[0].get_legend_handles_labels()
    target_label = get_param_label(target, target_unit)
    legend_title = f"{target_label} quartile"
    fig.legend(handles, labels,
               title=legend_title,
               loc='lower right',
               fontsize=12, title_fontsize=14)
    
    # Adjust layout
    target_symbol = get_param_symbol(target)
    fig.suptitle(f"KDE of Inputs by {target_symbol} quartile for {file_type}", fontsize=14)
    plt.subplots_adjust(
        left=0.05,
        right=0.85,
        top=0.88,
        bottom=0.05,
        wspace=0.3,
        hspace=0.4
    )
    
    plt.savefig(outputs_dir / plot_name, dpi=150)
    plt.close(fig)
