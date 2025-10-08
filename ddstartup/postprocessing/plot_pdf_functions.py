"""
PDF (Probability Density Function) Plot Functions

This module contains functions for generating PDF plots comparing distributions
across different datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


def generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path):
    """
    Generate a probability density function plot.
    
    Args:
        dataframes_dict: Dictionary mapping filenames to data dictionaries
        var: Variable name to plot
        label_list: List of labels for each dataset
        filters: Dictionary of filters (min/max) for variables
        output_path: Path to save plot file
    """
    vmin = filters.get(var, {}).get('min', None)
    vmax = filters.get(var, {}).get('max', None)
    
    plt.figure(figsize=(7, 5))
    has_data = False
    for filename, label in zip(dataframes_dict.keys(), label_list):
        df_data = dataframes_dict[filename]
        arr = df_data.get(var)
        if arr is not None:
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                counts, bins = np.histogram(arr, bins=10000, density=True)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                mask = np.ones_like(bin_centers, dtype=bool)
                if vmin is not None:
                    mask &= bin_centers >= vmin
                if vmax is not None:
                    mask &= bin_centers <= vmax
                plt.plot(bin_centers[mask], counts[mask], drawstyle='steps-mid', label=f"{label}")
                has_data = True
    
    plt.xlabel(var)
    plt.ylabel('Probability Density')
    plt.title(f'PDF of {var}')
    if has_data:
        plt.legend()
    if vmin is not None or vmax is not None:
        plt.xlim(left=vmin, right=vmax)
    plt.xscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
