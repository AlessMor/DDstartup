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


def plot_pdf(data_list, var, label_list):
    vmin = FILTERS.get(var, {}).get('min', None)
    vmax = FILTERS.get(var, {}).get('max', None)
    plt.figure(figsize=(7,5))
    for data, label in zip(data_list, label_list):
        arr = data.get(var)
        if arr is not None:
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                # Compute histogram on all finite data
                counts, bins = np.histogram(arr, bins=10000, density=True)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                # For display, mask bins outside vmin/vmax
                mask = np.ones_like(bin_centers, dtype=bool)
                if vmin is not None:
                    mask &= bin_centers >= vmin
                if vmax is not None:
                    mask &= bin_centers <= vmax
                plt.plot(bin_centers[mask], counts[mask], drawstyle='steps-mid', label=label)
    plt.xlabel(var)
    plt.ylabel('Probability Density')
    plt.title(f'PDF of {var}')
    plt.legend()
    if vmin is not None or vmax is not None:
        plt.xlim(left=1e6, right=vmax)
    plt.xscale('log')
    # Reduce number of xticks for log scale
    from matplotlib.ticker import LogLocator, NullFormatter
    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    plt.tight_layout()
    plt.show()


# Main
variables = ['t_startup', 'Dollar_Lost']

# Set custom labels for the legend
custom_labels = ['T seeded startup', 'lump startup']
label_list = custom_labels[:len(dataframes)]

for var in variables:
    data_list = [df for df in dataframes]
    plot_pdf(data_list, var, label_list)
