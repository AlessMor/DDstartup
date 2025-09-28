
# ======================== IMPORTS ========================
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from utils.postprocess import load_h5_to_dataframe, filter_finite, prepare_dataframes

# ======================== SETUP ========================
# File selection - Choose one or more HDF5 file to analyze
SELECTED_FILES = ["dd_startup_20250927_182615_parametric_T_seeded.h5"]
# Target variable selection - Choose which output metrics to visualize
TARGET_VARIABLES = ['unrealized_gains', 't_startup']
# Optional: set min/max filters for each variable (None means no filter)
FILTERS = {
    't_startup': {'min': 1e6, 'max': None},
    'unrealized_gains': {'min': None, 'max': 2e9},
}

# ======================== FUNCTION DEFINITIONS ========================
def plot_pdf(dataframes_dict, var, label_list, filters):
    """
    Plot PDF for a variable from multiple dataframes.
    """
    vmin = filters.get(var, {}).get('min', None)
    vmax = filters.get(var, {}).get('max', None)
    plt.figure(figsize=(7,5))
    for filename, label in zip(dataframes_dict.keys(), label_list):
        df = dataframes_dict[filename]
        arr = df.get(var)
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
                plt.plot(bin_centers[mask], counts[mask], drawstyle='steps-mid', label=label)
    plt.xlabel(var)
    plt.ylabel('Probability Density')
    plt.title(f'PDF of {var}')
    plt.legend()
    if vmin is not None or vmax is not None:
        plt.xlim(left=1e6, right=vmax)
    plt.xscale('log')
    from matplotlib.ticker import LogLocator, NullFormatter
    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    plt.tight_layout()
    plt.show()

# ======================== LOAD AND FILTER DATA ========================
dataframes_dict = prepare_dataframes(SELECTED_FILES, TARGET_VARIABLES, FILTERS)
label_list = [Path(f).stem for f in SELECTED_FILES]

# ======================== MAIN ROUTINE ========================
variables = ['t_startup', 'unrealized_gains']
for var in variables:
    # Collect filtered dataframes for each file
    dfs = {f: dataframes_dict[f][var] for f in SELECTED_FILES if var in dataframes_dict[f]}
    plot_pdf(dfs, var, label_list, FILTERS)
