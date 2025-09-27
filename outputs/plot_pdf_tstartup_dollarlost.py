import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# List your HDF5 files here (relative to this script or absolute paths)
h5_files = [
    "dd_startup_20250923_112837_parametric_T_seeded.h5",
    "dd_startup_20250924_142215_parametric_lump.h5",
    # Add more file names as needed
]

outputs_dir = Path(__file__).parent

def load_vars_from_h5(h5_path, variables):
    data = {}
    with h5py.File(h5_path, 'r') as f:
        for var in variables:
            if var in f:
                arr = f[var][:]
            elif 'parameter_fields' in f and var in f['parameter_fields']:
                arr = f['parameter_fields'][var][:]
            else:
                arr = None
            if arr is not None and arr.ndim == 1:
                data[var] = arr
    return data


# Optional: set min/max filters for each variable (None means no filter)
FILTERS = {
    't_startup': {'min': None, 'max': None},  # 3 years in seconds
    'Dollar_Lost': {'min': None, 'max': None},
}

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

data_list = []
# Set custom labels for the legend
custom_labels = ['T seeded startup', 'lump startup']
label_list = []
for i, fname in enumerate(h5_files):
    fpath = outputs_dir / fname
    if fpath.exists():
        data = load_vars_from_h5(fpath, variables)
        data_list.append(data)
        # Use custom label if available, else fallback to filename
        if i < len(custom_labels):
            label_list.append(custom_labels[i])
        else:
            label_list.append(fname)
    else:
        print(f"File not found: {fpath}")

for var in variables:
    plot_pdf(data_list, var, label_list)
