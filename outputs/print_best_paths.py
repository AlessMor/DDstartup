
# ======================== IMPORTS ========================
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from utils.postprocess import load_h5_to_dataframe, prepare_dataframes
from utils.tools import inputs_names, outputs_names

# ======================== SETUP ========================
# File selection - Choose one or more HDF5 file to analyze
SELECTED_FILES = ["dd_startup_20250927_182615_parametric_T_seeded.h5"]
# Target variable selection - Choose which output metrics to visualize
TARGET_VARIABLES = ['unrealized_gains', 't_startup']
# Optional: set min/max filters for each variable (None means no filter)
FILTERS = {
    't_startup': {'min': None, 'max': None},
    'unrealized_gains': {'min': None, 'max': None},
}

# ======================== FUNCTION DEFINITIONS ========================
def find_best(df, output):
    """
    For each output, find the best row (min for cost/time, max for power/efficiency)
    """
    # Only process numeric columns
    if output not in df.columns or not pd.api.types.is_numeric_dtype(df[output]):
        return None, None, None
    finite_mask = np.isfinite(df[output].values)
    df_filtered = df[finite_mask]
    if df_filtered.empty:
        return None, None, None
    if output in ['unrealized_gains', 't_startup', 'E_lost']:
        idx = df_filtered[output].idxmin()
        best_val = df_filtered[output].min()
        direction = "minimized"
    else:
        idx = df_filtered[output].idxmax()
        best_val = df_filtered[output].max()
        direction = "maximized"
    best_row = df_filtered.loc[idx]
    return best_row, best_val, direction

# ======================== LOAD AND FILTER DATA ========================
dataframes = prepare_dataframes(SELECTED_FILES, TARGET_VARIABLES, FILTERS)

# ======================== MAIN ROUTINE ========================
for selected_file in SELECTED_FILES:
    print(f"\n✅ Selected file: {selected_file}")
    df = load_h5_to_dataframe(selected_file)
    # Use outputs_names and inputs_names from utils.tools
    input_parameters = [col for col in df.columns if col in inputs_names]
    best_results = {}
    for output in outputs_names:
        if output in df.columns:
            best_row, best_val, direction = find_best(df, output)
            if best_row is None:
                continue
            idx = best_row.name
            if idx not in best_results:
                best_results[idx] = {
                    'params': [],
                    'values': [],
                    'directions': [],
                    'row': best_row
                }
            best_results[idx]['params'].append(output)
            best_results[idx]['values'].append(best_val)
            best_results[idx]['directions'].append(direction)

    for idx, result in best_results.items():
        param_strs = [
            f"{p} ({d}, value={v:.4e})"
            for p, v, d in zip(result['params'], result['values'], result['directions'])
        ]
        print(f"Best combination at index {idx} for: {', '.join(param_strs)}")
        for param in input_parameters:
            print(f"  {param}: {result['row'][param]}")
        print("-" * 40)