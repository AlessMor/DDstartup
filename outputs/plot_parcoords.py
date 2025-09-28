import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys
# --- Custom modules ---
sys.path.append("..")
from utils.postprocess import get_input_parameters, scale_target, get_discrete_colorscale, prepare_dataframes
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

# Color and styling options
N_COLOR_CHUNKS = 6           # Number of discrete color levels (None sets to gradient)



# ======================== LOAD AND FILTER DATA ========================
# Prepare filtered dataframes for each file and target variable
dataframes = prepare_dataframes(SELECTED_FILES, TARGET_VARIABLES, FILTERS)


# ======================== START PARCOORDS CREATION ========================

for selected_file in SELECTED_FILES:
    for TARGET_VARIABLE in TARGET_VARIABLES:
        df_filtered = dataframes[selected_file][TARGET_VARIABLE]

        # Scale target variable if needed
        df_filtered, target_unit = scale_target(df_filtered, TARGET_VARIABLE)
        PARAM_UNITS[TARGET_VARIABLE] = target_unit

        # Randomly sample up to 1e6 rows for plotting
        max_plot_rows = int(1e6)
        if len(df_filtered) > max_plot_rows:
            df_filtered = df_filtered.sample(n=max_plot_rows, random_state=42)
            print(f"Sampled {max_plot_rows} rows for plotting (out of {len(df_filtered)} finite rows)")
        else:
            print(f"Plotting all {len(df_filtered)} finite rows")

        # Identify input parameters
        selected_file_str = str(selected_file)
        input_parameters = get_input_parameters(df_filtered, TARGET_VARIABLE, filename=selected_file_str)
        # Color mapping
        if N_COLOR_CHUNKS:
            target_values = df_filtered[TARGET_VARIABLE]
            quantiles = np.linspace(0, 1, N_COLOR_CHUNKS+1)
            chunk_bounds = target_values.quantile(quantiles).values
            color_indices = np.zeros(len(target_values), dtype=int)
            for i in range(N_COLOR_CHUNKS):
                if i == N_COLOR_CHUNKS-1:
                    mask = target_values > chunk_bounds[i]
                else:
                    mask = (target_values > chunk_bounds[i]) & (target_values <= chunk_bounds[i+1])
                color_indices[mask] = i
            color_data = color_indices
            colorscale = get_discrete_colorscale(N_COLOR_CHUNKS)
            cmin, cmax = 0, N_COLOR_CHUNKS-1
            tickvals = list(range(N_COLOR_CHUNKS))
            ticktext = [f"{chunk_bounds[i]:.2e} – {chunk_bounds[i+1]:.2e}" for i in range(N_COLOR_CHUNKS)]
        else:
            color_data = df_filtered[TARGET_VARIABLE]
            colorscale = 'RdYlGn_r'
            cmin, cmax = color_data.min(), color_data.max()
            tickvals = None
            ticktext = None

        # Parallel coordinates plot
        dimensions = []
        for param in input_parameters:
            values = df_filtered[param]
            # Skip if values are arrays (vector fields)
            if hasattr(values.iloc[0], "__len__") and not isinstance(values.iloc[0], str):
                print(f"skipping {param}")
                continue  # skip vector fields
            unique_vals = np.sort(np.unique(values))
            if param == "cost_of_electricity":
                label = f"C_kWh<br>[{PARAM_UNITS[param]}]"
                dim = dict(
                    label=label,
                    values=values,
                    range=[values.min(), values.max()],
                    tickformat=".2g"  # Only C_kWh axis has 2 significant digits
                )
            else:
                label = f"{param}<br>[{PARAM_UNITS[param]}]" if param in PARAM_UNITS else param
                dim = dict(
                    label=label,
                    values=values,
                    range=[values.min(), values.max()]
                )
            if len(unique_vals) <= 20:
                dim['tickvals'] = unique_vals.tolist()
            dimensions.append(dim)
        values = df_filtered[TARGET_VARIABLE]
        target_label = f"{TARGET_VARIABLE}<br>[{PARAM_UNITS[TARGET_VARIABLE]}]" if TARGET_VARIABLE in PARAM_UNITS else TARGET_VARIABLE
        dimensions.append(dict(
            label=target_label,
            values=values,
            range=[values.min(), values.max()]
        ))

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=color_data,
                colorscale=colorscale,
                showscale=True,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title=target_label,
                    thickness=20,
                    len=0.8,
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickmode='array',
                    dtick=1
                )
            ),
            dimensions=dimensions
        ))

        fig.update_layout(
            title=f"DD Startup Parallel Coordinates Plot ({selected_file})",
            font=dict(size=12),
            width=1400,
            height=700,
            margin=dict(l=100, r=120, t=120, b=100),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        plot_name = f"paracoords_plot_{TARGET_VARIABLE}_{Path(selected_file).stem}.html"
        fig.write_html(plot_name)
        print(f"Plot saved as {plot_name}")