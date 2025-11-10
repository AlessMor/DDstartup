"""
Parallel Coordinates Plot Functions

This module contains functions for generating interactive parallel coordinates plots
using Plotly.
"""

import numpy as np
import plotly.graph_objects as go

from ddstartup.postprocessing.postprocess_functions import get_discrete_colorscale


def generate_parcoords_plot(df_filtered, target, input_parameters, target_unit, file_type, output_path, registry=None):
    """
    Generate a parallel coordinates plot using Plotly.
    
    Args:
        df_filtered: Filtered DataFrame with data
        target: Target variable name
        input_parameters: List of input parameter names
        target_unit: Unit string for target variable
        file_type: Type of file (for plot title)
        output_path: Path to save HTML plot file
        registry: ParameterRegistry instance (optional, will create if not provided)
    """
    # Get registry if not provided
    if registry is None:
        from ddstartup.utils.parameter_registry import get_registry
        registry = get_registry()
    
    # Sample if too many rows
    max_plot_rows = int(1e6)
    if len(df_filtered) > max_plot_rows:
        df_filtered = df_filtered.sample(n=max_plot_rows, random_state=42)
    
    # Color mapping (6 chunks)
    N_COLOR_CHUNKS = 6
    target_values = df_filtered[target]
    quantiles = np.linspace(0, 1, N_COLOR_CHUNKS+1)
    chunk_bounds = target_values.quantile(quantiles).values
    color_indices = np.zeros(len(target_values), dtype=int)
    for i in range(N_COLOR_CHUNKS):
        if i == N_COLOR_CHUNKS-1:
            mask = target_values >= chunk_bounds[i]
        else:
            mask = (target_values >= chunk_bounds[i]) & (target_values < chunk_bounds[i+1])
        color_indices[mask] = i
    
    colorscale = get_discrete_colorscale(N_COLOR_CHUNKS)
    
    # Build dimensions
    dimensions = []
    for param in input_parameters:
        values = df_filtered[param]
        if hasattr(values.iloc[0], "__len__") and not isinstance(values.iloc[0], str):
            continue  # skip vector fields
        
        unique_vals = np.sort(np.unique(values))
        
        # Use symbol instead of parameter name
        param_symbol = registry.get_symbol(param)
        unit = registry.get_unit(param)
        label = f"{param_symbol}<br>[{unit}]" if unit else param_symbol
        
        dim = dict(label=label, values=values, range=[values.min(), values.max()])
        if len(unique_vals) <= 20:
            dim['tickvals'] = unique_vals.tolist()
        dimensions.append(dim)
    
    # Add target dimension
    values = df_filtered[target]
    target_symbol = registry.get_symbol(target)
    target_label = f"{target_symbol}<br>[{target_unit}]"
    dimensions.append(dict(label=target_label, values=values, range=[values.min(), values.max()]))
    
    # Create figure
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=color_indices,
            colorscale=colorscale,
            showscale=True,
            cmin=0,
            cmax=N_COLOR_CHUNKS-1,
            colorbar=dict(
                title=target_label,
                thickness=20,
                len=0.8,
                tickvals=list(range(N_COLOR_CHUNKS)),
                ticktext=[f"{chunk_bounds[i]:.2e}–{chunk_bounds[i+1]:.2e}" 
                         for i in range(N_COLOR_CHUNKS)],
                tickmode='array'
            )
        ),
        dimensions=dimensions
    ))
    
    fig.update_layout(
        title=f"Parallel Coordinates Plot - {file_type} ({target_symbol})",
        font=dict(size=12),
        width=1400,
        height=700,
        margin=dict(l=100, r=120, t=120, b=100),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    fig.write_html(str(output_path))
