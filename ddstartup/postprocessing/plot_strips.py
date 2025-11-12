"""
Strip Plot Functions

This module contains functions for generating strip plots that compare multiple metrics
across parameter combinations, with trend lines and uncertainty bands.

USAGE:
------
Strip plots visualize how multiple metrics vary across simulations, sorted by a chosen
metric. They show:
  1. Raw data (thin, semi-transparent lines)
  2. LOWESS trend lines (thick colored lines)
  3. 80% uncertainty bands (shaded regions)
  4. Optimal point marker (gold star, optional)

Strip plots are ideal for:
  - Comparing 2-3 metrics simultaneously (e.g., cost, time, power)
  - Identifying trade-offs between objectives
  - Finding optimal operating points that balance multiple goals
  - Visualizing trends and variability in large parameter sweeps

CONFIGURATION (postprocess_config.yaml):
---------------------------------------
plots:
  strip: true
  
strip_settings:
  # Metrics to plot (1-3, can include computed metrics like "P_DT_eq - P_aux")
  y_metrics: 
    - unrealized_profits
    - t_startup
    - P_DT_eq
  
  # Metric to sort x-axis by
  sort_by: t_startup
  
  # Unit conversions for readability
  unit_conversions:
    t_startup:
      factor: 1.1574074074074073e-05  # seconds → days
      unit: 'days'
    unrealized_profits:
      factor: 1.0e-6  # dollars → M$
      unit: 'M$'
    P_DT_eq:
      factor: 1.0e-6  # watts → MW
      unit: 'MW'
  
  # Mark optimal point (minimizes all metrics in normalized space)
  optimal_point: true
  
  # LOWESS smoothing parameter (0.05-0.25)
  frac: 0.12
  
  # Figure size [width, height]
  figsize: [14, 6]

EXAMPLE - Net Power Strip Plot:
-------------------------------
To plot net power (P_DT_eq - P_aux) instead of gross power:

strip_settings:
  y_metrics: 
    - unrealized_profits
    - t_startup
    - "P_DT_eq - P_aux"  # Computed on-the-fly
  
  unit_conversions:
    "P_DT_eq - P_aux":
      factor: 1.0e-6
      unit: 'MW'

PROGRAMMATIC USAGE:
------------------
from ddstartup.postprocessing.plot_strips import generate_strip_plot
from ddstartup.utils.parameter_registry import get_registry

registry = get_registry()

generate_strip_plot(
    h5_file='outputs/results.h5',
    y_metrics=['unrealized_profits', 't_startup', 'P_DT_eq'],
    x_sort_by='t_startup',
    filters={
        't_startup': {'max': 100*24*3600},  # 100 days max
        'unrealized_profits': {'max': 2e6}   # 2 M$ max
    },
    unit_conversions={
        't_startup': {'factor': 1/(24*3600), 'unit': 'days'},
        'unrealized_profits': {'factor': 1e-6, 'unit': 'M$'},
        'P_DT_eq': {'factor': 1e-6, 'unit': 'MW'}
    },
    optimal_point=True,
    registry=registry
)

NOTES:
------
- Strip plots work directly with HDF5 files (no DataFrame preloading needed)
- Automatically extracts last value from time-series data (vectors)
- Supports computed metrics using simple arithmetic (e.g., "A - B")
- Uses parameter_registry.py for symbols and units (no hardcoded mappings)
- Filters are applied before plotting for focused analysis
- Memory efficient: only loads requested metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess

# Import hdf5plugin for LZ4 compression support
try:
    import hdf5plugin
except ImportError:
    pass


def extract_scalar_from_vector(data, take_last=True):
    """
    Extract scalar values from potentially vector data.
    
    Args:
        data: Array that may contain vectors or scalars
        take_last: If True, take last value of vectors; if False, take first
        
    Returns:
        1D numpy array of scalar values
    """
    if len(data) == 0:
        return np.array([])
    
    # Check if data contains vectors
    if isinstance(data[0], (list, np.ndarray)):
        if take_last:
            return np.array([arr[-1] if len(arr) > 0 else np.nan for arr in data])
        else:
            return np.array([arr[0] if len(arr) > 0 else np.nan for arr in data])
    else:
        # Already scalar
        return np.array(data)


def load_and_prepare_data(h5_file, metrics, filters, sort_by, registry):
    """
    Load data from HDF5 file, extract specified metrics, apply filters, and sort.
    
    Args:
        h5_file: Path to HDF5 file
        metrics: List of metric names to load
        filters: Dictionary of filter conditions {metric: {'min': val, 'max': val}}
        sort_by: Metric name to sort by (must be in metrics list)
        registry: ParameterRegistry instance
        
    Returns:
        pandas DataFrame with filtered and sorted data
    """
    import h5py
    
    data = {}
    
    with h5py.File(h5_file, 'r') as f:
        # Load sol_success if available
        if 'sol_success' in f.keys():
            success_flag = f['sol_success'][:]
            valid_mask = success_flag.astype(bool)
        else:
            # Assume all are valid if no success flag
            # Use first metric to determine length
            first_metric = metrics[0]
            if first_metric in f.keys():
                valid_mask = np.ones(f[first_metric].shape[0], dtype=bool)
            else:
                raise ValueError(f"Metric '{first_metric}' not found in HDF5 file")
        
        # Load each requested metric
        for metric in metrics:
            if metric not in f.keys():
                # Check if it's a computed metric (e.g., P_DT_eq - P_aux)
                if '-' in metric and metric.count('-') == 1:
                    parts = [p.strip() for p in metric.split('-')]
                    if len(parts) == 2 and all(p in f.keys() for p in parts):
                        # Compute difference
                        data1 = f[parts[0]][:]
                        data2 = f[parts[1]][:]
                        
                        # Extract scalars if needed
                        if len(data1.shape) > 1:
                            data1 = extract_scalar_from_vector(data1[valid_mask])
                        else:
                            data1 = data1[valid_mask]
                        
                        if len(data2.shape) > 1:
                            data2 = extract_scalar_from_vector(data2[valid_mask])
                        else:
                            data2 = data2[valid_mask]
                        
                        data[metric] = data1 - data2
                        continue
                
                print(f"   ⚠️  Warning: Metric '{metric}' not found in HDF5 file, skipping")
                continue
            
            dataset = f[metric][:]
            
            # Apply success filter
            if len(dataset.shape) > 1:
                # Vector data - extract last value
                dataset_filtered = dataset[valid_mask]
                data[metric] = extract_scalar_from_vector(dataset_filtered)
            else:
                # Scalar data
                data[metric] = dataset[valid_mask]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Apply filters
    mask = np.ones(len(df), dtype=bool)
    for metric, conditions in filters.items():
        if metric not in df.columns:
            continue
        
        # Ensure finite values
        mask &= np.isfinite(df[metric])
        
        if conditions.get('min') is not None:
            mask &= df[metric] >= conditions['min']
        if conditions.get('max') is not None:
            mask &= df[metric] <= conditions['max']
    
    df_filtered = df[mask].copy()
    
    print(f"   Loaded {len(df)} rows, filtered to {len(df_filtered)} rows ({len(df_filtered)/len(df)*100:.1f}%)")
    
    # Sort by specified metric
    if sort_by not in df_filtered.columns:
        print(f"   ⚠️  Warning: Cannot sort by '{sort_by}' (not in data), using first metric instead")
        sort_by = df_filtered.columns[0]
    
    df_sorted = df_filtered.sort_values(sort_by).reset_index(drop=True)
    
    return df_sorted


def add_trend_and_band(ax, x, y, color, label, frac=0.12, window_size=None, alpha_data=0.4, alpha_band=0.15):
    """
    Add LOWESS trend line and uncertainty band to axis.
    
    Args:
        ax: Matplotlib axis
        x: X coordinates (simulation indices)
        y: Y values
        color: Color for line and band
        label: Label for the data series
        frac: LOWESS fraction parameter (default: 0.12)
        window_size: Rolling window size for uncertainty band (default: adaptive)
        alpha_data: Alpha for raw data line (default: 0.4)
        alpha_band: Alpha for uncertainty band (default: 0.15)
        
    Returns:
        Tuple of (line_raw, line_trend) matplotlib Line2D objects
    """
    # Plot raw data
    line_raw = ax.plot(x, y, color=color, linewidth=1, label=label, alpha=alpha_data, zorder=1)
    
    # Compute LOWESS trend
    try:
        trend = lowess(y, x, frac=frac, it=1, return_sorted=False)
    except Exception as e:
        print(f"   ⚠️  Warning: LOWESS failed for {label}: {e}")
        # Fall back to moving average
        window = max(5, len(x) // 50)
        trend = pd.Series(y).rolling(window, center=True, min_periods=1).mean().values
    
    # Plot trend line
    line_trend = ax.plot(x, trend, color=color, linewidth=2.5, label=f'{label} (trend)', zorder=3)
    
    # Compute uncertainty band using rolling quantiles
    if window_size is None:
        window_size = max(25, len(x) // 20)  # Adaptive window size
    
    residuals = pd.Series(y - trend, index=pd.Index(x, name="x")).sort_index()
    min_periods = max(10, window_size // 3)
    
    lo_offset = residuals.rolling(window_size, min_periods=min_periods).quantile(0.10)
    hi_offset = residuals.rolling(window_size, min_periods=min_periods).quantile(0.90)
    
    lo = trend + lo_offset.values
    hi = trend + hi_offset.values
    
    # Plot uncertainty band
    ax.fill_between(x, lo, hi, color=color, alpha=alpha_band, linewidth=0, zorder=2)
    
    return line_raw, line_trend


def generate_strip_plot(h5_file, y_metrics, x_sort_by=None, filters=None, output_path=None,
                       unit_conversions=None, optimal_point=True, registry=None,
                       figsize=(14, 6), frac=0.12):
    """
    Generate a strip plot comparing multiple metrics with trend lines and uncertainty bands.
    
    This function creates a single plot with multiple y-axes (up to 3), showing trends
    and variability for each metric. Data is sorted by a specified metric for visualization.
    
    Args:
        h5_file: Path to HDF5 file
        y_metrics: List of 1-3 metric names to plot on y-axes (e.g., ['unrealized_profits', 't_startup', 'P_DT_eq'])
        x_sort_by: Metric to sort by for x-axis (default: first metric in y_metrics)
        filters: Dictionary of filter conditions {metric: {'min': val, 'max': val}}
        output_path: Path to save plot (PNG file). If None, uses default naming.
        unit_conversions: Dictionary of unit conversions {metric: {'factor': val, 'unit': 'unit_str'}}
                         Example: {'t_startup': {'factor': 1/(24*3600), 'unit': 'days'}}
        optimal_point: If True, mark optimal point minimizing all metrics (default: True)
        registry: ParameterRegistry instance (optional, will create if not provided)
        figsize: Figure size tuple (default: (14, 6))
        frac: LOWESS smoothing fraction (default: 0.12)
        
    Returns:
        Path to saved plot file
        
    Example:
        >>> generate_strip_plot(
        ...     h5_file='outputs/results.h5',
        ...     y_metrics=['unrealized_profits', 't_startup', 'P_DT_eq'],
        ...     x_sort_by='t_startup',
        ...     filters={'t_startup': {'max': 100*24*3600}, 'unrealized_profits': {'max': 2e6}},
        ...     unit_conversions={
        ...         't_startup': {'factor': 1/(24*3600), 'unit': 'days'},
        ...         'unrealized_profits': {'factor': 1e-6, 'unit': 'M$'},
        ...         'P_DT_eq': {'factor': 1e-6, 'unit': 'MW'}
        ...     }
        ... )
    """
    # Get registry if not provided
    if registry is None:
        from ddstartup.utils.parameter_registry import get_registry
        registry = get_registry()
    
    # Validate inputs
    if not y_metrics or len(y_metrics) == 0:
        raise ValueError("At least one y-metric must be specified")
    if len(y_metrics) > 3:
        raise ValueError("Maximum of 3 y-metrics supported")
    
    # Default sort metric
    if x_sort_by is None:
        x_sort_by = y_metrics[0]
    
    # Default filters
    if filters is None:
        filters = {}
    
    # Default unit conversions
    if unit_conversions is None:
        unit_conversions = {}
    
    # Load all required metrics (y_metrics + sort metric)
    all_metrics = list(set(y_metrics + [x_sort_by]))
    
    print(f"\n📊 Generating strip plot...")
    print(f"   File: {Path(h5_file).name}")
    print(f"   Y-metrics: {y_metrics}")
    print(f"   Sort by: {x_sort_by}")
    
    # Load and prepare data
    df = load_and_prepare_data(h5_file, all_metrics, filters, x_sort_by, registry)
    
    if len(df) == 0:
        print("   ❌ No data remaining after filtering!")
        return None
    
    # Apply unit conversions
    for metric in y_metrics:
        if metric in unit_conversions:
            conversion = unit_conversions[metric]
            df[metric] = df[metric] * conversion['factor']
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    
    x = np.arange(len(df))
    
    # Color scheme
    colors = ['tab:red', 'tab:blue', 'tab:green']
    axes = [ax1]
    
    # Plot first metric on left y-axis
    metric1 = y_metrics[0]
    y1 = df[metric1].values
    color1 = colors[0]
    
    line1_raw, line1_trend = add_trend_and_band(ax1, x, y1, color1, get_label(metric1, unit_conversions, registry))
    
    # Get y-axis label
    ylabel1 = get_axis_label(metric1, unit_conversions, registry)
    ax1.set_ylabel(ylabel1, color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot second metric on right y-axis (if provided)
    if len(y_metrics) >= 2:
        ax2 = ax1.twinx()
        axes.append(ax2)
        
        metric2 = y_metrics[1]
        y2 = df[metric2].values
        color2 = colors[1]
        
        line2_raw, line2_trend = add_trend_and_band(ax2, x, y2, color2, get_label(metric2, unit_conversions, registry))
        
        ylabel2 = get_axis_label(metric2, unit_conversions, registry)
        ax2.set_ylabel(ylabel2, color=color2, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color2)
    
    # Plot third metric on second right y-axis (if provided)
    if len(y_metrics) >= 3:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        axes.append(ax3)
        
        metric3 = y_metrics[2]
        y3 = df[metric3].values
        color3 = colors[2]
        
        line3_raw, line3_trend = add_trend_and_band(ax3, x, y3, color3, get_label(metric3, unit_conversions, registry))
        
        ylabel3 = get_axis_label(metric3, unit_conversions, registry)
        ax3.set_ylabel(ylabel3, color=color3, fontsize=12)
        ax3.tick_params(axis='y', labelcolor=color3)
    
    # Set x-axis label
    xlabel = f'Simulation Index (sorted by {get_label(x_sort_by, unit_conversions, registry)})'
    ax1.set_xlabel(xlabel, fontsize=12)
    
    # Add title
    file_name = Path(h5_file).stem
    title = f'Strip Plot: {", ".join([get_label(m, unit_conversions, registry) for m in y_metrics])}\n{file_name}'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    # Add optimal point marker if requested
    if optimal_point and len(y_metrics) >= 2:
        # Find optimal point by minimizing normalized Euclidean distance
        metrics_for_opt = y_metrics[:min(len(y_metrics), 3)]  # Use up to 3 metrics
        
        # Normalize each metric
        normalized = []
        for metric in metrics_for_opt:
            y_vals = df[metric].values
            y_min, y_max = y_vals.min(), y_vals.max()
            if y_max > y_min:
                y_norm = (y_vals - y_min) / (y_max - y_min)
            else:
                y_norm = np.zeros_like(y_vals)
            normalized.append(y_norm)
        
        # Compute Euclidean distance in normalized space
        combined_distance = np.sqrt(sum(norm**2 for norm in normalized))
        optimal_idx = combined_distance.argmin()
        
        # Add star marker (on ax1, visible across all)
        star = ax1.scatter([optimal_idx], [df.iloc[optimal_idx][y_metrics[0]]], 
                          color='gold', s=200, zorder=5, marker='*', 
                          edgecolors='black', linewidths=2, label='Optimal Point')
        
        print(f"\n   🎯 Optimal point found at index {optimal_idx}:")
        for metric in y_metrics:
            val = df.iloc[optimal_idx][metric]
            label = get_label(metric, unit_conversions, registry)
            print(f"      {label}: {val:.3e}")
    
    # Combine legends
    lines = []
    labels = []
    
    if len(y_metrics) >= 1:
        lines.extend(line1_raw + line1_trend)
        labels.extend([l.get_label() for l in line1_raw + line1_trend])
    if len(y_metrics) >= 2:
        lines.extend(line2_raw + line2_trend)
        labels.extend([l.get_label() for l in line2_raw + line2_trend])
    if len(y_metrics) >= 3:
        lines.extend(line3_raw + line3_trend)
        labels.extend([l.get_label() for l in line3_raw + line3_trend])
    
    if optimal_point and len(y_metrics) >= 2:
        lines.append(star)
        labels.append('Optimal Point')
    
    ax1.legend(lines, labels, loc='upper left', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        output_dir = Path(h5_file).parent
        plot_name = f"strip_{'_'.join(y_metrics[:3])}_{file_name}.png"
        output_path = output_dir / plot_name
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n   ✅ Saved: {output_path}")
    
    return output_path


def get_label(metric, unit_conversions, registry):
    """
    Get human-readable label for a metric.
    
    Args:
        metric: Metric name
        unit_conversions: Dictionary of unit conversions
        registry: ParameterRegistry instance
        
    Returns:
        String label
    """
    # Check if it's a computed metric (e.g., "P_DT_eq - P_aux")
    if '-' in metric and metric.count('-') == 1:
        parts = [p.strip() for p in metric.split('-')]
        if len(parts) == 2:
            label1 = registry.get_symbol(parts[0])
            label2 = registry.get_symbol(parts[1])
            return f'{label1} - {label2}'
    
    # Use symbol from registry
    return registry.get_symbol(metric)


def get_axis_label(metric, unit_conversions, registry):
    """
    Get axis label with units for a metric.
    
    Args:
        metric: Metric name
        unit_conversions: Dictionary of unit conversions
        registry: ParameterRegistry instance
        
    Returns:
        String axis label with units
    """
    label = get_label(metric, unit_conversions, registry)
    
    # Get unit
    if metric in unit_conversions:
        unit = unit_conversions[metric]['unit']
    else:
        # Check if it's a computed metric
        if '-' in metric and metric.count('-') == 1:
            parts = [p.strip() for p in metric.split('-')]
            if len(parts) == 2:
                unit1 = registry.get_unit(parts[0])
                unit2 = registry.get_unit(parts[1])
                if unit1 == unit2:
                    unit = unit1
                else:
                    unit = f'{unit1}'  # Use first unit if different
            else:
                unit = registry.get_unit(metric)
        else:
            unit = registry.get_unit(metric)
    
    if unit and unit != 'dimensionless' and unit != 'boolean':
        return f'{label} [{unit}]'
    else:
        return label
