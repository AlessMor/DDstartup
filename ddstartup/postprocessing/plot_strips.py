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
from tqdm import tqdm

# Import hdf5plugin for LZ4 compression support
try:
    import hdf5plugin
except ImportError:
    pass

# Check if datashader is available
HAS_DATASHADER = False
try:
    import datashader as ds
    import datashader.transfer_functions as tf
    HAS_DATASHADER = True
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


def plot_with_datashader(ax, x, y, color, width=1200, height=400, verbose=False):
    """
    Plot scatter using datashader for efficient rendering of large datasets.
    Falls back to matplotlib if datashader is not available.
    
    Args:
        ax: Matplotlib axis
        x: X coordinates
        y: Y values
        color: Color for the plot (used for matplotlib fallback)
        width: Canvas width in pixels
        height: Canvas height in pixels
        verbose: Print progress info
        
    Returns:
        True if datashader was used, False if matplotlib fallback
    """
    if not HAS_DATASHADER:
        return False
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Remove NaNs/Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    
    if len(x) == 0:
        return True
    
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    
    # Pad ranges slightly
    x_pad = (x_max - x_min) * 0.02 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.02 if y_max > y_min else 1.0
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad
    
    if verbose:
        print(f"      Datashader: rendering {len(x):,} points to {width}x{height} canvas...")
    
    cvs = ds.Canvas(plot_width=width, plot_height=height,
                    x_range=(x_min, x_max), y_range=(y_min, y_max))
    agg = cvs.points(pd.DataFrame({'x': x, 'y': y}), 'x', 'y')
    
    # Use color-based shading
    img = tf.shade(agg, cmap=[color, color], how='log')
    
    ax.imshow(img.to_pil(), origin='lower',
              extent=(x_min, x_max, y_min, y_max), aspect='auto', alpha=0.6)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    return True


def plot_scatter_rasterized(ax, x, y, color, alpha=0.4, s=1, verbose=False):
    """
    Plot scatter with rasterization for efficient rendering.
    Uses small markers and rasterizes the PathCollection.
    
    Args:
        ax: Matplotlib axis
        x: X coordinates
        y: Y values
        color: Color for markers
        alpha: Transparency
        s: Marker size
        verbose: Print progress info
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) == 0:
        return
    
    if verbose:
        print(f"      Matplotlib rasterized: plotting {len(x):,} points...")
    
    sc = ax.scatter(x, y, c=color, s=s, alpha=alpha, linewidths=0, rasterized=True)
    # Ensure all collections are rasterized
    for coll in ax.collections:
        coll.set_rasterized(True)


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


def generate_strip_plot(
    h5_file=None,
    y_metrics=None,
    x_sort_by=None,
    filters=None,
    output_path=None,
    output_dir=None,
    outputs_dir=None,
    plot_name_prefix=None,
    plot_name=None,
    unit_conversions=None,
    optimal_point=True,
    registry=None,
    show_titles=True,
    figsize=(14, 6),
    frac=0.12,
    df=None,
    strip_settings=None,
    use_datashader=True,
    verbose=False,
    **_,
):
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
        show_titles: If False, omit the figure title
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
    
    # Allow settings dict from dispatcher/config
    strip_settings = strip_settings or {}
    if y_metrics is None:
        y_metrics = strip_settings.get("y_metrics")
    if x_sort_by is None:
        x_sort_by = strip_settings.get("sort_by")
    if filters is None:
        filters = strip_settings.get("filters")
    if unit_conversions is None:
        unit_conversions = strip_settings.get("unit_conversions")
    # Allow config to override the default; fall back to True only if neither provided
    cfg_optimal = strip_settings.get("optimal_point")
    if cfg_optimal is not None:
        optimal_point = cfg_optimal
    if optimal_point is None:
        optimal_point = True
    if frac is None:
        frac = strip_settings.get("frac", 0.12)
    show_titles = strip_settings.get("show_titles", show_titles)
    use_datashader = strip_settings.get("use_datashader", use_datashader)
    verbose = strip_settings.get("verbose", verbose)

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
    
    # Load all required metrics (y_metrics + sort metric) preserving order
    all_metrics = list(y_metrics)
    if x_sort_by not in all_metrics:
        all_metrics.append(x_sort_by)
    
    print(f"\n📊 Generating strip plot...")
    if h5_file is not None:
        print(f"   File: {Path(h5_file).name}")
    print(f"   Y-metrics: {y_metrics}")
    print(f"   Sort by: {x_sort_by}")
    
    # Load and prepare data (prefer provided DataFrame from dispatcher)
    if df is not None:
        df_source = df.copy()
        # take scalars from vector/object columns (last element)
        inner = (getattr(df_source, "attrs", {}) or {}).get("_inner_dims", {})
        for col in list(df_source.columns):
            if int(inner.get(col, 1)) != 1 and col in all_metrics:
                df_source[col] = extract_scalar_from_vector(df_source[col].values)
        df = df_source
        # Apply filters
        mask = np.ones(len(df), dtype=bool)
        for metric, bounds in filters.items():
            if metric not in df.columns:
                continue
            if 'min' in bounds:
                mask &= df[metric] >= bounds['min']
            if 'max' in bounds:
                mask &= df[metric] <= bounds['max']
        df = df.loc[mask].reset_index(drop=True)
        # Keep only needed metrics
        missing = [m for m in all_metrics if m not in df.columns]
        if missing:
            print(f"   ⚠️  Missing metrics in DataFrame for strip plot: {missing}. Skipping.")
            return None
        df = df[all_metrics].copy()
        # Sort rows for consistency with HDF5 code path
        if x_sort_by not in df.columns:
            print(f"   ⚠️  Warning: Cannot sort by '{x_sort_by}' (not in data), using first metric instead")
            x_sort_by = df.columns[0]
        df = df.sort_values(x_sort_by).reset_index(drop=True)
    else:
        if h5_file is None:
            print("   ⚠️  No data source provided for strip plot. Skipping.")
            return None
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
    
    # Color scheme (hex colors for datashader compatibility)
    colors = ['#d62728', '#1f77b4', '#2ca02c']  # red, blue, green
    axes = [ax1]
    
    # Determine if we should use datashader (for large datasets)
    n_points = len(df)
    datashader_threshold = 50000  # Use datashader for datasets larger than this
    should_use_datashader = use_datashader and HAS_DATASHADER and n_points > datashader_threshold
    
    if verbose:
        print(f"   Dataset size: {n_points:,} points")
        if should_use_datashader:
            print(f"   Using datashader for efficient rendering")
        elif use_datashader and not HAS_DATASHADER:
            print(f"   Datashader not available, using matplotlib (install with: pip install datashader)")
        else:
            print(f"   Using matplotlib for rendering")
    
    # Store trend lines for legend
    all_lines = []
    all_labels = []
    
    # Progress bar for metrics
    metrics_iter = tqdm(enumerate(y_metrics), total=len(y_metrics), 
                        desc="   Plotting metrics", disable=not verbose)
    
    for idx, metric in metrics_iter:
        color = colors[idx]
        y_vals = df[metric].values
        
        if idx == 0:
            ax = ax1
        elif idx == 1:
            ax2 = ax1.twinx()
            axes.append(ax2)
            ax = ax2
        else:  # idx == 2
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            axes.append(ax3)
            ax = ax3
        
        # Plot raw data (datashader or matplotlib)
        if should_use_datashader:
            plot_with_datashader(ax, x, y_vals, color, verbose=verbose)
            # For datashader, we only add trend line (no raw line in legend)
            line_raw = []
        else:
            # Use rasterized matplotlib for medium datasets
            if n_points > 10000:
                plot_scatter_rasterized(ax, x, y_vals, color, verbose=verbose)
                line_raw = []
            else:
                # Standard matplotlib for small datasets
                line_raw = ax.plot(x, y_vals, color=color, linewidth=1, 
                                   label=get_label(metric, unit_conversions, registry), 
                                   alpha=0.4, zorder=1)
        
        # Always add LOWESS trend line (downsample for performance)
        if verbose:
            print(f"      Computing LOWESS trend for {metric}...")
        
        # LOWESS is O(n²) - must downsample for large datasets
        max_lowess_points = 10000
        if len(x) > max_lowess_points:
            # Uniform sampling to preserve distribution
            step = len(x) // max_lowess_points
            idx_sample = np.arange(0, len(x), step)
            x_sample = x[idx_sample]
            y_sample = y_vals[idx_sample]
            if verbose:
                print(f"      Downsampled {len(x):,} -> {len(x_sample):,} points for LOWESS")
        else:
            x_sample = x
            y_sample = y_vals
        
        try:
            trend_sample = lowess(y_sample, x_sample, frac=frac, it=1, return_sorted=False)
            # Interpolate back to full x range
            if len(x) > max_lowess_points:
                trend = np.interp(x, x_sample, trend_sample)
            else:
                trend = trend_sample
        except Exception as e:
            if verbose:
                print(f"      ⚠️  LOWESS failed: {e}, using moving average")
            window = max(5, len(x) // 50)
            trend = pd.Series(y_vals).rolling(window, center=True, min_periods=1).mean().values
        
        line_trend = ax.plot(x, trend, color=color, linewidth=2.5, 
                             label=f'{get_label(metric, unit_conversions, registry)} (trend)', zorder=3)
        
        # Add uncertainty band
        window_size = max(25, len(x) // 20)
        residuals = pd.Series(y_vals - trend, index=pd.Index(x, name="x")).sort_index()
        min_periods = max(10, window_size // 3)
        
        lo_offset = residuals.rolling(window_size, min_periods=min_periods).quantile(0.10)
        hi_offset = residuals.rolling(window_size, min_periods=min_periods).quantile(0.90)
        
        lo = trend + lo_offset.values
        hi = trend + hi_offset.values
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=2)
        
        # Set y-axis label
        ylabel = get_axis_label(metric, unit_conversions, registry)
        ax.set_ylabel(ylabel, color=color, fontsize=12)
        ax.tick_params(axis='y', labelcolor=color)
        
        if idx == 0:
            ax.grid(True, alpha=0.3)
        
        # Collect for legend
        all_lines.extend(line_raw + line_trend)
        all_labels.extend([l.get_label() for l in line_raw + line_trend])
    
    # Set x-axis label
    xlabel = f'Simulation Index (sorted by {get_label(x_sort_by, unit_conversions, registry)})'
    ax1.set_xlabel(xlabel, fontsize=12)
    
    # Add title
    file_name = Path(h5_file).stem if h5_file is not None else "dataframe"
    if show_titles:
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
    if optimal_point and len(y_metrics) >= 2:
        all_lines.append(star)
        all_labels.append('Optimal Point')
    
    ax1.legend(all_lines, all_labels, loc='upper left', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        outdir = output_dir if output_dir is not None else outputs_dir
        if outdir is None:
            outdir = Path(h5_file).parent if h5_file is not None else Path(".")
        else:
            outdir = Path(outdir)
        stem = plot_name_prefix or plot_name or f"strip_{'_'.join(y_metrics[:3])}_{file_name}"
        output_path = outdir / (stem if stem.lower().endswith(".png") else f"{stem}.png")
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
