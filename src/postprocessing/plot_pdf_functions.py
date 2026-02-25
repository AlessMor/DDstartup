"""
PDF (Probability Density Function) Plot Functions

This module contains functions for generating PDF plots comparing distributions
across different datasets.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.stats import gaussian_kde

from src.postprocessing.plot_utils_functions import ensure_registry, resolve_outdir_and_stem


def generate_pdf_plot(
    dataframes_dict=None,
    var=None,
    label_list=None,
    filters=None,
    output_path=None,
    *,
    df=None,
    target=None,
    output_dir=None,
    outputs_dir=None,
    plot_name_prefix=None,
    plot_name=None,
    pdf_smooth=False,
    kde_bandwidth='scott',
    registry=None,
    show_titles=True,
    **_,
):
    """
    Generate a probability density function plot.
    
    Args:
        dataframes_dict: Dictionary mapping filenames to data dictionaries
        var: Variable name to plot
        label_list: List of labels for each dataset
        filters: Dictionary of filters (min/max) for variables
        output_path: Path to save plot file
        smooth: If True, use KDE smoothing instead of histogram bins (default: False)
        kde_bandwidth: Bandwidth method for KDE ('scott', 'silverman', or float) (default: 'scott')
        registry: ParameterRegistry instance (optional, will create if not provided)
        show_titles: If False, omit the figure title
    """
    # Normalize arguments
    filters = filters or {}
    if var is None:
        var = target
    if output_path is None:
        outdir, stem = resolve_outdir_and_stem(
            output_dir=output_dir,
            outputs_dir=outputs_dir,
            plot_name_prefix=plot_name_prefix,
            plot_name=plot_name,
            default_stem=f"pdf_{var}",
        )
        output_path = outdir / f"{stem}.png"

    registry = ensure_registry(registry)
    
    vmin = filters.get(var, {}).get('min', None)
    vmax = filters.get(var, {}).get('max', None)
    
    plt.figure(figsize=(7, 5))
    has_data = False
    
    # Support legacy dict-of-arrays or new single-DataFrame call
    if dataframes_dict is None and df is not None:
        dataframes_dict = {plot_name_prefix or "data": df}
        label_list = [plot_name_prefix or "data"]
    if label_list is None:
        label_list = list(dataframes_dict.keys())

    for key, label in zip(dataframes_dict.keys(), label_list):
        df_data = dataframes_dict[key]
        # accept pre-serialized arrays or DataFrames/Series
        if hasattr(df_data, "get"):
            arr = df_data.get(var)
            if isinstance(df_data, (pd.DataFrame, pd.Series)) and arr is None and var in df_data:
                arr = df_data[var]
            if isinstance(arr, pd.Series):
                arr = arr.to_numpy()
        else:
            arr = None
        if arr is not None:
            arr = np.asarray(arr)
            # skip vector-valued data
            if arr.dtype == object and len(arr) and hasattr(arr[0], "__len__") and not isinstance(arr[0], str):
                continue
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                if pdf_smooth or smooth:
                    # Use Kernel Density Estimation for smooth curves
                    # Filter data first if limits are specified
                    arr_filtered = arr.copy()
                    if vmin is not None:
                        arr_filtered = arr_filtered[arr_filtered >= vmin]
                    if vmax is not None:
                        arr_filtered = arr_filtered[arr_filtered <= vmax]
                    
                    if len(arr_filtered) < 2:
                        # Fall back to histogram if insufficient data
                        counts, bins = np.histogram(arr_filtered, bins=10000, density=True)
                        bin_centers = 0.5 * (bins[:-1] + bins[1:])
                        plt.plot(bin_centers, counts, drawstyle='steps-mid', label=f"{label}")
                    else:
                        # Compute KDE
                        try:
                            kde = gaussian_kde(arr_filtered, bw_method=kde_bandwidth)
                            
                            # Create evaluation points in log space for better coverage
                            data_min = arr_filtered.min()
                            data_max = arr_filtered.max()
                            
                            # Ensure data_min > 0 for log scale
                            if data_min <= 0:
                                data_min = arr_filtered[arr_filtered > 0].min() if np.any(arr_filtered > 0) else 1e-10
                            
                            # Generate points in log space
                            x_eval = np.logspace(np.log10(data_min), np.log10(data_max), 1000)
                            
                            # Evaluate KDE
                            density = kde(x_eval)
                            
                            # Plot smooth curve
                            plt.plot(x_eval, density, linewidth=2, label=f"{label}")
                            has_data = True
                            
                        except Exception as e:
                            # Fall back to histogram if KDE fails
                            print(f"   ⚠️  KDE failed for {label}, using histogram: {e}")
                            counts, bins = np.histogram(arr_filtered, bins=10000, density=True)
                            bin_centers = 0.5 * (bins[:-1] + bins[1:])
                            plt.plot(bin_centers, counts, drawstyle='steps-mid', label=f"{label}")
                            has_data = True
                else:
                    # Original histogram-based approach
                    counts, bins = np.histogram(arr, bins=10000, density=True)
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    mask = np.ones_like(bin_centers, dtype=bool)
                    if vmin is not None:
                        mask &= bin_centers >= vmin
                    if vmax is not None:
                        mask &= bin_centers <= vmax
                    plt.plot(bin_centers[mask], counts[mask], drawstyle='steps-mid', label=f"{label}")
                    has_data = True
    
    # Format axis labels using registry
    xlabel = registry.get_param_label(var)
    symbol = registry.get_symbol(var)
    
    plt.xlabel(xlabel)
    plt.ylabel('Probability Density')
    
    # Add subtitle indicating mode
    if show_titles:
        title_text = f'PDF of {symbol}'
        if smooth:
            title_text += '\n(Kernel Density Estimation)'
        plt.title(title_text)
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
