"""
PDF (Probability Density Function) Plot Functions

This module contains functions for generating PDF plots comparing distributions
across different datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.stats import gaussian_kde


def generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path, 
                     smooth=False, kde_bandwidth='scott', registry=None):
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
    """
    # Get registry if not provided
    if registry is None:
        from ddstartup.utils.parameter_registry import get_registry
        registry = get_registry()
    
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
                if smooth:
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
    
    # Get unit and symbol for variable
    unit = registry.get_unit(var)
    symbol = registry.get_symbol(var)
    
    # Format labels with symbols
    xlabel = f"{symbol} [{unit}]" if unit else symbol
    plt.xlabel(xlabel)
    plt.ylabel('Probability Density')
    
    # Add subtitle indicating mode
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
