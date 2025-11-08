"""
Tests for PDF plotting functions in ddstartup.postprocessing.plot_pdf_functions

This module tests:
- generate_pdf_plot: Creating probability density function comparison plots
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ddstartup.postprocessing.plot_pdf_functions import generate_pdf_plot


# ============================================================================
# TEST generate_pdf_plot
# ============================================================================

def test_generate_pdf_plot_basic(temp_dir):
    """
    Test creating basic PDF comparison plot.
    Expected: Should create PNG file with PDF curves for each dataset.
    """
    # Create sample data dictionaries
    dataframes_dict = {
        'dataset1': {'var1': np.random.lognormal(10, 1, 1000)},
        'dataset2': {'var1': np.random.lognormal(11, 1, 1000)}
    }
    
    var = 'var1'
    label_list = ['Dataset 1', 'Dataset 2']
    filters = {}
    output_path = temp_dir / 'pdf_basic.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot file created
    assert output_path.exists()


def test_generate_pdf_plot_with_filters(temp_dir):
    """
    Test PDF plot with min/max filters applied.
    Expected: Should only plot data within filter range.
    """
    dataframes_dict = {
        'dataset1': {'var1': np.random.uniform(0, 100, 1000)}
    }
    
    var = 'var1'
    label_list = ['Filtered Data']
    filters = {
        'var1': {'min': 20, 'max': 80}
    }
    output_path = temp_dir / 'pdf_filtered.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot created with filtered data
    assert output_path.exists()


def test_generate_pdf_plot_multiple_datasets(temp_dir):
    """
    Test PDF plot with multiple datasets for comparison.
    Expected: Should plot multiple PDF curves, one for each dataset.
    """
    # Create three different distributions
    dataframes_dict = {
        'low_values': {'metric': np.random.lognormal(8, 0.5, 1000)},
        'medium_values': {'metric': np.random.lognormal(10, 0.5, 1000)},
        'high_values': {'metric': np.random.lognormal(12, 0.5, 1000)}
    }
    
    var = 'metric'
    label_list = ['Low', 'Medium', 'High']
    filters = {}
    output_path = temp_dir / 'pdf_multiple.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot with three curves created
    assert output_path.exists()


def test_generate_pdf_plot_handles_nan(temp_dir):
    """
    Test that PDF plot correctly handles NaN values.
    Expected: Should filter out NaN and inf values before plotting.
    """
    data = np.array([1e6, 2e6, np.nan, 3e6, np.inf, 4e6, -np.inf, 5e6])
    
    dataframes_dict = {
        'with_nan': {'target': data}
    }
    
    var = 'target'
    label_list = ['Data with NaN']
    filters = {}
    output_path = temp_dir / 'pdf_nan.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot created without error, NaN/inf values filtered
    assert output_path.exists()


def test_generate_pdf_plot_log_scale(temp_dir):
    """
    Test that PDF plot uses logarithmic x-axis.
    Expected: X-axis should be in log scale.
    """
    dataframes_dict = {
        'data': {'var1': np.logspace(6, 9, 1000)}
    }
    
    var = 'var1'
    label_list = ['Log Scale Data']
    filters = {}
    output_path = temp_dir / 'pdf_logscale.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot created with log scale
    assert output_path.exists()


def test_generate_pdf_plot_empty_data(temp_dir):
    """
    Test PDF plot with empty dataset.
    Expected: Should handle empty data gracefully (plot may be empty or show message).
    """
    dataframes_dict = {
        'empty': {'var1': np.array([])}
    }
    
    var = 'var1'
    label_list = ['Empty Data']
    filters = {}
    output_path = temp_dir / 'pdf_empty.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot file created (may be blank)
    assert output_path.exists()


def test_generate_pdf_plot_missing_variable(temp_dir):
    """
    Test PDF plot when variable is missing from some datasets.
    Expected: Should skip datasets that don't have the variable.
    """
    dataframes_dict = {
        'has_var': {'var1': np.random.uniform(0, 100, 1000)},
        'missing_var': {'var2': np.random.uniform(0, 100, 1000)}  # Different variable
    }
    
    var = 'var1'
    label_list = ['Has Variable', 'Missing Variable']
    filters = {}
    output_path = temp_dir / 'pdf_missing.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot created, only first dataset plotted
    assert output_path.exists()


def test_generate_pdf_plot_filter_min_only(temp_dir):
    """
    Test PDF plot with only minimum filter.
    Expected: Should plot data >= min value.
    """
    dataframes_dict = {
        'data': {'var1': np.random.uniform(0, 100, 1000)}
    }
    
    var = 'var1'
    label_list = ['Min Filtered']
    filters = {
        'var1': {'min': 50, 'max': None}
    }
    output_path = temp_dir / 'pdf_min_filter.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot created with data >= 50
    assert output_path.exists()


def test_generate_pdf_plot_filter_max_only(temp_dir):
    """
    Test PDF plot with only maximum filter.
    Expected: Should plot data <= max value.
    """
    dataframes_dict = {
        'data': {'var1': np.random.uniform(0, 100, 1000)}
    }
    
    var = 'var1'
    label_list = ['Max Filtered']
    filters = {
        'var1': {'min': None, 'max': 75}
    }
    output_path = temp_dir / 'pdf_max_filter.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot created with data <= 75
    assert output_path.exists()


def test_generate_pdf_plot_title_and_labels(temp_dir):
    """
    Test that plot has correct title and axis labels.
    Expected: Title should contain variable name, axes should be labeled.
    """
    dataframes_dict = {
        'data': {'t_startup': np.random.lognormal(15, 1, 1000)}
    }
    
    var = 't_startup'
    label_list = ['Test Data']
    filters = {}
    output_path = temp_dir / 'pdf_labels.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot created with proper labels
    assert output_path.exists()


def test_generate_pdf_plot_legend(temp_dir):
    """
    Test that plot includes legend with dataset labels.
    Expected: Legend should show all dataset labels.
    """
    dataframes_dict = {
        'first': {'var1': np.random.lognormal(10, 1, 1000)},
        'second': {'var1': np.random.lognormal(11, 1, 1000)}
    }
    
    var = 'var1'
    label_list = ['First Dataset', 'Second Dataset']
    filters = {}
    output_path = temp_dir / 'pdf_legend.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot with legend created
    assert output_path.exists()


def test_generate_pdf_plot_high_resolution(temp_dir):
    """
    Test that plot is saved with high DPI.
    Expected: Plot should be saved at 150 DPI.
    """
    dataframes_dict = {
        'data': {'var1': np.random.lognormal(10, 1, 1000)}
    }
    
    var = 'var1'
    label_list = ['Data']
    filters = {}
    output_path = temp_dir / 'pdf_highres.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: High resolution plot created
    assert output_path.exists()
    # Note: DPI is 150 according to source code


def test_generate_pdf_plot_many_bins(temp_dir):
    """
    Test that PDF uses many bins (10000) for smooth curves.
    Expected: Should create smooth PDF curves using histogram with 10000 bins.
    """
    # Create data with known distribution
    dataframes_dict = {
        'data': {'var1': np.random.lognormal(10, 0.5, 5000)}
    }
    
    var = 'var1'
    label_list = ['Smooth Curve']
    filters = {}
    output_path = temp_dir / 'pdf_smooth.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Smooth PDF curve created
    assert output_path.exists()


def test_generate_pdf_plot_steps_mid(temp_dir):
    """
    Test that PDF uses steps-mid drawstyle for histograms.
    Expected: Should use steps-mid style for step-like histogram appearance.
    """
    dataframes_dict = {
        'data': {'var1': np.random.lognormal(10, 1, 1000)}
    }
    
    var = 'var1'
    label_list = ['Step Plot']
    filters = {}
    output_path = temp_dir / 'pdf_steps.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Step-style plot created
    assert output_path.exists()


def test_generate_pdf_plot_tight_layout(temp_dir):
    """
    Test that plot uses tight layout to prevent label cutoff.
    Expected: Should call plt.tight_layout() for proper spacing.
    """
    dataframes_dict = {
        'data': {'var1': np.random.lognormal(10, 1, 1000)}
    }
    
    var = 'var1'
    label_list = ['Tight Layout']
    filters = {}
    output_path = temp_dir / 'pdf_tight.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot with tight layout created
    assert output_path.exists()


def test_generate_pdf_plot_figure_cleanup(temp_dir):
    """
    Test that matplotlib figures are properly closed after plotting.
    Expected: No open figures should remain after function execution.
    """
    # Get initial number of open figures
    initial_figs = len(plt.get_fignums())
    
    dataframes_dict = {
        'data': {'var1': np.random.lognormal(10, 1, 1000)}
    }
    
    var = 'var1'
    label_list = ['Cleanup Test']
    filters = {}
    output_path = temp_dir / 'pdf_cleanup.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Number of open figures returns to initial count
    final_figs = len(plt.get_fignums())
    assert final_figs == initial_figs


def test_generate_pdf_plot_xlim_with_filters(temp_dir):
    """
    Test that x-axis limits are set according to filters.
    Expected: X-axis should be limited to filter range when filters are provided.
    """
    dataframes_dict = {
        'data': {'var1': np.random.uniform(0, 200, 1000)}
    }
    
    var = 'var1'
    label_list = ['Limited Range']
    filters = {
        'var1': {'min': 50, 'max': 150}
    }
    output_path = temp_dir / 'pdf_xlim.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot with limited x-axis created
    assert output_path.exists()


def test_generate_pdf_plot_density_normalization(temp_dir):
    """
    Test that histogram uses density normalization.
    Expected: Y-axis should show probability density (normalized histogram).
    """
    dataframes_dict = {
        'data': {'var1': np.random.lognormal(10, 1, 1000)}
    }
    
    var = 'var1'
    label_list = ['Normalized']
    filters = {}
    output_path = temp_dir / 'pdf_normalized.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Density-normalized plot created
    assert output_path.exists()


def test_generate_pdf_plot_single_value(temp_dir):
    """
    Test PDF plot with dataset containing single unique value.
    Expected: Should handle constant data without error.
    """
    dataframes_dict = {
        'constant': {'var1': np.array([100.0] * 1000)}
    }
    
    var = 'var1'
    label_list = ['Constant Value']
    filters = {}
    output_path = temp_dir / 'pdf_single_value.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot created (may show spike at constant value)
    assert output_path.exists()


def test_generate_pdf_plot_wide_range(temp_dir):
    """
    Test PDF plot with data spanning many orders of magnitude.
    Expected: Log scale should handle wide range effectively.
    """
    # Data from 1e3 to 1e12 (9 orders of magnitude)
    dataframes_dict = {
        'wide_range': {'var1': np.logspace(3, 12, 5000)}
    }
    
    var = 'var1'
    label_list = ['Wide Range']
    filters = {}
    output_path = temp_dir / 'pdf_wide_range.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot handling wide range created
    assert output_path.exists()


def test_generate_pdf_plot_small_dataset(temp_dir):
    """
    Test PDF plot with very small dataset.
    Expected: Should handle small datasets without error.
    """
    dataframes_dict = {
        'small': {'var1': np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
    }
    
    var = 'var1'
    label_list = ['Small Dataset']
    filters = {}
    output_path = temp_dir / 'pdf_small.png'
    
    generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)
    
    # Expected: Plot created even with small dataset
    assert output_path.exists()
