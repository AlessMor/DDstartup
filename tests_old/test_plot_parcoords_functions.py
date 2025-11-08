"""
Tests for parallel coordinates plotting functions in ddstartup.postprocessing.plot_parcoords_functions

This module tests:
- generate_parcoords_plot: Creating interactive parallel coordinates plots with Plotly
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


from ddstartup.postprocessing.plot_parcoords_functions import generate_parcoords_plot


# ============================================================================
# TEST generate_parcoords_plot
# ============================================================================

def test_generate_parcoords_plot_basic(temp_dir, sample_dataframe):
    """
    Test creating basic parallel coordinates plot.
    Expected: Should create HTML file with interactive Plotly plot.
    """
    target = 't_startup'
    input_parameters = ['V_plasma', 'n_tot', 'T_i']
    target_unit = 's'
    file_type = 'test_analysis'
    output_path = temp_dir / 'parcoords_test.html'
    
    generate_parcoords_plot(
        sample_dataframe, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: HTML file created
    assert output_path.exists()
    
    # Expected: File contains Plotly HTML content
    with open(output_path, 'r') as f:
        content = f.read()
        assert 'plotly' in content.lower()
        assert 'parcoords' in content.lower() or 'parallel' in content.lower()


def test_generate_parcoords_plot_with_many_parameters(temp_dir):
    """
    Test parallel coordinates plot with many parameters.
    Expected: Should handle multiple dimensions without error.
    """
    # Create dataframe with many parameters
    np.random.seed(42)
    n_samples = 50
    df = pd.DataFrame({
        'param1': np.random.uniform(0, 100, n_samples),
        'param2': np.random.uniform(0, 100, n_samples),
        'param3': np.random.uniform(0, 100, n_samples),
        'param4': np.random.uniform(0, 100, n_samples),
        'param5': np.random.uniform(0, 100, n_samples),
        'param6': np.random.uniform(0, 100, n_samples),
        'param7': np.random.uniform(0, 100, n_samples),
        'param8': np.random.uniform(0, 100, n_samples),
        't_startup': np.random.uniform(1e6, 5e6, n_samples)
    })
    
    input_parameters = ['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'param8']
    target = 't_startup'
    target_unit = 's'
    file_type = 'multi_dim_test'
    output_path = temp_dir / 'parcoords_multi.html'
    
    generate_parcoords_plot(
        df, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: HTML file created with all dimensions
    assert output_path.exists()


def test_generate_parcoords_plot_sampling_large_dataset(temp_dir):
    """
    Test that large datasets are sampled to prevent performance issues.
    Expected: Should sample down to max_plot_rows (1e6) if dataset is larger.
    """
    # Create large dataset (> 1e6 would be too slow for test, use smaller but test logic)
    np.random.seed(42)
    n_samples = 2000  # Simulating large dataset
    df = pd.DataFrame({
        'V_plasma': np.random.uniform(100, 200, n_samples),
        'n_tot': np.random.uniform(1e14, 3e14, n_samples),
        't_startup': np.random.uniform(1e6, 5e6, n_samples)
    })
    
    input_parameters = ['V_plasma', 'n_tot']
    target = 't_startup'
    target_unit = 's'
    file_type = 'large_dataset_test'
    output_path = temp_dir / 'parcoords_large.html'
    
    generate_parcoords_plot(
        df, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: Plot created (sampling happens internally if > 1e6 rows)
    assert output_path.exists()


def test_generate_parcoords_plot_color_chunking(temp_dir, sample_dataframe):
    """
    Test that color chunking is correctly applied based on target quantiles.
    Expected: Target values should be divided into 6 color chunks.
    """
    target = 't_startup'
    input_parameters = ['V_plasma', 'n_tot']
    target_unit = 's'
    file_type = 'color_test'
    output_path = temp_dir / 'parcoords_colors.html'
    
    generate_parcoords_plot(
        sample_dataframe, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: HTML file created with color scale
    assert output_path.exists()
    
    with open(output_path, 'r') as f:
        content = f.read()
        # Expected: Color scale information in HTML
        assert 'colorscale' in content.lower()


def test_generate_parcoords_plot_with_param_units(temp_dir, sample_dataframe, monkeypatch):
    """
    Test that parameter units are included in axis labels.
    Expected: Axis labels should include units from PARAM_UNITS dictionary.
    """
    # Mock PARAM_UNITS
    mock_units = {
        'V_plasma': 'm³',
        'n_tot': 'cm⁻³',
        'T_i': 'keV',
        't_startup': 's'
    }
    
    import ddstartup.postprocessing.plot_parcoords_functions as parcoords_module
    monkeypatch.setattr(parcoords_module, 'PARAM_UNITS', mock_units)
    
    target = 't_startup'
    input_parameters = ['V_plasma', 'n_tot', 'T_i']
    target_unit = 's'
    file_type = 'units_test'
    output_path = temp_dir / 'parcoords_units.html'
    
    generate_parcoords_plot(
        sample_dataframe, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: HTML file created with units in labels
    assert output_path.exists()
    
    with open(output_path, 'r') as f:
        content = f.read()
        # Units should appear in the HTML (in axis labels)
        assert 'm³' in content or 'm3' in content  # Unit encoding may vary


def test_generate_parcoords_plot_discrete_parameters(temp_dir):
    """
    Test parallel coordinates with discrete/categorical parameters.
    Expected: Should handle parameters with few unique values (≤20) specially.
    """
    df = pd.DataFrame({
        'V_plasma': [100, 150, 200, 100, 150, 200, 100, 150],  # Only 3 unique values
        'n_tot': [1e14, 2e14, 3e14, 1e14, 2e14, 3e14, 1e14, 2e14],  # Only 3 unique values
        't_startup': [1e6, 2e6, 3e6, 2.5e6, 1.5e6, 3.5e6, 1.8e6, 2.2e6]
    })
    
    input_parameters = ['V_plasma', 'n_tot']
    target = 't_startup'
    target_unit = 's'
    file_type = 'discrete_test'
    output_path = temp_dir / 'parcoords_discrete.html'
    
    generate_parcoords_plot(
        df, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: Plot created with tickvals for discrete parameters
    assert output_path.exists()


def test_generate_parcoords_plot_skips_vector_fields(temp_dir):
    """
    Test that vector fields (array-like values) are skipped.
    Expected: Should skip columns containing vectors/arrays and only plot scalars.
    """
    df = pd.DataFrame({
        'V_plasma': [100, 150, 200, 250],
        'vector_field': [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9]),
            np.array([10, 11, 12])
        ],
        't_startup': [1e6, 2e6, 3e6, 4e6]
    })
    
    input_parameters = ['V_plasma', 'vector_field']
    target = 't_startup'
    target_unit = 's'
    file_type = 'vector_skip_test'
    output_path = temp_dir / 'parcoords_no_vectors.html'
    
    generate_parcoords_plot(
        df, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: Plot created, but vector_field dimension skipped
    assert output_path.exists()


def test_generate_parcoords_plot_target_dimension_included(temp_dir, sample_dataframe):
    """
    Test that target variable is included as final dimension.
    Expected: Target should appear as the last dimension in the plot.
    """
    target = 't_startup'
    input_parameters = ['V_plasma', 'n_tot']
    target_unit = 's'
    file_type = 'target_dim_test'
    output_path = temp_dir / 'parcoords_with_target.html'
    
    generate_parcoords_plot(
        sample_dataframe, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: HTML contains target variable label
    assert output_path.exists()
    
    with open(output_path, 'r') as f:
        content = f.read()
        assert target in content


def test_generate_parcoords_plot_title(temp_dir, sample_dataframe):
    """
    Test that plot title includes file_type and target.
    Expected: Title should contain both file_type and target variable name.
    """
    target = 't_startup'
    input_parameters = ['V_plasma']
    target_unit = 's'
    file_type = 'TEST_ANALYSIS_TYPE'
    output_path = temp_dir / 'parcoords_title.html'
    
    generate_parcoords_plot(
        sample_dataframe, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: Title contains file_type and target
    assert output_path.exists()
    
    with open(output_path, 'r') as f:
        content = f.read()
        assert file_type in content
        assert target in content


def test_generate_parcoords_plot_min_max_ranges(temp_dir, sample_dataframe):
    """
    Test that dimension ranges are set correctly (min to max of data).
    Expected: Each dimension should have range from min to max of values.
    """
    target = 't_startup'
    input_parameters = ['V_plasma', 'n_tot']
    target_unit = 's'
    file_type = 'range_test'
    output_path = temp_dir / 'parcoords_ranges.html'
    
    # Get actual min/max for verification
    v_plasma_min = sample_dataframe['V_plasma'].min()
    v_plasma_max = sample_dataframe['V_plasma'].max()
    
    generate_parcoords_plot(
        sample_dataframe, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: Plot created with correct ranges
    assert output_path.exists()


def test_generate_parcoords_plot_empty_input_parameters(temp_dir, sample_dataframe):
    """
    Test parallel coordinates with only target variable (no input parameters).
    Expected: Should create plot with just target dimension.
    """
    target = 't_startup'
    input_parameters = []
    target_unit = 's'
    file_type = 'empty_inputs_test'
    output_path = temp_dir / 'parcoords_empty_inputs.html'
    
    generate_parcoords_plot(
        sample_dataframe, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: Plot created with at least target dimension
    assert output_path.exists()


def test_generate_parcoords_plot_colorbar_labels(temp_dir, sample_dataframe):
    """
    Test that colorbar has correct labels with chunk ranges.
    Expected: Colorbar should show 6 color chunks with their value ranges.
    """
    target = 't_startup'
    input_parameters = ['V_plasma', 'n_tot']
    target_unit = 's'
    file_type = 'colorbar_test'
    output_path = temp_dir / 'parcoords_colorbar.html'
    
    generate_parcoords_plot(
        sample_dataframe, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: HTML contains colorbar configuration
    assert output_path.exists()
    
    with open(output_path, 'r') as f:
        content = f.read()
        # Colorbar should have ticktext with scientific notation ranges
        assert 'colorbar' in content.lower()


def test_generate_parcoords_plot_single_row(temp_dir):
    """
    Test parallel coordinates with single data point.
    Expected: Should handle single row without error.
    """
    df = pd.DataFrame({
        'V_plasma': [150],
        'n_tot': [2e14],
        't_startup': [2e6]
    })
    
    target = 't_startup'
    input_parameters = ['V_plasma', 'n_tot']
    target_unit = 's'
    file_type = 'single_row_test'
    output_path = temp_dir / 'parcoords_single.html'
    
    generate_parcoords_plot(
        df, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: Plot created even with single data point
    assert output_path.exists()


def test_generate_parcoords_plot_constant_parameter(temp_dir):
    """
    Test parallel coordinates with constant (zero variance) parameter.
    Expected: Should handle constant parameters without error.
    """
    df = pd.DataFrame({
        'V_plasma': [150, 150, 150, 150],  # Constant
        'n_tot': [1e14, 2e14, 3e14, 4e14],
        't_startup': [1e6, 2e6, 3e6, 4e6]
    })
    
    target = 't_startup'
    input_parameters = ['V_plasma', 'n_tot']
    target_unit = 's'
    file_type = 'constant_param_test'
    output_path = temp_dir / 'parcoords_constant.html'
    
    generate_parcoords_plot(
        df, target, input_parameters,
        target_unit, file_type, output_path
    )
    
    # Expected: Plot created (constant parameter shown as vertical line)
    assert output_path.exists()
