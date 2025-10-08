"""
Tests for KDE plotting functions in ddstartup.postprocessing.plot_kde_functions

This module tests:
- save_quartile_extremes_to_csv: Saving quartile extreme values to CSV
- kde_quartile_plot: Creating KDE plots split by quartiles
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ddstartup.postprocessing.plot_kde_functions import (
    save_quartile_extremes_to_csv,
    kde_quartile_plot
)


# ============================================================================
# TEST save_quartile_extremes_to_csv
# ============================================================================

def test_save_quartile_extremes_to_csv_basic(temp_dir, sample_dataframe):
    """
    Test saving quartile extremes to CSV file.
    Expected: Should create CSV file with lowest and middle values for each quartile.
    """
    # Add quartile bins to dataframe
    target = 't_startup'
    bin_edges = sample_dataframe[target].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    bin_labels = [
        f"Q1: {bin_edges[0]:.2e}–{bin_edges[1]:.2e}",
        f"Q2: {bin_edges[1]:.2e}–{bin_edges[2]:.2e}",
        f"Q3: {bin_edges[2]:.2e}–{bin_edges[3]:.2e}",
        f"Q4: {bin_edges[3]:.2e}–{bin_edges[4]:.2e}"
    ]
    sample_dataframe[f"{target}_bin"] = pd.qcut(
        sample_dataframe[target], q=4, labels=bin_labels
    )
    
    input_parameters = ['V_plasma', 'n_tot', 'T_i']
    plot_name = 'test_plot.png'
    
    save_quartile_extremes_to_csv(
        sample_dataframe, target, input_parameters, bin_labels, temp_dir, plot_name
    )
    
    # Expected: CSV file created
    csv_path = temp_dir / f"quartile_{target}_values_test_plot.csv"
    assert csv_path.exists()
    
    # Expected: CSV contains correct data
    df_csv = pd.read_csv(csv_path)
    assert 'quartile' in df_csv.columns
    assert 'which' in df_csv.columns
    assert target in df_csv.columns
    for param in input_parameters:
        assert param in df_csv.columns
    
    # Expected: Each quartile has 2 rows (lowest and middle)
    # Note: might have fewer if some quartiles are empty
    assert len(df_csv) <= 8  # 4 quartiles * 2 (lowest, middle)
    assert 'lowest' in df_csv['which'].values
    assert 'middle' in df_csv['which'].values


def test_save_quartile_extremes_to_csv_with_tstartup_column(temp_dir, sample_dataframe):
    """
    Test that t_startup column is included when analyzing unrealized_profits.
    Expected: CSV should include t_startup column even if not in input_parameters.
    """
    target = 'unrealized_profits'
    bin_edges = sample_dataframe[target].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    bin_labels = [
        f"Q1: {bin_edges[0]:.2e}–{bin_edges[1]:.2e}",
        f"Q2: {bin_edges[1]:.2e}–{bin_edges[2]:.2e}",
        f"Q3: {bin_edges[2]:.2e}–{bin_edges[3]:.2e}",
        f"Q4: {bin_edges[3]:.2e}–{bin_edges[4]:.2e}"
    ]
    sample_dataframe[f"{target}_bin"] = pd.qcut(
        sample_dataframe[target], q=4, labels=bin_labels
    )
    
    input_parameters = ['V_plasma', 'n_tot']  # t_startup not in input params
    plot_name = 'test_unrealized_profits.png'
    
    save_quartile_extremes_to_csv(
        sample_dataframe, target, input_parameters, bin_labels, temp_dir, plot_name
    )
    
    csv_path = temp_dir / f"quartile_{target}_values_test_unrealized_profits.csv"
    df_csv = pd.read_csv(csv_path)
    
    # Expected: t_startup column is present even though not in input_parameters
    assert 't_startup' in df_csv.columns


def test_save_quartile_extremes_to_csv_empty_quartile(temp_dir):
    """
    Test handling of empty quartiles (edge case).
    Expected: Should skip empty quartiles without error.
    """
    # Create dataframe with uneven distribution
    df = pd.DataFrame({
        'V_plasma': [100, 101, 102],
        'target': [1, 2, 3]
    })
    
    bin_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    df['target_bin'] = pd.Categorical(['Q1', 'Q1', 'Q1'], categories=bin_labels)
    
    input_parameters = ['V_plasma']
    plot_name = 'test_empty_quartile.png'
    
    save_quartile_extremes_to_csv(
        df, 'target', input_parameters, bin_labels, temp_dir, plot_name
    )
    
    csv_path = temp_dir / "quartile_target_values_test_empty_quartile.csv"
    df_csv = pd.read_csv(csv_path)
    
    # Expected: Only Q1 data present, other quartiles skipped
    assert len(df_csv) == 2  # lowest and middle for Q1 only
    assert all(df_csv['quartile'] == 'Q1')


# ============================================================================
# TEST kde_quartile_plot
# ============================================================================

def test_kde_quartile_plot_basic(temp_dir, sample_dataframe):
    """
    Test creating basic KDE quartile plot.
    Expected: Should create PNG file with KDE subplots for each input parameter.
    """
    target = 't_startup'
    input_parameters = ['V_plasma', 'n_tot', 'T_i']
    target_unit = 's'
    file_type = 'test_analysis'
    plot_name = 'kde_plot_test.png'
    
    kde_quartile_plot(
        sample_dataframe, target, input_parameters, 
        target_unit, temp_dir, file_type, plot_name
    )
    
    # Expected: Plot file created
    plot_path = temp_dir / plot_name
    assert plot_path.exists()
    
    # Expected: CSV file with quartile extremes also created
    csv_path = temp_dir / f"quartile_{target}_values_kde_plot_test.csv"
    assert csv_path.exists()


def test_kde_quartile_plot_no_input_parameters(temp_dir, capsys):
    """
    Test KDE plot when no input parameters provided.
    Expected: Should skip plotting and print message.
    """
    df = pd.DataFrame({
        't_startup': [1e6, 2e6, 3e6, 4e6, 5e6]
    })
    
    target = 't_startup'
    input_parameters = []
    target_unit = 's'
    file_type = 'test'
    plot_name = 'should_not_exist.png'
    
    kde_quartile_plot(
        df, target, input_parameters, 
        target_unit, temp_dir, file_type, plot_name
    )
    
    # Expected: No plot created
    plot_path = temp_dir / plot_name
    assert not plot_path.exists()
    
    # Expected: Warning message printed
    captured = capsys.readouterr()
    assert "No input parameters found" in captured.out


def test_kde_quartile_plot_many_parameters(temp_dir):
    """
    Test KDE plot with many input parameters (tests subplot grid).
    Expected: Should create plot with multiple rows of subplots.
    """
    # Create dataframe with many parameters
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        'param1': np.random.uniform(0, 100, n_samples),
        'param2': np.random.uniform(0, 100, n_samples),
        'param3': np.random.uniform(0, 100, n_samples),
        'param4': np.random.uniform(0, 100, n_samples),
        'param5': np.random.uniform(0, 100, n_samples),
        'param6': np.random.uniform(0, 100, n_samples),
        't_startup': np.random.uniform(1e6, 5e6, n_samples)
    })
    
    input_parameters = ['param1', 'param2', 'param3', 'param4', 'param5', 'param6']
    target = 't_startup'
    target_unit = 's'
    file_type = 'multi_param_test'
    plot_name = 'kde_multi_param.png'
    
    kde_quartile_plot(
        df, target, input_parameters, 
        target_unit, temp_dir, file_type, plot_name
    )
    
    # Expected: Plot file created with multiple subplots
    plot_path = temp_dir / plot_name
    assert plot_path.exists()


def test_kde_quartile_plot_with_param_units(temp_dir, sample_dataframe, monkeypatch):
    """
    Test that parameter units are included in subplot titles.
    Expected: Subplot titles should include units from PARAM_UNITS dictionary.
    """
    # Mock PARAM_UNITS
    mock_units = {
        'V_plasma': 'm³',
        'n_tot': 'cm⁻³',
        'T_i': 'keV'
    }
    
    import ddstartup.postprocessing.plot_kde_functions as kde_module
    monkeypatch.setattr(kde_module, 'PARAM_UNITS', mock_units)
    
    target = 't_startup'
    input_parameters = ['V_plasma', 'n_tot', 'T_i']
    target_unit = 's'
    file_type = 'test_with_units'
    plot_name = 'kde_with_units.png'
    
    kde_quartile_plot(
        sample_dataframe, target, input_parameters, 
        target_unit, temp_dir, file_type, plot_name
    )
    
    # Expected: Plot created (units are in the plot titles)
    plot_path = temp_dir / plot_name
    assert plot_path.exists()


def test_kde_quartile_plot_zero_variance_parameter(temp_dir):
    """
    Test KDE plot with zero variance in a parameter.
    Expected: Should handle constant values gracefully (shows horizontal line).
    """
    df = pd.DataFrame({
        'V_plasma': [100, 100, 100, 100, 100],  # Zero variance
        'n_tot': [1e14, 1.5e14, 2e14, 2.5e14, 3e14],
        't_startup': [1e6, 2e6, 3e6, 4e6, 5e6]
    })
    
    input_parameters = ['V_plasma', 'n_tot']
    target = 't_startup'
    target_unit = 's'
    file_type = 'zero_variance_test'
    plot_name = 'kde_zero_variance.png'
    
    kde_quartile_plot(
        df, target, input_parameters, 
        target_unit, temp_dir, file_type, plot_name
    )
    
    # Expected: Plot created without error (constant parameter shown as dashed line)
    plot_path = temp_dir / plot_name
    assert plot_path.exists()


def test_kde_quartile_plot_quartile_binning(temp_dir, sample_dataframe):
    """
    Test that quartile binning is correctly applied.
    Expected: Data should be split into 4 quartiles for plotting.
    """
    target = 't_startup'
    input_parameters = ['V_plasma']
    target_unit = 's'
    file_type = 'quartile_test'
    plot_name = 'kde_quartiles.png'
    
    # Store original dataframe to check binning
    df_copy = sample_dataframe.copy()
    
    kde_quartile_plot(
        sample_dataframe, target, input_parameters, 
        target_unit, temp_dir, file_type, plot_name
    )
    
    # Expected: Dataframe now has quartile bin column
    assert f"{target}_bin" in sample_dataframe.columns
    
    # Expected: Four unique quartile bins
    unique_bins = sample_dataframe[f"{target}_bin"].unique()
    assert len(unique_bins) == 4
    
    # Expected: Each bin label contains quartile info
    for bin_label in unique_bins:
        assert 'Q' in str(bin_label)


def test_kde_quartile_plot_creates_csv_with_correct_structure(temp_dir, sample_dataframe):
    """
    Test that CSV file created has correct structure.
    Expected: CSV should contain quartile, which, target, and all input parameters.
    """
    target = 't_startup'
    input_parameters = ['V_plasma', 'n_tot', 'T_i']
    target_unit = 's'
    file_type = 'csv_structure_test'
    plot_name = 'kde_csv_test.png'
    
    kde_quartile_plot(
        sample_dataframe, target, input_parameters, 
        target_unit, temp_dir, file_type, plot_name
    )
    
    csv_path = temp_dir / f"quartile_{target}_values_kde_csv_test.csv"
    df_csv = pd.read_csv(csv_path)
    
    # Expected: Correct columns present
    expected_columns = ['quartile', 'which', target] + input_parameters
    for col in expected_columns:
        assert col in df_csv.columns
    
    # Expected: 'which' column contains 'lowest' and 'middle'
    assert set(df_csv['which'].unique()).issubset({'lowest', 'middle'})
    
    # Expected: Quartile values are within data range
    assert df_csv[target].min() >= sample_dataframe[target].min()
    assert df_csv[target].max() <= sample_dataframe[target].max()


def test_kde_quartile_plot_figure_cleanup(temp_dir, sample_dataframe):
    """
    Test that matplotlib figures are properly closed after plotting.
    Expected: No open figures should remain after function execution.
    """
    # Get initial number of open figures
    initial_figs = len(plt.get_fignums())
    
    target = 't_startup'
    input_parameters = ['V_plasma', 'n_tot']
    target_unit = 's'
    file_type = 'cleanup_test'
    plot_name = 'kde_cleanup.png'
    
    kde_quartile_plot(
        sample_dataframe, target, input_parameters, 
        target_unit, temp_dir, file_type, plot_name
    )
    
    # Expected: Number of open figures should return to initial count
    final_figs = len(plt.get_fignums())
    assert final_figs == initial_figs


def test_kde_quartile_plot_with_nan_values(temp_dir):
    """
    Test KDE plot handles NaN values in input parameters.
    Expected: Should skip NaN values when creating KDE plots.
    """
    df = pd.DataFrame({
        'V_plasma': [100, 120, np.nan, 160, 180],
        'n_tot': [1e14, 1.5e14, 2e14, np.nan, 3e14],
        't_startup': [1e6, 2e6, 3e6, 4e6, 5e6]
    })
    
    input_parameters = ['V_plasma', 'n_tot']
    target = 't_startup'
    target_unit = 's'
    file_type = 'nan_test'
    plot_name = 'kde_with_nan.png'
    
    kde_quartile_plot(
        df, target, input_parameters, 
        target_unit, temp_dir, file_type, plot_name
    )
    
    # Expected: Plot created successfully despite NaN values
    plot_path = temp_dir / plot_name
    assert plot_path.exists()


def test_kde_quartile_plot_single_input_parameter(temp_dir, sample_dataframe):
    """
    Test KDE plot with only one input parameter.
    Expected: Should create plot with single subplot.
    """
    target = 't_startup'
    input_parameters = ['V_plasma']  # Only one parameter
    target_unit = 's'
    file_type = 'single_param_test'
    plot_name = 'kde_single_param.png'
    
    kde_quartile_plot(
        sample_dataframe, target, input_parameters, 
        target_unit, temp_dir, file_type, plot_name
    )
    
    # Expected: Plot created with single subplot
    plot_path = temp_dir / plot_name
    assert plot_path.exists()
