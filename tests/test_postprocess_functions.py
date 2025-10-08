"""
Tests for postprocessing functions in ddstartup.postprocessing.postprocess_functions

This module tests:
- File and directory utilities (find_latest_output_folder, find_latest_h5_file)
- YAML configuration loading
- Filter parsing and application
- HDF5 data loading
- Data filtering and parameter extraction
- Target variable scaling
- Color scale generation
"""

import pytest
import numpy as np
import pandas as pd
import h5py
import yaml
from pathlib import Path
import time

from ddstartup.postprocessing.postprocess_functions import (
    find_latest_output_folder,
    find_latest_h5_file,
    load_yaml_config,
    parse_filter_expression,
    clean_filters,
    apply_filters,
    load_h5_to_dataframe,
    filter_finite,
    get_input_parameters,
    scale_target,
    get_discrete_colorscale,
    prepare_dataframes
)


# ============================================================================
# TEST FILE AND DIRECTORY UTILITIES
# ============================================================================

def test_find_latest_output_folder_empty(temp_dir):
    """
    Test find_latest_output_folder with empty directory.
    Expected: Should return (None, None) when no folders exist.
    """
    result_folder, result_files = find_latest_output_folder(temp_dir)
    assert result_folder is None
    assert result_files is None


def test_find_latest_output_folder_with_folders(temp_dir):
    """
    Test find_latest_output_folder with multiple folders.
    Expected: Should return the most recently modified folder.
    """
    # Create multiple folders with different timestamps
    folder1 = temp_dir / "run_001"
    folder2 = temp_dir / "run_002"
    folder3 = temp_dir / "run_003"
    
    folder1.mkdir()
    time.sleep(0.01)
    folder2.mkdir()
    time.sleep(0.01)
    folder3.mkdir()
    
    # Add an h5 file to the latest folder
    h5_file = folder3 / "results.h5"
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('dummy', data=[1, 2, 3])
    
    result_folder, result_files = find_latest_output_folder(temp_dir)
    
    # Expected: folder3 is most recent and contains h5 files
    assert result_folder == folder3
    assert result_files is not None
    assert len(result_files) == 1
    assert result_files[0] == h5_file


def test_find_latest_output_folder_ignores_hidden(temp_dir):
    """
    Test that hidden directories (starting with .) are ignored.
    Expected: Hidden folders should not be considered.
    """
    # Create visible and hidden folders
    visible_folder = temp_dir / "visible"
    hidden_folder = temp_dir / ".hidden"
    
    hidden_folder.mkdir()
    time.sleep(0.01)
    visible_folder.mkdir()
    
    result_folder, _ = find_latest_output_folder(temp_dir)
    
    # Expected: Only visible folder is found
    assert result_folder == visible_folder
    assert result_folder != hidden_folder


def test_find_latest_h5_file_in_folder(temp_dir):
    """
    Test find_latest_h5_file when h5 file is in a subfolder.
    Expected: Should find h5 file in the most recent output folder.
    """
    folder = temp_dir / "run_001"
    folder.mkdir()
    
    h5_file = folder / "data.h5"
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('test', data=[1, 2, 3])
    
    result = find_latest_h5_file(temp_dir)
    
    # Expected: Should find the h5 file in the folder
    assert result == h5_file


def test_find_latest_h5_file_fallback_to_root(temp_dir):
    """
    Test find_latest_h5_file fallback to root directory.
    Expected: Should search root directory if no folders have h5 files.
    """
    # Create h5 file directly in root
    h5_file = temp_dir / "root_data.h5"
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('test', data=[1, 2, 3])
    
    result = find_latest_h5_file(temp_dir)
    
    # Expected: Should find h5 file in root directory
    assert result == h5_file


def test_find_latest_h5_file_none(temp_dir):
    """
    Test find_latest_h5_file with no h5 files.
    Expected: Should return None when no h5 files exist.
    """
    result = find_latest_h5_file(temp_dir)
    
    # Expected: No h5 files found
    assert result is None


# ============================================================================
# TEST YAML CONFIGURATION LOADING
# ============================================================================

def test_load_yaml_config(temp_dir):
    """
    Test loading YAML configuration file.
    Expected: Should correctly parse and return YAML content as dictionary.
    """
    config_path = temp_dir / "config.yaml"
    config_data = {
        'target_variables': ['t_startup', 'unrealized_gains'],
        'input_filters': {'V_plasma': {'min': 100, 'max': 200}},
        'output_filters': {'t_startup': {'min': 0, 'max': 1e8}}
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    result = load_yaml_config(config_path)
    
    # Expected: Loaded config matches original data
    assert result == config_data
    assert result['target_variables'] == ['t_startup', 'unrealized_gains']
    assert result['input_filters']['V_plasma']['min'] == 100


# ============================================================================
# TEST FILTER UTILITIES
# ============================================================================

def test_parse_filter_expression_single_less_than():
    """
    Test parsing single less than filter expression.
    Expected: Should parse "V_plasma<150" correctly.
    """
    result = parse_filter_expression("V_plasma<150")
    
    # Expected: V_plasma has max of 150, no min
    assert 'V_plasma' in result
    assert result['V_plasma']['max'] == 150
    assert result['V_plasma']['min'] is None


def test_parse_filter_expression_single_greater_than():
    """
    Test parsing single greater than filter expression.
    Expected: Should parse "n_tot>1e14" correctly.
    """
    result = parse_filter_expression("n_tot>1e14")
    
    # Expected: n_tot has min of 1e14, no max
    assert 'n_tot' in result
    assert result['n_tot']['min'] == 1e14
    assert result['n_tot']['max'] is None


def test_parse_filter_expression_equality():
    """
    Test parsing equality filter expression.
    Expected: Should parse "T_i==50" correctly (min and max equal).
    """
    result = parse_filter_expression("T_i==50")
    
    # Expected: T_i has both min and max equal to 50
    assert 'T_i' in result
    assert result['T_i']['min'] == 50
    assert result['T_i']['max'] == 50


def test_parse_filter_expression_multiple():
    """
    Test parsing multiple filter expressions.
    Expected: Should parse comma-separated filters correctly.
    """
    result = parse_filter_expression("V_plasma<150,n_tot>1e14,T_i>=30")
    
    # Expected: All three filters parsed correctly
    assert len(result) == 3
    assert result['V_plasma']['max'] == 150
    assert result['n_tot']['min'] == 1e14
    assert result['T_i']['min'] == 30


def test_parse_filter_expression_empty():
    """
    Test parsing empty filter expression.
    Expected: Should return empty dictionary for empty string.
    """
    result = parse_filter_expression("")
    
    # Expected: Empty dictionary
    assert result == {}


def test_parse_filter_expression_none():
    """
    Test parsing None filter expression.
    Expected: Should return empty dictionary for None input.
    """
    result = parse_filter_expression(None)
    
    # Expected: Empty dictionary
    assert result == {}


def test_clean_filters_removes_none():
    """
    Test clean_filters removes entries with both min and max as None.
    Expected: Entries with no constraints should be removed.
    """
    filters = {
        'V_plasma': {'min': 100, 'max': 200},
        'n_tot': {'min': None, 'max': None},  # Should be removed
        'T_i': {'min': 30, 'max': None},
        'empty': None  # Should be removed
    }
    
    result = clean_filters(filters)
    
    # Expected: Only V_plasma and T_i remain
    assert len(result) == 2
    assert 'V_plasma' in result
    assert 'T_i' in result
    assert 'n_tot' not in result
    assert 'empty' not in result


def test_clean_filters_empty_input():
    """
    Test clean_filters with empty or None input.
    Expected: Should return empty dictionary for None input.
    """
    result = clean_filters(None)
    
    # Expected: Empty dictionary
    assert result == {}


def test_apply_filters_basic(sample_dataframe):
    """
    Test basic filter application on dataframe.
    Expected: Should filter rows based on input and output filters.
    """
    input_filters = {'V_plasma': {'min': 120, 'max': 180}}
    output_filters = {}
    
    result = apply_filters(sample_dataframe, input_filters, output_filters, 't_startup')
    
    # Expected: Only rows with V_plasma between 120 and 180
    assert len(result) < len(sample_dataframe)
    assert result['V_plasma'].min() >= 120
    assert result['V_plasma'].max() <= 180


def test_apply_filters_removes_nan(sample_dataframe_with_nan):
    """
    Test that apply_filters removes NaN values in target variable.
    Expected: Rows with NaN target values should be filtered out.
    """
    input_filters = {}
    output_filters = {}
    
    result = apply_filters(sample_dataframe_with_nan, input_filters, output_filters, 't_startup')
    
    # Expected: All NaN values in t_startup removed
    assert not result['t_startup'].isna().any()
    assert len(result) < len(sample_dataframe_with_nan)


def test_apply_filters_output_filters(sample_dataframe):
    """
    Test output variable filtering.
    Expected: Should filter based on output variable constraints.
    """
    input_filters = {}
    output_filters = {'unrealized_gains': {'min': 1e8, 'max': None}}
    
    result = apply_filters(sample_dataframe, input_filters, output_filters, 't_startup')
    
    # Expected: Only rows with unrealized_gains >= 1e8
    assert result['unrealized_gains'].min() >= 1e8


# ============================================================================
# TEST HDF5 DATA LOADING
# ============================================================================

def test_load_h5_to_dataframe_1d_data(temp_dir):
    """
    Test loading 1D datasets from HDF5 file.
    Expected: Should load 1D arrays as DataFrame columns.
    """
    h5_path = temp_dir / "test_1d.h5"
    
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('V_plasma', data=[100, 150, 200])
        f.create_dataset('n_tot', data=[1e14, 2e14, 3e14])
        f.create_dataset('t_startup', data=[1e6, 2e6, 3e6])
    
    result = load_h5_to_dataframe(h5_path)
    
    # Expected: DataFrame with 3 columns and 3 rows
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert 'V_plasma' in result.columns
    assert 'n_tot' in result.columns
    assert 't_startup' in result.columns
    assert result['V_plasma'].tolist() == [100, 150, 200]


def test_load_h5_to_dataframe_2d_data(temp_dir):
    """
    Test loading 2D datasets from HDF5 file.
    Expected: Should load 2D arrays as lists of vectors in DataFrame.
    """
    h5_path = temp_dir / "test_2d.h5"
    
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('scalar', data=[1, 2, 3])
        f.create_dataset('vector', data=[[10, 20], [30, 40], [50, 60]])
    
    result = load_h5_to_dataframe(h5_path)
    
    # Expected: scalar as column, vector as list of arrays
    assert len(result) == 3
    assert 'scalar' in result.columns
    assert 'vector' in result.columns
    assert isinstance(result['vector'].iloc[0], np.ndarray)
    assert result['vector'].iloc[0].tolist() == [10, 20]


def test_load_h5_to_dataframe_parameter_fields(temp_dir):
    """
    Test loading data from parameter_fields group.
    Expected: Should load datasets from parameter_fields group.
    """
    h5_path = temp_dir / "test_param_fields.h5"
    
    with h5py.File(h5_path, 'w') as f:
        param_group = f.create_group('parameter_fields')
        param_group.create_dataset('V_plasma', data=[100, 200, 300])
        f.create_dataset('result', data=[1, 2, 3])
    
    result = load_h5_to_dataframe(h5_path)
    
    # Expected: Both parameter_fields and root datasets loaded
    assert 'V_plasma' in result.columns
    assert 'result' in result.columns
    assert len(result) == 3


# ============================================================================
# TEST DATA FILTERING AND PARAMETER EXTRACTION
# ============================================================================

def test_filter_finite_basic(sample_dataframe_with_nan):
    """
    Test filtering out non-finite values in target variable.
    Expected: Should remove rows with NaN, inf, or -inf in target.
    """
    result = filter_finite(sample_dataframe_with_nan, 't_startup')
    
    # Expected: All finite values in t_startup
    assert result['t_startup'].isna().sum() == 0
    assert np.isfinite(result['t_startup']).all()


def test_filter_finite_with_min_max(sample_dataframe):
    """
    Test filter_finite with min/max constraints.
    Expected: Should filter based on min/max values as well as finiteness.
    """
    filter_dict = {'min': 1.5e6, 'max': 2.5e6}
    
    result = filter_finite(sample_dataframe, 't_startup', filter_dict)
    
    # Expected: Values within min/max range
    assert result['t_startup'].min() >= 1.5e6
    assert result['t_startup'].max() <= 2.5e6


def test_get_input_parameters_basic(sample_dataframe):
    """
    Test extracting input parameters from dataframe.
    Expected: Should return list of input parameter names in desired order.
    """
    result = get_input_parameters(sample_dataframe, 't_startup')
    
    # Expected: Input parameters excluding target variable
    assert isinstance(result, list)
    assert 't_startup' not in result
    assert 'V_plasma' in result
    assert 'n_tot' in result


def test_get_input_parameters_removes_target(sample_dataframe):
    """
    Test that target variable is excluded from input parameters.
    Expected: Target variable should not appear in input parameters list.
    """
    result = get_input_parameters(sample_dataframe, 'V_plasma')
    
    # Expected: V_plasma not in result
    assert 'V_plasma' not in result


def test_get_input_parameters_t_seeded_removes_specific(sample_dataframe):
    """
    Test parameter removal for T_seeded analysis type.
    Expected: Should remove tau_p_He3 and I_target when filename contains 'T_seeded'.
    """
    # Add parameters that should be removed for T_seeded
    df = sample_dataframe.copy()
    df['tau_p_He3'] = [1, 2, 3, 4, 5]
    df['I_target'] = [10, 20, 30, 40, 50]
    
    result = get_input_parameters(df, 't_startup', filename='T_seeded_analysis.h5')
    
    # Expected: tau_p_He3 and I_target removed
    assert 'tau_p_He3' not in result
    assert 'I_target' not in result


def test_scale_target_unrealized_gains(sample_dataframe):
    """
    Test scaling unrealized_gains to millions of dollars.
    Expected: Should divide by 1e6 and return 'M$' unit.
    """
    df = sample_dataframe.copy()
    
    result_df, unit = scale_target(df, 'unrealized_gains')
    
    # Expected: Values scaled down by 1e6, unit is 'M$'
    assert unit == 'M$'
    assert result_df['unrealized_gains'].iloc[0] == sample_dataframe['unrealized_gains'].iloc[0] / 1e6


def test_scale_target_startup_to_days():
    """
    Test scaling t_startup to days when values are large.
    Expected: Should scale to days and return 'days' unit for large values.
    """
    df = pd.DataFrame({'t_startup': [50 * 24 * 3600]})  # 50 days in seconds
    
    result_df, unit = scale_target(df, 't_startup')
    
    # Expected: Scaled to days
    assert unit == 'days'
    assert abs(result_df['t_startup'].iloc[0] - 50) < 0.1


def test_scale_target_startup_to_hours():
    """
    Test scaling t_startup to hours for intermediate values.
    Expected: Should scale to hours and return 'hours' unit.
    """
    df = pd.DataFrame({'t_startup': [30 * 3600]})  # 30 hours in seconds
    
    result_df, unit = scale_target(df, 't_startup')
    
    # Expected: Scaled to hours
    assert unit == 'hours'
    assert abs(result_df['t_startup'].iloc[0] - 30) < 0.1


def test_scale_target_no_scaling():
    """
    Test that small t_startup values remain in seconds.
    Expected: Should keep seconds unit for small values.
    """
    df = pd.DataFrame({'t_startup': [3600]})  # 1 hour in seconds
    
    result_df, unit = scale_target(df, 't_startup')
    
    # Expected: No scaling, unit is 's'
    assert unit == 's'
    assert result_df['t_startup'].iloc[0] == 3600


# ============================================================================
# TEST COLOR SCALE GENERATION
# ============================================================================

def test_get_discrete_colorscale_small():
    """
    Test generating discrete colorscale with small number of chunks.
    Expected: Should return colorscale list with correct structure.
    """
    n_chunks = 4
    result = get_discrete_colorscale(n_chunks)
    
    # Expected: List with 2*n_chunks entries (start and end for each chunk)
    assert isinstance(result, list)
    assert len(result) == 2 * n_chunks
    # Each entry should be [fraction, color]
    for entry in result:
        assert len(entry) == 2
        assert 0 <= entry[0] <= 1  # Fraction
        assert isinstance(entry[1], str)  # Color hex string


def test_get_discrete_colorscale_large():
    """
    Test generating discrete colorscale with large number of chunks.
    Expected: Should use matplotlib colormap for many chunks.
    """
    n_chunks = 10
    result = get_discrete_colorscale(n_chunks)
    
    # Expected: List with 2*n_chunks entries
    assert len(result) == 2 * n_chunks
    # Colors should be hex strings
    for entry in result:
        assert entry[1].startswith('#')


def test_get_discrete_colorscale_values_monotonic():
    """
    Test that colorscale fractions are monotonically increasing.
    Expected: Fraction values should increase from 0 to 1.
    """
    result = get_discrete_colorscale(6)
    
    fractions = [entry[0] for entry in result]
    
    # Expected: Monotonically increasing from 0 to 1
    assert fractions[0] == 0
    assert fractions[-1] == 1
    for i in range(len(fractions) - 1):
        assert fractions[i] <= fractions[i + 1]


# ============================================================================
# TEST PREPARE DATAFRAMES
# ============================================================================

def test_prepare_dataframes_single_file(temp_dir):
    """
    Test preparing dataframes from single file with multiple targets.
    Expected: Should return nested dict with filtered dataframes.
    """
    # Create sample h5 file
    h5_path = temp_dir / "data.h5"
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('V_plasma', data=[100, 150, 200, 250])
        f.create_dataset('t_startup', data=[1e6, 2e6, 3e6, np.nan])
        f.create_dataset('unrealized_gains', data=[1e8, 2e8, np.nan, 4e8])
    
    target_variables = ['t_startup', 'unrealized_gains']
    filters = {
        't_startup': {'min': 0, 'max': 5e6},
        'unrealized_gains': {'min': 0, 'max': 5e8}
    }
    
    result = prepare_dataframes([h5_path], target_variables, filters)
    
    # Expected: Nested dict with file -> target -> dataframe
    assert h5_path in result
    assert 't_startup' in result[h5_path]
    assert 'unrealized_gains' in result[h5_path]
    
    # Expected: t_startup dataframe has finite values only
    df_startup = result[h5_path]['t_startup']
    assert df_startup['t_startup'].isna().sum() == 0
    
    # Expected: unrealized_gains dataframe has finite values only
    df_gains = result[h5_path]['unrealized_gains']
    assert df_gains['unrealized_gains'].isna().sum() == 0


def test_prepare_dataframes_empty_filters(temp_dir):
    """
    Test prepare_dataframes with no filters.
    Expected: Should only filter out non-finite target values.
    """
    h5_path = temp_dir / "data.h5"
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('V_plasma', data=[100, 150, 200])
        f.create_dataset('t_startup', data=[1e6, 2e6, 3e6])
    
    result = prepare_dataframes([h5_path], ['t_startup'], {})
    
    # Expected: All finite values retained
    assert len(result[h5_path]['t_startup']) == 3
