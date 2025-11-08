"""
Test parameter filtering functionality.
"""

import numpy as np
import pytest
from ddstartup.utils.filters import (
    ParameterFilter,
    FilterError,
    apply_filter_to_combinations
)


def test_simple_comparison():
    """Test simple comparison filters."""
    param_names = ['P_aux', 'T_i']
    
    # Less than
    f = ParameterFilter("P_aux < 1e6", param_names)
    assert f.evaluate({'P_aux': 5e5, 'T_i': 10}) == True
    assert f.evaluate({'P_aux': 2e6, 'T_i': 10}) == False
    
    # Greater than
    f = ParameterFilter("T_i > 15", param_names)
    assert f.evaluate({'P_aux': 1e6, 'T_i': 20}) == True
    assert f.evaluate({'P_aux': 1e6, 'T_i': 10}) == False


def test_logical_operators():
    """Test AND/OR logical operators."""
    param_names = ['P_aux', 'T_i', 'n_tot']
    
    # AND
    f = ParameterFilter("P_aux < 1e6 and T_i > 10", param_names)
    assert f.evaluate({'P_aux': 5e5, 'T_i': 15, 'n_tot': 1e20}) == True
    assert f.evaluate({'P_aux': 2e6, 'T_i': 15, 'n_tot': 1e20}) == False
    assert f.evaluate({'P_aux': 5e5, 'T_i': 5, 'n_tot': 1e20}) == False
    
    # OR
    f = ParameterFilter("P_aux > 1e6 or T_i > 20", param_names)
    assert f.evaluate({'P_aux': 2e6, 'T_i': 10, 'n_tot': 1e20}) == True
    assert f.evaluate({'P_aux': 5e5, 'T_i': 25, 'n_tot': 1e20}) == True
    assert f.evaluate({'P_aux': 5e5, 'T_i': 15, 'n_tot': 1e20}) == False


def test_parameter_comparison():
    """Test comparing two parameters."""
    param_names = ['P_aux', 'P_aux_DT_eq']
    
    f = ParameterFilter("P_aux_DT_eq < P_aux", param_names)
    assert f.evaluate({'P_aux': 1e6, 'P_aux_DT_eq': 5e5}) == True
    assert f.evaluate({'P_aux': 1e6, 'P_aux_DT_eq': 2e6}) == False


def test_complex_expression():
    """Test complex nested expressions."""
    param_names = ['P_aux', 'T_i', 'n_tot', 'V_plasma']
    
    f = ParameterFilter(
        "(P_aux > 1e6 and T_i > 15) or (n_tot > 5e20 and V_plasma < 100)",
        param_names
    )
    
    # First condition true
    assert f.evaluate({'P_aux': 2e6, 'T_i': 20, 'n_tot': 1e20, 'V_plasma': 200}) == True
    
    # Second condition true
    assert f.evaluate({'P_aux': 5e5, 'T_i': 10, 'n_tot': 6e20, 'V_plasma': 50}) == True
    
    # Neither true
    assert f.evaluate({'P_aux': 5e5, 'T_i': 10, 'n_tot': 1e20, 'V_plasma': 200}) == False


def test_invalid_syntax():
    """Test that invalid syntax raises FilterError."""
    param_names = ['P_aux']
    
    with pytest.raises(FilterError):
        ParameterFilter("P_aux < ", param_names)
    
    with pytest.raises(FilterError):
        ParameterFilter("P_aux +", param_names)


def test_unknown_parameter():
    """Test that unknown parameters raise FilterError."""
    param_names = ['P_aux', 'T_i']
    
    with pytest.raises(FilterError, match="Unknown parameter"):
        ParameterFilter("P_fusion > 1e6", param_names)


def test_apply_filter_to_combinations():
    """Test applying filter to full parameter grid."""
    # Create simple parameter grid
    input_data = {
        'P_aux': np.array([1e5, 5e5, 1e6, 2e6]),
        'T_i': np.array([10, 15, 20])
    }
    
    # Filter: P_aux < 1e6 and T_i > 12
    # Should keep: P_aux=[1e5, 5e5] x T_i=[15, 20] = 4 combinations
    filtered, indices, total = apply_filter_to_combinations(
        input_data, 
        "P_aux < 1e6 and T_i > 12",
        verbose=False
    )
    
    assert total == 12  # 4 * 3 = 12 total combinations
    assert len(indices) == 4  # 4 valid combinations
    assert len(filtered['P_aux']) == 4
    assert len(filtered['T_i']) == 4


def test_no_filter():
    """Test that no filter returns all combinations."""
    input_data = {
        'P_aux': np.array([1e5, 1e6]),
        'T_i': np.array([10, 20])
    }
    
    filtered, indices, total = apply_filter_to_combinations(
        input_data, 
        None,
        verbose=False
    )
    
    assert total == 4
    assert len(indices) == 4
    assert np.array_equal(indices, np.arange(4))


def test_filter_excludes_all():
    """Test filter that excludes all combinations."""
    input_data = {
        'P_aux': np.array([1e5, 1e6]),
        'T_i': np.array([10, 20])
    }
    
    # Impossible condition
    filtered, indices, total = apply_filter_to_combinations(
        input_data,
        "P_aux > 1e7",
        verbose=False
    )
    
    assert total == 4
    assert len(indices) == 0  # No valid combinations


def test_arithmetic_operations():
    """Test arithmetic operations in filters."""
    param_names = ['P_aux', 'T_i']
    
    # Addition
    f = ParameterFilter("P_aux + T_i > 1e6", param_names)
    assert f.evaluate({'P_aux': 9e5, 'T_i': 2e5}) == True
    
    # Multiplication
    f = ParameterFilter("P_aux * 2 > 1e6", param_names)
    assert f.evaluate({'P_aux': 6e5, 'T_i': 10}) == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
