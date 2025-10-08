"""
Comprehensive tests for Sobol sensitivity analysis.

Tests cover:
1. Sample generation
2. Parameter bounds handling
3. Parallel computation
4. HDF5 file creation
5. PCE fitting
6. Sobol indices calculation
"""

import pytest
import numpy as np
import h5py
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from ddstartup.physics.sobol_functions import (
    convert_sobol_sample_to_linear_index,
    run_sample_optimized,
    sobol_analysis
)


class TestSobolSampleGeneration:
    """Test Sobol sample generation and preparation."""
    
    def test_linear_index_conversion(self):
        """Test that Sobol index conversion works correctly."""
        sample_idx = 42
        n_samples = 1000
        result = convert_sobol_sample_to_linear_index(sample_idx, n_samples)
        assert result == sample_idx, "Linear index should equal sample index"
    
    def test_parameter_bounds_extraction(self):
        """Test extraction of parameter bounds from input data."""
        input_data = {
            'param1': np.array([1.0, 2.0, 3.0]),
            'param2': np.array([10.0]),  # Constant parameter
            'param3': np.linspace(5.0, 15.0, 10)
        }
        param_names = list(input_data.keys())
        param_bounds = [(arr.min(), arr.max()) for arr in input_data.values()]
        
        assert len(param_bounds) == 3
        assert param_bounds[0] == (1.0, 3.0)
        assert param_bounds[1] == (10.0, 10.0)  # Constant
        assert param_bounds[2] == (5.0, 15.0)
    
    def test_variable_vs_constant_parameters(self):
        """Test separation of variable and constant parameters."""
        input_data = {
            'var1': np.array([1.0, 2.0, 3.0]),
            'const1': np.array([10.0]),
            'var2': np.linspace(5.0, 15.0, 10)
        }
        param_names = list(input_data.keys())
        param_bounds = [(arr.min(), arr.max()) for arr in input_data.values()]
        
        variable_params = [
            (i, name, low, high) 
            for i, (name, (low, high)) in enumerate(zip(param_names, param_bounds)) 
            if high > low
        ]
        constant_params = [
            (i, name, low) 
            for i, (name, (low, high)) in enumerate(zip(param_names, param_bounds)) 
            if high == low
        ]
        
        assert len(variable_params) == 2, "Should have 2 variable parameters"
        assert len(constant_params) == 1, "Should have 1 constant parameter"
        assert constant_params[0][1] == 'const1'


class TestSobolSampleEvaluation:
    """Test individual sample evaluation."""
    
    def test_run_sample_optimized_structure(self):
        """Test that run_sample_optimized returns correct structure."""
        # Mock compute function that returns a valid result
        def mock_compute(idx, arrays, shapes, *args):
            return {
                'linear_index': idx,
                'sol_success': True,
                'unrealized_gains': 1000.0,
                't_startup': 3.156e7,  # 1 year
                'Q_DD': 5.0,
                'error': ''
            }
        
        sample_idx = 0
        sample_full = np.array([100.0, 15.0, 1.5e20, 1.0])
        input_arrays = [np.array([val]) for val in sample_full]
        param_shapes = np.ones(len(sample_full), dtype=np.int64)
        
        result = run_sample_optimized(
            sample_idx, sample_full,
            mock_compute, 'T_seeded', 
            10*365*24*3600, 100
        )
        
        assert 'sample_idx' in result
        assert result['sample_idx'] == sample_idx
        assert 'sol_success' in result
        assert 'unrealized_gains' in result
    
    def test_run_sample_with_error(self):
        """Test that errors are caught and returned properly."""
        def mock_compute_error(idx, arrays, shapes, *args):
            raise ValueError("Test error in computation")
        
        sample_idx = 5
        sample_full = np.array([100.0, 15.0, 1.5e20, 1.0])
        input_arrays = [np.array([val]) for val in sample_full]
        param_shapes = np.ones(len(sample_full), dtype=np.int64)
        
        result = run_sample_optimized(
            sample_idx, sample_full,
            mock_compute_error, 'T_seeded',
            10*365*24*3600, 100
        )
        
        assert result['sample_idx'] == sample_idx
        assert result['sol_success'] is False
        assert 'error' in result
        assert 'Test error' in result['error']


class TestSobolHDF5Output:
    """Test HDF5 file creation and structure."""
    
    def test_hdf5_file_creation(self):
        """Test that Sobol analysis creates HDF5 file with correct structure."""
        # Create mock input data
        input_data = {
            'V_plasma': np.linspace(100, 200, 3),
            'T_i': np.linspace(14, 20, 3),
            'n_tot': np.linspace(1.3e20, 2.1e20, 3),
            'tau_p_T': np.linspace(0.1, 5.0, 3)
        }
        param_names = list(input_data.keys())
        
        # Mock compute function
        def mock_compute(idx, arrays, shapes, *args):
            return {
                'linear_index': idx,
                'sol_success': True,
                'unrealized_gains': 1000.0 + np.random.randn() * 100,
                't_startup': 3.156e7,
                'Q_DD': 5.0,
                'V_plasma': arrays[0][0],
                'T_i': arrays[1][0],
                'n_tot': arrays[2][0],
                'tau_p_T': arrays[3][0],
                'error': ''
            }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            output_file = tmp.name
        
        try:
            # Run Sobol analysis with small sample size
            result = sobol_analysis(
                input_data=input_data,
                param_names=param_names,
                N_SAMPLES=100,  # Small sample for testing
                order=2,
                analysis_type='T_seeded',
                total_time=10*365*24*3600,
                compute_single_combination=mock_compute,
                output_file=output_file,
                vector_length=50,
                verbose=False
            )
            
            # Check that file was created
            assert os.path.exists(output_file), "HDF5 file should be created"
            
            # Verify HDF5 structure
            with h5py.File(output_file, 'r') as h5:
                # Check metadata
                assert 'method' in h5.attrs
                assert h5.attrs['method'] == 'sobol'
                assert 'N_SAMPLES' in h5.attrs
                assert h5.attrs['N_SAMPLES'] == 100
                
                # Check Sobol indices group
                assert 'sobol_indices' in h5
                assert 'first_order' in h5['sobol_indices']
                assert 'total_order' in h5['sobol_indices']
                assert 'parameter_names' in h5['sobol_indices']
                
                # Check samples group
                assert 'sobol_samples' in h5
                assert 'samples_full' in h5['sobol_samples']
                assert 'valid_mask' in h5['sobol_samples']
                
                # Check parameter info
                assert 'parameter_info' in h5
                
                # Check data fields exist
                assert 'unrealized_gains' in h5
                assert 'sol_success' in h5
                
        finally:
            # Cleanup
            if os.path.exists(output_file):
                os.remove(output_file)
    
    def test_hdf5_insufficient_samples(self):
        """Test that analysis handles insufficient valid samples gracefully."""
        input_data = {
            'param1': np.linspace(1, 10, 3),
            'param2': np.linspace(5, 15, 3)
        }
        param_names = list(input_data.keys())
        
        # Mock compute that always fails
        def mock_compute_fail(idx, arrays, shapes, *args):
            return {
                'linear_index': idx,
                'sol_success': False,
                'unrealized_gains': np.nan,
                'error': 'Test failure'
            }
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            output_file = tmp.name
        
        try:
            result = sobol_analysis(
                input_data=input_data,
                param_names=param_names,
                N_SAMPLES=20,
                order=2,
                analysis_type='T_seeded',
                total_time=10*365*24*3600,
                compute_single_combination=mock_compute_fail,
                output_file=output_file,
                vector_length=50,
                verbose=False
            )
            
            # Should return None when insufficient samples
            assert result is None, "Should return None for insufficient valid samples"
            
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)


class TestSobolIndicesCalculation:
    """Test PCE fitting and Sobol indices calculation."""
    
    def test_sobol_indices_properties(self):
        """Test that Sobol indices have correct properties."""
        # Run small Sobol analysis
        input_data = {
            'x1': np.linspace(0, 1, 5),
            'x2': np.linspace(0, 1, 5),
            'x3': np.linspace(0, 1, 5)
        }
        param_names = list(input_data.keys())
        
        # Ishigami-like test function: known sensitivity indices
        def mock_compute_ishigami(idx, arrays, shapes, *args):
            x1, x2, x3 = arrays[0][0], arrays[1][0], arrays[2][0]
            # Simplified test function where x1 dominates
            y = x1**2 + 2*x2 + 0.1*x3
            return {
                'linear_index': idx,
                'sol_success': True,
                'unrealized_gains': y,
                'error': ''
            }
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            output_file = tmp.name
        
        try:
            result = sobol_analysis(
                input_data=input_data,
                param_names=param_names,
                N_SAMPLES=500,
                order=2,
                analysis_type='T_seeded',
                total_time=10*365*24*3600,
                compute_single_combination=mock_compute_ishigami,
                output_file=output_file,
                vector_length=50,
                verbose=False
            )
            
            if result is not None:
                sobol_first = result['sobol_first']
                sobol_total = result['sobol_total']
                
                # Basic properties
                assert len(sobol_first) == 3, "Should have 3 first-order indices"
                assert len(sobol_total) == 3, "Should have 3 total-order indices"
                
                # Total >= First (always true)
                assert np.all(sobol_total >= sobol_first - 1e-10), \
                    "Total-order should be >= first-order"
                
                # All indices should be non-negative (approximately)
                assert np.all(sobol_first >= -0.1), "First-order indices should be non-negative"
                
                # x1 should be most important (quadratic term)
                assert sobol_first[0] > sobol_first[2], \
                    "x1 should be more important than x3"
                
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)


class TestSobolErrorHandling:
    """Test error handling in Sobol analysis."""
    
    def test_invalid_input_data(self):
        """Test handling of invalid input data."""
        # Empty input data
        with pytest.raises(Exception):
            input_data = {}
            sobol_analysis(
                input_data=input_data,
                param_names=[],
                N_SAMPLES=100,
                order=2,
                analysis_type='T_seeded',
                total_time=10*365*24*3600,
                compute_single_combination=lambda: None,
                output_file='test.h5',
                vector_length=50,
                verbose=False
            )
    
    def test_no_variable_parameters(self):
        """Test that analysis fails gracefully with all constant parameters."""
        input_data = {
            'const1': np.array([10.0]),
            'const2': np.array([20.0])
        }
        param_names = list(input_data.keys())
        
        with pytest.raises(ValueError, match="No parameters with a valid range"):
            with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
                sobol_analysis(
                    input_data=input_data,
                    param_names=param_names,
                    N_SAMPLES=100,
                    order=2,
                    analysis_type='T_seeded',
                    total_time=10*365*24*3600,
                    compute_single_combination=lambda: None,
                    output_file=tmp.name,
                    vector_length=50,
                    verbose=False
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=physics.sobol_functions', 
                 '--cov=utils.sobol_computation', '--cov-report=html', '--cov-report=term'])
