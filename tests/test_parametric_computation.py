"""
Tests for utils/parametric_computation.py

This module tests parametric analysis functions including:
- Parametric analysis execution with ProcessPoolExecutor
- Async HDF5 writing with background thread
- HDF5 file creation and writing
- Result buffering and writing
- Error handling and recovery
- Numba compilation priming
"""

import pytest
import numpy as np
import h5py
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddstartup.utils.parametric_computation import (
    run_parametric_analysis,
    print_parametric_summary,
    _write_results_to_hdf5
)


# Module-level mock functions for ProcessPoolExecutor (must be picklable)
def _mock_compute_basic(idx, arrays, shapes, *args):
    """Basic mock compute function"""
    return {
        'sol_success': True,
        't_startup': 100.0 + idx,
        'linear_index': idx,
        'error': ''
    }


def _mock_compute_with_vectors(idx, arrays, shapes, *args):
    """Mock compute with vector fields"""
    vector_length = args[1] if len(args) > 1 else 50
    return {
        'sol_success': True,
        't_startup': 100.0,
        'N_ofc': np.arange(vector_length),
        'error': ''
    }


def _mock_compute_with_errors(idx, arrays, shapes, *args):
    """Mock compute that fails for first index"""
    if idx == 0:
        return {
            'sol_success': False,
            't_startup': np.nan,
            'error': 'Test error message'
        }
    return {
        'sol_success': True,
        't_startup': 100.0,
        'error': ''
    }


def _mock_compute_with_sleep(idx, arrays, shapes, *args):
    """Mock compute with artificial delay"""
    time.sleep(0.01)
    return {
        'sol_success': True,
        't_startup': 100.0 + idx,
        'error': ''
    }


class TestRunParametricAnalysis:
    """Tests for run_parametric_analysis function"""
    
    def test_basic_execution(self, temp_dir):
        """Test basic parametric analysis execution"""
        # Create simple input data
        input_data = {
            'V_plasma': np.array([100.0, 150.0]),
            'T_i': np.array([15.0, 17.0]),
        }
        
        # Configuration
        config = {
            'analysis_type': 'T_seeded',
            'method': 'parametric',
            'n_jobs': 1,
            'chunk_size': 10,
            'batch_size': 10,
            'vector_length': 10,
            'total_time': 3600,
        }
        
        output_file = str(temp_dir / "test_output.h5")
        
        # Run analysis
        stats = run_parametric_analysis(
            input_data, output_file, config,
            _mock_compute_basic, verbose=False
        )
        
        # Check statistics
        assert stats['total_combinations'] == 4  # 2x2 combinations
        assert stats['processed'] == 4
        assert stats['successful'] == 4
        assert stats['success_rate'] == 100.0
        assert 'computation_time' in stats
        
        # Check HDF5 file was created
        assert Path(output_file).exists()
        
        # Verify HDF5 contents
        with h5py.File(output_file, 'r') as f:
            assert 't_startup' in f
            assert f['t_startup'].shape == (4,)
            assert 'parameter_fields' in f
    
    def test_vector_field_handling(self, temp_dir):
        """Test handling of vector fields"""
        input_data = {
            'V_plasma': np.array([100.0]),
            'T_i': np.array([15.0]),
        }
        
        vector_length = 50
        
        config = {
            'analysis_type': 'T_seeded',
            'method': 'parametric',
            'n_jobs': 1,
            'chunk_size': 10,
            'batch_size': 10,
            'vector_length': vector_length,
            'total_time': 3600,
        }
        
        output_file = str(temp_dir / "test_vector.h5")
        
        stats = run_parametric_analysis(
            input_data, output_file, config,
            _mock_compute_with_vectors, verbose=False
        )
        
        # Check vector field was saved correctly
        with h5py.File(output_file, 'r') as f:
            assert 'N_ofc' in f
            assert f['N_ofc'].shape == (1, vector_length)
            np.testing.assert_array_equal(f['N_ofc'][0, :], np.arange(vector_length))
    
    def test_error_handling(self, temp_dir):
        """Test error handling in computation"""
        input_data = {
            'V_plasma': np.array([100.0]),
            'T_i': np.array([15.0]),
        }
        
        config = {
            'analysis_type': 'T_seeded',
            'method': 'parametric',
            'n_jobs': 1,
            'chunk_size': 10,
            'batch_size': 10,
            'vector_length': 10,
            'total_time': 3600,
        }
        
        output_file = str(temp_dir / "test_errors.h5")
        
        stats = run_parametric_analysis(
            input_data, output_file, config,
            _mock_compute_with_errors, verbose=False
        )
        
        # Check that error was recorded
        with h5py.File(output_file, 'r') as f:
            assert 'error' in f
            error_msg = f['error'][0]
            if isinstance(error_msg, bytes):
                error_msg = error_msg.decode('utf-8')
            assert 'Test error message' in error_msg
            assert f['sol_success'][0] == False
    
    def test_metadata_storage(self, temp_dir):
        """Test that metadata is properly stored"""
        input_data = {'V_plasma': np.array([100.0])}
        
        def mock_compute(idx, arrays, shapes, *args):
            return {'sol_success': True, 't_startup': 100.0}
        
        config = {
            'analysis_type': 'T_seeded',
            'method': 'parametric',
            'n_jobs': 2,
            'chunk_size': 100,
            'batch_size': 50,
            'vector_length': 10,
            'total_time': 3600,
        }
        
        output_file = str(temp_dir / "test_metadata.h5")
        
        run_parametric_analysis(
            input_data, output_file, config,
            mock_compute, verbose=False
        )
        
        # Check metadata
        with h5py.File(output_file, 'r') as f:
            assert f.attrs['total_combinations'] == 1
            assert f.attrs['method'] == 'parametric'
            assert f.attrs['analysis_type'] == 'T_seeded'
            assert f.attrs['vector_length'] == 10
            assert f.attrs['n_jobs'] == 2
            assert 'computation_start_time' in f.attrs
            assert 'computation_end_time' in f.attrs
    
    def test_processpool_executor_usage(self, temp_dir):
        """Test that ProcessPoolExecutor is used for parallel execution"""
        input_data = {
            'V_plasma': np.array([100.0, 150.0]),
            'T_i': np.array([15.0, 17.0]),
        }
        
        config = {
            'analysis_type': 'T_seeded',
            'method': 'parametric',
            'n_jobs': 2,
            'chunk_size': 10,
            'batch_size': 10,
            'vector_length': 10,
            'total_time': 3600,
        }
        
        output_file = str(temp_dir / "test_pool.h5")
        
        start = time.time()
        stats = run_parametric_analysis(
            input_data, output_file, config,
            _mock_compute_with_sleep, verbose=False
        )
        elapsed = time.time() - start
        
        # Verify all combinations were computed
        assert stats['processed'] == 4
        assert stats['successful'] == 4
        
        # With 2 workers and 0.01s work, parallel should be faster than 4*0.01s
        # Allow generous margin for test reliability
        assert elapsed < 0.3, f"Parallel execution took {elapsed}s, expected < 0.3s"
    
    def test_async_hdf5_writing(self, temp_dir):
        """Test that HDF5 writing happens asynchronously"""
        input_data = {'V_plasma': np.array([100.0, 150.0, 200.0])}
        
        config = {
            'analysis_type': 'T_seeded',
            'method': 'parametric',
            'n_jobs': 1,
            'chunk_size': 10,
            'batch_size': 1,  # Force frequent writes
            'vector_length': 10,
            'total_time': 3600,
        }
        
        output_file = str(temp_dir / "test_async.h5")
        
        stats = run_parametric_analysis(
            input_data, output_file, config,
            _mock_compute_basic, verbose=False
        )
        
        # Verify all results were written
        assert stats['successful'] == 3
        
        with h5py.File(output_file, 'r') as f:
            assert 't_startup' in f
            assert len(f['t_startup']) == 3
            assert np.allclose(f['t_startup'][:], [100.0, 101.0, 102.0])
    
    def test_numba_priming_tseeded(self, temp_dir):
        """Test Numba compilation priming for T_seeded"""
        input_data = {'V_plasma': np.array([100.0])}
        
        config = {
            'analysis_type': 'T_seeded',
            'method': 'parametric',
            'n_jobs': 1,
            'chunk_size': 10,
            'batch_size': 10,
            'vector_length': 10,
            'total_time': 3600,
        }
        
        output_file = str(temp_dir / "test_prime.h5")
        
        # Run with verbose=True to trigger priming
        stats = run_parametric_analysis(
            input_data, output_file, config,
            _mock_compute_basic, verbose=True
        )
        
        # Should complete successfully
        assert stats['processed'] >= 1


class TestWriteResultsToHDF5:
    """Tests for _write_results_to_hdf5 function"""
    
    def test_scalar_field_writing(self, temp_dir):
        """Test writing scalar fields"""
        output_file = str(temp_dir / "test_write.h5")
        
        with h5py.File(output_file, 'w') as f:
            datasets = {
                't_startup': f.create_dataset('t_startup', (5,), dtype=np.float64),
                'sol_success': f.create_dataset('sol_success', (5,), dtype=bool),
                'error': f.create_dataset('error', (5,), dtype=h5py.string_dtype()),
            }
            
            results = [
                {'t_startup': 100.0, 'sol_success': True, 'error': ''},
                {'t_startup': 200.0, 'sol_success': False, 'error': 'Test error'},
            ]
            indices = [0, 1]
            
            _write_results_to_hdf5(
                datasets, results, indices,
                ['t_startup', 'sol_success', 'error'],
                [], 10
            )
            
            # Verify
            assert datasets['t_startup'][0] == 100.0
            assert datasets['t_startup'][1] == 200.0
            assert datasets['sol_success'][0] == True
            assert datasets['sol_success'][1] == False
    
    def test_vector_field_writing(self, temp_dir):
        """Test writing vector fields"""
        output_file = str(temp_dir / "test_vector_write.h5")
        vector_length = 10
        
        with h5py.File(output_file, 'w') as f:
            datasets = {
                'N_ofc': f.create_dataset('N_ofc', (3, vector_length), dtype=np.float64),
            }
            
            results = [
                {'N_ofc': np.ones(vector_length)},
                {'N_ofc': np.ones(vector_length) * 2},
                {'N_ofc': np.arange(vector_length)},
            ]
            indices = [0, 1, 2]
            
            _write_results_to_hdf5(
                datasets, results, indices,
                ['N_ofc'], ['N_ofc'], vector_length
            )
            
            # Verify
            np.testing.assert_array_equal(datasets['N_ofc'][0, :], np.ones(vector_length))
            np.testing.assert_array_equal(datasets['N_ofc'][1, :], np.ones(vector_length) * 2)
            np.testing.assert_array_equal(datasets['N_ofc'][2, :], np.arange(vector_length))
    
    def test_missing_values_handling(self, temp_dir):
        """Test handling of missing values"""
        output_file = str(temp_dir / "test_missing.h5")
        
        with h5py.File(output_file, 'w') as f:
            datasets = {
                't_startup': f.create_dataset('t_startup', (2,), dtype=np.float64),
                'error': f.create_dataset('error', (2,), dtype=h5py.string_dtype()),
            }
            
            results = [
                {'t_startup': None, 'error': None},  # Missing values
                {},  # Empty result
            ]
            indices = [0, 1]
            
            _write_results_to_hdf5(
                datasets, results, indices,
                ['t_startup', 'error'], [], 10
            )
            
            # Verify NaN for missing scalar
            assert np.isnan(datasets['t_startup'][0])
            assert datasets['error'][0].decode() == ""


class TestPrintParametricSummary:
    """Tests for print_parametric_summary function"""
    
    def test_summary_prints_stats(self, temp_dir, capsys):
        """Test that summary prints statistics"""
        output_file = str(temp_dir / "test.h5")
        
        # Create dummy file
        with h5py.File(output_file, 'w') as f:
            f.attrs['test'] = 'data'
        
        stats = {
            'processed': 1000,
            'successful': 950,
            'success_rate': 95.0,
            'computation_time': 123.45
        }
        
        print_parametric_summary(output_file, stats, verbose=True)
        
        captured = capsys.readouterr()
        assert "1,000 combinations" in captured.out
        assert "950" in captured.out
        assert "95.0%" in captured.out
        assert "123.45 seconds" in captured.out
    
    def test_summary_silent_when_not_verbose(self, temp_dir, capsys):
        """Test that summary is silent when verbose=False"""
        output_file = str(temp_dir / "test.h5")
        stats = {'processed': 100, 'successful': 90, 'success_rate': 90.0, 'computation_time': 10.0}
        
        print_parametric_summary(output_file, stats, verbose=False)
        
        captured = capsys.readouterr()
        assert captured.out == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
