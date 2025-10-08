"""
Tests for utils/sobol_computation.py

This module tests Sobol sensitivity analysis functions including:
- Sobol analysis execution
- Integration with physics module
- Statistics tracking
"""

import pytest
import numpy as np
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddstartup.utils.sobol_computation import (
    run_sobol_analysis,
    print_sobol_summary
)


class TestRunSobolAnalysis:
    """Tests for run_sobol_analysis function"""
    
    def test_basic_execution(self, temp_dir):
        """Test basic Sobol analysis execution"""
        # Mock the physics sobol function
        with patch('ddstartup.physics.sobol_functions.sobol_analysis') as mock_sobol:
            input_data = {
                'V_plasma': np.array([100.0, 150.0]),
                'T_i': np.array([15.0, 17.0]),
            }
            
            def mock_compute(idx, arrays, shapes):
                return {'sol_success': True, 't_startup': 100.0}
            
            config = {
                'analysis_type': 'T_seeded',
                'method': 'sobol',
                'N_SAMPLES': 1000,
                'order': 2,
                'total_time': 3600,
                'vector_length': 50,
            }
            
            output_file = str(temp_dir / "sobol_output.txt")
            
            # Run analysis
            stats = run_sobol_analysis(
                input_data, output_file, config,
                mock_compute, verbose=False
            )
            
            # Check that physics sobol function was called
            assert mock_sobol.called
            assert mock_sobol.call_count == 1
            
            # Check call arguments
            call_args = mock_sobol.call_args
            assert call_args[1]['N_SAMPLES'] == 1000
            assert call_args[1]['order'] == 2
            assert call_args[1]['analysis_type'] == 'T_seeded'
            
            # Check statistics
            assert stats['n_samples'] == 1000
            assert stats['n_parameters'] == 2
            assert stats['order'] == 2
            assert stats['analysis_type'] == 'T_seeded'
            assert 'computation_time' in stats
    
    def test_import_error_handling(self, temp_dir):
        """Test handling of missing physics module"""
        # Mock import to fail
        with patch('ddstartup.physics.sobol_functions.sobol_analysis', side_effect=ImportError("Module not found")):
            input_data = {'V_plasma': np.array([100.0])}
            config = {
                'analysis_type': 'T_seeded',
                'method': 'sobol',
                'N_SAMPLES': 100,
                'order': 2,
                'total_time': 3600,
                'vector_length': 50,
            }
            
            with pytest.raises(ImportError):
                run_sobol_analysis(
                    input_data, "output.txt", config,
                    lambda: None, verbose=False
                )
    
    def test_computation_error_handling(self, temp_dir):
        """Test handling of computation errors"""
        with patch('ddstartup.physics.sobol_functions.sobol_analysis',
                  side_effect=RuntimeError("Computation failed")):
            input_data = {'V_plasma': np.array([100.0])}
            config = {
                'analysis_type': 'T_seeded',
                'method': 'sobol',
                'N_SAMPLES': 100,
                'order': 2,
                'total_time': 3600,
                'vector_length': 50,
            }
            
            with pytest.raises(RuntimeError):
                run_sobol_analysis(
                    input_data, "output.txt", config,
                    lambda: None, verbose=False
                )
    
    def test_different_sample_sizes(self, temp_dir):
        """Test with different N_SAMPLES values"""
        sample_sizes = [100, 1000, 10000]
        
        for n_samples in sample_sizes:
            with patch('ddstartup.physics.sobol_functions.sobol_analysis') as mock_sobol:
                input_data = {'V_plasma': np.array([100.0])}
                config = {
                    'analysis_type': 'lump',
                    'method': 'sobol',
                    'N_SAMPLES': n_samples,
                    'order': 3,
                    'total_time': 3600,
                    'vector_length': 50,
                }
                
                stats = run_sobol_analysis(
                    input_data, "output.txt", config,
                    lambda: None, verbose=False
                )
                
                assert stats['n_samples'] == n_samples
                # Verify correct N_SAMPLES passed to physics function
                call_args = mock_sobol.call_args
                assert call_args[1]['N_SAMPLES'] == n_samples
    
    def test_different_orders(self, temp_dir):
        """Test with different Sobol orders"""
        for order in [1, 2, 3]:
            with patch('ddstartup.physics.sobol_functions.sobol_analysis') as mock_sobol:
                input_data = {'V_plasma': np.array([100.0])}
                config = {
                    'analysis_type': 'T_seeded',
                    'method': 'sobol',
                    'N_SAMPLES': 100,
                    'order': order,
                    'total_time': 3600,
                    'vector_length': 50,
                }
                
                stats = run_sobol_analysis(
                    input_data, "output.txt", config,
                    lambda: None, verbose=False
                )
                
                assert stats['order'] == order
                call_args = mock_sobol.call_args
                assert call_args[1]['order'] == order
    
    def test_verbose_output(self, temp_dir, capsys):
        """Test verbose output printing"""
        with patch('ddstartup.physics.sobol_functions.sobol_analysis'):
            input_data = {'V_plasma': np.array([100.0]), 'T_i': np.array([15.0])}
            config = {
                'analysis_type': 'T_seeded',
                'method': 'sobol',
                'N_SAMPLES': 1000,
                'order': 2,
                'total_time': 3600,
                'vector_length': 50,
            }
            
            run_sobol_analysis(
                input_data, "output.txt", config,
                lambda: None, verbose=True
            )
            
            captured = capsys.readouterr()
            assert "STARTING SOBOL SENSITIVITY ANALYSIS" in captured.out
            assert "Parameters: 2" in captured.out
            assert "Samples: 1,000" in captured.out
            assert "Order: 2" in captured.out
            assert "SOBOL ANALYSIS COMPLETE" in captured.out
    
    def test_analysis_type_passed_correctly(self, temp_dir):
        """Test that analysis_type is passed correctly to physics module"""
        for analysis_type in ['T_seeded', 'lump']:
            with patch('ddstartup.physics.sobol_functions.sobol_analysis') as mock_sobol:
                input_data = {'V_plasma': np.array([100.0])}
                config = {
                    'analysis_type': analysis_type,
                    'method': 'sobol',
                    'N_SAMPLES': 100,
                    'order': 2,
                    'total_time': 3600,
                    'vector_length': 50,
                }
                
                run_sobol_analysis(
                    input_data, "output.txt", config,
                    lambda: None, verbose=False
                )
                
                call_args = mock_sobol.call_args
                assert call_args[1]['analysis_type'] == analysis_type
    
    def test_integration_with_mock_compute(self, temp_dir):
        """Test full integration of run_sobol_analysis with mock compute function."""
        input_data = {
            'V_plasma': np.linspace(100, 200, 2),
            'T_i': np.linspace(14, 20, 2)
        }
        
        # Mock compute function
        def mock_compute(idx, arrays, shapes, *args):
            return {
                'linear_index': idx,
                'sol_success': True,
                'unrealized_profits': 1000.0,
                't_startup': 3.156e7,
                'error': ''
            }
        
        config = {
            'analysis_type': 'T_seeded',
            'N_SAMPLES': 50,
            'order': 2,
            'total_time': 10*365*24*3600,
            'vector_length': 50,
            'n_jobs': 2,
            'method': 'sobol'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            output_file = tmp.name
        
        try:
            stats = run_sobol_analysis(
                input_data=input_data,
                output_file=output_file,
                config=config,
                compute_function=mock_compute,
                verbose=False
            )
            
            # Check stats returned
            if stats is not None:
                assert 'n_samples' in stats
                assert 'computation_time' in stats
                assert stats['n_samples'] == 50
            
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)


class TestPrintSobolSummary:
    """Tests for print_sobol_summary function"""
    
    def test_summary_prints_stats(self, capsys):
        """Test that summary prints statistics"""
        stats = {
            'n_samples': 10000,
            'n_parameters': 5,
            'order': 2,
            'analysis_type': 'T_seeded',
            'computation_time': 456.78
        }
        
        print_sobol_summary(stats, verbose=True)
        
        captured = capsys.readouterr()
        assert "10,000" in captured.out
        assert "5" in captured.out
        assert "2" in captured.out
        assert "T_seeded" in captured.out
        assert "456.78 seconds" in captured.out
    
    def test_summary_silent_when_not_verbose(self, capsys):
        """Test that summary is silent when verbose=False"""
        stats = {
            'n_samples': 1000,
            'n_parameters': 3,
            'order': 2,
            'analysis_type': 'lump',
            'computation_time': 100.0
        }
        
        print_sobol_summary(stats, verbose=False)
        
        captured = capsys.readouterr()
        assert captured.out == ""
    
    def test_summary_formatting(self, capsys):
        """Test summary formatting with different values"""
        stats = {
            'n_samples': 123456,
            'n_parameters': 12,
            'order': 3,
            'analysis_type': 'lump',
            'computation_time': 1234.56
        }
        
        print_sobol_summary(stats, verbose=True)
        
        captured = capsys.readouterr()
        # Check comma formatting for large numbers
        assert "123,456" in captured.out
        assert "12" in captured.out
        assert "3" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
