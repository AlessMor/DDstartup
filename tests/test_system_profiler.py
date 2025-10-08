"""
Tests for system_profiler.py functions
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddstartup.utils.system_profiler import (
    get_system_info,
    calculate_optimal_n_jobs,
    calculate_optimal_chunk_size,
    calculate_optimal_batch_size,
    calculate_optimal_sobol_samples,
    calculate_optimal_sobol_order,
    get_optimal_parameters,
    print_system_profile,
    override_with_config
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_system_info_large():
    """Mock system info for a large system (16+ cores, 32 GB RAM)"""
    return {
        'n_cores': 24,
        'total_ram_gb': 32.0,
        'available_ram_gb': 24.0,
        'ram_percent_used': 25.0,
        'cpu_freq_mhz': 3600.0
    }


@pytest.fixture
def mock_system_info_medium():
    """Mock system info for a medium system (8 cores, 16 GB RAM)"""
    return {
        'n_cores': 8,
        'total_ram_gb': 16.0,
        'available_ram_gb': 12.0,
        'ram_percent_used': 25.0,
        'cpu_freq_mhz': 2800.0
    }


@pytest.fixture
def mock_system_info_small():
    """Mock system info for a small system (4 cores, 8 GB RAM)"""
    return {
        'n_cores': 4,
        'total_ram_gb': 8.0,
        'available_ram_gb': 4.0,
        'ram_percent_used': 50.0,
        'cpu_freq_mhz': 2400.0
    }


@pytest.fixture
def mock_system_info_minimal():
    """Mock system info for a minimal system (2 cores, 4 GB RAM)"""
    return {
        'n_cores': 2,
        'total_ram_gb': 4.0,
        'available_ram_gb': 2.0,
        'ram_percent_used': 50.0,
        'cpu_freq_mhz': None
    }


# ============================================================================
# Test get_system_info
# ============================================================================

class TestGetSystemInfo:
    """Tests for get_system_info function"""
    
    def test_returns_dict_with_required_keys(self):
        """Should return dictionary with all required keys"""
        info = get_system_info()
        
        required_keys = [
            'n_cores', 'total_ram_gb', 'available_ram_gb',
            'ram_percent_used', 'cpu_freq_mhz'
        ]
        
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
    
    def test_n_cores_positive(self):
        """Number of cores should be positive"""
        info = get_system_info()
        assert info['n_cores'] > 0
    
    def test_ram_values_positive(self):
        """RAM values should be positive"""
        info = get_system_info()
        assert info['total_ram_gb'] > 0
        assert info['available_ram_gb'] >= 0
        assert 0 <= info['ram_percent_used'] <= 100
    
    def test_available_ram_not_greater_than_total(self):
        """Available RAM should not exceed total RAM"""
        info = get_system_info()
        assert info['available_ram_gb'] <= info['total_ram_gb']
    
    def test_cpu_freq_none_or_positive(self):
        """CPU frequency should be None or positive (0.0 allowed for some systems)"""
        info = get_system_info()
        if info['cpu_freq_mhz'] is not None:
            assert info['cpu_freq_mhz'] >= 0


# ============================================================================
# Test calculate_optimal_n_jobs
# ============================================================================

class TestCalculateOptimalNJobs:
    """Tests for calculate_optimal_n_jobs function"""
    
    def test_large_system(self, mock_system_info_large):
        """Large system (24 cores) should use all cores"""
        n_jobs = calculate_optimal_n_jobs(mock_system_info_large)
        assert n_jobs == 24
    
    def test_medium_system(self, mock_system_info_medium):
        """Medium system (8 cores) should leave one core free"""
        n_jobs = calculate_optimal_n_jobs(mock_system_info_medium)
        assert n_jobs == 7
    
    def test_small_system(self, mock_system_info_small):
        """Small system (4 cores) should leave one core free"""
        n_jobs = calculate_optimal_n_jobs(mock_system_info_small)
        assert n_jobs == 3
    
    def test_minimal_system(self, mock_system_info_minimal):
        """Minimal system (2 cores) should use at least 1 job"""
        n_jobs = calculate_optimal_n_jobs(mock_system_info_minimal)
        assert n_jobs >= 1
    
    def test_without_system_info(self):
        """Should work without passing system_info"""
        n_jobs = calculate_optimal_n_jobs()
        assert n_jobs > 0


# ============================================================================
# Test calculate_optimal_chunk_size
# ============================================================================

class TestCalculateOptimalChunkSize:
    """Tests for calculate_optimal_chunk_size function"""
    
    def test_large_system(self, mock_system_info_large):
        """Large system should have large chunk size"""
        chunk_size = calculate_optimal_chunk_size(mock_system_info_large)
        assert chunk_size >= 5000
    
    def test_medium_system(self, mock_system_info_medium):
        """Medium system should have medium chunk size"""
        chunk_size = calculate_optimal_chunk_size(mock_system_info_medium)
        assert 2000 <= chunk_size <= 10000
    
    def test_small_system(self, mock_system_info_small):
        """Small system should have smaller chunk size"""
        chunk_size = calculate_optimal_chunk_size(mock_system_info_small)
        assert chunk_size >= 500
    
    def test_scales_with_n_jobs(self, mock_system_info_medium):
        """Chunk size should scale with n_jobs"""
        chunk_size_low = calculate_optimal_chunk_size(mock_system_info_medium, n_jobs=2)
        chunk_size_high = calculate_optimal_chunk_size(mock_system_info_medium, n_jobs=8)
        assert chunk_size_high >= chunk_size_low
    
    def test_without_system_info(self):
        """Should work without passing system_info"""
        chunk_size = calculate_optimal_chunk_size()
        assert chunk_size > 0


# ============================================================================
# Test calculate_optimal_batch_size
# ============================================================================

class TestCalculateOptimalBatchSize:
    """Tests for calculate_optimal_batch_size function"""
    
    def test_large_system(self, mock_system_info_large):
        """Large system should have large batch size"""
        batch_size = calculate_optimal_batch_size(mock_system_info_large)
        assert batch_size >= 500
    
    def test_medium_system(self, mock_system_info_medium):
        """Medium system should have medium batch size"""
        batch_size = calculate_optimal_batch_size(mock_system_info_medium)
        assert 250 <= batch_size <= 1000
    
    def test_small_system(self, mock_system_info_small):
        """Small system should have smaller batch size"""
        batch_size = calculate_optimal_batch_size(mock_system_info_small)
        assert batch_size >= 100
    
    def test_low_memory_constraint(self):
        """Low memory should reduce batch size"""
        low_mem_info = {
            'n_cores': 16,
            'total_ram_gb': 4.0,
            'available_ram_gb': 2.0,
            'ram_percent_used': 50.0,
            'cpu_freq_mhz': 3000.0
        }
        batch_size = calculate_optimal_batch_size(low_mem_info)
        assert batch_size <= 100
    
    def test_without_system_info(self):
        """Should work without passing system_info"""
        batch_size = calculate_optimal_batch_size()
        assert batch_size > 0


# ============================================================================
# Test calculate_optimal_sobol_samples
# ============================================================================

class TestCalculateOptimalSobolSamples:
    """Tests for calculate_optimal_sobol_samples function"""
    
    def test_large_system(self, mock_system_info_large):
        """Large system should support many samples"""
        n_samples = calculate_optimal_sobol_samples(mock_system_info_large)
        assert n_samples >= 50000
    
    def test_medium_system(self, mock_system_info_medium):
        """Medium system should support moderate samples"""
        n_samples = calculate_optimal_sobol_samples(mock_system_info_medium)
        assert 10000 <= n_samples <= 100000
    
    def test_small_system(self, mock_system_info_small):
        """Small system should have fewer samples"""
        n_samples = calculate_optimal_sobol_samples(mock_system_info_small)
        assert n_samples >= 5000
    
    def test_low_memory_constraint(self):
        """Low memory should limit sample count"""
        low_mem_info = {
            'n_cores': 16,
            'total_ram_gb': 4.0,
            'available_ram_gb': 2.0,
            'ram_percent_used': 50.0,
            'cpu_freq_mhz': 3000.0
        }
        n_samples = calculate_optimal_sobol_samples(low_mem_info)
        assert n_samples <= 5000
    
    def test_without_system_info(self):
        """Should work without passing system_info"""
        n_samples = calculate_optimal_sobol_samples()
        assert n_samples > 0


# ============================================================================
# Test calculate_optimal_sobol_order
# ============================================================================

class TestCalculateOptimalSobolOrder:
    """Tests for calculate_optimal_sobol_order function"""
    
    def test_large_system(self, mock_system_info_large):
        """Large system should support order 3"""
        order = calculate_optimal_sobol_order(mock_system_info_large)
        assert order == 3
    
    def test_medium_system(self, mock_system_info_medium):
        """Medium system should support order 3"""
        order = calculate_optimal_sobol_order(mock_system_info_medium)
        assert order == 3
    
    def test_small_system(self, mock_system_info_small):
        """Small system should use order 2"""
        order = calculate_optimal_sobol_order(mock_system_info_small)
        assert order == 2
    
    def test_minimal_system(self, mock_system_info_minimal):
        """Minimal system should use order 2"""
        order = calculate_optimal_sobol_order(mock_system_info_minimal)
        assert order == 2
    
    def test_without_system_info(self):
        """Should work without passing system_info"""
        order = calculate_optimal_sobol_order()
        assert order in [2, 3]


# ============================================================================
# Test get_optimal_parameters
# ============================================================================

class TestGetOptimalParameters:
    """Tests for get_optimal_parameters function"""
    
    def test_parametric_analysis(self):
        """Should return all required parameters for parametric analysis"""
        params = get_optimal_parameters(analysis_method='parametric')
        
        assert 'system_info' in params
        assert 'n_jobs' in params
        assert 'chunk_size' in params
        assert 'batch_size' in params
        assert 'N_SAMPLES' not in params  # Not needed for parametric
        assert 'order' not in params  # Not needed for parametric
    
    def test_sobol_analysis(self):
        """Should return all required parameters for Sobol analysis"""
        params = get_optimal_parameters(analysis_method='sobol')
        
        assert 'system_info' in params
        assert 'n_jobs' in params
        assert 'chunk_size' in params
        assert 'batch_size' in params
        assert 'N_SAMPLES' in params  # Needed for Sobol
        assert 'order' in params  # Needed for Sobol
    
    def test_verbose_false(self, capsys):
        """Should not print when verbose=False"""
        get_optimal_parameters(verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""
    
    def test_verbose_true(self, capsys):
        """Should print when verbose=True"""
        get_optimal_parameters(verbose=True)
        captured = capsys.readouterr()
        assert "SYSTEM PROFILE" in captured.out
        assert "CPU Cores" in captured.out
    
    def test_values_are_positive(self):
        """All calculated values should be positive"""
        params = get_optimal_parameters(analysis_method='sobol')
        
        assert params['n_jobs'] > 0
        assert params['chunk_size'] > 0
        assert params['batch_size'] > 0
        assert params['N_SAMPLES'] > 0
        assert params['order'] > 0


# ============================================================================
# Test print_system_profile
# ============================================================================

class TestPrintSystemProfile:
    """Tests for print_system_profile function"""
    
    def test_prints_parametric_info(self, mock_system_info_large, capsys):
        """Should print parametric analysis info"""
        params = {
            'system_info': mock_system_info_large,
            'n_jobs': 24,
            'chunk_size': 5000,
            'batch_size': 1000
        }
        
        print_system_profile(params, 'parametric')
        captured = capsys.readouterr()
        
        assert "SYSTEM PROFILE" in captured.out
        assert "CPU Cores: 24" in captured.out
        assert "n_jobs: 24" in captured.out
        assert "Sobol Analysis Parameters" not in captured.out
    
    def test_prints_sobol_info(self, mock_system_info_large, capsys):
        """Should print Sobol analysis info"""
        params = {
            'system_info': mock_system_info_large,
            'n_jobs': 24,
            'chunk_size': 5000,
            'batch_size': 1000,
            'N_SAMPLES': 100000,
            'order': 3
        }
        
        print_system_profile(params, 'sobol')
        captured = capsys.readouterr()
        
        assert "SYSTEM PROFILE" in captured.out
        assert "Sobol Analysis Parameters" in captured.out
        assert "N_SAMPLES: 100,000" in captured.out
        assert "order: 3" in captured.out
    
    def test_handles_none_cpu_freq(self, mock_system_info_minimal, capsys):
        """Should handle None CPU frequency gracefully"""
        params = {
            'system_info': mock_system_info_minimal,
            'n_jobs': 1,
            'chunk_size': 500,
            'batch_size': 100
        }
        
        print_system_profile(params, 'parametric')
        captured = capsys.readouterr()
        
        assert "SYSTEM PROFILE" in captured.out
        # Should not print CPU frequency line if None


# ============================================================================
# Test override_with_config
# ============================================================================

class TestOverrideWithConfig:
    """Tests for override_with_config function"""
    
    def test_no_override(self):
        """Should return original params if config is empty"""
        optimal = {
            'n_jobs': 8,
            'chunk_size': 2000,
            'batch_size': 500,
            'N_SAMPLES': 10000,
            'order': 3
        }
        config = {}
        
        result = override_with_config(optimal, config)
        assert result == optimal
    
    def test_partial_override(self):
        """Should override only specified parameters"""
        optimal = {
            'n_jobs': 8,
            'chunk_size': 2000,
            'batch_size': 500,
        }
        config = {
            'n_jobs': 16,
            'chunk_size': None,  # Should not override with None
        }
        
        result = override_with_config(optimal, config)
        assert result['n_jobs'] == 16
        assert result['chunk_size'] == 2000  # Not overridden
        assert result['batch_size'] == 500  # Not in config
    
    def test_full_override(self):
        """Should override all specified parameters"""
        optimal = {
            'n_jobs': 8,
            'chunk_size': 2000,
            'batch_size': 500,
            'N_SAMPLES': 10000,
            'order': 3
        }
        config = {
            'n_jobs': 16,
            'chunk_size': 5000,
            'batch_size': 1000,
            'N_SAMPLES': 50000,
            'order': 2
        }
        
        result = override_with_config(optimal, config)
        assert result['n_jobs'] == 16
        assert result['chunk_size'] == 5000
        assert result['batch_size'] == 1000
        assert result['N_SAMPLES'] == 50000
        assert result['order'] == 2
    
    def test_does_not_modify_original(self):
        """Should not modify the original optimal_params dict"""
        optimal = {
            'n_jobs': 8,
            'chunk_size': 2000,
        }
        config = {
            'n_jobs': 16,
        }
        
        result = override_with_config(optimal, config)
        assert optimal['n_jobs'] == 8  # Original unchanged
        assert result['n_jobs'] == 16  # Result modified
    
    def test_preserves_system_info(self):
        """Should preserve system_info in result"""
        system_info = {'n_cores': 8, 'total_ram_gb': 16.0}
        optimal = {
            'system_info': system_info,
            'n_jobs': 8,
        }
        config = {
            'n_jobs': 16,
        }
        
        result = override_with_config(optimal, config)
        assert result['system_info'] == system_info
