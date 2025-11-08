"""
Tests for profiling utilities.

Tests the profiling module functionality with small parametric analyses
to ensure profiling tools work correctly with both T-seeded and lump methods.
"""

import pytest
import time
import os
import shutil
from pathlib import Path

from ddstartup.utils.profiling import (
    Timer, 
    PerformanceMonitor, 
    time_function,
    profile_function,
    print_progress,
    ComputationProfiler,
    profile_analysis
)


# Fixtures
@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def output_cleanup():
    """Clean up test output directories after tests."""
    yield
    # Clean up after tests
    test_outputs = [
        "outputs/test_profiling_tseeded",
        "outputs/test_profiling_lump"
    ]
    for output_dir in test_outputs:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


class TestTimer:
    """Test Timer context manager."""
    
    def test_timer_basic(self, capsys):
        """Test basic timer functionality."""
        with Timer("Test operation"):
            time.sleep(0.1)
        
        captured = capsys.readouterr()
        assert "Test operation" in captured.out
        assert "Completed:" in captured.out or "completed in" in captured.out
    
    def test_timer_silent(self, capsys):
        """Test timer with verbose=False."""
        with Timer("Silent operation", verbose=False) as t:
            time.sleep(0.1)
        
        captured = capsys.readouterr()
        assert captured.out == ""
        assert t.elapsed >= 0.1
    
    def test_timer_elapsed_attribute(self):
        """Test that elapsed time is accessible."""
        with Timer("Test", verbose=False) as t:
            time.sleep(0.05)
        
        assert hasattr(t, 'elapsed')
        assert t.elapsed >= 0.05


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""
    
    def test_performance_monitor_tracking(self):
        """Test tracking multiple operations."""
        monitor = PerformanceMonitor()
        
        with monitor.track("Operation A"):
            time.sleep(0.05)
        
        with monitor.track("Operation B"):
            time.sleep(0.03)
        
        with monitor.track("Operation A"):
            time.sleep(0.02)
        
        assert "Operation A" in monitor.timings
        assert "Operation B" in monitor.timings
        assert len(monitor.timings["Operation A"]) == 2
        assert len(monitor.timings["Operation B"]) == 1
    
    def test_performance_monitor_summary(self, capsys):
        """Test summary printing."""
        monitor = PerformanceMonitor()
        
        for _ in range(3):
            with monitor.track("Test op"):
                time.sleep(0.01)
        
        monitor.print_summary()
        captured = capsys.readouterr()
        
        assert "Test op" in captured.out
        assert "Calls" in captured.out or "calls" in captured.out.lower()
        assert "Total" in captured.out or "total" in captured.out.lower()


class TestDecorators:
    """Test function decorators."""
    
    def test_time_function_decorator(self, capsys):
        """Test @time_function decorator."""
        @time_function
        def sample_function(x):
            time.sleep(0.05)
            return x * 2
        
        result = sample_function(5)
        assert result == 10
        
        captured = capsys.readouterr()
        assert "sample_function" in captured.out
        assert "took" in captured.out
    
    def test_profile_function_decorator(self, capsys):
        """Test @profile_function decorator."""
        @profile_function
        def another_function(n):
            time.sleep(0.02)
            return sum(range(n))
        
        result = another_function(100)
        assert result == sum(range(100))
        
        captured = capsys.readouterr()
        assert "another_function" in captured.out


class TestProgressBar:
    """Test progress bar functionality."""
    
    def test_print_progress(self, capsys):
        """Test progress bar printing."""
        start = time.time()
        for i in range(5):
            time.sleep(0.01)
            print_progress(i+1, 5, time.time() - start)
        
        captured = capsys.readouterr()
        assert "100%" in captured.out or "5/5" in captured.out


class TestComputationProfiler:
    """Test ComputationProfiler class."""
    
    def test_computation_profiler_basic(self):
        """Test basic profiler functionality."""
        profiler = ComputationProfiler()
        
        # Start and track a stage
        with profiler.start_stage("Load data"):
            time.sleep(0.05)
        
        with profiler.start_stage("Process"):
            time.sleep(0.1)
        
        # Check that stages were tracked
        assert len(profiler.monitor.timings) == 2
        assert "Load data" in profiler.monitor.timings
        assert "Process" in profiler.monitor.timings


class TestProfilingIntegration:
    """Integration tests with actual parametric analysis."""
    
    def test_profile_tseeded_analysis(self, fixtures_dir, output_cleanup):
        """Test profiling with T-seeded parametric analysis."""
        params_src = fixtures_dir / "params_test.yaml"
        config_src = fixtures_dir / "parametric_tseeded.yaml"
        
        assert params_src.exists(), "Test params fixture not found"
        assert config_src.exists(), "T-seeded config fixture not found"
        
        # Run profiling with absolute paths
        exit_code, metrics = profile_analysis(
            str(params_src),
            str(config_src), 
            verbose=False
        )
        
        # Check results
        assert exit_code == 0, "Analysis should complete successfully"
        assert 'total_time' in metrics
        assert metrics['total_time'] > 0
        assert str(params_src) in metrics['param_file']
        assert str(config_src) in metrics['config_file']
    
    def test_profile_lump_analysis(self, fixtures_dir, output_cleanup):
        """Test profiling with lump parametric analysis."""
        params_src = fixtures_dir / "params_test.yaml"
        config_src = fixtures_dir / "parametric_lump.yaml"
        
        assert params_src.exists(), "Test params fixture not found"
        assert config_src.exists(), "Lump config fixture not found"
        
        # Run profiling with absolute paths
        exit_code, metrics = profile_analysis(
            str(params_src),
            str(config_src),
            verbose=False
        )
        
        # Check results
        assert exit_code == 0, "Analysis should complete successfully"
        assert 'total_time' in metrics
        assert metrics['total_time'] > 0
        assert str(params_src) in metrics['param_file']
        assert str(config_src) in metrics['config_file']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
