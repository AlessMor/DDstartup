"""
Profiling utilities for DD Startup analysis.

This module provides utilities for profiling computational bottlenecks
in parametric analysis, Sobol analysis, and other computational tasks.

Usage as script:
    python -m ddstartup.utils.profiling <param_file> <config_file>
    
Usage in code:
    from ddstartup.utils.profiling import Timer, PerformanceMonitor
    
    with Timer("My operation"):
        result = expensive_function()
"""

import sys
import time
import cProfile
import pstats
import io
from functools import wraps
from typing import Callable, Any, Dict, Optional, Tuple
import numpy as np


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Initialize timer.
        
        Args:
            name: Name of the operation being timed
            verbose: Whether to print timing information
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Start timer."""
        if self.verbose:
            print(f"\n⏱️  Starting: {self.name}")
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Stop timer and print elapsed time."""
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"✅ Completed: {self.name} in {self.elapsed:.2f}s")


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile a function using cProfile.
    
    Usage:
        @profile_function
        def my_function():
            # code to profile
            pass
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function that profiles when called
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        # Print statistics
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        
        print(f"\n{'='*80}")
        print(f"Profile: {func.__name__}")
        print('='*80)
        stats.print_stats(20)
        
        return result
    
    return wrapper


def time_function(func: Callable) -> Callable:
    """
    Decorator to time a function execution.
    
    Usage:
        @time_function
        def my_function():
            # code to time
            pass
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that prints execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        print(f"⏱️  {func.__name__} took {elapsed:.2f}s")
        
        return result
    
    return wrapper


class PerformanceMonitor:
    """
    Monitor performance metrics during computation.
    
    Example:
        monitor = PerformanceMonitor()
        
        with monitor.track("Load data"):
            data = load_data()
        
        with monitor.track("Process"):
            result = process(data)
        
        monitor.print_summary()
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.timings = {}
        self.current_operation = None
        self.start_time = None
    
    def track(self, operation_name: str):
        """
        Context manager to track operation timing.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Self for context manager
        """
        self.current_operation = operation_name
        return self
    
    def __enter__(self):
        """Start tracking."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Stop tracking and record timing."""
        elapsed = time.time() - self.start_time
        
        if self.current_operation not in self.timings:
            self.timings[self.current_operation] = []
        
        self.timings[self.current_operation].append(elapsed)
    
    def print_summary(self):
        """Print performance summary."""
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        total_time = sum(sum(times) for times in self.timings.values())
        
        print(f"\n{'Operation':<40} {'Calls':<10} {'Total':<12} {'Average':<12} {'%':<8}")
        print("-"*80)
        
        # Sort by total time
        sorted_ops = sorted(
            self.timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )
        
        for operation, times in sorted_ops:
            total = sum(times)
            avg = total / len(times)
            percent = (total / total_time * 100) if total_time > 0 else 0
            
            print(f"{operation:<40} {len(times):<10} {total:>10.2f}s "
                  f"{avg:>10.4f}s {percent:>6.1f}%")
        
        print("-"*80)
        print(f"{'TOTAL':<40} {sum(len(t) for t in self.timings.values()):<10} "
              f"{total_time:>10.2f}s")
        print("="*80)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get performance summary as dictionary.
        
        Returns:
            dict: Performance metrics
        """
        total_time = sum(sum(times) for times in self.timings.values())
        
        summary = {}
        for operation, times in self.timings.items():
            total = sum(times)
            summary[operation] = {
                'calls': len(times),
                'total_time': total,
                'average_time': total / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'percent': (total / total_time * 100) if total_time > 0 else 0
            }
        
        return summary


def estimate_remaining_time(completed: int, total: int, elapsed: float) -> str:
    """
    Estimate remaining time based on progress.
    
    Args:
        completed: Number of completed items
        total: Total number of items
        elapsed: Elapsed time so far (seconds)
        
    Returns:
        str: Formatted remaining time estimate
    """
    if completed == 0:
        return "unknown"
    
    rate = completed / elapsed  # items per second
    remaining = total - completed
    remaining_seconds = remaining / rate
    
    # Format as human-readable
    if remaining_seconds < 60:
        return f"{remaining_seconds:.0f}s"
    elif remaining_seconds < 3600:
        minutes = remaining_seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = remaining_seconds / 3600
        return f"{hours:.1f}h"


def print_progress(completed: int, total: int, elapsed: float, prefix: str = "Progress"):
    """
    Print progress bar with time estimate.
    
    Args:
        completed: Number of completed items
        total: Total number of items
        elapsed: Elapsed time so far (seconds)
        prefix: Prefix for progress message
    """
    percent = (completed / total * 100) if total > 0 else 0
    bar_length = 40
    filled = int(bar_length * completed / total) if total > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)
    
    rate = completed / elapsed if elapsed > 0 else 0
    remaining = estimate_remaining_time(completed, total, elapsed)
    
    print(f"\r{prefix}: [{bar}] {percent:>5.1f}% | "
          f"{completed}/{total} | {rate:.1f} items/s | "
          f"ETA: {remaining}", end='', flush=True)
    
    if completed >= total:
        print()  # New line when complete


class ComputationProfiler:
    """
    Profiler specifically for parametric computations.
    
    Tracks timing for different stages of parametric analysis:
    - Parameter loading
    - Grid creation
    - Reactivity lookup
    - Worker initialization
    - Computation (per worker)
    - Result collection
    """
    
    def __init__(self):
        """Initialize computation profiler."""
        self.monitor = PerformanceMonitor()
        self.start_time = time.time()
        self.stage_times = {}
    
    def start_stage(self, stage_name: str):
        """Start timing a computation stage."""
        return self.monitor.track(stage_name)
    
    def print_summary(self):
        """Print profiling summary."""
        total_elapsed = time.time() - self.start_time
        
        print(f"\n\nTotal computation time: {total_elapsed:.2f}s")
        self.monitor.print_summary()
    
    def get_bottlenecks(self, threshold: float = 5.0) -> Dict[str, float]:
        """
        Identify computation bottlenecks.
        
        Args:
            threshold: Minimum percentage to be considered a bottleneck
            
        Returns:
            dict: Bottleneck operations and their percentage
        """
        summary = self.monitor.get_summary()
        bottlenecks = {
            op: data['percent']
            for op, data in summary.items()
            if data['percent'] >= threshold
        }
        
        return dict(sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True))


# Example usage function
def example_usage():
    """Example usage of profiling utilities."""
    print("="*80)
    print("PROFILING UTILITIES EXAMPLE")
    print("="*80)
    
    # Example 1: Timer context manager
    with Timer("Loading data"):
        time.sleep(0.1)  # Simulate work
    
    # Example 2: Performance monitor
    monitor = PerformanceMonitor()
    
    for i in range(5):
        with monitor.track("Process batch"):
            time.sleep(0.05)  # Simulate work
    
    with monitor.track("Save results"):
        time.sleep(0.1)
    
    monitor.print_summary()
    
    # Example 3: Progress bar
    total = 100
    start = time.time()
    for i in range(total):
        time.sleep(0.01)
        print_progress(i+1, total, time.time() - start)


def profile_analysis(param_file: str, config_file: str, verbose: bool = True) -> Tuple[int, Dict[str, Any]]:
    """
    Profile a DD Startup analysis run.
    
    Args:
        param_file: Parameter file path (absolute or relative to inputs/)
        config_file: Configuration file path (absolute or relative to inputs/)
        verbose: Whether to print detailed output
        
    Returns:
        tuple: (exit_code, profiling_stats)
    """
    import os
    from pathlib import Path
    
    # Convert to absolute paths if they exist as files
    param_path = Path(param_file)
    config_path = Path(config_file)
    
    # If files are absolute and exist, use them directly
    # Otherwise assume they're in inputs/
    if not param_path.is_absolute():
        param_file_arg = param_file
    else:
        param_file_arg = str(param_path)
        
    if not config_path.is_absolute():
        config_file_arg = config_file
    else:
        config_file_arg = str(config_path)
    
    if verbose:
        print("=" * 80)
        print("DD STARTUP PROFILING")
        print("=" * 80)
        print(f"\nParameter file: {param_file_arg}")
        print(f"Config file: {config_file_arg}")
    
    # Import main after setting up args
    from ddstartup.main import main as ddstartup_main
    
    # Set up arguments
    original_argv = sys.argv.copy()
    sys.argv = ['ddstartup', param_file_arg, config_file_arg]
    if verbose:
        sys.argv.append('--verbose')
    
    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    try:
        exit_code = ddstartup_main()
    except Exception as e:
        if verbose:
            print(f"\n❌ Analysis failed: {e}")
        exit_code = 1
    finally:
        elapsed = time.time() - start_time
        profiler.disable()
        sys.argv = original_argv
    
    # Get statistics
    stats = pstats.Stats(profiler)
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"PROFILING RESULTS (Total time: {elapsed:.2f}s)")
        print("=" * 80)
        
        print("\n" + "-" * 80)
        print("Top 20 functions by cumulative time:")
        print("-" * 80)
        stats.sort_stats('cumulative')
        stats.print_stats(20)
        
        print("\n" + "-" * 80)
        print("Top 20 functions by internal time:")
        print("-" * 80)
        stats.sort_stats('time')
        stats.print_stats(20)
    
    # Extract key metrics
    metrics = {
        'exit_code': exit_code,
        'total_time': elapsed,
        'param_file': param_file,
        'config_file': config_file
    }
    
    return exit_code, metrics


if __name__ == '__main__':
    if len(sys.argv) == 3:
        # Run profiling
        param_file = sys.argv[1]
        config_file = sys.argv[2]
        exit_code, _ = profile_analysis(param_file, config_file)
        sys.exit(exit_code)
    else:
        # Run example
        example_usage()
