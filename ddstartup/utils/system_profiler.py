"""
System Profiling Functions

This module provides functions for profiling the system and determining
optimal parameters for parallel computation based on available resources.
"""

import multiprocessing
import psutil
from typing import Dict, Tuple, Optional


def get_system_info() -> Dict[str, any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dictionary containing:
        - n_cores: Number of CPU cores
        - total_ram_gb: Total RAM in GB
        - available_ram_gb: Available RAM in GB
        - cpu_freq_mhz: CPU frequency in MHz (if available)
        - ram_percent_used: Percentage of RAM currently in use
    """
    n_cores = multiprocessing.cpu_count()
    mem = psutil.virtual_memory()
    
    system_info = {
        'n_cores': n_cores,
        'total_ram_gb': mem.total / 1e9,
        'available_ram_gb': mem.available / 1e9,
        'ram_percent_used': mem.percent,
    }
    
    # CPU frequency (may not be available on all systems)
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq and hasattr(cpu_freq, 'max'):
            system_info['cpu_freq_mhz'] = cpu_freq.max
        elif cpu_freq and hasattr(cpu_freq, 'current'):
            system_info['cpu_freq_mhz'] = cpu_freq.current
        else:
            system_info['cpu_freq_mhz'] = None
    except (AttributeError, RuntimeError):
        system_info['cpu_freq_mhz'] = None
    
    return system_info


def calculate_optimal_n_jobs(system_info: Optional[Dict] = None) -> int:
    """
    Calculate optimal number of parallel jobs based on CPU cores.
    
    Args:
        system_info: System information dictionary (if None, will be fetched)
        
    Returns:
        Optimal number of parallel jobs
    """
    if system_info is None:
        system_info = get_system_info()
    
    n_cores = system_info['n_cores']
    
    # Strategy: Use all cores for large systems, scale down for smaller ones
    if n_cores >= 16:
        return n_cores  # Use all cores for large systems
    elif n_cores >= 8:
        return n_cores - 1  # Leave one core free
    elif n_cores >= 4:
        return n_cores - 1
    else:
        return max(1, n_cores - 1)  # Always at least 1


def calculate_optimal_chunk_size(
    system_info: Optional[Dict] = None,
    n_jobs: Optional[int] = None
) -> int:
    """
    Calculate optimal chunk size for parallel processing.
    
    Args:
        system_info: System information dictionary
        n_jobs: Number of parallel jobs (if None, will be calculated)
        
    Returns:
        Optimal chunk size for batching computations
    """
    if system_info is None:
        system_info = get_system_info()
    
    if n_jobs is None:
        n_jobs = calculate_optimal_n_jobs(system_info)
    
    n_cores = system_info['n_cores']
    
    # Larger chunks for more cores to reduce overhead
    if n_cores >= 16:
        base_chunk = 5000
    elif n_cores >= 8:
        base_chunk = 2000
    elif n_cores >= 4:
        base_chunk = 1000
    else:
        base_chunk = 500
    
    # Scale by n_jobs
    chunk_size = max(base_chunk, n_jobs * 500)
    
    return chunk_size


def calculate_optimal_batch_size(system_info: Optional[Dict] = None) -> int:
    """
    Calculate optimal batch size for buffered operations.
    
    Args:
        system_info: System information dictionary
        
    Returns:
        Optimal batch size for buffering results
    """
    if system_info is None:
        system_info = get_system_info()
    
    n_cores = system_info['n_cores']
    available_ram_gb = system_info['available_ram_gb']
    
    # Base batch size on cores
    if n_cores >= 16:
        batch_size = 1000
    elif n_cores >= 8:
        batch_size = 500
    elif n_cores >= 4:
        batch_size = 250
    else:
        batch_size = 100
    
    # Reduce if low memory
    if available_ram_gb < 4:
        batch_size = min(batch_size, 100)
    elif available_ram_gb < 8:
        batch_size = min(batch_size, 250)
    
    return batch_size


def calculate_optimal_sobol_samples(system_info: Optional[Dict] = None) -> int:
    """
    Calculate optimal number of samples for Sobol analysis.
    
    Args:
        system_info: System information dictionary
        
    Returns:
        Recommended number of Sobol samples
    """
    if system_info is None:
        system_info = get_system_info()
    
    n_cores = system_info['n_cores']
    available_ram_gb = system_info['available_ram_gb']
    
    # Base on cores and memory
    if n_cores >= 16 and available_ram_gb >= 16:
        n_samples = 100000
    elif n_cores >= 8 and available_ram_gb >= 8:
        n_samples = 50000
    elif n_cores >= 4:
        n_samples = 10000
    else:
        n_samples = 5000
    
    # Adjust for memory constraints
    if available_ram_gb < 4:
        n_samples = min(n_samples, 5000)
    elif available_ram_gb < 8:
        n_samples = min(n_samples, 20000)
    
    return n_samples


def calculate_optimal_sobol_order(system_info: Optional[Dict] = None) -> int:
    """
    Calculate optimal order for Sobol sensitivity analysis.
    
    Args:
        system_info: System information dictionary
        
    Returns:
        Recommended Sobol analysis order (2 or 3)
    """
    if system_info is None:
        system_info = get_system_info()
    
    n_cores = system_info['n_cores']
    available_ram_gb = system_info['available_ram_gb']
    
    # Higher order analysis requires more computational resources
    if n_cores >= 8 and available_ram_gb >= 8:
        return 3
    else:
        return 2


def get_optimal_parameters(
    analysis_method: str = 'parametric',
    verbose: bool = False
) -> Dict[str, any]:
    """
    Get all optimal parameters for parallel computation.
    
    Args:
        analysis_method: Type of analysis ('parametric' or 'sobol')
        verbose: If True, print system information
        
    Returns:
        Dictionary containing:
        - system_info: System information
        - n_jobs: Optimal number of parallel jobs
        - chunk_size: Optimal chunk size
        - batch_size: Optimal batch size
        - N_SAMPLES: Optimal number of Sobol samples (if sobol method)
        - order: Optimal Sobol order (if sobol method)
    """
    # Get system information
    system_info = get_system_info()
    
    # Calculate optimal parameters
    n_jobs = calculate_optimal_n_jobs(system_info)
    chunk_size = calculate_optimal_chunk_size(system_info, n_jobs)
    batch_size = calculate_optimal_batch_size(system_info)
    
    params = {
        'system_info': system_info,
        'n_jobs': n_jobs,
        'chunk_size': chunk_size,
        'batch_size': batch_size,
    }
    
    # Add Sobol-specific parameters
    if analysis_method == 'sobol':
        params['N_SAMPLES'] = calculate_optimal_sobol_samples(system_info)
        params['order'] = calculate_optimal_sobol_order(system_info)
    
    if verbose:
        print_system_profile(params, analysis_method)
    
    return params


def print_system_profile(params: Dict[str, any], analysis_method: str = 'parametric') -> None:
    """
    Print formatted system profile information.
    
    Args:
        params: Dictionary containing system info and optimal parameters
        analysis_method: Type of analysis being performed
    """
    system_info = params['system_info']
    
    print("\n" + "="*60)
    print("SYSTEM PROFILE")
    print("="*60)
    
    # Hardware information
    print("Hardware:")
    print(f"  CPU Cores: {system_info['n_cores']}")
    if system_info['cpu_freq_mhz']:
        print(f"  CPU Frequency: {system_info['cpu_freq_mhz']:.0f} MHz")
    print(f"  Total RAM: {system_info['total_ram_gb']:.1f} GB")
    print(f"  Available RAM: {system_info['available_ram_gb']:.1f} GB ({100 - system_info['ram_percent_used']:.1f}% free)")
    
    # Recommended parameters
    print("\nRecommended Parallel Processing Parameters:")
    print(f"  n_jobs: {params['n_jobs']} (parallel workers)")
    print(f"  chunk_size: {params['chunk_size']} (computations per chunk)")
    print(f"  batch_size: {params['batch_size']} (results buffer size)")
    
    # Sobol-specific parameters
    if analysis_method == 'sobol':
        print(f"\nSobol Analysis Parameters:")
        print(f"  N_SAMPLES: {params['N_SAMPLES']:,} (number of samples)")
        print(f"  order: {params['order']} (sensitivity order)")
    
    print("="*60 + "\n")


def override_with_config(
    optimal_params: Dict[str, any],
    config: Dict[str, any]
) -> Dict[str, any]:
    """
    Override optimal parameters with user-specified config values.
    
    Args:
        optimal_params: Dictionary of optimal parameters
        config: User configuration dictionary
        
    Returns:
        Updated parameters dictionary
    """
    updated_params = optimal_params.copy()
    
    # Override if specified in config (and not None)
    if config.get('n_jobs') is not None:
        updated_params['n_jobs'] = config['n_jobs']
    
    if config.get('chunk_size') is not None:
        updated_params['chunk_size'] = config['chunk_size']
    
    if config.get('batch_size') is not None:
        updated_params['batch_size'] = config['batch_size']
    
    if config.get('N_SAMPLES') is not None:
        updated_params['N_SAMPLES'] = config['N_SAMPLES']
    
    if config.get('order') is not None:
        updated_params['order'] = config['order']
    
    return updated_params
