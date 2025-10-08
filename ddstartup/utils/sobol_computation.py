"""
Sobol sensitivity analysis computation module.

This module handles Sobol analysis workflow including:
- Sobol sample generation
- Parallel computation
- Sensitivity indices calculation
- Result visualization and export
"""

import numpy as np
from typing import Dict, Any, Callable
import time


def run_sobol_analysis(
    input_data: Dict[str, np.ndarray],
    output_file: str,
    config: Dict[str, Any],
    compute_function: Callable,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Sobol sensitivity analysis.
    
    Args:
        input_data: Dictionary of parameter arrays (distributions)
        output_file: Path to output file for results
        config: Configuration dictionary with analysis settings
        compute_function: Function to compute single combination
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with Sobol indices and statistics
        
    Raises:
        ImportError: If required sobol/chaospy packages not available
        ValueError: If configuration is invalid
    """
    # Import sobol analysis function from physics module
    try:
        from ddstartup.physics.sobol_functions import sobol_analysis as physics_sobol_analysis
    except ImportError:
        raise ImportError(
            "Sobol analysis requires the physics.sobol_functions module. "
            "Please ensure it is available in the physics/ directory."
        )
    
    # Extract configuration
    analysis_type = config['analysis_type']
    N_SAMPLES = config['N_SAMPLES']
    order = config['order']
    total_time = config['total_time']
    vector_length = config['vector_length']
    
    # Prepare parameter names
    param_names = list(input_data.keys())
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"STARTING SOBOL SENSITIVITY ANALYSIS")
        print(f"{'='*60}")
        print(f"Parameters: {len(param_names)}")
        print(f"Samples: {N_SAMPLES:,}")
        print(f"Order: {order}")
        print(f"Analysis type: {analysis_type}")
        print(f"Output file: {output_file}")
        print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Run Sobol analysis using existing physics module
    try:
        result = physics_sobol_analysis(
            input_data=input_data,
            param_names=param_names,
            N_SAMPLES=int(N_SAMPLES),
            order=order,
            analysis_type=analysis_type,
            total_time=total_time,
            compute_single_combination=compute_function,
            output_file=output_file,
            vector_length=vector_length,
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"\n❌ Sobol analysis failed: {e}")
        raise
    
    end_time = time.time()
    
    # Return statistics (merge with result from physics module)
    stats = {
        'n_samples': int(N_SAMPLES),
        'n_parameters': len(param_names),
        'order': order,
        'analysis_type': analysis_type,
        'computation_time': end_time - start_time,
        'output_file': output_file
    }
    
    # Add result data if available
    if result:
        stats.update({
            'n_valid': result.get('n_valid', 0),
            'throughput': result.get('throughput', 0),
            'plot_filename': result.get('plot_filename', '')
        })
    
    if verbose:
        print(f"\n{'='*60}")
        print("SOBOL ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Samples processed: {N_SAMPLES:,}")
        print(f"✅ Parameters analyzed: {len(param_names)}")
        print(f"⏱️  Computation time: {stats['computation_time']:.2f} seconds")
        print(f"📊 Sobol indices computed up to order {order}")
        print(f"{'='*60}\n")
    
    return stats


def print_sobol_summary(
    stats: Dict[str, Any],
    verbose: bool = True
) -> None:
    """
    Print summary of Sobol analysis results.
    
    Args:
        stats: Statistics dictionary from run_sobol_analysis
        verbose: Whether to print information
    """
    if not verbose:
        return
    
    print(f"\n{'='*60}")
    print("SOBOL SENSITIVITY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Samples: {stats['n_samples']:,}")
    print(f"Parameters: {stats['n_parameters']}")
    print(f"Analysis type: {stats['analysis_type']}")
    print(f"Order: {stats['order']}")
    print(f"Time: {stats['computation_time']:.2f} seconds")
    print(f"{'='*60}\n")
