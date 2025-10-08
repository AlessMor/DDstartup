"""
Parametric analysis computation module.

This module handles the parametric analysis workflow including:
- HDF5 file setup and dataset creation
- Parallel computation with joblib
- Progress tracking with tqdm
- Result writing and buffering
- Error handling and statistics
"""

import h5py
import numpy as np
from typing import Dict, Any, Callable, List
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import time

# Import tools for field names
from .tools import inputs_names, outputs_names


def run_parametric_analysis(
    input_data: Dict[str, np.ndarray],
    output_file: str,
    config: Dict[str, Any],
    compute_function: Callable,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run parametric analysis with parallel computation and HDF5 output.
    
    Args:
        input_data: Dictionary of parameter arrays
        output_file: Path to output HDF5 file
        config: Configuration dictionary with analysis settings
        compute_function: Function to compute single combination
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with analysis statistics
        
    Raises:
        ValueError: If configuration is invalid
        IOError: If HDF5 file cannot be created
    """
    # Extract configuration
    analysis_type = config['analysis_type']
    analysis_method = config['method']
    n_jobs = config['n_jobs']
    chunk_size = config['chunk_size']
    batch_size = config['batch_size']
    vector_length = config['vector_length']
    total_time = config['total_time']
    
    # Prepare parameter arrays
    param_names = list(input_data.keys())
    param_shapes = [arr.shape[0] for arr in input_data.values()]
    n_combinations = np.prod(param_shapes)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"STARTING PARAMETRIC ANALYSIS")
        print(f"{'='*60}")
        print(f"Total combinations: {n_combinations:,}")
        print(f"Parallel workers: {n_jobs}")
        print(f"Chunk size: {chunk_size}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")
    
    # Flatten arrays for easier indexing
    input_arrays = [np.asarray(arr) for arr in input_data.values()]
    input_arrays_flat = [arr.flatten() for arr in input_arrays]
    param_shapes_array = np.array(param_shapes, dtype=np.int64)
    
    # Fields to be saved in HDF5
    data_fields = list(dict.fromkeys(inputs_names + outputs_names))
    
    # Vector fields that need 2D storage
    vector_fields = ['N_ofc', 'N_ifc', 'N_stor', 'n_T', 'n_D', 'P_DDn', 'P_DDp', 'P_DT', 'TBE']
    
    # Statistics tracking
    start_time = time.time()
    processed_count = 0
    successful_count = 0
    
    # Create HDF5 file and run computation
    with h5py.File(output_file, 'w') as h5_file:
        # Add metadata
        h5_file.attrs.update({
            'total_combinations': int(n_combinations),
            'computation_start_time': start_time,
            'parameter_shapes': param_shapes,
            'method': analysis_method,
            'analysis_type': analysis_type,
            'vector_length': vector_length,
            'n_jobs': n_jobs,
            'chunk_size': chunk_size,
            'batch_size': batch_size,
        })
        
        # Pre-allocate datasets
        datasets = {}
        for field in data_fields:
            if field in vector_fields:
                # 2D array for vector fields
                datasets[field] = h5_file.create_dataset(
                    field,
                    (n_combinations, vector_length),
                    dtype=np.float64,
                    chunks=(min(chunk_size, n_combinations), vector_length),
                    compression='lzf',
                )
            elif field == 'error':
                # String field for errors
                datasets[field] = h5_file.create_dataset(
                    field,
                    (n_combinations,),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    chunks=(min(chunk_size, n_combinations),),
                    compression='lzf',
                )
            elif field == 'sol_success':
                # Boolean field
                datasets[field] = h5_file.create_dataset(
                    field,
                    (n_combinations,),
                    dtype=bool,
                    chunks=(min(chunk_size, n_combinations),),
                    compression='lzf',
                )
            else:
                # Scalar fields
                datasets[field] = h5_file.create_dataset(
                    field,
                    (n_combinations,),
                    dtype=np.float64,
                    chunks=(min(chunk_size, n_combinations),),
                    compression='lzf',
                )
        
        # Save parameter grids
        param_group = h5_file.create_group('parameter_fields')
        for name, arr in zip(param_names, input_arrays):
            param_group.create_dataset(f'{name}_values', data=arr, compression='gzip')
        
        # Initialize progress bar
        overall_pbar = tqdm(total=n_combinations, desc="Computing", unit="comb", disable=not verbose)
        
        # Prime Numba compilation if T_seeded
        if analysis_type == 'T_seeded' and verbose:
            print("Priming Numba compilation...")
            try:
                compute_function(0, input_arrays_flat, param_shapes_array, total_time, vector_length)
            except Exception as e:
                print(f"Warning during Numba priming: {e}")
        
        # Parallel computation with buffered writing
        buffer_results = []
        buffer_indices = []
        buffer_write_size = 1000  # Write every 1000 results
        
        # Ensure integer types for range()
        n_combinations = int(n_combinations)
        chunk_size = int(chunk_size)
        
        for chunk_start in range(0, n_combinations, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_combinations)
            chunk_indices = np.arange(chunk_start, chunk_end)
            
            # Prepare task arguments based on analysis type
            if analysis_type == 'lump':
                task_args = [
                    (idx, input_arrays_flat, param_shapes_array) 
                    for idx in chunk_indices
                ]
            elif analysis_type == 'T_seeded':
                task_args = [
                    (idx, input_arrays_flat, param_shapes_array, total_time, vector_length) 
                    for idx in chunk_indices
                ]
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            # Parallel computation
            chunk_results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(compute_function)(*args) for args in task_args
            )
            
            # Buffer results
            for i, result in enumerate(chunk_results):
                abs_idx = chunk_start + i
                buffer_results.append(result)
                buffer_indices.append(abs_idx)
                
                # Track successes
                if result.get('sol_success', False):
                    successful_count += 1
                
                # Update progress bar periodically
                if i % 10 == 0:
                    overall_pbar.update(min(10, len(chunk_results) - i))
            
            # Update remaining progress
            remainder = len(chunk_results) % 10
            if remainder > 0:
                overall_pbar.update(remainder)
            
            processed_count += len(chunk_results)
            
            # Write buffered results if buffer is full
            if len(buffer_results) >= buffer_write_size:
                _write_results_to_hdf5(
                    datasets, buffer_results, buffer_indices,
                    data_fields, vector_fields, vector_length
                )
                buffer_results.clear()
                buffer_indices.clear()
                h5_file.flush()
            
            # Update progress bar postfix
            success_rate = (successful_count / processed_count) * 100 if processed_count else 0
            overall_pbar.set_postfix({
                "Success": f"{success_rate:.1f}%",
                "Chunk": f"{len(chunk_indices)}"
            })
        
        # Write any remaining buffered results
        if buffer_results:
            _write_results_to_hdf5(
                datasets, buffer_results, buffer_indices,
                data_fields, vector_fields, vector_length
            )
            h5_file.flush()
        
        overall_pbar.close()
        
        # Final metadata
        end_time = time.time()
        h5_file.attrs.update({
            'successful_computations': successful_count,
            'computation_end_time': end_time,
            'total_computation_time': end_time - start_time
        })
    
    # Return statistics
    stats = {
        'total_combinations': int(n_combinations),
        'processed': processed_count,
        'successful': successful_count,
        'success_rate': (successful_count / processed_count) * 100 if processed_count else 0,
        'computation_time': end_time - start_time
    }
    
    return stats


def _write_results_to_hdf5(
    datasets: Dict[str, Any],
    results: List[Dict],
    indices: List[int],
    data_fields: List[str],
    vector_fields: List[str],
    vector_length: int
) -> None:
    """
    Write buffered results to HDF5 datasets.
    
    Args:
        datasets: Dictionary of HDF5 datasets
        results: List of result dictionaries
        indices: List of indices corresponding to results
        data_fields: List of all field names
        vector_fields: List of vector field names
        vector_length: Expected length of vector fields
    """
    for result, idx in zip(results, indices):
        for field in data_fields:
            value = result.get(field, None)
            
            if field == 'error':
                # Handle error strings
                if value is None or (isinstance(value, float) and not np.isfinite(value)):
                    datasets[field][idx] = ""
                else:
                    datasets[field][idx] = str(value)
                    
            elif field == 'sol_success':
                # Handle boolean
                datasets[field][idx] = bool(value) if value is not None else False
                
            elif field in vector_fields:
                # Handle vector data
                arr = np.asarray(value)
                if arr.ndim == 0 or arr.size == 0:
                    # Scalar or empty - fill with single value
                    scalar = float(value) if value is not None else np.nan
                    datasets[field][idx, :] = np.full(vector_length, scalar)
                elif arr.shape[0] == vector_length:
                    # Correct length
                    datasets[field][idx, :] = arr
                else:
                    # Wrong length - pad or truncate
                    padded = np.full(vector_length, np.nan)
                    copy_length = min(vector_length, arr.size)
                    padded[:copy_length] = arr[:copy_length]
                    datasets[field][idx, :] = padded
                    
            else:
                # Handle scalar data
                if value is None or (isinstance(value, float) and not np.isfinite(value)):
                    datasets[field][idx] = np.nan
                elif isinstance(value, str):
                    datasets[field][idx] = np.nan
                elif hasattr(value, "__len__") and not isinstance(value, str):
                    # Array-like - take last value
                    try:
                        scalar_value = float(value[-1]) if len(value) > 0 else np.nan
                    except Exception:
                        scalar_value = np.nan
                    datasets[field][idx] = scalar_value
                else:
                    # Direct scalar
                    try:
                        datasets[field][idx] = float(value)
                    except Exception:
                        datasets[field][idx] = np.nan


def print_parametric_summary(
    output_file: str,
    stats: Dict[str, Any],
    verbose: bool = True
) -> None:
    """
    Print summary statistics from parametric analysis.
    
    Args:
        output_file: Path to output HDF5 file
        stats: Statistics dictionary from run_parametric_analysis
        verbose: Whether to print detailed information
    """
    if not verbose:
        return
    
    print(f"\n{'='*60}")
    print("PARAMETRIC ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Processed: {stats['processed']:,} combinations")
    print(f"✅ Successful: {stats['successful']:,} ({stats['success_rate']:.1f}%)")
    print(f"⏱️  Computation time: {stats['computation_time']:.2f} seconds")
    print(f"📁 Results saved to: {output_file}")
    
    # Get file size
    import os
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024**2)  # MB
        print(f"💾 File size: {file_size:.1f} MB")
    
    print(f"{'='*60}\n")
