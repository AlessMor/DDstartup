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
        
        # Pre-allocate datasets WITH LZ4 compression
        # LZ4 is 10x faster than gzip while still achieving good compression
        # With 4 parallel writers, compression overhead is distributed
        # Result: Fast computation + reasonable file sizes
        try:
            import hdf5plugin  # Provides LZ4 compression for HDF5
            use_lz4 = True
        except ImportError:
            use_lz4 = False
            if verbose:
                print("⚠️  hdf5plugin not found - using gzip compression")
                print("   Install with: pip install hdf5plugin")
                print("   LZ4 is 10x faster than gzip!\n")
        
        datasets = {}
        for field in data_fields:
            if field in vector_fields:
                # 2D array for vector fields - LZ4 or fast gzip
                if use_lz4:
                    datasets[field] = h5_file.create_dataset(
                        field,
                        (n_combinations, vector_length),
                        dtype=np.float64,
                        chunks=(min(chunk_size, n_combinations), vector_length),
                        **hdf5plugin.LZ4(nbytes=0)  # Ultra-fast compression
                    )
                else:
                    datasets[field] = h5_file.create_dataset(
                        field,
                        (n_combinations, vector_length),
                        dtype=np.float64,
                        chunks=(min(chunk_size, n_combinations), vector_length),
                        compression='gzip',
                        compression_opts=1  # Fastest gzip fallback
                    )
            elif field == 'error':
                # String field for errors - LZ4 or fast gzip
                if use_lz4:
                    datasets[field] = h5_file.create_dataset(
                        field,
                        (n_combinations,),
                        dtype=h5py.string_dtype(encoding='utf-8'),
                        chunks=(min(chunk_size, n_combinations),),
                        **hdf5plugin.LZ4(nbytes=0)
                    )
                else:
                    datasets[field] = h5_file.create_dataset(
                        field,
                        (n_combinations,),
                        dtype=h5py.string_dtype(encoding='utf-8'),
                        chunks=(min(chunk_size, n_combinations),),
                        compression='gzip',
                        compression_opts=1
                    )
            elif field == 'sol_success':
                # Boolean field - LZ4 or fast gzip
                if use_lz4:
                    datasets[field] = h5_file.create_dataset(
                        field,
                        (n_combinations,),
                        dtype=bool,
                        chunks=(min(chunk_size, n_combinations),),
                        **hdf5plugin.LZ4(nbytes=0)
                    )
                else:
                    datasets[field] = h5_file.create_dataset(
                        field,
                        (n_combinations,),
                        dtype=bool,
                        chunks=(min(chunk_size, n_combinations),),
                        compression='gzip',
                        compression_opts=1
                    )
            else:
                # Scalar fields - LZ4 or fast gzip
                if use_lz4:
                    datasets[field] = h5_file.create_dataset(
                        field,
                        (n_combinations,),
                        dtype=np.float64,
                        chunks=(min(chunk_size, n_combinations),),
                        **hdf5plugin.LZ4(nbytes=0)
                    )
                else:
                    datasets[field] = h5_file.create_dataset(
                        field,
                        (n_combinations,),
                        dtype=np.float64,
                        chunks=(min(chunk_size, n_combinations),),
                        compression='gzip',
                        compression_opts=1
                    )
        
        # Save parameter grids
        param_group = h5_file.create_group('parameter_fields')
        for name, arr in zip(param_names, input_arrays):
            param_group.create_dataset(f'{name}_values', data=arr, compression='gzip')
        
        # Prime Numba compilation if T_seeded (before progress bar)
        if analysis_type == 'T_seeded' and verbose:
            print("Priming Numba compilation...")
            try:
                compute_function(0, input_arrays_flat, param_shapes_array, total_time, vector_length)
            except Exception as e:
                print(f"Warning during Numba priming: {e}")
        
        # Print parallelization info (before progress bar)
        if verbose:
            print(f"Starting BATCH PIPELINE computation with {n_jobs} workers...")
            print(f"Processing {n_combinations:,} combinations in batches of 5,000")
            compression_type = "LZ4 (ultra-fast)" if use_lz4 else "gzip-1 (fast fallback)"
            print(f"Compression: {compression_type} with VECTORIZED bulk writes")
            print(f"Architecture: COMPUTE phase (all {n_jobs} cores) → WRITE phase (<1s)")
            print(f"Expected: First ~{n_jobs * 2} tasks slow (Numba compilation), then fast")
            print(f"Monitor the speed - smooth progress with minimal write overhead!")
        
        # Initialize tqdm with dynamic status bar
        overall_pbar = tqdm(
            total=n_combinations, 
            desc="🔄 COMPUTE", 
            unit="comb",
            disable=not verbose,
            mininterval=0.5,  # Update twice per second for responsive status
            miniters=50,  # Update every 50 iterations
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # BATCH PIPELINE: Compute full batch, then write in parallel
        # Vectorized writes are MUCH faster (10-100x), can use larger batches
        # Larger batches = better compute efficiency, minimal write overhead
        write_batch_size = 5000  # Large batches with fast vectorized writes
        
        # Global error logging flag (across all chunks)
        first_error_logged = False
        
        # Ensure integer types for range()
        n_combinations = int(n_combinations)
        chunk_size = int(chunk_size)
        
        # Use ProcessPoolExecutor for persistent workers
        from concurrent.futures import ProcessPoolExecutor
        import concurrent.futures
        
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Process in large batches: compute all, then write all
                for batch_start in range(0, n_combinations, write_batch_size):
                    batch_end = min(batch_start + write_batch_size, n_combinations)
                    batch_size_actual = batch_end - batch_start
                    
                    # === COMPUTE PHASE: All workers compute in parallel ===
                    batch_indices = np.arange(batch_start, batch_end)
                    
                    # Prepare task arguments based on analysis type
                    if analysis_type == 'lump':
                        task_args = [
                            (idx, input_arrays_flat, param_shapes_array) 
                            for idx in batch_indices
                        ]
                    elif analysis_type == 'T_seeded':
                        task_args = [
                            (idx, input_arrays_flat, param_shapes_array, total_time, vector_length) 
                            for idx in batch_indices
                        ]
                    else:
                        raise ValueError(f"Unknown analysis type: {analysis_type}")
                    
                    # Submit all tasks for this batch
                    futures = {executor.submit(compute_function, *args): idx 
                              for idx, args in enumerate(task_args)}
                    
                    # Collect all results for this batch
                    batch_results = [None] * batch_size_actual
                    for future in concurrent.futures.as_completed(futures):
                        result_idx = futures[future]
                        try:
                            result = future.result()
                            batch_results[result_idx] = result
                        except Exception as exc:
                            batch_results[result_idx] = {'error': str(exc), 'sol_success': False}
                        
                        # Track successes
                        if batch_results[result_idx].get('sol_success', False):
                            successful_count += 1
                        else:
                            # Log first error for debugging (only once globally)
                            if not first_error_logged and verbose:
                                error_msg = batch_results[result_idx].get('error', 'Unknown error')
                                first_error_logged = True
                        
                        processed_count += 1
                        overall_pbar.update(1)
                    
                    # === WRITE PHASE: Write entire batch with compression ===
                    # Workers are idle during this phase, but write is fast with LZ4
                    # Update status to show we're writing
                    overall_pbar.set_description("💾 WRITE")
                    overall_pbar.refresh()  # Force immediate update
                    write_start = time.time()
                    
                    _write_results_to_hdf5(
                        datasets, batch_results, list(batch_indices),
                        data_fields, vector_fields, vector_length
                    )
                    
                    # Flush to disk after each batch (ensures data is written)
                    h5_file.flush()
                    
                    # Resume computing - show last write time and success rate
                    write_time = time.time() - write_start
                    success_rate = (successful_count / processed_count * 100) if processed_count > 0 else 0.0
                    overall_pbar.set_description(f"🔄 COMPUTE (write: {write_time:.2f}s, ✓ {success_rate:.1f}%)")
                    overall_pbar.refresh()  # Force immediate update
                    
        except Exception as e:
            if verbose:
                overall_pbar.close()
                print(f"\n❌ Error during computation: {e}")
                import traceback
                traceback.print_exc()
            raise
        finally:
            # Close progress bar
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
    Write buffered results to HDF5 datasets using VECTORIZED bulk writes.
    This is 10-100x faster than writing one result at a time!
    
    Args:
        datasets: Dictionary of HDF5 datasets
        results: List of result dictionaries
        indices: List of indices corresponding to results
        data_fields: List of all field names
        vector_fields: List of vector field names
        vector_length: Expected length of vector fields
    """
    batch_size = len(results)
    indices_array = np.array(indices)
    
    # Process each field with vectorized operations
    for field in data_fields:
        if field == 'error':
            # Collect all error strings
            error_values = [
                "" if (r.get(field) is None or (isinstance(r.get(field), float) and not np.isfinite(r.get(field))))
                else str(r.get(field))
                for r in results
            ]
            datasets[field][indices_array] = error_values
            
        elif field == 'sol_success':
            # Collect all boolean values
            bool_values = np.array([bool(r.get(field, False)) for r in results], dtype=bool)
            datasets[field][indices_array] = bool_values
            
        elif field in vector_fields:
            # Pre-allocate array for batch of vectors
            batch_array = np.full((batch_size, vector_length), np.nan, dtype=np.float64)
            
            for i, result in enumerate(results):
                value = result.get(field, None)
                arr = np.asarray(value)
                
                if arr.ndim == 0 or arr.size == 0:
                    # Scalar or empty - fill with single value
                    scalar = float(value) if value is not None else np.nan
                    batch_array[i, :] = scalar
                elif arr.shape[0] == vector_length:
                    # Correct length
                    batch_array[i, :] = arr
                else:
                    # Wrong length - pad or truncate
                    copy_length = min(vector_length, arr.size)
                    batch_array[i, :copy_length] = arr[:copy_length]
            
            # Single bulk write for entire batch!
            datasets[field][indices_array, :] = batch_array
            
        else:
            # Collect all scalar values
            scalar_values = []
            for result in results:
                value = result.get(field, None)
                
                if value is None or (isinstance(value, float) and not np.isfinite(value)):
                    scalar_values.append(np.nan)
                elif isinstance(value, str):
                    scalar_values.append(np.nan)
                elif hasattr(value, "__len__") and not isinstance(value, str):
                    # Array-like - take last value
                    try:
                        scalar_value = float(value[-1]) if len(value) > 0 else np.nan
                    except Exception:
                        scalar_value = np.nan
                    scalar_values.append(scalar_value)
                else:
                    # Direct scalar
                    try:
                        scalar_values.append(float(value))
                    except Exception:
                        scalar_values.append(np.nan)
            
            # Single bulk write for entire batch!
            datasets[field][indices_array] = np.array(scalar_values, dtype=np.float64)


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
        print(f"💾 File size: {file_size:.1f} MB (compressed with LZ4/gzip-1)")
    
    print(f"{'='*60}\n")
