"""
Parallel computation utilities for parametric analysis.
Handles chunked processing, HDF5 output, and parallelization logic.
"""

import os
import time
import itertools
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from .ode_solver import solve_single_combination

# Add h5py for HDF5 appends
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


def h5_append_rows(h5_path: str, dataset_name: str, rows_array: np.ndarray,
                   column_names: list[str], chunks_rows: int | None = None,
                   compression: str | None = None) -> int:
    """Append 2D numpy array rows to an HDF5 dataset, creating it if needed.
    Returns number of rows written. Does nothing if rows_array is empty.
    """
    if rows_array is None or len(rows_array) == 0:
        return 0
    os.makedirs(os.path.dirname(h5_path) or '.', exist_ok=True)
    with h5py.File(h5_path, 'a') as f:
        n_rows, n_cols = rows_array.shape
        if dataset_name not in f:
            # Initialize chunking
            chunk_r = min(chunks_rows or max(1000, n_rows), max(1000, n_rows))
            dset = f.create_dataset(
                dataset_name,
                data=rows_array,
                maxshape=(None, n_cols),
                chunks=(chunk_r, n_cols),
                compression=compression
            )
            dset.attrs['columns'] = [c.encode('utf-8') for c in column_names]
            return n_rows
        else:
            dset = f[dataset_name]
            current = dset.shape[0]
            new_size = current + n_rows
            dset.resize((new_size, n_cols))
            dset[current:new_size, :] = rows_array
            return n_rows


def update_statistics(stats, chunk_array):
    """Update statistics with results from a chunk."""
    t_startup_col = chunk_array[:, -8]
    e_lost_col = chunk_array[:, -6]
    dollar_lost_col = chunk_array[:, -5]
    finite_mask = np.isfinite(t_startup_col)
    successful_count = int(np.sum(finite_mask))
    failed_count = len(chunk_array) - successful_count
    
    stats['successful_count'] += successful_count
    stats['failed_count'] += failed_count
    
    if successful_count > 0:
        st = t_startup_col[finite_mask]
        stats['t_startup_sum'] += float(np.sum(st))
        stats['t_startup_min'] = min(stats['t_startup_min'], float(np.min(st)))
        stats['t_startup_max'] = max(stats['t_startup_max'], float(np.max(st)))
        
        valid_e = e_lost_col[finite_mask]
        valid_d = dollar_lost_col[finite_mask]
        valid_e = valid_e[~np.isnan(valid_e)]
        valid_d = valid_d[~np.isnan(valid_d)]
        
        if len(valid_e) > 0:
            stats['e_lost_sum'] += float(np.sum(valid_e))
        if len(valid_d) > 0:
            stats['dollar_lost_sum'] += float(np.sum(valid_d))


def run_chunked_analysis(input_data, total_time, total_combinations, h5_filename, 
                        column_names, profiler):
    """Run analysis with chunked processing for large parameter spaces."""
    # Use optimized settings with dynamic tuning
    n_jobs = profiler.optimization['n_jobs']
    chunk_size = profiler.optimization['chunk_size']
    batch_size = profiler.optimization['batch_size']
    backend = profiler.optimization['backend']
    use_parallel_chunks = profiler.optimization['use_parallel_chunks']
    
    print("Using chunked processing with HDF5 appends...")
    
    # Check for resume capability
    dataset_name = 'results'
    resume_from = 0
    if HDF5_AVAILABLE and os.path.exists(h5_filename):
        try:
            with h5py.File(h5_filename, 'r') as f:
                if dataset_name in f:
                    resume_from = int(f[dataset_name].shape[0])
                    print(f"Will resume appending to existing file with {resume_from:,} rows")
        except Exception as e:
            print(f"Resume check failed ({e}); starting a new file on first write")
    
    # Statistics tracking (memory-efficient)
    stats = {
        'successful_count': 0,
        'failed_count': 0,
        't_startup_sum': 0.0,
        't_startup_min': float('inf'),
        't_startup_max': 0.0,
        'e_lost_sum': 0.0,
        'dollar_lost_sum': 0.0
    }
    
    processed = 0
    param_ranges = [range(data.shape[0]) for data in input_data]
    
    def optimized_param_generator():
        for combo in itertools.product(*param_ranges):
            yield combo
    
    param_iter = optimized_param_generator()
    
    # Skip to resume point
    if resume_from > 0:
        for _ in range(resume_from):
            next(param_iter, None)
        processed = resume_from
    
    main_pbar = tqdm(
        total=total_combinations,
        desc="🚀 Parametric Analysis",
        unit="combo",
        unit_scale=True,
        dynamic_ncols=True,
        position=0,
        initial=processed,
        smoothing=0.1,
        miniters=max(1, total_combinations // 1000),  # Adaptive update frequency
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )
    
    while processed < total_combinations:
        chunk = list(itertools.islice(param_iter, chunk_size))
        if not chunk:
            break
        chunk_num = processed // chunk_size + 1
        
        if use_parallel_chunks:
            dynamic_batch_size = max(1, min(batch_size, len(chunk) // max(1, n_jobs)))
            chunk_results = Parallel(
                n_jobs=n_jobs,
                verbose=0,
                backend=backend,
                batch_size=dynamic_batch_size,
                prefer="threads",
                pre_dispatch='2*n_jobs',
                temp_folder=None,
                max_nbytes=None
            )(
                delayed(solve_single_combination)(param_combo, input_data, total_time)
                for param_combo in chunk
            )
        else:
            chunk_results = []
            for param_combo in chunk:
                chunk_results.append(solve_single_combination(param_combo, input_data, total_time))
        
        chunk_array = np.array(chunk_results, dtype=np.float64)
        
        # Update stats vectorized
        update_statistics(stats, chunk_array)
        
        # Append this chunk directly to HDF5
        h5_append_rows(
            h5_filename,
            dataset_name,
            chunk_array,
            column_names,
            chunks_rows=chunk_size,
            compression=None  # set to 'lzf' for light compression
        )
        
        processed += len(chunk_array)
        
        # Update progress bar with success rate
        success_rate = stats['successful_count'] / max(processed, 1) * 100
        main_pbar.set_postfix({
            'Success': f'{success_rate:.1f}%',
            'Chunk': f'{chunk_num}',
            'Successful': stats['successful_count'],
            'Failed': stats['failed_count']
        })
        main_pbar.update(len(chunk_array))
    
    main_pbar.close()
    return stats


def run_small_analysis(input_data, total_time, total_combinations, h5_filename, 
                      column_names, profiler):
    """Run analysis for small parameter spaces with direct processing."""
    print("Using direct processing with HDF5 appends...")
    
    n_jobs = profiler.optimization['n_jobs']
    batch_size = profiler.optimization['batch_size']
    dataset_name = 'results'
    
    stats = {
        'successful_count': 0,
        'failed_count': 0,
        't_startup_sum': 0.0,
        't_startup_min': float('inf'),
        't_startup_max': 0.0,
        'e_lost_sum': 0.0,
        'dollar_lost_sum': 0.0
    }
    
    param_ranges = [range(data.shape[0]) for data in input_data]
    
    def param_combinations_generator():
        for combo in itertools.product(*param_ranges):
            yield combo
    
    pbar = tqdm(
        total=total_combinations,
        desc="🚀 Processing Combinations",
        unit="combo",
        unit_scale=True,
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )
    
    batch = []
    processed = 0
    for combo in param_combinations_generator():
        batch.append(combo)
        if len(batch) >= batch_size:
            batch_results = Parallel(
                n_jobs=n_jobs,
                verbose=0,
                backend='threading',
                batch_size=1,
                prefer="threads"
            )(
                delayed(solve_single_combination)(param_combo, input_data, total_time)
                for param_combo in batch
            )
            chunk_array = np.array(batch_results, dtype=np.float64)
            h5_append_rows(h5_filename, dataset_name, chunk_array, column_names, 
                          chunks_rows=max(batch_size, 1000))
            
            # Update stats
            update_statistics(stats, chunk_array)
            
            processed += len(batch)
            
            # Update progress bar with real-time statistics
            success_rate = stats['successful_count'] / max(processed, 1) * 100
            pbar.set_postfix({
                'Success': f'{success_rate:.1f}%',
                'Successful': stats['successful_count'],
                'Failed': stats['failed_count']
            })
            pbar.update(len(batch))
            batch = []
    
    # Flush any remaining
    if batch:
        batch_results = [solve_single_combination(param_combo, input_data, total_time) 
                        for param_combo in batch]
        chunk_array = np.array(batch_results, dtype=np.float64)
        h5_append_rows(h5_filename, dataset_name, chunk_array, column_names, 
                      chunks_rows=max(batch_size, 1000))
        update_statistics(stats, chunk_array)
        processed += len(batch)
        pbar.update(len(batch))
    
    pbar.close()
    return stats


def run_parallel_analysis(input_data, total_time, total_combinations, h5_filename, 
                         column_names, profiler):
    """Main function to run parallel analysis with appropriate strategy."""
    print(f"Output file: {h5_filename} (HDF5, append-per-batch)")
    
    # Show diagnostic information for reasonable-sized datasets
    if total_combinations <= 1000:
        param_ranges = [range(data.shape[0]) for data in input_data]
        param_iter_test = itertools.product(*param_ranges)
        first_few = list(itertools.islice(param_iter_test, 3))
        print(f"First few parameter combinations to test:")
        for i, combo in enumerate(first_few):
            print(f"  {i+1}: {combo}")
    
    # Quick test of solve function (silent for large datasets)
    if total_combinations <= 100000:
        print("Testing solve function with first combination...")
        try:
            param_ranges = [range(data.shape[0]) for data in input_data]
            first_combo = next(itertools.product(*param_ranges))
            test_result = solve_single_combination(first_combo, input_data, total_time)
            print(f"Test successful! Result length: {len(test_result)}")
            print(f"Sample values: t_startup={test_result[-8]:.2f}, P_fusion={test_result[-7]:.2e}")
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            print("Switching to sequential mode for debugging...")
            # Update profiler settings for debug mode
            profiler.optimization['use_parallel_chunks'] = False
            profiler.optimization['n_jobs'] = 1
    
    # Choose processing strategy based on parameter space size
    if total_combinations > 10000:  # > 10K combinations
        return run_chunked_analysis(input_data, total_time, total_combinations, 
                                   h5_filename, column_names, profiler)
    else:
        return run_small_analysis(input_data, total_time, total_combinations, 
                                 h5_filename, column_names, profiler)