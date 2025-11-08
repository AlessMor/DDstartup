import numpy as np
import chaospy as cp
import time
import h5py
from joblib import Parallel, delayed
from numba import njit


@njit(cache=True)
def convert_sobol_sample_to_linear_index(sample_idx, n_samples):
    """
    Convert Sobol sample index to linear index format.
    
    For Sobol, we use direct indexing since samples are pre-generated.
    This is a placeholder for consistency with parametric interface.
    
    Args:
        sample_idx: Sobol sample index
        n_samples: Total number of samples (unused, for interface compatibility)
        
    Returns:
        Linear index (same as sample_idx)
    """
    return sample_idx


def run_sample_optimized(sample_idx, sample_values, 
                         compute_single_combination, analysis_type, max_simulation_time, vector_length):
    """
    Optimized single sample evaluation for Sobol analysis.
    
    Uses pre-allocated parameter arrays and leverages JIT-compiled
    compute_single_combination for speed. Minimizes memory allocation
    by using shared read-only arrays.
    
    Args:
        sample_idx: Index of Sobol sample to evaluate
        sample_values: Array of parameter values for this sample (all parameters)
        compute_single_combination: JIT-compiled worker function
        analysis_type: 'T_seeded' or 'lump'
        max_simulation_time: Maximum simulation time
        vector_length: Length of vector outputs
        
    Returns:
        Result dictionary from compute_single_combination
    """
    try:
        # Convert sample to format expected by compute_single_combination
        # Each parameter becomes a single-element array
        input_arrays_sobol = [np.array([val]) for val in sample_values]
        param_shapes_array = np.ones(len(sample_values), dtype=np.int64)
        
        # Direct call with linear index = 0 (single sample evaluation)
        if analysis_type == 'T_seeded':
            result = compute_single_combination(0, input_arrays_sobol, 
                                               param_shapes_array, max_simulation_time, vector_length)
        else:
            result = compute_single_combination(0, input_arrays_sobol, 
                                               param_shapes_array)
        result['sample_idx'] = sample_idx
        return result
    except Exception as e:
        import traceback
        error_msg = f"{str(e)} | Traceback: {traceback.format_exc()}"
        return {'sample_idx': sample_idx, 'error': error_msg, 'sol_success': False}

def compute_sobol_indices_parallel(
    input_data, param_names, N_SAMPLES, order, analysis_type, max_simulation_time,
    vector_length, compute_single_combination, n_jobs, verbose
):
    """
    Compute Sobol sensitivity indices using parallel computation.
    
    Generates Sobol samples, computes outputs in parallel, then analyzes sensitivities
    using SALib.
    
    Args:
        input_data: Dictionary with parameter arrays and metadata
        param_names: List of parameter names
        N_SAMPLES: Number of Sobol samples to generate
        order: Sobol analysis order (2 or 3)
        analysis_type: 'lump' or 'T_seeded'
        max_simulation_time: Maximum simulation time
        vector_length: Number of time points (T_seeded only)
        compute_single_combination: Physics compute function
        n_jobs: Number of parallel jobs
        verbose: Whether to print progress
    """
    t_start = time.time()
    
    # 1. Get parameter bounds
    t1 = time.time()
    param_bounds = [(arr.min(), arr.max()) for arr in input_data.values()]
    if verbose:
        print("\n" + "=" * 60)
        print("SOBOL SENSITIVITY ANALYSIS - SETUP")
        print("=" * 60)
        print("Parameter bounds for Sobol analysis:")
        for name, (low, high) in zip(param_names, param_bounds):
            print(f"  {name}: min={low}, max={high}")

    # 2. Separate variable and constant parameters
    variable_params = [(i, name, low, high) for i, (name, (low, high)) in enumerate(zip(param_names, param_bounds)) if high > low]
    constant_params = [(i, name, low) for i, (name, (low, high)) in enumerate(zip(param_names, param_bounds)) if high == low]

    if constant_params:
        print("The following parameters are constants in Sobol analysis:")
        for i, name, val in constant_params:
            print(f"  {name}: {val}")

    if not variable_params:
        raise ValueError("No parameters with a valid range for Sobol analysis.")

    # 3. Build distributions for variable parameters
    sobol_param_names = [name for i, name, low, high in variable_params]
    distr_params = {name: cp.Uniform(low, high) for i, name, low, high in variable_params}
    joint_dist = cp.J(*distr_params.values())

    # 4. Generate Sobol samples for variable parameters
    t2 = time.time()
    if verbose:
        print(f"⏱️  Parameter setup: {t2-t1:.2f}s")
        print(f"\nGenerating {N_SAMPLES} Sobol samples...")
    
    samples_var = joint_dist.sample(N_SAMPLES, rule='sobol').T  # shape: (N_SAMPLES, n_var_params)
    
    t3 = time.time()
    if verbose:
        print(f"✅ Samples generated in {t3-t2:.2f}s")

    # 5. For each sample, build the full parameter vector (variable + constants)
    def build_full_sample(sample_var):
        full = []
        var_iter = iter(sample_var)
        for i in range(len(param_names)):
            if any(i == idx for idx, _, _, _ in variable_params):
                full.append(next(var_iter))
            else:
                # Find the constant value
                val = [val for idx, _, val in constant_params if idx == i][0]
                full.append(val)
        return np.array(full)

    samples_full = np.array([build_full_sample(sample_var) for sample_var in samples_var])
    
    t4 = time.time()
    if verbose:
        print(f"⏱️  Sample preparation: {t4-t3:.2f}s")

    # 6. Parallel evaluation using joblib with tqdm progress bar
    if verbose:
        print("\n" + "=" * 60)
        print(f"PARALLEL EVALUATION - {N_SAMPLES} SAMPLES")
        print("=" * 60)
    
    # Import tools for field names and progress bar
    from ddstartup.utils.tools import inputs_names, outputs_names
    from tqdm import tqdm
    
    # Get number of jobs from config or use all cores
    n_jobs = -1  # Use all available cores
    chunk_size = max(1, N_SAMPLES // 20)  # Process in 20 chunks
    
    if verbose:
        print(f"Workers: {n_jobs}, Chunk size: {chunk_size}\n")
    
    # Initialize progress bar
    pbar = tqdm(total=N_SAMPLES, desc="Computing samples", unit="sample", disable=not verbose)
    
    # Parallel execution with chunking (like parametric analysis)
    results = [None] * N_SAMPLES
    successful_count = 0
    
    for chunk_start in range(0, N_SAMPLES, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N_SAMPLES)
        chunk_indices = range(chunk_start, chunk_end)
        
        # Prepare chunk samples
        chunk_samples = samples_full[chunk_start:chunk_end]
        
        # Parallel execution for this chunk
        chunk_results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(run_sample_optimized)(
                i,
                sample,  # Pass full sample array directly
                compute_single_combination,
                analysis_type,
                max_simulation_time,
                vector_length
            )
            for i, sample in zip(chunk_indices, chunk_samples)
        )
        
        # Store results and track successes
        for i, result in enumerate(chunk_results):
            abs_idx = chunk_start + i
            results[abs_idx] = result
            if result.get('sol_success', False):
                successful_count += 1
        
        # Update progress bar
        pbar.update(len(chunk_results))
        success_rate = (successful_count / (chunk_end)) * 100
        pbar.set_postfix({"Success": f"{success_rate:.1f}%"})
    
    pbar.close()
    
    t5 = time.time()
    if verbose:
        print(f"\n✅ Parallel evaluation complete in {t5-t4:.2f}s")
        print(f"⚡ Average: {(t5-t4)/N_SAMPLES*1000:.2f} ms/sample")
        print(f"🚀 Throughput: {N_SAMPLES/(t5-t4):.1f} samples/s")

    # 7. Extract output metric (unrealized_profits) - vectorized approach
    if verbose:
        print("\nExtracting results...")
    
    # Pre-allocate arrays for better performance
    n_results = len(results)
    unrealized_profits = np.full(n_results, np.nan, dtype=np.float64)
    
    # Vectorized extraction with error checking
    for i, r in enumerate(results):
        if r is not None and isinstance(r, dict):
            ug = r.get('unrealized_profits', np.nan)
            if isinstance(ug, (int, float, np.number)) and np.isfinite(ug):
                unrealized_profits[i] = float(ug)
    
    # Filter valid results
    valid_mask = np.isfinite(unrealized_profits)
    n_valid = np.sum(valid_mask)
    
    # Debug: check first result
    if verbose and len(results) > 0:
        print(f"\nDEBUG - First result keys: {list(results[0].keys()) if results[0] else 'None'}")
        if results[0] and 'unrealized_profits' in results[0]:
            print(f"DEBUG - First unrealized_profits: {results[0]['unrealized_profits']}")
        print(f"DEBUG - First 10 unrealized_profits values: {unrealized_profits[:10]}")
    
    if verbose:
        print(f"\nValid results: {n_valid}/{N_SAMPLES} ({100*n_valid/N_SAMPLES:.1f}%)")
    
    if n_valid < 10:
        print("❌ Not enough valid samples for Sobol analysis (minimum 10 required).")
        return None
    
    # Extract only variable parameter columns for valid samples
    # This reduces PCE fitting dimensionality (ignore constant parameters)
    valid_samples_var = samples_var[valid_mask]  # Only variable parameters
    valid_dollars = unrealized_profits[valid_mask]

    # 8. Fit PCE surrogate model and compute Sobol indices
    # Use variable parameters only (more efficient)
    t6 = time.time()
    
    if verbose:
        print(f"\nFitting PCE surrogate model (order={order}, {len(sobol_param_names)} variables)...")
    
    poly_expansion = cp.generate_expansion(order, joint_dist)
    pce_model = cp.fit_regression(poly_expansion, valid_samples_var.T, valid_dollars)
    
    if verbose:
        print("Computing Sobol indices...")
    
    sobol_first = cp.Sens_m(pce_model, joint_dist)
    sobol_total = cp.Sens_t(pce_model, joint_dist)
    
    t7 = time.time()
    if verbose:
        print(f"✅ Completed in {t7-t6:.2f}s")

    # 9. Save all results to HDF5 (consistent with parametric analysis)
    save_start = time.time()
    
    if verbose:
        print("\nSaving results to HDF5...")
    
    from pathlib import Path
    
    data_fields = list(dict.fromkeys(inputs_names + outputs_names))
    vector_fields = ['N_ofc', 'N_ifc', 'N_stor', 'n_T', 'n_D', 'P_DDn', 'P_DDp', 'P_DT', 'TBE']
    
    # Save to HDF5 with progress bar
    with h5py.File(output_file, 'w') as h5_file:
        # Add metadata
        h5_file.attrs.update({
            'method': 'sobol',
            'analysis_type': analysis_type,
            'N_SAMPLES': N_SAMPLES,
            'n_valid': n_valid,
            'pce_order': order,
            'n_variable_params': len(sobol_param_names),
            'n_constant_params': len(constant_params),
            'variable_param_names': ','.join(sobol_param_names),
            'computation_time': t7 - t_start,
            'evaluation_time': t5 - t4,
            'vector_length': vector_length,
        })
        
        # Save Sobol indices
        sobol_group = h5_file.create_group('sobol_indices')
        sobol_group.create_dataset('first_order', data=sobol_first, compression='gzip')
        sobol_group.create_dataset('total_order', data=sobol_total, compression='gzip')
        sobol_group.create_dataset('interaction', data=sobol_total - sobol_first, compression='gzip')
        sobol_group.create_dataset('parameter_names', 
                                   data=[p.encode('utf-8') for p in sobol_param_names],
                                   dtype=h5py.string_dtype(encoding='utf-8'))
        
        # Pre-allocate all datasets (like parametric analysis)
        datasets = {}
        chunk_size_h5 = min(1000, N_SAMPLES)
        
        for field in data_fields:
            if field in vector_fields:
                datasets[field] = h5_file.create_dataset(
                    field, (N_SAMPLES, vector_length), dtype=np.float64,
                    chunks=(chunk_size_h5, vector_length), compression='lzf'
                )
            elif field == 'error':
                datasets[field] = h5_file.create_dataset(
                    field, (N_SAMPLES,), dtype=h5py.string_dtype(encoding='utf-8'),
                    chunks=(chunk_size_h5,), compression='lzf'
                )
            elif field == 'sol_success':
                datasets[field] = h5_file.create_dataset(
                    field, (N_SAMPLES,), dtype=bool,
                    chunks=(chunk_size_h5,), compression='lzf'
                )
            else:
                datasets[field] = h5_file.create_dataset(
                    field, (N_SAMPLES,), dtype=np.float64,
                    chunks=(chunk_size_h5,), compression='lzf'
                )
        
        # Write results with progress bar
        write_pbar = tqdm(total=N_SAMPLES, desc="Writing to HDF5", unit="sample", disable=not verbose)
        
        write_batch_size = 1000
        for batch_start in range(0, N_SAMPLES, write_batch_size):
            batch_end = min(batch_start + write_batch_size, N_SAMPLES)
            batch_results = results[batch_start:batch_end]
            
            for field in data_fields:
                if field in vector_fields:
                    # 2D array for vector fields
                    batch_data = np.full((len(batch_results), vector_length), np.nan, dtype=np.float64)
                    for i, r in enumerate(batch_results):
                        if r is not None and field in r:
                            data = r[field]
                            if hasattr(data, '__len__'):
                                batch_data[i, :min(len(data), vector_length)] = data[:vector_length]
                            else:
                                batch_data[i, 0] = data
                    datasets[field][batch_start:batch_end, :] = batch_data
                    
                elif field == 'error':
                    batch_data = [r.get('error', '') if r is not None else '' for r in batch_results]
                    # Convert all error values to strings before encoding
                    datasets[field][batch_start:batch_end] = [str(e).encode('utf-8') for e in batch_data]
                    
                elif field == 'sol_success':
                    batch_data = [r.get('sol_success', False) if r is not None else False for r in batch_results]
                    datasets[field][batch_start:batch_end] = batch_data
                    
                else:
                    batch_data = np.full(len(batch_results), np.nan, dtype=np.float64)
                    for i, r in enumerate(batch_results):
                        if r is not None and field in r:
                            val = r[field]
                            if isinstance(val, (int, float, np.number)):
                                batch_data[i] = float(val)
                    datasets[field][batch_start:batch_end] = batch_data
            
            write_pbar.update(len(batch_results))
            h5_file.flush()
        
        write_pbar.close()
        
        # Save Sobol samples
        samples_group = h5_file.create_group('sobol_samples')
        samples_group.create_dataset('samples_full', data=samples_full, compression='gzip')
        samples_group.create_dataset('samples_variable', data=samples_var, compression='gzip')
        samples_group.create_dataset('valid_mask', data=valid_mask, compression='gzip')
        
        # Save parameter information
        param_group = h5_file.create_group('parameter_info')
        param_group.create_dataset('all_param_names',
                                   data=[p.encode('utf-8') for p in param_names],
                                   dtype=h5py.string_dtype(encoding='utf-8'))
        for i, (idx, name, low, high) in enumerate(variable_params):
            param_group.attrs[f'variable_param_{i}_name'] = name
            param_group.attrs[f'variable_param_{i}_min'] = float(low)
            param_group.attrs[f'variable_param_{i}_max'] = float(high)
        for i, (idx, name, val) in enumerate(constant_params):
            param_group.attrs[f'constant_param_{i}_name'] = name
            param_group.attrs[f'constant_param_{i}_value'] = float(val)
    
    t8 = time.time()
    t_total = t8 - t_start
    
    if verbose:
        print(f"✅ Results saved to HDF5: {output_file}")
        print(f"   Saving time: {t8 - save_start:.2f}s\n")
        print("=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Setup & sampling:      {t4-t1:.2f}s ({100*(t4-t1)/t_total:.1f}%)")
        print(f"Parallel evaluation:   {t5-t4:.2f}s ({100*(t5-t4)/t_total:.1f}%)")
        print(f"PCE fitting:           {t7-t6:.2f}s ({100*(t7-t6)/t_total:.1f}%)")
        print(f"Saving results:        {t8-save_start:.2f}s ({100*(t8-save_start)/t_total:.1f}%)")
        print("-" * 60)
        print(f"TOTAL TIME:            {t_total:.2f}s")
        print(f"THROUGHPUT:            {N_SAMPLES/t_total:.1f} samples/s")
        print("=" * 60 + "\n")

    return {
        'sobol_first': sobol_first,
        'sobol_total': sobol_total,
        'param_names': list(distr_params.keys()),
        'output_file': str(output_file),
        'n_samples': N_SAMPLES,
        'n_valid': n_valid,
        'success_rate': n_valid / N_SAMPLES,
        'computation_time': t_total,
        'evaluation_time': t5 - t4,
        'pce_fitting_time': t7 - t6,
        'saving_time': t8 - save_start,
        'throughput': N_SAMPLES / t_total
    }