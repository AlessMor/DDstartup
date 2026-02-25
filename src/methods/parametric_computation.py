"""
Parametric analysis computation module.

This module handles the parametric analysis workflow including:
- HDF5 file setup and dataset creation
- Parallel computation with ProcessPoolExecutor
- Progress tracking with tqdm
- Result writing and buffering
- Error handling and statistics

Performance optimizations:
- ProcessPoolExecutor for persistent workers (no fork overhead)
- Vectorized HDF5 writes (10-100x faster than individual writes)
- Pre-cached reaction rates for all unique temperatures
- LZ4 compression (10x faster than gzip, good compression)
- Batch pipeline: compute phase → write phase (maximizes parallelism)
"""

import h5py
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import time

# Import centralized parameter management
from src.utils.parameter_registry import SPECIES, get_registry, get_multispecies_registry
from src.utils.filters import apply_filter_to_combinations


def _compute_lump(linear_index, input_arrays_flat, param_shapes_array, reactivity_lookup=None):
    """
    Compute lump analysis for a single parameter combination.
    
    This function is defined at module level to allow pickling for multiprocessing.
    Handles parameter extraction, physics solver call, and postprocessing.
    
    Args:
        linear_index: Linear index into parameter grid
        input_arrays_flat: Flattened parameter arrays
        param_shapes_array: Shape of parameter grid
        reactivity_lookup: Optional dict containing pre-computed reactivities
    """
    from src.physics.lump_functions import lump_solver
    from src.physics.power_balance import compute_lump_powers_and_energies, calculate_P_aux_from_power_balance
    from src.economics.economics_functions import compute_economics_from_energies
    from src.utils.tools import index_to_params
    
    # Extract parameters efficiently with tuple unpacking
    idx = index_to_params(linear_index, param_shapes_array)
    (V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, P_aux, P_aux_DT_eq,
     TBR_DT, TBR_DDn, I_target, eta_th, capacity_factor, price_of_electricity) = (
        arr[idx[i]] for i, arr in enumerate(input_arrays_flat)
    )
    
    # Get reactivities from lookup table (much faster than computing)
    if reactivity_lookup is not None:
        T_key = round(T_i / 0.1) * 0.1
        sigmav_DD_p = reactivity_lookup['sigmav_DD_p'][T_key]
        sigmav_DD_n = reactivity_lookup['sigmav_DD_n'][T_key]
        sigmav_DT = reactivity_lookup['sigmav_DT'][T_key]
        sigmav_DHe3 = reactivity_lookup['sigmav_DHe3'][T_key]
    else:
        # Fallback to old cached method
        from src.utils.physics_cache import get_cached_reaction_rates
        sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3 = get_cached_reaction_rates(T_i, include_DHe3=True)
    
    # Call the physics solver (returns only physics: n_T, n_D, n_He3, t_startup, sol_success)
    physics_result = lump_solver(
        V_plasma, n_tot, tau_p_T, tau_p_He3,
        TBR_DT, TBR_DDn, I_target, 
        sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3
    )
    
    # Extract physics results
    n_T = physics_result['n_T']
    n_D = physics_result['n_D']
    n_He3 = physics_result['n_He3']
    t_startup = physics_result['t_startup']
    sol_success = physics_result['sol_success']
    
    ###################################################
    #                 AUXILIARY POWER
    ###################################################
    
    # Calculate P_aux from power balance if not provided (None or NaN)
    if P_aux is None or (isinstance(P_aux, float) and np.isnan(P_aux)):
        P_aux = calculate_P_aux_from_power_balance(
            n_T, n_D, T_i, V_plasma,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT,
            tau_p_T
        )
    
    # Calculate P_aux_DT_eq from power balance if not provided (None or NaN)
    if P_aux_DT_eq is None or (isinstance(P_aux_DT_eq, float) and np.isnan(P_aux_DT_eq)):
        # For DT equilibrium: n_T = n_D = n_tot / 2
        n_eq = n_tot / 2.0
        P_aux_DT_eq = calculate_P_aux_from_power_balance(
            n_eq, n_eq, T_i, V_plasma,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT,
            tau_p_T
        )
    
    # Build result dictionary with input parameters
    result = {
        'linear_index': linear_index,
        'V_plasma': V_plasma,
        'T_i': T_i,
        'n_tot': n_tot,
        'tau_p_T': tau_p_T,
        'tau_p_He3': tau_p_He3,
        'P_aux': P_aux,
        'P_aux_DT_eq': P_aux_DT_eq,
        'TBR_DT': TBR_DT,
        'TBR_DDn': TBR_DDn,
        'I_target': I_target,
        'eta_th': eta_th,
        'capacity_factor': capacity_factor,
        'price_of_electricity': price_of_electricity,
        'n_T': n_T,
        'n_D': n_D,
        'n_He3': n_He3,
        't_startup': t_startup,
        'sol_success': sol_success,
        'error': ''
    }
    
    # Guard clause: Return early if computation failed
    if not (sol_success and np.isfinite(t_startup)):
        result.update({
            'P_DDn': np.nan,
            'P_DDp': np.nan,
            'P_DT': np.nan,
            'P_DT_eq': np.nan,
            'Q_DD': np.nan,
            'Q_DT_eq': np.nan,
            'E_lost': np.nan,
            'unrealized_profits': np.nan,
            'error': 'Physics solver failed or t_startup infinite'
        })
        return result
    
    #########################################################
    #              POWER CALCULATION
    #########################################################    
    
    # Compute powers and energies for successful cases
    power_results = compute_lump_powers_and_energies(
        n_T, n_D, n_He3, t_startup,
        V_plasma, sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3,
        P_aux, P_aux_DT_eq
    )
    
    #########################################################
    #              ECONOMICS CALCULATION
    #########################################################
    
    # Compute economics
    econ_results = compute_economics_from_energies(
        power_results['E_fusion_DD'],
        power_results['E_fusion_DT_eq'],
        power_results['E_aux_DD'],
        power_results['E_aux_DT_eq'],
        eta_th, capacity_factor, price_of_electricity
    )
    
    # Add computed results to base result
    result.update({
        'P_DDn': power_results['P_DDn'],
        'P_DDp': power_results['P_DDp'],
        'P_DT': power_results['P_DT'],
        'P_DT_eq': power_results['P_DT_eq'],
        'Q_DD': econ_results['Q_DD'],
        'Q_DT_eq': econ_results['Q_DT_eq'],
        'E_lost': econ_results['E_lost'],
        'unrealized_profits': econ_results['unrealized_profits']
    })
    
    return result


def _compute_tseeded(linear_index, input_arrays_flat, param_shapes_array, max_simulation_time, vector_length, reactivity_lookup=None):
    """
    Compute T-seeded analysis for a single parameter combination.
    
    This function is defined at module level to allow pickling for multiprocessing.
    
    Args:
        linear_index: Linear index into parameter grid
        input_arrays_flat: Flattened parameter arrays
        param_shapes_array: Shape of parameter grid
        max_simulation_time: Maximum simulation time for ODE solver
        vector_length: Length of output time series vectors
        reactivity_lookup: Optional dict containing pre-computed reactivities
    """
    from src.physics.Tseeded_functions import solve_ode_system
    from src.physics.power_balance import compute_tseeded_powers_and_energies, calculate_P_aux_from_power_balance
    from src.economics.economics_functions import compute_economics_from_energies
    from src.utils.tools import index_to_params, fix_vector_length
    from src.utils.units_and_constants import tritium_mass
    from src.utils.parameter_registry import get_registry
    
    registry = get_registry()
    
    # Extract parameters efficiently with tuple unpacking
    idx = index_to_params(linear_index, param_shapes_array)
    (V_plasma, T_i, n_tot, tau_p_T, P_aux, P_aux_DT_eq,
     TBR_DT, TBR_DDn, tau_ifc, tau_ofc, eta_th, capacity_factor, price_of_electricity) = (
        arr[idx[i]] for i, arr in enumerate(input_arrays_flat)
    )
    
    # Create result dictionary with all expected fields for T_seeded analysis
    result_dict = registry.make_result_dict({
        'V_plasma': V_plasma,
        'T_i': T_i,
        'n_tot': n_tot,
        'tau_p_T': tau_p_T,
        'P_aux': P_aux,
        'P_aux_DT_eq': P_aux_DT_eq,
        'TBR_DT': TBR_DT,
        'TBR_DDn': TBR_DDn,
        'tau_ifc': tau_ifc,
        'tau_ofc': tau_ofc,
        'eta_th': eta_th,
        'capacity_factor': capacity_factor,
        'price_of_electricity': price_of_electricity,
        'error': ""
    }, analysis_type='T_seeded')
    
    # Get reactivities from lookup table (much faster than computing)
    if reactivity_lookup is not None:
        T_key = round(T_i / 0.1) * 0.1
        sigmav_DD_p = reactivity_lookup['sigmav_DD_p'][T_key]
        sigmav_DD_n = reactivity_lookup['sigmav_DD_n'][T_key]
        sigmav_DT = reactivity_lookup['sigmav_DT'][T_key]
    else:
        # Fallback to old cached method
        from src.utils.physics_cache import get_cached_reaction_rates
        sigmav_DD_p, sigmav_DD_n, sigmav_DT = get_cached_reaction_rates(T_i, include_DHe3=False)
    
    # Determine if P_aux needs to be computed (will be time-dependent vector if computed)
    compute_P_aux = P_aux is None or (isinstance(P_aux, float) and np.isnan(P_aux))
    compute_P_aux_DT_eq = P_aux_DT_eq is None or (isinstance(P_aux_DT_eq, float) and np.isnan(P_aux_DT_eq))
    
    # Precompute injection_rate_max and N_st_min
    injection_rate_max = (n_tot/2/tau_p_T*V_plasma + 0.25*n_tot**2*sigmav_DT*V_plasma - 
                         0.25/2*n_tot**2*sigmav_DD_p*V_plasma)
    N_st_min = 0.001 / tritium_mass  # Minimum storage tritium (0.001 kg)
    
    # Solve ODE system (pure physics solver - no power/economics parameters)
    # Note: ODE solver doesn't use P_aux, so we pass initial estimates for now
    ode_results = solve_ode_system(
        V_plasma, n_tot, tau_p_T,
        TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
        sigmav_DD_p, sigmav_DD_n, sigmav_DT,
        injection_rate_max, max_simulation_time, N_st_min
    )
    
    # Add ODE results
    result = {
        'linear_index': linear_index,
        **ode_results
    }
    result_dict.update(result)
    
    aux_len = registry.get_vector_length('P_aux', 5)

    # Guard clause: Return early if computation failed
    if not (ode_results.get('sol_success', False) and np.isfinite(ode_results.get('t_startup', np.inf))):
        nan_array = np.full(vector_length, np.nan)
        nan_aux = np.full(aux_len, np.nan)
        result_dict.update({
            'N_ofc': fix_vector_length(result_dict.get('N_ofc', nan_array), vector_length),
            'N_ifc': fix_vector_length(result_dict.get('N_ifc', nan_array), vector_length),
            'N_stor': fix_vector_length(result_dict.get('N_stor', nan_array), vector_length),
            'n_T': fix_vector_length(result_dict.get('n_T', nan_array), vector_length),
            'n_D': fix_vector_length(result_dict.get('n_D', nan_array), vector_length),
            'P_DDn': fix_vector_length(result_dict.get('P_DDn', nan_array), vector_length),
            'P_DDp': fix_vector_length(result_dict.get('P_DDp', nan_array), vector_length),
            'P_DT': fix_vector_length(result_dict.get('P_DT', nan_array), vector_length),
            'TBE': fix_vector_length(result_dict.get('TBE', nan_array), vector_length),
            'P_aux': fix_vector_length(result_dict.get('P_aux', nan_aux), aux_len),
            'P_aux_DT_eq': fix_vector_length(result_dict.get('P_aux_DT_eq', nan_aux), aux_len),
            'P_DT_eq': np.nan,
            'Q_DD': np.nan,
            'Q_DT_eq': np.nan,
            'E_lost': np.nan,
            'unrealized_profits': np.nan,
            'error': result_dict.get('error', 'ODE solver failed or t_startup infinite')    
        })
        return result_dict
    
    # Extract arrays (clean names, no "_raw" suffix)
    t = ode_results['t']
    N_ofc = ode_results['N_ofc']
    N_ifc = ode_results['N_ifc']
    N_stor = ode_results['N_stor']
    n_T = ode_results['n_T']
    n_D = n_tot - n_T
    t_startup = ode_results['t_startup']
    
    # Compute time-dependent P_aux vector if it was inferred from power balance
    if compute_P_aux:
        P_aux_vector = np.array([
            calculate_P_aux_from_power_balance(
                n_T_val, n_D_val, T_i, V_plasma, 
                sigmav_DD_p, sigmav_DD_n, sigmav_DT, tau_p_T
            ) for n_T_val, n_D_val in zip(n_T, n_D)
        ])
        P_aux_for_energy = np.mean(P_aux_vector)
    else:
        P_aux_vector = np.array([P_aux])
        P_aux_for_energy = P_aux
    
    if compute_P_aux_DT_eq:
        n_eq = n_tot / 2.0
        P_aux_DT_eq_vector = calculate_P_aux_from_power_balance(
            n_eq, n_eq, T_i, V_plasma, sigmav_DD_p, sigmav_DD_n, sigmav_DT, tau_p_T
        )
        P_aux_DT_eq_for_energy = P_aux_DT_eq_vector
    else:
        P_aux_DT_eq_vector = np.array([P_aux_DT_eq])
        P_aux_DT_eq_for_energy = P_aux_DT_eq
    
    # Compute powers and energies
    power_results = compute_tseeded_powers_and_energies(
        t_startup, t, n_T, n_D,
        N_ofc, N_ifc, N_stor,
        n_tot, V_plasma,
        sigmav_DD_p, sigmav_DD_n, sigmav_DT,
        tau_ifc,
        P_aux_for_energy, P_aux_DT_eq_for_energy,
        injection_rate_max, 0.001/tritium_mass,
        vector_length
    )
    # Keep the time-series profile for internal math but store a scalar value to HDF5
    P_DT_eq_scalar = np.asarray(power_results.get('P_DT_eq', np.nan)).reshape(-1)[-1]
    P_DT_eq_profile = power_results.get('P_DT_eq_profile')
    
    # Compute economics
    econ_results = compute_economics_from_energies(
        power_results['E_fusion_DD'],
        power_results['E_fusion_DT_eq'],
        power_results['E_aux_DD'],
        power_results['E_aux_DT_eq'],
        eta_th, capacity_factor, price_of_electricity
    )
    
    vectors = {
        'N_ofc': power_results['N_ofc'],
        'N_ifc': power_results['N_ifc'],
        'N_stor': power_results['N_st'],
        'n_T': power_results['n_T'],
        'n_D': power_results['n_D'],
        'P_DDn': power_results['P_DDn'],
        'P_DDp': power_results['P_DDp'],
        'P_DT': power_results['P_DT'],
        'TBE': power_results['TBE'],
        'P_aux': np.asarray(P_aux_vector, dtype=float),
        'P_aux_DT_eq': np.asarray(P_aux_DT_eq_vector, dtype=float),
    }

    # Store results with vectors padded/clipped to expected lengths
    for name, val in vectors.items():
        target_len = aux_len if name in ('P_aux', 'P_aux_DT_eq') else vector_length
        result_dict[name] = fix_vector_length(val, target_len)

    result_dict.update({
        'P_DT_eq': P_DT_eq_scalar,
        'Q_DD': econ_results['Q_DD'],
        'Q_DT_eq': econ_results['Q_DT_eq'],
        'E_lost': econ_results['E_lost'],
        'unrealized_profits': econ_results['unrealized_profits'],
    })
    
    # Keep the profile available for any downstream time-series calculations (not written to HDF5)
    if P_DT_eq_profile is not None:
        result_dict['P_DT_eq_profile'] = P_DT_eq_profile
    
    return result_dict


def run_parametric_analysis(
    input_data: Dict[str, np.ndarray],
    output_file: str,
    config: Dict[str, Any],
    verbose: bool = True,
    filter_expr: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run parametric analysis with parallel computation and HDF5 output.
    
    Args:
        input_data: Dictionary of parameter arrays
        output_file: Path to output HDF5 file
        config: Configuration dictionary with analysis settings
        verbose: Whether to print progress information
        filter_expr: Optional filter expression to reduce combinations
        
    Returns:
        Dictionary with analysis statistics
        
    Raises:
        ValueError: If configuration is invalid
        IOError: If HDF5 file cannot be created
    """
    analysis_type = str(config.get('analysis_type', '')).strip()

    # Merged model path: use internal multispecies parametric engine.
    if analysis_type == 'multispecies':
        return _run_multispecies_parametric_analysis(
            input_data=input_data,
            output_file=output_file,
            config=config,
            verbose=verbose,
            filter_expr=filter_expr,
        )

    # Extract configuration
    analysis_type = config['analysis_type']
    analysis_method = config['method']
    n_jobs = config['n_jobs']
    chunk_size = config['chunk_size']
    batch_size = config['batch_size']
    vector_length = config['vector_length']
    max_simulation_time = config['max_simulation_time']
    
    # Select compute function based on analysis type
    if analysis_type == 'lump':
        compute_function = _compute_lump
    elif analysis_type == 'T_seeded':
        compute_function = _compute_tseeded
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    # Apply filtering if specified
    if filter_expr:
        filtered_input_data, valid_indices, original_n_combinations = apply_filter_to_combinations(
            input_data, filter_expr, verbose
        )
        # Use filtered data for computation
        input_data = filtered_input_data
        n_combinations = len(valid_indices)
        # # Store original shapes for metadata
        # original_param_shapes = [arr.shape[0] for arr in [
        #     np.asarray(input_data[k]) if k in input_data else np.array([])
        #     for k in list(input_data.keys())
        # ]]
    else:
        valid_indices = None
        original_n_combinations = None
    
    # Prepare parameter arrays
    # Replace None values with scalar NaN arrays (will be computed during analysis)
    param_names = list(input_data.keys())
    param_arrays_processed = []
    for arr in input_data.values():
        if arr is None:
            # Create a scalar array with NaN for parameters that will be calculated
            param_arrays_processed.append(np.array([np.nan]))
        else:
            param_arrays_processed.append(arr)
    
    param_shapes = [arr.shape[0] for arr in param_arrays_processed]
    
    # For filtered data, all arrays are already flattened to the same length
    if filter_expr:
        n_combinations = param_shapes[0] if param_shapes else 0  # All have same length after filtering
    else:
        n_combinations = np.prod(param_shapes) if param_shapes else 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"STARTING PARAMETRIC ANALYSIS")
        print(f"{'='*60}")
        if filter_expr:
            print(f"Original combinations: {original_n_combinations:,}")
            print(f"Filtered combinations: {n_combinations:,}")
            print(f"Filter reduction: {(1 - n_combinations/original_n_combinations)*100:.1f}%")
        else:
            print(f"Total combinations: {n_combinations:,}")
        print(f"Parallel workers: {n_jobs}")
        print(f"Chunk size: {chunk_size}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")
    
    # Use the processed arrays (None replaced with NaN scalars)
    input_arrays = [np.asarray(arr) for arr in param_arrays_processed]
    
    if filter_expr:
        # For filtered data: arrays are already 1D with aligned indices
        # All parameters at index i correspond to combination i
        input_arrays_flat = input_arrays
        # Create shape array [1, 1, ..., 1, n_combinations]
        # This makes index_to_params return [0, 0, ..., 0, linear_index]
        # Then we modify compute functions to handle this special case
        param_shapes_array = np.ones(len(param_names), dtype=np.int64)
        param_shapes_array[-1] = n_combinations
    else:
        # Standard grid-based indexing
        input_arrays_flat = [arr.flatten() for arr in input_arrays]
        param_shapes_array = np.array(param_shapes, dtype=np.int64)
    
    # Get ALL parameter fields for uniform HDF5 structure
    # This ensures consistent file format regardless of analysis type
    # Fields not relevant to analysis_type will be NaN
    registry = get_registry()
    data_fields = registry.get_all_field_names(None)  # None = get all fields
    
    # Get ALL vector fields from parameter registry
    vector_fields = registry.get_vector_fields(None)  # None = get all vector fields
    
    # Statistics tracking
    start_time = time.perf_counter()
    processed_count = 0
    successful_count = 0
    negative_event_count = 0
    tmax_reached_count = 0
    solver_failed_count = 0
    
    # Create HDF5 file and run computation
    with h5py.File(output_file, 'w') as h5_file:
        # Add metadata
        metadata = {
            'total_combinations': int(n_combinations),
            'computation_start_time': start_time,
            'parameter_shapes': param_shapes,
            'method': analysis_method,
            'analysis_type': analysis_type,
            'vector_length': vector_length,
            'n_jobs': n_jobs,
            'chunk_size': chunk_size,
            'batch_size': batch_size,
        }
        
        # Add filter metadata if filtering was applied
        if filter_expr:
            metadata['filter_expression'] = filter_expr
            metadata['original_combinations'] = int(original_n_combinations)
            metadata['filtered_combinations'] = int(n_combinations)
            metadata['filter_efficiency'] = float(n_combinations / original_n_combinations * 100)
        
        h5_file.attrs.update(metadata)
        
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
                # Determine vector length for this field from registry metadata
                field_vector_length = registry.get_vector_length(field, vector_length)
                
                # 2D array for vector fields - LZ4 or fast gzip
                if use_lz4:
                    datasets[field] = h5_file.create_dataset(
                        field,
                        (n_combinations, field_vector_length),
                        dtype=np.float64,
                        chunks=(min(chunk_size, n_combinations), field_vector_length),
                        **hdf5plugin.LZ4(nbytes=0)  # Ultra-fast compression
                    )
                else:
                    datasets[field] = h5_file.create_dataset(
                        field,
                        (n_combinations, field_vector_length),
                        dtype=np.float64,
                        chunks=(min(chunk_size, n_combinations), field_vector_length),
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
        
        # ========== REACTIVITY LOOKUP TABLE (NEW OPTIMIZATION) ==========
        # Pre-compute reactivity lookup table for all unique T_i values
        # This is ~1000x faster than computing reactivities on-demand
        from src.utils.reactivity_lookup import ReactivityLookupTable
        
        # T_i is always the second parameter
        T_i_array = input_arrays_flat[1]
        unique_Ti = np.unique(T_i_array)
        
        if verbose:
            print(f"🔧 Building reactivity lookup table for {len(unique_Ti)} unique T_i values...")
        
        include_DHe3 = (analysis_type == 'lump')
        reactivity_table = ReactivityLookupTable(unique_Ti, include_DHe3=include_DHe3)
        reactivity_lookup = reactivity_table.to_dict()
        
        if verbose:
            print(f"✅ Reactivity lookup table created ({len(reactivity_table)} temperatures)")
        # ================================================================
        
        # Prime Numba compilation if T_seeded (before progress bar)
        if analysis_type == 'T_seeded' and verbose:
            print("Priming Numba compilation...")
            try:
                compute_function(0, input_arrays_flat, param_shapes_array, max_simulation_time, vector_length, reactivity_lookup)
            except Exception as e:
                print(f"Warning during Numba priming: {e}")
        # BATCH PIPELINE: Compute full batch, then write in parallel
        # Vectorized writes are MUCH faster (10-100x), can use larger batches
        # Larger batches = better compute efficiency, minimal write overhead
        write_batch_size = 10*batch_size  # Large batches with fast vectorized writes
        
        # Print parallelization info (before progress bar)
        if verbose:
            print(f"Starting BATCH PIPELINE computation with {n_jobs} workers...")
            print(f"Writing in batches of {write_batch_size}")
        
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
                # ========== DYNAMIC WORK QUEUE IMPLEMENTATION ==========
                # Instead of submitting all tasks at once, we use a work queue
                # that dynamically submits new tasks as workers finish.
                # This provides better load balancing when task durations vary.
                
                # Track all pending futures and their indices
                pending_futures = {}
                next_index_to_submit = 0
                results_buffer = {}  # Store results by index for ordered writing
                
                # Helper function to submit a single task
                def submit_task(idx):
                    """Submit a single task to the executor."""
                    if analysis_type == 'lump':
                        args = (idx, input_arrays_flat, param_shapes_array, reactivity_lookup)
                    elif analysis_type == 'T_seeded':
                        args = (idx, input_arrays_flat, param_shapes_array, max_simulation_time, vector_length, reactivity_lookup)
                    else:
                        raise ValueError(f"Unknown analysis type: {analysis_type}")
                    
                    future = executor.submit(compute_function, *args)
                    return future
                
                # Initialize queue: Submit initial batch (n_jobs * 2 tasks to keep workers busy)
                initial_queue_size = min(n_jobs * 2, n_combinations)
                for idx in range(initial_queue_size):
                    future = submit_task(idx)
                    pending_futures[future] = idx
                    next_index_to_submit += 1
                
                # Process tasks as they complete, submitting new ones dynamically
                while pending_futures:
                    # Wait for next task to complete
                    done, _ = concurrent.futures.wait(
                        pending_futures.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done:
                        result_idx = pending_futures.pop(future)
                        
                        # Get result
                        try:
                            result = future.result()
                            results_buffer[result_idx] = result
                        except Exception as exc:
                            results_buffer[result_idx] = {'error': str(exc), 'sol_success': False}
                        
                        # Track successes
                        if results_buffer[result_idx].get('sol_success', False):
                            successful_count += 1
                        else:
                            # Categorize failure type
                            error_msg = results_buffer[result_idx].get('error', '')
                            if 'Negative population' in error_msg or 'Physics failure' in error_msg:
                                negative_event_count += 1
                            elif 'not reached within max_simulation_time' in error_msg or 'DT equilibrium not reached' in error_msg:
                                tmax_reached_count += 1
                            else:
                                # ODE solver failures, step size issues, etc.
                                solver_failed_count += 1
                            
                            # Log first error for debugging (only once globally)
                            if not first_error_logged and verbose:
                                first_error_logged = True
                                error_msg = results_buffer[result_idx].get('error', 'Unknown error')
                        
                        processed_count += 1
                        overall_pbar.update(1)
                        
                        # Submit next task if more work available
                        if next_index_to_submit < n_combinations:
                            new_future = submit_task(next_index_to_submit)
                            pending_futures[new_future] = next_index_to_submit
                            next_index_to_submit += 1
                    
                    # === BATCH WRITE PHASE ===
                    # Write results to HDF5 when buffer reaches batch size
                    if len(results_buffer) >= write_batch_size or next_index_to_submit >= n_combinations:
                        overall_pbar.set_description("💾 WRITE")
                        overall_pbar.refresh()
                        write_start = time.perf_counter()
                        
                        # Get ordered indices and results for this batch
                        batch_indices = sorted(results_buffer.keys())
                        batch_results = [results_buffer[idx] for idx in batch_indices]
                        
                        _write_results_to_hdf5(
                            datasets, batch_results, batch_indices,
                            data_fields, vector_fields, vector_length
                        )
                        
                        # Flush to disk after each batch
                        h5_file.flush()
                        
                        # Clear buffer for next batch
                        results_buffer.clear()
                        
                        # Resume computing - show last write time and success rate with failure breakdown
                        write_time = time.perf_counter() - write_start
                        success_rate = (successful_count / processed_count * 100) if processed_count > 0 else 0.0
                        
                        # Calculate failure breakdown percentages
                        neg_pct = (negative_event_count / processed_count * 100) if processed_count > 0 else 0.0
                        tmax_pct = (tmax_reached_count / processed_count * 100) if processed_count > 0 else 0.0
                        solver_pct = (solver_failed_count / processed_count * 100) if processed_count > 0 else 0.0
                        
                        overall_pbar.set_description(
                            f"🔄 COMPUTE [successes: ✓ {success_rate:.1f}% | failures: ❌ {neg_pct+tmax_pct:.1f}% physical events + {solver_pct:.1f}% solver errors | last writing time: 🕐 {write_time:.2f}s ]"
                        )
                
                # === FINAL WRITE: Flush any remaining results ===
                if results_buffer:
                    overall_pbar.set_description("💾 WRITE (final)")
                    overall_pbar.refresh()
                    write_start = time.perf_counter()
                    
                    batch_indices = sorted(results_buffer.keys())
                    batch_results = [results_buffer[idx] for idx in batch_indices]
                    
                    _write_results_to_hdf5(
                        datasets, batch_results, batch_indices,
                        data_fields, vector_fields, vector_length
                    )
                    
                    h5_file.flush()
                    write_time = time.perf_counter() - write_start
                    success_rate = (successful_count / processed_count * 100) if processed_count > 0 else 0.0
                    
                    # Calculate failure breakdown percentages
                    neg_pct = (negative_event_count / processed_count * 100) if processed_count > 0 else 0.0 # % due to negative events
                    tmax_pct = (tmax_reached_count / processed_count * 100) if processed_count > 0 else 0.0 # % due to tmax reached
                    solver_pct = (solver_failed_count / processed_count * 100) if processed_count > 0 else 0.0 # % due to solver failures
                    
                    overall_pbar.set_description(
                        f"✅ COMPLETE (✓ {success_rate:.1f}% [❌ {neg_pct:.1f}% neg + {tmax_pct:.1f}% tmax + {solver_pct:.1f}% solver] 🕐 {write_time:.2f}s)"
                    )
                # =======================================================
                    
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
        end_time = time.perf_counter()
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
            
            # Special handling for P_aux and P_aux_DT_eq with length 5
            is_p_aux_field = field in ['P_aux', 'P_aux_DT_eq']
            target_length = 5 if is_p_aux_field else vector_length
            
            for i, result in enumerate(results):
                value = result.get(field, None)
                arr = np.asarray(value)
                
                if arr.ndim == 0 or arr.size == 0:
                    # Scalar or empty - fill with single value
                    scalar = float(value) if value is not None else np.nan
                    if is_p_aux_field:
                        # For P_aux fields with length 5, replicate scalar value
                        batch_array[i, :5] = scalar
                    else:
                        batch_array[i, :] = scalar
                elif arr.shape[0] == target_length:
                    # Correct length
                    if is_p_aux_field:
                        batch_array[i, :5] = arr
                    else:
                        batch_array[i, :] = arr
                else:
                    # Wrong length - special handling for P_aux fields
                    if is_p_aux_field:
                        if arr.size < 5:
                            # Shorter than 5: keep as-is and pad with last value
                            batch_array[i, :arr.size] = arr
                            if arr.size > 0:
                                batch_array[i, arr.size:5] = arr[-1]
                        else:
                            # Longer than 5: interpolate (first, last, and 3 intermediate points)
                            indices = np.linspace(0, arr.size - 1, 5, dtype=int)
                            batch_array[i, :5] = arr[indices]
                    else:
                        # Regular vector field - pad or truncate
                        copy_length = min(vector_length, arr.size)
                        batch_array[i, :copy_length] = arr[:copy_length]
            
            # Single bulk write for entire batch!
            if is_p_aux_field:
                datasets[field][indices_array, :5] = batch_array[:, :5]
            else:
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


def _ms_to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_multispecies_species_dict(
    combo: Dict[str, Any],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, float], Dict[str, float]]:
    species_params: Dict[str, Dict[str, Any]] = {}
    injection_mode: Dict[str, str] = {}
    automatic_injection_weights: Dict[str, float] = {}
    f0: Dict[str, float] = {}

    for sp in SPECIES:
        species_params[sp] = {
            "tau_p": float(combo[f"tau_p_{sp}"]),
            "lambda_decay": float(combo[f"lambda_decay_{sp}"]),
            "tau_ifc": float(combo[f"tau_ifc_{sp}"]),
            "tau_ofc": float(combo[f"tau_ofc_{sp}"]),
            "N_st_min": float(combo[f"N_st_min_{sp}"]),
            "Ndot_max": float(combo[f"Ndot_max_{sp}"]),
            "enable_plasma_channel": bool(combo[f"enable_plasma_channel_{sp}"]),
        }
        injection_mode[sp] = str(combo[f"injection_mode_{sp}"])
        automatic_injection_weights[sp] = float(combo[f"automatic_injection_weight_{sp}"])
        f0[sp] = float(combo[f"f_{sp}_0"])

    return species_params, injection_mode, automatic_injection_weights, f0


def _compute_tseeded_direct_cap(
    *,
    V_plasma: float,
    n_tot: float,
    tau_p_T: float,
    sigmav_DD_p: float,
    sigmav_DT: float,
) -> float:
    return (
        n_tot / 2.0 / tau_p_T * V_plasma
        + 0.25 * n_tot * n_tot * sigmav_DT * V_plasma
        - 0.125 * n_tot * n_tot * sigmav_DD_p * V_plasma
    )


def _apply_multispecies_dd_startup_overrides(
    *,
    dd_startup_method: str,
    combo: Dict[str, Any],
    species_params: Dict[str, Dict[str, Any]],
    injection_mode: Dict[str, str],
    automatic_injection_weights: Dict[str, float],
    f0: Dict[str, float],
    targets: Optional[List[Dict[str, Any]]],
    reactivities: Optional[Dict[str, float]],
) -> tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, str],
    Dict[str, float],
    Dict[str, float],
    List[Dict[str, Any]],
    bool,
    bool,
    bool,
]:
    from src.utils.units_and_constants import lambda_T, species_mass
    from src.physics.reactivity_functions import sigmav_DD_BoschHale, sigmav_DT_BoschHale

    method = str(dd_startup_method)
    if method == "none":
        return (
            species_params,
            injection_mode,
            automatic_injection_weights,
            f0,
            [] if targets is None else [dict(t) for t in targets],
            bool(combo["enforce_constant_total_density"]),
            bool(combo["allow_negative_auto_injection"]),
            bool(combo["auto_injection_use_storage_limits"]),
        )

    if method not in {"T-seeded_old", "lump_old", "T-seeded", "lump"}:
        raise ValueError(f"Unsupported dd_startup_method: {method}")

    species_params_out = {sp: dict(species_params[sp]) for sp in SPECIES}
    injection_mode_out = dict(injection_mode)
    auto_weights_out = dict(automatic_injection_weights)
    f0_out = dict(f0)

    if reactivities is not None:
        sigmav_DD_p = float(reactivities["sigmav_DD_p"])
        sigmav_DT = float(reactivities["sigmav_DT"])
    else:
        T_i = float(combo["T_i"])
        _, _, sigmav_DD_p = sigmav_DD_BoschHale(T_i)
        sigmav_DT = sigmav_DT_BoschHale(T_i)

    tau_p_T = float(species_params_out["T"]["tau_p"])
    tritium_cap = _compute_tseeded_direct_cap(
        V_plasma=float(combo["V_plasma"]),
        n_tot=float(combo["n_tot"]),
        tau_p_T=tau_p_T,
        sigmav_DD_p=float(sigmav_DD_p),
        sigmav_DT=float(sigmav_DT),
    )
    if not np.isfinite(tritium_cap):
        tritium_cap = np.inf
    tritium_cap = max(float(tritium_cap), 0.0)

    f0_out.update({"D": 1.0, "T": 0.0, "He3": 0.0, "He4": 0.0})
    auto_weights_out.update({"D": 1.0, "T": 0.0, "He3": 0.0, "He4": 0.0})

    species_params_out["D"]["enable_plasma_channel"] = True
    species_params_out["D"]["tau_p"] = tau_p_T
    species_params_out["D"]["tau_ifc"] = np.inf
    species_params_out["D"]["tau_ofc"] = np.inf
    species_params_out["D"]["lambda_decay"] = 0.0
    species_params_out["D"]["N_st_min"] = 0.0
    species_params_out["D"]["Ndot_max"] = np.inf

    species_params_out["T"]["enable_plasma_channel"] = True
    species_params_out["T"]["lambda_decay"] = float(lambda_T)
    species_params_out["T"]["N_st_min"] = 0.001 / species_mass["T"]
    species_params_out["T"]["Ndot_max"] = tritium_cap

    injection_mode_out.update({"D": "auto", "He3": "off", "He4": "off"})
    injection_mode_out["T"] = "direct" if method in {"T-seeded_old", "T-seeded"} else "off"

    if method in {"T-seeded_old", "lump_old"}:
        for sp in ("He3", "He4"):
            species_params_out[sp]["enable_plasma_channel"] = False
            species_params_out[sp]["tau_ifc"] = np.inf
            species_params_out[sp]["tau_ofc"] = np.inf
            species_params_out[sp]["N_st_min"] = 0.0
            species_params_out[sp]["Ndot_max"] = 0.0
    else:
        species_params_out["He3"]["enable_plasma_channel"] = True
        species_params_out["He4"]["enable_plasma_channel"] = True

    if method in {"T-seeded_old", "T-seeded"}:
        targets_out = [{"target_specie": "T", "target_fraction_in_plasma": 0.5}]
    else:
        targets_in = [] if targets is None else [dict(t) for t in targets]
        storage_targets = [t for t in targets_in if "target_inventory_storage" in t]
        if len(storage_targets) == 0:
            raise ValueError(
                "dd_startup_method 'lump'/'lump_old' requires a target with "
                "'target_inventory_storage' in config.targets"
            )
        targets_out = storage_targets

    return (
        species_params_out,
        injection_mode_out,
        auto_weights_out,
        f0_out,
        targets_out,
        True,
        False,
        False,
    )


def _compute_mixedfuel_combination(
    linear_index: int,
    input_arrays_flat: List[np.ndarray],
    param_shapes_array: np.ndarray,
    param_names: List[str],
    output_vector_length: int,
    targets: Optional[List[Dict[str, Any]]] = None,
    reactivity_lookup: Optional[Dict[str, Dict[float, float]]] = None,
    dd_startup_method: str = "none",
    route_the3_ch3_to_he4: bool = False,
) -> Dict[str, Any]:
    from src.physics.multispecies_functions import solve_multispecies_ode_system
    from src.utils.tools import fix_vector_length, index_to_params

    idx = index_to_params(linear_index, param_shapes_array)
    combo: Dict[str, Any] = {}
    for i, name in enumerate(param_names):
        combo[name] = _ms_to_python_scalar(input_arrays_flat[i][idx[i]])

    species_params, injection_mode, automatic_injection_weights, f0 = _build_multispecies_species_dict(combo)

    reactivities = None
    if reactivity_lookup is not None:
        T_i = float(combo["T_i"])
        T_key = round(T_i / 0.1) * 0.1
        if T_key in reactivity_lookup["sigmav_DT"]:
            reactivities = {
                "sigmav_DD_p": float(reactivity_lookup["sigmav_DD_p"][T_key]),
                "sigmav_DD_n": float(reactivity_lookup["sigmav_DD_n"][T_key]),
                "sigmav_DT": float(reactivity_lookup["sigmav_DT"][T_key]),
                "sigmav_DHe3": float(reactivity_lookup["sigmav_DHe3"][T_key]),
                "sigmav_TT": float(reactivity_lookup.get("sigmav_TT", {}).get(T_key, 0.0)),
                "sigmav_He3He3": float(reactivity_lookup.get("sigmav_He3He3", {}).get(T_key, 0.0)),
                "sigmav_THe3_ch1": float(reactivity_lookup.get("sigmav_THe3_ch1", {}).get(T_key, 0.0)),
                "sigmav_THe3_ch2": float(reactivity_lookup.get("sigmav_THe3_ch2", {}).get(T_key, 0.0)),
                "sigmav_THe3_ch3": float(reactivity_lookup.get("sigmav_THe3_ch3", {}).get(T_key, 0.0)),
            }

    combo_vector_length = int(round(float(combo.get("vector_length", output_vector_length))))
    combo_vector_length = max(2, combo_vector_length)

    (
        species_params,
        injection_mode,
        automatic_injection_weights,
        f0,
        targets_for_solver,
        enforce_constant_total_density,
        allow_negative_auto_injection,
        auto_injection_use_storage_limits,
    ) = _apply_multispecies_dd_startup_overrides(
        dd_startup_method=dd_startup_method,
        combo=combo,
        species_params=species_params,
        injection_mode=injection_mode,
        automatic_injection_weights=automatic_injection_weights,
        f0=f0,
        targets=targets,
        reactivities=reactivities,
    )

    solver_result = solve_multispecies_ode_system(
        V_plasma=float(combo["V_plasma"]),
        T_i=float(combo["T_i"]),
        n_tot=float(combo["n_tot"]),
        f0=f0,
        species_params=species_params,
        injection_mode=injection_mode,
        automatic_injection_weights=automatic_injection_weights,
        TBR_DT=float(combo["TBR_DT"]),
        TBR_DDn=float(combo["TBR_DDn"]),
        max_simulation_time=float(combo["max_simulation_time"]),
        vector_length=combo_vector_length,
        targets=targets_for_solver,
        enforce_constant_total_density=enforce_constant_total_density,
        allow_negative_auto_injection=allow_negative_auto_injection,
        auto_injection_use_storage_limits=auto_injection_use_storage_limits,
        route_THe3_ch3_to_He4=bool(route_the3_ch3_to_he4),
        reactivities=reactivities,
    )

    result = {"linear_index": int(linear_index)}
    result.update(combo)
    result.update(solver_result)

    for key in (
        ["t", "sum_dn_dt", "required_auto_total", "unmet_auto_total", "normalized_residual", "n_total_rel_drift"]
        + [f"n_{sp}" for sp in SPECIES]
        + [f"N_ofc_{sp}" for sp in SPECIES]
        + [f"N_ifc_{sp}" for sp in SPECIES]
        + [f"N_st_{sp}" for sp in SPECIES]
    ):
        result[key] = fix_vector_length(result.get(key, np.array([np.nan], dtype=float)), output_vector_length)

    result.setdefault("error", "")
    result.setdefault("sol_success", False)
    result.setdefault("t_startup", np.inf)
    return result


def _write_multispecies_results_to_hdf5(
    *,
    datasets: Dict[str, Any],
    results: List[Dict[str, Any]],
    indices: List[int],
    data_fields: List[str],
    vector_fields: set[str],
    vector_length: int,
    registry,
) -> None:
    from src.utils.tools import fix_vector_length

    idx_arr = np.asarray(indices, dtype=np.int64)

    for field in data_fields:
        props = registry.parameters[field]
        dtype_name = props.get("dtype", "float")

        if field in vector_fields:
            batch = np.full((len(results), vector_length), np.nan, dtype=float)
            for i, res in enumerate(results):
                batch[i, :] = fix_vector_length(res.get(field, np.array([np.nan])), vector_length)
            datasets[field][idx_arr, :] = batch
            continue

        if dtype_name == "str":
            values = []
            for res in results:
                v = res.get(field, "")
                if v is None:
                    v = ""
                values.append(str(v))
            datasets[field][idx_arr] = values
            continue

        if dtype_name == "bool":
            values = np.array([bool(res.get(field, False)) for res in results], dtype=bool)
            datasets[field][idx_arr] = values
            continue

        if dtype_name == "int":
            values = []
            for res in results:
                v = res.get(field, np.nan)
                if isinstance(v, (list, tuple, np.ndarray)):
                    v = np.asarray(v).reshape(-1)[-1] if np.asarray(v).size else 0
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    v = 0
                values.append(int(v))
            datasets[field][idx_arr] = np.asarray(values, dtype=np.int64)
            continue

        values = []
        for res in results:
            v = res.get(field, np.nan)
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v).reshape(-1)
                v = float(arr[-1]) if arr.size else np.nan
            elif v is None:
                v = np.nan
            try:
                values.append(float(v))
            except Exception:
                values.append(np.nan)
        datasets[field][idx_arr] = np.asarray(values, dtype=np.float64)


def _run_multispecies_parametric_analysis(
    input_data: Dict[str, np.ndarray],
    output_file: str,
    config: Dict[str, Any],
    verbose: bool = True,
    filter_expr: Optional[str] = None,
) -> Dict[str, Any]:
    from src.utils.reactivity_lookup import ReactivityLookupTable
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    analysis_type = str(config["analysis_type"]).strip()
    if analysis_type != "multispecies":
        raise ValueError(f"Unsupported analysis_type: {analysis_type}")

    n_jobs = int(config["n_jobs"])
    chunk_size = int(config["chunk_size"])
    batch_size = int(config["batch_size"])
    output_vector_length = int(config["vector_length"])

    if filter_expr:
        filtered_input_data, valid_indices, original_n_combinations = apply_filter_to_combinations(
            input_data, filter_expr, verbose
        )
        input_data = filtered_input_data
        n_combinations = len(valid_indices)
    else:
        valid_indices = None
        original_n_combinations = None
        n_combinations = int(np.prod([arr.shape[0] for arr in input_data.values()]))

    param_names = list(input_data.keys())
    input_arrays = [np.asarray(arr) for arr in input_data.values()]
    param_shapes = [arr.shape[0] for arr in input_arrays]

    if filter_expr:
        input_arrays_flat = input_arrays
        param_shapes_array = np.ones(len(param_names), dtype=np.int64)
        param_shapes_array[-1] = n_combinations
    else:
        input_arrays_flat = [arr.flatten() for arr in input_arrays]
        param_shapes_array = np.array(param_shapes, dtype=np.int64)

    registry = get_multispecies_registry()
    data_fields = registry.get_all_field_names(analysis_type)
    vector_fields = set(registry.get_vector_fields(analysis_type))

    T_i_idx = param_names.index("T_i")
    unique_Ti = np.unique(np.asarray(input_arrays_flat[T_i_idx], dtype=float))
    reactivity_lookup = ReactivityLookupTable(
        unique_Ti,
        include_DHe3=True,
        include_extra_channels=True,
    ).to_dict()

    start_time = time.perf_counter()
    processed_count = 0
    successful_count = 0
    write_batch_size = max(batch_size, 8 * max(1, n_jobs))

    with h5py.File(output_file, "w") as h5_file:
        h5_file.attrs.update(
            {
                "analysis_type": analysis_type,
                "method": config["method"],
                "dd_startup_method": str(config.get("dd_startup_method", "none")),
                "total_combinations": int(n_combinations),
                "vector_length": int(output_vector_length),
                "n_jobs": n_jobs,
                "chunk_size": chunk_size,
                "batch_size": batch_size,
                "route_the3_ch3_to_he4": bool(config.get("route_the3_ch3_to_he4", False)),
                "computation_start_time": start_time,
            }
        )
        if filter_expr:
            h5_file.attrs["filter_expression"] = str(filter_expr)
            h5_file.attrs["original_combinations"] = int(original_n_combinations)
            h5_file.attrs["filtered_combinations"] = int(n_combinations)

        datasets: Dict[str, Any] = {}
        for field in data_fields:
            props = registry.parameters[field]
            dtype_name = props.get("dtype", "float")
            is_vector = field in vector_fields
            if is_vector:
                datasets[field] = h5_file.create_dataset(
                    field,
                    (n_combinations, output_vector_length),
                    dtype=np.float64,
                    chunks=(min(chunk_size, max(1, n_combinations)), output_vector_length),
                    compression="gzip",
                    compression_opts=1,
                )
                continue

            if dtype_name == "str":
                dtype = h5py.string_dtype(encoding="utf-8")
            elif dtype_name == "bool":
                dtype = bool
            elif dtype_name == "int":
                dtype = np.int64
            else:
                dtype = np.float64

            datasets[field] = h5_file.create_dataset(
                field,
                (n_combinations,),
                dtype=dtype,
                chunks=(min(chunk_size, max(1, n_combinations)),),
                compression="gzip",
                compression_opts=1,
            )

        pbar = tqdm(total=n_combinations, disable=not verbose, desc="multispecies", unit="comb")

        def _compute_one(i: int) -> Dict[str, Any]:
            return _compute_mixedfuel_combination(
                linear_index=i,
                input_arrays_flat=input_arrays_flat,
                param_shapes_array=param_shapes_array,
                param_names=param_names,
                output_vector_length=output_vector_length,
                targets=config.get("targets"),
                reactivity_lookup=reactivity_lookup,
                dd_startup_method=str(config.get("dd_startup_method", "none")),
                route_the3_ch3_to_he4=bool(config.get("route_the3_ch3_to_he4", False)),
            )

        def _flush_buffer(results_buffer: Dict[int, Dict[str, Any]]) -> None:
            if not results_buffer:
                return
            batch_indices = sorted(results_buffer.keys())
            batch_results = [results_buffer[i] for i in batch_indices]
            _write_multispecies_results_to_hdf5(
                datasets=datasets,
                results=batch_results,
                indices=batch_indices,
                data_fields=data_fields,
                vector_fields=vector_fields,
                vector_length=output_vector_length,
                registry=registry,
            )
            h5_file.flush()
            results_buffer.clear()

        results_buffer: Dict[int, Dict[str, Any]] = {}
        try:
            if n_jobs <= 1:
                for i in range(n_combinations):
                    try:
                        res = _compute_one(i)
                    except Exception as e:
                        res = {
                            "linear_index": i,
                            "sol_success": False,
                            "error": f"Worker exception: {e}",
                            "t_startup": np.inf,
                        }
                    results_buffer[i] = res
                    processed_count += 1
                    if bool(res.get("sol_success", False)):
                        successful_count += 1
                    pbar.update(1)
                    if len(results_buffer) >= write_batch_size:
                        _flush_buffer(results_buffer)
                _flush_buffer(results_buffer)
            else:
                def submit_task(executor: ProcessPoolExecutor, i: int):
                    return executor.submit(
                        _compute_mixedfuel_combination,
                        linear_index=i,
                        input_arrays_flat=input_arrays_flat,
                        param_shapes_array=param_shapes_array,
                        param_names=param_names,
                        output_vector_length=output_vector_length,
                        targets=config.get("targets"),
                        reactivity_lookup=reactivity_lookup,
                        dd_startup_method=str(config.get("dd_startup_method", "none")),
                        route_the3_ch3_to_he4=bool(config.get("route_the3_ch3_to_he4", False)),
                    )

                next_index_to_submit = 0
                pending = {}
                try:
                    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                        initial_queue = min(n_combinations, max(1, 2 * n_jobs))
                        for i in range(initial_queue):
                            fut = submit_task(executor, i)
                            pending[fut] = i
                            next_index_to_submit += 1

                        while pending:
                            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                            for fut in done:
                                idx_i = pending.pop(fut)
                                try:
                                    res = fut.result()
                                except Exception as e:
                                    res = {
                                        "linear_index": idx_i,
                                        "sol_success": False,
                                        "error": f"Worker exception: {e}",
                                        "t_startup": np.inf,
                                    }
                                results_buffer[idx_i] = res
                                processed_count += 1
                                if bool(res.get("sol_success", False)):
                                    successful_count += 1
                                pbar.update(1)

                                if next_index_to_submit < n_combinations:
                                    new_fut = submit_task(executor, next_index_to_submit)
                                    pending[new_fut] = next_index_to_submit
                                    next_index_to_submit += 1

                            if len(results_buffer) >= write_batch_size or (
                                next_index_to_submit >= n_combinations and not pending
                            ):
                                _flush_buffer(results_buffer)
                except (PermissionError, OSError):
                    for i in range(processed_count, n_combinations):
                        try:
                            res = _compute_one(i)
                        except Exception as e:
                            res = {
                                "linear_index": i,
                                "sol_success": False,
                                "error": f"Worker exception: {e}",
                                "t_startup": np.inf,
                            }
                        results_buffer[i] = res
                        processed_count += 1
                        if bool(res.get("sol_success", False)):
                            successful_count += 1
                        pbar.update(1)
                        if len(results_buffer) >= write_batch_size:
                            _flush_buffer(results_buffer)
                    _flush_buffer(results_buffer)
        finally:
            pbar.close()

        end_time = time.perf_counter()
        h5_file.attrs["computation_end_time"] = end_time
        h5_file.attrs["total_computation_time"] = end_time - start_time
        h5_file.attrs["processed"] = int(processed_count)
        h5_file.attrs["successful"] = int(successful_count)

    stats = {
        "total_combinations": int(n_combinations),
        "processed": int(processed_count),
        "successful": int(successful_count),
        "success_rate": (100.0 * successful_count / processed_count) if processed_count else 0.0,
        "computation_time": float(end_time - start_time),
    }
    return stats


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
