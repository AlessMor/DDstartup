import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp

# Configuration selector - set to True for test, False for full run
USE_TEST = True  # Change this to False for full 24,300 combination run

verbose = True
from utils.physics import sigmav_DT_BoschHale, sigmav_DD_BoschHale, sigmav_DHe3_BoschHale

############ INPUT SETUP ############

# constants
from utils.units_and_constants import u
import utils.units_and_constants as consts

# parameters - conditional import based on test mode
if USE_TEST:
    from utils.config_test import *
    print("🔬 Using TEST configuration")
else:
    from utils.config import *
    print("🚀 Using FULL configuration")

# IMPORTANT: Override ALL pint quantities with units-free versions AFTER config import
E_DDp = consts.E_DDp.to('J').magnitude
E_DDn = consts.E_DDn.to('J').magnitude
E_DT = consts.E_DT.to('J').magnitude
E_DHe3 = consts.E_DHe3.to('J').magnitude
tritium_mass = consts.tritium_mass.to('kg').magnitude  # [kg] - plain number, no units
lambda_T = np.log(2) / (12.32 * 365*24*3600)  # [1/s] - plain number, no units

if verbose:
    print("Plasma parameter fields:")
    print(f"V_plasma: {V_plasma_field}")
    print(f"T_i_field: {T_i_field}")
    print(f"n_tot_field: {n_tot_field}")
    print(f"tau_p_T: {tau_p_T_field}")
    print(f"tau_p_He3: {tau_p_He3_field}")
    print(f"P_aux: {P_aux_field}")
    print(f"P_aux_all_DT: {P_aux_all_DT_field}")
    print(f"P_lost_rad: {P_lost_rad_field}")
    print(f"P_lost_rad_all_DT: {P_lost_rad_all_DT_field}")

    print("Breeding parameters:")
    print(f"TBR_DT: {TBR_DT_field}")
    print(f"TBR_DDn: {TBR_DDn_field}")
    print(f"tau_ifc: {tau_ifc_field}")
    print(f"tau_ofc: {tau_ofc_field}")

    print("Economic parameters:")
    print(f"eta_th: {eta_th_field}")
    print(f"plant_avail: {plant_avail_field}")
    print(f"Cost_per_kWh: {Cost_per_kWh_field}")

# build iterator
from itertools import product
input_data = [
    V_plasma_field.data.to_base_units().magnitude,
    T_i_field.data.to('keV').magnitude,
    n_tot_field.data.to_base_units().magnitude,
    tau_p_T_field.data.to_base_units().magnitude, 
    tau_p_He3_field.data.to_base_units().magnitude,
    P_aux_field.data.to_base_units().magnitude,
    P_lost_rad_field.data.to_base_units().magnitude,
    P_aux_all_DT_field.data.to_base_units().magnitude,
    P_lost_rad_all_DT_field.data.to_base_units().magnitude,
        
    TBR_DT_field.data.to_base_units().magnitude,
    TBR_DDn_field.data.to_base_units().magnitude,
    tau_ifc_field.data.to_base_units().magnitude,
    tau_ofc_field.data.to_base_units().magnitude,
    
    eta_th_field.data.to_base_units().magnitude,
    plant_avail_field.data.to_base_units().magnitude,
    Cost_per_kWh_field.data.to_base_units().magnitude,
]

# Create parameter ranges and calculate total combinations
param_shapes = [data.shape[0] for data in input_data]
n_combinations = np.prod(param_shapes)
if verbose:
    print(f"Total number of parameter combinations: {n_combinations}")

# Convert input_data to numpy arrays for faster indexing
input_arrays = [np.asarray(data) for data in input_data]

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def index_to_params(linear_index, param_shapes):
    """Convert linear index to multi-dimensional parameter indices"""
    n_params = len(param_shapes)
    indices = np.zeros(n_params, dtype=np.int64)
    
    remaining = linear_index
    for i in range(n_params - 1, -1, -1):
        indices[i] = remaining % param_shapes[i]
        remaining = remaining // param_shapes[i]
    
    return indices

def solve_ode_system(V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, P_aux, P_lost_rad, 
                           P_aux_all_DT, P_lost_rad_all_DT, TBR_DT, TBR_DDn, tau_ifc, tau_ofc, 
                           eta_th, plant_avail, Cost_per_kWh, sigmav_DD_p, sigmav_DD_n, sigmav_DT):
    """Simple ODE system implementation without external dependencies"""
    
    
    # Units-free ODE function
    def tritium_inventory_odes_unitless(t, y):
        """Units-free ODE system: y = [N_ofc, N_ifc, N_st, n_T]"""
        N_ofc, N_ifc, N_st, n_T = y
        
        n_D = n_tot - n_T
        N_st_min = 0.001/tritium_mass  # minimum storage inventory
        
        # Injection rate logic
        if N_st < N_st_min:
            injection_rate = 0.0
        else:
            injection_rate_max = (n_tot/2/tau_p_T*V_plasma + 0.25*n_tot**2*sigmav_DT*V_plasma - 0.25/2*n_tot**2*sigmav_DD_p*V_plasma)
            injection_rate = min((N_ifc/tau_ifc - lambda_T*N_st), injection_rate_max)
        
        # Reaction rates
        Tdot_DDn = TBR_DDn * 0.5 * n_D * n_D * sigmav_DD_n * V_plasma
        Tdot_DDp = 0.5 * n_D * n_D * sigmav_DD_p * V_plasma
        Tdot_DT = TBR_DT * n_D * n_T * sigmav_DT * V_plasma
        Tdot_burn = n_D * n_T * sigmav_DT * V_plasma
        
        # ODE system
        dN_ofc_dt = Tdot_DT + Tdot_DDn - N_ofc / tau_ofc - N_ofc * lambda_T
        dN_ifc_dt = N_ofc / tau_ofc - N_ifc / tau_ifc - lambda_T * N_ifc + n_T / tau_p_T * V_plasma
        dN_stor_dt = N_ifc / tau_ifc - lambda_T * N_st - injection_rate
        dnT_dt = injection_rate / V_plasma + Tdot_DDp / V_plasma - n_T / tau_p_T - Tdot_burn / V_plasma
        
        return [dN_ofc_dt, dN_ifc_dt, dN_stor_dt, dnT_dt]
        
        return [dN_ofc_dt, dN_ifc_dt, dN_stor_dt, dnT_dt]
    
    # Event functions
    def DT_reached_event(t, y):
        """Event function for DT condition"""
        n_T = y[3]
        return n_T - 0.5 * n_tot
    
    def negative_event(t, y):
        """Event function for negative states"""
        return min(y[0]+100, y[1]+100, y[2]+100, y[3]+100)
    
    # Configure events
    DT_reached_event.terminal = True
    negative_event.terminal = True 
    
    # Initial conditions [N_ofc, N_ifc, N_stor, n_T]
    y0 = [0.0, 0.0, 0.0, 0.0]  # Start with minimal tritium storage
    
    # Time span
    t_span = (0, total_time)
    t_eval = np.linspace(0, total_time, 100)
    
    # Solve ODE system
    try:
        sol = solve_ivp(
            fun=tritium_inventory_odes_unitless,
            t_span=t_span,
            t_eval=t_eval,
            y0=y0,
            method='BDF',  # Good for stiff systems
            dense_output=False,
            events=[DT_reached_event, negative_event],
            rtol=1e-6,
            atol=1e-9
        )
    except Exception as e:
        # Return failure case with detailed error information
        import traceback
        full_error = f"{str(e)} | Traceback: {traceback.format_exc()}"
        return {
            't_startup': np.inf,
            'P_DT': np.inf,
            'P_DDn': np.inf, 
            'P_DDp': np.inf,
            'P_DT_full': np.inf,
            'P_fusion_DD_avg': np.inf,
            'P_e_net_DD_avg': np.inf,
            'P_e_net_DT_full_avg': np.inf,
            'Q_DD_total': np.inf,
            'Q_DT_full_total': np.inf,
            'E_fusion_total_DD': np.inf,
            'E_fusion_DT_full': np.inf,
            'E_e_net_DD': np.inf,
            'E_e_net_DT_full': np.inf,
            'E_lost': np.inf,
            'Dollar_Lost': np.inf,
            'error': full_error
        }
    
    # Process results (rest of the function same as before)
    if not sol.success:
        # Enhanced error reporting for ODE solve failures
        failure_reason = f"ODE solve failed: {sol.message if hasattr(sol, 'message') else 'Unknown reason'}"
        if hasattr(sol, 't') and len(sol.t) > 0:
            failure_reason += f" | Last time: {sol.t[-1]:.2e}s"
        if hasattr(sol, 'status'):
            failure_reason += f" | Status code: {sol.status}"
        
        return {
            't_startup': np.inf,
            'P_DT': np.inf,
            'P_DDn': np.inf,
            'P_DDp': np.inf,
            'P_DT_full': np.inf,
            'P_fusion_DD_avg': np.inf,
            'P_e_net_DD_avg': np.inf,
            'P_e_net_DT_full_avg': np.inf,
            'Q_DD_total': np.inf,
            'Q_DT_full_total': np.inf,
            'E_fusion_total_DD': np.inf,
            'E_fusion_DT_full': np.inf,
            'E_e_net_DD': np.inf,
            'E_e_net_DT_full': np.inf,
            'E_lost': np.inf,
            'Dollar_Lost': np.inf,
            'error': failure_reason
        }
    
    # Check for negative event (index 1)
    if len(sol.t_events) > 1 and sol.t_events[1].size > 0:
        return {
            't_startup': np.inf,
            'P_DT': np.inf,
            'P_DDn': np.inf,
            'P_DDp': np.inf,
            'P_DT_full': np.inf,
            'P_fusion_DD_avg': np.inf,
            'P_e_net_DD_avg': np.inf,
            'P_e_net_DT_full_avg': np.inf,
            'Q_DD_total': np.inf,
            'Q_DT_full_total': np.inf,
            'E_fusion_total_DD': np.inf,
            'E_fusion_DT_full': np.inf,
            'E_e_net_DD': np.inf,
            'E_e_net_DT_full': np.inf,
            'E_lost': np.inf,
            'Dollar_Lost': np.inf,
            'negative_event_time': sol.t_events[1][0]
        }
    
    # Determine startup time
    if len(sol.t_events) > 0 and sol.t_events[0].size > 0:
        t_startup = sol.t_events[0][0]  # DT reached event
    else:
        t_startup = np.inf
    
    # Extract solution
    n_T = sol.y[3]  # Tritium density evolution
    n_D = n_tot - n_T  # Deuterium density
    
    # Calculate fusion powers (all in Watts)
    P_DDn = n_D * n_D * sigmav_DD_n / 2 * V_plasma * E_DDn
    P_DDp = n_D * n_D * sigmav_DD_p / 2 * V_plasma * E_DDp  
    P_DT = n_D * n_T * sigmav_DT * V_plasma * E_DT
    P_DT_full = n_tot/2 * n_tot/2 * sigmav_DT * V_plasma * E_DT  # Equivalent if always DT
    
    # Determine integration time
    t_end = t_startup if np.isfinite(t_startup) else total_time
    mask = sol.t <= t_end
    
    # Calculate net energies (integrate up to startup time) - TOTAL ENERGY APPROACH
    if np.sum(mask) > 1:
         # Total fusion energies by integration
        E_fusion_DDn = np.trapz(P_DDn[mask], sol.t[mask])
        E_fusion_DDp = np.trapz(P_DDp[mask], sol.t[mask])
        E_fusion_DT = np.trapz(P_DT[mask], sol.t[mask])
        E_fusion_total_DD = E_fusion_DDn + E_fusion_DDp + E_fusion_DT
        E_fusion_DT_full = P_DT_full * t_end  # Constant power * time
        
        # Total auxiliary energies (constant power * time)
        E_aux_DD = P_aux * t_end
        E_aux_DT_full = P_aux_all_DT * t_end
        
        # Total radiation losses (constant power * time)
        E_rad_DD = P_lost_rad * t_end
        E_rad_DT_full = P_lost_rad_all_DT * t_end
        
        # Net electrical energies (with thermal efficiency and plant availability)
        E_e_net_DD = plant_avail * (eta_th * (E_fusion_total_DD - E_rad_DD) - E_aux_DD)
        E_e_net_DT_full = plant_avail * (eta_th * (E_fusion_DT_full - E_rad_DT_full) - E_aux_DT_full)
        
        # Q factors based on total energy
        Q_DD_total = E_fusion_total_DD / E_aux_DD if E_aux_DD > 0 else np.inf
        Q_DT_full_total = E_fusion_DT_full / E_aux_DT_full if E_aux_DT_full > 0 else np.inf
        
        # Average powers for reference (divide total energy by time)
        time_duration = t_end  # Use actual time duration, not sol.t[mask][-1]
        P_fusion_DD_avg = E_fusion_total_DD / time_duration if time_duration > 0 else 0.0
        P_e_net_DD_avg = E_e_net_DD / time_duration if time_duration > 0 else 0.0
        P_e_net_DT_full_avg = E_e_net_DT_full / time_duration if time_duration > 0 else 0.0
        
    else:
        E_fusion_total_DD = 0.0
        E_fusion_DT_full = 0.0
        E_e_net_DD = 0.0
        E_e_net_DT_full = 0.0
        Q_DD_total = 0.0
        Q_DT_full_total = 0.0
        P_fusion_DD_avg = 0.0
        P_e_net_DD_avg = 0.0
        P_e_net_DT_full_avg = 0.0
    
    return {
        't_startup': t_startup,
        'P_DT': np.mean(P_DT[mask]) if np.sum(mask) > 0 else 0.0,
        'P_DDn': np.mean(P_DDn[mask]) if np.sum(mask) > 0 else 0.0,
        'P_DDp': np.mean(P_DDp[mask]) if np.sum(mask) > 0 else 0.0,
        'P_DT_full': P_DT_full,  # Constant scalar value
        'P_fusion_DD_avg': P_fusion_DD_avg,  # Average fusion power during startup
        'P_e_net_DD_avg': P_e_net_DD_avg,    # Average net electrical power
        'P_e_net_DT_full_avg': P_e_net_DT_full_avg,  # Average net electrical power (DT case)
        'Q_DD_total': Q_DD_total,            # Q factor based on total energy
        'Q_DT_full_total': Q_DT_full_total,  # Q factor based on total energy
        'E_fusion_total_DD': E_fusion_total_DD,      # Total fusion energy produced
        'E_fusion_DT_full': E_fusion_DT_full,        # Total DT-equivalent fusion energy
        'E_e_net_DD': E_e_net_DD,                    # Total net electrical energy (DD startup)
        'E_e_net_DT_full': E_e_net_DT_full,          # Total net electrical energy (DT full)
        'E_lost': E_e_net_DT_full - E_e_net_DD,      # Energy difference
        'Dollar_Lost': (E_e_net_DT_full - E_e_net_DD) * Cost_per_kWh,  # Cost in dollars
        'n_T_final': n_T[-1] if len(n_T) > 0 else 0.0,
        'sol_success': sol.success
    }
       
def compute_single_combination(linear_index, input_arrays_flat, param_shapes_array):
    """Compute results for a single parameter combination"""
    # Get parameter indices from linear index
    param_indices = index_to_params(linear_index, param_shapes_array)
    
    # Extract parameters directly - fastest approach
    V_plasma = input_arrays_flat[0][param_indices[0]]
    T_i = input_arrays_flat[1][param_indices[1]]
    n_tot = input_arrays_flat[2][param_indices[2]]
    tau_p_T = input_arrays_flat[3][param_indices[3]]
    tau_p_He3 = input_arrays_flat[4][param_indices[4]]
    P_aux = input_arrays_flat[5][param_indices[5]]
    P_lost_rad = input_arrays_flat[6][param_indices[6]]
    P_aux_all_DT = input_arrays_flat[7][param_indices[7]]
    P_lost_rad_all_DT = input_arrays_flat[8][param_indices[8]]
    TBR_DT = input_arrays_flat[9][param_indices[9]]
    TBR_DDn = input_arrays_flat[10][param_indices[10]]
    tau_ifc = input_arrays_flat[11][param_indices[11]]
    tau_ofc = input_arrays_flat[12][param_indices[12]]
    eta_th = input_arrays_flat[13][param_indices[13]]
    plant_avail = input_arrays_flat[14][param_indices[14]]
    Cost_per_kWh = input_arrays_flat[15][param_indices[15]]
    
    # Calculate T_i-dependent parameters (ensure scalar inputs to physics functions)
    T_i_array = np.array([T_i])  # Convert scalar to array for physics functions
    sigmav_DD_results = sigmav_DD_BoschHale(T_i_array)
    sigmav_DD_p = sigmav_DD_results[1][0]  # Extract scalar from array result
    sigmav_DD_n = sigmav_DD_results[2][0]  # Extract scalar from array result
    sigmav_DT = sigmav_DT_BoschHale(T_i_array)[0]  # Extract scalar from array result
    sigmav_DHe3 = sigmav_DHe3_BoschHale(T_i_array)[0]  # Extract scalar from array result
    
    # Maximum injection rate
    injection_rate_max = (n_tot/2/tau_p_T*V_plasma + 0.25*n_tot**2*sigmav_DT*V_plasma - 0.25/2*n_tot**2*sigmav_DD_p*V_plasma)
    
    # SOLVE COMPLEX ODE SYSTEM
    ode_results = solve_ode_system(
        V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, P_aux, P_lost_rad,
        P_aux_all_DT, P_lost_rad_all_DT, TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
        eta_th, plant_avail, Cost_per_kWh, sigmav_DD_p, sigmav_DD_n, sigmav_DT
    )
    
    # Package results (combine input parameters and ODE outputs)
    result = {
        'linear_index': linear_index,
        'V_plasma': V_plasma,
        'T_i': T_i,
        'n_tot': n_tot,
        'tau_p_T': tau_p_T,
        'tau_p_He3': tau_p_He3,
        'P_aux': P_aux,
        'P_lost_rad': P_lost_rad,
        'P_aux_all_DT': P_aux_all_DT,
        'P_lost_rad_all_DT': P_lost_rad_all_DT,
        'TBR_DT': TBR_DT,
        'TBR_DDn': TBR_DDn,
        'tau_ifc': tau_ifc,
        'tau_ofc': tau_ofc,
        'eta_th': eta_th,
        'plant_avail': plant_avail,
        'Cost_per_kWh': Cost_per_kWh,
        'injection_rate_max': injection_rate_max,
        'sigmav_DT': sigmav_DT,
        'sigmav_DD_p': sigmav_DD_p,
        'sigmav_DD_n': sigmav_DD_n,
        # Add all ODE results
        **ode_results
    }
    
    return result

# Prepare data for parallel computation
input_arrays_flat = [arr.flatten() for arr in input_arrays]
param_shapes_array = np.array(param_shapes, dtype=np.int64)

# Parallel computation
if verbose:
    print("Starting parallel computation...")

# Create indices for all combinations
combination_indices = np.arange(n_combinations)

# Parallel execution using joblib
n_jobs = -1  # Use all available cores

# Setup streaming HDF5 file for incremental saving
import h5py
import time
import os

# Generate timestamped filename
timestamp = time.strftime("%Y%m%d_%H%M%S")
if USE_TEST:
    output_filename = f"outputs/dd_startup_results_test_{timestamp}.h5"
else:
    output_filename = f"outputs/dd_startup_results_{timestamp}.h5"

if verbose:
    print(f"Streaming results to {output_filename}...")

# Define all result fields for pre-allocation
result_fields = [
    'linear_index', 'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'tau_p_He3', 
    'P_aux', 'P_lost_rad', 'P_aux_all_DT', 'P_lost_rad_all_DT',
    'TBR_DT', 'TBR_DDn', 'tau_ifc', 'tau_ofc', 'eta_th', 'plant_avail', 'Cost_per_kWh',
    'injection_rate_max', 'sigmav_DT', 'sigmav_DD_p', 'sigmav_DD_n',
    't_startup', 'P_DT', 'P_DDn', 'P_DDp', 'P_DT_full', 'P_fusion_DD_avg',
    'P_e_net_DD_avg', 'P_e_net_DT_full_avg', 'Q_DD_total', 'Q_DT_full_total',
    'E_fusion_total_DD', 'E_fusion_DT_full', 'E_e_net_DD', 'E_e_net_DT_full',
    'E_lost', 'Dollar_Lost', 'n_T_final', 'sol_success'
]

# Create HDF5 file and pre-allocate datasets
chunk_size = 10000 if USE_TEST else 1000 
write_chunk_size = min(chunk_size, n_combinations)

with h5py.File(output_filename, 'w') as h5_file:
    # Store metadata
    h5_file.attrs['total_combinations'] = n_combinations
    h5_file.attrs['computation_start_time'] = time.time()
    h5_file.attrs['parameter_shapes'] = param_shapes
    h5_file.attrs['test_mode'] = 'small_test' if USE_TEST else 'full'
    
    # Pre-allocate datasets for all numerical fields
    datasets = {}
    for field in result_fields:
        if field == 'sol_success':
            datasets[field] = h5_file.create_dataset(field, (n_combinations,), dtype=bool, 
                                                   chunks=(write_chunk_size,), compression='gzip')
        else:
            datasets[field] = h5_file.create_dataset(field, (n_combinations,), dtype=np.float64, 
                                                   chunks=(write_chunk_size,), compression='gzip')
    
    # Pre-allocate error tracking (estimate max 1000 errors)
    max_errors = min(1000, n_combinations // 10)
    error_indices_list = []
    error_messages_list = []
    negative_event_indices_list = []
    negative_event_times_list = []
    
    # Save parameter field information
    param_group = h5_file.create_group('parameter_fields')
    param_names = ['V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'tau_p_He3', 'P_aux', 'P_lost_rad',
                   'P_aux_all_DT', 'P_lost_rad_all_DT', 'TBR_DT', 'TBR_DDn', 'tau_ifc', 'tau_ofc',
                   'eta_th', 'plant_avail', 'Cost_per_kWh']
    
    for i, name in enumerate(param_names):
        param_group.create_dataset(f'{name}_values', data=input_arrays[i], compression='gzip')
    
    # Process in chunks for streaming
    processed_count = 0
    successful_count = 0
    
    # Calculate total chunks for progress bar
    total_chunks = (n_combinations + chunk_size - 1) // chunk_size
    
    # Create progress bars
    chunk_pbar = tqdm(total=total_chunks, desc="Processing chunks", unit="chunk", position=0)
    overall_pbar = tqdm(total=n_combinations, desc="Overall progress", unit="comb", position=1)
    
    for chunk_start in range(0, n_combinations, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_combinations)
        chunk_indices = combination_indices[chunk_start:chunk_end]
        current_chunk_size = len(chunk_indices)
        
        # Parallel execution for this chunk (no verbose output)
        chunk_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(compute_single_combination)(idx, input_arrays_flat, param_shapes_array) 
            for idx in chunk_indices
        )
        
        # Write chunk results to HDF5 immediately
        chunk_successes = 0
        for i, result in enumerate(chunk_results):
            abs_idx = chunk_start + i
            
            # Fill numerical datasets
            for field in result_fields:
                if field in result:
                    value = result[field]
                    if field == 'sol_success':
                        datasets[field][abs_idx] = bool(value)
                    else:
                        datasets[field][abs_idx] = float(value) if np.isfinite(value) else float(value)
                else:
                    # Default values for missing fields
                    if field == 'sol_success':
                        datasets[field][abs_idx] = False
                    else:
                        datasets[field][abs_idx] = np.inf
            
            # Collect error and event information
            if 'error' in result:
                error_indices_list.append(abs_idx)
                error_messages_list.append(result['error'])
            if 'negative_event_time' in result:
                negative_event_indices_list.append(abs_idx)
                negative_event_times_list.append(result['negative_event_time'])
            
            # Count successes
            if np.isfinite(result.get('t_startup', np.inf)):
                successful_count += 1
                chunk_successes += 1
        
        processed_count += len(chunk_results)
        
        # Force write to disk after each chunk
        h5_file.flush()
        
        # Update progress bars
        chunk_pbar.update(1)
        overall_pbar.update(current_chunk_size)
        
        # Update progress bar descriptions with success rate
        success_rate = (successful_count / processed_count) * 100 if processed_count > 0 else 0
        chunk_pbar.set_postfix({"Success Rate": f"{success_rate:.1f}%", "Successes": successful_count})
        overall_pbar.set_postfix({"Success Rate": f"{success_rate:.1f}%", "Current Chunk": f"{chunk_successes}/{current_chunk_size}"})
    
    # Close progress bars
    chunk_pbar.close()
    overall_pbar.close()
    
    # Save error and event data at the end
    if error_indices_list:
        h5_file.create_dataset('error_indices', data=np.array(error_indices_list), compression='gzip')
        dt = h5py.special_dtype(vlen=str)
        h5_file.create_dataset('error_messages', data=error_messages_list, dtype=dt, compression='gzip')
    
    if negative_event_indices_list:
        h5_file.create_dataset('negative_event_indices', data=np.array(negative_event_indices_list), compression='gzip')
        h5_file.create_dataset('negative_event_times', data=np.array(negative_event_times_list), compression='gzip')
    
    # Update final metadata
    h5_file.attrs['successful_startups'] = successful_count
    h5_file.attrs['computation_end_time'] = time.time()
    h5_file.attrs['total_computation_time'] = h5_file.attrs['computation_end_time'] - h5_file.attrs['computation_start_time']

if verbose:
    print(f"\n✅ Completed {processed_count} combinations")
    print(f"📁 Results saved to {output_filename}")
    print(f"💾 File size: {os.path.getsize(output_filename) / 1024**2:.1f} MB")

# Load results from file for final statistics (memory efficient)
with h5py.File(output_filename, 'r') as f:
    t_startup_array = f['t_startup'][:]
    dollar_lost_array = f['Dollar_Lost'][:]
    
    # Calculate final statistics without loading all data into memory
    finite_mask = np.isfinite(t_startup_array)
    successful_startups = np.sum(finite_mask)
    
    if verbose:
        print(f"\n📊 Final Statistics from saved file:")
        print(f"🎯 Successful startups: {successful_startups}/{len(t_startup_array)} ({100*successful_startups/len(t_startup_array):.1f}%)")
        
        if successful_startups > 0:
            finite_startups = t_startup_array[finite_mask]
            finite_dollars = dollar_lost_array[finite_mask]
            print(f"⏱️  Startup times (finite): min={np.min(finite_startups/(3600*24)):.1f} days, max={np.max(finite_startups)/(3600*24):.1f} days, "
                  f"mean={np.mean(finite_startups/(3600*24)):.1f} days")
            print(f"💰 Dollar losses (finite): min=M${np.min(finite_dollars/1e6):.2e}, max=M${np.max(finite_dollars/1e6):.2e}")

print(f"\n🎉 Streaming HDF5 save completed! All results are available in {output_filename}")

