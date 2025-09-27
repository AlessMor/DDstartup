import os, time, h5py
import numpy as np
from tqdm import tqdm
import importlib
import warnings
import concurrent.futures
warnings.filterwarnings("ignore", module="scipy.integrate")
# --- Custom modules ---
from utils.tools import profile_system
from utils.tools import inputs_names, outputs_names
# ======================== SETUP ========================

verbose = True
input_file_name = "config_test"  # or any other config file name (without .py)
analysis_type = 'T_seeded'               # 'T_seeded'/'lump'
analysis_method = 'parametric'          # 'parametric'/'sobol'
vector_length = 100 # length of vectors to be stored if T_seeded

config_module = importlib.import_module(f"inputs.{input_file_name}")
for k, v in config_module.__dict__.items():
    if not k.startswith("_"):
        globals()[k] = v
        
# ======================== INPUT PREPARATION ========================

# Select input_data and param_names based on analysis_type
if analysis_type == 'T_seeded':
    input_data = {
        'V_plasma': V_plasma_field.data.to('m^3').magnitude,
        'T_i': T_i_field.data.to('keV').magnitude,
        'n_tot': n_tot_field.data.to('1/m^3').magnitude,
        'tau_p_T': tau_p_T_field.data.to('s').magnitude,
        'P_aux': P_aux_field.data.to('W').magnitude,
        'P_aux_DT_eq': P_aux_DT_eq_field.data.to('W').magnitude,
        'TBR_DT': TBR_DT_field.data.to_base_units().magnitude,
        'TBR_DDn': TBR_DDn_field.data.to_base_units().magnitude,
        'tau_ifc': tau_ifc_field.data.to('s').magnitude,
        'tau_ofc': tau_ofc_field.data.to('s').magnitude,
        'eta_th': eta_th_field.data.to_base_units().magnitude,
        'capacity_factor': capacity_factor_field.data.to_base_units().magnitude,
        'cost_of_electricity': cost_of_electricity_field.data.to('1/J').magnitude
    }
elif analysis_type == 'lump':
    input_data = {
        'V_plasma': V_plasma_field.data.to('m^3').magnitude,
        'T_i': T_i_field.data.to('keV').magnitude,
        'n_tot': n_tot_field.data.to('1/m^3').magnitude,
        'tau_p_T': tau_p_T_field.data.to('s').magnitude,
        'tau_p_He3': tau_p_He3_field.data.to('s').magnitude,
        'P_aux': P_aux_field.data.to('W').magnitude,
        'P_aux_DT_eq': P_aux_DT_eq_field.data.to('W').magnitude,
        'TBR_DT': TBR_DT_field.data.to_base_units().magnitude,
        'TBR_DDn': TBR_DDn_field.data.to_base_units().magnitude,
        'I_target': I_target_field.data.to('kg').magnitude,
        'eta_th': eta_th_field.data.to_base_units().magnitude,
        'capacity_factor': capacity_factor_field.data.to_base_units().magnitude,
        'cost_of_electricity': cost_of_electricity_field.data.to('1/J').magnitude
    }
param_names = list(input_data.keys())
if verbose:
    print("Input parameter fields:")
    for name, arr in input_data.items():
        print(f" - {name}: {arr.shape[0]} points, range [{arr.min():.3e}, {arr.max():.3e}]")


if analysis_type == 'T_seeded':
    from utils.Tseeded_functions import compute_single_combination
elif analysis_type == 'lump':
    from utils.lump_functions import compute_single_combination
else:
    print(f"Error: Unknown analysis type: {analysis_type}")
    exit()
# ======================== SYSTEM PROFILER ========================
chunk_size, batch_size, n_jobs, N_SAMPLES, order = profile_system()


# ======================== SOBOL/CHAOSPY BRANCH ========================
if analysis_method == 'sobol':
    from utils.sobol_functions import sobol_analysis

    sobol_analysis(
        input_data=input_data,
        param_names=param_names,
        N_SAMPLES=int(N_SAMPLES),
        order=order,
        analysis_type=analysis_type,
        total_time=10*365*24*3600,
        compute_single_combination=compute_single_combination,
        verbose=True
    )
    exit()

# ======================== PARAMETRIC ANALYSIS BRANCH ========================

# Compute parameter space size
param_shapes = [arr.shape[0] for arr in input_data.values()] # Shape of each parameter array
n_combinations = np.prod(param_shapes) # Total number of combinations

if verbose:
    print(f"Total number of parameter combinations: {n_combinations}")

# Flatten arrays for easier indexing later
input_arrays = [np.asarray(arr) for arr in input_data.values()]
input_arrays_flat = [arr.flatten() for arr in input_arrays]
param_shapes_array = np.array(param_shapes, dtype=np.int64)

# ======================== OUTPUT SETUP ========================

# Timestamped filename
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_filename = f"outputs/dd_startup_{timestamp}_{analysis_method}_{analysis_type}.h5"

if verbose:
    print(f"Streaming results to {output_filename}...")

# Fields to be saved in HDF5
data_fields = list(dict.fromkeys(inputs_names + outputs_names)) # creates a unique list preserving order and avoiding duplicates

# ======================== MAIN COMPUTATION ========================
with h5py.File(output_filename, 'w') as h5_file:
    # --- Metadata of h5 file ---
    h5_file.attrs.update({
        'total_combinations': n_combinations,
        'computation_start_time': time.time(),
        'parameter_shapes': param_shapes,
        'method': analysis_method,
        'analysis_type': analysis_type,
    })

    vector_fields = ['N_ofc', 'N_ifc', 'N_stor', 'n_T', 'n_D', 'P_DDn', 'P_DDp', 'P_DT', 'TBE_vector']  # Add any other fields that should store vectors
    vlen_dtype = h5py.special_dtype(vlen=np.float64)

    # --- Pre-allocate datasets ---
    vector_length = 100
    datasets = {
        field: h5_file.create_dataset(
            field,
            (n_combinations, vector_length) if field in vector_fields else (n_combinations,),
            # Use string dtype for 'error', bool for 'sol_success', float64 otherwise
            dtype=(
                h5py.string_dtype(encoding='utf-8') if field == 'error'
                else (bool if field == 'sol_success' else np.float64)
                if field not in vector_fields else np.float64
            ),
            chunks=(min(chunk_size, n_combinations), vector_length) if field in vector_fields else (min(chunk_size, n_combinations),),
            compression='lzf',
        )
        for field in data_fields
    }

    # --- Save parameter grids ---
    param_group = h5_file.create_group('parameter_fields')
    for name, arr in zip(param_names, input_arrays):
        param_group.create_dataset(f'{name}_values', data=arr, compression='gzip')

    # --- initiate errors/events tracking ---
    error_indices, error_messages = [], []
    
    # --- initialize progress bars ---
    processed_count, successful_count = 0, 0
    total_chunks = (n_combinations + chunk_size - 1) // chunk_size
    overall_pbar = tqdm(total=n_combinations, desc="Total computation", unit="comb", position=0)
    chunk_pbar = tqdm(total=chunk_size, desc="Current batch", unit="comb", position=1, leave=False)

    # --- Process parameter space in chunks ---
    for chunk_start in range(0, n_combinations, chunk_size):

        chunk_end = min(chunk_start + chunk_size, n_combinations)
        chunk_indices = np.arange(chunk_start, chunk_end)

        # Reset chunk_pbar for this batch
        chunk_pbar.reset(total=chunk_end - chunk_start)

        # Prepare tasks for concurrent.futures
        if analysis_type == 'lump':
            task_args = [(idx, input_arrays_flat, param_shapes_array) for idx in chunk_indices]
            task_func = compute_single_combination
        elif analysis_type == 'T_seeded':
            task_args = [(idx, input_arrays_flat, param_shapes_array, total_time, vector_length) for idx in chunk_indices]
            task_func = compute_single_combination
        else: 
            print(f"Error: Unknown analysis type: {analysis_type}")
            break

        chunk_results = [None] * (chunk_end - chunk_start)
        chunk_successes = 0
        
        # --- Concurrent execution ---
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            future_to_idx = {executor.submit(task_func, *args): i for i, args in enumerate(task_args)}
            
            # --- As tasks (parameter combinations) finish in the process pool, this loop collects their results. ---
            for future in concurrent.futures.as_completed(future_to_idx):
                i = future_to_idx[future]
                abs_idx = chunk_start + i
                try:
                    result = future.result()
                except Exception as exc:
                    result = {'error': str(exc)}
                chunk_results[i] = result
                

                # --- For each result, save all its fields (inputs, outputs) into the correct HDF5 datasets.
                for field in data_fields:
                    value = result.get(field, None)
                                    
                    if field == 'error':
                        # Always convert to string before saving
                        if value is None or (isinstance(value, float) and not np.isfinite(value)):
                            datasets[field][abs_idx] = ""
                        else:
                            datasets[field][abs_idx] = str(value)
                    elif field == 'sol_success':
                        datasets[field][abs_idx] = bool(value)
                    elif field in vector_fields:
                        arr = np.asarray(value)
                        if arr.ndim == 0 or arr.size == 0:
                            # Fill with scalar value if not a valid vector
                            scalar = float(value) if value is not None else np.nan
                            padded = np.full(vector_length, scalar)
                            datasets[field][abs_idx, :] = padded
                        elif arr.shape[0] == vector_length:
                            datasets[field][abs_idx, :] = arr
                        else:
                            # Pad or truncate to vector_length
                            padded = np.full(vector_length, arr[-1] if arr.size > 0 else np.nan)
                            padded[:min(vector_length, arr.size)] = arr[:vector_length]
                            datasets[field][abs_idx, :] = padded
                            datasets['error'][abs_idx] += f"Warning: {field} length {arr.size} != expected {vector_length}, padded with NaN"
                    else:
                        if value is None or (isinstance(value, float) and not np.isfinite(value)):
                            datasets[field][abs_idx] = np.inf
                        elif isinstance(value, str):
                            datasets[field][abs_idx] = np.nan
                        elif hasattr(value, "__len__") and not isinstance(value, str):
                            try:
                                if len(value) > 0:
                                    scalar_value = float(value[-1])
                                else:
                                    scalar_value = np.inf
                            except Exception:
                                scalar_value = float(value)
                            datasets[field][abs_idx] = scalar_value if np.isfinite(scalar_value) else np.inf
                        else:
                            try:
                                scalar_value = float(value)
                            except Exception:
                                scalar_value = np.inf
                            datasets[field][abs_idx] = scalar_value if np.isfinite(scalar_value) else np.inf
                # Count successes
                if np.isfinite(result.get('t_startup', np.inf)):
                    successful_count += 1
                    chunk_successes += 1

                # Update progress bars for each combination
                chunk_pbar.update(1)
                overall_pbar.update(1)

        processed_count += len(chunk_results)

        # Save to disk
        h5_file.flush()

        # Set postfix for chunk and overall
        success_rate = (successful_count / processed_count) * 100 if processed_count else 0
        chunk_pbar.set_postfix({"Success Rate": f"{success_rate:.1f}%", "Successes": successful_count})
        overall_pbar.set_postfix({"Success Rate": f"{success_rate:.1f}%", "Current Chunk": f"{chunk_successes}/{len(chunk_indices)}"})

    # --- Close progress bars ---
    chunk_pbar.close()
    overall_pbar.close()

    # --- Close progress bars ---
    chunk_pbar.close()
    overall_pbar.close()

    # --- Final metadata ---
    h5_file.attrs.update({
        'successful_startups': successful_count,
        'computation_end_time': time.time(),
        'total_computation_time': time.time() - h5_file.attrs['computation_start_time']
    })


# ======================== SUMMARY ========================

if verbose:

    print(f"\n✅ Completed {processed_count} combinations")
    print(f"📁 Results saved to {output_filename}")
    print(f"💾 File size: {os.path.getsize(output_filename) / 1024**2:.1f} MB")

# Print error messages if present
if os.path.exists(output_filename):
    with h5py.File(output_filename, 'r') as f:
        if 'error' in f:
            error_msgs = f['error'][:]
            # Decode bytes and skip empty errors
            error_msgs_str = [
                msg.decode('utf-8') if isinstance(msg, bytes) else str(msg)
                for msg in error_msgs
            ]
            non_empty_errors = [(i, msg) for i, msg in enumerate(error_msgs_str) if msg]
            if non_empty_errors:
                print("\n❌ Errors encountered during computation:")
                for i, msg in non_empty_errors:
                    print(f"  [{i}] {msg}")

# Reload results for quick statistics
with h5py.File(output_filename, 'r') as f:
    t_startup_array = f['t_startup'][:]
    unrealized_gains_array = f['unrealized_gains'][:]

    finite_mask = np.isfinite(t_startup_array)
    successful_startups = np.sum(finite_mask)

    if verbose:
        print("\n📊 Final Statistics:")
        print(f"🎯 Successful startups: {successful_startups}/{len(t_startup_array)} ({100*successful_startups/len(t_startup_array):.1f}%)")

        if successful_startups:
            finite_startups = t_startup_array[finite_mask] / (3600 * 24)  # convert to days
            finite_dollars = unrealized_gains_array[finite_mask] / 1e6         # convert to million $
            print(f"⏱️ Startup times: min={finite_startups.min():.1f} days, "
                  f"max={finite_startups.max():.1f} days, mean={finite_startups.mean():.1f} days")
            print(f"💰 Dollar losses: min=M${finite_dollars.min():.2e}, max=M${finite_dollars.max():.2e}")
            
            # print the list of input parameters that led to the minimum startup time
            min_index = np.where(t_startup_array == t_startup_array[finite_mask].min())[0][0]
            print(f"\n⚡ Fastest startup [SI units]:")
            with h5py.File(output_filename, 'r') as f:
                for field in data_fields:
                    if field in f:
                        value = f[field][min_index]
                        # If value is a vector, print the last value
                        if isinstance(value, np.ndarray) and value.ndim > 0 and value.size > 1:
                            print(f"  {field}: {value[-1]} (last value of array)")
                        else:
                            print(f"  {field}: {value}")

print(f"\n🎉 Streaming HDF5 save completed! Results are in {output_filename}")