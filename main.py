import os, time, h5py
import numpy as np
from tqdm import tqdm

# --- Custom modules ---
from utils.tools import choose_batch_size
# ======================== SETUP ========================

from inputs.config_test import *            # input fields and parameters
verbose = True
analysis_type = 'T_seeded'               # 'T_seeded' or 'lumped'


# ======================== INPUT PREPARATION ========================


# Select input_data and param_names based on analysis_type
if analysis_type == 'T_seeded':
    input_data = [
        V_plasma_field.data.to('m^3').magnitude,
        T_i_field.data.to('keV').magnitude,
        n_tot_field.data.to('1/m^3').magnitude,
        tau_p_T_field.data.to('s').magnitude,
        P_aux_field.data.to('W').magnitude,
        P_lost_rad_field.data.to('W').magnitude,
        P_aux_all_DT_field.data.to('W').magnitude,
        P_lost_rad_all_DT_field.data.to('W').magnitude,
        TBR_DT_field.data.to_base_units().magnitude,
        TBR_DDn_field.data.to_base_units().magnitude,
        tau_ifc_field.data.to('s').magnitude,
        tau_ofc_field.data.to('s').magnitude,
        eta_th_field.data.to_base_units().magnitude,
        plant_avail_field.data.to_base_units().magnitude,
        Cost_per_kWh_field.data.to('1/J').magnitude
    ]
    param_names = [
        'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'P_aux',
        'P_lost_rad', 'P_aux_all_DT', 'P_lost_rad_all_DT', 
        'TBR_DT','TBR_DDn','tau_ifc', 'tau_ofc',
        'eta_th', 'plant_avail', 'Cost_per_kWh'
    ]
elif analysis_type == 'lumped':
    input_data = [
        V_plasma_field.data.to('m^3').magnitude,
        T_i_field.data.to('keV').magnitude,
        n_tot_field.data.to('1/m^3').magnitude,
        tau_p_T_field.data.to('s').magnitude,
        tau_p_He3_field.data.to('s').magnitude,
        P_aux_field.data.to('W').magnitude,
        P_lost_rad_field.data.to('W').magnitude,
        P_aux_all_DT_field.data.to('W').magnitude,
        P_lost_rad_all_DT_field.data.to('W').magnitude,
        TBR_DT_field.data.to_base_units().magnitude,
        TBR_DDn_field.data.to_base_units().magnitude,
        I_target_field.data.to('kg').magnitude,
        eta_th_field.data.to_base_units().magnitude,
        plant_avail_field.data.to_base_units().magnitude,
        Cost_per_kWh_field.data.to('1/J').magnitude,
    ]
    param_names = [
        'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'tau_p_He3',
        'P_aux', 'P_lost_rad', 'P_aux_all_DT', 'P_lost_rad_all_DT',
        'TBR_DT', 'TBR_DDn', 'I_target', 'eta_th', 'plant_avail', 'Cost_per_kWh', 
    ]

if verbose:
    print("Input parameter fields:")
    for name, arr in zip(param_names, input_data):
        print(f" - {name}: {arr.shape[0]} points, range [{arr.min():.3e}, {arr.max():.3e}]")

# Compute parameter space size
param_shapes = [arr.shape[0] for arr in input_data]
n_combinations = np.prod(param_shapes)

if verbose:
    print(f"Total number of parameter combinations: {n_combinations}")

# Flatten arrays for easier indexing later
input_arrays = [np.asarray(arr) for arr in input_data]
input_arrays_flat = [arr.flatten() for arr in input_arrays]
param_shapes_array = np.array(param_shapes, dtype=np.int64)


# ======================== OUTPUT SETUP ========================

# Timestamped filename
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_filename = f"outputs/dd_startup_results_{timestamp}.h5"

if verbose:
    print(f"Streaming results to {output_filename}...")

# Fields to be saved in HDF5
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

chunk_size = min(10_000, n_combinations)  
batch_size = choose_batch_size(n_combinations)
n_jobs = -1             # use all available cores



# ======================== MAIN COMPUTATION ========================
with h5py.File(output_filename, 'w') as h5_file:
    # --- Metadata ---
    h5_file.attrs.update({
        'total_combinations': n_combinations,
        'computation_start_time': time.time(),
        'parameter_shapes': param_shapes,
    })

    # --- Pre-allocate datasets ---
    datasets = {
        field: h5_file.create_dataset(
            field,
            (n_combinations,),
            dtype=bool if field == 'sol_success' else np.float64,
            chunks=(min(chunk_size, n_combinations),),
            compression='lzf',
        )
        for field in result_fields
    }

    # --- Save parameter grids ---
    param_group = h5_file.create_group('parameter_fields')
    for name, arr in zip(param_names, input_arrays):
        param_group.create_dataset(f'{name}_values', data=arr, compression='gzip')

    # --- Track errors/events ---
    error_indices, error_messages = [], []
    negative_event_indices, negative_event_times = [], []

    processed_count, successful_count = 0, 0
    total_chunks = (n_combinations + chunk_size - 1) // chunk_size


    # Progress bars
    overall_pbar = tqdm(total=n_combinations, desc="Total computation", unit="comb", position=0)
    chunk_pbar = tqdm(total=chunk_size, desc="Current batch", unit="comb", position=1, leave=False)

    # --- Process parameter space in chunks ---
    for chunk_start in range(0, n_combinations, chunk_size):

        import concurrent.futures
        chunk_end = min(chunk_start + chunk_size, n_combinations)
        chunk_indices = np.arange(chunk_start, chunk_end)

        # Reset chunk_pbar for this batch
        chunk_pbar.reset(total=chunk_end - chunk_start)

        # Prepare tasks for concurrent.futures
        if analysis_type == 'lumped':
            from utils.lump_functions import compute_single_combination
            task_args = [(idx, input_arrays_flat, param_shapes_array) for idx in chunk_indices]
            task_func = compute_single_combination
        elif analysis_type == 'T_seeded':
            from utils.Tseeded_functions import compute_single_combination
            task_args = [(idx, input_arrays_flat, param_shapes_array, total_time) for idx in chunk_indices]
            task_func = compute_single_combination
        else: 
            print("Error: Unknown analysis type.")
            break

        chunk_results = [None] * (chunk_end - chunk_start)
        chunk_successes = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            future_to_idx = {executor.submit(task_func, *args): i for i, args in enumerate(task_args)}
            for future in concurrent.futures.as_completed(future_to_idx):
                i = future_to_idx[future]
                abs_idx = chunk_start + i
                try:
                    result = future.result()
                except Exception as exc:
                    result = {'error': str(exc)}
                chunk_results[i] = result

                # Fill datasets with results (handle arrays vs scalars)
                for field in result_fields:
                    value = result.get(field, np.inf if field != 'sol_success' else False)

                    if field == 'sol_success':
                        datasets[field][abs_idx] = bool(value)
                    else:
                        if hasattr(value, "__len__") and not isinstance(value, str):
                            if field in ['P_DT', 'P_DDn', 'P_DDp']:
                                scalar_value = float(value[-1]) if len(value) else np.inf
                            else:
                                scalar_value = float(np.mean(value)) if len(value) else np.inf
                        else:
                            scalar_value = float(value)

                        datasets[field][abs_idx] = scalar_value if np.isfinite(scalar_value) else np.inf

                # Track errors and special events
                if 'error' in result:
                    error_indices.append(abs_idx)
                    error_messages.append(result['error'])
                if 'negative_event_time' in result:
                    negative_event_indices.append(abs_idx)
                    negative_event_times.append(result['negative_event_time'])

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

    # --- Save error/event data ---
    if error_indices:
        h5_file.create_dataset('error_indices', data=np.array(error_indices), compression='gzip')
        dt = h5py.special_dtype(vlen=str)
        h5_file.create_dataset('error_messages', data=error_messages, dtype=dt, compression='gzip')

    if negative_event_indices:
        h5_file.create_dataset('negative_event_indices', data=np.array(negative_event_indices), compression='gzip')
        h5_file.create_dataset('negative_event_times', data=np.array(negative_event_times), compression='gzip')

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
        if 'error_messages' in f:
            error_msgs = f['error_messages'][:]
            if len(error_msgs) > 0:
                print("\n❌ Errors encountered during computation:")
                for i, msg in enumerate(error_msgs):
                    print(f"  [{i}] {msg}")

# Reload results for quick statistics
with h5py.File(output_filename, 'r') as f:
    t_startup_array = f['t_startup'][:]
    dollar_lost_array = f['Dollar_Lost'][:]

    finite_mask = np.isfinite(t_startup_array)
    successful_startups = np.sum(finite_mask)

    if verbose:
        print("\n📊 Final Statistics:")
        print(f"🎯 Successful startups: {successful_startups}/{len(t_startup_array)} ({100*successful_startups/len(t_startup_array):.1f}%)")

        if successful_startups:
            finite_startups = t_startup_array[finite_mask] / (3600 * 24)  # convert to days
            finite_dollars = dollar_lost_array[finite_mask] / 1e6         # convert to million $
            print(f"⏱️ Startup times: min={finite_startups.min():.1f} days, "
                  f"max={finite_startups.max():.1f} days, mean={finite_startups.mean():.1f} days")
            print(f"💰 Dollar losses: min=M${finite_dollars.min():.2e}, max=M${finite_dollars.max():.2e}")
            
            # print the list of input parameters that led to the minimum startup time
            min_index = np.where(t_startup_array == t_startup_array[finite_mask].min())[0][0]
            print(f"\n🏆 Fastest startup at index {min_index}:")
            for param_name, arr in zip(param_names, input_arrays_flat):

                if min_index < len(arr):
                    param_value = arr[min_index]
                else:
                    param_value = arr[-1]
                print(f"  {param_name}: {param_value}")

            # Print the entire row for all result fields
            print("\nFull result row for fastest startup:")
            with h5py.File(output_filename, 'r') as f:
                for field in result_fields:
                    if field in f:
                        value = f[field][min_index]
                        print(f"  {field}: {value}")

print(f"\n🎉 Streaming HDF5 save completed! Results are in {output_filename}")

