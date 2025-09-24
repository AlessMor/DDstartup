import os, time, h5py
import numpy as np
from tqdm import tqdm
import importlib
import warnings
warnings.filterwarnings("ignore", module="scipy.integrate")
# --- Custom modules ---
from utils.tools import profile_system
# ======================== SETUP ========================

verbose = True
input_file_name = "config"  # or any other config file name (without .py)
analysis_type = 'lumped'               # 'T_seeded'/'lumped'
analysis_method = 'sobol'          # 'parametric'/'sobol'



config_module = importlib.import_module(f"inputs.{input_file_name}")
for k, v in config_module.__dict__.items():
    if not k.startswith("_"):
        globals()[k] = v
# ======================== INPUT PREPARATION ========================

# Select input_data and param_names based on analysis_type
if analysis_type == 'T_seeded':
    input_data = [
        V_plasma_field.data.to('m^3').magnitude,
        T_i_field.data.to('keV').magnitude,
        # n_tot_field.data.to('1/m^3').magnitude,
        # tau_p_T_field.data.to('s').magnitude,
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
        # n_tot_field.data.to('1/m^3').magnitude,
        # tau_p_T_field.data.to('s').magnitude,
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

chunk_size, batch_size, n_jobs, N_SAMPLES, order = profile_system()
if verbose:
    print(f"System profile suggests chunk_size={chunk_size}, batch_size={batch_size}, n_jobs={n_jobs}, N_SAMPLES={N_SAMPLES}, order={order}")


# ======================== SOBOL/CHAOSPY BRANCH ========================
if analysis_method == 'sobol':
    import chaospy as cp
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt
    if analysis_type == 'T_seeded':
        from utils.Tseeded_functions import compute_single_combination
    elif analysis_type == 'lumped':
        from utils.lump_functions import compute_single_combination
    else:
        print(f"Error: Unknown analysis type: {analysis_type}")
        exit()
    # Define parameter distributions (adjust as needed)
    param_bounds = [(arr.min(), arr.max()) for arr in input_data]
    print("Parameter bounds for Sobol analysis:")
    for name, (low, high) in zip(param_names, param_bounds):
        print(f"  {name}: min={low}, max={high}")

    # Separate variable and constant parameters
    variable_params = [(i, name, low, high) for i, (name, (low, high)) in enumerate(zip(param_names, param_bounds)) if high > low]
    constant_params = [(i, name, low) for i, (name, (low, high)) in enumerate(zip(param_names, param_bounds)) if high == low]

    if constant_params:
        print("The following parameters are constants in Sobol analysis:")
        for i, name, val in constant_params:
            print(f"  {name}: {val}")

    if not variable_params:
        raise ValueError("No parameters with a valid range for Sobol analysis.")

    # Build distributions for variable parameters
    sobol_param_names = [name for i, name, low, high in variable_params]
    distr_params = {name: cp.Uniform(low, high) for i, name, low, high in variable_params}
    joint_dist = cp.J(*distr_params.values())

    # Generate Sobol samples for variable parameters

    samples_var = joint_dist.sample(N_SAMPLES, rule='sobol').T  # shape: (N_SAMPLES, n_var_params)

    # For each sample, build the full parameter vector (variable + constants)
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

    # Prepare for parallel evaluation
    def run_sample(idx, sample):
        try:
            input_arrays_flat_sample = [np.array([val]) for val in sample]
            param_shapes_array_sample = np.array([1]*len(sample), dtype=np.int64)
            if analysis_type == 'T_seeded':
                result = compute_single_combination(0, input_arrays_flat_sample, param_shapes_array_sample, total_time)
            else:
                result = compute_single_combination(0, input_arrays_flat_sample, param_shapes_array_sample)
            return result
        except Exception as e:
            return {'error': str(e)}

    print(f"Evaluating {N_SAMPLES} Sobol samples in parallel...")
    results = Parallel(n_jobs=-1, verbose=0)(delayed(run_sample)(i, sample) for i, sample in enumerate(samples_full))

    # Extract output metric (Dollar_Lost)
    dollar_lost = np.array([
        r['Dollar_Lost'] if (r is not None and 'Dollar_Lost' in r and np.isfinite(r['Dollar_Lost'])) else np.nan
        for r in results
    ])
    valid_mask = np.isfinite(dollar_lost)
    valid_samples = samples_full[valid_mask]
    valid_dollars = dollar_lost[valid_mask]

    print(f"Valid results: {np.sum(valid_mask)}/{N_SAMPLES}")
    if np.sum(valid_mask) < 10:
        print("Not enough valid samples for Sobol analysis.")
    else:

        poly_expansion = cp.generate_expansion(order, joint_dist)
        print("Fitting PCE surrogate model...") if verbose else None
        pce_model = cp.fit_regression(poly_expansion, valid_samples.T, valid_dollars)
        print("Computing first order Sobol indices...") if verbose else None
        sobol_first = cp.Sens_m(pce_model, joint_dist)
        print("Computing total order Sobol indices...") if verbose else None
        sobol_total = cp.Sens_t(pce_model, joint_dist)
        print("\nSensitivity Analysis Results:") if verbose else None
        print("Parameter\t\tFirst-order\tTotal-order")  if verbose else None
        print("------------------------------------------------") if verbose else None
        for i, param_name in enumerate(distr_params.keys()):
            print(f"{param_name:15s}\t{sobol_first[i]:.6f}\t{sobol_total[i]:.6f}")
        plt.figure(figsize=(10, 6))
        x = np.arange(len(distr_params))
        width = 0.35
        plt.bar(x - width/2, sobol_first, width, label='First-order')
        plt.bar(x + width/2, sobol_total, width, label='Total-order')
        plt.xlabel('Parameters')
        plt.ylabel('Sobol Indices')
        plt.title('Sensitivity Analysis (Dollar_Lost)')
        plt.xticks(x, list(distr_params.keys()), rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save plot to outputs/ with timestamp and analysis info
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_filename = f"outputs/sobol_indices_{timestamp}_{analysis_method}_{analysis_type}.png"
        plt.savefig(plot_filename, dpi=200)
        print(f"Sobol indices plot saved to {plot_filename}")
        plt.close()

        # Save Sobol indices to txt file
        txt_filename = f"outputs/sobol_indices_{timestamp}_{analysis_method}_{analysis_type}.txt"
        with open(txt_filename, 'w') as ftxt:
            ftxt.write("Parameter\tFirst-order\tTotal-order\n")
            for i, param_name in enumerate(distr_params.keys()):
                ftxt.write(f"{param_name}\t{sobol_first[i]:.6f}\t{sobol_total[i]:.6f}\n")
        print(f"Sobol indices saved to {txt_filename}")
    exit()

# ======================== PARAMETRIC ANALYSIS BRANCH ========================
# (rest of your code remains unchanged)

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
output_filename = f"outputs/dd_startup_{timestamp}_{analysis_method}_{analysis_type}.h5"

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
            print(f"Error: Unknown analysis type: {analysis_type}")
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

