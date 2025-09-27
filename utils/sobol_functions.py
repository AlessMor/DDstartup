import numpy as np
import chaospy as cp
import time
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_sample(idx, sample, compute_single_combination, analysis_type, total_time):
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

def sobol_analysis(
    input_data, param_names, N_SAMPLES, order, analysis_type, total_time,
    compute_single_combination, verbose=True
):
    # 1. Get parameter bounds
    param_bounds = [(arr.min(), arr.max()) for arr in input_data.values()]
    if verbose:
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
    samples_var = joint_dist.sample(N_SAMPLES, rule='sobol').T  # shape: (N_SAMPLES, n_var_params)

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

    # 6. Parallel evaluation using ProcessPoolExecutor
    print(f"Evaluating {N_SAMPLES} Sobol samples in parallel...")
    results = [None] * N_SAMPLES
    with ProcessPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(run_sample, i, sample, compute_single_combination, analysis_type, total_time): i
            for i, sample in enumerate(samples_full)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                results[i] = future.result()
            except Exception as exc:
                results[i] = {'error': str(exc)}

    # 7. Extract output metric (unrealized_gains)
    unrealized_gains = np.array([
        r['unrealized_gains'] if (r is not None and 'unrealized_gains' in r and np.isfinite(r['unrealized_gains'])) else np.nan
        for r in results
    ])
    valid_mask = np.isfinite(unrealized_gains)
    valid_samples = samples_full[valid_mask]
    valid_dollars = unrealized_gains[valid_mask]

    print(f"Valid results: {np.sum(valid_mask)}/{N_SAMPLES}")
    if np.sum(valid_mask) < 10:
        print("Not enough valid samples for Sobol analysis.")
        return None

    # 8. Fit PCE surrogate model and compute Sobol indices
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

    # 9. Save plot and results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    x = np.arange(len(distr_params))
    width = 0.35
    plt.bar(x - width/2, sobol_first, width, label='First-order')
    plt.bar(x + width/2, sobol_total, width, label='Total-order')
    plt.xlabel('Parameters')
    plt.ylabel('Sobol Indices')
    plt.title('Sensitivity Analysis (unrealized_gains)')
    plt.xticks(x, list(distr_params.keys()), rotation=45)
    plt.legend()
    plt.tight_layout()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_filename = f"outputs/sobol_indices_{timestamp}_sobol_{analysis_type}.png"
    plt.savefig(plot_filename, dpi=200)
    print(f"Sobol indices plot saved to {plot_filename}")
    plt.close()

    txt_filename = f"outputs/sobol_indices_{timestamp}_sobol_{analysis_type}.txt"
    with open(txt_filename, 'w') as ftxt:
        ftxt.write("Parameter\tFirst-order\tTotal-order\n")
        for i, param_name in enumerate(distr_params.keys()):
            ftxt.write(f"{param_name}\t{sobol_first[i]:.6f}\t{sobol_total[i]:.6f}\n")
    print(f"Sobol indices saved to {txt_filename}")

    return {
        'sobol_first': sobol_first,
        'sobol_total': sobol_total,
        'param_names': list(distr_params.keys()),
        'plot_filename': plot_filename,
        'txt_filename': txt_filename
    }