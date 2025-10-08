import numpy as np
import multiprocessing
import psutil
import argparse

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='DD Startup Analysis Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'params',
        type=str,
        help='Parameter config file name (e.g., "config") or path to file (e.g., "inputs/config.py")'
    )
    
    parser.add_argument(
        'config',
        type=str,
        help='YAML configuration file name (e.g., "parametric_tseeded") or path to file (e.g., "run_configs/parametric.yaml")'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration without running analysis'
    )
    
    return parser.parse_args()


def index_to_params(linear_index, param_shapes):
    """Convert linear index to multi-dimensional parameter indices"""
    n_params = len(param_shapes)
    indices = np.zeros(n_params, dtype=np.int64)
    
    remaining = linear_index
    for i in range(n_params - 1, -1, -1):
        indices[i] = remaining % param_shapes[i]
        remaining = remaining // param_shapes[i]
    
    return indices


def check_analysis_field(analysis_type, input_data):
    """
    Checks if input_data contains all required fields for the given analysis_type.
    Raises ValueError if any required field is missing.
    """
    # Common required fields
    required_fields = [
        'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'P_aux', 'P_aux_all_DT',
        'TBR_DT', 'TBR_DDn',
        'eta_th', 'plant_avail', 'Cost_per_kWh'
    ]
    # Add analysis-specific fields
    if analysis_type == 'T_seeded':
        required_fields += ['tau_ifc', 'tau_ofc']
    elif analysis_type == 'lump':
        required_fields += ['I_st', 'tau_p_He3']

    missing = [field for field in required_fields if field not in input_data]
    if missing:
        raise ValueError(f"Missing required input fields for analysis '{analysis_type}': {missing}")
    return True

def choose_batch_size(n_combinations, min_batch=10, max_batch=500):
    """
    Dynamically choose a joblib batch size based on available cores
    and problem size.
    """
    n_cores = multiprocessing.cpu_count()
    
    # Heuristic: more cores → smaller batch size, but not too small
    base_batch = max(min_batch, n_combinations // (n_cores * 50))
    batch_size = min(max_batch, max(min_batch, base_batch))
    
    return batch_size

def system_profiler():
    """
    Returns a dict with system info: n_cores, total_gb, available_gb, cpu_freq, etc.
    """
    n_cores = multiprocessing.cpu_count()
    mem = psutil.virtual_memory()
    total_gb = mem.total / 1e9
    available_gb = mem.available / 1e9
    cpu_freq = psutil.cpu_freq().max if hasattr(psutil.cpu_freq(), 'max') else None
    return {
        'n_cores': n_cores,
        'total_gb': total_gb,
        'available_gb': available_gb,
        'cpu_freq': cpu_freq
    }

def optimal_n_jobs(sysinfo=None):
    if sysinfo is None:
        sysinfo = system_profiler()
    n_cores = sysinfo['n_cores']
    if n_cores >= 16:
        return 2*n_cores
    elif n_cores >= 8:
        return 8
    elif n_cores >= 4:
        return 4
    else:
        return 2

def optimal_chunk_size(sysinfo=None):
    if sysinfo is None:
        sysinfo = system_profiler()
    n_cores = sysinfo['n_cores']
    if n_cores >= 16:
        return 5000
    elif n_cores >= 8:
        return 1000
    elif n_cores >= 4:
        return 500
    else:
        return 200

def optimal_batch_size(sysinfo=None):
    if sysinfo is None:
        sysinfo = system_profiler()
    n_cores = sysinfo['n_cores']
    if n_cores >= 16:
        return 500
    elif n_cores >= 8:
        return 250
    elif n_cores >= 4:
        return 100
    else:
        return 50

def optimal_N_SAMPLES(sysinfo=None):
    if sysinfo is None:
        sysinfo = system_profiler()
    n_cores = sysinfo['n_cores']
    total_gb = sysinfo['total_gb']
    if n_cores >= 16:
        n = 100e3
    elif n_cores >= 8:
        n = 100e3
    elif n_cores >= 4:
        n = 1024
    else:
        n = 512
    if total_gb < 8:
        n = min(n, 512)
    return n

def optimal_sobol_order(sysinfo=None):
    if sysinfo is None:
        sysinfo = system_profiler()
    n_cores = sysinfo['n_cores']
    if n_cores >= 16:
        return 3
    elif n_cores >= 8:
        return 3
    else:
        return 2

def profile_system():
    """
    Profile system and return recommended chunk_size, batch_size, n_jobs, N_SAMPLES, order for Sobol analysis.
    Returns: (chunk_size, batch_size, n_jobs, N_SAMPLES, order)
    """
    sysinfo = system_profiler()
    n_jobs = optimal_n_jobs(sysinfo)
    chunk_size = optimal_chunk_size(sysinfo)
    batch_size = optimal_batch_size(sysinfo)
    N_SAMPLES = optimal_N_SAMPLES(sysinfo)
    order = optimal_sobol_order(sysinfo)
    
    print(f"[System profiling] Cores: {sysinfo['n_cores']}, RAM: {sysinfo['total_gb']:.1f} GB")
    print(f"[System profiling] n_jobs={n_jobs}, chunk_size={chunk_size}, batch_size={batch_size}, N_SAMPLES={N_SAMPLES}, order={order}")
    return chunk_size, batch_size, n_jobs, N_SAMPLES, order

inputs_names = [
    'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'tau_He3',
    'P_aux', 'P_aux_DT_eq', 
    'TBR_DT', 'TBR_DDn', 'tau_ifc', 'tau_ofc',
    'I_target', 'eta_th', 'capacity_factor', 'cost_of_electricity'
]

outputs_names = [
    'n_T', 'n_D', 'n_He3', 
    'N_ofc', 'N_ifc', 'N_stor',
    'P_DDn', 'P_DDp', 'P_DT', 'P_DT_eq', 
    't_startup', 
    'Q_DD', 'Q_DT_eq', 'TBE',
    'E_lost', 'unrealized_profits',
    'sol_success', 'linear_index', 'error'
    ]
PARAM_KEYS = {
    'T_seeded': [
        'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'P_aux', 'P_aux_DT_eq',
        'TBR_DT', 'TBR_DDn', 'tau_ifc', 'tau_ofc', 'eta_th',
        'capacity_factor', 'cost_of_electricity'
    ],
    'lump': [
        'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'tau_p_He3', 'P_aux', 'P_aux_DT_eq',
        'TBR_DT', 'TBR_DDn', 'I_target', 'eta_th', 'capacity_factor', 'cost_of_electricity'
    ]
}
PARAM_UNITS = {
    'V_plasma': 'm³','T_i': 'keV', 'n_tot': 'm⁻³',
    'tau_p_T': 's', 'tau_p_He3': 's',
    'P_aux': 'W', 'P_aux_DT_eq': 'W',
    'TBR_DT': '-', 'TBR_DDn': '-', 'tau_ifc': 's', 'tau_ofc': 's',
    'I_target': 'kg', 'eta_th': '-', 'capacity_factor': '-', 'cost_of_electricity': '1/J',
    'n_T': 'm⁻³', 'n_D': 'm⁻³', 'n_He3': 'm⁻³',
    'N_ofc': 'kg', 'N_ifc': 'kg', 'N_stor': 'kg',
    'P_DDn': 'W', 'P_DDp': 'W', 'P_DT': 'W', 'P_DT_eq': 'W',    
    't_startup': 's',
    'Q_DD': '-', 'Q_DT_eq': '-', 'TBE': '-',
    'E_lost': 'J', 'unrealized_profits': 'J',
    # Add more as needed
}

def make_output_dict(actual_results):
    """
    Returns a dictionary with all outputs_names as keys.
    Fills with np.nan (or np.inf) by default, then updates with actual_results.
    """
    # Use np.nan or np.inf as default, depending on your convention
    output_dict = {k: np.nan for k in outputs_names}
    output_dict.update(actual_results)
    return output_dict
def make_input_dict(actual_results):
    """
    Returns a dictionary with all inputs_names as keys.
    Fills with np.nan (or np.inf) by default, then updates with actual_results.
    """
    # Use np.nan or np.inf as default, depending on your convention
    output_dict = {k: np.nan for k in inputs_names}
    output_dict.update(actual_results)
    return output_dict

def fix_vector_length(vec, target_length=100):
    vec = np.asarray(vec)
    if vec.ndim == 0 or vec.size == 0:
        # If scalar or empty, fill with scalar value or nan
        scalar = float(vec) if vec.size == 1 or vec.ndim == 0 else np.nan
        return np.full(target_length, scalar)
    if vec.size == 1:
        # Repeat the single value
        return np.full(target_length, vec[0])
    if vec.size == target_length:
        return vec
    # Interpolate to target length, preserving first and last value
    x_old = np.linspace(0, 1, vec.size)
    x_new = np.linspace(0, 1, target_length)
    vec_interp = np.interp(x_new, x_old, vec)
    return vec_interp

