import numpy as np
import multiprocessing
import psutil

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
        'P_lost_rad', 'P_lost_rad_all_DT', 'TBR_DT', 'TBR_DDn',
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
        return 16
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
        return 2000
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
        n = 4096
    elif n_cores >= 8:
        n = 2048
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
        return 4
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