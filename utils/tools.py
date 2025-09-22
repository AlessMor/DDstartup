import numpy as np
import multiprocessing

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