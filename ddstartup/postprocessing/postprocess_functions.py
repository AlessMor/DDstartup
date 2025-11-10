import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import hdf5plugin to enable LZ4 compression support
# This MUST be imported before opening any HDF5 files with LZ4 compression
try:
    import hdf5plugin
except ImportError:
    pass  # Will fall back to standard compression formats


# ============================================================================
# FILE AND DIRECTORY UTILITIES
# ============================================================================

def find_latest_output_folder(outputs_dir):
    """
    Find the most recent output folder in the outputs directory.
    Prioritizes folders with timestamp prefixes (YYYYMMDD_HHMMSS format),
    then falls back to filesystem creation time.
    
    Args:
        outputs_dir: Path to outputs directory
        
    Returns:
        Tuple of (folder_path, h5_files) or (None, None) if not found
    """
    import re
    
    # Find all directories (excluding hidden dirs)
    folders = [d for d in outputs_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not folders:
        return None, None
    
    # Sort by timestamp in folder name if present, otherwise by creation time
    def sort_key(path):
        # Try to extract timestamp from folder name (YYYYMMDD_HHMMSS format)
        match = re.match(r'(\d{8})_(\d{6})', path.name)
        if match:
            # Return timestamp as sortable string (YYYYMMDDHHMMSS)
            return (1, match.group(1) + match.group(2))  # Priority 1 (highest)
        else:
            # Fall back to filesystem creation time for folders without timestamp
            stat = path.stat()
            # Use st_birthtime if available (macOS), otherwise st_mtime (best proxy on Linux)
            ctime = getattr(stat, 'st_birthtime', stat.st_mtime)
            return (0, ctime)  # Priority 0 (lower than named folders)
    
    folders.sort(key=sort_key, reverse=True)
    latest_folder = folders[0]
    
    # Find HDF5 files in the latest folder
    h5_files = list(latest_folder.glob("*.h5"))
    
    if not h5_files:
        return latest_folder, None
    
    return latest_folder, h5_files


def find_latest_h5_file(outputs_dir):
    """
    Find the most recent HDF5 file in the outputs directory.
    Falls back to searching root outputs/ if no folders found.
    
    Args:
        outputs_dir: Path to outputs directory
        
    Returns:
        Path to the latest HDF5 file, or None if not found
    """
    # First try to find latest folder
    latest_folder, h5_files = find_latest_output_folder(outputs_dir)
    
    if h5_files:
        return h5_files[0]  # Return first (and likely only) HDF5 file in folder
    
    # Fallback: search root outputs directory
    h5_files = list(outputs_dir.glob("*.h5"))
    if not h5_files:
        return None
    # Sort by modification time, most recent first
    h5_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return h5_files[0]


def load_yaml_config(config_path):
    """
    Load postprocessing configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary with configuration settings
    """
    import sys
    try:
        import yaml
    except ImportError:
        print("❌ Error: PyYAML is not installed. Install it with: pip install pyyaml")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


# ============================================================================
# FILTER UTILITIES
# ============================================================================

def parse_filter_expression(filter_str):
    """
    Parse a filter expression like "V_plasma<150" or "t_startup>1e6,unrealized_profits<2e9".
    
    Args:
        filter_str: String with filter expressions separated by commas
        
    Returns:
        Dictionary with filter conditions: {variable: {'min': val, 'max': val}}
    """
    if not filter_str:
        return {}
    
    filters = {}
    expressions = filter_str.split(',')
    
    for expr in expressions:
        expr = expr.strip()
        if not expr:
            continue
            
        # Parse operator and value
        if '<=' in expr:
            var, val = expr.split('<=')
            var = var.strip()
            val = float(val.strip())
            if var not in filters:
                filters[var] = {'min': None, 'max': None}
            filters[var]['max'] = val
        elif '>=' in expr:
            var, val = expr.split('>=')
            var = var.strip()
            val = float(val.strip())
            if var not in filters:
                filters[var] = {'min': None, 'max': None}
            filters[var]['min'] = val
        elif '<' in expr:
            var, val = expr.split('<')
            var = var.strip()
            val = float(val.strip())
            if var not in filters:
                filters[var] = {'min': None, 'max': None}
            filters[var]['max'] = val
        elif '>' in expr:
            var, val = expr.split('>')
            var = var.strip()
            val = float(val.strip())
            if var not in filters:
                filters[var] = {'min': None, 'max': None}
            filters[var]['min'] = val
        elif '==' in expr:
            var, val = expr.split('==')
            var = var.strip()
            val = float(val.strip())
            if var not in filters:
                filters[var] = {'min': None, 'max': None}
            filters[var]['min'] = val
            filters[var]['max'] = val
        else:
            print(f"⚠️  Warning: Could not parse filter expression: {expr}")
            continue
    
    return filters


def clean_filters(filters_dict):
    """
    Remove entries with both min and max as None from filters dictionary.
    
    Args:
        filters_dict: Dictionary of filters from YAML
        
    Returns:
        Cleaned dictionary with only active filters
    """
    if not filters_dict:
        return {}
    
    cleaned = {}
    for var, conditions in filters_dict.items():
        if conditions is None:
            continue
        if isinstance(conditions, dict):
            min_val = conditions.get('min')
            max_val = conditions.get('max')
            if min_val is not None or max_val is not None:
                cleaned[var] = {'min': min_val, 'max': max_val}
    
    return cleaned


def apply_filters(df, input_filters, output_filters, target_variable):
    """
    Apply both input and output filters to a dataframe.
    
    Args:
        df: DataFrame to filter
        input_filters: Dictionary of filters for input parameters
        output_filters: Dictionary of filters for output variables
        target_variable: Name of target output variable (always kept finite)
        
    Returns:
        Filtered DataFrame
    """
    # Start with finite target values
    mask = np.isfinite(df[target_variable])
    
    # Apply input filters
    for var, conditions in input_filters.items():
        if var not in df.columns:
            print(f"⚠️  Warning: Input variable '{var}' not found in data")
            continue
        if conditions.get('min') is not None:
            mask &= df[var] >= conditions['min']
        if conditions.get('max') is not None:
            mask &= df[var] <= conditions['max']
    
    # Apply output filters
    for var, conditions in output_filters.items():
        if var not in df.columns:
            print(f"⚠️  Warning: Output variable '{var}' not found in data")
            continue
        mask &= np.isfinite(df[var])
        if conditions.get('min') is not None:
            mask &= df[var] >= conditions['min']
        if conditions.get('max') is not None:
            mask &= df[var] <= conditions['max']
    
    filtered = df[mask].copy()
    print(f"   Filtered: {len(df)} → {len(filtered)} rows ({len(filtered)/len(df)*100:.1f}%)")
    return filtered


# ============================================================================
# HDF5 DATA LOADING
# ============================================================================

def load_h5_to_dataframe(h5_path, chunk_size=500000, target_variables=None):
    """
    Load HDF5 data efficiently in chunks to manage memory for large datasets.
    
    Only loads input parameters and specified target variables, not all outputs.
    This dramatically reduces memory usage for large datasets.
    
    For datasets with >1M rows, loads data in chunks and builds the DataFrame incrementally,
    reducing peak memory usage by ~50%.
    
    IMPORTANT: For parametric analyses, if input parameter datasets contain NaN values,
    this function will reconstruct them from linear_index and parameter_fields group.
    
    Args:
        h5_path: Path to HDF5 file
        chunk_size: Number of rows to load per chunk (default: 500k)
        target_variables: List of target variables to load. If None, loads all datasets.
    
    Returns:
        pandas DataFrame with input parameters and target variables
    """
    # Try to import hdf5plugin for LZ4 compression support (optional)
    try:
        import hdf5plugin
    except ImportError:
        pass  # Will still work with gzip compression
    
    # Define input parameter names
    INPUT_PARAMS = ['V_plasma', 'n_tot', 'T_i', 'tau_p_T', 'tau_p_He3', 'P_aux', 'P_aux_DT_eq', 
                    'tau_ifc', 'tau_ofc', 'TBR_DT', 'TBR_DDn', 'eta_th', 'capacity_factor', 
                    'price_of_electricity', 'I_target']
    
    data = {}
    expected_length = None
    
    with h5py.File(h5_path, 'r') as f:
        # First pass: determine dataset length and identify keys to load
        dataset_keys = []
        param_keys = []
        
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                if expected_length is None and f[key].ndim >= 1:
                    expected_length = f[key].shape[0]
                
                # Only load inputs and specified targets
                is_input = key in INPUT_PARAMS
                is_target = target_variables is None or key in target_variables
                is_success = key == 'sol_success'  # Always load success flag
                
                if is_input or is_target or is_success:
                    dataset_keys.append((key, f[key].ndim, f[key].shape))
        
        if 'parameter_fields' in f:
            # Check which parameter_fields are actual data (match expected_length)
            for subkey in f['parameter_fields'].keys():
                arr_shape = f['parameter_fields'][subkey].shape
                if arr_shape[0] == expected_length:
                    param_keys.append(subkey)
        
        # If dataset is small (<1M rows), load all at once (old behavior)
        if expected_length <= 1_000_000:
            for key, ndim, shape in dataset_keys:
                arr = f[key][:]
                if ndim == 1 and len(arr) == expected_length:
                    data[key] = arr
                elif ndim == 2 and shape[0] == expected_length:
                    data[key] = [arr[i, :] for i in range(shape[0])]
            
            for subkey in param_keys:
                arr = f['parameter_fields'][subkey][:]
                if arr.ndim == 1 and len(arr) == expected_length:
                    data[subkey] = arr
        
        # For large datasets, load in chunks
        else:
            print(f"   Large dataset detected ({expected_length:,} rows), loading in chunks of {chunk_size:,}...")
            n_chunks = int(np.ceil(expected_length / chunk_size))
            
            # Allocate memory for datasets
            for key, ndim, shape in dataset_keys:
                data[key] = [] if ndim == 2 else np.empty(expected_length, dtype=f[key].dtype)
            
            # Allocate memory only for parameter_fields that match expected_length
            for subkey in param_keys:
                data[subkey] = np.empty(expected_length, dtype=f['parameter_fields'][subkey].dtype)
            
            # Load data in chunks with progress bar
            for chunk_idx in tqdm(range(n_chunks), desc="   Loading chunks", unit="chunk"):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, expected_length)
                
                for key, ndim, shape in dataset_keys:
                    if ndim == 1:
                        data[key][start_idx:end_idx] = f[key][start_idx:end_idx]
                    elif ndim == 2:
                        chunk_arr = f[key][start_idx:end_idx]
                        data[key].extend([chunk_arr[i, :] for i in range(len(chunk_arr))])
                
                for subkey in param_keys:
                    data[subkey][start_idx:end_idx] = f['parameter_fields'][subkey][start_idx:end_idx]
        
        # ========== RECONSTRUCT INPUT PARAMETERS FROM LINEAR INDEX (FIX FOR NaN BUG) ==========
        # Check if input parameters are all NaN (bug in older parametric analysis runs)
        # If so, reconstruct them from linear_index and parameter_fields
        if 'linear_index' in data and 'parameter_fields' in f:
            need_reconstruction = False
            for param in INPUT_PARAMS:
                if param in data:
                    sample_vals = data[param][:min(100, len(data[param]))]
                    if np.all(np.isnan(sample_vals)):
                        need_reconstruction = True
                        break
            
            if need_reconstruction:
                print("   ⚠️  Input parameters contain NaN - reconstructing from linear_index...")
                
                # Get parameter shapes from HDF5 metadata
                param_shapes = f.attrs.get('parameter_shapes', None)
                if param_shapes is None:
                    print("   ❌ Cannot reconstruct: parameter_shapes attribute missing")
                else:
                    param_shapes = np.array(param_shapes, dtype=np.int64)
                    
                    # Load parameter grid values from parameter_fields group
                    param_field_names = []
                    param_grids = []
                    for key in sorted(f['parameter_fields'].keys()):
                        if key.endswith('_values'):
                            param_name = key.replace('_values', '')
                            param_field_names.append(param_name)
                            param_grids.append(f['parameter_fields'][key][:])
                    
                    # Reconstruct each input parameter using linear_index
                    linear_indices = data['linear_index']
                    for i, param_name in enumerate(param_field_names):
                        if param_name in INPUT_PARAMS and param_name in data:
                            # Convert linear indices to multi-dimensional indices
                            # Using same logic as in index_to_params
                            param_values = np.empty(len(linear_indices))
                            for j, lin_idx in enumerate(linear_indices):
                                # Compute multi-index for this parameter
                                idx = int(lin_idx)
                                temp_idx = np.zeros(len(param_shapes), dtype=np.int64)
                                for k in range(len(param_shapes) - 1, -1, -1):
                                    temp_idx[k] = idx % param_shapes[k]
                                    idx //= param_shapes[k]
                                # Get value from parameter grid
                                param_values[j] = param_grids[i][temp_idx[i]]
                            
                            data[param_name] = param_values
                            print(f"      ✓ Reconstructed {param_name}: {np.unique(param_values)[:5]} ...")
                
                print("   ✓ Reconstruction complete")
    
    return pd.DataFrame(data)

def filter_finite(df, target_variable, filter_dict=None):
    mask = np.isfinite(df[target_variable])
    if filter_dict is not None:
        if filter_dict.get('min') is not None:
            mask &= df[target_variable] >= filter_dict['min']
        if filter_dict.get('max') is not None:
            mask &= df[target_variable] <= filter_dict['max']
    return df[mask].copy()

def get_input_parameters(df, target_variable, filename=None):
    DESIRED_ORDER = [
        'V_plasma','n_tot', 'T_i', 'tau_p_T','tau_p_He3','P_aux', 'P_aux_DT_eq', 'tau_ifc', 'tau_ofc', 'TBR_DT', 'TBR_DDn', 'eta_th', 'capacity_factor', 'price_of_electricity',  'I_target',
    ]
    base_inputs = [col for col in df.columns if col in DESIRED_ORDER and col != target_variable]
    input_parameters = [p for p in DESIRED_ORDER if p in base_inputs] + [p for p in base_inputs if p not in DESIRED_ORDER]
    REMOVE_PARAMS = []
    if filename:
        if 'T_seeded' in filename:
            REMOVE_PARAMS = ['tau_p_He3', 'I_target']
        elif 'lump' in filename:
            REMOVE_PARAMS = ['tau_ifc', 'tau_ofc']
    if target_variable == 't_startup':
        REMOVE_PARAMS += ['eta_th', 'capacity_factor', 'price_of_electricity','P_aux', 'P_aux_DT_eq']
    
    # Filter out constant parameters (zero variance)
    # This automatically excludes parameters with a single value across all samples
    varying_params = []
    for param in input_parameters:
        if param not in REMOVE_PARAMS and param in df.columns:
            # Check if parameter varies (std > threshold to account for floating point errors)
            if df[param].std() > 1e-10:
                varying_params.append(param)
    
    return varying_params

def scale_target(df, target_variable):
    if target_variable == 'unrealized_profits':
        df[target_variable] = df[target_variable] / 1e6
        return df, 'M$'
    if target_variable != 't_startup':
        return df, 's'
    max_val = df[target_variable].max()
    scale = 1
    unit = "s"
    if max_val > 30*24*3600:
        scale = 1/(24*3600)
        unit = "days"
    elif max_val > 24*3600:
        scale = 1/3600
        unit = "hours"
    df[target_variable] = df[target_variable] * scale
    return df, unit

def get_discrete_colorscale(n_chunks):
    """
    Returns a Plotly-compatible discrete colorscale from blue to orange/red.
    
    This colorblind-friendly palette works for ~99% of people, including those
    with red-green colorblindness (deuteranopia/protanopia).
    
    Color progression: Blue (good) → Teal → Yellow → Orange → Red (bad)
    
    Logic:
    - If n_chunks <= 7: Uses base colors with interpolation for smooth transitions
    - If n_chunks > 7: Uses matplotlib's RdYlBu_r colormap for more colors
    
    Args:
        n_chunks: Number of discrete color bins
        
    Returns:
        List of [fraction, color] pairs for Plotly colorscale
    """
    # Colorblind-friendly blue to orange/red palette
    # Based on ColorBrewer's RdYlBu reversed, optimized for accessibility
    base_colors = [
        "#2166AC",  # dark blue (best/lowest)
        "#4393C3",  # medium blue
        "#92C5DE",  # light blue
        "#FFFFBF",  # pale yellow (neutral)
        "#FDAE61",  # light orange
        "#F46D43",  # orange
        "#D73027",  # red (worst/highest)
    ]
    
    import matplotlib.colors as mcolors
    
    if n_chunks > len(base_colors):
        # Case 1: Need MORE colors than base palette
        # Use matplotlib's RdYlBu_r (Red-Yellow-Blue reversed) colormap
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap('RdYlBu_r')
        color_values = cmap(np.linspace(0, 1, n_chunks))
        color_list = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' 
                     for r, g, b, _ in color_values]
    else:
        # Case 2: Need FEWER or EQUAL colors than base palette
        # Interpolate between base colors for smooth gradients
        base_rgb = [mcolors.to_rgb(c) for c in base_colors]
        positions = np.linspace(0, 1, len(base_rgb))
        interp_positions = np.linspace(0, 1, n_chunks)
        
        # Interpolate R, G, B channels separately
        interp_rgb = np.array([
            np.interp(interp_positions, positions, [rgb[i] for rgb in base_rgb]) 
            for i in range(3)
        ]).T
        
        color_list = [mcolors.to_hex(rgb) for rgb in interp_rgb]
    
    # Build Plotly colorscale: each color spans its interval
    # Format: [[0.0, color0], [0.25, color0], [0.25, color1], [0.5, color1], ...]
    colorscale = []
    for i, color in enumerate(color_list):
        frac_start = i / n_chunks
        frac_end = (i + 1) / n_chunks
        colorscale.append([frac_start, color])
        colorscale.append([frac_end, color])
    
    return colorscale

def prepare_dataframes(selected_files, target_variables, filters):
    """
    Load and filter dataframes for each selected file and target variable.
    Returns a nested dict: {filename: {target: filtered_df}}
    """
    dataframes = {}
    for selected_file in selected_files:
        file_data = {}
        df = load_h5_to_dataframe(selected_file)
        for target in target_variables:
            filter_dict = filters.get(target, {})
            df_filtered = filter_finite(df, target, filter_dict)
            file_data[target] = df_filtered
        dataframes[selected_file] = file_data
    return dataframes