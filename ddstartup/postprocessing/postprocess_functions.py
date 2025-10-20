import h5py
import numpy as np
import pandas as pd
from pathlib import Path

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

def load_h5_to_dataframe(h5_path):
    data = {}
    expected_length = None
    with h5py.File(h5_path, 'r') as f:
        # Load all datasets (1D and 2D)
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                arr = f[key][:]
                if arr.ndim == 1:
                    if expected_length is None:
                        expected_length = len(arr)
                    if len(arr) == expected_length:
                        data[key] = arr
                elif arr.ndim == 2 and arr.shape[0] == expected_length:
                    # Store each vector as a list in the DataFrame
                    data[key] = [arr[i, :] for i in range(arr.shape[0])]
        # Load parameter_fields group if present
        if 'parameter_fields' in f:
            param_group = f['parameter_fields']
            for subkey in param_group.keys():
                arr = param_group[subkey][:]
                if arr.ndim == 1 and len(arr) == expected_length:
                    data[subkey] = arr
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
        'V_plasma','n_tot', 'T_i', 'tau_p_T','tau_p_He3','P_aux', 'P_aux_DT_eq', 'tau_ifc', 'tau_ofc', 'TBR_DT', 'TBR_DDn', 'eta_th', 'capacity_factor', 'cost_of_electricity',  'I_target',
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
        REMOVE_PARAMS += ['eta_th', 'capacity_factor', 'cost_of_electricity','P_aux', 'P_aux_DT_eq']
    
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