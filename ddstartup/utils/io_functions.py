"""
I/O functions for DD Startup analysis tool.

This module contains functions for:
- File path resolution
- Configuration loading and validation
- Parameter field loading
- Input data preparation
- Configuration display
- Output directory and file creation
"""

import os
import yaml
import importlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time


def resolve_file_path(filename: str, default_dir: str, extensions: Optional[List[str]] = None) -> Path:
    """
    Resolve file path - check if it's a direct path or needs default directory.
    
    Args:
        filename: File name or path
        default_dir: Default directory to search in (e.g., 'inputs')
        extensions: List of extensions to try (e.g., ['.yaml', '.yml'])
    
    Returns:
        Path object to the file
        
    Raises:
        FileNotFoundError: If file cannot be found in any of the expected locations
    """
    if extensions is None:
        extensions = ['']
    
    # Check if filename is already a valid path
    file_path = Path(filename)
    if file_path.exists():
        return file_path
    
    # Remove extension from filename if present
    name_without_ext = filename.replace('.py', '').replace('.yaml', '').replace('.yml', '')
    
    # Try with default directory in multiple locations
    search_paths = []
    for ext in extensions:
        # Try in current working directory's default_dir
        test_path = Path(default_dir) / f"{name_without_ext}{ext}"
        search_paths.append(str(test_path))
        if test_path.exists():
            return test_path
        
        # Try in parent directory's default_dir (for when running from ddstartup/)
        parent_test_path = Path('..') / default_dir / f"{name_without_ext}{ext}"
        search_paths.append(str(parent_test_path))
        if parent_test_path.exists():
            return parent_test_path.resolve()
    
    # If not found, raise error with helpful message
    raise FileNotFoundError(
        f"File not found: {filename}\n"
        f"Searched in: {filename}, {', '.join(search_paths)}"
    )


def load_config(yaml_path: Path) -> Dict[str, Any]:
    """
    Load and validate YAML configuration.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration with defaults applied
        
    Raises:
        ValueError: If required fields are missing
        yaml.YAMLError: If YAML file is malformed
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['analysis_type', 'method']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Set defaults for optional fields
    config.setdefault('vector_length', 100)
    config.setdefault('max_simulation_time', 10 * 365 * 24 * 3600)
    config.setdefault('verbose', False)
    config.setdefault('output_dir', 'outputs')
    config.setdefault('n_jobs', None)
    config.setdefault('chunk_size', None)
    config.setdefault('batch_size', 500)
    config.setdefault('N_SAMPLES', 100000)
    config.setdefault('order', 3)
    config.setdefault('filter', None)  # Parameter filter expression
    
    return config


def load_parameter_fields(param_module_path: Path) -> Dict[str, Any]:
    """
    Load parameter fields from YAML configuration file.
    
    Args:
        param_module_path: Path to YAML parameter configuration file
        
    Returns:
        Dictionary mapping field names to tuples of (values_array, unit_string, metadata)
        or scalar values for simple parameters
        
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If file format is unsupported or YAML structure is invalid
    """
    param_path = Path(param_module_path)
    
    # Only YAML files are supported
    if param_path.suffix not in ['.yaml', '.yml']:
        raise ValueError(
            f"Only YAML parameter files are supported (.yaml or .yml), got: {param_path.suffix}\n"
            f"Legacy Python parameter files (.py) are no longer supported.\n"
            f"Please convert to YAML format. See inputs/README_YAML.md for migration guide."
        )
    
    from .parameter_loader import ParameterLoader
    return ParameterLoader.load_from_yaml(param_path)


def prepare_input_data(param_fields: Dict[str, Any], analysis_type: str) -> Dict[str, np.ndarray]:
    """
    Prepare input data dictionary based on analysis type.
    
    Converts parameter field tuples (values, unit, metadata) to numpy arrays with proper units.
    
    Args:
        param_fields: Dictionary of parameter field tuples from YAML loader
        analysis_type: Type of analysis ('T_seeded' or 'lump')
        
    Returns:
        Dictionary mapping parameter names to numpy arrays in correct units
        
    Raises:
        ValueError: If analysis_type is not recognized
    """
    from .units_and_constants import u
    
    def convert_to_unit(field_data: tuple, target_unit: str) -> np.ndarray:
        """Convert parameter field data to target unit."""
        values, unit_str, metadata = field_data
        quantity = values * u(unit_str)
        return quantity.to(target_unit).magnitude
    
    if analysis_type == 'T_seeded':
        input_data = {
            'V_plasma': convert_to_unit(param_fields['V_plasma_field'], 'm^3'),
            'T_i': convert_to_unit(param_fields['T_i_field'], 'keV'),
            'n_tot': convert_to_unit(param_fields['n_tot_field'], '1/m^3'),
            'tau_p_T': convert_to_unit(param_fields['tau_p_T_field'], 's'),
            'P_aux': convert_to_unit(param_fields['P_aux_field'], 'W'),
            'P_aux_DT_eq': convert_to_unit(param_fields['P_aux_DT_eq_field'], 'W'),
            'TBR_DT': convert_to_unit(param_fields['TBR_DT_field'], 'dimensionless'),
            'TBR_DDn': convert_to_unit(param_fields['TBR_DDn_field'], 'dimensionless'),
            'tau_ifc': convert_to_unit(param_fields['tau_ifc_field'], 's'),
            'tau_ofc': convert_to_unit(param_fields['tau_ofc_field'], 's'),
            'eta_th': convert_to_unit(param_fields['eta_th_field'], 'dimensionless'),
            'capacity_factor': convert_to_unit(param_fields['capacity_factor_field'], 'dimensionless'),
            'price_of_electricity': convert_to_unit(param_fields['price_of_electricity_field'], '1/J')
        }
    elif analysis_type == 'lump':
        input_data = {
            'V_plasma': convert_to_unit(param_fields['V_plasma_field'], 'm^3'),
            'T_i': convert_to_unit(param_fields['T_i_field'], 'keV'),
            'n_tot': convert_to_unit(param_fields['n_tot_field'], '1/m^3'),
            'tau_p_T': convert_to_unit(param_fields['tau_p_T_field'], 's'),
            'tau_p_He3': convert_to_unit(param_fields['tau_p_He3_field'], 's'),
            'P_aux': convert_to_unit(param_fields['P_aux_field'], 'W'),
            'P_aux_DT_eq': convert_to_unit(param_fields['P_aux_DT_eq_field'], 'W'),
            'TBR_DT': convert_to_unit(param_fields['TBR_DT_field'], 'dimensionless'),
            'TBR_DDn': convert_to_unit(param_fields['TBR_DDn_field'], 'dimensionless'),
            'I_target': convert_to_unit(param_fields['I_target_field'], 'kg'),
            'eta_th': convert_to_unit(param_fields['eta_th_field'], 'dimensionless'),
            'capacity_factor': convert_to_unit(param_fields['capacity_factor_field'], 'dimensionless'),
            'price_of_electricity': convert_to_unit(param_fields['price_of_electricity_field'], '1/J')
        }
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    return input_data


def print_configuration(
    config: Dict[str, Any],
    param_fields: Dict[str, Any],
    input_data: Dict[str, np.ndarray],
    param_file: Path,
    config_file: Path
) -> None:
    """
    Print configuration summary.
    
    Args:
        config: Configuration dictionary
        param_fields: Parameter fields dictionary
        input_data: Prepared input data dictionary
        param_file: Path to parameter config file
        config_file: Path to YAML config file
    """
    print("\n" + "="*60)
    print("DD STARTUP ANALYSIS CONFIGURATION")
    print("="*60)
    print(f"Parameter file: {param_file}")
    print(f"Config file: {config_file}")
    print(f"Analysis type: {config['analysis_type']}")
    print(f"Method: {config['method']}")
    print(f"Max simulation time: {config['max_simulation_time']/365/24/3600:.2f} years")
    
    if config.get('filter'):
        print(f"Filter: {config['filter']}")
    
    if config['method'] == 'sobol':
        print(f"N_SAMPLES: {config['N_SAMPLES']}")
        print(f"Order: {config['order']}")
    else:
        print(f"Vector length: {config['vector_length']}")
    
    print(f"n_jobs: {config['n_jobs'] or 'auto'}")
    print(f"chunk_size: {config['chunk_size'] or 'auto'}")
    print(f"batch_size: {config['batch_size']}")
    print(f"Output directory: {config['output_dir']}")
    
    print("\nInput parameter fields:")
    for name, arr in input_data.items():
        print(f"  {name:20s}: shape={arr.shape}, range=[{arr.min():.3e}, {arr.max():.3e}]")
    
    param_shapes = [arr.shape[0] for arr in input_data.values()]
    n_combinations = np.prod(param_shapes)
    print(f"\nTotal parameter combinations: {n_combinations:,}")
    print("="*60 + "\n")


def create_output_directory(base_dir: str, timestamp: str, analysis_method: str, analysis_type: str) -> Path:
    """
    Create output directory with timestamp and analysis info.
    
    Creates a directory structure: base_dir/timestamp_method_type/
    
    Args:
        base_dir: Base output directory (e.g., 'outputs')
        timestamp: Timestamp string (e.g., '20251006_123045')
        analysis_method: Analysis method ('parametric', 'sobol', 'lhs')
        analysis_type: Analysis type ('T_seeded', 'lump')
        
    Returns:
        Path object to the created directory
        
    Raises:
        OSError: If directory cannot be created
        
    Example:
        >>> output_dir = create_output_directory('outputs', '20251006_123045', 'parametric', 'T_seeded')
        >>> print(output_dir)
        outputs/20251006_123045_parametric_T_seeded
    """
    # If base_dir is 'outputs' (relative), make it relative to parent directory
    # This ensures outputs go to dd_startup/outputs instead of dd_startup/ddstartup/outputs
    if base_dir == 'outputs':
        # Get the parent directory of the current script location
        script_dir = Path(__file__).resolve().parent.parent  # Go up from utils/ to ddstartup/
        base_dir = script_dir.parent / 'outputs'  # Go up from ddstartup/ to dd_startup/ and add outputs/
    
    # Create directory name with timestamp and analysis info
    dir_name = f"{timestamp}_{analysis_method}_{analysis_type}"
    output_path = Path(base_dir) / dir_name
    
    # Create directory (including parent directories if needed)
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path


def generate_output_path(
    base_dir: str = 'outputs',
    analysis_method: str = 'parametric',
    analysis_type: str = 'T_seeded',
    timestamp: Optional[str] = None
) -> Tuple[Path, str]:
    """
    Generate output directory and file path for HDF5 results.
    
    Creates directory structure and generates full path to output file:
    base_dir/timestamp_method_type/ddstartup_timestamp_method_type.h5
    
    Args:
        base_dir: Base output directory (default: 'outputs')
        analysis_method: Analysis method ('parametric', 'sobol', 'lhs')
        analysis_type: Analysis type ('T_seeded', 'lump')
        timestamp: Optional timestamp string. If None, generates current timestamp
        
    Returns:
        Tuple of (output_directory_path, full_output_file_path)
        
    Example:
        >>> output_dir, output_file = generate_output_path('outputs', 'parametric', 'T_seeded')
        >>> print(output_dir)
        outputs/20251006_123045_parametric_T_seeded
        >>> print(output_file)
        outputs/20251006_123045_parametric_T_seeded/ddstartup_20251006_123045_parametric_T_seeded.h5
    """
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = create_output_directory(base_dir, timestamp, analysis_method, analysis_type)
    
    # Generate output filename
    filename = f"ddstartup_{timestamp}_{analysis_method}_{analysis_type}.h5"
    output_file = output_dir / filename
    
    return output_dir, str(output_file)
