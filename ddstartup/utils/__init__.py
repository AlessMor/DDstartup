"""
Utils package initialization

This module automatically exports all classes, functions, and constants from:
- units_and_constants.py
- custom_classes.py

This allows users to simply import:
    from ddstartup.utils import u, ParameterField
    
Instead of:
    from ddstartup.utils.units_and_constants import u
    from ddstartup.utils.custom_classes import ParameterField
"""

# Import and re-export everything from units_and_constants
from .units_and_constants import *

# Import and re-export everything from custom_classes
from .custom_classes import *

# Import and re-export I/O functions
from .io_functions import (
    resolve_file_path,
    load_config,
    load_parameter_fields,
    prepare_input_data,
    print_configuration,
    create_output_directory,
    generate_output_path,
)

# Import and re-export computation functions
from .parametric_computation import (
    run_parametric_analysis,
    print_parametric_summary,
)

from .sobol_computation import (
    run_sobol_analysis,
    print_sobol_summary,
)

# Optional: Define what gets exported with "from utils import *"
# This makes it explicit and helps with IDE autocomplete
__all__ = [
    # From units_and_constants
    'u',  # Pint unit registry
    # Add other constants/units you want to export
    
    # From custom_classes
    'ParameterField',
    
    # From io_functions
    'resolve_file_path',
    'load_config',
    'load_parameter_fields',
    'prepare_input_data',
    'print_configuration',
    'create_output_directory',
    'generate_output_path',
    
    # From parametric_computation
    'run_parametric_analysis',
    'print_parametric_summary',
    
    # From sobol_computation
    'run_sobol_analysis',
    'print_sobol_summary',
]