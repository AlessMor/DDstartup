"""
Utils package initialization

This module automatically exports key functions and constants from:
- units_and_constants.py
- io_functions.py

This allows users to simply import:
    from ddstartup.utils import u, load_config
    
Instead of:
    from ddstartup.utils.units_and_constants import u
    from ddstartup.utils.io_functions import load_config
"""

# Import and re-export everything from units_and_constants
from .units_and_constants import *

# Import and re-export I/O functions
from .io_functions import (
    parse_arguments,
    resolve_file_path,
    load_config,
    load_params,
    prepare_input_data,
    print_configuration,
    generate_output_path,
)

# Optional: Define what gets exported with "from utils import *"
# This makes it explicit and helps with IDE autocomplete
__all__ = [
    # From units_and_constants
    'u',  # Pint unit registry
    # Add other constants/units you want to export
    
    # From io_functions
    'parse_arguments',
    'resolve_file_path',
    'load_config',
    'load_params',
    'prepare_input_data',
    'print_configuration',
    'generate_output_path',
]
