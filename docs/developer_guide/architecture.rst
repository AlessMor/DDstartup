System Architecture
===================

This document describes the architecture and design patterns of the DD Startup Analysis Tool, focusing on the package structure, import system, and module organization.

Overview
--------

The DD Startup Analysis Tool follows a modular Python package architecture with:

* **Clear separation of concerns** - Each module has a specific purpose
* **Automatic imports** - Simplified import system through package ``__init__.py``
* **Type safety** - Type hints throughout the codebase
* **Testability** - Designed for easy unit testing

Package Structure
-----------------

.. code-block:: text

   dd_startup/
   ├── __init__.py                    # Makes dd_startup a package
   ├── ddstartup/                     # Main package directory
   │   ├── __init__.py                # Package initialization
   │   ├── main.py                    # Application entry point
   │   ├── utils/                     # Utilities package
   │   │   ├── __init__.py            # Auto-export utilities
   │   │   ├── units_and_constants.py # Pint units, constants
   │   │   ├── custom_classes.py      # ParameterField class
   │   │   ├── io_functions.py        # I/O operations
   │   │   └── system_profiler.py     # Hardware profiling
   │   ├── inputs/                    # Configuration files
   │   │   ├── __init__.py            # Path setup
   │   │   ├── config_test.py         # Test parameters
   │   │   └── *.yaml                 # Configuration files
   │   ├── outputs/                   # Analysis results
   │   └── tests/                     # Test suite
   │       ├── conftest.py            # Pytest fixtures
   │       ├── test_io_functions.py   # I/O tests
   │       ├── test_main.py           # Integration tests
   │       └── test_system_profiler.py # Profiler tests
   └── docs/                          # Sphinx documentation

Module Responsibilities
-----------------------

Core Modules
~~~~~~~~~~~~

**main.py** - Application Orchestration
  
  * Command-line interface
  * Workflow coordination
  * Configuration management
  * Exit handling

  .. code-block:: python
  
     def main() -> int:
         """Main entry point for DD startup analysis"""
         args = parse_arguments()
         config = load_config(args.config)
         # ... orchestrate workflow
         return 0

**utils/__init__.py** - Package Exports

  * Auto-exports utilities
  * Simplifies imports
  * Defines public API

  .. code-block:: python
  
     from .units_and_constants import *
     from .custom_classes import *
     from .io_functions import *
     from .system_profiler import *
     
     __all__ = ['u', 'ParameterField', 'load_config', ...]

Utility Modules
~~~~~~~~~~~~~~~

**utils/units_and_constants.py** - Physical Units

  * Pint unit registry (``u``)
  * Physical constants
  * Unit conversions

  .. code-block:: python
  
     import pint
     u = pint.UnitRegistry()
     
     # Usage
     energy = 17.0 * u.keV
     volume = 150.0 * u.m**3

**utils/custom_classes.py** - Data Structures

  * ``ParameterField`` class
  * Parameter metadata
  * Data validation

  .. code-block:: python
  
     class ParameterField:
         """Stores parameter data with units and metadata"""
         def __init__(self, unit, name, parametrization_type, ...):
             self.unit = unit
             self.name = name
             # ...

**utils/io_functions.py** - Input/Output Operations

  * File path resolution
  * YAML configuration loading
  * Parameter module loading
  * Data extraction with unit conversion
  * Configuration display

  Key functions:
  
  * ``resolve_file_path()`` - Smart file resolution
  * ``load_config()`` - YAML loading with validation
  * ``load_parameter_fields()`` - Module importing
  * ``prepare_input_data()`` - Data extraction
  * ``print_configuration()`` - Display utility

**utils/system_profiler.py** - Hardware Profiling

  * CPU/RAM detection
  * Optimal parameter calculation
  * User override support

  Key functions:
  
  * ``get_system_info()`` - Hardware information
  * ``get_optimal_parameters()`` - Calculate recommendations
  * ``print_system_profile()`` - Display profile

Import System
-------------

The import system is designed for simplicity and flexibility.

Package Initialization
~~~~~~~~~~~~~~~~~~~~~~

**utils/__init__.py** Configuration:

.. code-block:: python

   """
   Utils package initialization - exports all utilities
   """
   
   # Import and re-export everything
   from .units_and_constants import *
   from .custom_classes import *
   from .io_functions import *
   from .system_profiler import *
   
   # Define explicit exports
   __all__ = [
       # Units and constants
       'u',
       
       # Classes
       'ParameterField',
       
       # I/O functions
       'resolve_file_path',
       'load_config',
       'load_parameter_fields',
       'prepare_input_data',
       'print_configuration',
       
       # System profiler
       'get_system_info',
       'calculate_optimal_n_jobs',
       'calculate_optimal_chunk_size',
       'calculate_optimal_batch_size',
       'calculate_optimal_sobol_samples',
       'calculate_optimal_sobol_order',
       'get_optimal_parameters',
       'override_with_config',
       'print_system_profile',
   ]

Import Patterns
~~~~~~~~~~~~~~~

Three import styles are supported:

**1. Wildcard Import (Simplest)**

.. code-block:: python

   from ddstartup.utils import *
   
   # Now use directly
   field = ParameterField(unit=u.keV, ...)

Pros: Clean, minimal code
Cons: Less explicit, potential namespace pollution

**2. Explicit Import (Recommended)**

.. code-block:: python

   from ddstartup.utils import u, ParameterField
   
   # Explicit about dependencies
   field = ParameterField(unit=u.keV, ...)

Pros: Clear dependencies, IDE-friendly
Cons: Slightly more verbose

**3. Module Import (Most Explicit)**

.. code-block:: python

   from ddstartup import utils
   
   # Fully qualified names
   field = utils.ParameterField(unit=utils.u.keV, ...)

Pros: No namespace pollution, very explicit
Cons: Most verbose

Usage Examples
~~~~~~~~~~~~~~

**In configuration files (inputs/config.py):**

.. code-block:: python

   # Recommended: Explicit imports
   from ddstartup.utils import u, ParameterField
   import numpy as np
   
   # Define parameter fields
   V_plasma_field = ParameterField(
       unit=u.m**3,
       name="plasma_volume",
       parametrization_type="normal",
       mean=150.0,
       param_points=1
   )
   
   T_i_field = ParameterField(
       unit=u.keV,
       name="ion_temperature",
       parametrization_type="normal",
       mean=17.0,
       param_points=1
   )

**In application code (main.py):**

.. code-block:: python

   # Explicit imports for clarity
   from utils.io_functions import (
       resolve_file_path,
       load_config,
       load_parameter_fields,
       prepare_input_data,
       print_configuration
   )
   from utils.system_profiler import (
       get_optimal_parameters,
       override_with_config,
       print_system_profile
   )

Best Practices
--------------

Import Guidelines
~~~~~~~~~~~~~~~~~

✅ **DO:**

* Use explicit imports in library code:

  .. code-block:: python
  
     from ddstartup.utils import u, ParameterField

* Use wildcard imports in convenience scripts/config files:

  .. code-block:: python
  
     from ddstartup.utils import *

* Keep ``__all__`` updated when adding exports:

  .. code-block:: python
  
     __all__ = ['u', 'ParameterField', 'new_function']

* Use type hints for all functions:

  .. code-block:: python
  
     def load_config(yaml_path: Path) -> Dict[str, Any]:

⚠️ **AVOID:**

* Mixing import styles unnecessarily:

  .. code-block:: python
  
     from ddstartup.utils import u
     from ddstartup.utils.units_and_constants import u  # Redundant!

* Importing from private modules directly:

  .. code-block:: python
  
     from ddstartup.utils._internal import something  # Bad!

* Circular imports between modules

Code Organization
~~~~~~~~~~~~~~~~~

**1. Function Length**

* Keep functions under 50 lines
* Extract complex logic into helper functions
* Use clear, descriptive names

**2. Module Size**

* Target 200-400 lines per module
* Split larger modules by functionality
* Keep related functions together

**3. Type Hints**

* Always include type hints:

  .. code-block:: python
  
     def resolve_file_path(
         filename: str,
         default_dir: str,
         extensions: List[str]
     ) -> Path:

**4. Documentation**

* Docstrings for all public functions:

  .. code-block:: python
  
     def load_config(yaml_path: Path) -> Dict[str, Any]:
         """
         Load and validate YAML configuration file.
         
         Parameters
         ----------
         yaml_path : Path
             Path to YAML file
         
         Returns
         -------
         Dict[str, Any]
             Configuration dictionary
         
         Raises
         ------
         FileNotFoundError
             If file doesn't exist
         ValueError
             If required fields missing
         """

Adding New Modules
------------------

To add a new utility module:

**Step 1: Create Module File**

.. code-block:: python

   # utils/new_module.py
   from typing import Any
   
   def new_function(param: str) -> str:
       """
       Brief description of function.
       
       Parameters
       ----------
       param : str
           Description
       
       Returns
       -------
       str
           Description
       """
       return param.upper()
   
   __all__ = ['new_function']

**Step 2: Update utils/__init__.py**

.. code-block:: python

   # utils/__init__.py
   from .units_and_constants import *
   from .custom_classes import *
   from .io_functions import *
   from .system_profiler import *
   from .new_module import *  # Add new import
   
   __all__ = [
       'u',
       'ParameterField',
       # ... existing exports
       'new_function',  # Add new export
   ]

**Step 3: Write Tests**

.. code-block:: python

   # tests/test_new_module.py
   from ddstartup.utils.new_module import new_function
   
   def test_new_function():
       result = new_function("hello")
       assert result == "HELLO"

**Step 4: Document Function**

Add to appropriate API reference document in ``docs/api_reference/``.

Dependency Management
---------------------

Core Dependencies
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Package
     - Purpose
     - Usage
   * - pint
     - Unit conversions
     - ``u = pint.UnitRegistry()``
   * - pyyaml
     - Config files
     - ``yaml.safe_load()``
   * - numpy
     - Array operations
     - ``np.array()``
   * - psutil
     - System profiling
     - ``psutil.cpu_count()``
   * - pytest
     - Testing
     - ``pytest tests/``

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For documentation:

* sphinx >= 7.0.0
* sphinx_rtd_theme >= 2.0.0

For development:

* black (code formatting)
* flake8 (linting)
* mypy (type checking)

Design Patterns
---------------

Factory Pattern
~~~~~~~~~~~~~~~

Used in ``ParameterField`` creation:

.. code-block:: python

   # Different parametrization types
   scalar_field = ParameterField(
       parametrization_type="scalar",
       mean=1.0
   )
   
   normal_field = ParameterField(
       parametrization_type="normal",
       mean=1.0,
       std=0.1
   )

Strategy Pattern
~~~~~~~~~~~~~~~~

Used in system profiling:

.. code-block:: python

   # Different strategies for different methods
   def get_optimal_parameters(method: str, verbose: bool):
       if method == 'parametric':
           return parametric_strategy()
       elif method == 'sobol':
           return sobol_strategy()
       elif method == 'lhs':
           return lhs_strategy()

Dependency Injection
~~~~~~~~~~~~~~~~~~~~

Used in configuration loading:

.. code-block:: python

   # Inject dependencies, not hardcode
   def main(config_loader=load_config, 
            profiler=get_optimal_parameters):
       config = config_loader(config_file)
       params = profiler(config['method'])

Testing Architecture
--------------------

Test Organization
~~~~~~~~~~~~~~~~~

.. code-block:: text

   tests/
   ├── conftest.py                  # Shared fixtures
   ├── test_io_functions.py         # Unit tests
   ├── test_main.py                 # Integration tests
   └── test_system_profiler.py      # Unit tests

Fixture Strategy
~~~~~~~~~~~~~~~~

.. code-block:: python

   # conftest.py - Shared fixtures
   @pytest.fixture
   def temp_dir():
       """Temporary directory for file operations"""
       with tempfile.TemporaryDirectory() as tmpdir:
           yield Path(tmpdir)
   
   @pytest.fixture
   def sample_yaml_file(temp_dir):
       """Sample YAML configuration"""
       yaml_path = temp_dir / "test_config.yaml"
       # ... create file
       return yaml_path

See :doc:`testing` for complete testing guide.

Performance Considerations
--------------------------

Memory Management
~~~~~~~~~~~~~~~~~

* Use generators for large datasets:

  .. code-block:: python
  
     def generate_data():
         for item in large_dataset:
             yield process(item)

* Avoid unnecessary copies:

  .. code-block:: python
  
     # Bad: Creates copy
     result = data.copy()
     
     # Good: Modify in place when possible
     data[:] = process(data)

Parallel Processing
~~~~~~~~~~~~~~~~~~~

System profiler automatically calculates optimal parameters:

.. code-block:: python

   optimal = get_optimal_parameters(method='parametric')
   # Returns: n_jobs, chunk_size, batch_size
   
   # Use with multiprocessing
   with Pool(processes=optimal['n_jobs']) as pool:
       results = pool.map(func, data, 
                         chunksize=optimal['chunk_size'])

Error Handling
--------------

Exception Strategy
~~~~~~~~~~~~~~~~~~

* Use specific exceptions:

  .. code-block:: python
  
     if not yaml_path.exists():
         raise FileNotFoundError(f"Config file not found: {yaml_path}")
     
     if 'analysis_type' not in config:
         raise ValueError("Missing required field: analysis_type")

* Provide helpful error messages:

  .. code-block:: python
  
     raise ValueError(
         f"Invalid analysis type: {analysis_type}. "
         f"Must be one of: {', '.join(VALID_TYPES)}"
     )

Validation
~~~~~~~~~~

Validate early and fail fast:

.. code-block:: python

   def load_config(yaml_path: Path) -> Dict[str, Any]:
       """Load configuration with validation"""
       if not yaml_path.exists():
           raise FileNotFoundError(f"Not found: {yaml_path}")
       
       config = yaml.safe_load(yaml_path.read_text())
       
       # Validate required fields immediately
       required = ['analysis_type', 'method']
       for field in required:
           if field not in config:
               raise ValueError(f"Missing required: {field}")
       
       return config

Future Architecture
-------------------

Planned Improvements
~~~~~~~~~~~~~~~~~~~~

**Short-term:**

* Plugin system for custom analyses
* Configuration schema validation
* Caching layer for expensive operations
* Async I/O for large files

**Long-term:**

* Web API interface
* Database backend for results
* Distributed computing support
* Real-time monitoring dashboard

See Also
--------

* :doc:`testing` - Testing guide
* :doc:`refactoring_history` - Refactoring details
* :doc:`contributing` - How to contribute
* :doc:`../api_reference/index` - API documentation
