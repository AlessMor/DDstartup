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
   │   │   ├── __init__.py            # Package initialization
   │   │   ├── units_and_constants.py # Physical constants
   │   │   ├── parameter_registry.py  # Parameter schema
   │   │   ├── io_functions.py        # I/O operations
   │   │   ├── tools.py               # Utility functions
   │   │   └── system_profiler.py     # Performance optimization
   │   ├── physics/                   # Physics models
   │   │   ├── Tseeded_functions.py   # Time-dependent model
   │   │   ├── lump_functions.py      # Steady-state model
   │   │   └── reactivity_functions.py # Fusion reactivity
   │   ├── inputs/                    # Configuration files
   │   │   ├── __init__.py            # Path setup
   │   │   └── *.yaml                 # YAML configs
   │   ├── outputs/                   # Analysis results
   │   └── tests/                     # Test suite
   │       ├── conftest.py            # Pytest fixtures
   │       └── unit/                  # Unit tests
   │           ├── physics/           # Physics tests
   │           ├── io/                # I/O tests
   │           └── utils/             # Utility tests
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

Utility Modules
~~~~~~~~~~~~~~~

**utils/units_and_constants.py** - Physical Constants

  * Physical constants (c, e, ε₀, etc.)
  * Mathematical constants (π, e)
  * Conversion factors
  * Unit definitions

**utils/parameter_registry.py** - Parameter Schema

  * Central parameter registry
  * Parameter definitions and metadata
  * Type validation
  * Unit handling

  .. code-block:: python
  
     from ddstartup.utils.parameter_registry import get_registry
     
     registry = get_registry()
     result_dict = registry.make_result_dict(
         param_values, outputs
     )

**utils/io_functions.py** - Input/Output Operations

  * YAML configuration loading
  * File path resolution
  * Data extraction
  * Configuration display

  Key functions:
  
  * ``resolve_file_path()`` - Smart file resolution
  * ``load_config()`` - YAML loading with validation
  * ``load_parameter_fields()`` - Module importing
  * ``prepare_input_data()`` - Data extraction
  * ``print_configuration()`` - Display utility

**utils/tools.py** - Utility Functions

  * Parameter parsing and indexing
  * Vector length fixing
  * General helper functions

**utils/system_profiler.py** - Performance Optimization

  * Hardware detection
  * Optimal parameter calculation for parallel processing
  * Used internally for performance tuning

Import System
-------------

The import system is designed for simplicity and explicit imports.Package Initialization
~~~~~~~~~~~~~~~~~~~~~~

**utils/__init__.py** Configuration:

.. code-block:: python

   """
   Utils package initialization
   """
   
   # Package marker - explicit imports preferred

Import Patterns
~~~~~~~~~~~~~~~

Use explicit imports for clarity:

**Direct Module Import (Recommended)**

.. code-block:: python

   from ddstartup.utils.parameter_registry import get_registry
   from ddstartup.utils.io_functions import load_config
   
   # Clear and explicit
   registry = get_registry()
   config = load_config('config.yaml')

**Function-level Import**

.. code-block:: python

   from ddstartup.physics.lump_functions import compute_lump
   from ddstartup.physics.Tseeded_functions import compute_Tseeded
   
   # Direct function access
   result = compute_lump(params)

Usage Examples
~~~~~~~~~~~~~~

**In analysis scripts:**

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
   from ddstartup.utils.io_functions import load_config
   from ddstartup.utils.parameter_registry import get_registry
   from ddstartup.physics.lump_functions import compute_lump

Best Practices
--------------

Import Guidelines
~~~~~~~~~~~~~~~~~

✅ **DO:**

* Use explicit imports:

  .. code-block:: python
  
     from ddstartup.utils.parameter_registry import get_registry
     from ddstartup.utils.io_functions import load_config

* Use type hints for all functions:

  .. code-block:: python
  
     from pathlib import Path
     from typing import Dict, Any
     
     def load_config(yaml_path: Path) -> Dict[str, Any]:
         """Load YAML configuration file"""
         ...

* Keep imports organized (stdlib, third-party, local)

⚠️ **AVOID:**

* Wildcard imports in library code:

  .. code-block:: python
  
     from ddstartup.utils import *  # Avoid in library code

* Importing from private modules:

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
   # Package marker - explicit imports preferred
   
   # When adding new modules, import them explicitly where needed

**Step 3: Write Tests**

.. code-block:: python

   # tests/unit/utils/test_new_module.py
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
   └── unit/                        # Unit tests
       ├── physics/                 # Physics tests
       ├── io/                      # I/O tests
       └── utils/                   # Utility tests

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
