Getting Started
===============

Installation
------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.11 or higher
* conda (recommended) or venv for environment management

Required Packages
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt

Core dependencies:

* ``pyyaml`` - YAML configuration parsing
* ``numpy`` - Numerical computations
* ``h5py`` - HDF5 file I/O
* ``pint`` - Unit conversions
* ``psutil`` - System profiling
* ``tqdm`` - Progress bars
* ``joblib`` - Parallel processing

Development dependencies:

* ``pytest`` - Testing framework
* ``pytest-cov`` - Test coverage

Environment Setup
-----------------

Using conda (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create environment
   conda create -n ddstartupenv python=3.13
   conda activate ddstartupenv

   # Install dependencies
   cd dd_startup
   pip install -r requirements.txt

Using venv
~~~~~~~~~~

.. code-block:: bash

   # Create environment
   python -m venv ddstartupenv
   source ddstartupenv/bin/activate  # On Windows: ddstartupenv\Scripts\activate

   # Install dependencies
   cd dd_startup
   pip install -r requirements.txt

Directory Structure
-------------------

.. code-block:: text

   dd_startup/
   ├── ddstartup/                    # Main package
   │   ├── main.py                   # Entry point
   │   ├── inputs/                   # Configuration files
   │   │   ├── config.py             # Parameter definitions
   │   │   ├── config_test.py        # Test parameters
   │   │   ├── parametric_tseeded.yaml
   │   │   ├── sobol_tseeded.yaml
   │   │   └── parametric_lump.yaml
   │   ├── outputs/                  # Generated results (HDF5 files)
   │   ├── utils/                    # Utility modules
   │   │   ├── io_functions.py       # I/O operations
   │   │   ├── system_profiler.py    # Hardware profiling
   │   │   ├── custom_classes.py     # ParameterField class
   │   │   └── units_and_constants.py
   │   ├── physics/                  # Physics computations
   │   │   ├── lump_functions.py     # Lump analysis
   │   │   ├── Tseeded_functions.py  # T-seeded analysis
   │   │   └── sobol_functions.py    # Sobol sensitivity
   │   └── tests/                    # Test suite
   │       ├── test_io_functions.py
   │       ├── test_main.py
   │       └── test_system_profiler.py
   ├── docs/                         # Documentation
   └── requirements.txt              # Python dependencies

Quick Test
----------

Verify Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd ddstartup
   python main.py config_test parametric_tseeded --dry-run

Expected output:

.. code-block:: text

   ============================================================
   SYSTEM PROFILE
   ============================================================
   Hardware:
     CPU Cores: 12
     Total RAM: 8.2 GB
     Available RAM: 2.9 GB (35.5% free)
   
   Recommended Parallel Processing Parameters:
     n_jobs: 11 (parallel workers)
     chunk_size: 5500 (computations per chunk)
     batch_size: 100 (results buffer size)
   ============================================================
   
   ✅ Dry run completed. Configuration validated successfully.

Run Tests
~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run with coverage
   pytest tests/ --cov=. --cov-report=html

   # Run specific test file
   pytest tests/test_system_profiler.py -v

Next Steps
----------

* Read the :doc:`user_guide/index` for detailed usage instructions
* Explore :doc:`examples/index` for common use cases
* Check :doc:`api_reference/index` for module documentation
* See :doc:`developer_guide/index` for contribution guidelines
