Configuration Files
===================

Overview
--------

YAML configuration files define run-specific settings for the DD Startup Analysis Tool. They control the analysis method, performance parameters, and output options.

File Format
-----------

Configuration files use YAML format with key-value pairs:

.. code-block:: yaml

   analysis_type: T_seeded
   method: parametric
   vector_length: 100
   total_time: 315360000
   verbose: true

Required Fields
---------------

analysis_type
~~~~~~~~~~~~~

**Required**. Type of analysis to perform.

**Valid values:**

* ``T_seeded`` - T-seeded startup analysis
* ``lump`` - Lumped parameter analysis

**Example:**

.. code-block:: yaml

   analysis_type: T_seeded

method
~~~~~~

**Required**. Analysis method to use.

**Valid values:**

* ``parametric`` - Parametric sweep across parameter space
* ``sobol`` - Sobol sensitivity analysis

**Example:**

.. code-block:: yaml

   method: parametric

Optional Fields
---------------

Common Fields
~~~~~~~~~~~~~

total_time
^^^^^^^^^^

Total simulation time in seconds.

* **Type:** Integer or float
* **Default:** 315360000 (10 years)
* **Units:** seconds

.. code-block:: yaml

   total_time: 315360000  # 10 years

verbose
^^^^^^^

Enable verbose output.

* **Type:** Boolean
* **Default:** false
* **Note:** Overridden by ``--verbose`` command-line flag

.. code-block:: yaml

   verbose: true

output_dir
^^^^^^^^^^

Output directory for results.

* **Type:** String
* **Default:** "outputs"
* **Note:** Directory will be created if it doesn't exist

.. code-block:: yaml

   output_dir: outputs

Performance Fields
~~~~~~~~~~~~~~~~~~

n_jobs
^^^^^^

Number of parallel worker processes.

* **Type:** Integer or null
* **Default:** null (auto-detect based on CPU cores)
* **Range:** 1 to number of CPU cores

.. code-block:: yaml

   n_jobs: 8  # Use 8 parallel workers
   # or
   n_jobs: null  # Auto-detect optimal value

chunk_size
^^^^^^^^^^

Number of computations per chunk in parallel processing.

* **Type:** Integer or null
* **Default:** null (auto-calculate based on system)
* **Note:** Larger values reduce overhead but increase memory usage

.. code-block:: yaml

   chunk_size: 5000
   # or
   chunk_size: null  # Auto-calculate

batch_size
^^^^^^^^^^

Buffer size for batched HDF5 writes.

* **Type:** Integer
* **Default:** 500
* **Range:** 10 to 10000 (depends on available memory)

.. code-block:: yaml

   batch_size: 1000

T_seeded Specific Fields
~~~~~~~~~~~~~~~~~~~~~~~~~

vector_length
^^^^^^^^^^^^^

Length of output time-series vectors.

* **Type:** Integer
* **Default:** 100
* **Only for:** ``analysis_type: T_seeded``

.. code-block:: yaml

   vector_length: 100

Sobol Specific Fields
~~~~~~~~~~~~~~~~~~~~~

N_SAMPLES
^^^^^^^^^

Number of samples for Sobol analysis.

* **Type:** Integer
* **Default:** null (auto-calculate based on system)
* **Only for:** ``method: sobol``
* **Note:** Higher values give better accuracy but take longer

.. code-block:: yaml

   N_SAMPLES: 100000

order
^^^^^

Order of Sobol sensitivity analysis.

* **Type:** Integer (2 or 3)
* **Default:** null (auto-calculate: 2 or 3 based on system)
* **Only for:** ``method: sobol``
* **Values:**
  
  * ``2`` - First and second-order indices
  * ``3`` - First, second, and third-order indices

.. code-block:: yaml

   order: 3

Example Configurations
----------------------

Parametric T_seeded
~~~~~~~~~~~~~~~~~~~

File: ``inputs/parametric_tseeded.yaml``

.. code-block:: yaml

   # Analysis settings
   analysis_type: T_seeded
   method: parametric
   
   # Simulation parameters
   vector_length: 100
   total_time: 315360000  # 10 years
   
   # Performance settings (auto-detect)
   n_jobs: null
   chunk_size: null
   batch_size: 500
   
   # Output settings
   output_dir: outputs
   verbose: true

Parametric Lump
~~~~~~~~~~~~~~~

File: ``inputs/parametric_lump.yaml``

.. code-block:: yaml

   # Analysis settings
   analysis_type: lump
   method: parametric
   
   # Simulation parameters
   total_time: 315360000  # 10 years
   
   # Performance settings (auto-detect)
   n_jobs: null
   chunk_size: null
   batch_size: 500
   
   # Output settings
   output_dir: outputs
   verbose: true

Sobol T_seeded
~~~~~~~~~~~~~~

File: ``inputs/sobol_tseeded.yaml``

.. code-block:: yaml

   # Analysis settings
   analysis_type: T_seeded
   method: sobol
   
   # Sobol parameters
   N_SAMPLES: 100000
   order: 3
   
   # Simulation parameters
   total_time: 315360000  # 10 years
   
   # Performance settings (manual)
   n_jobs: 8
   chunk_size: 2000
   batch_size: 500
   
   # Output settings
   output_dir: outputs
   verbose: true

High-Performance Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large systems (16+ cores, 32+ GB RAM):

.. code-block:: yaml

   analysis_type: T_seeded
   method: parametric
   
   # Optimized for high-performance systems
   n_jobs: 16
   chunk_size: 10000
   batch_size: 2000
   
   vector_length: 100
   total_time: 315360000
   output_dir: outputs
   verbose: true

Low-Memory Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

For systems with limited RAM (< 8 GB):

.. code-block:: yaml

   analysis_type: T_seeded
   method: parametric
   
   # Optimized for low-memory systems
   n_jobs: 4
   chunk_size: 500
   batch_size: 100
   
   vector_length: 100
   total_time: 315360000
   output_dir: outputs
   verbose: false

System Profiler Integration
----------------------------

When ``null`` values are used for performance parameters, the system profiler automatically:

1. Detects hardware (CPU cores, RAM, frequency)
2. Calculates optimal values based on:
   
   * Number of CPU cores
   * Available RAM
   * Analysis method (parametric vs sobol)
   * Analysis type (T_seeded vs lump)

3. Applies recommendations
4. Allows user overrides from YAML

**Example workflow:**

.. code-block:: yaml

   # YAML file
   n_jobs: null        # Will be auto-detected (e.g., 11 for 12-core system)
   chunk_size: 10000   # User override, will use this value
   batch_size: null    # Will be auto-detected (e.g., 100 for low RAM)

**System profiler output (with** ``--verbose`` **):**

.. code-block:: text

   ============================================================
   SYSTEM PROFILE
   ============================================================
   Hardware:
     CPU Cores: 12
     Total RAM: 8.2 GB
     Available RAM: 2.9 GB (35.5% free)
   
   Recommended Parallel Processing Parameters:
     n_jobs: 11 (parallel workers)        ← auto-detected
     chunk_size: 10000 (from config)      ← user override
     batch_size: 100 (results buffer)     ← auto-detected
   ============================================================

Validation
----------

The configuration loader validates:

* Required fields are present
* Field types are correct
* Values are in valid ranges
* Analysis-specific fields are present

**Common validation errors:**

.. code-block:: text

   ValueError: Missing required field in config: analysis_type
   ValueError: Missing required field in config: method
   ValueError: Invalid analysis_type: invalid_type (must be T_seeded or lump)
   ValueError: Invalid method: invalid_method (must be parametric or sobol)

See Also
--------

* :doc:`command_line_interface` - CLI usage
* :doc:`parameter_definitions` - Parameter file format
