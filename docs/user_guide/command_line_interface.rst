Command-Line Interface
======================

Overview
--------

The DD Startup Analysis Tool provides a command-line interface for running fusion reactor startup simulations.

Basic Syntax
------------

.. code-block:: bash

   python main.py <params> <config> [--verbose] [--dry-run]

Positional Arguments
--------------------

params
~~~~~~

**Required**. Parameter configuration file defining the simulation parameter space.

* Can be just the filename (e.g., ``config``) - will search in ``inputs/`` directory
* Or full path to file (e.g., ``inputs/config.py`` or ``/path/to/config.py``)
* Must be a valid Python file defining ``ParameterField`` objects

**Examples:**

.. code-block:: bash

   # Short name (searches in inputs/)
   python main.py config parametric_tseeded
   
   # Relative path
   python main.py inputs/config.py parametric_tseeded
   
   # Absolute path
   python main.py /home/user/configs/custom_params.py parametric_tseeded

config
~~~~~~

**Required**. YAML configuration file defining the analysis run settings.

* Can be just the filename (e.g., ``parametric_tseeded``) - will search in ``inputs/`` directory
* Or full path to file (e.g., ``inputs/parametric_tseeded.yaml``)
* Must be a valid YAML file with required fields

**Examples:**

.. code-block:: bash

   # Short name (searches in inputs/)
   python main.py config parametric_tseeded
   
   # With .yaml extension
   python main.py config parametric_tseeded.yaml
   
   # Full path
   python main.py config /path/to/run_configs/my_analysis.yaml

Optional Arguments
------------------

--verbose
~~~~~~~~~

Enable verbose output. Prints detailed information about:

* System hardware profile
* Configuration parameters
* Parameter field details
* Input data ranges
* Total parameter combinations

Overrides the ``verbose`` setting in the YAML configuration file.

**Example:**

.. code-block:: bash

   python main.py config parametric_tseeded --verbose

**Output:**

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
   
   ============================================================
   DD STARTUP ANALYSIS CONFIGURATION
   ============================================================
   Parameter file: inputs/config.py
   Config file: inputs/parametric_tseeded.yaml
   Analysis type: T_seeded
   Method: parametric
   Total time: 10.00 years
   Vector length: 100
   ...

--dry-run
~~~~~~~~~

Print configuration without running the analysis. Useful for:

* Validating configuration files
* Checking parameter combinations
* Verifying system profiling
* Testing file paths

**Example:**

.. code-block:: bash

   python main.py config parametric_tseeded --dry-run

**Output:**

.. code-block:: text

   [System profile and configuration details]
   
   ✅ Dry run completed. Configuration validated successfully.

Combining Flags
~~~~~~~~~~~~~~~

Flags can be combined in any order:

.. code-block:: bash

   # Both flags
   python main.py config parametric_tseeded --verbose --dry-run
   
   # Same result, different order
   python main.py config parametric_tseeded --dry-run --verbose

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

Files in ``inputs/`` directory:

.. code-block:: bash

   # Parametric T_seeded analysis
   python main.py config parametric_tseeded
   
   # Sobol T_seeded analysis
   python main.py config sobol_tseeded
   
   # Parametric lump analysis
   python main.py config parametric_lump

Testing Configuration
~~~~~~~~~~~~~~~~~~~~~

Always test with ``--dry-run`` first:

.. code-block:: bash

   # Validate configuration
   python main.py config_test parametric_tseeded --dry-run
   
   # Check with verbose output
   python main.py config_test parametric_tseeded --verbose --dry-run
   
   # Run actual analysis after validation
   python main.py config_test parametric_tseeded

Production Runs
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Silent run (no verbose output)
   python main.py config parametric_tseeded
   
   # Verbose run for monitoring
   python main.py config parametric_tseeded --verbose
   
   # Long-running background process
   nohup python main.py config sobol_tseeded --verbose > run.log 2>&1 &

Multiple Sequential Runs
~~~~~~~~~~~~~~~~~~~~~~~~~

Batch script example:

.. code-block:: bash

   #!/bin/bash
   # run_all.sh
   
   echo "Starting batch analysis..."
   
   # Run parametric analyses
   python main.py config parametric_tseeded --verbose
   python main.py config parametric_lump --verbose
   
   # Run Sobol analyses
   python main.py config sobol_tseeded --verbose
   
   echo "Batch analysis completed!"

Exit Codes
----------

The script returns standard exit codes:

* ``0`` - Success
* ``1`` - Error (file not found, invalid config, import error, etc.)

**Example usage in scripts:**

.. code-block:: bash

   python main.py config parametric_tseeded
   if [ $? -eq 0 ]; then
       echo "Analysis completed successfully"
   else
       echo "Analysis failed"
       exit 1
   fi

Error Messages
--------------

File Not Found
~~~~~~~~~~~~~~

.. code-block:: text

   Error: File not found: config
   Searched in: config, inputs/config, config.py, inputs/config.py

**Solution:** Ensure the file exists in ``inputs/`` directory or provide full path.

Invalid Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Error loading configuration: Missing required field in config: analysis_type

**Solution:** Verify YAML file has all required fields (``analysis_type``, ``method``).

Import Error
~~~~~~~~~~~~

.. code-block:: text

   Error loading parameter fields: Cannot import module: config

**Solution:** Check that parameter file is valid Python and all imports work.

System Profiling Error
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Error during system profiling: [error details]

**Solution:** Ensure ``psutil`` is installed and system resources are accessible.

See Also
--------

* :doc:`configuration_files` - YAML configuration reference
* :doc:`parameter_definitions` - Parameter file format
* :doc:`troubleshooting` - Common issues and solutions
