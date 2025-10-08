System Profiler Module
======================

The ``system_profiler`` module provides automatic hardware detection and optimal parameter calculation for parallel computation.

.. module:: ddstartup.utils.system_profiler

Overview
--------

The system profiler automatically detects:

* CPU cores and frequency
* Total and available RAM
* System load

And calculates optimal values for:

* Number of parallel jobs (``n_jobs``)
* Chunk size for batching (``chunk_size``)
* Buffer size for results (``batch_size``)
* Sobol sample count (``N_SAMPLES``)
* Sobol sensitivity order (``order``)

Core Functions
--------------

get_system_info
~~~~~~~~~~~~~~~

.. function:: get_system_info() -> Dict[str, any]

   Get comprehensive system information.

   :returns: Dictionary containing:
      
      * ``n_cores`` (int) - Number of CPU cores
      * ``total_ram_gb`` (float) - Total RAM in GB
      * ``available_ram_gb`` (float) - Available RAM in GB  
      * ``ram_percent_used`` (float) - RAM usage percentage
      * ``cpu_freq_mhz`` (float or None) - CPU frequency in MHz

   :rtype: dict

   **Example:**

   .. code-block:: python

      from ddstartup.utils.system_profiler import get_system_info

      info = get_system_info()
      print(f"Cores: {info['n_cores']}")
      print(f"RAM: {info['total_ram_gb']:.1f} GB")
      # Cores: 12
      # RAM: 8.2 GB

calculate_optimal_n_jobs
~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: calculate_optimal_n_jobs(system_info: Optional[Dict] = None) -> int

   Calculate optimal number of parallel jobs based on CPU cores.

   :param system_info: System information dictionary (if None, will be fetched)
   :type system_info: dict or None
   :returns: Optimal number of parallel jobs
   :rtype: int

   **Strategy:**

   * 16+ cores: Use all cores
   * 8-15 cores: Leave 1 core free
   * 4-7 cores: Leave 1 core free
   * 2-3 cores: Leave 1 core free (minimum 1)

   **Example:**

   .. code-block:: python

      from ddstartup.utils.system_profiler import calculate_optimal_n_jobs

      n_jobs = calculate_optimal_n_jobs()
      print(f"Recommended n_jobs: {n_jobs}")
      # Recommended n_jobs: 11 (for 12-core system)

calculate_optimal_chunk_size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: calculate_optimal_chunk_size(system_info: Optional[Dict] = None, n_jobs: Optional[int] = None) -> int

   Calculate optimal chunk size for parallel processing.

   :param system_info: System information dictionary
   :type system_info: dict or None
   :param n_jobs: Number of parallel jobs (if None, will be calculated)
   :type n_jobs: int or None
   :returns: Optimal chunk size
   :rtype: int

   **Strategy:**

   * 16+ cores: 5000+ base chunk size
   * 8-15 cores: 2000 base chunk size
   * 4-7 cores: 1000 base chunk size
   * 2-3 cores: 500 base chunk size
   * Scales with n_jobs (minimum: base + n_jobs × 500)

   **Example:**

   .. code-block:: python

      from ddstartup.utils.system_profiler import calculate_optimal_chunk_size

      chunk_size = calculate_optimal_chunk_size()
      print(f"Recommended chunk_size: {chunk_size}")
      # Recommended chunk_size: 5500

calculate_optimal_batch_size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: calculate_optimal_batch_size(system_info: Optional[Dict] = None) -> int

   Calculate optimal batch size for buffered operations.

   :param system_info: System information dictionary
   :type system_info: dict or None
   :returns: Optimal batch size
   :rtype: int

   **Strategy:**

   * 16+ cores: 1000 batch size
   * 8-15 cores: 500 batch size
   * 4-7 cores: 250 batch size
   * 2-3 cores: 100 batch size
   * Reduced if available RAM < 8 GB

   **Example:**

   .. code-block:: python

      from ddstartup.utils.system_profiler import calculate_optimal_batch_size

      batch_size = calculate_optimal_batch_size()
      print(f"Recommended batch_size: {batch_size}")
      # Recommended batch_size: 100

calculate_optimal_sobol_samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: calculate_optimal_sobol_samples(system_info: Optional[Dict] = None) -> int

   Calculate optimal number of samples for Sobol analysis.

   :param system_info: System information dictionary
   :type system_info: dict or None
   :returns: Recommended number of Sobol samples
   :rtype: int

   **Strategy:**

   * 16+ cores & 16+ GB RAM: 100,000 samples
   * 8+ cores & 8+ GB RAM: 50,000 samples
   * 4+ cores: 10,000 samples
   * < 4 cores: 5,000 samples
   * Reduced if available RAM is limited

   **Example:**

   .. code-block:: python

      from ddstartup.utils.system_profiler import calculate_optimal_sobol_samples

      n_samples = calculate_optimal_sobol_samples()
      print(f"Recommended N_SAMPLES: {n_samples:,}")
      # Recommended N_SAMPLES: 5,000

calculate_optimal_sobol_order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: calculate_optimal_sobol_order(system_info: Optional[Dict] = None) -> int

   Calculate optimal order for Sobol sensitivity analysis.

   :param system_info: System information dictionary
   :type system_info: dict or None
   :returns: Recommended Sobol analysis order (2 or 3)
   :rtype: int

   **Strategy:**

   * 8+ cores & 8+ GB RAM: Order 3
   * < 8 cores or < 8 GB RAM: Order 2

   **Example:**

   .. code-block:: python

      from ddstartup.utils.system_profiler import calculate_optimal_sobol_order

      order = calculate_optimal_sobol_order()
      print(f"Recommended order: {order}")
      # Recommended order: 2

Master Functions
----------------

get_optimal_parameters
~~~~~~~~~~~~~~~~~~~~~~

.. function:: get_optimal_parameters(analysis_method: str = 'parametric', verbose: bool = False) -> Dict[str, any]

   Get all optimal parameters for parallel computation.

   :param analysis_method: Type of analysis ('parametric' or 'sobol')
   :type analysis_method: str
   :param verbose: If True, print system information
   :type verbose: bool
   :returns: Dictionary containing:

      * ``system_info`` (dict) - System hardware information
      * ``n_jobs`` (int) - Optimal number of parallel jobs
      * ``chunk_size`` (int) - Optimal chunk size
      * ``batch_size`` (int) - Optimal batch size
      * ``N_SAMPLES`` (int) - Optimal Sobol samples (if sobol method)
      * ``order`` (int) - Optimal Sobol order (if sobol method)

   :rtype: dict

   **Example:**

   .. code-block:: python

      from ddstartup.utils.system_profiler import get_optimal_parameters

      # Parametric analysis
      params = get_optimal_parameters(analysis_method='parametric', verbose=True)
      print(f"n_jobs: {params['n_jobs']}")
      print(f"chunk_size: {params['chunk_size']}")

      # Sobol analysis
      params = get_optimal_parameters(analysis_method='sobol', verbose=True)
      print(f"N_SAMPLES: {params['N_SAMPLES']:,}")
      print(f"order: {params['order']}")

override_with_config
~~~~~~~~~~~~~~~~~~~~

.. function:: override_with_config(optimal_params: Dict[str, any], config: Dict[str, any]) -> Dict[str, any]

   Override optimal parameters with user-specified config values.

   :param optimal_params: Dictionary of optimal parameters
   :type optimal_params: dict
   :param config: User configuration dictionary
   :type config: dict
   :returns: Updated parameters dictionary (original not modified)
   :rtype: dict

   **Behavior:**

   * Only overrides if config value is not None
   * Preserves system_info in result
   * Does not modify original optimal_params

   **Example:**

   .. code-block:: python

      from ddstartup.utils.system_profiler import (
          get_optimal_parameters,
          override_with_config
      )

      # Get recommendations
      optimal = get_optimal_parameters('parametric')
      print(f"Recommended n_jobs: {optimal['n_jobs']}")
      # Recommended n_jobs: 11

      # User overrides
      config = {
          'n_jobs': 8,          # Override
          'chunk_size': None,   # Don't override
          'batch_size': 1000    # Override
      }

      final = override_with_config(optimal, config)
      print(f"Final n_jobs: {final['n_jobs']}")          # 8 (overridden)
      print(f"Final chunk_size: {final['chunk_size']}")  # 5500 (kept)
      print(f"Final batch_size: {final['batch_size']}")  # 1000 (overridden)

Utility Functions
-----------------

print_system_profile
~~~~~~~~~~~~~~~~~~~~

.. function:: print_system_profile(params: Dict[str, any], analysis_method: str = 'parametric') -> None

   Print formatted system profile information.

   :param params: Dictionary containing system info and optimal parameters
   :type params: dict
   :param analysis_method: Type of analysis being performed
   :type analysis_method: str

   **Example:**

   .. code-block:: python

      from ddstartup.utils.system_profiler import (
          get_optimal_parameters,
          print_system_profile
      )

      params = get_optimal_parameters('sobol')
      print_system_profile(params, 'sobol')

   **Output:**

   .. code-block:: text

      ============================================================
      SYSTEM PROFILE
      ============================================================
      Hardware:
        CPU Cores: 12
        CPU Frequency: 2800 MHz
        Total RAM: 8.2 GB
        Available RAM: 2.9 GB (35.5% free)

      Recommended Parallel Processing Parameters:
        n_jobs: 11 (parallel workers)
        chunk_size: 5500 (computations per chunk)
        batch_size: 100 (results buffer size)

      Sobol Analysis Parameters:
        N_SAMPLES: 5,000 (number of samples)
        order: 2 (sensitivity order)
      ============================================================

Integration Example
-------------------

Complete workflow example:

.. code-block:: python

   from ddstartup.utils.system_profiler import (
       get_optimal_parameters,
       override_with_config,
       print_system_profile
   )

   # User configuration from YAML
   config = {
       'method': 'sobol',
       'n_jobs': None,      # Auto-detect
       'chunk_size': 2000,  # User override
       'batch_size': None,  # Auto-detect
       'N_SAMPLES': 100000, # User override
       'order': None        # Auto-detect
   }

   # Get optimal parameters
   optimal = get_optimal_parameters(
       analysis_method=config['method'],
       verbose=True
   )

   # Apply user overrides
   final_params = override_with_config(optimal, config)

   # Update config with final values
   config.update({
       'n_jobs': final_params['n_jobs'],
       'chunk_size': final_params['chunk_size'],
       'batch_size': final_params['batch_size'],
       'N_SAMPLES': final_params['N_SAMPLES'],
       'order': final_params['order']
   })

   print(f"Final configuration:")
   print(f"  n_jobs: {config['n_jobs']}")         # 11 (auto)
   print(f"  chunk_size: {config['chunk_size']}") # 2000 (user)
   print(f"  batch_size: {config['batch_size']}") # 100 (auto)
   print(f"  N_SAMPLES: {config['N_SAMPLES']}")   # 100000 (user)
   print(f"  order: {config['order']}")           # 2 (auto)

Dependencies
------------

* ``multiprocessing`` - CPU core detection
* ``psutil`` - System resource monitoring

See Also
--------

* :doc:`../user_guide/configuration_files` - YAML configuration format
* :doc:`io_functions` - I/O operations module
