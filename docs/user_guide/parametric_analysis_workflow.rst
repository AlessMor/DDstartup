Parametric Analysis Workflow
=============================

This guide provides a detailed, step-by-step explanation of how parametric analysis works in the DD Startup Analysis Tool, following the execution flow from command-line invocation through to HDF5 output generation.

Overview
--------

The parametric analysis workflow consists of these major phases:

1. **Initialization** - Parse arguments, load configuration
2. **Input Preparation** - Load parameters, prepare data structures
3. **System Optimization** - Determine optimal parallel processing settings
4. **Parallel Execution** - Compute all parameter combinations
5. **Results Writing** - Save results to HDF5 file

Entry Point: main.py
--------------------

Location: ``ddstartup/main.py``

The ``main()`` function orchestrates the entire analysis workflow.

Command-Line Invocation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m ddstartup params_file config_file [--verbose] [--dry-run]


Phase 1: Initialization
-----------------------

Step 1.1: Argument Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ddstartup.utils.tools.parse_arguments` parses command-line arguments to extract:

- ``params``: Parameter file name (e.g., 'params_test')
- ``config``: Configuration file name (e.g., 'parametric_lump')
- ``--verbose``: Enable detailed output
- ``--dry-run``: Validate files without running analysis

.. code-block:: python

   args = parse_arguments()

Step 1.2: File Path Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ddstartup.utils.io_functions.resolve_file_path` locates parameter and configuration files in the ``inputs/`` directory.

**File structure:**

- **Parameter file**: Defines physics parameters (YAML format)
  
  Example: ``inputs/params_test.yaml``
  
  .. code-block:: yaml
  
     V_plasma:
       values: [100, 150, 200]
       unit: m^3
     
     T_i:
       values: [15, 20, 25]
       unit: keV
     
     ... (additional parameters)

- **Configuration file**: Analysis settings (YAML format)
  
  Example: ``inputs/parametric_lump.yaml``
  
  .. code-block:: yaml
  
     analysis_type: lump
     method: parametric
     n_jobs: null  # Auto-detect
     chunk_size: 1000
     batch_size: 100
     verbose: true
     filter: "t_startup < 1000 and sol_success == True"

.. code-block:: python

   param_file = resolve_file_path(args.params, 'inputs', ['.yaml', '.yml', '.py'])
   config_file = resolve_file_path(args.config, 'inputs', ['.yaml', '.yml'])

Step 1.3: Configuration Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ddstartup.utils.io_functions.load_config` parses the YAML configuration file into a Python dictionary.

**Key configuration fields:**

- ``analysis_type``: 'lump' or 'T_seeded'
- ``method``: 'parametric', 'sobol', or 'lhs'
- ``n_jobs``: Number of parallel workers (null = auto-detect)
- ``chunk_size``: Work distribution granularity
- ``batch_size``: HDF5 write buffer size
- ``verbose``: Enable detailed logging
- ``filter``: Expression to filter valid results (e.g., ``"t_startup < 1000 and sol_success == True"``), this allows to reduce the dimension of the h5df5 file
- ``total_time``: Maximum simulation time for T_seeded analysis

.. code-block:: python

   config = load_config(config_file)
   if args.verbose:
       config['verbose'] = True

Phase 2: Input Preparation
---------------------------

Step 2.1: Parameter Field Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ddstartup.utils.io_functions.load_parameter_fields` loads parameter definitions from YAML file and converts to structured format.

**Parameter field structure:**

Each parameter has:

- ``values``: Single value or list of values for parametric sweep
- ``unit``: Physical unit (e.g., 'm^3', 'keV', '1/m^3')
- ``parametrization_type``: 'range', 'list', or 'constant'

**Internal representation:** Dictionary of parameter names → metadata

.. code-block:: python

   param_fields = load_parameter_fields(param_file)

Step 2.2: Input Data Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ddstartup.utils.io_functions.prepare_input_data` converts parameter fields into numpy arrays ready for computation.

**Transformation:**

1. Extract parameter values from field definitions
2. Create meshgrid for all parameter combinations
3. Flatten arrays for efficient iteration
4. Store grid shapes for index reconstruction

**Output structure:**

.. code-block:: python

   input_data = {
       'input_arrays': [V_plasma_grid, T_i_grid, n_tot_grid, ...],
       'input_arrays_flat': [V_plasma_flat, T_i_flat, n_tot_flat, ...],
       'param_names': ['V_plasma', 'T_i', 'n_tot', ...],
       'param_units': ['m^3', 'keV', '1/m^3', ...],
       'param_shapes': (3, 3, 1, ...),  # Grid dimensions
       'n_combinations': 9,  # Total parameter combinations
   }

**Example:** If V_plasma has 3 values and T_i has 3 values, we get 3×3 = 9 combinations.

.. code-block:: python

   input_data = prepare_input_data(param_fields, config['analysis_type'])

Phase 3: System Optimization
-----------------------------

Step 3.1: Hardware Profiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ddstartup.utils.system_profiler.get_optimal_parameters` detects system hardware and calculates optimal parallel processing parameters.

**Detection:**

- Number of CPU cores
- Available RAM
- Current system load

**Calculated parameters:**

- ``n_jobs``: Number of parallel workers (typically n_cores - 1)
- ``chunk_size``: Work distribution size (1000-10000)
- ``batch_size``: HDF5 write buffer (100-1000)

.. code-block:: python

   optimal_params = get_optimal_parameters(
       analysis_method=config['method'],
       verbose=verbose
   )

Step 3.2: Configuration Override
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ddstartup.utils.system_profiler.override_with_config` allows user-specified values to override auto-detected settings.

**Behavior:**

- If config value is ``null`` → use auto-detected value
- If config value is specified → use user value

.. code-block:: python

   optimal_params = override_with_config(optimal_params, config)
   config.update(optimal_params)

Step 3.3: Configuration Display
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ddstartup.utils.io_functions.print_configuration` displays the complete configuration for user verification (if verbose mode).

**Output includes:**

- Analysis type and method
- Parameter ranges and sweep dimensions
- Parallel processing settings
- Total number of combinations

Phase 4: Parallel Execution
----------------------------

This is where the actual physics computations happen.

Step 4.1: Output File Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ddstartup.utils.io_functions.generate_output_path` creates a timestamped output directory and filename.

**Output structure:**

.. code-block:: text

   outputs/
   └── YYYYMMDD_HHMMSS_method_analysistype/
       └── ddstartup_YYYYMMDD_HHMMSS_method_analysistype.h5

Example: ``outputs/20251107_143022_parametric_lump/ddstartup_20251107_143022_parametric_lump.h5``

Step 4.2: Parametric Analysis Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ddstartup.utils.parametric_computation.run_parametric_analysis` orchestrates the parallel computation of all parameter combinations.

**Architecture:**

1. **Compute Function Creation** (Internal Factory)
   
   Based on ``analysis_type``, creates appropriate compute function:
   
   - ``analysis_type='lump'`` → ``_compute_lump()``
   - ``analysis_type='T_seeded'`` → ``_compute_tseeded()``
   
   These functions are defined at module level in ``parametric_computation.py`` for pickling.

2. **HDF5 File Initialization**
   
   Internal function ``_create_hdf5_structure()`` creates HDF5 datasets for:
   
   - Input parameters (V_plasma, T_i, n_tot, ...)
   - Output results (n_T, n_D, t_startup, ...)
   - Economic outputs (unrealized_profits, ...)
   - Metadata (configuration, timestamps)
   
   **Compression:** LZ4 (fast compression, ~10x faster than gzip)

3. **Work Distribution**
   
   Generate linear indices: 0, 1, 2, ..., n_combinations-1
   
   Each index maps to a unique parameter combination via ``index_to_params()``.

4. **Parallel Computation with Dynamic Work Queue**
   
   **Executor:** ``ProcessPoolExecutor`` (from ``concurrent.futures``)
   
   **Why ProcessPoolExecutor?**
   
   - Persistent worker processes (no fork overhead per task)
   - Better than joblib for long-running compute functions
   - Dynamic work queue for optimal load balancing
   
   **Dynamic Work Queue Strategy:**
   
   The implementation uses an adaptive task submission approach:
   
   a. **Initial Queue:** Submit 2×n_jobs tasks to start (e.g., 22 tasks for 11 workers)
   b. **Continuous Submission:** As each task completes, immediately submit the next task
   c. **Load Balancing:** Ensures no workers sit idle while tasks remain
   d. **Adaptive:** Automatically adjusts to variable task durations
   
   **Benefits over batch processing:**
   
   - Eliminates idle workers at batch boundaries
   - Better utilization for heterogeneous workloads
   - Improved throughput (5-15% faster for variable-duration tasks)
   
   **Process:**
   
   1. Each worker receives: ``(linear_index, input_arrays_flat, param_shapes_array, reactivity_lookup)``
   2. Worker calls compute function with pre-computed reactivity table
   3. Results collected as they complete and buffered for batch writing
   4. New task submitted immediately when worker becomes available
   
   **Code pattern:**
   
   .. code-block:: python
   
      with ProcessPoolExecutor(max_workers=n_jobs) as executor:
          pending_futures = {}
          next_index = 0
          results_buffer = {}
          
          # Submit initial queue (2x workers)
          for idx in range(min(n_jobs * 2, n_combinations)):
              future = executor.submit(compute_fn, idx, input_arrays, param_shapes, reactivity_lookup)
              pending_futures[future] = idx
              next_index += 1
          
          # Process as completed, submit next task
          while pending_futures:
              done, _ = concurrent.futures.wait(pending_futures.keys(), return_when=FIRST_COMPLETED)
              for future in done:
                  result_idx = pending_futures.pop(future)
                  results_buffer[result_idx] = future.result()
                  
                  # Submit next task immediately
                  if next_index < n_combinations:
                      new_future = executor.submit(compute_fn, next_index, ...)
                      pending_futures[new_future] = next_index
                      next_index += 1
                  
                  # Batch write when buffer full
                  if len(results_buffer) >= write_batch_size:
                      write_to_hdf5(results_buffer)
                      results_buffer.clear()

5. **Batch Writing**
   
   **Purpose:** Minimize HDF5 I/O overhead by buffering results
   
   **Process:**
   
   - Accumulate results in memory buffer
   - When buffer reaches ``batch_size`` → write to HDF5
   - Final write at end for remaining results
   
   **Vectorized writes:** 100x faster than individual writes

6. **Result Filtering** (Optional)
   
   :func:`ddstartup.utils.filters.apply_filter_to_combinations` filters results if a ``filter`` expression is provided.
   Only valid results are written:
   
   Example: ``"t_startup < 1000 and sol_success == True"``
   
   Filter evaluated after computation, before HDF5 write.

.. code-block:: python

   stats = run_parametric_analysis(
       input_data=input_data,
       output_file=output_file,
       config=config,
       verbose=verbose,
       filter_expr=config.get('filter')
   )

Step 4.3: Compute Function Deep Dive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For Lump Analysis:** The internal function ``_compute_lump()`` in ``parametric_computation.py`` follows this execution flow:

2. **Parameter Extraction**
   
   Convert linear index → multi-dimensional indices → parameter values:
   
   .. code-block:: python
   
      idx = index_to_params(linear_index, param_shapes_array)
      V_plasma = input_arrays_flat[0][idx[0]]
      T_i = input_arrays_flat[1][idx[1]]
      # ... extract all parameters

3. **Reactivity Lookup**
   
   :class:`ddstartup.utils.reactivity_lookup.ReactivityLookupTable` provides O(1) dictionary lookups for fusion reactivities.
   
   **Pre-computed rates:**
   
   - σv_DD_p: D+D → He-3 + n
   - σv_DD_n: D+D → T + p  
   - σv_DT: D+T → He-4 + n
   - σv_DHe3: D+He-3 → He-4 + p (optional)
   
   **Lookup mechanism:**
   
   .. code-block:: python
   
      sigmav_DD_p, sigmav_DD_n, sigmav_DT = reactivity_lookup.get_reactivities(T_i)
   
   Temperature keys are rounded to 0.1 eV precision for dictionary access.

4. **Physics Solver Call**   :func:`ddstartup.physics.lump_functions.lump_solver` solves the tritium breeding balance equations.
   
   **Inputs:** Plasma parameters, reaction rates, breeding ratios
   
   **Outputs:** 
   
   - n_T, n_D, n_He3: Steady-state densities
   - t_startup: Time to reach steady state
   - sol_success: Whether solver converged

5. **Power Balance Calculations**
   
   :func:`ddstartup.physics.power_balance.compute_lump_powers_and_energies` calculates fusion power, radiation losses, and net power.
   
   **Calculations:**
   
   - P_fusion: Total fusion power output
   - P_rad: Radiation power loss
   - P_net: Net electrical power output
   - Q: Fusion gain (P_fusion / P_aux)
   
   Safe division with ``np.errstate`` context manager prevents warnings when dividing by zero.

6. **Economic Calculations**
   
   :func:`ddstartup.economics.economics_functions.compute_economics_from_energies` calculates economic metrics.
   
   **Calculations:**
   
   - LCOE: Levelized Cost of Electricity
   - NPV: Net Present Value
   - Payback time
   - IRR: Internal Rate of Return

6. **Result Dictionary Construction**
   
   Build complete result with inputs + outputs:
   
   .. code-block:: python
   
      result = {
          # Inputs
          'linear_index': linear_index,
          'V_plasma': V_plasma,
          'T_i': T_i,
          # ... all input parameters
          
          # Physics outputs
          'n_T': n_T,
          'n_D': n_D,
          't_startup': t_startup,
          'sol_success': sol_success,
          
          # Power outputs
          'P_fusion': P_fusion,
          'Q': Q,
          
          # Economics
          'LCOE': LCOE,
          'payback_time': payback_time,
      }

7. **Parameter Registry Integration**
   
   :func:`ddstartup.utils.parameter_registry.get_registry` provides the central registry.
   The ``make_result_dict()`` method ensures all required fields are present and applies units.
   
   Returns standardized result dictionary ready for HDF5 storage.

**For T_seeded Analysis:** The internal function ``_compute_tseeded()`` in ``parametric_computation.py`` has a similar structure but calls the time-dependent ODE solver:

- :func:`ddstartup.physics.Tseeded_functions.compute_Tseeded` for time evolution
- Uses ``solve_ivp()`` from scipy for ODE integration
- **Outputs:** Time series of densities, final state values

Phase 5: Results Output
------------------------

Step 5.1: HDF5 Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

**File format:** HDF5 with LZ4 compression

**Dataset organization:**

.. code-block:: text

   ddstartup_20251107_143022_parametric_lump.h5
   ├── inputs/
   │   ├── V_plasma          [N×1 array]
   │   ├── T_i               [N×1 array]
   │   ├── n_tot             [N×1 array]
   │   └── ...
   ├── outputs/
   │   ├── n_T               [N×1 array]
   │   ├── n_D               [N×1 array]
   │   ├── t_startup         [N×1 array]
   │   ├── sol_success       [N×1 array]
   │   └── ...
   ├── economics/
   │   ├── LCOE              [N×1 array]
   │   ├── payback_time      [N×1 array]
   │   └── ...
   └── metadata/
       ├── analysis_type     (attribute)
       ├── method            (attribute)
       ├── timestamp         (attribute)
       ├── n_combinations    (attribute)
       └── param_names       (dataset)

Where N = number of valid combinations (after filtering).

Step 5.2: Statistics Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ddstartup.utils.parametric_computation.print_parametric_summary` displays analysis statistics.

**Output includes:**

- Total combinations computed
- Valid results (passed filter)
- Failed computations
- Total execution time
- Combinations per second
- Output file location

.. code-block:: python

   print_parametric_summary(output_file, stats, verbose)

Performance Optimizations
-------------------------

The parametric analysis is optimized for performance:

1. **Persistent Workers**
   
   ProcessPoolExecutor creates worker pool once, reuses for all tasks.
   Avoids fork() overhead of creating new process per combination.

2. **Reactivity Lookup Table**
   
   Pre-compute fusion reactivities for all unique temperatures before spawning workers.
   Each worker receives the lookup table for O(1) access instead of O(n) recalculation.
   
   **Implementation:**
   
   .. code-block:: python
   
      # Before parallel execution
      unique_temperatures = np.unique(T_i_flat)
      reactivity_lookup = ReactivityLookupTable(unique_temperatures)
      
      # In worker process
      sigmav_DD_p, sigmav_DD_n, sigmav_DT = reactivity_lookup.get_reactivities(T_i)
   
   **Speedup:** 20-30% reduction in computation time.

3. **Dynamic Work Queue**
   
   Maintain active task queue (2×n_workers) and submit new tasks as workers complete.
   Eliminates idle workers when task durations vary.
   
   **Speedup:** 5-15% improvement for heterogeneous workloads.

4. **Vectorized HDF5 Writes**
   
   Buffer results in memory, write in batches of 5000.
   ~100x faster than individual writes.

5. **LZ4 Compression**
   
   Fast compression algorithm (10x faster than gzip).
   Reduces file size with minimal CPU overhead.

5. **Efficient Index Mapping**
   
   Linear index → multi-dimensional indices via modular arithmetic.
   Avoids creating full meshgrid in memory.

6. **Minimal Data Transfer**
   
   Worker processes receive only:
   - Linear index (single integer)
   - Flattened arrays (shared read-only)
   - Shape information
   - Reactivity lookup table (pre-computed)
   
   Minimizes inter-process communication overhead.

Error Handling
--------------

The workflow includes comprehensive error handling:

1. **File Validation**
   
   - Check file existence before loading
   - Validate YAML syntax
   - Verify required fields present

2. **Computation Errors**
   
   - Catch solver convergence failures
   - Mark failed combinations with ``sol_success=False``
   - Continue processing remaining combinations

3. **HDF5 Errors**
   
   - Verify write permissions
   - Handle disk space issues
   - Ensure atomic writes (write to temp, then rename)

4. **Memory Management**
   
   - Monitor buffer sizes
   - Flush to disk periodically
   - Prevent out-of-memory errors

Complete Example
----------------

Here's a complete example from command to output:

**Command:**

.. code-block:: bash

   python -m ddstartup params_test parametric_lump --verbose

**Execution trace:**

1. Parse: ``params_test`` → ``inputs/params_test.yaml``
2. Parse: ``parametric_lump`` → ``inputs/parametric_lump.yaml``
3. Load 13 parameters: V_plasma, T_i, n_tot, ...
4. Create grid: 3×3×1×... = 9 combinations
5. Auto-detect: n_jobs=11, chunk_size=5500, batch_size=100
6. Create output: ``outputs/20251107_143022_parametric_lump/``
7. Spawn 11 worker processes
8. Compute 9 combinations in parallel
9. Filter results: 7 valid (t_startup < 1000)
10. Write to HDF5: 7 valid results
11. Complete in 2.3 seconds

**Output:**

.. code-block:: text

   ✅ Analysis completed successfully!
   📁 Results saved to: outputs/20251107_143022_parametric_lump/ddstartup_20251107_143022_parametric_lump.h5
   
   Statistics:
     Total combinations: 9
     Valid results: 7
     Failed: 2
     Execution time: 2.3 seconds
     Rate: 3.9 combinations/second

See Also
--------

* :doc:`../api_reference/parametric_computation` - API documentation
* :doc:`configuration_files` - Configuration file format
* :doc:`parameter_definitions` - Parameter definitions
* :doc:`../developer_guide/architecture` - System architecture
* :doc:`../examples/parametric_analysis` - Usage examples
