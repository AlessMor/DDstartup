==================================
Parametric Analysis Module
==================================

The ``utils.parametric_computation`` module implements parallel parametric analysis with HDF5 output.

Overview
========

This module orchestrates the execution of parametric studies where parameter combinations are systematically evaluated across a multi-dimensional grid. It provides:

- **Parallel Execution**: Uses ProcessPoolExecutor to distribute computations across persistent worker processes
- **Async I/O**: Background thread for HDF5 writing that doesn't block computation
- **Progress Tracking**: Real-time progress bars with tqdm
- **HDF5 Output**: Efficient storage of large result sets with compression
- **Memory Management**: Buffered writing with queue-based async writes
- **Error Handling**: Graceful handling of failed computations
- **Numba Optimization**: Automatic JIT compilation priming for optimal performance

How Parallel Computation Works
===============================

The parametric analysis uses a **ProcessPoolExecutor with async I/O** pattern for maximum performance:

**1. Data Preparation** (Main Process):

.. code-block:: python

   # Parameters stored as flat 1D arrays
   input_arrays_flat = [
       np.array([100, 150, 200]),        # V_plasma: 3 values
       np.array([15, 17, 19]),           # T_i: 3 values  
       np.array([0.1, 0.5, 1.0, 2.0]),   # tau_p_T: 4 values
   ]
   
   # Total combinations = 3 × 3 × 4 = 36
   # Each combination assigned a linear index: 0, 1, 2, ..., 35

**2. Worker Pool Creation** (ProcessPoolExecutor):

.. code-block:: python

   # Spawn persistent worker processes (e.g., 11 workers)
   # Workers stay alive for entire analysis (unlike joblib which recreates)
   
   with ProcessPoolExecutor(max_workers=11) as executor:
       # Workers initialized once, reused for all tasks
       # Eliminates process startup overhead
       # Numba JIT compilation happens once per worker

**3. Task Distribution** (Futures-based):

.. code-block:: python

   # Submit tasks in chunks to maintain order
   for chunk_start in range(0, n_combinations, chunk_size):
       chunk_indices = [chunk_start, chunk_start+1, ..., chunk_end]
       
       # Submit all tasks for this chunk
       futures = {
           executor.submit(compute_function, idx, ...): idx
           for idx in chunk_indices
       }
       
       # Collect results as they complete (any order)
       for future in as_completed(futures):
           result = future.result()  # Get computed result
           buffer_results.append(result)

**4. Async HDF5 Writing** (Background Thread):

.. code-block:: python

   # Main computation thread:
   buffer_results.append(result)
   if len(buffer_results) >= 1000:
       write_queue.put((buffer_results, indices))  # Non-blocking
       buffer_results.clear()
   
   # Background writer thread (runs concurrently):
   while True:
       results, indices = write_queue.get()
       _write_results_to_hdf5(h5file, results, indices)
       h5file.flush()

**5. Worker Execution** (Each Worker Process):

.. code-block:: python

   # Each worker calls:
   result = compute_function(
       linear_index=5,              # Unique index
       input_arrays_flat=arrays,    # Shared read-only
       param_shapes_array=shapes    # Grid dimensions
   )
   
   # Worker converts: index=5 → (i=1, j=2, k=1)
   # Extracts: V_plasma[1]=150, T_i[2]=19, tau_p_T[1]=0.5
   # Runs physics simulation (with Numba JIT)
   # Returns result dictionary

**Key Improvements Over Previous Implementation**:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Old (joblib)
     - New (ProcessPoolExecutor + Async)
   * - Worker Creation
     - Per-batch process spawn
     - Persistent workers
   * - Numba Compilation
     - Repeated per batch
     - Once per worker
   * - HDF5 Writing
     - Blocks computation
     - Async background thread
   * - Task Ordering
     - Sequential batches
     - Futures-based (out-of-order OK)
   * - Memory Efficiency
     - Batch buffers
     - Queue-based streaming

**Benefits**:

- ✅ **Minimal Data Transfer**: Only indices passed, not full parameter grids
- ✅ **Independent Workers**: No synchronization needed between workers
- ✅ **Fault Tolerant**: Failed computations don't crash entire run
- ✅ **Scalable**: Efficient from 1 to 1000s of cores
- ✅ **Zero I/O Blocking**: HDF5 writes never pause computation
- ✅ **Optimal Numba**: JIT compilation overhead eliminated after warmup

Core Functions
==============

run_parametric_analysis
-----------------------

.. code-block:: python

   def run_parametric_analysis(
       input_data: Dict[str, np.ndarray],
       output_file: str,
       config: Dict[str, Any],
       compute_function: Callable,
       verbose: bool = True
   ) -> Dict[str, Any]

**Purpose**: Execute parallel parametric analysis and write results to HDF5.

This is the main entry point for parametric studies. It:

1. Prepares parameter arrays and grid shape information
2. Creates HDF5 file with pre-allocated datasets
3. Spawns parallel workers using joblib
4. Distributes linear indices to workers
5. Collects results and writes to HDF5 in batches
6. Tracks progress and success statistics

**Arguments**:

- ``input_data``: Dictionary of parameter arrays
  
  .. code-block:: python
  
     {
         'V_plasma': np.array([100, 150, 200]),
         'T_i': np.array([15, 17, 19]),
         'tau_p_T': np.array([0.1, 0.5, 1.0, 2.0]),
         # ... other parameters
     }

- ``output_file``: Path to HDF5 output file (e.g., 'results/analysis.h5')

- ``config``: Configuration dictionary containing:
  
  - ``analysis_type``: 'T_seeded' or 'lump'
  - ``method``: 'parametric'
  - ``n_jobs``: Number of parallel workers (or None for auto)
  - ``chunk_size``: Chunk size for parallel processing (or None for auto)
  - ``batch_size``: Buffer size for HDF5 writing (or None for auto)
  - ``total_time``: Simulation time in seconds (T_seeded only)
  - ``vector_length``: Number of time points (T_seeded only, default: 100)

- ``compute_function``: Function to compute single combination
  
  Must have signature: ``func(linear_index, input_arrays_flat, param_shapes_array, **kwargs) -> Dict``
  
  Examples:
  
  - ``physics.Tseeded_functions.compute_single_combination``
  - ``physics.lump_functions.compute_single_combination``

- ``verbose``: Enable progress output

**Returns**: Dictionary with statistics:

.. code-block:: python

   {
       'n_combinations': 1000,
       'n_success': 873,
       'success_rate': 87.3,
       'computation_time': 45.2,
       'output_file': '/path/to/results.h5'
   }

**HDF5 File Structure**:

The output HDF5 file contains:

.. code-block:: text

   results.h5
   ├── Attributes (metadata)
   │   ├── analysis_type: "T_seeded"
   │   ├── total_combinations: 1000
   │   ├── successful_computations: 873
   │   ├── computation_time: 45.2
   │   └── ...
   ├── Datasets (results, one per output field)
   │   ├── V_plasma: [n_combinations] float64
   │   ├── T_i: [n_combinations] float64
   │   ├── t_startup: [n_combinations] float64
   │   ├── Q_DD: [n_combinations] float64
   │   ├── N_ofc: [n_combinations, vector_length] float64
   │   ├── sol_success: [n_combinations] bool
   │   ├── error: [n_combinations] object (strings)
   │   └── ...
   └── parameters/ (input parameter grid)
       ├── V_plasma: [3] float64
       ├── T_i: [3] float64
       ├── tau_p_T: [4] float64
       └── ...

**Example**:

.. code-block:: python

   from utils.parametric_computation import run_parametric_analysis
   from physics.Tseeded_functions import compute_single_combination
   
   # Prepare input data
   input_data = {
       'V_plasma': np.linspace(100, 200, 10),
       'T_i': np.linspace(15, 20, 5),
       'tau_p_T': np.logspace(-1, 0, 20),
       # ... 10 more parameters
   }
   
   # Configuration
   config = {
       'analysis_type': 'T_seeded',
       'method': 'parametric',
       'total_time': 10*365*24*3600,  # 10 years
       'vector_length': 100,
       'n_jobs': 11,              # Auto-detected
       'chunk_size': 5500,        # Auto-calculated
       'batch_size': 100          # Auto-calculated
   }
   
   # Run analysis (parallel execution automatic)
   stats = run_parametric_analysis(
       input_data=input_data,
       output_file='results.h5',
       config=config,
       compute_function=compute_single_combination,
       verbose=True
   )
   
   # Output:
   # ============================================================
   # STARTING PARAMETRIC ANALYSIS
   # ============================================================
   # Total combinations: 10,000
   # Parallel workers: 11
   # Chunk size: 5500
   # Batch size: 100
   # ============================================================
   #
   # Computing: 100%|████████| 10000/10000 [02:15<00:00, 73.9comb/s, Success=87.5%]
   #
   # ============================================================
   # PARAMETRIC ANALYSIS COMPLETE
   # ============================================================
   # ✅ Processed: 10,000 combinations
   # ✅ Successful: 8,750 (87.5%)
   # ⏱️  Computation time: 135.2 seconds
   # 📁 Results saved to: results.h5
   # 💾 File size: 82.3 MB
   # ============================================================

print_parametric_summary
------------------------

.. code-block:: python

   def print_parametric_summary(
       output_file: str,
       stats: Dict[str, Any],
       verbose: bool = True
   ) -> None

**Purpose**: Print formatted summary of parametric analysis results.

Displays statistics including:

- Total combinations processed
- Success count and rate
- Computation time
- Output file path and size

**Arguments**:

- ``output_file``: Path to HDF5 results file
- ``stats``: Statistics dictionary from run_parametric_analysis
- ``verbose``: Whether to print (False for silent operation)

Helper Functions
================

_write_results_to_hdf5
-----------------------

.. code-block:: python

   def _write_results_to_hdf5(
       datasets: Dict[str, Any],
       results: List[Dict],
       indices: List[int],
       data_fields: List[str],
       vector_fields: List[str],
       vector_length: int
   ) -> None

**Purpose**: Internal function for buffered HDF5 writing.

Writes accumulated results to HDF5 datasets. Handles:

- Scalar fields (1D arrays)
- Vector fields (2D arrays with time dimension)
- String fields (error messages)
- Boolean fields (success flags)
- Missing values (NaN/empty string defaults)

**Async I/O Implementation**:

The module uses a producer-consumer pattern for non-blocking HDF5 writes:

.. code-block:: python

   # Main computation thread (producer):
   while computing:
       result = compute_next()
       buffer.append(result)
       
       if len(buffer) >= 1000:
           write_queue.put(buffer.copy())  # Non-blocking enqueue
           buffer.clear()
   
   # Background writer thread (consumer):
   def writer_thread():
       while not done or not queue.empty():
           batch = write_queue.get(timeout=0.1)
           _write_results_to_hdf5(...)
           h5file.flush()

**Key Design Decisions**:

1. **Unbounded Queue**: Never blocks computation, trades memory for speed
2. **Graceful Shutdown**: Uses poison pill pattern to signal completion
3. **Error Handling**: Writer errors logged but don't crash computation
4. **Thread Safety**: HDF5 writes serialized by background thread

**Buffering Strategy**:

- Results accumulate in memory buffer
- Buffer flushed when reaching ``batch_size`` (typically 1000)
- Queue transfers buffer ownership to writer thread (via copy)
- Reduces HDF5 write operations by ~1000x
- Computation and I/O happen concurrently
- Significant performance improvement for large runs (20-30% faster)

Performance Tuning
==================

System Profiler Integration
----------------------------

The module integrates with ``utils.system_profiler`` to automatically determine optimal parameters:

.. code-block:: python

   from utils.system_profiler import get_optimal_parameters
   
   # Auto-detect optimal settings
   optimal = get_optimal_parameters(
       analysis_type='T_seeded',
       method='parametric'
   )
   
   # Returns:
   {
       'n_jobs': 11,        # CPU cores - 1
       'chunk_size': 5500,  # Based on RAM and cores
       'batch_size': 100    # HDF5 write buffer
   }

Manual Tuning
-------------

**n_jobs** (Parallel Workers):

- Rule of thumb: ``n_cores - 1``
- Leave one core for system tasks
- More workers = faster (up to core count)
- Beyond core count = diminishing returns

**chunk_size** (Work Distribution):

- Controls granularity of work distribution
- Larger = less overhead, less load balancing
- Smaller = better load balancing, more overhead
- Typical: 1000-10000 per worker

**batch_size** (HDF5 Buffer):

- Number of results before writing to HDF5
- Larger = fewer writes, more memory
- Smaller = more writes, less memory
- Typical: 100-1000

**Example**:

.. code-block:: python

   # For memory-constrained system
   config['batch_size'] = 50     # Smaller buffer
   
   # For fast I/O system
   config['batch_size'] = 2000   # Larger buffer
   
   # For very large parameter space
   config['chunk_size'] = 10000  # Bigger chunks

Performance Benchmarks
======================

**Typical Performance** (Intel i7, 12 cores, 16GB RAM):

+-------------------+----------------+------------------+-------------------+----------------------+
| Combinations      | Analysis Type  | Time             | Rate              | Improvement vs Old   |
+===================+================+==================+===================+======================+
| 256               | T_seeded       | 8 seconds        | 32 comb/s         | +28% faster          |
+-------------------+----------------+------------------+-------------------+----------------------+
| 1,000             | T_seeded       | 30 seconds       | 33 comb/s         | +32% faster          |
+-------------------+----------------+------------------+-------------------+----------------------+
| 10,000            | T_seeded       | 5 minutes        | 33 comb/s         | +27% faster          |
+-------------------+----------------+------------------+-------------------+----------------------+
| 1,000             | lump           | 3 seconds        | 333 comb/s        | +33% faster          |
+-------------------+----------------+------------------+-------------------+----------------------+
| 10,000            | lump           | 30 seconds       | 333 comb/s        | +33% faster          |
+-------------------+----------------+------------------+-------------------+----------------------+

**Speedup Analysis**:

- Serial Python (no optimization): ~0.01 comb/s
- With Numba JIT: ~1 comb/s (100x)
- With 11 parallel workers (old joblib): ~25 comb/s (2500x)
- With ProcessPoolExecutor + async I/O: ~33 comb/s (3300x overall)

**Why is it Faster?**

1. **Persistent Workers**: No process creation overhead after initial spawn
2. **One-time Numba Compilation**: JIT happens once per worker, not per batch
3. **Async I/O**: HDF5 writes happen in background, never block computation
4. **Better Task Scheduling**: Futures allow out-of-order completion for better load balancing

Error Handling
==============

The module handles failures gracefully:

**Failed Computations**:

- Recorded in ``sol_success`` dataset (False)
- Error message stored in ``error`` dataset
- Other fields filled with NaN or empty values
- Analysis continues with remaining combinations

**Common Failure Modes**:

1. **ODE Integration Failure** (T_seeded):
   
   - Stiff system convergence issues
   - Unphysical parameter combinations
   - Solution: Adjust ODE tolerances or parameter bounds

2. **Timeout** (T_seeded):
   
   - D-T not reached within total_time
   - Solution: Increase total_time or adjust parameters

3. **Impossible Startup** (lump):
   
   - Insufficient tritium breeding
   - Solution: Increase TBR or reduce target inventory

**Debugging**:

.. code-block:: python

   import h5py
   
   # Find failed runs
   with h5py.File('results.h5', 'r') as f:
       success = f['sol_success'][:]
       errors = f['error'][:]
       
       failed_indices = np.where(~success)[0]
       print(f"Failed: {len(failed_indices)} runs")
       
       # Most common errors
       unique_errors = np.unique(errors[failed_indices])
       for err in unique_errors:
           count = np.sum(errors == err)
           print(f"  {count:4d}: {err}")

See Also
========

- :doc:`physics_tseeded` - T_seeded compute function
- :doc:`physics_lump` - Lump compute function
- :doc:`system_profiler` - Performance optimization
- :doc:`../user_guide/running_analysis` - Usage guide
