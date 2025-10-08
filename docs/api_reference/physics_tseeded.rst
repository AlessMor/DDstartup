==================================
T_seeded Analysis Module
==================================

The ``physics.Tseeded_functions`` module implements time-resolved ODE-based simulation of DD startup with tritium breeding and inventory tracking.

Overview
========

The T_seeded approach models the detailed evolution of tritium through different fuel cycle stages:

- **Out-of-fuel-cycle inventory** (N_ofc): Tritium in breeding blankets
- **In-fuel-cycle inventory** (N_ifc): Tritium in processing systems  
- **Stored inventory** (N_st): Tritium available for injection
- **Plasma tritium density** (n_T): Tritium concentration in plasma

The system evolves from pure DD operation (n_T = 0) until D-T operation is reached (n_T = n_D = 0.5*n_tot).

Core Functions
==============

compute_single_combination
--------------------------

.. code-block:: python

   def compute_single_combination(
       linear_index: int,
       input_arrays_flat: List[np.ndarray],
       param_shapes_array: np.ndarray,
       total_time: float = 10*365*24*3600,
       vector_length: int = 100
   ) -> Dict[str, Any]

**Purpose**: Compute T_seeded analysis for a single parameter combination.

This is the core function called by parallel workers during parametric analysis. It:

1. Converts linear index to multi-dimensional parameter indices
2. Extracts parameter values from flattened arrays
3. Computes temperature-dependent reaction rates
4. Solves the ODE system for tritium inventory evolution
5. Computes fusion powers and economic metrics
6. Returns comprehensive results dictionary

**Arguments**:

- ``linear_index``: Integer index (0 to n_combinations-1) identifying parameter set
- ``input_arrays_flat``: List of 1D arrays, one per parameter
- ``param_shapes_array``: Array of parameter grid shapes for index conversion
- ``total_time``: Maximum simulation time in seconds (default: 10 years)
- ``vector_length``: Number of time points in output arrays (default: 100)

**Returns**: Dictionary containing:

- **Input echoes**: V_plasma, T_i, n_tot, tau_p_T, P_aux, P_aux_DT_eq, TBR_DT, TBR_DDn, tau_ifc, tau_ofc, eta_th, capacity_factor, cost_of_electricity
- **Time series** (length = vector_length):
  
  - N_ofc, N_ifc, N_stor: Tritium inventories (atoms)
  - n_T, n_D: Tritium and deuterium densities (m⁻³)
  - P_DDn, P_DDp, P_DT: Fusion powers (W)
  - TBE: Tritium breeding efficiency

- **Scalars**:
  
  - t_startup: Time to reach D-T operation (s)
  - P_DT_eq: Equivalent D-T fusion power (W)
  - Q_DD: Energy gain factor during DD startup
  - Q_DT_eq: Energy gain factor for equivalent D-T operation
  - E_lost: Energy lost due to DD startup (J)
  - unrealized_gains: Economic cost of startup ($)

- **Status**:
  
  - linear_index: Echo of input index
  - sol_success: Boolean indicating successful computation
  - error: Error message if computation failed (empty string if successful)

**Example**:

.. code-block:: python

   from physics.Tseeded_functions import compute_single_combination
   import numpy as np
   
   # Prepare inputs
   linear_index = 0
   input_arrays_flat = [
       np.array([150.0]),      # V_plasma
       np.array([17.0]),       # T_i
       np.array([1.7e20]),     # n_tot
       np.array([0.1, 1.0]),   # tau_p_T (2 values)
       # ... other parameters
   ]
   param_shapes_array = np.array([1, 1, 1, 2, ...])
   
   # Compute
   result = compute_single_combination(
       linear_index, 
       input_arrays_flat, 
       param_shapes_array,
       total_time=10*365*24*3600,
       vector_length=100
   )
   
   # Access results
   print(f"Startup time: {result['t_startup']/86400:.1f} days")
   print(f"Q factor: {result['Q_DD']:.2f}")
   print(f"Success: {result['sol_success']}")

solve_ode_system
----------------

.. code-block:: python

   def solve_ode_system(
       total_time: float,
       V_plasma: float, n_tot: float, tau_p_T: float,
       P_aux: float, P_aux_DT_eq: float,
       TBR_DT: float, TBR_DDn: float,
       tau_ifc: float, tau_ofc: float,
       eta_th: float, capacity_factor: float, cost_of_electricity: float,
       sigmav_DD_p: float, sigmav_DD_n: float, sigmav_DT: float,
       injection_rate_max: float, vector_length: int,
       N_st_min: float = 0.001/tritium_mass
   ) -> Dict[str, Any]

**Purpose**: Solve tritium inventory ODE system and compute startup metrics.

Integrates the coupled ODEs for tritium evolution until D-T operation is reached (n_T = 0.5*n_tot) or total_time is exceeded. Handles three termination conditions:

1. **Success**: D-T operation reached
2. **Timeout**: Total time exceeded without reaching D-T
3. **Failure**: Negative inventories or integration errors

**Integration Method**: BDF (Backward Differentiation Formula) - suitable for stiff systems

**Event Detection**:

- ``DT_reached_event``: Triggers when n_T = 0.5*n_tot
- ``negative_event``: Triggers if any inventory becomes negative

**Returns**: Dictionary with time series arrays and scalar metrics (see compute_single_combination for details)

ode_system
----------

.. code-block:: python

   @njit(cache=True)
   def ode_system(
       t: float, y: np.ndarray,
       V_plasma: float, n_tot: float, tau_p_T: float,
       TBR_DT: float, TBR_DDn: float,
       tau_ifc: float, tau_ofc: float,
       sigmav_DD_p: float, sigmav_DD_n: float, sigmav_DT: float,
       injection_rate_max: float, N_st_min: float
   ) -> np.ndarray

**Purpose**: Numba-compiled ODE system for maximum performance.

Computes time derivatives for the four state variables:

.. math::

   \\frac{dN_{ofc}}{dt} &= \\dot{T}_{DT} + \\dot{T}_{DDn} - \\frac{N_{ofc}}{\\tau_{ofc}} - \\lambda_T N_{ofc} \\\\
   \\frac{dN_{ifc}}{dt} &= \\frac{N_{ofc}}{\\tau_{ofc}} - \\frac{N_{ifc}}{\\tau_{ifc}} - \\lambda_T N_{ifc} + \\frac{n_T}{\\tau_{p,T}} V_{plasma} \\\\
   \\frac{dN_{st}}{dt} &= \\frac{N_{ifc}}{\\tau_{ifc}} - \\lambda_T N_{st} - \\dot{I}_{inj} \\\\
   \\frac{dn_T}{dt} &= \\frac{\\dot{I}_{inj}}{V_{plasma}} + \\frac{\\dot{T}_{DDp}}{V_{plasma}} - \\frac{n_T}{\\tau_{p,T}} - \\frac{\\dot{T}_{burn}}{V_{plasma}}

where:

- :math:`\\dot{T}_{DDn} = TBR_{DDn} \\cdot 0.5 n_D^2 \\langle\\sigma v\\rangle_{DDn} V_{plasma}` (breeding from DD-n)
- :math:`\\dot{T}_{DDp} = 0.5 n_D^2 \\langle\\sigma v\\rangle_{DDp} V_{plasma}` (production from DD-p)
- :math:`\\dot{T}_{DT} = TBR_{DT} \\cdot n_D n_T \\langle\\sigma v\\rangle_{DT} V_{plasma}` (breeding from D-T)
- :math:`\\dot{T}_{burn} = n_D n_T \\langle\\sigma v\\rangle_{DT} V_{plasma}` (consumption by D-T)
- :math:`\\dot{I}_{inj} = \\min(\\max(0, N_{ifc}/\\tau_{ifc} - \\lambda_T N_{st}), \\dot{I}_{max})` (injection rate)

**Compilation**: JIT-compiled with Numba for ~100x speedup. Cached after first run.

postprocess_fusion_results_Tseeded
-----------------------------------

.. code-block:: python

   @njit(cache=True)
   def postprocess_fusion_results_Tseeded(
       t_startup: float, N_ofc: np.ndarray, N_ifc: np.ndarray,
       N_st: np.ndarray, n_T: np.ndarray, n_tot: float,
       V_plasma: float, sigmav_DD_p: float, sigmav_DD_n: float,
       sigmav_DT: float, TBR_DT: float, TBR_DDn: float,
       tau_ifc: float, eta_th: float, capacity_factor: float,
       cost_of_electricity: float, P_aux: float, P_aux_DT_eq: float,
       E_DDn: float, E_DDp: float, E_DT: float,
       injection_rate_max: float, N_st_min: float, vector_length: int
   ) -> Tuple

**Purpose**: JIT-compiled postprocessing for performance.

Computes fusion powers, energy integrals, Q factors, and economic metrics from ODE solution. Uses trapezoidal integration over time to compute total energies.

**Returns**: Tuple of (P_DDn, P_DDp, P_DT, P_DT_eq, Q_DD, Q_DT_eq, E_lost, unrealized_gains, TBE_vector, n_D)

Parallel Computation
====================

The T_seeded module is designed for efficient parallel execution:

**Data Layout**:

- Parameters stored as 1D arrays (flattened)
- Linear indexing: 0, 1, 2, ..., n_combinations-1
- Minimal data transfer between processes

**Worker Execution**:

1. Each worker receives ``compute_single_combination``
2. Worker converts linear_index → multi-dimensional indices
3. Worker extracts parameters and runs physics
4. Worker returns result dictionary
5. Main process collects and writes to HDF5

**Performance**:

- Numba JIT compilation: ~100x speedup for ODE system
- Parallel workers: ~10x speedup (11 cores)
- Overall: ~1000x faster than naive Python

**Memory Efficiency**:

- Shared read-only parameter arrays
- Each worker holds only one result at a time
- Buffered HDF5 writing (1000 results)

Example Workflow
================

.. code-block:: python

   # 1. Prepare parameter grid
   from utils.io_functions import prepare_input_data
   
   param_fields = {
       'V_plasma': np.linspace(100, 200, 5),
       'T_i': np.linspace(14, 20, 3),
       'tau_p_T': np.logspace(-1, 0, 10),
       # ... other parameters
   }
   
   input_data = prepare_input_data(param_fields, 'T_seeded')
   
   # 2. Run parametric analysis (parallel execution happens automatically)
   from utils.parametric_computation import run_parametric_analysis
   from physics.Tseeded_functions import compute_single_combination
   
   stats = run_parametric_analysis(
       input_data=input_data,
       output_file='results.h5',
       config={'total_time': 10*365*24*3600, 'n_jobs': 11},
       compute_function=compute_single_combination,
       verbose=True
   )
   
   # 3. Access results
   import h5py
   with h5py.File('results.h5', 'r') as f:
       t_startup = f['t_startup'][:]
       Q_DD = f['Q_DD'][:]
       success = f['sol_success'][:]

Performance Notes
=================

**Typical Performance** (256 combinations, 11 workers):

- Total time: ~10 seconds
- Per combination: ~40 ms
- Success rate: ~85-90%

**Optimization Tips**:

1. Use ``vector_length=100`` for balance of detail vs. speed
2. Set ``total_time`` appropriately (10 years typical)
3. Let system profiler auto-detect ``n_jobs``
4. Use HDF5 compression for large parameter sweeps

**Common Failure Modes**:

- **Timeout**: D-T not reached within total_time
- **Negative inventories**: Unphysical parameter combinations
- **Integration errors**: Stiff ODE solver issues

See Also
========

- :doc:`physics_lump` - Simplified lumped parameter model
- :doc:`parametric_computation` - Parallel execution framework
- :doc:`../user_guide/analysis_types` - Choosing between T_seeded and lump
