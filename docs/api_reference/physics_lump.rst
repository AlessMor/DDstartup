==================================
Lump Parameter Model
==================================

The ``physics.lump_functions`` module implements a simplified steady-state model for DD startup analysis.

Overview
========

The lump model uses algebraic equations and steady-state assumptions instead of solving ODEs, making it:

- **Faster**: ~10x quicker than T_seeded
- **Simpler**: No ODE integration, direct calculations
- **Less detailed**: No time evolution, only final equilibrium

**Key Assumptions**:

1. Tritium and He-3 reach steady-state densities immediately
2. Tritium inventory grows exponentially with radioactive decay
3. Power output is constant at steady-state values

**Use Cases**:

- Quick parameter scans
- Sensitivity analysis where time evolution isn't critical
- Validation of T_seeded results
- Preliminary design studies

Core Functions
==============

compute_single_combination
--------------------------

.. code-block:: python

   def compute_single_combination(
       linear_index: int,
       input_arrays_flat: List[np.ndarray],
       param_shapes_array: np.ndarray
   ) -> Dict[str, Any]

**Purpose**: Compute lump analysis for a single parameter combination.

This is the core function called by parallel workers during parametric analysis. It:

1. Converts linear index to parameter indices
2. Extracts parameter values
3. Computes temperature-dependent reaction rates
4. Runs steady-state lump model
5. Returns results dictionary

**Arguments**:

- ``linear_index``: Integer index (0 to n_combinations-1) identifying parameter set
- ``input_arrays_flat``: List of 1D arrays, one per parameter
- ``param_shapes_array``: Array of parameter grid shapes for index conversion

**Returns**: Dictionary containing:

- **Input echoes**: V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, P_aux, P_aux_DT_eq, TBR_DT, TBR_DDn, I_target, eta_th, capacity_factor, cost_of_electricity

- **Results** (all scalars):
  
  - n_T: Steady-state tritium density (m⁻³)
  - n_D: Deuterium density (m⁻³)
  - n_He3: Steady-state helium-3 density (m⁻³)
  - t_startup: Time to reach target inventory (s)
  - P_DDn, P_DDp: DD fusion powers (W)
  - P_DT: D-T fusion power at steady-state (W)
  - P_DT_eq: Equivalent D-T power for comparison (W)
  - Q_DD: Energy gain factor during DD startup
  - Q_DT_eq: Energy gain factor for equivalent D-T
  - E_lost: Energy lost due to DD startup (J)
  - unrealized_gains: Economic cost of startup ($)

- **Status**:
  
  - linear_index: Echo of input index
  - sol_success: Boolean (False if startup impossible)

**Example**:

.. code-block:: python

   from physics.lump_functions import compute_single_combination
   import numpy as np
   
   # Prepare inputs
   linear_index = 0
   input_arrays_flat = [
       np.array([150.0]),      # V_plasma
       np.array([17.0]),       # T_i
       np.array([1.7e20]),     # n_tot
       np.array([0.5]),        # tau_p_T
       np.array([1.0]),        # tau_p_He3
       np.array([60e6]),       # P_aux
       np.array([60e6]),       # P_aux_DT_eq
       np.array([1.1]),        # TBR_DT
       np.array([0.7]),        # TBR_DDn
       np.array([1.0]),        # I_target (kg)
       np.array([0.35]),       # eta_th
       np.array([0.7]),        # capacity_factor
       np.array([7e-8]),       # cost_of_electricity ($/J)
   ]
   param_shapes_array = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
   
   # Compute
   result = compute_single_combination(
       linear_index, 
       input_arrays_flat, 
       param_shapes_array
   )
   
   # Access results
   print(f"Startup time: {result['t_startup']/86400:.1f} days")
   print(f"Steady-state n_T: {result['n_T']:.2e} m⁻³")
   print(f"Q factor: {result['Q_DD']:.2f}")

lump_numba
----------

.. code-block:: python

   @njit(cache=True)
   def lump_numba(
       V_plasma: float, n_tot: float,
       tau_p_T: float, tau_p_He3: float,
       P_aux: float, P_aux_DT_eq: float,
       TBR_DT: float, TBR_DDn: float, I_target: float,
       eta_th: float, capacity_factor: float, cost_of_electricity: float,
       sigmav_DD_p: float, sigmav_DD_n: float,
       sigmav_DT: float, sigmav_DHe3: float
   ) -> Tuple

**Purpose**: JIT-compiled lumped parameter calculation.

Uses steady-state balance equations to compute:

**Steady-State Densities**:

.. math::

   n_T &= \\frac{0.5 n_D^2 \\langle\\sigma v\\rangle_{DDp}}{n_D \\langle\\sigma v\\rangle_{DT} + 1/\\tau_{p,T}} \\\\
   n_{He3} &= \\frac{0.5 n_D^2 \\langle\\sigma v\\rangle_{DDn}}{n_D \\langle\\sigma v\\rangle_{DHe3} + 1/\\tau_{p,He3}}

**Startup Time**:

.. math::

   t_{startup} = -\\frac{1}{\\lambda_T} \\ln\\left(1 - \\frac{N_{ST} \\lambda_T}{\\dot{T}_{tot}}\\right)

where :math:`N_{ST}` is the target inventory and :math:`\\dot{T}_{tot}` is the total tritium production rate.

**Fusion Powers**:

.. math::

   P_{f,DDp} &= n_D^2 \\langle\\sigma v\\rangle_{DDp} V_{plasma} E_{DDp} \\\\
   P_{f,DDn} &= n_D^2 \\langle\\sigma v\\rangle_{DDn} V_{plasma} E_{DDn} \\\\
   P_{f,DT} &= n_D n_T \\langle\\sigma v\\rangle_{DT} V_{plasma} E_{DT} \\\\
   P_{f,DHe3} &= n_D n_{He3} \\langle\\sigma v\\rangle_{DHe3} V_{plasma} E_{DHe3}

**Returns**: Tuple of (n_T, n_D, n_He3, t_startup, Pf_DDn, Pf_DDp, Pf_DD_DT, Pf_DT_eq, Q_DD, Q_DT_eq, E_lost, unrealized_gains, sol_success)

**Failure Condition**: If :math:`N_{ST} \\lambda_T / \\dot{T}_{tot} \\geq 1`, startup is impossible and ``t_startup = inf``, ``sol_success = False``.

lump_solver
-----------

.. code-block:: python

   def lump_solver(
       V_plasma: float, n_tot: float,
       tau_p_T: float, tau_p_He3: float,
       P_aux: float, P_aux_DT_eq: float,
       TBR_DT: float, TBR_DDn: float, I_target: float,
       eta_th: float, capacity_factor: float, cost_of_electricity: float,
       sigmav_DD_p: float, sigmav_DD_n: float,
       sigmav_DT: float, sigmav_DHe3: float
   ) -> Dict[str, Any]

**Purpose**: Wrapper for lump_numba that returns dictionary format.

Converts the tuple output from lump_numba into a standardized dictionary matching the compute_single_combination interface.

compute_lump_batch
------------------

.. code-block:: python

   def compute_lump_batch(
       V_plasma: np.ndarray, T_i: np.ndarray, n_tot: np.ndarray,
       tau_p_T: np.ndarray, tau_p_He3: np.ndarray,
       P_aux: np.ndarray, P_aux_DT_eq: np.ndarray,
       TBR_DT: np.ndarray, TBR_DDn: np.ndarray, I_target: np.ndarray,
       eta_th: np.ndarray, capacity_factor: np.ndarray,
       cost_of_electricity: np.ndarray
   ) -> Dict[str, np.ndarray]

**Purpose**: Vectorized computation for multiple parameter combinations.

Alternative to looping compute_single_combination. All inputs are arrays of the same length, and results are returned as arrays.

**Use Case**: When you have arrays of parameters and want to process them in batch without the parallel framework overhead.

**Performance**: Slightly faster than compute_single_combination in loop due to vectorized reaction rate calculations.

Model Comparison
================

T_seeded vs Lump
----------------

+------------------+---------------------------+---------------------------+
| Feature          | T_seeded                  | Lump                      |
+==================+===========================+===========================+
| **Method**       | ODE integration           | Algebraic equations       |
+------------------+---------------------------+---------------------------+
| **Time detail**  | Full evolution (100 pts)  | No time series            |
+------------------+---------------------------+---------------------------+
| **Speed**        | ~40 ms/combination        | ~4 ms/combination         |
+------------------+---------------------------+---------------------------+
| **Accuracy**     | High (tracks transients)  | Medium (steady-state)     |
+------------------+---------------------------+---------------------------+
| **Complexity**   | Complex (4 ODEs, events)  | Simple (direct calc)      |
+------------------+---------------------------+---------------------------+
| **Use case**     | Detailed analysis         | Quick scans               |
+------------------+---------------------------+---------------------------+

**Recommendation**: 

- Use **T_seeded** for final analysis and when time evolution matters
- Use **lump** for initial exploration and sensitivity studies

Parallel Computation
====================

The lump model uses the same parallel framework as T_seeded:

**Execution Flow**:

1. Main process prepares flattened parameter arrays
2. ``joblib.Parallel`` spawns worker processes
3. Each worker calls ``compute_single_combination(index, ...)``
4. Worker computes reaction rates and runs ``lump_numba``
5. Results collected and written to HDF5

**Performance**:

- Numba JIT compilation: ~50x speedup
- 11 parallel workers: ~10x speedup
- Overall: ~500x faster than naive Python

Example Workflow
================

.. code-block:: python

   # 1. Prepare parameter grid
   from utils.io_functions import prepare_input_data
   
   param_fields = {
       'V_plasma': np.linspace(100, 200, 10),
       'T_i': np.linspace(14, 20, 5),
       'tau_p_T': np.logspace(-1, 0, 20),
       'tau_p_He3': np.array([1.0]),
       'I_target': np.linspace(0.5, 2.0, 10),
       # ... other parameters
   }
   
   input_data = prepare_input_data(param_fields, 'lump')
   
   # 2. Run parametric analysis (parallel)
   from utils.parametric_computation import run_parametric_analysis
   from physics.lump_functions import compute_single_combination
   
   stats = run_parametric_analysis(
       input_data=input_data,
       output_file='lump_results.h5',
       config={'n_jobs': 11},
       compute_function=compute_single_combination,
       verbose=True
   )
   
   # 3. Quick batch computation alternative
   from physics.lump_functions import compute_lump_batch
   
   # Arrays of parameters (all same length)
   results = compute_lump_batch(
       V_plasma=np.array([100, 150, 200]),
       T_i=np.array([15, 17, 19]),
       n_tot=np.array([1.5e20, 1.7e20, 1.9e20]),
       # ... other parameters
   )
   
   # Results are arrays
   print(f"Startup times: {results['t_startup']/86400}")  # days

Performance Notes
=================

**Typical Performance** (10,000 combinations, 11 workers):

- Total time: ~40 seconds
- Per combination: ~4 ms
- Success rate: ~95-98% (higher than T_seeded)

**When Lump Fails**:

- Target inventory too large: :math:`N_{ST} \\lambda_T / \\dot{T}_{tot} \\geq 1`
- Insufficient breeding: TBR values too low
- Returns ``t_startup = inf`` and ``sol_success = False``

**Optimization Tips**:

1. Use batch mode for quick checks (no HDF5 overhead)
2. Lump is ideal for Sobol analysis (many samples needed)
3. Validates T_seeded results quickly

See Also
========

- :doc:`physics_tseeded` - Detailed ODE-based model
- :doc:`parametric_computation` - Parallel execution framework
- :doc:`../user_guide/analysis_types` - Choosing analysis type
