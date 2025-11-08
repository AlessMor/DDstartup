==================================
Sobol Sensitivity Analysis
==================================

The ``utils.sobol_computation`` module provides wrapper functions for Sobol sensitivity analysis.

Overview
========

Sobol analysis is a global sensitivity method that quantifies how much each input parameter contributes to output variance. It's useful for:

- **Identifying critical parameters**: Which inputs matter most?
- **Model simplification**: Which parameters can be fixed?
- **Uncertainty quantification**: How does input uncertainty propagate?

**Key Concepts**:

- **First-order indices** (S_i): Direct effect of parameter i
- **Total-order indices** (ST_i): Total effect including interactions
- **Second-order indices** (S_ij): Two-way interaction effects

Functions
=========

run_sobol_analysis
------------------

.. code-block:: python

   def run_sobol_analysis(
       input_data: Dict[str, np.ndarray],
       output_file: str,
       config: Dict[str, Any],
       compute_function: Callable,
       verbose: bool = True
   ) -> Dict[str, Any]

**Purpose**: Execute Sobol sensitivity analysis.

This function wraps the ``physics.sobol_functions.sobol_analysis`` module to provide a consistent interface with parametric analysis.

**Arguments**:

- ``input_data``: Dictionary of parameter distributions (not grids!)
  
  For Sobol, these should be ranges/distributions, not fixed grids:
  
  .. code-block:: python
  
     {
         'V_plasma': np.array([100, 200]),     # [min, max]
         'T_i': np.array([15, 20]),            # [min, max]
         'tau_p_T': np.array([0.1, 2.0]),      # [min, max]
         # ...
     }

- ``output_file``: Path for Sobol results (used by physics module)

- ``config``: Configuration dictionary containing:
  
  - ``analysis_type``: 'T_seeded' or 'lump'
  - ``method``: 'sobol' or 'lhs'
  - ``N_SAMPLES``: Number of Sobol samples (typically 1000-10000)
  - ``order``: Sobol order (1=first-order only, 2=with interactions, 3=full)
  - ``total_time``: Simulation time (T_seeded only)

- ``compute_function``: Physics function (T_seeded or lump)

- ``verbose``: Enable progress output

**Returns**: Dictionary with statistics:

.. code-block:: python

   {
       'n_samples': 10000,
       'n_parameters': 13,
       'order': 2,
       'analysis_type': 'T_seeded',
       'computation_time': 450.2
   }

**Example**:

.. code-block:: python

   from utils.sobol_computation import run_sobol_analysis
   from physics.Tseeded_functions import compute_single_combination
   
   # Parameter ranges for Sobol
   input_data = {
       'V_plasma': np.array([100, 200]),
       'T_i': np.array([15, 20]),
       'tau_p_T': np.array([0.1, 2.0]),
       'P_aux': np.array([20e6, 100e6]),
       # ... all 13 parameters with [min, max]
   }
   
   # Configuration
   config = {
       'analysis_type': 'T_seeded',
       'method': 'sobol',
       'N_SAMPLES': 5000,      # 5000 base samples
       'order': 2,             # First + second order
       'total_time': 10*365*24*3600
   }
   
   # Run Sobol analysis
   stats = run_sobol_analysis(
       input_data=input_data,
       output_file='sobol_results.txt',
       config=config,
       compute_function=compute_single_combination,
       verbose=True
   )
   
   # Output:
   # ============================================================
   # STARTING SOBOL SENSITIVITY ANALYSIS
   # ============================================================
   # Parameters: 13
   # Samples: 5,000
   # Order: 2
   # Analysis type: T_seeded
   # ============================================================
   #
   # [Sobol analysis runs...]
   #
   # ============================================================
   # SOBOL ANALYSIS COMPLETE
   # ============================================================
   # Samples evaluated: 5,000
   # Computation time: 450.2 seconds
   # Results saved to: sobol_results.txt

print_sobol_summary
-------------------

.. code-block:: python

   def print_sobol_summary(
       stats: Dict[str, Any],
       verbose: bool = True
   ) -> None

**Purpose**: Print formatted summary of Sobol analysis.

Displays:

- Number of samples evaluated
- Number of parameters analyzed
- Sobol order used
- Analysis type
- Computation time

Sobol Analysis Workflow
========================

The Sobol module delegates to ``physics.sobol_functions.sobol_analysis`` which:

1. **Sample Generation**: Creates Sobol quasi-random sequences
2. **Parameter Scaling**: Maps [0,1] samples to parameter ranges
3. **Model Evaluation**: Runs physics for each sample
4. **Index Calculation**: Computes Sobol sensitivity indices
5. **Output**: Saves indices to text file and generates plots

Sample Size Guidelines
======================

**Rule of Thumb**:

.. math::

   N_{total} = N_{samples} \\times (2p + 2)

where :math:`p` is the number of parameters.

**Example** (13 parameters):

- N_samples = 1,000 → N_total = 28,000 evaluations
- N_samples = 5,000 → N_total = 140,000 evaluations
- N_samples = 10,000 → N_total = 280,000 evaluations

**Recommendations**:

+-------------------+-------------------+-------------------------+
| N_samples         | Use Case          | Accuracy                |
+===================+===================+=========================+
| 500-1,000         | Quick screening   | Low (±0.1)              |
+-------------------+-------------------+-------------------------+
| 2,000-5,000       | Standard analysis | Medium (±0.05)          |
+-------------------+-------------------+-------------------------+
| 10,000+           | Publication       | High (±0.02)            |
+-------------------+-------------------+-------------------------+

Sobol Order Selection
=====================

**Order 1** (First-order only):

- Fastest: :math:`N(p+2)` evaluations
- Shows direct effects only
- Use for: initial screening

**Order 2** (First + second order):

- Moderate: :math:`N(2p+2)` evaluations  
- Shows direct effects + pairwise interactions
- Use for: most analyses (recommended)

**Order 3** (Full decomposition):

- Slowest: More complex sampling
- Shows all interaction effects
- Use for: detailed mechanism understanding

Interpreting Results
====================

Sobol analysis produces text output like:

.. code-block:: text

   SOBOL SENSITIVITY INDICES
   ==========================
   
   Output: t_startup (startup time)
   
   First-order indices (S_i):
     tau_p_T:              0.654 ± 0.023  ████████████████
     TBR_DT:               0.189 ± 0.015  ████
     tau_ifc:              0.087 ± 0.012  ██
     V_plasma:             0.042 ± 0.008  █
     T_i:                  0.018 ± 0.006  
     (others < 0.01)
   
   Total-order indices (ST_i):
     tau_p_T:              0.723 ± 0.019
     TBR_DT:               0.245 ± 0.017
     tau_ifc:              0.112 ± 0.014
     V_plasma:             0.058 ± 0.010
   
   Second-order indices (S_ij):
     tau_p_T × TBR_DT:     0.045 ± 0.012
     tau_p_T × tau_ifc:    0.023 ± 0.009

**Key Insights**:

1. **Most Important**: tau_p_T (65% of variance)
2. **Moderate**: TBR_DT, tau_ifc
3. **Interactions**: Small (total > first-order by ~10%)
4. **Negligible**: T_i, P_aux, etc.

**Actions**:

- Focus calibration efforts on tau_p_T
- TBR_DT and tau_ifc need good estimates
- Other parameters can use nominal values

Comparison with Parametric
===========================

+------------------+---------------------------+---------------------------+
| Feature          | Parametric                | Sobol                     |
+==================+===========================+===========================+
| **Sampling**     | Regular grid              | Quasi-random              |
+------------------+---------------------------+---------------------------+
| **Coverage**     | Full space systematically | Representative samples    |
+------------------+---------------------------+---------------------------+
| **Evaluations**  | :math:`\\prod_i n_i`      | :math:`N(2p+2)`           |
+------------------+---------------------------+---------------------------+
| **Output**       | All combinations          | Sensitivity indices       |
+------------------+---------------------------+---------------------------+
| **Use case**     | Design optimization       | Uncertainty analysis      |
+------------------+---------------------------+---------------------------+

**Example**:

- 13 parameters, 5 points each
- **Parametric**: :math:`5^{13}` = 1.2 billion evaluations
- **Sobol** (N=5000): 140,000 evaluations (~10,000x fewer!)

**When to Use Each**:

- **Parametric**: 
  
  - Finding optimal designs
  - Mapping full parameter space
  - 2-4 parameters varying

- **Sobol**:
  
  - Identifying critical parameters
  - Uncertainty propagation
  - >5 parameters

Performance Notes
=================

**Typical Performance** (T_seeded, 13 parameters):

- N=1,000: ~60 seconds (28,000 evals)
- N=5,000: ~5 minutes (140,000 evals)
- N=10,000: ~10 minutes (280,000 evals)

**Optimization**:

Sobol uses the same parallel framework as parametric:

- Workers evaluate samples in parallel
- Speedup scales with cores (up to ~10x on 12 cores)
- Lump model ~10x faster than T_seeded

Example Workflow
================

.. code-block:: python

   # 1. Define parameter ranges
   param_ranges = {
       'V_plasma': [100, 200],
       'T_i': [14, 20],
       'n_tot': [1.3e20, 2.1e20],
       'tau_p_T': [0.1, 5.0],
       'P_aux': [20e6, 100e6],
       'P_aux_DT_eq': [20e6, 100e6],
       'TBR_DT': [1.05, 1.15],
       'TBR_DDn': [0.5, 0.9],
       'tau_ifc': [3600, 86400],
       'tau_ofc': [3600, 172800],
       'eta_th': [0.3, 0.4],
       'capacity_factor': [0.5, 0.9],
       'price_of_electricity': [5e-8, 1e-7],
   }
   
   # Convert to arrays
   input_data = {k: np.array(v) for k, v in param_ranges.items()}
   
   # 2. Configure analysis
   config = {
       'analysis_type': 'T_seeded',
       'method': 'sobol',
       'N_SAMPLES': 5000,
       'order': 2,
       'total_time': 10*365*24*3600
   }
   
   # 3. Run Sobol
   from utils.sobol_computation import run_sobol_analysis
   from physics.Tseeded_functions import compute_single_combination
   
   stats = run_sobol_analysis(
       input_data, 'sobol_output.txt', config,
       compute_single_combination, verbose=True
   )
   
   # 4. Analyze results
   # Results in: sobol_output.txt and .png plots

See Also
========

- :doc:`parametric_computation` - Grid-based parameter sweep
- :doc:`physics_tseeded` - T_seeded physics model
- :doc:`physics_lump` - Lump physics model
- :doc:`../user_guide/sensitivity_analysis` - Sobol methodology
