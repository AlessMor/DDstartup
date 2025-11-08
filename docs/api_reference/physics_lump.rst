==================================
Lump Parameter Model
==================================

The ``physics.lump_functions`` module implements a simplified steady-state model for DD startup analysis.

Overview
========

The lump model considers a reactor operating in a pure deuterium environment with tritium breeding.
Tritium is produced by DD-p reactions (and partially consumed via D-T reactions) and by tritium 
breeding in the blanket (from neutrons produced by DD-n and D-T reactions). The model accounts 
for radioactive decay of tritium during the startup phase.

**Key Assumptions**:

1. Tritium and He-3 reach steady-state densities immediately
2. Deuterium density remains constant (n_D ≈ n_tot) - better approximation would find a value of n_D accounting for n_T and n_He3 production
3. Power output is constant at steady-state values
4. No time evolution of inventories (algebraic solution)

Core Functions
==============

lump_numba
----------

.. code-block:: python

   @njit(cache=True)
   def lump_numba(
       V_plasma: float, n_tot: float,
       tau_p_T: float, tau_p_He3: float,
       TBR_DT: float, TBR_DDn: float, I_target: float,
       sigmav_DD_p: float, sigmav_DD_n: float,
       sigmav_DT: float, sigmav_DHe3: float
   ) -> Tuple[float, float, float, float, bool]

**Purpose**: JIT-compiled lumped parameter calculation for steady-state DD startup.

Solves for equilibrium particle densities and startup time using steady-state balance equations:

**Steady-State Densities**:

.. math::

   n_T &= \\frac{0.5 n_D^2 \\langle\\sigma v\\rangle_{DDp}}{n_D \\langle\\sigma v\\rangle_{DT} + 1/\\tau_{p,T}} \\\\
   n_{He3} &= \\frac{0.5 n_D^2 \\langle\\sigma v\\rangle_{DDn}}{n_D \\langle\\sigma v\\rangle_{DHe3} + 1/\\tau_{p,He3}}

where :math:`n_D \\approx n_{tot}` (pure deuterium assumption during DD phase).

**Tritium Production Rates**:

.. math::

   \\dot{T}_{DDn} &= TBR_{DDn} \\cdot 0.5 n_D^2 \\langle\\sigma v\\rangle_{DDn} V_{plasma} \\\\
   \\dot{T}_{DDp} &= 0.5 n_D^2 \\langle\\sigma v\\rangle_{DDp} V_{plasma} \\\\
   \\dot{T}_{DT} &= TBR_{DT} \\cdot n_D n_T \\langle\\sigma v\\rangle_{DT} V_{plasma}

**Startup Time**:

.. math::

   t_{startup} = -\\frac{1}{\\lambda_T} \\ln\\left(1 - \\frac{N_{ST} \\lambda_T}{\\dot{T}_{tot}}\\right)

where :math:`N_{ST} = I_{target} / m_T` is the target inventory in atoms and :math:`\\dot{T}_{tot} = \\dot{T}_{DDn} + \\dot{T}_{DDp} + \\dot{T}_{DT}`.

**Returns**: ``Tuple[float, float, float, float, bool]``

- ``n_T``: Steady-state tritium density (m⁻³)
- ``n_D``: Deuterium density (m⁻³) - equals n_tot
- ``n_He3``: Steady-state helium-3 density (m⁻³)
- ``t_startup``: Time to reach target inventory (s)
- ``sol_success``: True if physically valid solution found

**Failure Conditions**: 

- If :math:`N_{ST} \\lambda_T / \\dot{T}_{tot} \\geq 1`: Startup impossible (decay dominates production)
- If denominators near zero (< 1e-20): Unphysical parameters
- Returns ``t_startup = np.inf`` and ``sol_success = False``


lump_solver
-----------

.. code-block:: python

   def lump_solver(
       V_plasma: float, n_tot: float,
       tau_p_T: float, tau_p_He3: float,
       TBR_DT: float, TBR_DDn: float, I_target: float,
       sigmav_DD_p: float, sigmav_DD_n: float,
       sigmav_DT: float, sigmav_DHe3: float
   ) -> Dict[str, Any]

**Purpose**: Python wrapper for lump_numba that returns results as a dictionary.

Converts the tuple output from lump_numba into a dictionary format for easier result handling.

**Returns**: ``Dict[str, Any]`` with keys:

- ``n_T``: Tritium density (m⁻³) - **scalar**
- ``n_D``: Deuterium density (m⁻³) - **scalar**
- ``n_He3``: Helium-3 density (m⁻³) - **scalar**
- ``t_startup``: Startup time (s) - **scalar**
- ``sol_success``: Success flag (bool) - **scalar**
- ``error``: Error message if failed, None otherwise (str or None) - **scalar**

**Note**: Unlike T_seeded which returns time-series arrays, lump returns only scalar values
since it uses steady-state approximation.

**Example**:

.. code-block:: python

   from ddstartup.physics.lump_functions import lump_solver
   from ddstartup.physics.reactivity_functions import (
       sigmav_DT_BoschHale, sigmav_DD_BoschHale, sigmav_DHe3_BoschHale
   )
   import numpy as np
   
   # Define parameters
   V_plasma = 150.0  # m³
   n_tot = 2e20      # m⁻³
   T_i = 14.0        # keV
   tau_p_T = 1.0     # s
   tau_p_He3 = 1.0   # s
   TBR_DT = 1.05
   TBR_DDn = 0.5
   I_target = 1.0    # kg
   
   # Compute reaction rates at temperature T_i
   sigmav_DT = sigmav_DT_BoschHale(np.array([T_i]))[0]
   _, sigmav_DD_p, sigmav_DD_n = sigmav_DD_BoschHale(np.array([T_i]))
   sigmav_DD_p = sigmav_DD_p[0]
   sigmav_DD_n = sigmav_DD_n[0]
   sigmav_DHe3 = sigmav_DHe3_BoschHale(np.array([T_i]))[0]
   
   # Solve
   result = lump_solver(
       V_plasma=V_plasma, n_tot=n_tot,
       tau_p_T=tau_p_T, tau_p_He3=tau_p_He3,
       TBR_DT=TBR_DT, TBR_DDn=TBR_DDn, I_target=I_target,
       sigmav_DD_p=sigmav_DD_p, sigmav_DD_n=sigmav_DD_n,
       sigmav_DT=sigmav_DT, sigmav_DHe3=sigmav_DHe3
   )
   
   # Check results
   if result['sol_success']:
       print(f"Startup time: {result['t_startup']/86400:.1f} days")
       print(f"Tritium density: {result['n_T']:.2e} m⁻³")
       print(f"He-3 density: {result['n_He3']:.2e} m⁻³")
   else:
       print(f"Solution failed: {result['error']}")

Model Comparison
================

T_seeded vs Lump
----------------

+------------------+---------------------------+---------------------------+
| Feature          | T_seeded                  | Lump                      |
+==================+===========================+===========================+
| **Method**       | ODE integration           | Algebraic equations       |
+------------------+---------------------------+---------------------------+
| **Time detail**  | Full evolution            | No time series            |
+------------------+---------------------------+---------------------------+
| **Complexity**   | Complex (4 ODEs, events)  | Simple (direct calc)      |
+------------------+---------------------------+---------------------------+

See Also
========

- :doc:`physics_tseeded` - Time-resolved ODE-based model
- :doc:`physics_reactivity` - Reaction rate functions
