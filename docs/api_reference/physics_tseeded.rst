==================================
T_seeded Analysis Module
==================================

The ``physics.Tseeded_functions`` module implements time-resolved ODE-based simulation of DD startup with tritium breeding and inventory tracking.

Overview
========

The T_seeded approach models the detailed evolution of tritium through different fuel cycle stages:

- **Outer fuel-cycle inventory** (N_ofc)
- **Inner fuel-cycle inventory** (N_ifc)
- **Storage inventory** (N_st): Tritium available for injection
- **Plasma tritium density** (n_T): Tritium concentration in plasma

The system evolves from pure DD operation (n_T = 0) until D-T operation is reached (n_T = n_D = 0.5*n_tot).

Core Functions
==============

solve_ode_system
----------------

.. code-block:: python

   def solve_ode_system(
       V_plasma: float, n_tot: float, tau_p_T: float,
       TBR_DT: float, TBR_DDn: float,
       tau_ifc: float, tau_ofc: float,
       sigmav_DD_p: float, sigmav_DD_n: float, sigmav_DT: float,
       injection_rate_max: float,
       total_time: float = 10*365*24*3600,
       N_st_min: float = 0.001/tritium_mass
   ) -> Dict[str, Any]

**Purpose**: Solve the ODE system for tritium inventory evolution during DD startup.

Integrates the coupled ODEs describing tritium evolution through the fuel cycle until 
D-T operation is reached (n_T = 0.5*n_tot) or total_time is exceeded.

**Integration Method**: BDF - best for stiff systems

**Event Detection**:

- ``DT_reached_event``: Triggers when n_T = 0.5*n_tot (success)
- ``negative_event``: Triggers if any inventory becomes negative (failure)

**Termination Conditions**:

1. **Success**: D-T operation reached (n_T = 0.5*n_tot)
2. **Timeout**: Total time exceeded without reaching D-T
3. **Failure**: Negative inventories or integration errors

**Returns**: ``Dict[str, Any]`` with keys:

- **Time series arrays**:
  
  - ``t``: Time points (s) - 1D array (variable length)
  - ``N_ofc``: Out-of-fuel-cycle tritium (atoms) - 1D array
  - ``N_ifc``: In-fuel-cycle tritium (atoms) - 1D array
  - ``N_stor``: Storage tritium (atoms) - 1D array
  - ``n_T``: Tritium density in plasma (m⁻³) - 1D array

- **Scalar results**:
  
  - ``t_startup``: Time to reach D-T operation (s)
  - ``sol_success``: True if D-T reached, False otherwise
  - ``error``: Error message if failed, None if successful

**Example**:

.. code-block:: python

   from ddstartup.physics.Tseeded_functions import solve_ode_system
   from ddstartup.physics.reactivity_functions import (
       sigmav_DT_BoschHale, sigmav_DD_BoschHale
   )
   import numpy as np
   
   # Define parameters
   V_plasma = 150.0  # m³
   n_tot = 2e20      # m⁻³
   T_i = 17.0        # keV
   tau_p_T = 0.5     # s
   TBR_DT = 1.1
   TBR_DDn = 0.7
   tau_ifc = 1.0 * 86400    # 1 day
   tau_ofc = 10.0 * 86400   # 10 days
   injection_rate_max = 1e20  # atoms/s
   total_time = 10 * 365 * 86400  # 10 years
   
   # Compute reaction rates
   sigmav_DT = sigmav_DT_BoschHale(np.array([T_i]))[0]
   _, sigmav_DD_p, sigmav_DD_n = sigmav_DD_BoschHale(np.array([T_i]))
   sigmav_DD_p = sigmav_DD_p[0]
   sigmav_DD_n = sigmav_DD_n[0]
   
   # Solve ODE system
   result = solve_ode_system(
       V_plasma=V_plasma, n_tot=n_tot, tau_p_T=tau_p_T,
       TBR_DT=TBR_DT, TBR_DDn=TBR_DDn,
       tau_ifc=tau_ifc, tau_ofc=tau_ofc,
       sigmav_DD_p=sigmav_DD_p, sigmav_DD_n=sigmav_DD_n,
       sigmav_DT=sigmav_DT, injection_rate_max=injection_rate_max,
       total_time=total_time
   )
   
   # Check results
   if result['sol_success']:
       print(f"Startup time: {result['t_startup']/86400:.1f} days")
       print(f"Final tritium density: {result['n_T'][-1]:.2e} m⁻³")
       print(f"Number of time points: {len(result['t'])}")
   else:
       print(f"Solution failed: {result['error']}")

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

**Optional adjustments**:

1. Set ``total_time`` appropriately (10 years is typical)
2. Adjust ``N_st_min`` if injection control causes issues
3. Modify initial inventories (N_ifc, N_ofc, N_st) - they are hardcoded to small nonzero values to avoid division by zero at t=0.

See Also
========

- :doc:`physics_lump` - Simplified steady-state model
- :doc:`physics_reactivity` - Reaction rate functions
