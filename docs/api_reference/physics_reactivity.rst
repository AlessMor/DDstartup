Reactivity Functions
====================

.. module:: ddstartup.physics.reactivity_functions

The ``reactivity_functions`` module provides fusion reactivity (⟨σv⟩) calculations using the Bosch-Hale parameterization :cite:`bosch_improved_1992`, as implemented in cfspopcon :cite: `cfspopcon`.
The functions return reactivities in SI units (m³/s) and support both scalar and NumPy array temperature inputs for efficient vectorized computations.

Overview
--------

This module implements temperature-dependent fusion reactivities for:

* **D-T (Deuterium-Tritium)** - Primary fusion reaction
* **D-D (Deuterium-Deuterium)** - Two branches: D(d,n)³He and D(d,p)T
* **D-³He (Deuterium-Helium-3)** - Advanced fuel cycle

All functions use the Bosch-Hale parameterization :cite:`bosch_improved_1992` for accurate cross-section calculations and are tested against reference data from the NRL Plasma Formulary :cite:`nrl_plasma_formulary_2019`.

Functions
---------

.. autofunction:: sigmav_DT_BoschHale

.. autofunction:: sigmav_DD_BoschHale

.. autofunction:: sigmav_DHe3_BoschHale

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

Calculate D-T reactivity at typical fusion temperatures:

.. code-block:: python

   from ddstartup.physics.reactivity_functions import sigmav_DT_BoschHale
   import numpy as np
   
   # Single temperature
   T_i = 14.0  # keV
   reactivity = sigmav_DT_BoschHale(T_i)
   print(f"DT reactivity at {T_i} keV: {reactivity:.3e} m³/s")
   
   # Array of temperatures
   T_array = np.array([10, 20, 50, 100])  # keV
   reactivities = sigmav_DT_BoschHale(T_array)
   for T, react in zip(T_array, reactivities):
       print(f"T={T:3.0f} keV: {react:.3e} m³/s")

D-D Reaction Branches
~~~~~~~~~~~~~~~~~~~~~

The D-D reaction has two equally probable branches:

.. code-block:: python

   from ddstartup.physics.reactivity_functions import sigmav_DD_BoschHale
   
   T_i = 14.0  # keV
   react_total, react_dn, react_dp = sigmav_DD_BoschHale(T_i)
   
   print(f"Total DD reactivity: {react_total:.3e} m³/s")
   print(f"D(d,n)³He branch:    {react_dn:.3e} m³/s")
   print(f"D(d,p)T branch:      {react_dp:.3e} m³/s")

Comparing Fuel Cycles
~~~~~~~~~~~~~~~~~~~~~~

Compare reactivities of different fusion reactions:

.. code-block:: python

   from ddstartup.physics.reactivity_functions import (
       sigmav_DT_BoschHale,
       sigmav_DD_BoschHale,
       sigmav_DHe3_BoschHale
   )
   
   T_i = 14.0  # keV
   
   react_DT = sigmav_DT_BoschHale(T_i)
   react_DD_total, _, _ = sigmav_DD_BoschHale(T_i)
   react_DHe3 = sigmav_DHe3_BoschHale(T_i)
   
   print(f"Reactivity comparison at {T_i} keV:")
   print(f"  D-T:    {react_DT:.3e} m³/s")
   print(f"  D-D (total):    {react_DD_total:.3e} m³/s")
   print(f"  D-³He:  {react_DHe3:.3e} m³/s")

Temperature Scan
~~~~~~~~~~~~~~~~

Generate reactivity curves over temperature range:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from ddstartup.physics.reactivity_functions import (
       sigmav_DT_BoschHale,
       sigmav_DD_BoschHale,
       sigmav_DHe3_BoschHale
   )
   
   # Temperature range (linear spacing): 1-200 keV
   T_range = np.linspace(1, 200, 100)
   
   react_DT = sigmav_DT_BoschHale(T_range)
   react_DD_total = sigmav_DD_BoschHale(T_range)[0]
   react_DHe3 = sigmav_DHe3_BoschHale(T_range)
   
   plt.figure(figsize=(10, 6))
   # Plot on log-log axes while computing on a linear temperature grid
   plt.loglog(T_range, react_DT, label='D-T', linewidth=2)
   plt.loglog(T_range, react_DD_total, label='D-D', linewidth=2)
   plt.loglog(T_range, react_DHe3, label='D-³He', linewidth=2)
   plt.xlabel('Ion Temperature (keV)')
   plt.ylabel('Reactivity ⟨σv⟩ (m³/s)')
   plt.title('Fusion Reactivities vs Temperature (computed on linear T grid, plotted log-log)')
   plt.legend()
   plt.grid(True, alpha=0.3, which='both')
   plt.show()

See Also
--------

* :doc:`physics_tseeded` - T-seeded analysis using these reactivities
* :doc:`physics_lump` - Lumped parameter model
* :doc:`../user_guide/parameter_definitions` - Parameter definitions
* :doc:`../bibliography` - Complete bibliography

References
----------

.. bibliography::
   :filter: key in {"bosch_improved_1992", "cfspopcon", "nrl_plasma_formulary_2019"}
   :style: plain

