Parameter Definitions
=====================

Overview
--------

Parameter definition files specify the simulation parameter space using either **YAML files** (recommended) or Python modules with ``ParameterField`` objects. These files define values and ranges for physical and operational parameters.

.. note::
   **YAML format is now the recommended approach** for all new parameter files due to better separation of configuration from code, improved readability, and easier maintenance. Python parameter files are still fully supported for backward compatibility.

YAML Parameter Files (Recommended)
-----------------------------------

Quick Start
~~~~~~~~~~~

Create a YAML parameter file (e.g., ``params.yaml``):

.. code-block:: yaml

   parameters:
     V_plasma_field:
       type: linear
       min: 100
       max: 200
       points: 5
       unit: m^3
       description: Plasma volume
     
     T_i_field:
       type: linear
       min: 15
       max: 20
       points: 3
       unit: keV
       description: Ion temperature
     
     n_tot_field:
       type: linear
       min: 1.3e20
       max: 2.0e20
       points: 3
       unit: 1/m^3
       description: Total particle density

Run with the YAML file:

.. code-block:: bash

   python -m ddstartup params.yaml parametric_tseeded.yaml

Parameter Types
~~~~~~~~~~~~~~~

YAML parameters support four field types:

**scalar** - Single fixed value:

.. code-block:: yaml

   V_plasma_field:
     type: scalar
     value: 150
     unit: m^3
     description: Plasma volume

**linear** - Uniformly spaced range:

.. code-block:: yaml

   T_i_field:
     type: linear
     min: 14
     max: 20
     points: 10
     unit: keV
     description: Ion temperature

**normal** - Normally distributed values:

.. code-block:: yaml

   tau_p_T_field:
     type: normal
     mean: 0.5
     std: 0.1
     points: 5
     unit: s
     description: Particle confinement time

**vector** - Explicit list of values:

.. code-block:: yaml

   TBR_DT_field:
     type: vector
     values: [1.05, 1.10, 1.15, 1.20]
     unit: ""
     description: DT tritium breeding ratio

Special Features
~~~~~~~~~~~~~~~~

**Automatic P_aux Calculation**

Set P_aux to NaN to calculate from power balance:

.. code-block:: yaml

   P_aux_field:
     type: scalar
     value: .nan  # Will be calculated automatically
     unit: MW
     description: Auxiliary heating power

The system will calculate: ``P_aux = P_rad + 3*n_tot*T_i*V_plasma - P_charged``

**Scientific Notation**

YAML supports scientific notation:

.. code-block:: yaml

   n_tot_field:
     type: linear
     min: 1.3e20
     max: 2.0e20
     points: 5
     unit: 1/m^3

**Multi-line Descriptions**

Use YAML multi-line syntax for long descriptions:

.. code-block:: yaml

   tau_ifc_field:
     type: vector
     values: [3600, 21600, 43200]
     unit: s
     description: |
       In-fuel cycle time.
       Represents time between fuel processing cycles.
       Values: 1, 6, 12 hours converted to seconds.

Complete Example
~~~~~~~~~~~~~~~~

File: ``inputs/params.yaml``

.. code-block:: yaml

   # DD Startup Analysis Parameters
   # Complete configuration for T_seeded parametric analysis
   
   parameters:
     # Plasma parameters
     V_plasma_field:
       type: linear
       min: 100
       max: 200
       points: 5
       unit: m^3
       description: Plasma volume
     
     T_i_field:
       type: linear
       min: 14
       max: 20
       points: 10
       unit: keV
       description: Ion temperature
     
     n_tot_field:
       type: linear
       min: 1.3e20
       max: 2.0e20
       points: 5
       unit: 1/m^3
       description: Total particle density
     
     # Confinement
     tau_p_T_field:
       type: normal
       mean: 0.5
       std: 0.1
       points: 5
       unit: s
       description: Tritium particle confinement time
     
     # Auxiliary power
     P_aux_field:
       type: linear
       min: 20e6
       max: 100e6
       points: 5
       unit: W
       description: Auxiliary heating power
     
     P_aux_DT_eq_field:
       type: scalar
       value: 60e6
       unit: W
       description: DT-equivalent auxiliary power
     
     # Breeding ratios
     TBR_DT_field:
       type: vector
       values: [1.05, 1.10, 1.15]
       unit: ""
       description: DT tritium breeding ratio
     
     TBR_DDn_field:
       type: vector
       values: [0.5, 0.7, 0.9]
       unit: ""
       description: DDn tritium breeding ratio
     
     # Fuel cycle times
     tau_ifc_field:
       type: vector
       values: [3600, 21600, 43200]
       unit: s
       description: Inner fuel cycle time (1, 6, 12 hours)
     
     tau_ofc_field:
       type: vector
       values: [3600, 43200, 86400]
       unit: s
       description: Outer fuel cycle time (1, 12, 24 hours)
     
     # Economic parameters
     eta_th_field:
       type: vector
       values: [0.30, 0.35, 0.40]
       unit: ""
       description: Thermal conversion efficiency
     
     capacity_factor_field:
       type: vector
       values: [0.5, 0.7, 0.9]
       unit: ""
       description: Plant capacity factor
     
     price_of_electricity_field:
       type: vector
       values: [50, 75, 100]
       unit: $/MWh
       description: Price of electricity

Advantages of YAML
~~~~~~~~~~~~~~~~~~~

**Compared to Python parameter files:**

1. **Clean and readable** - No Python syntax required
2. **Safe** - No code execution risks
3. **Portable** - Can be used by other tools
4. **Version control friendly** - Cleaner git diffs
5. **Easy validation** - Programmatic structure checking
6. **Better separation** - Configuration separate from code

Python Parameter Files (Legacy)
--------------------------------

Overview
~~~~~~~~

Parameter definition files are Python modules that define the simulation parameter space using ``ParameterField`` objects. These files specify the values and ranges for physical and operational parameters.

.. note::
   Python parameter files are still fully supported but YAML format is recommended for new configurations.

ParameterField Class
--------------------

The ``ParameterField`` class defines a single parameter with its values and metadata.

Constructor
~~~~~~~~~~~

.. code-block:: python

   ParameterField(
       name: str,
       field_type: str,
       value: any,
       unit: str,
       description: str = ""
   )

**Parameters:**

* ``name`` - Parameter name (string)
* ``field_type`` - Type of field (see Field Types below)
* ``value`` - Parameter value(s)
* ``unit`` - Physical unit (for Pint unit registry)
* ``description`` - Optional description

Field Types
-----------

scalar
~~~~~~

Single constant value.

.. code-block:: python

   from ddstartup.utils import u, ParameterField
   
   V_plasma_field = ParameterField(
       name='V_plasma',
       field_type='scalar',
       value=150,
       unit='m^3',
       description='Plasma volume'
   )
   # Results in: 150 m³

linear
~~~~~~

Linearly-spaced array between two values.

.. code-block:: python

   T_i_field = ParameterField(
       name='T_i',
       field_type='linear',
       value=(14, 20, 2),  # (min, max, num_points)
       unit='keV',
       description='Ion temperature'
   )
   # Results in: [14.0, 20.0] keV (2 points)

normal
~~~~~~

Normally-distributed values (mean, std, num_points).

.. code-block:: python

   tau_p_He3_field = ParameterField(
       name='tau_p_He3',
       field_type='normal',
       value=(1.0, 0.215, 2),  # (mean, std, num_points)
       unit='s',
       description='He3 particle confinement time'
   )
   # Results in: [0.785, 1.215] s (approximately)

array
~~~~~

Explicitly specified array of values.

.. code-block:: python

   TBR_DT_field = ParameterField(
       name='TBR_DT',
       field_type='array',
       value=[1.05, 1.10, 1.15],
       unit='',  # dimensionless
       description='DT tritium breeding ratio'
   )
   # Results in: [1.05, 1.10, 1.15]

Example Parameter Files
------------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

File: ``inputs/config.py``

.. code-block:: python

   """
   Basic parameter configuration for DD startup analysis
   """
   from ddstartup.utils import u, ParameterField
   
   # Plasma parameters
   V_plasma_field = ParameterField(
       name='V_plasma',
       field_type='scalar',
       value=150,
       unit='m^3',
       description='Plasma volume'
   )
   
   T_i_field = ParameterField(
       name='T_i',
       field_type='scalar',
       value=17,
       unit='keV',
       description='Ion temperature'
   )
   
   n_tot_field = ParameterField(
       name='n_tot',
       field_type='scalar',
       value=1.7e20,
       unit='1/m^3',
       description='Total particle density'
   )
   
   # Confinement
   tau_p_T_field = ParameterField(
       name='tau_p_T',
       field_type='linear',
       value=(0.1, 1.0, 10),  # 0.1 to 1.0 s, 10 points
       unit='s',
       description='Tritium particle confinement time'
   )
   
   # Auxiliary power
   P_aux_field = ParameterField(
       name='P_aux',
       field_type='linear',
       value=(20e6, 100e6, 5),  # 20 to 100 MW, 5 points
       unit='W',
       description='Auxiliary heating power'
   )
   
   # Tritium breeding
   TBR_DT_field = ParameterField(
       name='TBR_DT',
       field_type='array',
       value=[1.05, 1.10, 1.15],
       unit='',
       description='DT tritium breeding ratio'
   )
   
   TBR_DDn_field = ParameterField(
       name='TBR_DDn',
       field_type='array',
       value=[0.5, 0.7, 0.9],
       unit='',
       description='DDn tritium breeding ratio'
   )
   
   # Fuel cycle times
   tau_ifc_field = ParameterField(
       name='tau_ifc',
       field_type='array',
       value=[1, 6, 12],  # hours
       unit='hour',
       description='Inner fuel cycle time'
   )
   
   tau_ofc_field = ParameterField(
       name='tau_ofc',
       field_type='array',
       value=[1, 12, 24],  # hours
       unit='hour',
       description='Outer fuel cycle time'
   )
   
   # Economic parameters
   eta_th_field = ParameterField(
       name='eta_th',
       field_type='array',
       value=[0.3, 0.35, 0.4],
       unit='',
       description='Thermal conversion efficiency'
   )
   
   capacity_factor_field = ParameterField(
       name='capacity_factor',
       field_type='array',
       value=[0.5, 0.7, 0.9],
       unit='',
       description='Plant capacity factor'
   )
   
   price_of_electricity_field = ParameterField(
       name='price_of_electricity',
       field_type='array',
       value=[50, 75, 100],  # $/MWh
       unit='$/kWh',
       description='Price of electricity'
   )
   
   # Simulation time
   total_time = 10 * 365 * 24 * 3600  # 10 years in seconds

Test Configuration
~~~~~~~~~~~~~~~~~~

File: ``inputs/config_test.py``

Reduced parameter space for quick testing:

.. code-block:: python

   """
   Test configuration with minimal parameter space
   """
   from ddstartup.utils import u, ParameterField
   
   # Single values for quick testing
   V_plasma_field = ParameterField('V_plasma', 'scalar', 150, 'm^3')
   T_i_field = ParameterField('T_i', 'scalar', 17, 'keV')
   n_tot_field = ParameterField('n_tot', 'scalar', 1.7e20, '1/m^3')
   
   # Minimal ranges (2 points each)
   tau_p_T_field = ParameterField('tau_p_T', 'linear', (0.1, 1.0, 2), 's')
   P_aux_field = ParameterField('P_aux', 'scalar', 60e6, 'W')
   
   TBR_DT_field = ParameterField('TBR_DT', 'linear', (1.05, 1.15, 2), '')
   TBR_DDn_field = ParameterField('TBR_DDn', 'linear', (0.5, 0.9, 2), '')
   
   tau_ifc_field = ParameterField('tau_ifc', 'linear', (1, 12, 2), 'hour')
   tau_ofc_field = ParameterField('tau_ofc', 'linear', (1, 24, 2), 'hour')
   
   eta_th_field = ParameterField('eta_th', 'linear', (0.3, 0.4, 2), '')
   capacity_factor_field = ParameterField('capacity_factor', 'linear', (0.5, 0.9, 2), '')
   price_of_electricity_field = ParameterField('price_of_electricity', 'linear', (50, 100, 2), '$/kWh')
   
   # Additional lump-specific parameter
   I_target_field = ParameterField('I_target', 'linear', (0.5, 2.0, 2), 'kg')
   tau_p_He3_field = ParameterField('tau_p_He3', 'normal', (1.0, 0.215, 2), 's')
   
   P_aux_DT_eq_field = ParameterField('P_aux_DT_eq', 'scalar', 60e6, 'W')
   
   total_time = 10 * 365 * 24 * 3600  # 10 years

Required Parameters
-------------------

T_seeded Analysis
~~~~~~~~~~~~~~~~~

Required parameter fields for T_seeded analysis:

.. code-block:: python

   V_plasma_field         # Plasma volume [m³]
   T_i_field              # Ion temperature [keV]
   n_tot_field            # Total particle density [1/m³]
   tau_p_T_field          # Tritium confinement time [s]
   P_aux_field            # Auxiliary power [W]
   P_aux_DT_eq_field      # DT-equivalent auxiliary power [W]
   TBR_DT_field           # DT breeding ratio [-]
   TBR_DDn_field          # DDn breeding ratio [-]
   tau_ifc_field          # Inner fuel cycle time [s]
   tau_ofc_field          # Outer fuel cycle time [s]
   eta_th_field           # Thermal efficiency [-]
   capacity_factor_field  # Capacity factor [-]
   price_of_electricity_field  # Electricity cost [$/J]

lump Analysis
~~~~~~~~~~~~~

Required parameter fields for lump analysis:

.. code-block:: python

   V_plasma_field         # Plasma volume [m³]
   T_i_field              # Ion temperature [keV]
   n_tot_field            # Total particle density [1/m³]
   tau_p_T_field          # Tritium confinement time [s]
   tau_p_He3_field        # He3 confinement time [s]
   P_aux_field            # Auxiliary power [W]
   P_aux_DT_eq_field      # DT-equivalent auxiliary power [W]
   TBR_DT_field           # DT breeding ratio [-]
   TBR_DDn_field          # DDn breeding ratio [-]
   I_target_field         # Target inventory [kg]
   eta_th_field           # Thermal efficiency [-]
   capacity_factor_field  # Capacity factor [-]
   price_of_electricity_field  # Electricity cost [$/J]

Unit Conversions
----------------

The system automatically converts units to SI base units:

.. code-block:: python

   # Input
   tau_ifc_field = ParameterField('tau_ifc', 'array', [1, 6, 12], 'hour')
   
   # Automatic conversion to seconds
   # Results in: [3600, 21600, 43200] seconds

Supported units:

* **Time:** s, hour, day, year
* **Energy:** J, eV, keV, MeV
* **Power:** W, kW, MW, GW
* **Length:** m, cm, km
* **Volume:** m^3, L
* **Density:** 1/m^3, 1/cm^3
* **Cost:** $/J, $/kWh, $/MWh
* **Dimensionless:** '' (empty string)

Parameter Space Size
--------------------

The total number of parameter combinations is the product of all parameter array sizes:

.. code-block:: python

   # Example
   tau_p_T_field: 10 points
   P_aux_field: 5 points
   TBR_DT_field: 3 points
   TBR_DDn_field: 3 points
   # ... other parameters ...
   
   # Total combinations = 10 × 5 × 3 × 3 × ... = large number

For test configurations, use 2-3 points per parameter to keep the parameter space manageable.

Best Practices
--------------

1. **Start small**: Use ``config_test.py`` with 2 points per parameter for validation
2. **Document parameters**: Add descriptions to all ``ParameterField`` objects
3. **Use appropriate types**: Choose field types that match your analysis needs
4. **Check units**: Verify units are compatible with Pint unit registry
5. **Test imports**: Ensure parameter file imports successfully before running analysis
6. **Version control**: Track parameter files in git to maintain history

Common Patterns
---------------

Parameter Sweeps
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Logarithmic sweep (manual)
   import numpy as np
   
   power_values = np.logspace(6, 8, 20)  # 1 MW to 100 MW
   P_aux_field = ParameterField('P_aux', 'array', power_values, 'W')

Parameter Correlations
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Same values for related parameters
   base_tau = (0.1, 1.0, 10)
   
   tau_p_T_field = ParameterField('tau_p_T', 'linear', base_tau, 's')
   tau_p_D_field = ParameterField('tau_p_D', 'linear', base_tau, 's')

Dimensionless Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use empty string for dimensionless quantities
   TBR_DT_field = ParameterField('TBR_DT', 'array', [1.0, 1.1, 1.2], '')
   eta_th_field = ParameterField('eta_th', 'array', [0.3, 0.4], '')

See Also
--------

* :doc:`../api_reference/custom_classes` - ParameterField API reference
* :doc:`../api_reference/units_and_constants` - Unit registry documentation
* :doc:`configuration_files` - YAML configuration files
