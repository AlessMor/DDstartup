Parameter Filtering
===================

Overview
--------

Parameter filtering allows you to apply constraints to reduce the number of parameter combinations computed during analysis. This saves time and resources by excluding invalid or uninteresting cases **before** running expensive computations.

Key Benefits
------------

* **Faster Analysis** - Skip invalid or uninteresting parameter combinations
* **Focused Results** - Only compute physically meaningful cases  
* **Resource Efficient** - Reduce computation time and storage requirements
* **Flexible Constraints** - Support for complex logical expressions

Quick Start
-----------

Add a ``filter`` field to your configuration YAML:

.. code-block:: yaml

   # your_config.yaml
   analysis_type: T_seeded
   method: parametric
   filter: "P_aux < 1e6 and T_i > 10"

Run the analysis:

.. code-block:: bash

   python -m ddstartup params.yaml your_config.yaml

Only parameter combinations satisfying the filter will be computed.

Filter Syntax
-------------

Operators
~~~~~~~~~

**Comparison Operators**

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Operator
     - Description
     - Example
   * - ``<``
     - Less than
     - ``P_aux < 1e6``
   * - ``<=``
     - Less than or equal
     - ``T_i <= 25``
   * - ``>``
     - Greater than
     - ``T_i > 10``
   * - ``>=``
     - Greater than or equal
     - ``n_tot >= 1e20``
   * - ``==``
     - Equal to
     - ``TBR_DT == 1.1``
   * - ``!=``
     - Not equal to
     - ``method != 'lump'``

**Logical Operators**

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Operator
     - Description
     - Example
   * - ``and``
     - Both conditions true
     - ``T_i > 10 and T_i < 25``
   * - ``or``
     - Either condition true
     - ``P_aux < 5e5 or P_aux > 2e6``
   * - ``not``
     - Negate condition
     - ``not (T_i < 10)``

**Arithmetic Operators**

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Operator
     - Description
     - Example
   * - ``+``
     - Addition
     - ``P_aux + P_rad < 1e6``
   * - ``-``
     - Subtraction
     - ``TBR_DT - TBR_DDn > 0.1``
   * - ``*``
     - Multiplication
     - ``n_tot * V_plasma < 1e24``
   * - ``/``
     - Division
     - ``P_aux / V_plasma < 10``
   * - ``**``
     - Power
     - ``T_i**2 > 100``

Variable Names
~~~~~~~~~~~~~~

Use the **base parameter name** without the ``_field`` suffix:

.. code-block:: yaml

   # ✓ Correct
   filter: "T_i > 10 and n_tot < 1e21"
   
   # ✗ Wrong
   filter: "T_i_field > 10"  # Don't include _field suffix

Common Examples
---------------

Physical Constraints
~~~~~~~~~~~~~~~~~~~~

**Temperature range:**

.. code-block:: yaml

   filter: "T_i >= 10 and T_i <= 30"

**Power limits:**

.. code-block:: yaml

   filter: "P_aux > 0 and P_aux < 1e6"

**Density constraints:**

.. code-block:: yaml

   filter: "n_tot > 1e19 and n_tot < 1e21"

Combined Conditions
~~~~~~~~~~~~~~~~~~~

**Fusion power requirements:**

.. code-block:: yaml

   filter: "(P_fus > 1e6) and (T_i > 15) and (n_tot > 5e20)"

**Exclude specific regions:**

.. code-block:: yaml

   filter: "not (T_i < 12 and P_aux > 1e6)"

**Multiple alternatives:**

.. code-block:: yaml

   filter: "(T_i > 20) or (P_aux < 5e5 and n_tot > 8e20)"

Derived Quantities
~~~~~~~~~~~~~~~~~~

**Power density:**

.. code-block:: yaml

   filter: "P_aux / V_plasma < 10e6"

**Confinement product:**

.. code-block:: yaml

   filter: "n_tot * T_i * tau_p_T > 1e21"

**Energy ratios:**

.. code-block:: yaml

   filter: "P_fus / P_aux > 5"

Advanced Usage
--------------

Complex Expressions
~~~~~~~~~~~~~~~~~~~

Filters support arbitrarily complex expressions:

.. code-block:: yaml

   filter: >
     (T_i > 15 and T_i < 25) and
     (P_aux < 1e6) and
     (n_tot * V_plasma < 1e23) and
     ((TBR_DT > 1.05 and TBR_DT < 1.2) or (TBR_DDn > 0.1))

Multi-line Filters
~~~~~~~~~~~~~~~~~~

Use YAML multi-line syntax for readability:

.. code-block:: yaml

   filter: |
     T_i >= 10 and T_i <= 30 and
     P_aux > 0 and P_aux < 1e6 and
     n_tot > 1e19 and n_tot < 1e21

Operator Precedence
~~~~~~~~~~~~~~~~~~~

Standard mathematical precedence applies:

1. Parentheses ``()``
2. Exponentiation ``**``
3. Multiplication/Division ``*`` ``/``
4. Addition/Subtraction ``+`` ``-``
5. Comparison ``<`` ``<=`` ``>`` ``>=`` ``==`` ``!=``
6. Logical NOT ``not``
7. Logical AND ``and``
8. Logical OR ``or``

**Example:**

.. code-block:: yaml

   # This filter:
   filter: "T_i > 10 and P_aux < 1e6 or n_tot > 1e21"
   
   # Is evaluated as:
   filter: "(T_i > 10 and P_aux < 1e6) or (n_tot > 1e21)"
   
   # To change precedence, use parentheses:
   filter: "T_i > 10 and (P_aux < 1e6 or n_tot > 1e21)"

Output Information
------------------

Filter Metadata
~~~~~~~~~~~~~~~

When filtering is active, the output includes:

* **Filter expression** - The constraint applied
* **Efficiency** - Percentage of combinations kept
* **Filtered count** - Number of combinations excluded

Example output:

.. code-block:: text

   Filtering enabled: P_aux < 1e6 and T_i > 10
   Total combinations: 10000
   After filtering: 2847 (28.47% kept)
   Running analysis on 2847 combinations...

Result Files
~~~~~~~~~~~~

Filtered results include metadata in the output file:

.. code-block:: python

   # results.pkl contains:
   {
       'filter_expression': 'P_aux < 1e6 and T_i > 10',
       'filter_efficiency': 0.2847,
       'original_combinations': 10000,
       'filtered_combinations': 2847,
       # ... actual results ...
   }

Validation
----------

Test Before Running
~~~~~~~~~~~~~~~~~~~

Validate filter syntax before running expensive analyses:

.. code-block:: bash

   python -m ddstartup params.yaml config.yaml --validate-only

Common Errors
~~~~~~~~~~~~~

**Unknown variable:**

.. code-block:: text

   Error: Variable 'T_ion' not found in parameters
   Available: T_i, n_tot, P_aux, V_plasma, ...

**Syntax error:**

.. code-block:: text

   Error: Invalid filter syntax: 'T_i > 10 and'
   Expected: comparison expression after 'and'

**Type error:**

.. code-block:: text

   Error: Cannot compare string 'lump' with number
   Filter: 'analysis_type > 10'

Best Practices
--------------

Performance Tips
~~~~~~~~~~~~~~~~

1. **Apply restrictive filters early** - Reduces computation time significantly
2. **Use simple expressions** - Complex filters add overhead  
3. **Test on small parameter sets** - Validate before large runs
4. **Combine related constraints** - More efficient than separate checks

Physical Considerations
~~~~~~~~~~~~~~~~~~~~~~~

1. **Check dimensional consistency** - Ensure units match
2. **Validate physical bounds** - Temperature > 0, density > 0, etc.
3. **Consider coupled parameters** - T_i and n_tot affect fusion power together
4. **Test edge cases** - Verify filter behavior at boundaries

Maintainability
~~~~~~~~~~~~~~~

1. **Document complex filters** - Add comments explaining constraints
2. **Use descriptive names** - Keep parameter names meaningful
3. **Version control filters** - Track changes to constraints
4. **Test thoroughly** - Verify filters don't exclude valid cases

Examples
--------

Realistic Scenarios
~~~~~~~~~~~~~~~~~~~

**Economically viable designs:**

.. code-block:: yaml

   analysis_type: T_seeded
   method: parametric
   filter: |
     P_aux < 50e6 and
     P_fus > 100e6 and
     COE < 0.15 and
     capacity_factor > 0.8

**Compact devices:**

.. code-block:: yaml

   analysis_type: lump
   method: parametric
   filter: |
     V_plasma < 200 and
     P_aux / V_plasma < 5e6 and
     n_tot * V_plasma < 1e23

**High-temperature regime:**

.. code-block:: yaml

   analysis_type: T_seeded
   method: sobol
   filter: |
     T_i > 20 and T_i < 40 and
     tau_p_T > 0.5 and
     P_rad / P_fus < 0.3

See Also
--------

* :doc:`configuration_files` - YAML configuration reference
* :doc:`parameter_definitions` - Parameter definition guide
* :doc:`/api_reference/index` - API documentation

References
----------

Implementation details and code examples:

* ``ddstartup/utils/filtering.py`` - Filter evaluation engine
* ``ddstartup/io/yaml_loader.py`` - Configuration loading
* ``tests/unit/utils/test_filtering.py`` - Unit tests and examples
