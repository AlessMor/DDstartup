Testing Guide
=============

Overview
--------

This guide explains how to run and write tests for the DD Startup Analysis Tool.

Test Structure
--------------

.. code-block:: text

   tests/
   ├── conftest.py              # Pytest fixtures and configuration
   ├── unit/                    # Unit tests
   │   ├── physics/            # Physics module tests
   │   │   ├── test_tseeded_functions.py
   │   │   ├── test_lump_functions.py
   │   │   └── test_reactivity_functions.py
   │   ├── io/                 # I/O tests
   │   └── utils/              # Utility tests
   └── pytest.ini              # Pytest configuration

Current Test Status
-------------------

.. code-block:: text

   ✅ Total: 15 tests
   ✅ All passing (100% success rate)
   ⏱️ Execution time: ~2 seconds

Test Breakdown:

* **test_tseeded_functions.py**: 9 tests - ODE system and time evolution
* **test_reactivity_functions.py**: 5 tests - Reactivity calculations
* **test_lump_functions.py**: 1 test - Comprehensive lump model validation

Prerequisites
-------------

Install testing dependencies:

.. code-block:: bash

   pip install pytest pytest-cov pyyaml numpy

Running Tests
-------------

Run All Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # From project root
   python -m pytest tests/
   
   # With verbose output
   python -m pytest tests/ -v

Run Specific Test Files
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Test T_seeded functions
   pytest tests/unit/physics/test_tseeded_functions.py
   
   # Test reactivity functions
   pytest tests/unit/physics/test_reactivity_functions.py
   
   # Test lump functions
   pytest tests/unit/physics/test_lump_functions.py

Run Specific Test Classes or Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run specific test class
   pytest tests/unit/physics/test_tseeded_functions.py::TestSolveODESystem
   
   # Run specific test function
   pytest tests/unit/physics/test_tseeded_functions.py::TestSolveODESystem::test_exact_solution
   
   # Run tests matching a pattern
   pytest -k "ode_system"

Verbose Output
~~~~~~~~~~~~~~

.. code-block:: bash

   # Detailed output
   pytest tests/ -v
   
   # Extra verbose with full output
   pytest tests/ -vv
   
   # Show print statements
   pytest tests/ -s

Coverage Reports
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run with coverage
   pytest tests/ --cov=ddstartup --cov-report=html
   
   # View coverage report
   xdg-open htmlcov/index.html

Test Categories
---------------

Physics Tests
~~~~~~~~~~~~~

**test_tseeded_functions.py** - Time-dependent T_seeded model:

* ODE system behavior and tritium conservation
* Time evolution with different plasma volumes
* Breeding ratio impact on tritium dynamics

**test_reactivity_functions.py** - Fusion reactivity calculations:

* D-T, D-D, and D-He3 reactivity values
* Temperature dependence validation
* Vectorized input handling

**test_lump_functions.py** - Lump model validation:

* Comprehensive parameter sweep testing
* Power balance and economic calculations

Writing New Tests
-----------------

Test File Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from ddstartup.physics import function_to_test
   
   class TestMyFunction:
       """Tests for my_function"""
       
       def test_basic_case(self):
           """Test basic functionality"""
           result = function_to_test(input)
           assert result == expected
       
       def test_error_case(self):
           """Test error handling"""
           with pytest.raises(ValueError):
               function_to_test(invalid_input)

Using Fixtures
~~~~~~~~~~~~~~

Fixtures are defined in ``conftest.py``:

.. code-block:: python

   def test_with_fixture(sample_params):
       """Use fixture for test data"""
       result = compute(sample_params)
       assert result['success'] == True
* ``sample_yaml_file_missing_fields`` - Invalid YAML file
* ``sample_param_module`` - Sample parameter module file
* ``inputs_dir`` - Inputs directory structure

System Profiler Fixtures:

* ``mock_system_info_large`` - 24 cores, 32 GB RAM
* ``mock_system_info_medium`` - 8 cores, 16 GB RAM
* ``mock_system_info_small`` - 4 cores, 8 GB RAM
* ``mock_system_info_minimal`` - 2 cores, 4 GB RAM

Test Best Practices
-------------------

1. **One concept per test**: Each test should verify one specific behavior
2. **Clear test names**: Use descriptive names that explain what is being tested

Test Best Practices
-------------------

1. **One concept per test**: Each test verifies one specific behavior
2. **Clear test names**: Use descriptive names explaining what is tested
3. **Arrange-Act-Assert**: Structure tests clearly:

   .. code-block:: python
   
      def test_something(self):
          # Arrange - set up test data
          input_value = create_test_data()
          
          # Act - execute the function
          result = function_under_test(input_value)
          
          # Assert - verify the result
          assert result == expected_value

4. **Use fixtures**: Reuse common setup code
5. **Test edge cases**: Include boundary values and error conditions
6. **Keep tests fast**: Aim for quick execution

Test Results Summary
--------------------

Current test status:

.. code-block:: text

   ======================== 15 passed in ~2s =========================
   
   Platform: Linux (Python 3.11.7)
   Test Framework: pytest 7.4.0
   
   Breakdown:
   ├── test_tseeded_functions.py      9 passed ✅
   ├── test_reactivity_functions.py   5 passed ✅
   └── test_lump_functions.py         1 passed ✅

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

Ensure correct directory:

.. code-block:: bash

   cd /path/to/dd_startup
   python -m pytest tests/

Missing Dependencies
~~~~~~~~~~~~~~~~~~~~

Install required packages:

.. code-block:: bash

   pip install -r requirements.txt
   pip install pytest pytest-cov

Debugging Failed Tests
~~~~~~~~~~~~~~~~~~~~~~

Use pytest debugging features:

.. code-block:: bash

   # Stop at first failure
   pytest tests/ -x
   
   # Enter debugger on failure
   pytest tests/ --pdb
   
   # Show local variables on failure
   pytest tests/ -l

See Also
--------

* :doc:`../api_reference/physics_tseeded` - T_seeded physics module
* :doc:`../api_reference/physics_lump` - Lump physics module
* :doc:`contributing` - Contribution guidelines
