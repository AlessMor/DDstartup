Testing Guide
=============

Overview
--------

This guide explains how to run and write tests for the DD Startup Analysis Tool.

Test Structure
--------------

.. code-block:: text

   ddstartup/
   ├── tests/
   │   ├── conftest.py              # Pytest fixtures and configuration
   │   ├── test_io_functions.py     # Tests for I/O functions (18 tests)
   │   ├── test_main.py             # Tests for main entry point (13 tests)
   │   └── test_system_profiler.py  # Tests for system profiler (43 tests)
   ├── pytest.ini                   # Pytest configuration
   └── run_tests.py                 # Test runner script

Current Test Status
-------------------

.. code-block:: text

   ✅ Total: 74 tests
   ✅ All passing (100% success rate)
   ⏱️ Execution time: ~0.45 seconds

Test Breakdown:

* **test_io_functions.py**: 18 unit tests
* **test_main.py**: 13 integration tests  
* **test_system_profiler.py**: 43 unit tests
* **Fixtures**: 6+ fixtures in conftest.py

Prerequisites
-------------

Install testing dependencies:

.. code-block:: bash

   pip install pytest pytest-cov pyyaml

Running Tests
-------------

Run All Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # From ddstartup/ directory
   python run_tests.py
   
   # Or use pytest directly
   pytest tests/
   
   # Or from parent directory
   cd dd_startup/ddstartup
   python -m pytest tests/

Run Specific Test Files
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Test only I/O functions
   pytest tests/test_io_functions.py
   
   # Test only main function
   pytest tests/test_main.py
   
   # Test only system profiler
   pytest tests/test_system_profiler.py

Run Specific Test Classes or Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run specific test class
   pytest tests/test_io_functions.py::TestResolveFilePath
   
   # Run specific test function
   pytest tests/test_io_functions.py::TestResolveFilePath::test_resolve_existing_file
   
   # Run tests matching a pattern
   pytest -k "test_load_config"

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
   pytest tests/ --cov=. --cov-report=html
   
   # View coverage report (Linux)
   xdg-open htmlcov/index.html

Test Categories
---------------

Unit Tests
~~~~~~~~~~

Test individual functions in isolation.

**test_io_functions.py** (18 tests):

* ``TestResolveFilePath`` (4 tests) - File path resolution
* ``TestLoadConfig`` (5 tests) - YAML configuration loading
* ``TestLoadParameterFields`` (3 tests) - Parameter module loading
* ``TestPrepareInputData`` (3 tests) - Input data preparation
* ``TestPrintConfiguration`` (3 tests) - Configuration display

**test_system_profiler.py** (43 tests):

* ``TestGetSystemInfo`` (5 tests) - System info retrieval
* ``TestCalculateOptimalNJobs`` (5 tests) - Parallel worker calculation
* ``TestCalculateOptimalChunkSize`` (5 tests) - Chunk size calculation
* ``TestCalculateOptimalBatchSize`` (5 tests) - Batch size calculation
* ``TestCalculateOptimalSobolSamples`` (5 tests) - Sobol sample count
* ``TestCalculateOptimalSobolOrder`` (5 tests) - Sobol order selection
* ``TestGetOptimalParameters`` (5 tests) - Master function testing
* ``TestPrintSystemProfile`` (3 tests) - Output formatting
* ``TestOverrideWithConfig`` (5 tests) - Config override logic

Integration Tests
~~~~~~~~~~~~~~~~~

Test complete workflows.

**test_main.py** (13 tests):

* ``TestParseArguments`` (6 tests) - CLI argument parsing
* ``TestMainFunction`` (6 tests) - Main workflow and error handling
* ``TestMainIntegration`` (1 test) - Full end-to-end workflow

Writing New Tests
-----------------

Test File Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from module import function_to_test
   
   
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

   def test_with_temp_dir(temp_dir):
       """Use temporary directory fixture"""
       test_file = temp_dir / "test.txt"
       test_file.write_text("content")
       assert test_file.exists()

Available Fixtures
~~~~~~~~~~~~~~~~~~

Core Fixtures:

* ``temp_dir`` - Temporary directory (cleaned up after test)
* ``sample_yaml_config`` - Sample YAML configuration dict
* ``sample_yaml_file`` - Sample YAML configuration file
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
3. **Arrange-Act-Assert**: Structure tests in three clear sections:

   .. code-block:: python
   
      def test_something(self):
          # Arrange - set up test data
          input_value = create_test_data()
          
          # Act - execute the function
          result = function_under_test(input_value)
          
          # Assert - verify the result
          assert result == expected_value

4. **Use fixtures**: Reuse common setup code via fixtures
5. **Test edge cases**: Include tests for error conditions and boundary values
6. **Mock external dependencies**: Use mocking for file I/O and imports
7. **Fast tests**: Keep tests under 1 second total execution time

Test Results Summary
--------------------

Current test results (as of refactoring completion):

.. code-block:: text

   ======================== 74 passed in 0.45s =========================
   
   Platform: Linux (Python 3.13.7)
   Test Framework: pytest 8.4.2
   Coverage: Core modules
   
   Breakdown:
   ├── test_io_functions.py      18 passed ✅
   ├── test_main.py               13 passed ✅
   └── test_system_profiler.py    43 passed ✅

Continuous Integration
----------------------

Pre-commit Hook
~~~~~~~~~~~~~~~

Create ``.git/hooks/pre-commit``:

.. code-block:: bash

   #!/bin/bash
   echo "Running tests..."
   cd ddstartup
   pytest tests/ -x
   if [ $? -ne 0 ]; then
       echo "Tests failed. Commit aborted."
       exit 1
   fi

Make it executable:

.. code-block:: bash

   chmod +x .git/hooks/pre-commit

GitHub Actions
~~~~~~~~~~~~~~

Add ``.github/workflows/tests.yml``:

.. code-block:: yaml

   name: Tests
   
   on: [push, pull_request]
   
   jobs:
     test:
       runs-on: ubuntu-latest
       
       steps:
       - uses: actions/checkout@v2
       
       - name: Set up Python
         uses: actions/setup-python@v2
         with:
           python-version: '3.13'
       
       - name: Install dependencies
         run: |
           pip install -r requirements.txt
           pip install pytest pytest-cov
       
       - name: Run tests
         run: |
           cd ddstartup
           pytest tests/ --cov --cov-report=xml
       
       - name: Upload coverage
         uses: codecov/codecov-action@v2

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

If you get import errors, make sure you're running tests from the correct directory:

.. code-block:: bash

   cd dd_startup/ddstartup
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   pytest tests/

Missing Dependencies
~~~~~~~~~~~~~~~~~~~~

Install required packages:

.. code-block:: bash

   pip install pytest pytest-cov pyyaml numpy psutil

Fixture Not Found
~~~~~~~~~~~~~~~~~

Make sure ``conftest.py`` is in the tests directory and properly formatted.

Expected Test Output
--------------------

Successful test run:

.. code-block:: text

   ========================= test session starts ==========================
   platform linux -- Python 3.13.7, pytest-8.4.2
   rootdir: /path/to/dd_startup/ddstartup
   plugins: cov-4.1.0
   collected 74 items
   
   tests/test_io_functions.py::TestResolveFilePath::test_resolve_existing_file PASSED [  1%]
   tests/test_io_functions.py::TestResolveFilePath::test_resolve_file_in_default_dir PASSED [  2%]
   ...
   tests/test_system_profiler.py::TestOverrideWithConfig::test_preserves_system_info PASSED [100%]
   
   ========================== 74 passed in 0.45s ==========================

Debugging Failed Tests
----------------------

Use pytest's debugging features:

.. code-block:: bash

   # Stop at first failure
   pytest tests/ -x
   
   # Enter debugger on failure
   pytest tests/ --pdb
   
   # Show local variables on failure
   pytest tests/ -l
   
   # Increase verbosity
   pytest tests/ -vv

Adding New Tests
----------------

When adding new features, follow this workflow:

1. **Write the test first** (TDD approach)
2. **Run the test** - it should fail
3. **Implement the feature**
4. **Run the test again** - it should pass
5. **Refactor if needed** - tests ensure you don't break anything

Example:

.. code-block:: python

   # 1. Write test in tests/test_new_feature.py
   def test_new_feature():
       result = new_feature(input_data)
       assert result == expected_output
   
   # 2. Run test - should fail
   # $ pytest tests/test_new_feature.py
   
   # 3. Implement feature
   # ... add code ...
   
   # 4. Run test again - should pass
   # $ pytest tests/test_new_feature.py

Maintaining Test Quality
-------------------------

Regularly check:

* **Test coverage**: Aim for >80% coverage
* **Test speed**: Keep total time under 1 second
* **Test isolation**: Tests shouldn't depend on each other
* **Test clarity**: Names and docstrings should be clear
* **Mock usage**: Don't test external dependencies

See Also
--------

* :doc:`../api_reference/io_functions` - Functions being tested
* :doc:`../api_reference/system_profiler` - System profiler being tested
* :doc:`contributing` - Contribution guidelines
