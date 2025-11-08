# DD Startup Testing Guide

## Overview

This document outlines the testing conventions, best practices, and common patterns used in the DD Startup project. Follow these guidelines when writing new tests to maintain consistency and code quality.

---

## Table of Contents

1. [Test Organization](#test-organization)
2. [Naming Conventions](#naming-conventions)
3. [Test Structure](#test-structure)
4. [Common Patterns](#common-patterns)
5. [Fixtures and Test Data](#fixtures-and-test-data)
6. [Assertions and Validation](#assertions-and-validation)
7. [Testing Best Practices](#testing-best-practices)
8. [Running Tests](#running-tests)
9. [Examples](#examples)

---

## Test Organization

### Directory Structure

```
tests/
├── conftest.py                          # Shared fixtures and configuration
├── README_TESTS.md                      # This file
├── test_<module_name>.py                # Unit tests for each module
├── manual_<feature>_verification.ipynb  # Manual verification notebooks
└── profile_<feature>_performance.ipynb  # Performance profiling notebooks
```

### File Organization

- **One test file per module**: `test_tseeded_functions.py` tests `ddstartup/physics/Tseeded_functions.py`
- **Group related tests in classes**: Use `TestClassName` for logical grouping
- **Order tests logically**: Simple → Complex, Setup → Execution → Validation

---

## Naming Conventions

### Test Files

- **Format**: `test_<module_name>.py`
- **Examples**:
  - `test_filters.py` → tests `ddstartup/utils/filters.py`
  - `test_tseeded_functions.py` → tests `ddstartup/physics/Tseeded_functions.py`
  - `test_postprocess_functions.py` → tests `ddstartup/postprocessing/postprocess_functions.py`

### Test Functions

- **Format**: `test_<what_it_tests>()`
- **Be descriptive**: Name should explain what is being tested
- **Use underscores**: Separate words for readability

**Good Examples**:
```python
def test_simple_comparison()
def test_ode_system_returns_correct_shape()
def test_parameter_loader_handles_nan_values()
def test_filter_raises_error_for_invalid_syntax()
```

### Test Classes

- **Format**: `TestClassName`
- **Group related tests**: Tests for same function or feature
- **Use descriptive names**: Class name indicates what's being tested

**Examples**:
```python
class TestODESystem:
    """Test the ODE system function."""
    
class TestParameterFilter:
    """Test parameter filtering functionality."""
    
class TestDataLoading:
    """Test HDF5 data loading functions."""
```

---

## Test Structure

### Standard Test Pattern

```python
def test_function_name():
    """
    Brief description of what this test validates.
    
    Include any important context or edge cases being tested.
    """
    # 1. ARRANGE: Set up test data and conditions
    input_value = 10
    expected_output = 20
    
    # 2. ACT: Execute the function being tested
    result = function_to_test(input_value)
    
    # 3. ASSERT: Verify the results
    assert result == expected_output, "Result should be double the input"
```

### Docstring Requirements

Every test function **must** include a docstring:

```python
def test_ode_system_returns_correct_shape():
    """Test that ODE system returns 4 derivatives."""
    # Test implementation
```

For complex tests, add more detail:

```python
def test_apply_filter_to_combinations():
    """
    Test applying filter to full parameter grid.
    
    Validates that:
    - Filter correctly identifies matching combinations
    - Mask array has correct shape
    - Edge cases (all pass, none pass) are handled
    """
    # Test implementation
```

---

## Common Patterns

### 1. Testing Exceptions

Use `pytest.raises` to test that errors are raised correctly:

```python
def test_invalid_syntax():
    """Test that invalid syntax raises FilterError."""
    param_names = ['P_aux']
    
    with pytest.raises(FilterError):
        ParameterFilter("P_aux < ", param_names)
    
    # Can also check error message
    with pytest.raises(FilterError, match="Unknown parameter"):
        ParameterFilter("P_fusion > 1e6", param_names)
```

### 2. Testing Numerical Results

For floating-point comparisons, use appropriate tolerance:

```python
def test_numerical_calculation():
    """Test numerical calculation with tolerance."""
    result = compute_value()
    expected = 1.23456789
    
    # Use np.allclose for arrays
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-9)
    
    # Or pytest.approx for scalars
    assert result == pytest.approx(expected, rel=1e-6)
```

### 3. Testing Array Shapes and Types

Always validate return types and shapes:

```python
def test_function_returns_correct_shape():
    """Test that function returns correct array shape."""
    result = function_returning_array()
    
    assert isinstance(result, np.ndarray), "Should return numpy array"
    assert result.shape == (100, 4), "Shape should be (100, 4)"
    assert result.dtype == np.float64, "Dtype should be float64"
```

### 4. Parametrized Tests

Use `@pytest.mark.parametrize` for testing multiple inputs:

```python
@pytest.mark.parametrize("input_val,expected", [
    (0, 0),
    (1, 2),
    (5, 10),
    (-3, -6),
])
def test_double_function(input_val, expected):
    """Test doubling function with various inputs."""
    assert double(input_val) == expected
```

### 5. Testing Edge Cases

Always test boundary conditions:

```python
def test_function_edge_cases():
    """Test function with edge cases."""
    # Zero
    assert function(0) == expected_for_zero
    
    # Negative
    assert function(-1) == expected_for_negative
    
    # Very large
    assert function(1e100) == expected_for_large
    
    # Very small
    assert function(1e-100) == expected_for_small
```

### 6. Testing with NaN and Inf

Handle special float values explicitly:

```python
def test_function_handles_nan():
    """Test that function correctly handles NaN values."""
    result = function_with_nan(np.nan)
    
    assert np.isnan(result), "Should return NaN for NaN input"
    
def test_function_handles_inf():
    """Test that function handles infinite values."""
    result = function_with_inf(np.inf)
    
    assert np.isinf(result) or result == expected_max_value
```

---

## Fixtures and Test Data

### Using Fixtures from conftest.py

Fixtures defined in `conftest.py` are automatically available to all tests:

```python
def test_with_temp_dir(temp_dir):
    """Test using temporary directory fixture."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test data")
    assert test_file.exists()
```

### Common Fixtures Available

From `conftest.py`:

- **`temp_dir`**: Temporary directory (cleaned up automatically)
- **`sample_yaml_config`**: Sample configuration dictionary
- **`sample_yaml_file`**: Sample YAML configuration file
- **`sample_param_module`**: Sample parameter module
- **`sample_dataframe`**: Sample DataFrame for postprocessing
- **`sample_h5_file`**: Sample HDF5 file

### Creating Custom Fixtures

Add module-specific fixtures at the top of test files:

```python
import pytest

@pytest.fixture
def sample_input_data():
    """Create sample input data for testing."""
    return {
        'V_plasma': np.array([100, 150, 200]),
        'T_i': np.array([10, 15, 20]),
        'n_tot': np.array([1e20, 2e20, 3e20])
    }

def test_with_custom_fixture(sample_input_data):
    """Test using custom fixture."""
    result = process_data(sample_input_data)
    assert len(result) == 3
```

### Fixture Scope

Control fixture lifetime with scope:

```python
@pytest.fixture(scope="module")
def expensive_setup():
    """
    Expensive setup run once per module.
    
    Use when setup is slow but can be shared across tests.
    """
    data = load_large_dataset()
    return data

@pytest.fixture(scope="function")  # Default
def fresh_data():
    """New instance for each test function."""
    return create_test_data()
```

---

## Assertions and Validation

### Standard Assertions

```python
# Equality
assert result == expected

# Inequality
assert result != unexpected

# Boolean
assert condition is True
assert condition is False

# None checks
assert result is not None
assert result is None

# Container membership
assert item in collection
assert item not in collection

# Type checks
assert isinstance(obj, ExpectedType)
```

### Numerical Assertions

```python
# NumPy arrays
assert np.allclose(actual, expected, rtol=1e-5, atol=1e-8)
assert np.array_equal(actual, expected)  # Exact equality

# Scalars with pytest.approx
assert value == pytest.approx(3.14159, rel=1e-5)

# Check for NaN/Inf
assert np.isnan(value)
assert np.isinf(value)
assert np.isfinite(value)
```

### Custom Error Messages

Add descriptive messages to assertions:

```python
assert result > 0, f"Result should be positive, got {result}"
assert len(output) == expected_length, \
    f"Expected {expected_length} items, got {len(output)}"
```

---

## Testing Best Practices

### 1. Test One Thing at a Time

Each test should validate **one specific behavior**:

```python
# GOOD: Tests one specific aspect
def test_filter_accepts_valid_syntax():
    """Test that filter accepts valid comparison syntax."""
    f = ParameterFilter("P_aux > 1e6", ['P_aux'])
    assert f.evaluate({'P_aux': 2e6}) == True

# BAD: Tests multiple unrelated things
def test_filter():
    """Test filter."""
    # Tests syntax validation
    f = ParameterFilter("P_aux > 1e6", ['P_aux'])
    
    # Tests evaluation
    assert f.evaluate({'P_aux': 2e6}) == True
    
    # Tests error handling
    with pytest.raises(FilterError):
        ParameterFilter("invalid", ['P_aux'])
```

### 2. Make Tests Independent

Tests should not depend on execution order:

```python
# GOOD: Self-contained test
def test_function_a():
    data = create_test_data()
    result = function_a(data)
    assert result == expected

# BAD: Depends on previous test
# (Don't do this!)
shared_data = None

def test_setup():
    global shared_data
    shared_data = create_test_data()

def test_function_b():
    # Fails if test_setup() hasn't run
    result = function_b(shared_data)
```

### 3. Use Descriptive Variable Names

Make test code readable:

```python
# GOOD
def test_parameter_filter_with_complex_expression():
    param_names = ['P_aux', 'T_i', 'n_tot']
    filter_expression = "(P_aux > 1e6 and T_i > 15) or n_tot > 5e20"
    test_params = {'P_aux': 2e6, 'T_i': 20, 'n_tot': 1e20}
    
    f = ParameterFilter(filter_expression, param_names)
    assert f.evaluate(test_params) == True

# BAD
def test_filter():
    x = ['P_aux', 'T_i', 'n_tot']
    y = "(P_aux > 1e6 and T_i > 15) or n_tot > 5e20"
    z = {'P_aux': 2e6, 'T_i': 20, 'n_tot': 1e20}
    
    f = ParameterFilter(y, x)
    assert f.evaluate(z) == True
```

### 4. Test Both Success and Failure

Test both valid inputs and error conditions:

```python
def test_parameter_loader_with_valid_yaml():
    """Test loading valid YAML parameter file."""
    params = ParameterLoader.load_from_yaml(valid_yaml_path)
    assert params is not None
    assert 'V_plasma_field' in params

def test_parameter_loader_with_invalid_yaml():
    """Test that invalid YAML raises appropriate error."""
    with pytest.raises(ValueError, match="Missing required field"):
        ParameterLoader.load_from_yaml(invalid_yaml_path)
```

### 5. Keep Tests Fast

- Mock expensive operations (file I/O, network calls)
- Use small test datasets
- Skip slow tests with `@pytest.mark.slow`

```python
@pytest.mark.slow
def test_full_parametric_analysis():
    """
    Test full parametric analysis (slow).
    
    Run with: pytest -m slow
    Skip with: pytest -m "not slow"
    """
    # Long-running test
```

### 6. Document Complex Tests

Add comments for non-obvious test logic:

```python
def test_ode_solver_with_discontinuity():
    """Test ODE solver handles discontinuity in forcing function."""
    # Initial conditions: plasma just starting
    y0 = [0, 0, 1e18, 0]
    
    # Create discontinuous injection rate
    # (simulates sudden fuel injection)
    def injection_rate(t):
        return 1e20 if t > 100 else 0
    
    # Solver should handle this without errors
    result = solve_ode(y0, injection_rate, t_span=(0, 200))
    
    # Verify solution is continuous despite discontinuous input
    assert np.all(np.isfinite(result))
```

### 7. Use Appropriate Test Coverage

Aim for:
- **Critical functions**: 100% coverage (physics calculations, data I/O)
- **Utility functions**: >80% coverage
- **Visualization code**: >50% coverage (focus on data handling, not matplotlib)

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_filters.py

# Run specific test function
pytest tests/test_filters.py::test_simple_comparison

# Run specific test class
pytest tests/test_tseeded_functions.py::TestODESystem

# Run with verbose output
pytest -v

# Run with output capture disabled (see print statements)
pytest -s

# Run and show local variables on failure
pytest -l
```

### Advanced Options

```bash
# Run only tests matching pattern
pytest -k "filter"

# Run only slow tests
pytest -m slow

# Run excluding slow tests
pytest -m "not slow"

# Stop on first failure
pytest -x

# Show test coverage
pytest --cov=ddstartup --cov-report=html

# Run in parallel (requires pytest-xdist)
pytest -n auto
```

### Continuous Testing

```bash
# Watch for changes and rerun tests
pytest-watch

# Or use pytest-testmon for smart test selection
pytest --testmon
```

---

## Examples

### Example 1: Simple Function Test

```python
def test_trapz_integration():
    """Test trapezoidal integration matches analytical result."""
    # Arrange: Create simple function (y = x)
    x = np.linspace(0, 10, 100)
    y = x
    
    # Act: Integrate using trapz
    result = trapz_numba(y, x)
    
    # Assert: Should equal 0.5 * base * height = 0.5 * 10 * 10 = 50
    expected = 50.0
    assert abs(result - expected) < 0.1, f"Integration error: {result} vs {expected}"
```

### Example 2: Class-Based Tests

```python
class TestParameterFilter:
    """Test parameter filtering functionality."""
    
    @pytest.fixture
    def param_names(self):
        """Common parameter names for tests."""
        return ['P_aux', 'T_i', 'n_tot']
    
    def test_simple_comparison(self, param_names):
        """Test simple comparison filters."""
        f = ParameterFilter("P_aux < 1e6", param_names)
        assert f.evaluate({'P_aux': 5e5, 'T_i': 10, 'n_tot': 1e20}) == True
    
    def test_logical_operators(self, param_names):
        """Test AND/OR logical operators."""
        f = ParameterFilter("P_aux < 1e6 and T_i > 10", param_names)
        assert f.evaluate({'P_aux': 5e5, 'T_i': 15, 'n_tot': 1e20}) == True
    
    def test_invalid_syntax(self, param_names):
        """Test that invalid syntax raises FilterError."""
        with pytest.raises(FilterError):
            ParameterFilter("P_aux < ", param_names)
```

### Example 3: Parametrized Test

```python
@pytest.mark.parametrize("expression,params,expected", [
    ("P_aux < 1e6", {'P_aux': 5e5}, True),
    ("P_aux < 1e6", {'P_aux': 2e6}, False),
    ("T_i > 15", {'T_i': 20}, True),
    ("T_i > 15", {'T_i': 10}, False),
    ("P_aux > 1e6 or T_i > 20", {'P_aux': 2e6, 'T_i': 10}, True),
])
def test_filter_expressions(expression, params, expected):
    """Test various filter expressions."""
    param_names = list(params.keys())
    f = ParameterFilter(expression, param_names)
    assert f.evaluate(params) == expected
```

### Example 4: Testing with Fixtures

```python
def test_load_h5_file(sample_h5_file):
    """Test loading HDF5 file into DataFrame."""
    # Act
    df = load_h5_to_dataframe(sample_h5_file)
    
    # Assert
    assert isinstance(df, pd.DataFrame)
    assert 'V_plasma' in df.columns
    assert len(df) == 5
    assert df['V_plasma'].iloc[0] == pytest.approx(100.0)
```

### Example 5: Testing Exception Messages

```python
def test_parameter_loader_missing_field():
    """Test that missing required field raises ValueError with helpful message."""
    incomplete_yaml = """
    parameters:
      V_plasma_field:
        type: scalar
        value: 100
    """
    
    yaml_path = tmp_path / "incomplete.yaml"
    yaml_path.write_text(incomplete_yaml)
    
    with pytest.raises(ValueError) as exc_info:
        ParameterLoader.load_from_yaml(yaml_path)
    
    # Check error message contains helpful information
    assert "Missing required field" in str(exc_info.value)
    assert "T_i_field" in str(exc_info.value) or "required" in str(exc_info.value)
```

---

## Common Pitfalls to Avoid

### ❌ Don't: Test Implementation Details

```python
# BAD: Tests internal implementation
def test_filter_uses_correct_regex():
    """Test that filter uses regex for parsing."""
    f = ParameterFilter("P_aux > 1e6", ['P_aux'])
    assert hasattr(f, '_regex_pattern')  # Internal detail
```

```python
# GOOD: Tests behavior/interface
def test_filter_accepts_valid_syntax():
    """Test that filter accepts valid comparison syntax."""
    f = ParameterFilter("P_aux > 1e6", ['P_aux'])
    assert f.evaluate({'P_aux': 2e6}) == True
```

### ❌ Don't: Use Magic Numbers

```python
# BAD
def test_calculation():
    result = calculate(10, 20)
    assert result == 30

# GOOD
def test_addition():
    """Test that calculate correctly adds two numbers."""
    first_value = 10
    second_value = 20
    expected_sum = 30
    
    result = calculate(first_value, second_value)
    assert result == expected_sum
```

### ❌ Don't: Write Tests That Can't Fail

```python
# BAD: This test can never fail
def test_function():
    result = function()
    assert True  # Always passes!

# GOOD
def test_function_returns_positive_value():
    result = function()
    assert result > 0, f"Expected positive value, got {result}"
```

### ❌ Don't: Ignore Warnings

```python
# BAD: Suppresses all warnings
import warnings
warnings.filterwarnings("ignore")

# GOOD: Suppress specific expected warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    result = function_with_deprecated_feature()
```

---

## Quick Reference Checklist

When writing a new test, ensure:

- [ ] Test file named `test_<module>.py`
- [ ] Test function named `test_<descriptive_name>`
- [ ] Docstring explains what is being tested
- [ ] Uses Arrange-Act-Assert pattern
- [ ] Tests one specific behavior
- [ ] Uses appropriate assertions with tolerance for floats
- [ ] Handles edge cases (zero, negative, NaN, inf)
- [ ] Tests both success and error conditions
- [ ] Uses fixtures for common setup
- [ ] Runs independently of other tests
- [ ] Completes quickly (< 1 second if possible)
- [ ] Has descriptive variable names
- [ ] Includes helpful error messages in assertions

---

## Additional Resources

### Pytest Documentation
- [Pytest Official Docs](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Parametrize](https://docs.pytest.org/en/stable/parametrize.html)

### Testing Best Practices
- [Testing Best Practices (Real Python)](https://realpython.com/pytest-python-testing/)
- [Effective Python Testing](https://effectivepython.com/)

### Project-Specific
- `conftest.py` - Available fixtures
- `docs/` - Project documentation
- `MAIN_COMMANDS.md` - Common commands and workflows

---

## Questions or Issues?

If you have questions about testing conventions or need help writing tests:

1. Check existing tests for similar examples
2. Review this document
3. Consult `conftest.py` for available fixtures
4. Ask in project discussions/issues

---

**Last Updated**: November 4, 2025  
**Maintained By**: DD Startup Development Team
