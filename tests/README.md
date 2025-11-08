# Test Suite for DD Startup Analysis

This directory contains a comprehensive test suite organized with pytest.

## 📁 Directory Structure

```
tests/
├── conftest.py                 # Shared fixtures and pytest configuration
├── test_module_template.py     # Generic test template (copy and modify)
│
├── unit/                       # Unit tests for individual components
│   ├── physics/                # Physics module tests
│   │   ├── test_reaction_rates.py
│   │   ├── test_lump_functions.py
│   │   └── test_tseeded_functions.py
│   ├── io/                     # I/O module tests
│   │   └── test_io_functions.py
│   └── utils/                  # Utility module tests
│       └── test_custom_classes.py
│
├── integration/                # Integration tests (multi-module workflows)
│   └── test_full_workflow.py
│
└── fixtures/                   # Shared test data and fixtures
    └── sample_data.py
```

## 🚀 Running Tests

### Run all tests
```bash
pytest tests/
```

### Run tests in a specific directory
```bash
pytest tests/unit/physics/
```

### Run a specific test file
```bash
pytest tests/unit/physics/test_reaction_rates.py
```

### Run tests with specific markers
```bash
# Run only unit tests
pytest tests/ -m unit

# Skip slow tests
pytest tests/ -m "not slow"

# Run only physics tests
pytest tests/ -m physics
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with coverage
```bash
pytest tests/ --cov=ddstartup --cov-report=html
```

### Run and stop at first failure
```bash
pytest tests/ -x
```

## 📝 Writing New Tests

1. **Copy the template:**
   ```bash
   cp tests/test_module_template.py tests/unit/physics/test_your_module.py
   ```

2. **Modify the template:**
   - Replace imports with your actual module
   - Implement test functions
   - Use fixtures from `conftest.py`

3. **Use appropriate markers:**
   ```python
   @pytest.mark.slow
   @pytest.mark.physics
   @pytest.mark.numba
   ```

## 🎯 Available Fixtures

Fixtures are defined in `conftest.py` and automatically available in all tests:

### Directory fixtures
- `temp_dir` - Temporary directory (auto-cleanup)
- `sample_output_dir` - Sample output directory structure

### Physics parameter fixtures
- `typical_plasma_params` - Standard plasma parameters
- `typical_power_params` - Power and breeding parameters
- `typical_fuel_cycle_params` - Fuel cycle parameters
- `typical_economic_params` - Economic parameters
- `typical_reaction_rates` - Reaction rate values
- `all_typical_params` - Combined dictionary

### Numerical data fixtures
- `sample_time_series` - Time array for testing
- `sample_density_evolution` - Density evolution data
- `sample_inventory_evolution` - Inventory evolution data

### Helper fixtures
- `assert_close` - Numerical comparison helper
- `assert_positive` - Positivity check helper
- `assert_finite` - Finite value check helper

## 🏷️ Test Markers

Custom markers for organizing tests:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (skip with `-m "not slow"`)
- `@pytest.mark.physics` - Physics-related tests
- `@pytest.mark.io` - I/O-related tests
- `@pytest.mark.numba` - Tests involving Numba JIT

## ✅ Best Practices

1. **One test, one assertion (mostly)**
   - Each test should verify one specific behavior
   - Use parametrize for testing multiple inputs

2. **Use descriptive names**
   ```python
   def test_function_returns_positive_value_for_valid_input():
   ```

3. **Use fixtures for setup**
   ```python
   def test_with_typical_params(typical_plasma_params):
       result = my_function(**typical_plasma_params)
       assert result > 0
   ```

4. **Test edge cases**
   - Zero inputs
   - Negative inputs
   - Very large/small values
   - None values

5. **Test error handling**
   ```python
   def test_raises_value_error_for_negative_input():
       with pytest.raises(ValueError):
           my_function(-1.0)
   ```

6. **Use parametrize for multiple cases**
   ```python
   @pytest.mark.parametrize("input,expected", [
       (1.0, 1.0),
       (2.0, 4.0),
       (3.0, 9.0),
   ])
   def test_function(input, expected):
       assert my_function(input) == expected
   ```

## 🔧 Configuration

The existing `pytest.ini` in the project root is used for global configuration.
You can override settings in `tests/conftest.py` if needed.

## 📊 Coverage Reports

Generate HTML coverage reports:
```bash
pytest tests/ --cov=ddstartup --cov-report=html
```

View the report:
```bash
firefox htmlcov/index.html
```

## 🐛 Debugging Tests

### Run with more verbose output
```bash
pytest tests/ -vv
```

### Show print statements
```bash
pytest tests/ -s
```

### Drop into debugger on failure
```bash
pytest tests/ --pdb
```

### Run last failed tests only
```bash
pytest tests/ --lf
```

## 📚 Example Test

```python
import pytest
import numpy as np
from ddstartup.physics.reaction_rates import sigmav_DT_BoschHale

class TestReactionRates:
    """Test reaction rate calculations."""
    
    def test_DT_rate_at_typical_temperature(self, assert_close):
        """Test DT reaction rate at typical ion temperature."""
        T_i = 69.0  # keV
        result = sigmav_DT_BoschHale(T_i)
        
        # Should return reasonable value in m^3/s
        assert result > 0
        assert result < 1e-20  # Upper bound
        assert_close(result, 1e-21, rtol=0.5)  # Order of magnitude check
    
    @pytest.mark.parametrize("T_i", [10, 50, 100, 200])
    def test_DT_rate_increases_with_temperature(self, T_i):
        """Test that DT rate generally increases with temperature."""
        result = sigmav_DT_BoschHale(T_i)
        assert np.isfinite(result)
        assert result > 0
```

## 🔗 Related Files

- `../pytest.ini` - Global pytest configuration
- `../conftest.py` (old) - Original conftest (for reference)
- `../tests/` - Original test suite (for migration reference)
