# Quick Start Guide for New Test Structure

## ✅ What Was Created

A complete pytest test structure has been set up in `tests/` with:

```
tests/
├── conftest.py                          # Shared fixtures and configuration
├── test_module_template.py              # Generic template (copy and modify)
├── README.md                            # Full documentation
│
├── unit/                                # Unit tests directory
│   ├── physics/                         # Physics tests
│   │   └── test_reaction_rates.py      # Example working test
│   ├── io/                              # I/O tests (empty, ready for you)
│   └── utils/                           # Utility tests (empty, ready for you)
│
├── integration/                         # Integration tests (empty, ready for you)
└── fixtures/                            # Shared test data (empty, ready for you)
```

## 🎯 Key Features

✅ **Organized subfolders** - Tests are organized by type (unit/integration) and module
✅ **Shared fixtures** - 15+ pre-configured fixtures in `conftest.py`
✅ **Custom markers** - `@pytest.mark.physics`, `@pytest.mark.slow`, etc.
✅ **Helper functions** - `assert_close()`, `assert_positive()`, `assert_finite()`
✅ **Working example** - 22 passing tests for reaction rates
✅ **Template file** - Copy and modify for new tests

## 🚀 Getting Started

### 1. Run the example tests
```bash
cd /home/alessmor/Scrivania/dd_startup
pytest tests/ -v
```

Expected: **44 tests passed** ✅

### 2. Create a new test file
```bash
# Copy the template
cp tests/test_module_template.py tests/unit/physics/test_my_module.py

# Edit and add your tests
nano tests/unit/physics/test_my_module.py
```

### 3. Use the fixtures
```python
def test_with_typical_params(typical_plasma_params):
    """Fixtures are automatically available!"""
    V_plasma = typical_plasma_params['V_plasma']
    # ... your test code
```

### 4. Run specific tests
```bash
# Run all physics tests
pytest tests/unit/physics/ -v

# Run only fast tests (skip slow ones)
pytest tests/ -m "not slow"

# Run with coverage
Generate HTML coverage reports:
```bash
pytest tests/ --cov=ddstartup --cov-report=html
```

View the report:
```

## 📝 Available Fixtures

All these are available in any test file without imports:

**Directories:**
- `temp_dir` - Temporary directory (auto-cleanup)

**Physics Parameters:**
- `typical_plasma_params` - V_plasma, n_tot, T_i, tau_p_T, tau_p_He3
- `typical_power_params` - P_aux, TBR_DT, TBR_DDn
- `typical_fuel_cycle_params` - tau_ifc, tau_ofc, I_target
- `typical_economic_params` - eta_th, capacity_factor, price_of_electricity
- `typical_reaction_rates` - sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3

**Helpers:**
- `assert_close(a, b, rtol=1e-5)` - Numerical comparison
- `assert_positive(value)` - Check positivity
- `assert_finite(value)` - Check for NaN/Inf

## 💡 Quick Example

```python
import pytest
from ddstartup.your_module import your_function

@pytest.mark.physics
class TestYourFunction:
    """Test your function."""
    
    def test_basic_case(self, typical_plasma_params):
        """Test with typical parameters."""
        result = your_function(**typical_plasma_params)
        assert result > 0
    
    @pytest.mark.parametrize("input,expected", [
        (1.0, 1.0),
        (2.0, 4.0),
    ])
    def test_multiple_inputs(self, input, expected):
        """Test multiple cases."""
        assert your_function(input) == expected
```

## 🔧 Configuration

The existing `pytest.ini` in the project root is used for global configuration.
You can override settings in `tests/conftest.py` if needed.

## 📚 Next Steps

1. **Copy the template** for each module you want to test
2. **Use the example** (`test_reaction_rates.py`) as a reference
3. **Start with core physics** (reaction rates, lump functions, tseeded functions)
4. **Move to I/O** (HDF5 read/write, parameter loading)
5. **Add integration tests** for complete workflows

## 📖 Full Documentation

See `tests/README.md` for complete documentation including:
- Directory structure details
- All available fixtures
- Best practices
- Marker descriptions
- Debugging tips

## ✨ Key Advantages

- **Organized**: Subfolders keep tests tidy
- **Reusable**: Fixtures eliminate boilerplate
- **Fast**: Skip slow tests during development
- **Clear**: Markers show what each test does
- **Maintainable**: Template makes new tests easy
