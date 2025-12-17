"""
Pytest configuration and shared fixtures.

This file is automatically discovered by pytest and provides fixtures
that can be used across all test files.
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import tempfile
import shutil

# Ensure Numba caching does not require repository write access during tests
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())
# Prefer pure-Python execution in tests to avoid cache locator issues
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# Add parent directory to path for imports
dd_startup_root = Path(__file__).parent.parent
sys.path.insert(0, str(dd_startup_root))


# ============================================================================
# DIRECTORY AND FILE FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


# ============================================================================
# PHYSICS PARAMETER FIXTURES
# ============================================================================

@pytest.fixture
def typical_plasma_params():
    """Typical plasma parameters for testing."""
    return {
        'V_plasma': 100.0,      # m^3
        'n_tot': 1.5e20,        # m^-3
        'T_i': 14.0,            # keV
        'tau_p_T': 1.0,         # s
        'tau_p_He3': 1.0,       # s
    }


@pytest.fixture
def assert_close():
    """Helper function for numerical comparisons."""
    def _assert_close(a, b, rtol=1e-5, atol=1e-8, msg=""):
        """Assert that two values are close within tolerances."""
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            raise AssertionError(
                f"{msg}\nExpected: {b}\nGot: {a}\n"
                f"Relative error: {np.abs((a - b) / b) if b != 0 else np.abs(a - b)}"
            )
    return _assert_close


@pytest.fixture
def assert_positive():
    """Helper to assert values are positive."""
    def _assert_positive(value, name="value"):
        """Assert that a value is positive."""
        if not np.all(value > 0):
            raise AssertionError(f"{name} should be positive, got {value}")
    return _assert_positive


@pytest.fixture
def assert_finite():
    """Helper to assert values are finite."""
    def _assert_finite(value, name="value"):
        """Assert that a value is finite."""
        if not np.all(np.isfinite(value)):
            raise AssertionError(f"{name} should be finite, got {value}")
    return _assert_finite
