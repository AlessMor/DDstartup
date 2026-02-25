import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def trapz_numba(y, x):
    """
    JIT-compiled trapezoidal integration for Numba compatibility.
    
    Optimized with fastmath for additional speed. Equivalent to numpy.trapz
    but works inside @njit decorated functions. Automatically detects uniform
    grids for faster computation.
    
    Args:
        y: Function values (array)
        x: Independent variable values (array)
        
    Returns:
        float: Integrated value using trapezoidal rule
        
    Notes:
        - For uniform grids: O(n) with optimized sum
        - For non-uniform grids: O(n) with pairwise differences
        
    Examples:
        >>> x = np.linspace(0, 10, 100)
        >>> y = x**2
        >>> integral = trapz_numba(y, x)  # Should be ~333.33
    """
    n = len(x)
    if n < 2:
        return 0.0
    
    # For uniform grid (common case), use simplified formula
    dx = x[1] - x[0]
    is_uniform = True
    for i in range(2, n):
        if abs(x[i] - x[i-1] - dx) > 1e-10 * dx:
            is_uniform = False
            break
    
    if is_uniform:
        # FAST PATH: Uniform grid
        return dx * (0.5 * (y[0] + y[n-1]) + np.sum(y[1:n-1]))
    else:
        # SLOW PATH: Non-uniform grid
        s = 0.0
        for i in range(1, n):
            s += 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
        return s





def index_to_params(linear_index, param_shapes):
    """
    Convert linear index to multi-dimensional parameter indices.
    
    Special case for filtered data: If param_shapes is [1, 1, ..., 1, N],
    this indicates filtered data where all parameters are aligned at the same index.
    In this case, return [linear_index, linear_index, ..., linear_index].
    """
    n_params = len(param_shapes)
    indices = np.zeros(n_params, dtype=np.int64)
    
    # Check if this is filtered data (all shapes are 1 except possibly the last)
    if n_params > 1 and np.all(param_shapes[:-1] == 1):
        # Filtered data: all parameters share the same linear index
        indices[:] = linear_index
        return indices
    
    # Standard grid-based indexing
    remaining = linear_index
    for i in range(n_params - 1, -1, -1):
        indices[i] = remaining % param_shapes[i]
        remaining = remaining // param_shapes[i]
    
    return indices


# ============================================================================
# PARAMETER DEFINITIONS - Now using centralized parameter management
# ============================================================================
# These are maintained for backwards compatibility but now generated
# dynamically from the PARAMETER_SCHEMA in parameter_registry.py

from .parameter_registry import get_registry

# Get registry instance
_registry = get_registry()

# Generate lists dynamically from parameter registry
inputs_names = _registry.get_input_names()  # All inputs across all analysis types
outputs_names = _registry.get_output_names()  # All outputs across all analysis types

# Parameter keys by analysis type
PARAM_KEYS = {
    'T_seeded': _registry.get_input_names('T_seeded'),
    'lump': _registry.get_input_names('lump')
}

# Parameter units
PARAM_UNITS = _registry.get_units_dict()


def fix_vector_length(vec, target_length=100):
    vec = np.asarray(vec)
    if vec.ndim == 0 or vec.size == 0:
        # If scalar or empty, fill with scalar value or nan
        scalar = float(vec) if vec.size == 1 or vec.ndim == 0 else np.nan
        return np.full(target_length, scalar)
    if vec.size == 1:
        # Repeat the single value
        return np.full(target_length, vec[0])
    if vec.size == target_length:
        return vec
    # Interpolate to target length, preserving first and last value
    x_old = np.linspace(0, 1, vec.size)
    x_new = np.linspace(0, 1, target_length)
    vec_interp = np.interp(x_new, x_old, vec)
    return vec_interp

