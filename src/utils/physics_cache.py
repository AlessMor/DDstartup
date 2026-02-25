"""
Physics computation caching utilities.

This module provides caching for expensive physics calculations that are
frequently repeated, particularly reaction rate calculations.
"""

import numpy as np
from src.physics.reactivity_functions import sigmav_DD_BoschHale, sigmav_DT_BoschHale, sigmav_DHe3_BoschHale


# Global cache for reaction rates
_sigmav_cache = {}


def get_cached_reaction_rates(T_i, include_DHe3=False):
    """
    Get reaction rates with caching to avoid recomputation for repeated T_i values.
    
    Caches reaction rates at 0.1 eV resolution to balance memory and performance.
    Can be used by both T-seeded and lump analyses.
    
    Args:
        T_i: Ion temperature in eV (scalar or array element)
        include_DHe3: If True, also return sigmav_DHe3 (needed for lump analysis)
        
    Returns:
        If include_DHe3=False: Tuple of (sigmav_DD_p, sigmav_DD_n, sigmav_DT)
        If include_DHe3=True: Tuple of (sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3)
        
    Notes:
        - Rounding to 0.1 eV provides good balance between accuracy and cache hits
        - Cache persists across function calls within same Python session
        - For array inputs, extract scalar value first: T_i = T_i_array[0]
        
    Examples:
        >>> # For T-seeded analysis
        >>> sigmav_DD_p, sigmav_DD_n, sigmav_DT = get_cached_reaction_rates(17e3)
        
        >>> # For lump analysis (needs DHe3)
        >>> rates = get_cached_reaction_rates(17e3, include_DHe3=True)
        >>> sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3 = rates
    """
    # Round to 0.1 eV precision for caching
    T_i_key = round(T_i / 0.1) * 0.1
    
    # Create cache key that includes DHe3 flag
    cache_key = (T_i_key, include_DHe3)
    
    if cache_key not in _sigmav_cache:
        # Compute reaction rates
        T_i_array = np.array([T_i])
        sigmav_DD_results = sigmav_DD_BoschHale(T_i_array)
        sigmav_DD_p = sigmav_DD_results[2][0]
        sigmav_DD_n = sigmav_DD_results[1][0]
        sigmav_DT = sigmav_DT_BoschHale(T_i_array)[0]
        
        if include_DHe3:
            sigmav_DHe3 = sigmav_DHe3_BoschHale(T_i_array)[0]
            _sigmav_cache[cache_key] = (sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3)
        else:
            _sigmav_cache[cache_key] = (sigmav_DD_p, sigmav_DD_n, sigmav_DT)
    
    return _sigmav_cache[cache_key]
