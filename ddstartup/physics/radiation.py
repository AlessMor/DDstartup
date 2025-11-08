"""
Radiation power calculations for DD Startup Analysis.

This module provides functions for computing radiation losses from
bremsstrahlung, line radiation, and synchrotron radiation.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def calculate_bremsstrahlung_power(n_e, T_e, Z_eff, V_plasma):
    """
    Calculate Bremsstrahlung radiation power.
    
    Uses the standard formula for free-free radiation from electron-ion
    collisions in a fully ionized plasma.
    
    Args:
        n_e: Electron density (m⁻³)
        T_e: Electron temperature (keV)
        Z_eff: Effective charge (dimensionless)
        V_plasma: Plasma volume (m³)
        
    Returns:
        float: P_Bremsstrahlung in Watts
        
    Notes:
        Formula taken from Stott 2005 - as implemented in cfspopcon 
        
    """
    ne20 = n_e / 1e20  # Convert from m^-3 to units of 10^20 m^-3

    Tm = 511.0  # keV, Tm = m_e * c**2
    xrel = (1.0 + 2.0 * T_e / Tm) * (
        1.0 + (2.0 / Z_eff) * (1.0 - 1.0 / (1.0 + T_e / Tm))
    )  # relativistic correction factor

    Kb = ne20**2 * np.sqrt(T_e) * xrel * V_plasma
    
    P_brem: float = 5.35e-3 * Z_eff * Kb  # volume-averaged bremsstrahlung radiaton in MW

    return P_brem

@njit(cache=True, fastmath=True)
def calculate_line_radiation_power(n_e, T_e, V_plasma, L_z):
    """
    Calculate line radiation power (placeholder).
    
    Currently returns zero. Full implementation would include impurity
    species densities and line radiation coefficients.
    
    """
    return 0


@njit(cache=True, fastmath=True)
def calculate_synchrotron_power(n_e, T_e, V_plasma, B_field):
    """
    Calculate synchrotron radiation power (placeholder).
    
    Currently returns zero. Full implementation would include magnetic
    field strength and relativistic corrections.
    
    """
    # TODO: Implement full synchrotron calculation
    return 0.0


@njit(cache=True, fastmath=True)
def calculate_total_radiation_power(n_e, T_e, Z_eff, V_plasma, B_field=0.0, L_z=0.0):
    """
    Calculate total radiation power from all sources.
    
    Combines bremsstrahlung, line radiation, and synchrotron radiation.
    
    Args:
        n_e: Electron density (m⁻³)
        n_i: Ion density (m⁻³)
        n_impurity: Impurity density (m⁻³)
        T_e: Electron temperature (eV)
        Z_eff: Effective charge
        V_plasma: Plasma volume (m³)
        B_field: Magnetic field strength (T), default 0
        L_z: Radiative loss coefficient (W·m³), default 1e-37
        
    Returns:
        tuple: (P_rad_total, P_brems, P_line, P_sync) in Watts
        
    Examples:
        >>> P_total, P_b, P_l, P_s = calculate_total_radiation_power(
        ...     1e20, 1e20, 1e18, 10e3, 1.5, 100.0
        ... )
    """
    P_brems = calculate_bremsstrahlung_power(n_e, T_e, Z_eff, V_plasma)
    P_line = calculate_line_radiation_power(n_e, T_e, V_plasma, L_z)
    P_sync = calculate_synchrotron_power(n_e, T_e, V_plasma, B_field)
    
    P_rad_total = P_brems + P_line + P_sync
    
    return P_rad_total, P_brems, P_line, P_sync
