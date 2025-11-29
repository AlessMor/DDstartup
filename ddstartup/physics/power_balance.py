"""
Power balance calculations for DD Startup Analysis.

This module provides unified postprocessing functions for calculating
power and energy metrics from both lump and T-seeded solver outputs.
"""

import numpy as np
from numba import njit
from ddstartup.utils.units_and_constants import E_DDn, E_DDp, E_DT, E_DHe3
from ddstartup.utils.tools import trapz_numba
from ddstartup.physics.radiation import (
    calculate_total_radiation_power
)


@njit(cache=True, fastmath=True)
def calculate_P_aux_from_power_balance(
    n_T, n_D, T_i, V_plasma, 
    sigmav_DD_p, sigmav_DD_n, sigmav_DT,
    tau_E, n_He3=0.0,
    Z_eff=1
):
    """
    Calculate required auxiliary heating power from power balance.
    
    P_aux = P_rad + P_confinement - P_charged
    
    where:
    - P_rad = P_brem + P_line + P_synch (radiation losses)
    - P_confinement = 3*n*T*V/tau_E (energy confinement loss)
    - P_charged = power from charged fusion products (He3 from DDn, p from DDp, alpha from DT)
    
    Args:
        n_T: Tritium density (m⁻³)
        n_D: Deuterium density (m⁻³)
        T_i: Ion temperature (keV)
        V_plasma: Plasma volume (m³)
        sigmav_DD_p, sigmav_DD_n: DD reaction rates (m³/s)
        sigmav_DT: DT reaction rate (m³/s)
        tau_E: Energy confinement time (s)
        Z_eff: Effective charge (default 1)
        
    Returns:
        float: P_aux in Watts (minimum 0)
        
    Notes:
        Used by both lump and T-seeded analyses when P_aux is not specified.
        Negative P_aux (self-heating plasma) is clipped to 0.
    """
    # Calculate radiation power (returns tuple: (P_total, P_brems, P_line, P_sync))
    P_rad, _, _, _ = calculate_total_radiation_power(n_e = n_T+n_D+n_He3, T_e=T_i, Z_eff=Z_eff, V_plasma=V_plasma)
    
    # Confinement losses: 3*n*T*V/tau_E
    n_tot = n_T + n_D
    T_i_eV = T_i * 1000  # Convert keV to eV
    T_i_J = T_i_eV * 1.60218e-19  # Convert eV to Joules
    P_confinement = 3 * n_tot * T_i_J * V_plasma / tau_E
    
    # Energy from charged particles (in eV, convert to Joules)
    E_He3_DDn = 0.82e6 * 1.60218e-19  # He3 from D(d,n)He3, J
    E_p_DDp = 3.02e6 * 1.60218e-19    # Proton from D(d,p)T, J
    E_T_DDp = 1.01e6 * 1.60218e-19    # Triton from D(d,p)T, J
    E_alpha_DT = 3.5e6 * 1.60218e-19  # Alpha from D-T, J
    
    # Reaction rates
    R_DDn = 0.5 * n_D * n_D * sigmav_DD_n * V_plasma
    R_DDp = 0.5 * n_D * n_D * sigmav_DD_p * V_plasma
    R_DT = n_D * n_T * sigmav_DT * V_plasma
    
    # Charged particle power (particles that stay in plasma and heat it)
    P_charged = (R_DDn * E_He3_DDn +  # He3 from DDn
                 R_DDp * (E_p_DDp + E_T_DDp) +  # p and T from DDp
                 R_DT * E_alpha_DT)  # alpha from DT
    
    # Power balance: P_aux + P_charged = P_rad + P_confinement
    # Therefore: P_aux = P_rad + P_confinement - P_charged
    P_aux = P_rad + P_confinement - P_charged
    
    # Clip to minimum of 0 (can't have negative auxiliary heating)
    P_aux = max(0.0, P_aux)
    
    return P_aux


@njit(cache=True, fastmath=True)
def compute_fusion_powers(n_D, n_T, n_tot, V_plasma, 
                          sigmav_DD_p, sigmav_DD_n, sigmav_DT,
                          n_He3=0.0, sigmav_DHe3=0.0):
    """
    JIT-compiled core function to compute fusion powers from densities.
    
    Optimized with fastmath for both scalar and array inputs.
    Works for both lump (scalar) and T-seeded (array) analyses.
    
    Args:
        n_D: Deuterium density (m⁻³) - scalar or array
        n_T: Tritium density (m⁻³) - scalar or array
        n_tot: Total density (m⁻³) - scalar
        V_plasma: Plasma volume (m³) - scalar
        sigmav_DD_p, sigmav_DD_n, sigmav_DT: Reaction rates (m³/s) - scalars
        n_He3: Helium-3 density (m⁻³) - scalar or array (default 0.0)
        sigmav_DHe3: DHe3 reaction rate (m³/s) - scalar (default 0.0)
        
    Returns:
        tuple: (P_DDn, P_DDp, P_DT, P_DHe3, P_DT_eq) in Watts
            - P_DDn, P_DDp, P_DT, P_DHe3: Can be scalar or array
            - P_DT_eq: Always scalar (equilibrium reference)
            
    Notes:
        - For T-seeded analysis: pass n_He3=0.0 (default) to skip DHe3
        - For lump analysis: pass actual n_He3 and sigmav_DHe3 values
    """
    # Pre-compute common terms
    half_sigmav_DD_n = 0.5 * sigmav_DD_n
    half_sigmav_DD_p = 0.5 * sigmav_DD_p
    n_D_squared = n_D * n_D
    n_D_n_T = n_D * n_T
    
    # Pre-multiply volume and energy terms (constants)
    V_E_DDn = V_plasma * E_DDn
    V_E_DDp = V_plasma * E_DDp
    V_E_DT = V_plasma * E_DT
    V_E_DHe3 = V_plasma * E_DHe3
    
    # Calculate fusion powers
    P_DDn = n_D_squared * half_sigmav_DD_n * V_E_DDn
    P_DDp = n_D_squared * half_sigmav_DD_p * V_E_DDp
    P_DT = n_D_n_T * sigmav_DT * V_E_DT
    P_DHe3 = n_D * n_He3 * sigmav_DHe3 * V_E_DHe3
    
    # DT equilibrium power (50-50 mixture, always scalar)
    P_DT_eq = 0.25 * n_tot * n_tot * sigmav_DT * V_E_DT
    
    return P_DDn, P_DDp, P_DT, P_DHe3, P_DT_eq


@njit(cache=True, fastmath=True)
def compute_lump_powers_and_energies(
    n_T, n_D, n_He3, t_startup,
    V_plasma, sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3,
    P_aux, P_aux_DT_eq
):
    """
    Compute fusion powers and energies for lump analysis (steady-state).
    
    Uses JIT-compiled core function for power calculations, then applies
    steady-state energy calculation (power × time). Includes DHe3 reaction.
    
    Args:
        n_T: Steady-state tritium density (m⁻³)
        n_D: Steady-state deuterium density (m⁻³)
        n_He3: Steady-state helium-3 density (m⁻³)
        t_startup: Startup time (s)
        V_plasma: Plasma volume (m³)
        sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3: Reaction rates (m³/s)
        P_aux: Auxiliary heating power during DD startup (W)
        P_aux_DT_eq: Auxiliary power for DT equilibrium (W)
        
    Returns:
        dict: Power and energy metrics
            - P_DDn, P_DDp, P_DT, P_DT_eq: Fusion powers (W)
            - P_fusion_total: Total fusion power (W)
            - E_fusion_DD: Total DD fusion energy (J)
            - E_fusion_DT_eq: DT equilibrium fusion energy (J)
            - E_aux_DD: Auxiliary energy during DD startup (J)
            - E_aux_DT_eq: Auxiliary energy for DT equilibrium (J)
    """
    # Core power calculations (JIT-compiled, includes DHe3)
    n_tot = n_D  # Assuming n_D ≈ n_tot for DD startup
    P_DDn, P_DDp, P_DT, P_DHe3, P_DT_eq = compute_fusion_powers(
        n_D, n_T, n_tot, V_plasma, 
        sigmav_DD_p, sigmav_DD_n, sigmav_DT,
        n_He3, sigmav_DHe3
    )
    
    # Total fusion power
    P_fusion_total = P_DDn + P_DDp + P_DT + P_DHe3
    
    # Energies (steady-state × time)
    E_fusion_DD = P_fusion_total * t_startup
    E_fusion_DT_eq = P_DT_eq * t_startup
    E_aux_DD = P_aux * t_startup
    E_aux_DT_eq = P_aux_DT_eq * t_startup
    
    return {
        'P_DDn': P_DDn,
        'P_DDp': P_DDp,
        'P_DT': P_DT,
        'P_DT_eq': P_DT_eq,
        'P_fusion_total': P_fusion_total,
        'E_fusion_DD': E_fusion_DD,
        'E_fusion_DT_eq': E_fusion_DT_eq,
        'E_aux_DD': E_aux_DD,
        'E_aux_DT_eq': E_aux_DT_eq
    }


def compute_tseeded_powers_and_energies(
    t_startup, t_raw, n_T_raw, n_D_raw,
    N_ofc_raw, N_ifc_raw, N_st_raw,
    n_tot, V_plasma, 
    sigmav_DD_p, sigmav_DD_n, sigmav_DT,
    tau_ifc,
    P_aux, P_aux_DT_eq,
    injection_rate_max, N_st_min,
    vector_length
):
    """
    Compute fusion powers and energies for T-seeded analysis (time-dependent).
    
    Uses JIT-compiled core function for power calculations with interpolated
    density arrays, then integrates to get energies. No DHe3 reaction.
    
    Args:
        t_startup: Time to reach DT operation (s)
        t_raw: Raw time points from ODE solver (array)
        n_T_raw: Raw tritium density from solver (m⁻³, array)
        n_D_raw: Raw deuterium density from solver (m⁻³, array)
        N_ofc_raw, N_ifc_raw, N_st_raw: Raw tritium inventories (atoms, arrays)
        n_tot: Total particle density (m⁻³, scalar)
        V_plasma: Plasma volume (m³, scalar)
        sigmav_DD_p, sigmav_DD_n, sigmav_DT: Reaction rates (m³/s, scalars)
        TBR_DT, TBR_DDn: Tritium breeding ratios
        tau_ifc: In-fuel-cycle time (s)
        P_aux, P_aux_DT_eq: Auxiliary powers (W)
        injection_rate_max: Max injection rate (atoms/s)
        N_st_min: Min stored tritium (atoms)
        vector_length: Length of output arrays
        
    Returns:
        dict: Interpolated time series and integrated energies
            - t_interp: Uniform time grid (s)
            - n_T, n_D: Interpolated densities (m⁻³)
            - N_ofc, N_ifc, N_st: Interpolated inventories (atoms)
            - P_DDn, P_DDp, P_DT: Power arrays (W)
            - P_DT_eq: Scalar DT-equilibrium power (W)
            - P_DT_eq_profile: DT-equilibrium power repeated across the time grid (W)
            - TBE: Tritium breeding efficiency
            - E_fusion_DD: Total DD fusion energy (J)
            - E_fusion_DT_eq: DT equilibrium fusion energy (J)
            - E_aux_DD: Auxiliary energy during DD startup (J)
            - E_aux_DT_eq: Auxiliary energy for DT equilibrium (J)
    """
    # Create uniform time grid
    dt = t_startup / (vector_length - 1)
    t_interp = np.arange(vector_length) * dt
    
    # Interpolate all raw quantities
    N_ofc = np.interp(t_interp, t_raw, N_ofc_raw)
    N_ifc = np.interp(t_interp, t_raw, N_ifc_raw)
    N_st = np.interp(t_interp, t_raw, N_st_raw)
    n_T = np.interp(t_interp, t_raw, n_T_raw)
    n_D = np.interp(t_interp, t_raw, n_D_raw)
    
    # Core power calculations (JIT-compiled, n_He3=0.0 for T-seeded)
    P_DDn, P_DDp, P_DT, P_DHe3, P_DT_eq_scalar = compute_fusion_powers(
        n_D, n_T, n_tot, V_plasma, 
        sigmav_DD_p, sigmav_DD_n, sigmav_DT
        # n_He3 defaults to 0.0, sigmav_DHe3 defaults to 0.0
    )
    
    # Broadcast P_DT_eq for any time-series operations, but keep scalar for storage
    P_DT_eq_profile = np.full_like(P_DDn, P_DT_eq_scalar)
    
    # Integrate powers to get energies
    E_fusion_DDn = trapz_numba(P_DDn, t_interp)
    E_fusion_DDp = trapz_numba(P_DDp, t_interp)
    E_fusion_DT = trapz_numba(P_DT, t_interp)
    E_fusion_DD = E_fusion_DDn + E_fusion_DDp + E_fusion_DT
    E_fusion_DT_eq = P_DT_eq_scalar * t_startup
    
    # Auxiliary energies
    E_aux_DD = P_aux * t_startup
    E_aux_DT_eq = P_aux_DT_eq * t_startup
    
    # Compute TBE (tritium breeding efficiency)
    from ddstartup.utils.units_and_constants import lambda_T
    
    # Injection rate calculation
    inj_temp = N_ifc / tau_ifc - lambda_T * N_st
    inj_rate = np.clip(inj_temp, 0.0, injection_rate_max)
    inj_rate = np.where(N_st > N_st_min, inj_rate, 0.0)
    
    # TBE calculation with safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        TBE_raw = (n_D * n_T * sigmav_DT * V_plasma) / inj_rate
    
    TBE = np.where(
        (N_st > N_st_min) & (inj_rate > 0),
        TBE_raw,
        np.nan
    )
    
    return {
        't_interp': t_interp,
        'n_T': n_T,
        'n_D': n_D,
        'N_ofc': N_ofc,
        'N_ifc': N_ifc,
        'N_st': N_st,
        'P_DDn': P_DDn,
        'P_DDp': P_DDp,
        'P_DT': P_DT,
        'P_DT_eq': P_DT_eq_scalar,
        'P_DT_eq_profile': P_DT_eq_profile,
        'TBE': TBE,
        'E_fusion_DD': E_fusion_DD,
        'E_fusion_DT_eq': E_fusion_DT_eq,
        'E_aux_DD': E_aux_DD,
        'E_aux_DT_eq': E_aux_DT_eq
    }
