"""
ODE solver utilities for DD startup analysis.
Contains ODE functions, cross-section caching, and single combination solver.
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from .sigmav_functions import sigmav_DD_BoschHale, sigmav_DT_BoschHale
from .units_and_constants import *


# Cross-section caching for performance optimization
_sigmav_cache = {}

def get_cached_sigmav(T_i_keV):
    """Cache expensive Bosch-Hale cross-section calculations"""
    T_i_rounded = round(T_i_keV, 2)
    
    if T_i_rounded not in _sigmav_cache:
        sigmav_DD_results = sigmav_DD_BoschHale(T_i_rounded * u.keV)
        sigmav_DD_p = sigmav_DD_results[1].to('m^3/s').magnitude  
        sigmav_DD_n = sigmav_DD_results[2].to('m^3/s').magnitude
        sigmav_DT = sigmav_DT_BoschHale(T_i_rounded * u.keV).to('m^3/s').magnitude
        
        _sigmav_cache[T_i_rounded] = (sigmav_DD_p, sigmav_DD_n, sigmav_DT)
    
    return _sigmav_cache[T_i_rounded]


def pre_populate_sigmav_cache(T_i_field):
    """Pre-calculate cross-sections for all T_i values to avoid repeated calculations"""
    T_i_values = T_i_field.data.to('keV').magnitude
    for T_i in T_i_values:
        get_cached_sigmav(T_i)  # This will cache the values
    print(f"Pre-cached cross-sections for {len(T_i_values)} T_i values")


# Global variables for fast ODE computation (avoid units overhead)
_ode_globals = {}


def injection_rate_fun(t, N_ifc, N_st, n_T, N_st_min=None):
    """Injection rate function using global variables for numerical stability"""
    if N_st_min is None:
        N_st_min = _ode_globals['N_st_min']
    
    g = _ode_globals
    n_tot = g['n_tot'] * u.m**(-3)
    tau_p_T = g['tau_p_T'] * u.s
    V_plasma = g['V_plasma'] * u.m**3
    sigmav_DT = g['sigmav_DT'] * u.m**3/u.s
    sigmav_DD_p = g['sigmav_DD_p'] * u.m**3/u.s
    tau_ifc = g['tau_ifc'] * u.s
    lambda_T = g['lambda_T'] * u.s**(-1)
    
    # Calculate injection_rate_max like working script
    injection_rate_max = (n_tot/2/tau_p_T*V_plasma + 0.25*n_tot**2*sigmav_DT*V_plasma - 0.25/2*n_tot**2*sigmav_DD_p*V_plasma).to('1/s')
    
    if N_st < N_st_min:
        return 0 * u.s**(-1)
    else:
        return min((N_ifc/tau_ifc - lambda_T*N_st), injection_rate_max).to('1/s')


def tritium_inventory_odes_global_fast(t, y):
    """Optimized ODE function using global variables - no units conversion overhead"""
    N_ofc, N_ifc, N_st, n_T = y[0], y[1], y[2], y[3]
    
    g = _ode_globals
    n_tot = g['n_tot']
    V_plasma = g['V_plasma'] 
    tau_p_T = g['tau_p_T']
    tau_ifc = g['tau_ifc']
    tau_ofc = g['tau_ofc']
    sigmav_DD_p = g['sigmav_DD_p']
    sigmav_DD_n = g['sigmav_DD_n']
    sigmav_DT = g['sigmav_DT']
    TBR_DDn = g['TBR_DDn']
    TBR_DT = g['TBR_DT']
    lambda_T = g['lambda_T']
    N_st_min = g['N_st_min']
    
    n_D = n_tot - n_T
    
    # Calculate injection_rate_max (pre-computed and stored)
    injection_rate_max = g['injection_rate_max']
    
    # Injection rate logic - simplified
    if N_st < N_st_min:
        injection_rate = 0.0
    else:
        injection_rate = min((N_ifc/tau_ifc - lambda_T*N_st), injection_rate_max)
    
    # Reaction rates - no units overhead
    Tdot_DDn = TBR_DDn * 0.5 * n_D * n_D * sigmav_DD_n * V_plasma
    Tdot_DDp = 0.5 * n_D * n_D * sigmav_DD_p * V_plasma
    Tdot_DT = TBR_DT * n_D * n_T * sigmav_DT * V_plasma
    Tdot_burn = n_D * n_T * sigmav_DT * V_plasma

    # ODE system - direct calculations
    dN_ofc_dt = Tdot_DT + Tdot_DDn - N_ofc / tau_ofc - N_ofc * lambda_T
    dN_ifc_dt = N_ofc / tau_ofc - N_ifc / tau_ifc - lambda_T * N_ifc + n_T / tau_p_T * V_plasma
    dN_stor_dt = N_ifc / tau_ifc - lambda_T * N_st - injection_rate
    dnT_dt = injection_rate / V_plasma + Tdot_DDp / V_plasma - n_T / tau_p_T - Tdot_burn / V_plasma

    return [dN_ofc_dt, dN_ifc_dt, dN_stor_dt, dnT_dt]


def DT_reached_event_global_fast(t, y):
    """Fast event function for DT condition using global variables"""
    n_T = y[3]
    n_tot = _ode_globals['n_tot']
    return n_T - 0.5 * n_tot


def negative_event_global_fast(t, y):
    """Fast event function for negative states"""
    return min(y[0]+100, y[1]+100, y[2]+100, y[3]+100)


def create_result_row(param_combo, extracted_data, t_startup, sol, sigmav_DD_p, sigmav_DD_n, sigmav_DT):
    """
    Create a result row from the solution
    Optimized to return minimal objects for memory efficiency
    """
    # Unpack extracted_data again
    (V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, P_aux, P_lost_rad, 
     P_aux_all_DT, P_lost_rad_all_DT, TBR_DT, TBR_DDn, tau_ifc, 
     tau_ofc, eta_th, plant_avail, Cost_per_kWh) = extracted_data
    
    if sol is None or t_startup == np.inf or not sol.success:
        # Failed case - return all input parameters and fill outputs with np.nan or np.inf
        return [
            V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, P_aux, P_lost_rad,
            P_aux_all_DT, P_lost_rad_all_DT, TBR_DT, TBR_DDn, tau_ifc,
            tau_ofc, eta_th, plant_avail, Cost_per_kWh,
            np.inf,  # t_startup
            np.nan,  # P_fusion_startup
            np.nan,  # E_lost
            np.nan,  # Dollar_Lost
            np.nan,  # P_fusion_DD_startup
            np.nan,  # P_fusion_DT_startup
            np.nan,  # avg_n_T_startup
            np.nan,  # avg_n_D_startup
        ]
    
    # Calculate physics results using dense solution (more efficient)
    # Evaluate at startup time and a few intermediate points for averaging
    t_points = np.array([0, t_startup * 0.5, t_startup]) if t_startup < np.inf else np.array([0, sol.t[-1]])
    
    try:
        y_eval = sol.sol(t_points)  # Use dense output
        n_T_vals = y_eval[3]  # Tritium density values
        n_D_vals = n_tot - n_T_vals  # Deuterium density values
        
        # Use final values for startup calculations
        n_T_final = n_T_vals[-1]
        n_D_final = n_D_vals[-1]
        
    except Exception:
        # If dense evaluation fails, use last time point
        n_T_final = sol.y[3][-1] if len(sol.y[3]) > 0 else 0
        n_D_final = max(n_tot - n_T_final, 0)
        n_T_vals = np.array([n_T_final])
        n_D_vals = np.array([n_D_final])
    
    # Calculate fusion power at startup using precomputed cross-sections
    # Use scalar math for efficiency
    P_fusion_DD_startup = 0.5 * n_D_final * n_D_final * (sigmav_DD_n + sigmav_DD_p) * V_plasma * E_DDn.magnitude * 1.602e-13
    P_fusion_DT_startup = n_D_final * n_T_final * sigmav_DT * V_plasma * E_DT.magnitude * 1.602e-13
    P_fusion_startup = P_fusion_DD_startup + P_fusion_DT_startup
    
    # Simplified energy loss calculation (avoid expensive integration for screening)
    if t_startup < np.inf:
        # Rough estimate of energy lost during startup
        avg_n_T = np.mean(n_T_vals) if len(n_T_vals) > 1 else n_T_final
        avg_n_D = np.mean(n_D_vals) if len(n_D_vals) > 1 else n_D_final
        
        # Approximate average fusion power during startup
        avg_P_fusion = (0.5 * avg_n_D * avg_n_D * (sigmav_DD_n + sigmav_DD_p) * V_plasma * E_DDn.magnitude + 
                       avg_n_D * avg_n_T * sigmav_DT * V_plasma * E_DT.magnitude) * 1.602e-13
        
        # Energy lost is roughly (P_aux - avg_P_fusion) * t_startup
        P_net_loss = max(P_aux - avg_P_fusion - P_lost_rad, 0)
        E_lost = P_net_loss * t_startup
        
        # Economic calculation
        E_lost_kWh = E_lost / 3.6e6  # Convert J to kWh
        Dollar_Lost = E_lost_kWh * Cost_per_kWh
    else:
        avg_n_T = np.nan
        avg_n_D = np.nan
        E_lost = np.nan
        Dollar_Lost = np.nan
    
    return [
        V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, P_aux, P_lost_rad,
        P_aux_all_DT, P_lost_rad_all_DT, TBR_DT, TBR_DDn, tau_ifc,
        tau_ofc, eta_th, plant_avail, Cost_per_kWh,
        t_startup,
        P_fusion_startup,
        E_lost,
        Dollar_Lost,
        P_fusion_DD_startup,
        P_fusion_DT_startup,
        avg_n_T,
        avg_n_D,
    ]


def solve_single_combination(param_combo, input_data, total_time):
    """
    Solve ODE system for a single parameter combination
    Uses working script approach for numerical stability
    """
    solve_start_time = time.time()
    
    # Convert constants to magnitude values for speed (no units in calculations)
    tritium_mass_mag = tritium_mass.to('kg').magnitude
    lambda_T_mag = lambda_T.to('1/s').magnitude
    E_DDn_mag = E_DDn.to('J').magnitude
    E_DDp_mag = E_DDp.to('J').magnitude
    E_DT_mag = E_DT.to('J').magnitude
    
    # Extract data for each parameter
    extracted_data = [input_data[i][param_idx] for i, param_idx in enumerate(param_combo)]
        
    # Unpack the extracted data (all unit-stripped values)
    (V_plasma, T_i, n_tot,
     tau_p_T, tau_p_He3, 
     P_aux, P_lost_rad, P_aux_all_DT, P_lost_rad_all_DT,
     
     TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
     
     eta_th, plant_avail, Cost_per_kWh,
     ) = extracted_data

    # OPTIMIZATION: Quick validity check to avoid expensive computations
    if (T_i <= 0 or n_tot <= 0 or V_plasma <= 0 or 
        tau_p_T <= 0 or tau_p_He3 <= 0 or P_aux <= 0):
        # Return failure case quickly
        return create_result_row(param_combo, extracted_data, np.inf, None, 0, 0, 0)

    # Use cached cross-sections for better performance
    sigmav_DD_p, sigmav_DD_n, sigmav_DT = get_cached_sigmav(T_i)
    
    # Calculate injection_rate_max with proper physics
    injection_rate_max = (
        n_tot/2/tau_p_T*V_plasma + 
        0.25*n_tot*n_tot*sigmav_DT*V_plasma - 
        0.25/2*n_tot*n_tot*sigmav_DD_p*V_plasma
    )
    
    # Calculate N_st_min
    N_st_min = 0.001/tritium_mass_mag
    
    # Use proper initial conditions
    y0 = np.zeros(4)  # [N_ofc, N_ifc, N_st, n_T] all start at zero
    
    # Time span (total_time already in seconds)
    t_span = (0, total_time)
    
    # Enhanced solver settings selection based on stiffness analysis
    time_scales = [tau_p_T, tau_ifc, tau_ofc]
    min_time_scale = max(min(time_scales), 1e-12)  # Prevent zero
    max_time_scale = max(time_scales)
    stiffness_ratio = max_time_scale / min_time_scale
    
    # OPTIMIZATION: Adaptive solver settings based on stiffness
    if stiffness_ratio > 100000:  # Very stiff system
        method = 'Radau'         # Better for very stiff problems
        rtol = 1e-3              # More relaxed for stiff cases
        atol = 1e-6              
        max_step = min_time_scale * 10  # Much smaller steps for stiff systems
    elif stiffness_ratio > 10000:  # Moderately stiff
        method = 'BDF'           
        rtol = 1e-4              
        atol = 1e-7              
        max_step = min_time_scale * 50
    else:  # Not very stiff
        method = 'BDF'           
        rtol = 1e-4              
        atol = 1e-7              
        max_step = total_time/100
    
    first_step = min_time_scale / 100  # Start with very small step for stiff problems
    
    try:
        # Set up global variables for fast ODE function
        global _ode_globals
        _ode_globals = {
            'n_tot': n_tot,
            'sigmav_DD_p': sigmav_DD_p,
            'sigmav_DD_n': sigmav_DD_n,
            'sigmav_DT': sigmav_DT,
            'V_plasma': V_plasma,
            'TBR_DDn': TBR_DDn,
            'TBR_DT': TBR_DT,
            'tau_p_T': tau_p_T,
            'tau_ifc': tau_ifc,
            'tau_ofc': tau_ofc,
            'injection_rate_max': injection_rate_max,
            'N_st_min': N_st_min,
            'lambda_T': lambda_T_mag
        }
        
        solve_kwargs = {
            'fun': tritium_inventory_odes_global_fast,
            't_span': t_span,
            'y0': y0,
            'method': method,
            'dense_output': False,
            't_eval': None,
            'events': [DT_reached_event_global_fast, negative_event_global_fast],
            'rtol': rtol,
            'atol': atol
        }
        
        # Only add optional parameters if they're not None
        if max_step is not None:
            solve_kwargs['max_step'] = max_step
        if first_step is not None:
            solve_kwargs['first_step'] = first_step
        
        # Set up events for fast approach
        DT_reached_event_global_fast.terminal = True
        DT_reached_event_global_fast.direction = 1
        negative_event_global_fast.terminal = True
        negative_event_global_fast.direction = -1
            
        # Solve with optimized fast approach
        sol = solve_ivp(**solve_kwargs)
        
        solve_end_time = time.time()
        solve_duration = solve_end_time - solve_start_time
        
        # Process results - check events first
        if not sol.success:
            # Integration failed
            t_startup = np.inf
            return create_result_row(param_combo, extracted_data, t_startup, None, sigmav_DD_p, sigmav_DD_n, sigmav_DT)
        
        if len(sol.t_events) > 1 and sol.t_events[1].size > 0:
            # Negative event occurred (second event)
            t_startup = np.inf
            return create_result_row(param_combo, extracted_data, t_startup, None, sigmav_DD_p, sigmav_DD_n, sigmav_DT)
        elif len(sol.t_events) > 0 and sol.t_events[0].size > 0:
            # DT condition reached (first event)
            t_startup = sol.t_events[0][0]
        else:
            # Time limit reached without DT condition
            t_startup = np.inf
        
        # For successful cases, we can evaluate the dense solution at specific points if needed
        return create_result_row(param_combo, extracted_data, t_startup, sol, sigmav_DD_p, sigmav_DD_n, sigmav_DT)
    
    except Exception as e:
        # Handle any integration errors silently during parallel execution
        return create_result_row(param_combo, extracted_data, np.inf, None, 
                                sigmav_DD_p, sigmav_DD_n, sigmav_DT)