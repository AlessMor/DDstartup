import numpy as np
from ddstartup.utils.units_and_constants import *
from ddstartup.physics.reactionrates_functions import sigmav_DT_BoschHale, sigmav_DD_BoschHale, sigmav_DHe3_BoschHale
from numba import njit, prange
from ddstartup.utils.tools import index_to_params, make_input_dict, make_output_dict

@njit(cache=True)
def lump_numba(
    V_plasma, n_tot, tau_p_T, tau_p_He3, 
    P_aux, P_aux_DT_eq, 
    TBR_DT, TBR_DDn, I_target, 
    eta_th, capacity_factor, cost_of_electricity, 
    sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3):
    """
    JIT-compiled lumped parameter model for DD startup analysis.
    
    Uses steady-state assumptions to compute startup time and performance metrics
    without solving ODEs. Faster than T_seeded approach but less detailed.
    
    Args:
        V_plasma: Plasma volume (m³)
        n_tot: Total particle density (m⁻³)
        tau_p_T: Tritium confinement time (s)
        tau_p_He3: Helium-3 confinement time (s)
        P_aux: Auxiliary heating power (W)
        P_aux_DT_eq: Auxiliary power for D-T equivalent (W)
        TBR_DT: D-T tritium breeding ratio
        TBR_DDn: DD neutron tritium breeding ratio
        I_target: Target tritium inventory (kg)
        eta_th: Thermal efficiency
        capacity_factor: Plant capacity factor
        cost_of_electricity: Electricity cost ($/J)
        sigmav_DD_p, sigmav_DD_n: DD reaction rates (m³/s)
        sigmav_DT: D-T reaction rate (m³/s)
        sigmav_DHe3: D-He3 reaction rate (m³/s)
        
    Returns:
        Tuple of (n_T, n_D, n_He3, t_startup, Pf_DDn, Pf_DDp, Pf_DD_DT, 
                  Pf_DT_eq, Q_DD, Q_DT_eq, E_lost, unrealized_gains, sol_success)
    """
    
    n_D = n_tot
    # --- Startup tritium balance as in parametric_analysis_I_startup.py ---
    # n_T is the steady-state tritium density during startup
    n_T = (0.5 * n_D**2 * sigmav_DD_p) / (n_D * sigmav_DT + 1 / tau_p_T)
    n_He3 = (0.5 * n_D**2 * sigmav_DD_n) / (n_D * sigmav_DHe3 + 1 / tau_p_He3)

    # Total tritium production rate (from all sources)
    Tdot_DDn = TBR_DDn * 0.5 * n_D**2 * sigmav_DD_n * V_plasma
    Tdot_DDp = 0.5 * n_D**2 * sigmav_DD_p * V_plasma
    Tdot_DT = TBR_DT * n_D * n_T * sigmav_DT * V_plasma
    Tdot_tot = Tdot_DDn + Tdot_DDp + Tdot_DT
    # Required tritium inventory (atoms)
    N_ST = I_target / tritium_mass

    # Startup time (tritium inventory build-up) -- use same logic as parametric_analysis_I_startup.py
    ratio = N_ST * lambda_T / Tdot_tot if Tdot_tot > 0 else np.inf
    if ratio >= 1:
        t_startup = np.inf
        return (n_T, n_D, n_He3, t_startup, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False)
    else:
        t_startup = - (1/lambda_T) * np.log(1 - (ratio))

        # Power and energy calculations
        Pf_DDp = n_D**2 * sigmav_DD_p * V_plasma * E_DDp
        Pf_DDn = n_D**2 * sigmav_DD_n * V_plasma * E_DDn
        Pf_DD = Pf_DDp + Pf_DDn
        Pf_DD_DT = n_D * n_T * sigmav_DT * V_plasma * E_DT
        Pf_DD_DHe3 = n_D * n_He3 * sigmav_DHe3 * V_plasma * E_DHe3
        Pf_DD_tot = Pf_DD + Pf_DD_DT + Pf_DD_DHe3
        Pf_DT_eq = (n_tot/2)**2 * sigmav_DT * V_plasma * E_DT
        Q_DD = (Pf_DD_tot - P_aux) / P_aux if P_aux > 0 else np.inf
        P_e_net = capacity_factor * (eta_th * (Pf_DD_tot) - P_aux)
        Q_DT_eq = (Pf_DT_eq - P_aux_DT_eq) / P_aux_DT_eq if P_aux_DT_eq > 0 else np.inf
        P_e_net_DT_eq = capacity_factor * (eta_th * (Pf_DT_eq) - P_aux_DT_eq)

        # Energies
        E_e_net_DD = P_e_net * t_startup
        E_e_net_DT_eq = P_e_net_DT_eq * t_startup
        E_lost = E_e_net_DT_eq - E_e_net_DD
        unrealized_gains = E_lost * cost_of_electricity
        
        
        return (n_T, n_D, n_He3, t_startup, Pf_DDn, Pf_DDp, Pf_DD_DT, Pf_DT_eq, Q_DD, Q_DT_eq, E_lost, unrealized_gains, True)
        
    

def lump_solver(
    V_plasma, n_tot, tau_p_T, tau_p_He3, 
    P_aux, P_aux_DT_eq, 
    TBR_DT, TBR_DDn, I_target, 
    eta_th, capacity_factor, cost_of_electricity, 
    sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3):
    """
    Wrapper for lump_numba that returns results as a dictionary.
    
    Calls the JIT-compiled lump model and packages results in standard format.
    
    Args:
        (See lump_numba for parameter descriptions)
        
    Returns:
        Dictionary with keys: n_T, n_D, n_He3, t_startup, P_DDn, P_DDp, P_DT,
                             P_DT_eq, Q_DD, Q_DT_eq, E_lost, unrealized_gains, sol_success
    """

    (n_T, n_D, n_He3, 
     t_startup, 
     Pf_DDn, Pf_DDp, Pf_DD_DT, Pf_DT_eq, Q_DD, Q_DT_eq, 
     E_lost, unrealized_gains, sol_success) = lump_numba(
    V_plasma, n_tot, tau_p_T, tau_p_He3, 
    P_aux, P_aux_DT_eq, 
    TBR_DT, TBR_DDn, I_target, 
    eta_th, capacity_factor, cost_of_electricity, 
    sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3)

    return make_output_dict({
        'n_T': n_T,
        'n_D': n_D,
        'n_He3': n_He3,
        't_startup': t_startup,
        'P_DDn': Pf_DDn,
        'P_DDp': Pf_DDp,
        'P_DT': Pf_DD_DT,
        'P_DT_eq': Pf_DT_eq,
        'Q_DD': Q_DD,
        'Q_DT_eq': Q_DT_eq,
        'E_lost': E_lost,
        'unrealized_gains': unrealized_gains,
        'sol_success': sol_success
    })



# # --- Numba-parallelized batch computation ---
# def compute_lump_batch(
#     V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, P_aux, P_aux_DT_eq,
#     TBR_DT, TBR_DDn, I_target, eta_th, capacity_factor, cost_of_electricity
# ):
#     """
#     Compute lump model for multiple parameter combinations (vectorized).
    
#     Alternative to compute_single_combination for batch processing.
#     All inputs are arrays of the same length.
    
#     Args:
#         V_plasma: Array of plasma volumes (m³)
#         T_i: Array of ion temperatures (keV)
#         n_tot: Array of total densities (m⁻³)
#         tau_p_T: Array of tritium confinement times (s)
#         tau_p_He3: Array of He3 confinement times (s)
#         P_aux: Array of auxiliary powers (W)
#         P_aux_DT_eq: Array of D-T equivalent powers (W)
#         TBR_DT: Array of D-T breeding ratios
#         TBR_DDn: Array of DD neutron breeding ratios
#         I_target: Array of target inventories (kg)
#         eta_th: Array of thermal efficiencies
#         capacity_factor: Array of capacity factors
#         cost_of_electricity: Array of electricity costs ($/J)
        
#     Returns:
#         Dictionary of result arrays (n_T, n_D, n_He3, t_startup, etc.)
#     """
#     n = len(V_plasma)
#     # Preallocate output arrays
#     results = {
#         'n_T': np.empty(n),
#         'n_D': np.empty(n),
#         'n_He3': np.empty(n),
#         't_startup': np.empty(n),
#         'P_DDn': np.empty(n),
#         'P_DDp': np.empty(n),
#         'P_DT': np.empty(n),
#         'P_DT_eq': np.empty(n),
#         'Q_DD': np.empty(n),
#         'Q_DT_eq': np.empty(n),
#         'E_lost': np.empty(n),
#         'unrealized_gains': np.empty(n),
#         'sol_success': np.empty(n, dtype=bool),
#     }
#     # Physics functions (vectorized)
#     sigmav_DD_results = sigmav_DD_BoschHale(T_i)
#     sigmav_DD_p = sigmav_DD_results[1]
#     sigmav_DD_n = sigmav_DD_results[2]
#     sigmav_DT = sigmav_DT_BoschHale(T_i)
#     sigmav_DHe3 = sigmav_DHe3_BoschHale(T_i)
#     # Numba loop
#     for i in range(n):
#         tup = lump_numba(
#             V_plasma[i], n_tot[i], tau_p_T[i], tau_p_He3[i],
#             P_aux[i], P_aux_DT_eq[i], TBR_DT[i], TBR_DDn[i], I_target[i],
#             eta_th[i], capacity_factor[i], cost_of_electricity[i],
#             sigmav_DD_p[i], sigmav_DD_n[i], sigmav_DT[i], sigmav_DHe3[i]
#         )
#         # Unpack results
#         results['n_T'][i] = tup[0]
#         results['n_D'][i] = tup[1]
#         results['n_He3'][i] = tup[2]
#         results['t_startup'][i] = tup[3]
#         results['P_DDn'][i] = tup[4]
#         results['P_DDp'][i] = tup[5]
#         results['P_DT'][i] = tup[6]
#         results['P_DT_eq'][i] = tup[7]
#         results['Q_DD'][i] = tup[8]
#         results['Q_DT_eq'][i] = tup[9]
#         results['E_lost'][i] = tup[10]
#         results['unrealized_gains'][i] = tup[11]
#         results['sol_success'][i] = tup[12]
#     return results

# --- Old single-combination function preserved ---
def compute_single_combination(linear_index, input_arrays_flat, param_shapes_array):
    """
    Compute lump analysis for a single parameter combination.
    
    This is the core function called by parallel workers during parametric analysis.
    Converts linear index to parameter indices, extracts values, computes reaction
    rates, runs lump model, and packages results.
    
    Args:
        linear_index: Integer index (0 to n_combinations-1) identifying parameter set
        input_arrays_flat: List of 1D arrays, one per parameter
        param_shapes_array: Array of parameter grid shapes for index conversion
        
    Returns:
        Dictionary with input parameters and computed results:
            - Input echoes: V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, etc.
            - Results: n_T, n_D, n_He3, t_startup, P_DDn, P_DDp, P_DT, etc.
            - Status: linear_index, sol_success
    """
    idx = index_to_params(linear_index, param_shapes_array)
    V_plasma = input_arrays_flat[0][idx[0]]
    T_i = input_arrays_flat[1][idx[1]]
    n_tot = input_arrays_flat[2][idx[2]]
    tau_p_T = input_arrays_flat[3][idx[3]]
    tau_p_He3 = input_arrays_flat[4][idx[4]]
    P_aux = input_arrays_flat[5][idx[5]]
    P_aux_DT_eq = input_arrays_flat[6][idx[6]]
    TBR_DT = input_arrays_flat[7][idx[7]]
    TBR_DDn = input_arrays_flat[8][idx[8]]
    I_target = input_arrays_flat[9][idx[9]]
    eta_th = input_arrays_flat[10][idx[10]]
    capacity_factor = input_arrays_flat[11][idx[11]]
    cost_of_electricity = input_arrays_flat[12][idx[12]]
    result_dict = make_input_dict({
        'V_plasma': V_plasma,
        'T_i': T_i,
        'n_tot': n_tot,
        'tau_p_T': tau_p_T,
        'tau_p_He3':tau_p_He3,
        'P_aux': P_aux,
        'P_aux_DT_eq': P_aux_DT_eq,
        'TBR_DT': TBR_DT,
        'TBR_DDn': TBR_DDn,
        'I_target': I_target,
        'eta_th': eta_th,
        'capacity_factor': capacity_factor,
        'cost_of_electricity': cost_of_electricity,
    })
    T_i_array = np.array([T_i])
    sigmav_DD_results = sigmav_DD_BoschHale(T_i_array)
    sigmav_DD_p = sigmav_DD_results[1][0]
    sigmav_DD_n = sigmav_DD_results[2][0]
    sigmav_DT = sigmav_DT_BoschHale(T_i_array)[0]
    sigmav_DHe3 = sigmav_DHe3_BoschHale(T_i_array)[0]
    lump_results = lump_solver(
        V_plasma, n_tot, tau_p_T, tau_p_He3, P_aux, P_aux_DT_eq,
        TBR_DT, TBR_DDn, I_target, eta_th, capacity_factor, cost_of_electricity,
        sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3
    )
    result = {
        'linear_index': linear_index,
        **lump_results
    }
    result_dict.update(result)
    return result_dict

