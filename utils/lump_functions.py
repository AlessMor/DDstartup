import numpy as np
from utils.units_and_constants import *
from utils.physics import sigmav_DT_BoschHale, sigmav_DD_BoschHale, sigmav_DHe3_BoschHale
from utils.tools import index_to_params
from numba import njit
from utils.tools import make_input_dict, make_output_dict

@njit
def lump_numba(
    V_plasma, n_tot, tau_p_T, tau_p_He3, 
    P_aux, P_aux_DT_eq, 
    TBR_DT, TBR_DDn, I_target, 
    eta_th, capacity_factor, cost_of_electricity, 
    sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3):
    
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


def compute_single_combination(linear_index, input_arrays_flat, param_shapes_array):

    idx = index_to_params(linear_index, param_shapes_array)

    # Extract parameters in the same order as main.py expects for lump
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
    
    # Calculate T_i-dependent parameters (ensure scalar inputs to physics functions)
    T_i_array = np.array([T_i])  # Convert scalar to array for physics functions
    sigmav_DD_results = sigmav_DD_BoschHale(T_i_array)
    sigmav_DD_p = sigmav_DD_results[1][0]  # Extract scalar from array result
    sigmav_DD_n = sigmav_DD_results[2][0]  # Extract scalar from array result
    sigmav_DT = sigmav_DT_BoschHale(T_i_array)[0]  # Extract scalar from array result
    sigmav_DHe3 = sigmav_DHe3_BoschHale(T_i_array)[0]  # Extract scalar from array result
    
    # Call the core solver
    lump_results = lump_solver(
        V_plasma, n_tot, tau_p_T, tau_p_He3, P_aux, P_aux_DT_eq,
        TBR_DT, TBR_DDn, I_target, eta_th, capacity_factor, cost_of_electricity,
        sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3
    )
    
    # Package results (combine input parameters and ODE outputs)
    result = {
        'linear_index': linear_index,
        # Add all ODE results
        **lump_results
    }
    result_dict.update(result)

    return result_dict

