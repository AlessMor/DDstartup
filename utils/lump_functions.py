import numpy as np
from utils.units_and_constants import *
from utils.physics import sigmav_DT_BoschHale, sigmav_DD_BoschHale, sigmav_DHe3_BoschHale
from utils.tools import index_to_params



def compute_single_combination(linear_index, input_arrays_flat, param_shapes_array):

    idx = index_to_params(linear_index, param_shapes_array)

    # Extract parameters in the same order as main.py expects for lumped
    V_plasma = input_arrays_flat[0][idx[0]]
    T_i = input_arrays_flat[1][idx[1]]
    n_tot = input_arrays_flat[2][idx[2]]
    tau_p_T = input_arrays_flat[3][idx[3]]
    tau_p_He3 = input_arrays_flat[4][idx[4]]
    P_aux = input_arrays_flat[5][idx[5]]
    P_lost_rad = input_arrays_flat[6][idx[6]]
    P_aux_all_DT = input_arrays_flat[7][idx[7]]
    P_lost_rad_all_DT = input_arrays_flat[8][idx[8]]
    TBR_DT = input_arrays_flat[9][idx[9]]
    TBR_DDn = input_arrays_flat[10][idx[10]]
    I_target = input_arrays_flat[11][idx[11]]
    eta_th = input_arrays_flat[12][idx[12]]
    plant_avail = input_arrays_flat[13][idx[13]]
    Cost_per_kWh = input_arrays_flat[14][idx[14]]
    # Calculate T_i-dependent parameters (ensure scalar inputs to physics functions)
    T_i_array = np.array([T_i])  # Convert scalar to array for physics functions
    sigmav_DD_results = sigmav_DD_BoschHale(T_i_array)
    sigmav_DD_p = sigmav_DD_results[1][0]  # Extract scalar from array result
    sigmav_DD_n = sigmav_DD_results[2][0]  # Extract scalar from array result
    sigmav_DT = sigmav_DT_BoschHale(T_i_array)[0]  # Extract scalar from array result
    sigmav_DHe3 = sigmav_DHe3_BoschHale(T_i_array)[0]  # Extract scalar from array result
    
    # Call the core solver
    core_results = lumped_core_solver(
        V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, P_aux, P_lost_rad, P_aux_all_DT, P_lost_rad_all_DT,
        TBR_DT, TBR_DDn, I_target, eta_th, plant_avail, Cost_per_kWh,  sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3
    )
    
    result = {
        'linear_index': linear_index,
        'V_plasma': V_plasma,
        'T_i': T_i,
        'n_tot': n_tot,
        'tau_p_T': tau_p_T,
        'tau_p_He3': tau_p_He3,
        'P_aux': P_aux,
        'P_lost_rad': P_lost_rad,
        'P_aux_all_DT': P_aux_all_DT,
        'P_lost_rad_all_DT': P_lost_rad_all_DT,
        'TBR_DT': TBR_DT,
        'TBR_DDn': TBR_DDn,
        'I_target': I_target,
        'eta_th': eta_th,
        'plant_avail': plant_avail,
        'Cost_per_kWh': Cost_per_kWh,
        'sigmav_DT': sigmav_DT,
        'sigmav_DD_p': sigmav_DD_p,
        'sigmav_DD_n': sigmav_DD_n,
        'sigmav_DHe3': sigmav_DHe3,
        # Add all outputs from the core solver
        **core_results
    }
    return result


def lumped_core_solver(V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, P_aux, P_lost_rad, P_aux_all_DT, P_lost_rad_all_DT, TBR_DT, TBR_DDn, I_target, eta_th, plant_avail, Cost_per_kWh, sigmav_DD_p, sigmav_DD_n, sigmav_DT, sigmav_DHe3):

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
        result = {
            't_startup': t_startup,
            'P_DT': np.inf,
            'P_DDn': np.inf,
            'P_DDp': np.inf,
            'P_DT_full': np.inf,
            'P_fusion_DD_avg': np.inf,
            'P_e_net_DD_avg': np.inf,
            'P_e_net_DT_full_avg': np.inf,
            'Q_DD_total': np.inf,
            'Q_DT_full_total': np.inf,
            'E_fusion_total_DD': np.inf,
            'E_fusion_DT_full': np.inf,
            'E_e_net_DD': np.inf,
            'E_e_net_DT_full': np.inf,
            'E_lost': np.inf,
            'Dollar_Lost': np.inf,
            'n_T_final': np.inf,
            'sol_success': False,
        }
    else:
        t_startup = - (1/lambda_T) * np.log(1 - (ratio))

        # Power and energy calculations
        Pf_DDp = n_D**2 * sigmav_DD_p * V_plasma * E_DDp
        Pf_DDn = n_D**2 * sigmav_DD_n * V_plasma * E_DDn
        Pf_DD = Pf_DDp + Pf_DDn
        Pf_DD_DT = n_D * n_T * sigmav_DT * V_plasma * E_DT
        Pf_DD_DHe3 = n_D * n_He3 * sigmav_DHe3 * V_plasma * E_DHe3
        Pf_DD_tot = Pf_DD + Pf_DD_DT + Pf_DD_DHe3
        Pf_DT_full = (n_tot/2)**2 * sigmav_DT * V_plasma * E_DT
        Q_DD = (Pf_DD_tot - P_aux) / P_aux if P_aux > 0 else np.inf
        P_e_net = plant_avail * (eta_th * (Pf_DD_tot - P_lost_rad) - P_aux)
        Q_DT_full = (Pf_DT_full - P_aux_all_DT) / P_aux_all_DT if P_aux_all_DT > 0 else np.inf
        P_e_net_DT_full = plant_avail * (eta_th * (Pf_DT_full - P_lost_rad_all_DT) - P_aux_all_DT)

        # Energies
        E_fusion_total_DD = Pf_DD_tot * t_startup
        E_fusion_DT_full = Pf_DT_full * t_startup
        E_e_net_DD = E_fusion_total_DD * eta_th * plant_avail
        E_e_net_DT_full = E_fusion_DT_full * eta_th * plant_avail
        E_lost = E_e_net_DT_full - E_e_net_DD
        Dollar_Lost = E_lost * Cost_per_kWh

        result = {
            't_startup': t_startup,
            'P_DT': Pf_DD_DT,
            'P_DDn': Pf_DDn,
            'P_DDp': Pf_DDp,
            'P_DT_full': Pf_DT_full,
            'P_fusion_DD_avg': Pf_DD,
            'P_e_net_DD_avg': P_e_net,
            'P_e_net_DT_full_avg': P_e_net_DT_full,
            'Q_DD_total': Q_DD,
            'Q_DT_full_total': Q_DT_full,
            'E_fusion_total_DD': E_fusion_total_DD,
            'E_fusion_DT_full': E_fusion_DT_full,
            'E_e_net_DD': E_e_net_DD,
            'E_e_net_DT_full': E_e_net_DT_full,
            'E_lost': E_lost,
            'Dollar_Lost': Dollar_Lost,
            'n_T_final': n_T if np.isfinite(n_T) else 0.0,
            'sol_success': np.isfinite(t_startup) and t_startup > 0 and t_startup < 1e20
        }
    return result