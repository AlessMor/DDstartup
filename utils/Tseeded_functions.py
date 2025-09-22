import numpy as np
from scipy.integrate import solve_ivp
from .units_and_constants import *
from utils.physics import sigmav_DT_BoschHale, sigmav_DD_BoschHale
from utils.tools import index_to_params
from numba import njit
    


@njit
def compute_reaction_rates(n_tot, n_T, sigmav_DT, sigmav_DD_p, sigmav_DD_n, V_plasma, TBR_DT, TBR_DDn):
    n_D = n_tot - n_T
    Tdot_DDn = TBR_DDn * 0.5 * n_D * n_D * sigmav_DD_n * V_plasma
    Tdot_DDp = 0.5 * n_D * n_D * sigmav_DD_p * V_plasma
    Tdot_DT = TBR_DT * n_D * n_T * sigmav_DT * V_plasma
    Tdot_burn = n_D * n_T * sigmav_DT * V_plasma
    return Tdot_DDn, Tdot_DDp, Tdot_DT, Tdot_burn

@njit
def ode_system(t, y, V_plasma, T_i, n_tot, tau_p_T, P_aux, P_lost_rad,
               P_aux_all_DT, P_lost_rad_all_DT, TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
               eta_th, plant_avail, Cost_per_kWh, sigmav_DD_p, sigmav_DD_n, sigmav_DT, injection_rate_max, N_st_min):
    # Unpack state variables
    N_ofc = y[0]
    N_ifc = y[1]
    N_st = y[2]
    n_T = y[3]

    # Compute injection rate (Numba-compatible, no min/max)
    if N_st > N_st_min:
        inj_rate = N_ifc / tau_ifc - lambda_T * N_st
        if inj_rate > injection_rate_max:
            injection_rate = injection_rate_max
        elif inj_rate < 0.0:
            injection_rate = 0.0
        else:
            injection_rate = inj_rate
    else:
        injection_rate = 0.0

    # Compute reaction rates
    n_D = n_tot - n_T
    Tdot_DDn = TBR_DDn * 0.5 * n_D * n_D * sigmav_DD_n * V_plasma
    Tdot_DDp = 0.5 * n_D * n_D * sigmav_DD_p * V_plasma
    Tdot_DT = TBR_DT * n_D * n_T * sigmav_DT * V_plasma
    Tdot_burn = n_D * n_T * sigmav_DT * V_plasma

    # ODEs
    dN_ofc_dt = Tdot_DT + Tdot_DDn - N_ofc / tau_ofc - N_ofc * lambda_T
    dN_ifc_dt = N_ofc / tau_ofc - N_ifc / tau_ifc - lambda_T * N_ifc + n_T / tau_p_T * V_plasma
    dN_stor_dt = N_ifc / tau_ifc - lambda_T * N_st - injection_rate
    dnT_dt = injection_rate / V_plasma + Tdot_DDp / V_plasma - n_T / tau_p_T - Tdot_burn / V_plasma

    return np.array([dN_ofc_dt, dN_ifc_dt, dN_stor_dt, dnT_dt])

def solve_ode_system(total_time, V_plasma, T_i, n_tot, tau_p_T, P_aux, P_lost_rad, 
                           P_aux_all_DT, P_lost_rad_all_DT, TBR_DT, TBR_DDn, tau_ifc, tau_ofc, 
                           eta_th, plant_avail, Cost_per_kWh, sigmav_DD_p, sigmav_DD_n, sigmav_DT, injection_rate_max, N_st_min=0.001/tritium_mass):
    # Units-free ODE function
    
    def tritium_inventory_odes_unitless(t, y):
        # Call the njit-compiled ODE system for performance and correctness
        return ode_system(
            t, y, V_plasma, T_i, n_tot, tau_p_T, P_aux, P_lost_rad,
            P_aux_all_DT, P_lost_rad_all_DT, TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
            eta_th, plant_avail, Cost_per_kWh, sigmav_DD_p, sigmav_DD_n, sigmav_DT, injection_rate_max, N_st_min
        )
        
    
    # Event functions
    def DT_reached_event(t, y):
        n_T = y[3]
        return n_T - 0.5 * n_tot
    
    def negative_event(t, y):
        return min(y[0]+100, y[1]+100, y[2]+100, y[3]+100)
    
    # Configure events
    DT_reached_event.terminal = True
    negative_event.terminal = True 
    
    # Initial conditions [N_ofc, N_ifc, N_stor, n_T]
    y0 = [0.0, 0.0, 0.0, 0.0]  # Start with minimal tritium storage
    
    # Time span
    t_span = (0, total_time)
    t_eval = np.linspace(0, total_time, 100)
    
    # Solve ODE system
    try:
        sol = solve_ivp(
            fun=tritium_inventory_odes_unitless,
            t_span=t_span,
            t_eval=t_eval,
            y0=y0,
            method='BDF',  # Good for stiff systems
            dense_output=False,
            events=[DT_reached_event, negative_event],
            rtol=1e-6,
            atol=1e-9
        )
    except Exception as e:
        # Return failure case with detailed error information
        import traceback
        full_error = f"{str(e)} | Traceback: {traceback.format_exc()}"
        return {
            't_startup': np.inf,
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
            'error': full_error
        }
    
    # Process results (rest of the function same as before)
    if not sol.success:
        # Enhanced error reporting for ODE solve failures
        failure_reason = f"ODE solve failed: {sol.message if hasattr(sol, 'message') else 'Unknown reason'}"
        if hasattr(sol, 't') and len(sol.t) > 0:
            failure_reason += f" | Last time: {sol.t[-1]:.2e}s"
        if hasattr(sol, 'status'):
            failure_reason += f" | Status code: {sol.status}"
        
        return {
            't_startup': np.inf,
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
            'error': failure_reason
        }
    
    # Check for negative event (index 1)
    if len(sol.t_events) > 1 and sol.t_events[1].size > 0:
        return {
            't_startup': np.inf,
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
            'negative_event_time': sol.t_events[1][0]
        }
    
    # Determine startup time
    if len(sol.t_events) > 0 and sol.t_events[0].size > 0:
        t_startup = sol.t_events[0][0]  # DT reached event
    else:
        t_startup = np.inf
    
    # Extract solution
    n_T = sol.y[3]  # Tritium density evolution
    n_D = n_tot - n_T  # Deuterium density
    # Pre-compute common terms to avoid redundant calculations
    n_D_squared = n_D * n_D  # Compute once, use multiple times
    n_D_n_T = n_D * n_T      # Compute once for DT reactions
    V_E_DDn = V_plasma * E_DDn  # Pre-compute scalar products
    V_E_DDp = V_plasma * E_DDp
    V_E_DT = V_plasma * E_DT
    
    # Calculate fusion powers (all in Watts)
    P_DDn = n_D_squared * sigmav_DD_n / 2 * V_E_DDn
    P_DDp = n_D_squared * sigmav_DD_p / 2 * V_E_DDp
    P_DT = n_D_n_T * sigmav_DT * V_E_DT
    P_DT_full = n_tot/2 * n_tot/2 * sigmav_DT * V_E_DT  # Equivalent if always DT
    
    # Determine integration time
    t_end = t_startup if np.isfinite(t_startup) else np.inf
    
    if np.isfinite(t_end):
        # Total fusion energies by integration
        E_fusion_DDn = np.trapezoid(P_DDn, sol.t)
        E_fusion_DDp = np.trapezoid(P_DDp, sol.t)
        E_fusion_DT = np.trapezoid(P_DT, sol.t)
        E_fusion_total_DD = E_fusion_DDn + E_fusion_DDp + E_fusion_DT
        E_fusion_DT_full = P_DT_full * t_end  # Constant power * time
        
        # Pre-compute auxiliary and radiation energies (scalar operations)
        E_aux_DD = P_aux * t_end
        E_aux_DT_full = P_aux_all_DT * t_end
        E_rad_DD = P_lost_rad * t_end
        E_rad_DT_full = P_lost_rad_all_DT * t_end

        # Net electrical energies (with thermal efficiency and plant availability)
        E_e_net_DD = plant_avail * (eta_th * (E_fusion_total_DD - E_rad_DD) - E_aux_DD)
        E_e_net_DT_full = plant_avail * (eta_th * (E_fusion_DT_full - E_rad_DT_full) - E_aux_DT_full)
        
        # Q factors based on total energy
        Q_DD_total = E_fusion_total_DD / E_aux_DD if E_aux_DD > 0 else np.inf
        Q_DT_full_total = E_fusion_DT_full / E_aux_DT_full if E_aux_DT_full > 0 else np.inf
        
        # Average powers for reference (divide total energy by time)
        P_fusion_DD_avg = E_fusion_total_DD / t_end
        P_e_net_DD_avg = E_e_net_DD / t_end
        P_e_net_DT_full_avg = E_e_net_DT_full / t_end 

        E_lost = E_e_net_DT_full - E_e_net_DD
        Dollar_lost = E_lost * Cost_per_kWh  # Cost in dollars (NB Cost is in 1/J)
    else:
        E_fusion_total_DD = np.inf
        E_fusion_DT_full = np.inf
        E_e_net_DD = np.inf
        E_e_net_DT_full = np.inf
        Q_DD_total = np.inf
        Q_DT_full_total = np.inf
        Dollar_lost = np.inf
        E_lost = np.inf
        P_fusion_DD_avg = np.inf
        P_e_net_DD_avg = np.inf
        P_e_net_DT_full_avg = np.inf
        Dollar_lost = np.inf
        
    return {
        't_startup': t_startup,
        'P_DT': P_DT,
        'P_DDn': P_DDn,
        'P_DDp': P_DDp,
        'P_DT_full': P_DT_full,  # Constant scalar value
        'P_fusion_DD_avg': P_fusion_DD_avg,  # Average fusion power during startup
        'P_e_net_DD_avg': P_e_net_DD_avg,    # Average net electrical power
        'P_e_net_DT_full_avg': P_e_net_DT_full_avg,  # Average net electrical power (DT case)
        'Q_DD_total': Q_DD_total,            # Q factor based on total energy
        'Q_DT_full_total': Q_DT_full_total,  # Q factor based on total energy
        'E_fusion_total_DD': E_fusion_total_DD,      # Total fusion energy produced
        'E_fusion_DT_full': E_fusion_DT_full,        # Total DT-equivalent fusion energy
        'E_e_net_DD': E_e_net_DD,                    # Total net electrical energy (DD startup)
        'E_e_net_DT_full': E_e_net_DT_full,          # Total net electrical energy (DT full)
        'E_lost': E_lost,      # Energy difference
        'Dollar_Lost': Dollar_lost,  # Cost in dollars
        'n_T_final': n_T[-1] if len(n_T) > 0 else 0.0,
        'sol_success': sol.success
    }
       
def compute_single_combination(linear_index, input_arrays_flat, param_shapes_array, total_time = 10*365*24*3600):

    # Get parameter indices from linear index
    param_indices = index_to_params(linear_index, param_shapes_array)
    
    # Extract parameters directly - fastest approach
    V_plasma = input_arrays_flat[0][param_indices[0]]
    T_i = input_arrays_flat[1][param_indices[1]]
    n_tot = input_arrays_flat[2][param_indices[2]]
    tau_p_T = input_arrays_flat[3][param_indices[3]]
    P_aux = input_arrays_flat[4][param_indices[4]]
    P_lost_rad = input_arrays_flat[5][param_indices[5]]
    P_aux_all_DT = input_arrays_flat[6][param_indices[6]]
    P_lost_rad_all_DT = input_arrays_flat[7][param_indices[7]]
    TBR_DT = input_arrays_flat[8][param_indices[8]]
    TBR_DDn = input_arrays_flat[9][param_indices[9]]
    tau_ifc = input_arrays_flat[10][param_indices[10]]
    tau_ofc = input_arrays_flat[11][param_indices[11]]
    eta_th = input_arrays_flat[12][param_indices[12]]
    plant_avail = input_arrays_flat[13][param_indices[13]]
    Cost_per_kWh = input_arrays_flat[14][param_indices[14]]
    
    # Calculate T_i-dependent parameters (ensure scalar inputs to physics functions)
    T_i_array = np.array([T_i])  # Convert scalar to array for physics functions
    sigmav_DD_results = sigmav_DD_BoschHale(T_i_array)
    sigmav_DD_p = sigmav_DD_results[1][0]  # Extract scalar from array result
    sigmav_DD_n = sigmav_DD_results[2][0]  # Extract scalar from array result
    sigmav_DT = sigmav_DT_BoschHale(T_i_array)[0]  # Extract scalar from array result
    
    # Precompute injection_rate_max
    injection_rate_max = (n_tot/2/tau_p_T*V_plasma + 0.25*n_tot**2*sigmav_DT*V_plasma - 0.25/2*n_tot**2*sigmav_DD_p*V_plasma)

    # SOLVE ODE SYSTEM
    ode_results = solve_ode_system(total_time,
        V_plasma, T_i, n_tot, tau_p_T, P_aux, P_lost_rad,
        P_aux_all_DT, P_lost_rad_all_DT, TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
        eta_th, plant_avail, Cost_per_kWh, sigmav_DD_p, sigmav_DD_n, sigmav_DT, injection_rate_max
    )
    
    # Package results (combine input parameters and ODE outputs)
    result = {
        'linear_index': linear_index,
        'V_plasma': V_plasma,
        'T_i': T_i,
        'n_tot': n_tot,
        'tau_p_T': tau_p_T,
        'P_aux': P_aux,
        'P_lost_rad': P_lost_rad,
        'P_aux_all_DT': P_aux_all_DT,
        'P_lost_rad_all_DT': P_lost_rad_all_DT,
        'TBR_DT': TBR_DT,
        'TBR_DDn': TBR_DDn,
        'tau_ifc': tau_ifc,
        'tau_ofc': tau_ofc,
        'eta_th': eta_th,
        'plant_avail': plant_avail,
        'Cost_per_kWh': Cost_per_kWh,
        'injection_rate_max': injection_rate_max,
        'sigmav_DT': sigmav_DT,
        'sigmav_DD_p': sigmav_DD_p,
        'sigmav_DD_n': sigmav_DD_n,
        # Add all ODE results
        **ode_results
    }
    
    return result

