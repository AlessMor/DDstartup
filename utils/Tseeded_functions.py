import numpy as np
from scipy.integrate import solve_ivp
from .units_and_constants import *
from utils.physics import sigmav_DT_BoschHale, sigmav_DD_BoschHale
from numba import njit
from utils.tools import index_to_params, make_input_dict, make_output_dict, fix_vector_length

@njit
def ode_system(t, y, 
               V_plasma, n_tot, tau_p_T, 
               TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
               sigmav_DD_p, sigmav_DD_n, sigmav_DT, 
               injection_rate_max, N_st_min):
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

def solve_ode_system(total_time, 
                     V_plasma, n_tot, tau_p_T, 
                     P_aux, P_aux_DT_eq,
                     TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
                     eta_th, capacity_factor, cost_of_electricity,
                     sigmav_DD_p, sigmav_DD_n, sigmav_DT, 
                     injection_rate_max, N_st_min=0.001/tritium_mass, STORE_TBE=False):
    # Units-free ODE function
    
    def tritium_inventory_odes_unitless(t, y):
        # Call the njit-compiled ODE system for performance and correctness
        return ode_system(
            t, y, 
            V_plasma, n_tot, tau_p_T, 
            TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT, 
            injection_rate_max, N_st_min
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
    
    
    # --- Solve ODE system ---
    try:
        sol = solve_ivp(
            fun=tritium_inventory_odes_unitless,
            t_span=t_span,
            y0=y0,
            method='BDF',  # Good for stiff systems
            dense_output=True,
            events=[DT_reached_event, negative_event],
            rtol=1e-6,
            atol=1e-9
        )
        N_ofc = sol.y[0]
        N_ifc = sol.y[1]
        N_st  = sol.y[2]
        n_T   = sol.y[3]
        
        # --- the solver failed ---
        if not sol.success:
            # Enhanced error reporting for ODE solve failures
            failure_reason = f"ODE solve failed: {sol.message if hasattr(sol, 'message') else 'Unknown reason'}"
            if hasattr(sol, 't') and len(sol.t) > 0:
                failure_reason += f" | Last time: {sol.t[-1]:.2e}s"
            if hasattr(sol, 'status'):
                failure_reason += f" | Status code: {sol.status}"
            
            return make_output_dict({
                'N_ofc': N_ofc,
                'N_ifc': N_ifc,
                'N_stor': N_st,
                'n_T': n_T,
                'error': failure_reason,
                'sol_success': False
            })
        
        # --- there were negative events ---
        if len(sol.t_events) > 1 and sol.t_events[1].size > 0:
            negative_event_msg = f"Negative population event at t={sol.t_events[1][0]:.2e}s"
            return make_output_dict({
                'N_ofc': N_ofc,
                'N_ifc': N_ifc,
                'N_stor': N_st,
                'n_T': n_T,
                'error': negative_event_msg,
                'sol_success': False
            })

        # --- DT opertion reached ---
        elif len(sol.t_events) > 0 and sol.t_events[0].size > 0:
            t_startup = sol.t_events[0][0]  # DT reached event
            
            # check again if t_startup is finite and a float
            if not np.isfinite(t_startup) or not isinstance(t_startup, (float, np.floating)):
                error_msg = "Invalid t_startup value"
                return make_output_dict({
                    'N_ofc': N_ofc,
                    'N_ifc': N_ifc,
                    'N_stor': N_st,
                    'n_T': n_T,
                    'error': error_msg,
                    'sol_success': False
                })
                
            # Extract solution
            N_ofc = sol.y[0]  # Out-of-fuel-cycle inventory evolution
            N_ifc = sol.y[1]  # In-fuel-cycle inventory evolution
            N_st = sol.y[2]   # Stored tritium inventory evolution
            n_T = sol.y[3]  # Tritium density evolution
            n_D = n_tot*np.ones_like(n_T) - n_T  # Deuterium density

            # Pre-compute common terms to avoid redundant calculations
            n_D_squared = n_D**2  # Compute once, use multiple times
            n_D_n_T = n_D * n_T      # Compute once for DT reactions
            V_E_DDn = V_plasma * E_DDn  # Pre-compute scalar products
            V_E_DDp = V_plasma * E_DDp
            V_E_DT = V_plasma * E_DT
            
            # Calculate fusion powers (all in Watts)
            P_DDn = n_D_squared * sigmav_DD_n / 2 * V_E_DDn
            P_DDp = n_D_squared * sigmav_DD_p / 2 * V_E_DDp
            P_DT = n_D_n_T * sigmav_DT * V_E_DT
            P_DT_eq = n_tot/2 * n_tot/2 * sigmav_DT * V_E_DT  # Equivalent if always DT
        
            # Total fusion energies by integration
            E_fusion_DDn = np.trapezoid(P_DDn, sol.t)
            E_fusion_DDp = np.trapezoid(P_DDp, sol.t)
            E_fusion_DT = np.trapezoid(P_DT, sol.t)
            E_fusion_total_DD = E_fusion_DDn + E_fusion_DDp + E_fusion_DT
            E_fusion_DT_eq = P_DT_eq * t_startup  # Constant power * time
            
            # Pre-compute auxiliary power energy (scalar operations)
            E_aux_DD = P_aux * t_startup
            E_aux_DT_eq = P_aux_DT_eq * t_startup

            # Net electrical energies (with thermal efficiency and plant availability)
            E_e_net_DD = capacity_factor * (eta_th * (E_fusion_total_DD) - E_aux_DD)
            E_e_net_DT_eq = capacity_factor * (eta_th * (E_fusion_DT_eq) - E_aux_DT_eq)
            
            # Q factors based on total energy
            Q_DD = E_fusion_total_DD / E_aux_DD if E_aux_DD > 0 else np.inf
            Q_DT_eq = E_fusion_DT_eq / E_aux_DT_eq if E_aux_DT_eq > 0 else np.inf
            
            # # Average powers for reference (divide total energy by time)
            # P_fusion_DD_avg = E_fusion_total_DD / t_startup
            # P_e_net_DD_avg = E_e_net_DD / t_startup
            # P_e_net_DT_eq_avg = E_e_net_DT_eq / t_startup 

            E_lost = E_e_net_DT_eq - E_e_net_DD
            unrealized_gains = E_lost * cost_of_electricity  # Cost in dollars (NB Cost is in 1/J)

            # Compute TBE_vector if requested
            mask = N_st > N_st_min
            inj_rate = N_ifc[mask] / tau_ifc - lambda_T * N_st[mask]
            inj_rate = np.clip(inj_rate, 0.0, injection_rate_max)
            TBE_vector = np.full_like(N_st, np.nan)
            TBE_vector[mask] = (n_D[mask] * n_T[mask] * sigmav_DT) / inj_rate
            
            # #save all to a txt for debugging
            # with open("debug_output.txt", "w") as f:
            #     f.write("t_startup: {}\n".format(t_startup))
            #     f.write("N_ofc: {}\n".format(N_ofc))
            #     f.write("N_ifc: {}\n".format(N_ifc))
            #     f.write("N_st: {}\n".format(N_st))
            #     f.write("n_T: {}\n".format(n_T))
            #     f.write("n_D: {}\n".format(n_D))
            #     f.write("P_DDn: {}\n".format(P_DDn))
            #     f.write("P_DDp: {}\n".format(P_DDp))
            #     f.write("P_DT: {}\n".format(P_DT))
            #     f.write("P_DT_eq: {}\n".format(P_DT_eq))
            #     f.write("Q_DD: {}\n".format(Q_DD))
            #     f.write("Q_DT_eq: {}\n".format(Q_DT_eq))
            #     f.write("E_lost: {}\n".format(E_lost))
            #     f.write("unrealized_gains: {}\n".format(unrealized_gains))
            #     f.write("TBE_vector: {}\n".format(TBE_vector))
            
            
            return make_output_dict({
                'N_ofc': N_ofc,
                'N_ifc': N_ifc,
                'N_stor': N_st,
                'n_D': n_D,
                'n_T': n_T,
                't_startup': t_startup,
                'P_DDn': P_DDn,
                'P_DDp': P_DDp,
                'P_DT': P_DT,
                'P_DT_eq': P_DT_eq,
                'Q_DD': Q_DD,
                'Q_DT_eq': Q_DT_eq,
                'E_lost': E_lost,
                'unrealized_gains': unrealized_gains,
                'TBE': TBE_vector if STORE_TBE else np.nan,
                'sol_success': True
            })
            
            
        # --- cannot reach DT operation ---
        else:
            t_startup = np.inf
            error_msg = "DT concentration not reached within total_time"
            return make_output_dict({
                'N_ofc': N_ofc,
                'N_ifc': N_ifc,
                'N_stor': N_st,
                'n_T': n_T,
                't_startup': t_startup,
                'error': error_msg,
                'sol_success': False
            })
    
    # --- error solve_ivp cannot handle ---
    except Exception as e:
        # Return failure case with detailed error information
        import traceback
        full_error = f"{str(e)} | Traceback: {traceback.format_exc()}"
        return make_output_dict({
            'error': full_error
        })
    
       
def compute_single_combination(linear_index, input_arrays_flat, param_shapes_array, total_time = 10*365*24*3600, STORE_TBE=False, vector_length = 100):

    # Get parameter indices from linear index
    param_indices = index_to_params(linear_index, param_shapes_array)
    
    # Extract parameters directly - fastest approach
    V_plasma = input_arrays_flat[0][param_indices[0]]
    T_i = input_arrays_flat[1][param_indices[1]]
    n_tot = input_arrays_flat[2][param_indices[2]]
    tau_p_T = input_arrays_flat[3][param_indices[3]]
    P_aux = input_arrays_flat[4][param_indices[4]]
    P_aux_DT_eq = input_arrays_flat[5][param_indices[5]]
    TBR_DT = input_arrays_flat[6][param_indices[6]]
    TBR_DDn = input_arrays_flat[7][param_indices[7]]
    tau_ifc = input_arrays_flat[8][param_indices[8]]
    tau_ofc = input_arrays_flat[9][param_indices[9]]
    eta_th = input_arrays_flat[10][param_indices[10]]
    capacity_factor = input_arrays_flat[11][param_indices[11]]
    cost_of_electricity = input_arrays_flat[12][param_indices[12]]
    
    
    result_dict = make_input_dict({
        'V_plasma': V_plasma,
        'T_i': T_i,
        'n_tot': n_tot,
        'tau_p_T': tau_p_T,
        'P_aux': P_aux,
        'P_aux_DT_eq': P_aux_DT_eq,
        'TBR_DT': TBR_DT,
        'TBR_DDn': TBR_DDn,
        'tau_ifc': tau_ifc,
        'tau_ofc': tau_ofc,
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
    
    # Precompute injection_rate_max
    injection_rate_max = (n_tot/2/tau_p_T*V_plasma + 0.25*n_tot**2*sigmav_DT*V_plasma - 0.25/2*n_tot**2*sigmav_DD_p*V_plasma)

    # SOLVE ODE SYSTEM
    ode_results = solve_ode_system(total_time,
        V_plasma, n_tot, tau_p_T, 
        P_aux, P_aux_DT_eq, 
        TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
        eta_th, capacity_factor, cost_of_electricity, 
        sigmav_DD_p, sigmav_DD_n, sigmav_DT, 
        injection_rate_max,
        STORE_TBE=STORE_TBE,  
    )
    
    # Package results (combine input parameters and ODE outputs)
    result = {
        'linear_index': linear_index,
        # Add all ODE results
        **ode_results
    }
    result_dict.update(result)
    print(f"results dictionary: {result_dict}")
    result_dict['N_ofc'] = fix_vector_length(result_dict['N_ofc'], vector_length)
    result_dict['N_ifc'] = fix_vector_length(result_dict['N_ifc'], vector_length)
    result_dict['N_stor'] = fix_vector_length(result_dict['N_stor'], vector_length)
    result_dict['n_T']   = fix_vector_length(result_dict['n_T'], vector_length)
    result_dict['n_D']   = fix_vector_length(result_dict['n_D'], vector_length)
    result_dict['P_DDn'] = fix_vector_length(result_dict['P_DDn'], vector_length)
    result_dict['P_DDp'] = fix_vector_length(result_dict['P_DDp'], vector_length)
    result_dict['P_DT']  = fix_vector_length(result_dict['P_DT'], vector_length)
    if STORE_TBE:
        result_dict['TBE']   = fix_vector_length(result_dict['TBE'], vector_length)

                
    return result_dict

