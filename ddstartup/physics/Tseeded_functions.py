import numpy as np
from scipy.integrate import solve_ivp
from ddstartup.utils.units_and_constants import *
from ddstartup.physics.reactionrates_functions import sigmav_DT_BoschHale, sigmav_DD_BoschHale
from numba import njit, prange
from ddstartup.utils.tools import index_to_params, make_input_dict, make_output_dict, fix_vector_length

@njit(cache=True)
def ode_system(t, y, 
               V_plasma, n_tot, tau_p_T, 
               TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
               sigmav_DD_p, sigmav_DD_n, sigmav_DT, 
               injection_rate_max, N_st_min):
    """
    Numba-compiled ODE system for tritium inventory evolution during DD startup.
    
    Computes time derivatives for the four state variables:
    - N_ofc: Out-of-fuel-cycle tritium inventory (atoms)
    - N_ifc: In-fuel-cycle tritium inventory (atoms)
    - N_st: Stored tritium inventory (atoms)
    - n_T: Tritium density in plasma (particles/m³)
    
    Args:
        t: Time (seconds)
        y: State vector [N_ofc, N_ifc, N_st, n_T]
        V_plasma: Plasma volume (m³)
        n_tot: Total particle density (m⁻³)
        tau_p_T: Tritium particle confinement time (s)
        TBR_DT: D-T tritium breeding ratio
        TBR_DDn: DD neutron tritium breeding ratio
        sigmav_DD_p: DD proton reaction rate (m³/s)
        sigmav_DD_n: DD neutron reaction rate (m³/s)
        sigmav_DT: D-T reaction rate (m³/s)
        injection_rate_max: Maximum tritium injection rate (atoms/s)
        N_st_min: Minimum stored tritium for injection (atoms)
        
    Returns:
        Array of time derivatives [dN_ofc/dt, dN_ifc/dt, dN_st/dt, dn_T/dt]
    """
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
                     injection_rate_max, vector_length, N_st_min=0.001/tritium_mass):
    """
    Solve tritium inventory ODE system and compute startup metrics.
    
    Integrates the ODE system until D-T operation is reached (n_T = 0.5*n_tot)
    or total_time is exceeded. Computes fusion powers, Q factors, and economic metrics.
    
    Args:
        total_time: Maximum simulation time (s)
        V_plasma: Plasma volume (m³)
        n_tot: Total particle density (m⁻³)
        tau_p_T: Tritium particle confinement time (s)
        P_aux: Auxiliary heating power (W)
        P_aux_DT_eq: Auxiliary power for equivalent D-T operation (W)
        TBR_DT: D-T tritium breeding ratio
        TBR_DDn: DD neutron tritium breeding ratio
        tau_ifc: In-fuel-cycle processing time (s)
        tau_ofc: Out-of-fuel-cycle processing time (s)
        eta_th: Thermal conversion efficiency
        capacity_factor: Plant capacity factor
        cost_of_electricity: Electricity cost ($/J)
        sigmav_DD_p: DD proton reaction rate (m³/s)
        sigmav_DD_n: DD neutron reaction rate (m³/s)
        sigmav_DT: D-T reaction rate (m³/s)
        injection_rate_max: Maximum tritium injection rate (atoms/s)
        vector_length: Number of time points for output arrays
        N_st_min: Minimum stored tritium for injection (atoms)
        
    Returns:
        Dictionary containing:
            - Time series: N_ofc, N_ifc, N_stor, n_T, n_D, P_DDn, P_DDp, P_DT, TBE
            - Scalars: t_startup, P_DT_eq, Q_DD, Q_DT_eq, E_lost, unrealized_gains
            - Status: sol_success (bool), error (str)
    """
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
    t_eval = np.linspace(0, total_time, vector_length)
    
    # --- Solve ODE system ---
    try:
        sol = solve_ivp(
            fun=tritium_inventory_odes_unitless,
            t_span=t_span,
            t_eval = t_eval,
            y0=y0,
            method='BDF',  # Good for stiff systems
            dense_output=False,
            events=[DT_reached_event, negative_event],
            rtol=1e-5,
            atol=1e10
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
            y_event = sol.y_events[0][0]    # Exact state at event time
            
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
            
            # Add exact event point to solution data for accurate interpolation
            # This ensures n_T[-1] = 0.5*n_tot exactly at t_startup
            t_with_event = np.append(sol.t, t_startup)
            y0_with_event = np.append(sol.y[0], y_event[0])
            y1_with_event = np.append(sol.y[1], y_event[1])
            y2_with_event = np.append(sol.y[2], y_event[2])
            y3_with_event = np.append(sol.y[3], y_event[3])
            
            # Sort by time (event time should be at end, but be safe)
            sort_idx = np.argsort(t_with_event)
            t_sorted = t_with_event[sort_idx]
            y0_sorted = y0_with_event[sort_idx]
            y1_sorted = y1_with_event[sort_idx]
            y2_sorted = y2_with_event[sort_idx]
            y3_sorted = y3_with_event[sort_idx]
            
            # Create uniform time grid from 0 to t_startup with vector_length points
            t_interp = np.linspace(0, t_startup, vector_length)
            
            # Interpolate solution onto the new time grid using augmented data
            # This ensures we have exactly vector_length points from 0 to t_startup
            # with correct final values at t_startup
            N_ofc = np.interp(t_interp, t_sorted, y0_sorted)
            N_ifc = np.interp(t_interp, t_sorted, y1_sorted)
            N_st = np.interp(t_interp, t_sorted, y2_sorted)
            n_T = np.interp(t_interp, t_sorted, y3_sorted)
            n_D = n_tot - n_T  # Deuterium density

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
        
            # Total fusion energies by integration (use interpolated time grid)
            E_fusion_DDn = np.trapezoid(P_DDn, t_interp)
            E_fusion_DDp = np.trapezoid(P_DDp, t_interp)
            E_fusion_DT = np.trapezoid(P_DT, t_interp)
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
                'TBE': TBE_vector,
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
    
       
def compute_single_combination(linear_index, input_arrays_flat, param_shapes_array, total_time = 10*365*24*3600, vector_length = 100):
    """
    Compute T_seeded analysis for a single parameter combination.
    
    This is the core function called by parallel workers during parametric analysis.
    Converts linear index to multi-dimensional parameter indices, extracts parameter
    values, computes reaction rates, solves ODE system, and packages results.
    
    Args:
        linear_index: Integer index (0 to n_combinations-1) identifying parameter set
        input_arrays_flat: List of 1D arrays, one per parameter
        param_shapes_array: Array of parameter grid shapes for index conversion
        total_time: Maximum simulation time in seconds (default: 10 years)
        vector_length: Number of time points in output arrays (default: 100)
        
    Returns:
        Dictionary with input parameters and computed results:
            - Input echoes: V_plasma, T_i, n_tot, tau_p_T, P_aux, etc.
            - Time series: N_ofc, N_ifc, N_stor, n_T, n_D, P_DDn, P_DDp, P_DT, TBE
            - Scalars: t_startup, Q_DD, Q_DT_eq, E_lost, unrealized_gains
            - Status: linear_index, sol_success, error
    """

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
        'error': ""
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
        injection_rate_max, vector_length
    )

    # Package results (combine input parameters and ODE outputs)
    result = {
        'linear_index': linear_index,
        # Add all ODE results
        **ode_results
    }
    result_dict.update(result)

    # JIT-accelerated postprocessing for successful ODE results
    # Only run if ODE was successful and t_startup is finite
    if ode_results.get('sol_success', False) and np.isfinite(ode_results.get('t_startup', np.inf)):
        # Fix vector lengths for ODE outputs
        N_ofc = fix_vector_length(ode_results['N_ofc'], vector_length)
        N_ifc = fix_vector_length(ode_results['N_ifc'], vector_length)
        N_st = fix_vector_length(ode_results['N_stor'], vector_length)
        n_T = fix_vector_length(ode_results['n_T'], vector_length)
        t_startup = ode_results.get('t_startup')
        # Call JIT postprocessing
        P_DDn, P_DDp, P_DT, P_DT_eq, Q_DD, Q_DT_eq, E_lost, unrealized_gains, TBE_vector, n_D = postprocess_fusion_results_Tseeded(t_startup,
            N_ofc, N_ifc, N_st, n_T, n_tot, V_plasma, sigmav_DD_p, sigmav_DD_n, sigmav_DT, TBR_DT, TBR_DDn, tau_ifc, eta_th, capacity_factor, cost_of_electricity, P_aux, P_aux_DT_eq, E_DDn, E_DDp, E_DT, injection_rate_max, 0.001/tritium_mass, vector_length
        )
        result_dict['N_ofc'] = N_ofc
        result_dict['N_ifc'] = N_ifc
        result_dict['N_stor'] = N_st
        result_dict['n_T'] = n_T
        result_dict['n_D'] = n_D
        result_dict['P_DDn'] = P_DDn
        result_dict['P_DDp'] = P_DDp
        result_dict['P_DT'] = P_DT
        result_dict['P_DT_eq'] = P_DT_eq
        result_dict['Q_DD'] = Q_DD
        result_dict['Q_DT_eq'] = Q_DT_eq
        result_dict['E_lost'] = E_lost
        result_dict['unrealized_gains'] = unrealized_gains
        result_dict['TBE'] = TBE_vector
    else:
        # Fix vector lengths for error cases
        result_dict['N_ofc'] = fix_vector_length(result_dict['N_ofc'], vector_length)
        result_dict['N_ifc'] = fix_vector_length(result_dict['N_ifc'], vector_length)
        result_dict['N_stor'] = fix_vector_length(result_dict['N_stor'], vector_length)
        result_dict['n_T']   = fix_vector_length(result_dict['n_T'], vector_length)
        result_dict['n_D']   = fix_vector_length(result_dict.get('n_D', np.full(vector_length, np.nan)), vector_length)
        result_dict['P_DDn'] = fix_vector_length(result_dict.get('P_DDn', np.full(vector_length, np.nan)), vector_length)
        result_dict['P_DDp'] = fix_vector_length(result_dict.get('P_DDp', np.full(vector_length, np.nan)), vector_length)
        result_dict['P_DT']  = fix_vector_length(result_dict.get('P_DT', np.full(vector_length, np.nan)), vector_length)
        result_dict['TBE']   = fix_vector_length(result_dict.get('TBE', np.full(vector_length, np.nan)), vector_length)

    return result_dict

@njit(cache=True)
def postprocess_fusion_results_Tseeded(t_startup, N_ofc, N_ifc, N_st, n_T, n_tot, V_plasma, sigmav_DD_p, sigmav_DD_n, sigmav_DT, TBR_DT, TBR_DDn, tau_ifc, eta_th, capacity_factor, cost_of_electricity, P_aux, P_aux_DT_eq, E_DDn, E_DDp, E_DT, injection_rate_max, N_st_min, vector_length):
    """
    JIT-compiled postprocessing of fusion results for performance.
    
    Computes fusion powers, energy integrals, Q factors, and economic metrics
    from ODE solution. Uses Numba JIT compilation for speed.
    
    Args:
        t_startup: Time to reach D-T operation (s)
        N_ofc, N_ifc, N_st: Tritium inventory time series (atoms)
        n_T: Tritium density time series (m⁻³)
        n_tot: Total particle density (m⁻³)
        V_plasma: Plasma volume (m³)
        sigmav_DD_p, sigmav_DD_n, sigmav_DT: Reaction rates (m³/s)
        TBR_DT, TBR_DDn: Tritium breeding ratios
        tau_ifc: In-fuel-cycle time (s)
        eta_th: Thermal efficiency
        capacity_factor: Plant capacity factor
        cost_of_electricity: Electricity cost ($/J)
        P_aux, P_aux_DT_eq: Auxiliary powers (W)
        E_DDn, E_DDp, E_DT: Reaction energies (J)
        injection_rate_max: Max injection rate (atoms/s)
        N_st_min: Min stored tritium (atoms)
        vector_length: Length of output arrays
        
    Returns:
        Tuple of (P_DDn, P_DDp, P_DT, P_DT_eq, Q_DD, Q_DT_eq, 
                  E_lost, unrealized_gains, TBE_vector, n_D)
    """
    n_D = n_tot - n_T
    n_D_squared = n_D ** 2
    n_D_n_T = n_D * n_T
    V_E_DDn = V_plasma * E_DDn
    V_E_DDp = V_plasma * E_DDp
    V_E_DT = V_plasma * E_DT

    P_DDn = n_D_squared * sigmav_DD_n / 2 * V_E_DDn
    P_DDp = n_D_squared * sigmav_DD_p / 2 * V_E_DDp
    P_DT = n_D_n_T * sigmav_DT * V_E_DT
    P_DT_eq = n_tot/2 * n_tot/2 * sigmav_DT * V_E_DT

    t_vec = np.linspace(0, t_startup, vector_length)

    E_fusion_DDn = trapz_numba(P_DDn, t_vec)
    E_fusion_DDp = trapz_numba(P_DDp, t_vec)
    E_fusion_DT = trapz_numba(P_DT, t_vec)
    E_fusion_total_DD = E_fusion_DDn + E_fusion_DDp + E_fusion_DT
    E_fusion_DT_eq = P_DT_eq * t_vec[-1]

    E_aux_DD = P_aux * t_vec[-1]
    E_aux_DT_eq = P_aux_DT_eq * t_vec[-1]

    E_e_net_DD = capacity_factor * (eta_th * (E_fusion_total_DD) - E_aux_DD)
    E_e_net_DT_eq = capacity_factor * (eta_th * (E_fusion_DT_eq) - E_aux_DT_eq)

    Q_DD = E_fusion_total_DD / E_aux_DD if E_aux_DD > 0 else np.inf
    Q_DT_eq = E_fusion_DT_eq / E_aux_DT_eq if E_aux_DT_eq > 0 else np.inf

    E_lost = E_e_net_DT_eq - E_e_net_DD
    unrealized_gains = E_lost * cost_of_electricity

    mask = N_st > N_st_min
    inj_rate = np.empty_like(N_ifc)
    for i in prange(len(N_ifc)):
        if mask[i]:
            inj = N_ifc[i] / tau_ifc - lambda_T * N_st[i]
            inj_rate[i] = min(max(inj, 0.0), injection_rate_max)
        else:
            inj_rate[i] = 0.0
    TBE_vector = np.full_like(N_st, np.nan)
    for i in prange(len(N_st)):
        if mask[i] and inj_rate[i] > 0:
            TBE_vector[i] = (n_D[i] * n_T[i] * sigmav_DT) / inj_rate[i]
    return P_DDn, P_DDp, P_DT, P_DT_eq, Q_DD, Q_DT_eq, E_lost, unrealized_gains, TBE_vector, n_D

@njit(cache=True)
def trapz_numba(y, x):
    """
    JIT-compiled trapezoidal integration for Numba compatibility.
    
    Equivalent to numpy.trapz but works inside @njit decorated functions.
    
    Args:
        y: Function values
        x: Independent variable values
        
    Returns:
        Integrated value using trapezoidal rule
    """
    s = 0.0
    for i in range(1, len(x)):
        s += 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
    return s