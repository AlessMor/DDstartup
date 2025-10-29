import numpy as np
from scipy.integrate import solve_ivp
from ddstartup.utils.units_and_constants import *
from ddstartup.physics.reactionrates_functions import sigmav_DT_BoschHale, sigmav_DD_BoschHale
from numba import njit
from ddstartup.utils.tools import index_to_params, make_input_dict, make_output_dict, fix_vector_length

# OPTIMIZATION: Cache for reaction rates to avoid recomputation
_sigmav_cache = {}

def get_cached_reaction_rates(T_i):
    """
    Get reaction rates with caching to avoid recomputation for repeated T_i values.
    
    Args:
        T_i: Ion temperature in eV
        
    Returns:
        Tuple of (sigmav_DD_p, sigmav_DD_n, sigmav_DT)
    """
    # Round to 0.1 eV precision for caching
    T_i_key = round(T_i / 0.1) * 0.1
    
    if T_i_key not in _sigmav_cache:
        T_i_array = np.array([T_i])
        sigmav_DD_results = sigmav_DD_BoschHale(T_i_array)
        sigmav_DD_p = sigmav_DD_results[1][0]
        sigmav_DD_n = sigmav_DD_results[2][0]
        sigmav_DT = sigmav_DT_BoschHale(T_i_array)[0]
        _sigmav_cache[T_i_key] = (sigmav_DD_p, sigmav_DD_n, sigmav_DT)
    
    return _sigmav_cache[T_i_key]

@njit(cache=True, fastmath=True)
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

    # Compute injection rate (Numba-compatible, optimized)
    if N_st > N_st_min:
        inj_rate = N_ifc / tau_ifc - lambda_T * N_st
        # Single comparison chain (faster than nested ifs)
        injection_rate = max(0.0, min(inj_rate, injection_rate_max))
    else:
        injection_rate = 0.0

    # Compute reaction rates (optimized with pre-computed terms)
    n_D = n_tot - n_T
    n_D_squared = n_D * n_D
    half_n_D_squared = 0.5 * n_D_squared
    n_D_n_T = n_D * n_T
    
    # Reaction rate products (reduce multiplications)
    Tdot_DDn = TBR_DDn * half_n_D_squared * sigmav_DD_n * V_plasma
    Tdot_DDp = half_n_D_squared * sigmav_DD_p * V_plasma
    Tdot_DT_breeding = TBR_DT * n_D_n_T * sigmav_DT * V_plasma
    Tdot_burn = n_D_n_T * sigmav_DT * V_plasma

    # Pre-compute common terms
    N_ofc_decay = N_ofc * (1.0 / tau_ofc + lambda_T)
    N_ifc_decay = N_ifc * (1.0 / tau_ifc + lambda_T)
    N_st_decay = N_st * lambda_T
    n_T_loss = n_T / tau_p_T

    # ODEs (optimized)
    dN_ofc_dt = Tdot_DT_breeding + Tdot_DDn - N_ofc_decay
    dN_ifc_dt = N_ofc / tau_ofc - N_ifc_decay + n_T_loss * V_plasma
    dN_stor_dt = N_ifc / tau_ifc - N_st_decay - injection_rate
    dnT_dt = (injection_rate + Tdot_DDp - n_T_loss * V_plasma - Tdot_burn) / V_plasma

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
            - Scalars: t_startup, P_DT_eq, Q_DD, Q_DT_eq, E_lost, unrealized_profits
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
    # OPTIMIZATION: Don't specify t_eval - let solver choose adaptive timesteps
    # This is faster and we interpolate later anyway
    
    # --- Solve ODE system ---
    # OPTIMIZATION: LSODA with proven settings for 87.5% success rate
    try:
        sol = solve_ivp(
            fun=tritium_inventory_odes_unitless,
            t_span=t_span,
            y0=y0,
            method='BDF',  # Fastest adaptive method
            dense_output=False,
            jac = lambda t, y: jacobian(
                    t, y, V_plasma, n_tot, tau_p_T,
                    TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
                    sigmav_DD_p, sigmav_DD_n, sigmav_DT,
                    injection_rate_max, N_st_min
                ),
            events=[DT_reached_event, negative_event],
            rtol=1e-4,  # Proven tolerance for reliability
            atol=1e10   # Appropriate for large atom numbers
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
            
            # OPTIMIZATION: Store raw solution arrays instead of interpolating here
            # Interpolation will happen ONCE in postprocessing
            # Add exact event point to solution data for accurate interpolation later
            t_with_event = np.append(sol.t, t_startup)
            y0_with_event = np.append(sol.y[0], y_event[0])
            y1_with_event = np.append(sol.y[1], y_event[1])
            y2_with_event = np.append(sol.y[2], y_event[2])
            y3_with_event = np.append(sol.y[3], y_event[3])
            
            # Sort by time (event time should be at end, but be safe)
            sort_idx = np.argsort(t_with_event)
            t_raw = t_with_event[sort_idx]
            N_ofc_raw = y0_with_event[sort_idx]
            N_ifc_raw = y1_with_event[sort_idx]
            N_st_raw = y2_with_event[sort_idx]
            n_T_raw = y3_with_event[sort_idx]
            
            # Return RAW arrays (no interpolation yet) - this avoids double interpolation
            # Postprocessing will interpolate once and compute all derived quantities
            return make_output_dict({
                't_raw': t_raw,
                'N_ofc_raw': N_ofc_raw,
                'N_ifc_raw': N_ifc_raw,
                'N_stor_raw': N_st_raw,
                'n_T_raw': n_T_raw,
                't_startup': t_startup,
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
            - Scalars: t_startup, Q_DD, Q_DT_eq, E_lost, unrealized_profits
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
    # OPTIMIZATION: Use cached reaction rates
    sigmav_DD_p, sigmav_DD_n, sigmav_DT = get_cached_reaction_rates(T_i)
    
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
    # OPTIMIZATION: Raw arrays from ODE solver are processed here with SINGLE interpolation
    # Only run if ODE was successful and t_startup is finite
    if ode_results.get('sol_success', False) and np.isfinite(ode_results.get('t_startup', np.inf)):
        # Extract raw arrays from ODE solution
        t_raw = ode_results['t_raw']
        N_ofc_raw = ode_results['N_ofc_raw']
        N_ifc_raw = ode_results['N_ifc_raw']
        N_st_raw = ode_results['N_stor_raw']
        n_T_raw = ode_results['n_T_raw']
        t_startup = ode_results['t_startup']
        
        # Call optimized JIT postprocessing (single interpolation + all calculations)
        N_ofc, N_ifc, N_st, n_T, n_D, P_DDn, P_DDp, P_DT, P_DT_eq, Q_DD, Q_DT_eq, E_lost, unrealized_profits, TBE_vector = postprocess_fusion_results_Tseeded(
            t_startup, t_raw, N_ofc_raw, N_ifc_raw, N_st_raw, n_T_raw,
            n_tot, V_plasma, sigmav_DD_p, sigmav_DD_n, sigmav_DT, 
            TBR_DT, TBR_DDn, tau_ifc, eta_th, capacity_factor, cost_of_electricity, 
            P_aux, P_aux_DT_eq, E_DDn, E_DDp, E_DT, 
            injection_rate_max, 0.001/tritium_mass, vector_length
        )
        
        # Store interpolated and computed results
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
        result_dict['unrealized_profits'] = unrealized_profits
        result_dict['TBE'] = TBE_vector
    else:
        # Fix vector lengths for error cases
        result_dict['N_ofc'] = fix_vector_length(result_dict.get('N_ofc', np.full(vector_length, np.nan)), vector_length)
        result_dict['N_ifc'] = fix_vector_length(result_dict.get('N_ifc', np.full(vector_length, np.nan)), vector_length)
        result_dict['N_stor'] = fix_vector_length(result_dict.get('N_stor', np.full(vector_length, np.nan)), vector_length)
        result_dict['n_T']   = fix_vector_length(result_dict.get('n_T', np.full(vector_length, np.nan)), vector_length)
        result_dict['n_D']   = fix_vector_length(result_dict.get('n_D', np.full(vector_length, np.nan)), vector_length)
        result_dict['P_DDn'] = fix_vector_length(result_dict.get('P_DDn', np.full(vector_length, np.nan)), vector_length)
        result_dict['P_DDp'] = fix_vector_length(result_dict.get('P_DDp', np.full(vector_length, np.nan)), vector_length)
        result_dict['P_DT']  = fix_vector_length(result_dict.get('P_DT', np.full(vector_length, np.nan)), vector_length)
        result_dict['TBE']   = fix_vector_length(result_dict.get('TBE', np.full(vector_length, np.nan)), vector_length)

    return result_dict

@njit(cache=True, fastmath=True)
def postprocess_fusion_results_Tseeded(t_startup, t_raw, N_ofc_raw, N_ifc_raw, N_st_raw, n_T_raw, n_tot, V_plasma, sigmav_DD_p, sigmav_DD_n, sigmav_DT, TBR_DT, TBR_DDn, tau_ifc, eta_th, capacity_factor, cost_of_electricity, P_aux, P_aux_DT_eq, E_DDn, E_DDp, E_DT, injection_rate_max, N_st_min, vector_length):
    """
    JIT-compiled postprocessing of fusion results for performance.
    
    OPTIMIZED VERSION: Performs interpolation ONCE and computes all derived quantities.
    This avoids the double-interpolation bottleneck.
    
    Computes fusion powers, energy integrals, Q factors, and economic metrics
    from ODE solution. Uses Numba JIT compilation for speed.
    
    Args:
        t_startup: Time to reach D-T operation (s)
        t_raw: Raw time points from ODE solver
        N_ofc_raw, N_ifc_raw, N_st_raw: Raw tritium inventory from solver (atoms)
        n_T_raw: Raw tritium density from solver (m⁻³)
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
        Tuple of (N_ofc, N_ifc, N_st, n_T, n_D, P_DDn, P_DDp, P_DT, P_DT_eq, 
                  Q_DD, Q_DT_eq, E_lost, unrealized_profits, TBE_vector)
    """
    # OPTIMIZATION: Use faster uniform grid creation
    dt = t_startup / (vector_length - 1)
    t_interp = np.arange(vector_length) * dt
    
    # Interpolate RAW solution onto uniform grid (SINGLE INTERPOLATION)
    # Using np.interp which is fast for monotonic data
    N_ofc = np.interp(t_interp, t_raw, N_ofc_raw)
    N_ifc = np.interp(t_interp, t_raw, N_ifc_raw)
    N_st = np.interp(t_interp, t_raw, N_st_raw)
    n_T = np.interp(t_interp, t_raw, n_T_raw)
    
    # Compute derived quantities from interpolated data
    n_D = n_tot - n_T
    
    # Pre-compute common terms to avoid redundant multiplications
    half_sigmav_DD_n = 0.5 * sigmav_DD_n
    half_sigmav_DD_p = 0.5 * sigmav_DD_p
    n_D_squared = n_D * n_D
    n_D_n_T = n_D * n_T
    
    # Pre-multiply volume and energy terms (constants outside loop)
    V_E_DDn = V_plasma * E_DDn
    V_E_DDp = V_plasma * E_DDp
    V_E_DT = V_plasma * E_DT

    # Calculate fusion powers (optimized with pre-computed terms)
    P_DDn = n_D_squared * half_sigmav_DD_n * V_E_DDn
    P_DDp = n_D_squared * half_sigmav_DD_p * V_E_DDp
    P_DT = n_D_n_T * sigmav_DT * V_E_DT
    P_DT_eq = 0.25 * n_tot * n_tot * sigmav_DT * V_E_DT  # Optimized: 0.5*0.5 = 0.25

    # Energy integrals using trapezoid rule
    E_fusion_DDn = trapz_numba(P_DDn, t_interp)
    E_fusion_DDp = trapz_numba(P_DDp, t_interp)
    E_fusion_DT = trapz_numba(P_DT, t_interp)
    E_fusion_total_DD = E_fusion_DDn + E_fusion_DDp + E_fusion_DT
    E_fusion_DT_eq = P_DT_eq * t_startup

    # Pre-compute energy terms
    E_aux_DD = P_aux * t_startup
    E_aux_DT_eq = P_aux_DT_eq * t_startup

    # Net energy with pre-computed auxiliaries
    E_e_net_DD = capacity_factor * (eta_th * E_fusion_total_DD - E_aux_DD)
    E_e_net_DT_eq = capacity_factor * (eta_th * E_fusion_DT_eq - E_aux_DT_eq)

    # Q factors with safe division
    Q_DD = E_fusion_total_DD / E_aux_DD if E_aux_DD > 0 else np.inf
    Q_DT_eq = E_fusion_DT_eq / E_aux_DT_eq if E_aux_DT_eq > 0 else np.inf

    E_lost = E_e_net_DT_eq - E_e_net_DD
    unrealized_profits = E_lost * cost_of_electricity

    # Vectorized TBE computation (much faster than loops)
    # Compute injection rate vectorized
    inj_temp = N_ifc / tau_ifc - lambda_T * N_st
    inj_rate = np.clip(inj_temp, 0.0, injection_rate_max)
    inj_rate = np.where(N_st > N_st_min, inj_rate, 0.0)
    
    # Compute TBE vectorized
    TBE_vector = np.where(
        (N_st > N_st_min) & (inj_rate > 0),
        (n_D * n_T * sigmav_DT) / inj_rate,
        np.nan
    )
    
    return N_ofc, N_ifc, N_st, n_T, n_D, P_DDn, P_DDp, P_DT, P_DT_eq, Q_DD, Q_DT_eq, E_lost, unrealized_profits, TBE_vector

@njit(cache=True, fastmath=True)
def trapz_numba(y, x):
    """
    JIT-compiled trapezoidal integration for Numba compatibility.
    
    OPTIMIZED with fastmath for additional speed.
    Equivalent to numpy.trapz but works inside @njit decorated functions.
    
    Args:
        y: Function values
        x: Independent variable values
        
    Returns:
        Integrated value using trapezoidal rule
    """
    # OPTIMIZATION: Vectorized computation when possible
    n = len(x)
    if n < 2:
        return 0.0
    
    # For uniform grid (common case), use simplified formula
    dx = x[1] - x[0]
    is_uniform = True
    for i in range(2, n):
        if abs(x[i] - x[i-1] - dx) > 1e-10 * dx:
            is_uniform = False
            break
    
    if is_uniform:
        # FAST PATH: Uniform grid
        return dx * (0.5 * (y[0] + y[n-1]) + np.sum(y[1:n-1]))
    else:
        # SLOW PATH: Non-uniform grid
        s = 0.0
        for i in range(1, n):
            s += 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
        return s

def compute_batch_combinations(linear_indices, input_arrays_flat, param_shapes_array, total_time=10*365*24*3600, vector_length=100):
    """
    Compute multiple T_seeded cases in a batch with optimizations.
    
    This function processes multiple parameter combinations more efficiently than
    calling compute_single_combination repeatedly by:
    - Pre-computing unique reaction rates once per unique T_i value
    - Minimizing overhead from repeated function calls
    
    Args:
        linear_indices: List/array of integer indices to compute
        input_arrays_flat: List of 1D arrays, one per parameter
        param_shapes_array: Array of parameter grid shapes
        total_time: Maximum simulation time in seconds
        vector_length: Number of time points in output arrays
        
    Returns:
        List of result dictionaries, one per linear_index
    """
    results = []
    
    # Pre-compute all unique T_i values and cache reaction rates
    T_i_array = input_arrays_flat[1]  # T_i is second parameter
    unique_Ti = np.unique(T_i_array)
    
    print(f"🔧 Pre-computing reaction rates for {len(unique_Ti)} unique T_i values...")
    for T_i in unique_Ti:
        _ = get_cached_reaction_rates(T_i)
    
    # Process each combination
    for idx in linear_indices:
        result = compute_single_combination(
            idx, input_arrays_flat, param_shapes_array, 
            total_time, vector_length
        )
        results.append(result)
    
    return results

def jacobian(t, y,
             V_plasma, n_tot, tau_p_T,
             TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
             sigmav_DD_p, sigmav_DD_n, sigmav_DT,
             injection_rate_max, N_st_min):
    # y = [N_ofc, N_ifc, N_st, n_T]
    N_ofc, N_ifc, N_st, n_T = y
    n_D = n_tot - n_T

    # piecewise derivative of injection_rate wrt N_ifc, N_st
    dinj_dN_ifc = 0.0
    dinj_dN_st  = 0.0
    if N_st > N_st_min:
        inj_raw = N_ifc / tau_ifc - lambda_T * N_st
        if 0.0 < inj_raw < injection_rate_max:
            dinj_dN_ifc = 1.0 / tau_ifc
            dinj_dN_st  = -lambda_T

    # partials wrt n_T for the volumetric terms
    dTdot_DDn_dnT  = - TBR_DDn * n_D * sigmav_DD_n * V_plasma
    dTdot_DDp_dnT  = - n_D * sigmav_DD_p * V_plasma
    dTdot_DTbr_dnT =   TBR_DT * (n_D - n_T) * sigmav_DT * V_plasma
    dTdot_burn_dnT =   (n_D - n_T) * sigmav_DT * V_plasma
    dnTloss_dnT_V  =   (1.0 / tau_p_T) * V_plasma

    J = np.zeros((4, 4), dtype=float)

    # Row: dN_ofc/dt = Tdot_DT_breeding + Tdot_DDn - N_ofc*(1/tau_ofc + lambda_T)
    J[0, 0] = - (1.0 / tau_ofc + lambda_T)
    J[0, 3] = dTdot_DTbr_dnT + dTdot_DDn_dnT

    # Row: dN_ifc/dt = N_ofc/tau_ofc - N_ifc*(1/tau_ifc + lambda_T) + n_T/tau_p_T * V
    J[1, 0] = 1.0 / tau_ofc
    J[1, 1] = - (1.0 / tau_ifc + lambda_T)
    J[1, 3] = dnTloss_dnT_V

    # Row: dN_st/dt = N_ifc/tau_ifc - lambda_T*N_st - injection_rate
    J[2, 1] = 1.0 / tau_ifc - dinj_dN_ifc
    J[2, 2] = - lambda_T - dinj_dN_st

    # Row: dn_T/dt = (injection_rate + Tdot_DDp - n_T/tau_p_T*V - Tdot_burn)/V
    J[3, 1] =  dinj_dN_ifc / V_plasma
    J[3, 2] =  dinj_dN_st  / V_plasma
    J[3, 3] = (dTdot_DDp_dnT - dnTloss_dnT_V - dTdot_burn_dnT) / V_plasma

    return J
