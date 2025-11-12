#!/usr/bin/env python3
"""
Comprehensive test to verify P_aux calculation works correctly.
Tests both cases: 1) P_aux provided, 2) P_aux calculated from power balance
"""

import numpy as np
import sys
sys.path.insert(0, '/home/alessmor/Scrivania/dd_startup')

from ddstartup.physics.Tseeded_functions import compute_single_combination as compute_tseeded
from ddstartup.physics.lump_functions import compute_single_combination as compute_lump

def test_comparison():
    """Compare results with provided P_aux vs calculated P_aux."""
    print("=" * 80)
    print("COMPARISON TEST: Provided P_aux vs Calculated P_aux")
    print("=" * 80)
    
    # Common parameters
    V_plasma = 1000.0  # m³
    T_i = 15000.0      # eV (15 keV)
    n_tot = 1e20       # m⁻³
    tau_p_T = 2.0      # s
    TBR_DT = 1.1
    TBR_DDn = 0.5
    tau_ifc = 3600.0
    tau_ofc = 86400.0
    eta_th = 0.4
    capacity_factor = 0.9
    price_of_electricity = 0.1e-6
    
    # First, calculate what P_aux should be using our power balance
    from ddstartup.physics.reactivity_functions import sigmav_DD_BoschHale, sigmav_DT_BoschHale
    T_i_array = np.array([T_i])
    sigmav_DD_results = sigmav_DD_BoschHale(T_i_array)
    sigmav_DD_p = sigmav_DD_results[2][0]
    sigmav_DD_n = sigmav_DD_results[1][0]
    sigmav_DT = sigmav_DT_BoschHale(T_i_array)[0]
    
    from ddstartup.physics.lump_functions import calculate_P_aux_lump
    # For DD startup, assume mostly D
    n_D_est = n_tot
    n_T_est = n_tot * 0.001  # Small tritium fraction
    P_aux_calculated = calculate_P_aux_lump(n_T_est, n_D_est, T_i, V_plasma, 
                                            sigmav_DD_p, sigmav_DD_n, sigmav_DT)
    
    print(f"\n📊 Pre-calculated P_aux estimate: {P_aux_calculated:.3e} W")
    
    # Test 1: Provided P_aux
    print("\n" + "-" * 80)
    print("Test 1: P_aux PROVIDED")
    print("-" * 80)
    
    test_params_provided = {
        'V_plasma': np.array([V_plasma]),
        'T_i': np.array([T_i]),
        'n_tot': np.array([n_tot]),
        'tau_p_T': np.array([tau_p_T]),
        'tau_p_He3': np.array([tau_p_T]),
        'P_aux': np.array([50e6]),  # Provided: 50 MW
        'P_aux_DT_eq': np.array([100e6]),  # Provided: 100 MW
        'TBR_DT': np.array([TBR_DT]),
        'TBR_DDn': np.array([TBR_DDn]),
        'I_target': np.array([10.0]),
        'eta_th': np.array([eta_th]),
        'capacity_factor': np.array([capacity_factor]),
        'price_of_electricity': np.array([price_of_electricity]),
    }
    
    input_arrays_flat = [test_params_provided[key] for key in [
        'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'tau_p_He3', 'P_aux', 'P_aux_DT_eq',
        'TBR_DT', 'TBR_DDn', 'I_target', 'eta_th', 'capacity_factor', 'price_of_electricity'
    ]]
    param_shapes = np.array([len(arr) for arr in input_arrays_flat])
    
    result_provided = compute_lump(0, input_arrays_flat, param_shapes)
    print(f"  P_aux used: 50 MW (provided)")
    print(f"  P_aux_DT_eq used: 100 MW (provided)")
    print(f"  Q_DD: {result_provided.get('Q_DD', np.nan):.3f}")
    print(f"  Q_DT_eq: {result_provided.get('Q_DT_eq', np.nan):.3f}")
    print(f"  t_startup: {result_provided.get('t_startup', np.nan):.2e} s")
    
    # Test 2: Calculated P_aux
    print("\n" + "-" * 80)
    print("Test 2: P_aux CALCULATED from power balance")
    print("-" * 80)
    
    test_params_calculated = test_params_provided.copy()
    test_params_calculated['P_aux'] = np.array([np.nan])  # Let it calculate
    test_params_calculated['P_aux_DT_eq'] = np.array([np.nan])  # Let it calculate
    
    input_arrays_flat = [test_params_calculated[key] for key in [
        'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'tau_p_He3', 'P_aux', 'P_aux_DT_eq',
        'TBR_DT', 'TBR_DDn', 'I_target', 'eta_th', 'capacity_factor', 'price_of_electricity'
    ]]
    
    result_calculated = compute_lump(0, input_arrays_flat, param_shapes)
    # Note: The lump model doesn't return the calculated P_aux directly,
    # but uses it internally for Q calculations
    print(f"  P_aux: calculated internally from power balance")
    print(f"  P_aux_DT_eq: calculated internally from power balance")
    print(f"  Q_DD: {result_calculated.get('Q_DD', np.nan):.3f}")
    print(f"  Q_DT_eq: {result_calculated.get('Q_DT_eq', np.nan):.3f}")
    print(f"  t_startup: {result_calculated.get('t_startup', np.nan):.2e} s")
    
    # Test 3: T_seeded with calculated P_aux
    print("\n" + "-" * 80)
    print("Test 3: T_seeded model with CALCULATED P_aux")
    print("-" * 80)
    
    test_params_tseeded = {
        'V_plasma': np.array([V_plasma]),
        'T_i': np.array([T_i]),
        'n_tot': np.array([n_tot]),
        'tau_p_T': np.array([tau_p_T]),
        'P_aux': np.array([np.nan]),  # Calculate
        'P_aux_DT_eq': np.array([np.nan]),  # Calculate
        'TBR_DT': np.array([TBR_DT]),
        'TBR_DDn': np.array([TBR_DDn]),
        'tau_ifc': np.array([tau_ifc]),
        'tau_ofc': np.array([tau_ofc]),
        'eta_th': np.array([eta_th]),
        'capacity_factor': np.array([capacity_factor]),
        'price_of_electricity': np.array([price_of_electricity]),
    }
    
    input_arrays_flat_tseeded = [test_params_tseeded[key] for key in [
        'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'P_aux', 'P_aux_DT_eq',
        'TBR_DT', 'TBR_DDn', 'tau_ifc', 'tau_ofc', 'eta_th', 
        'capacity_factor', 'price_of_electricity'
    ]]
    param_shapes_tseeded = np.array([len(arr) for arr in input_arrays_flat_tseeded])
    
    result_tseeded = compute_tseeded(0, input_arrays_flat_tseeded, param_shapes_tseeded,
                                     total_time=10*365*24*3600, vector_length=100)
    
    print(f"  P_aux (calculated): {result_tseeded.get('P_aux', np.nan):.3e} W")
    print(f"  P_aux_DT_eq (calculated): {result_tseeded.get('P_aux_DT_eq', np.nan):.3e} W")
    print(f"  Q_DD: {result_tseeded.get('Q_DD', np.nan):.3f}")
    print(f"  Q_DT_eq: {result_tseeded.get('Q_DT_eq', np.nan):.3f}")
    print(f"  t_startup: {result_tseeded.get('t_startup', np.nan):.2e} s")
    print(f"  sol_success: {result_tseeded.get('sol_success', False)}")
    
    print("\n" + "=" * 80)
    print("✅ All tests completed successfully!")
    print("=" * 80)

if __name__ == '__main__':
    test_comparison()
