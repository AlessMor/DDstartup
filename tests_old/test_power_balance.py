#!/usr/bin/env python3
"""
Test script to verify P_aux calculation from power balance.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/alessmor/Scrivania/dd_startup')

from ddstartup.physics.Tseeded_functions import compute_single_combination as compute_tseeded
from ddstartup.physics.lump_functions import compute_single_combination as compute_lump

# Test parameters
test_params_tseeded = {
    'V_plasma': np.array([1000.0]),  # m³
    'T_i': np.array([15000.0]),      # eV (15 keV)
    'n_tot': np.array([1e20]),       # m⁻³
    'tau_p_T': np.array([2.0]),      # s
    'P_aux': np.array([np.nan]),     # Calculate from power balance
    'P_aux_DT_eq': np.array([np.nan]),  # Calculate from power balance
    'TBR_DT': np.array([1.1]),
    'TBR_DDn': np.array([0.5]),
    'tau_ifc': np.array([3600.0]),   # s (1 hour)
    'tau_ofc': np.array([86400.0]),  # s (1 day)
    'eta_th': np.array([0.4]),
    'capacity_factor': np.array([0.9]),
    'price_of_electricity': np.array([0.1e-6]),  # $/J
}

test_params_lump = {
    'V_plasma': np.array([1000.0]),  # m³
    'T_i': np.array([15000.0]),      # eV (15 keV)
    'n_tot': np.array([1e20]),       # m⁻³
    'tau_p_T': np.array([2.0]),      # s
    'tau_p_He3': np.array([2.0]),    # s
    'P_aux': np.array([np.nan]),     # Calculate from power balance
    'P_aux_DT_eq': np.array([np.nan]),  # Calculate from power balance
    'TBR_DT': np.array([1.1]),
    'TBR_DDn': np.array([0.5]),
    'I_target': np.array([10.0]),    # kg
    'eta_th': np.array([0.4]),
    'capacity_factor': np.array([0.9]),
    'price_of_electricity': np.array([0.1e-6]),  # $/J
}

def test_tseeded():
    """Test T_seeded model with calculated P_aux."""
    print("=" * 80)
    print("Testing T_seeded model with P_aux calculation from power balance")
    print("=" * 80)
    
    # Flatten parameter arrays
    input_arrays_flat = [test_params_tseeded[key] for key in [
        'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'P_aux', 'P_aux_DT_eq',
        'TBR_DT', 'TBR_DDn', 'tau_ifc', 'tau_ofc', 'eta_th', 
        'capacity_factor', 'price_of_electricity'
    ]]
    
    # Create param_shapes_array (all length 1 since we have single values)
    param_shapes = np.array([len(arr) for arr in input_arrays_flat])
    
    # Compute for linear_index = 0
    try:
        result = compute_tseeded(0, input_arrays_flat, param_shapes, 
                                 total_time=10*365*24*3600, vector_length=100)
        
        print("\n✅ Computation successful!")
        print(f"\nInput parameters:")
        print(f"  V_plasma: {test_params_tseeded['V_plasma'][0]:.2e} m³")
        print(f"  T_i: {test_params_tseeded['T_i'][0]:.2e} eV")
        print(f"  n_tot: {test_params_tseeded['n_tot'][0]:.2e} m⁻³")
        
        print(f"\nCalculated powers:")
        if 'P_aux' in result and result['P_aux'] is not None:
            print(f"  P_aux (calculated): {result.get('P_aux', np.nan):.2e} W")
        if 'P_aux_DT_eq' in result and result['P_aux_DT_eq'] is not None:
            print(f"  P_aux_DT_eq (calculated): {result.get('P_aux_DT_eq', np.nan):.2e} W")
        
        print(f"\nResults:")
        print(f"  t_startup: {result.get('t_startup', np.nan):.2e} s")
        print(f"  Q_DD: {result.get('Q_DD', np.nan):.3f}")
        print(f"  Q_DT_eq: {result.get('Q_DT_eq', np.nan):.3f}")
        print(f"  sol_success: {result.get('sol_success', False)}")
        if result.get('error'):
            print(f"  Error: {result.get('error')}")
            
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

def test_lump():
    """Test lump model with calculated P_aux."""
    print("\n" + "=" * 80)
    print("Testing Lump model with P_aux calculation from power balance")
    print("=" * 80)
    
    # Flatten parameter arrays
    input_arrays_flat = [test_params_lump[key] for key in [
        'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'tau_p_He3', 'P_aux', 'P_aux_DT_eq',
        'TBR_DT', 'TBR_DDn', 'I_target', 'eta_th', 
        'capacity_factor', 'price_of_electricity'
    ]]
    
    # Create param_shapes_array (all length 1 since we have single values)
    param_shapes = np.array([len(arr) for arr in input_arrays_flat])
    
    # Compute for linear_index = 0
    try:
        result = compute_lump(0, input_arrays_flat, param_shapes)
        
        print("\n✅ Computation successful!")
        print(f"\nInput parameters:")
        print(f"  V_plasma: {test_params_lump['V_plasma'][0]:.2e} m³")
        print(f"  T_i: {test_params_lump['T_i'][0]:.2e} eV")
        print(f"  n_tot: {test_params_lump['n_tot'][0]:.2e} m⁻³")
        
        print(f"\nResults:")
        print(f"  n_T: {result.get('n_T', np.nan):.2e} m⁻³")
        print(f"  n_D: {result.get('n_D', np.nan):.2e} m⁻³")
        print(f"  t_startup: {result.get('t_startup', np.nan):.2e} s")
        print(f"  Q_DD: {result.get('Q_DD', np.nan):.3f}")
        print(f"  Q_DT_eq: {result.get('Q_DT_eq', np.nan):.3f}")
        print(f"  sol_success: {result.get('sol_success', False)}")
            
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_tseeded()
    test_lump()
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
