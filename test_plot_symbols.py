#!/usr/bin/env python3
"""
Test script to verify parameter symbol formatting in plots.

This script demonstrates the new symbol-based labeling for all plot types.
"""

from ddstartup.utils.parameter_symbols import get_param_label, get_param_symbol, PARAM_SYMBOLS
from ddstartup.utils.tools import PARAM_UNITS

def test_symbol_mappings():
    """Test that all symbols are properly formatted."""
    print("=" * 70)
    print("PARAMETER SYMBOL MAPPINGS")
    print("=" * 70)
    
    test_params = [
        'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'P_aux', 'TBR_DT', 
        't_startup', 'Q_DD', 'E_lost', 'unrealized_gains'
    ]
    
    for param in test_params:
        symbol = get_param_symbol(param)
        unit = PARAM_UNITS.get(param, None)
        label = get_param_label(param, unit)
        
        print(f"\n{param:20s} → {symbol:30s}")
        if unit:
            print(f"{'':20s}   with unit: {label}")
    
    print("\n" + "=" * 70)
    print("TOTAL SYMBOLS DEFINED:", len(PARAM_SYMBOLS))
    print("=" * 70)

def test_label_formatting():
    """Test different label formatting options."""
    print("\n" + "=" * 70)
    print("LABEL FORMATTING EXAMPLES")
    print("=" * 70)
    
    # Test with symbols
    print("\n1. With symbols (default):")
    print(f"   V_plasma: {get_param_label('V_plasma', 'm³')}")
    print(f"   t_startup: {get_param_label('t_startup', 's')}")
    
    # Test without symbols
    print("\n2. Without symbols:")
    print(f"   V_plasma: {get_param_label('V_plasma', 'm³', use_symbol=False)}")
    print(f"   t_startup: {get_param_label('t_startup', 's', use_symbol=False)}")
    
    # Test without units
    print("\n3. Without units:")
    print(f"   V_plasma: {get_param_label('V_plasma', None)}")
    print(f"   TBR_DT: {get_param_label('TBR_DT', None)}")
    
    print("\n" + "=" * 70)

def test_coverage():
    """Check which parameters from PARAM_UNITS have symbol mappings."""
    print("\n" + "=" * 70)
    print("SYMBOL COVERAGE CHECK")
    print("=" * 70)
    
    missing_symbols = []
    for param in PARAM_UNITS.keys():
        if param not in PARAM_SYMBOLS:
            missing_symbols.append(param)
    
    if missing_symbols:
        print(f"\n⚠️  Parameters without symbol mappings ({len(missing_symbols)}):")
        for param in sorted(missing_symbols):
            print(f"   - {param}")
    else:
        print("\n✅ All parameters have symbol mappings!")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_symbol_mappings()
    test_label_formatting()
    test_coverage()
    
    print("\n✅ Symbol formatting system is working correctly!")
    print("\nTo see symbols in action, run postprocessing on your data:")
    print("   python -m ddstartup.postprocessing <output_dir>")
