#!/usr/bin/env python
"""
Reconstruct and fix input parameters in HDF5 files with NaN values.

This utility script fixes HDF5 files from parametric analyses where input
parameters were incorrectly stored as NaN due to a bug in the compute functions.

The script:
1. Loads parameter definitions from YAML files (same as used for analysis)
2. Reads the linear_index and parameter_shapes from the HDF5 file
3. Reconstructs all input parameter values from the parameter grid
4. Writes the corrected values back to the HDF5 file

Usage:
    python -m ddstartup.utils.reconstruct_hdf5_inputs \
        --hdf5 outputs/20251108_125645_parametric_T_seeded/ddstartup_20251108_125645_parametric_T_seeded.h5 \
        --params inputs/parametric_tseeded.yaml \
        --config inputs/parametric_tseeded.yaml

"""

import argparse
import h5py
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ddstartup.utils.io_functions import load_parameter_fields, prepare_input_data


def check_needs_reconstruction(h5_path: str, input_params: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if HDF5 file needs reconstruction by checking for NaN input parameters.
    
    Args:
        h5_path: Path to HDF5 file
        input_params: List of input parameter names to check
        
    Returns:
        Tuple of (needs_reconstruction, list of params with NaN)
    """
    try:
        import hdf5plugin  # For LZ4 compression
    except ImportError:
        print("⚠️  Warning: hdf5plugin not installed. Install with: pip install hdf5plugin")
    
    nan_params = []
    with h5py.File(h5_path, 'r') as f:
        for param in input_params:
            if param in f:
                sample = f[param][:min(100, f[param].shape[0])]
                if np.all(np.isnan(sample)):
                    nan_params.append(param)
    
    return len(nan_params) > 0, nan_params


def reconstruct_input_parameters(
    h5_path: str,
    param_file: str,
    config_file: str,
    dry_run: bool = False,
    backup: bool = True
) -> Dict:
    """
    Reconstruct input parameters in HDF5 file from parameter definitions.
    
    Args:
        h5_path: Path to HDF5 file to fix
        param_file: Path to parameter YAML file
        config_file: Path to configuration YAML file
        dry_run: If True, only check what would be done without modifying file
        backup: If True, create backup of original file before modifying
        
    Returns:
        Dictionary with reconstruction statistics
    """
    h5_path = Path(h5_path)
    
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    
    print(f"{'='*70}")
    print(f"HDF5 INPUT PARAMETER RECONSTRUCTION")
    print(f"{'='*70}")
    print(f"HDF5 file: {h5_path}")
    print(f"Parameter file: {param_file}")
    print(f"Config file: {config_file}")
    print(f"Mode: {'DRY RUN' if dry_run else 'WRITE'}")
    print(f"{'='*70}\n")
    
    # Load parameter definitions and prepare input data (same as parametric analysis)
    print("📋 Loading parameter definitions...")
    param_fields = load_parameter_fields(param_file)
    
    # Load config file to get analysis_type
    from ddstartup.utils.io_functions import load_config
    config = load_config(config_file)
    analysis_type = config.get('analysis_type', 'T_seeded')
    
    print(f"Analysis type: {analysis_type}")
    input_data = prepare_input_data(param_fields, analysis_type)
    param_names = list(input_data.keys())
    
    print(f"✓ Loaded {len(param_names)} parameters:")
    for name in param_names:
        arr = input_data[name]
        print(f"  - {name}: {len(arr)} values, range [{arr.min():.3e}, {arr.max():.3e}]")
    
    # Define input parameters (same as in load_h5_to_dataframe)
    INPUT_PARAMS = ['V_plasma', 'n_tot', 'T_i', 'tau_p_T', 'tau_p_He3', 'P_aux', 'P_aux_DT_eq',
                    'tau_ifc', 'tau_ofc', 'TBR_DT', 'TBR_DDn', 'eta_th', 'capacity_factor',
                    'price_of_electricity', 'I_target']
    
    # Check which parameters need reconstruction
    print("\n🔍 Checking HDF5 file for NaN input parameters...")
    needs_fix, nan_params = check_needs_reconstruction(str(h5_path), INPUT_PARAMS)
    
    if not needs_fix:
        print("✅ All input parameters have valid values - no reconstruction needed!")
        return {'status': 'no_fix_needed', 'nan_params': [], 'fixed_params': []}
    
    print(f"⚠️  Found {len(nan_params)} parameters with NaN values:")
    for param in nan_params:
        print(f"  - {param}")
    
    if dry_run:
        print("\n🔍 DRY RUN - Would reconstruct these parameters but not modifying file")
        return {'status': 'dry_run', 'nan_params': nan_params, 'fixed_params': []}
    
    # Create backup if requested
    if backup:
        backup_path = h5_path.with_suffix('.h5.backup')
        if backup_path.exists():
            print(f"\n⚠️  Backup already exists: {backup_path}")
            response = input("Overwrite backup? (y/N): ")
            if response.lower() != 'y':
                print("❌ Aborting - backup not overwritten")
                return {'status': 'aborted', 'nan_params': nan_params, 'fixed_params': []}
        
        import shutil
        print(f"\n💾 Creating backup: {backup_path}")
        shutil.copy2(h5_path, backup_path)
        print(f"✓ Backup created ({backup_path.stat().st_size / 1024**2:.1f} MB)")
    
    # Open HDF5 file for reconstruction
    print("\n🔧 Reconstructing input parameters...")
    
    try:
        import hdf5plugin  # For LZ4 compression
    except ImportError:
        pass
    
    fixed_params = []
    
    with h5py.File(h5_path, 'r+') as f:  # Read-write mode
        # Get metadata
        param_shapes = f.attrs.get('parameter_shapes', None)
        if param_shapes is None:
            raise ValueError("HDF5 file missing 'parameter_shapes' attribute - cannot reconstruct")
        
        param_shapes = np.array(param_shapes, dtype=np.int64)
        n_combinations = int(np.prod(param_shapes))
        
        # Load linear indices
        if 'linear_index' not in f:
            raise ValueError("HDF5 file missing 'linear_index' dataset - cannot reconstruct")
        
        linear_indices = f['linear_index'][:]
        
        # Check if linear_index is all NaN - if so, generate from row position
        if np.all(np.isnan(linear_indices)):
            print("  ⚠️  linear_index is all NaN - generating from row positions...")
            linear_indices = np.arange(len(linear_indices), dtype=np.float64)
        
        print(f"  Total combinations: {n_combinations:,}")
        print(f"  Parameter grid shape: {list(param_shapes)}")
        print(f"  Loaded {len(linear_indices):,} linear indices")
        
        # Create flattened input arrays (same as parametric analysis)
        input_arrays_flat = [np.asarray(input_data[name]) for name in param_names]
        
        # Reconstruct each parameter
        for i, param_name in enumerate(param_names):
            if param_name not in INPUT_PARAMS:
                continue
            
            if param_name not in f:
                print(f"  ⚠️  {param_name}: not in HDF5 file, skipping")
                continue
            
            # Check if needs fixing
            sample = f[param_name][:min(100, f[param_name].shape[0])]
            if not np.all(np.isnan(sample)):
                print(f"  ✓ {param_name}: already has valid values, skipping")
                continue
            
            print(f"  🔧 {param_name}: reconstructing...", end='', flush=True)
            
            # Reconstruct values from linear indices
            param_values = np.empty(len(linear_indices), dtype=np.float64)
            param_grid = input_arrays_flat[i]
            
            for j, lin_idx in enumerate(linear_indices):
                # Convert linear index to multi-dimensional index
                idx = int(lin_idx)
                temp_idx = np.zeros(len(param_shapes), dtype=np.int64)
                
                # Compute multi-index (same as index_to_params)
                for k in range(len(param_shapes) - 1, -1, -1):
                    temp_idx[k] = idx % param_shapes[k]
                    idx //= param_shapes[k]
                
                # Get value from parameter grid
                param_values[j] = param_grid[temp_idx[i]]
            
            # Write reconstructed values back to HDF5
            f[param_name][:] = param_values
            
            # Verify reconstruction
            unique_vals = np.unique(param_values)
            print(f" ✓ ({len(unique_vals)} unique values)")
            fixed_params.append(param_name)
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ RECONSTRUCTION COMPLETE")
    print(f"{'='*70}")
    print(f"Fixed {len(fixed_params)} parameters:")
    for param in fixed_params:
        print(f"  ✓ {param}")
    
    if backup:
        print(f"\n💾 Original file backed up to: {backup_path}")
    
    print(f"\n📁 Modified file: {h5_path}")
    print(f"   Size: {h5_path.stat().st_size / 1024**2:.1f} MB")
    
    return {
        'status': 'success',
        'nan_params': nan_params,
        'fixed_params': fixed_params,
        'backup_path': str(backup_path) if backup else None
    }


def main():
    """Command-line interface for HDF5 input reconstruction."""
    parser = argparse.ArgumentParser(
        description='Reconstruct input parameters in HDF5 files with NaN values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check what would be fixed (dry run)
  python -m ddstartup.utils.reconstruct_hdf5_inputs \\
      --hdf5 outputs/20251108_125645_parametric_T_seeded/ddstartup_20251108_125645_parametric_T_seeded.h5 \\
      --params inputs/params_test.yaml \\
      --config inputs/parametric_tseeded.yaml \\
      --dry-run
  
  # Fix the file (creates backup)
  python -m ddstartup.utils.reconstruct_hdf5_inputs \\
      --hdf5 outputs/20251108_125645_parametric_T_seeded/ddstartup_20251108_125645_parametric_T_seeded.h5 \\
      --params inputs/params_test.yaml \\
      --config inputs/parametric_tseeded.yaml
  
  # Fix without backup
  python -m ddstartup.utils.reconstruct_hdf5_inputs \\
      --hdf5 outputs/20251108_125645_parametric_T_seeded/ddstartup_20251108_125645_parametric_T_seeded.h5 \\
      --params inputs/params_test.yaml \\
      --config inputs/parametric_tseeded.yaml \\
      --no-backup
"""
    )
    
    parser.add_argument('--hdf5', required=True,
                       help='Path to HDF5 file to fix')
    parser.add_argument('--params', required=True,
                       help='Path to parameter YAML file used for the analysis')
    parser.add_argument('--config', required=True,
                       help='Path to configuration YAML file used for the analysis')
    parser.add_argument('--dry-run', action='store_true',
                       help='Check what would be done without modifying file')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup of original file')
    
    args = parser.parse_args()
    
    try:
        result = reconstruct_input_parameters(
            h5_path=args.hdf5,
            param_file=args.params,
            config_file=args.config,
            dry_run=args.dry_run,
            backup=not args.no_backup
        )
        
        # Exit codes
        if result['status'] == 'success':
            sys.exit(0)
        elif result['status'] == 'no_fix_needed':
            sys.exit(0)
        elif result['status'] == 'dry_run':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
