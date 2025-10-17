#!/usr/bin/env python3
"""Quick script to regenerate interactive contour plots with fixed dropdowns."""

import sys
import hdf5plugin  # Must import before h5py
import h5py
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ddstartup.postprocessing.plot_contour_functions import plot_interactive_pairwise_contours

# Load data
h5_file = '/home/phd/dd_startup/outputs/20251014_212826_parametric_T_seeded/ddstartup_20251014_212826_parametric_T_seeded.h5'
output_dir = Path('/home/phd/dd_startup/outputs/20251014_212826_parametric_T_seeded')

print(f"📂 Loading data from {h5_file}")

with h5py.File(h5_file, 'r') as f:
    # Load all data into dataframe
    data = {}
    
    # Get parameter fields group
    param_fields_group = f['parameter_fields']
    param_fields = [k.replace('_values', '') for k in param_fields_group.keys()]
    
    # Load input parameters from top level (not from parameter_fields group)
    for field in param_fields:
        if field in f:
            data[field] = f[field][:]
    
    # Load target outputs
    for field in ['t_startup', 'unrealized_profits']:
        if field in f:
            data[field] = f[field][:]
    
    df = pd.DataFrame(data)
    
    # Filter to only successful runs
    sol_success = f['sol_success'][:]
    df = df[sol_success]
    
    print(f"✅ Loaded {len(df)} successful parameter combinations")
    print(f"📊 Input parameters: {param_fields}")
    print(f"🎯 Target variables: ['t_startup', 'unrealized_profits']")

# Generate interactive plots for each target
targets = ['t_startup', 'unrealized_profits']

for target in targets:
    print(f"\n🎨 Generating interactive plot for {target}...")
    
    # Get all input parameter names (exclude targets)
    inputs_present = [col for col in df.columns if col not in targets]
    
    # Generate plot using the correct signature
    html_file = plot_interactive_pairwise_contours(
        df=df,
        inputs=inputs_present,
        target=target,
        outputs_dir=output_dir,
        plot_name=f'contour_interactive_FIXED_{Path(h5_file).stem}_{target}'
    )
    
    print(f"💾 Saved: {html_file}")

print("\n✅ Done! Open the HTML files in a browser to test the fixed dropdown.")
