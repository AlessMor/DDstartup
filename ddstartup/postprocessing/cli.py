"""
DD Startup Postprocessing Tool - Main Entry Point

This script provides a command-line interface for postprocessing DD startup
analysis results stored in HDF5 files. It can generate various plots and
analyses with flexible filtering options.

Usage:
    python -m ddstartup.postprocessing [config.yaml] [OPTIONS]

Examples:
    # Use YAML configuration file (recommended)
    python -m ddstartup.postprocessing postprocess_config.yaml
    
    # Use YAML with command-line overrides
    python -m ddstartup.postprocessing postprocess_config.yaml --plots kde
    
    # Process latest file with defaults (no config file)
    python -m ddstartup.postprocessing
    
    # Process specific file (command-line only)
    python -m ddstartup.postprocessing --files outputs/my_results.h5
    
    # Process multiple files
    python -m ddstartup.postprocessing --files file1.h5 file2.h5
    
    # Specify target variables
    python -m ddstartup.postprocessing --targets t_startup unrealized_profits
    
    # Apply filters (command-line)
    python -m ddstartup.postprocessing --input-filter "V_plasma<150" --output-filter "unrealized_profits>2e6"
    
    # Generate specific plot types
    python -m ddstartup.postprocessing --plots kde parcoords pdf

Note:
    Configuration file (YAML) is the recommended method for complex postprocessing.
    Command-line arguments override settings from the YAML file.
    See inputs/postprocess_config.yaml for a complete configuration example.
"""

import argparse
import sys
import warnings
from pathlib import Path

# Local imports
from ddstartup.postprocessing.postprocess_functions import (
    load_h5_to_dataframe,
    get_input_parameters,
    scale_target,
    find_latest_output_folder,
    find_latest_h5_file,
    load_yaml_config,
    parse_filter_expression,
    clean_filters,
    apply_filters
)
from ddstartup.postprocessing.plot_kde_functions import kde_quartile_plot
from ddstartup.postprocessing.plot_parcoords_functions import generate_parcoords_plot
from ddstartup.postprocessing.plot_pdf_functions import generate_pdf_plot

# Suppress warnings
warnings.filterwarnings("ignore")


def generate_plots(files, targets, input_filters, output_filters, plot_types, output_dir):
    """
    Generate requested plots for the given files and targets.
    
    Args:
        files: List of HDF5 file paths
        targets: List of target variables
        input_filters: Dictionary of input filters
        output_filters: Dictionary of output filters
        plot_types: List of plot types to generate ('kde', 'parcoords', 'pdf')
        output_dir: Directory to save plots
    """
    print(f"\n{'='*80}")
    print(f"GENERATING PLOTS")
    print(f"{'='*80}")
    
    for file_path in files:
        print(f"\n📁 Processing: {file_path.name}")
        
        # Determine file type for labeling
        if "lump" in file_path.name.lower():
            file_type = "lump"
        elif "t_seeded" in file_path.name.lower() or "tseeded" in file_path.name.lower():
            file_type = "Tseeded"
        else:
            file_type = "unknown"
        
        for target in targets:
            print(f"\n  🎯 Target: {target}")
            
            # Load and filter data
            print(f"   Loading data...")
            df = load_h5_to_dataframe(file_path)
            
            # Apply filters
            df_filtered = apply_filters(df, input_filters, output_filters, target)
            
            if len(df_filtered) == 0:
                print(f"   ⚠️  No data remaining after filtering. Skipping.")
                continue
            
            # Scale target variable
            df_filtered, target_unit = scale_target(df_filtered, target)
            
            # Get input parameters
            input_parameters = get_input_parameters(df_filtered, target, filename=str(file_path))
            
            # Generate KDE plot
            if 'kde' in plot_types:
                print(f"   Generating KDE plot...")
                plot_name = f"kde_quartiles_{file_path.stem}_{target}.png"
                try:
                    kde_quartile_plot(df_filtered, target, input_parameters, 
                                    target_unit, output_dir, file_type, plot_name)
                    print(f"   ✅ Saved: {plot_name}")
                except Exception as e:
                    print(f"   ❌ Error generating KDE plot: {e}")
            
            # Generate parallel coordinates plot
            if 'parcoords' in plot_types:
                print(f"   Generating parallel coordinates plot...")
                plot_name = f"parcoords_{file_path.stem}_{target}.html"
                try:
                    generate_parcoords_plot(df_filtered, target, input_parameters, 
                                          target_unit, file_type, output_dir / plot_name)
                    print(f"   ✅ Saved: {plot_name}")
                except Exception as e:
                    print(f"   ❌ Error generating parallel coordinates plot: {e}")
            
            # Generate PDF plot
            if 'pdf' in plot_types:
                print(f"   Generating PDF plot...")
                plot_name = f"pdf_{file_path.stem}_{target}.png"
                try:
                    generate_pdf_plot({str(file_path): {target: df_filtered[target].values}}, 
                                    target, [f"{file_type}"], output_filters, 
                                    output_dir / plot_name)
                    print(f"   ✅ Saved: {plot_name}")
                except Exception as e:
                    print(f"   ❌ Error generating PDF plot: {e}")


def main():
    """Main execution function."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Postprocess DD startup analysis HDF5 results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'config',
        nargs='?',
        default=None,
        help='YAML configuration file (e.g., postprocess_config.yaml)'
    )
    
    parser.add_argument(
        '--files', '-f',
        nargs='+',
        help='HDF5 file(s) to process (overrides config file)'
    )
    
    parser.add_argument(
        '--targets', '-t',
        nargs='+',
        help='Target variables to analyze (overrides config file)'
    )
    
    parser.add_argument(
        '--input-filter', '-if',
        type=str,
        help='Filter for input parameters (e.g., "V_plasma<150,n_tot>1e20")'
    )
    
    parser.add_argument(
        '--output-filter', '-of',
        type=str,
        help='Filter for output variables (e.g., "unrealized_profits>2e6,t_startup<1e8")'
    )
    
    parser.add_argument(
        '--plots', '-p',
        nargs='+',
        choices=['kde', 'parcoords', 'pdf', 'all'],
        help='Plot types to generate (overrides config file)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for plots (overrides config file)'
    )
    
    args = parser.parse_args()
    
    # ============================================================================
    # LOAD CONFIGURATION
    # ============================================================================
    
    # Get base directory (where main.py is located)
    base_dir = Path(__file__).parent.parent
    
    # Load YAML config if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            # Try relative to current directory first
            if not config_path.exists():
                # Try relative to inputs directory
                config_path = base_dir / 'inputs' / args.config
        
        if not config_path.exists():
            print(f"❌ Error: Configuration file not found: {args.config}")
            sys.exit(1)
        
        print(f"📋 Loading configuration from: {config_path.name}")
        config = load_yaml_config(config_path)
    else:
        # Use defaults if no config file
        config = {
            'files': 'latest',
            'target_variables': ['unrealized_profits', 't_startup'],
            'input_filters': {},
            'output_filters': {},
            'plots': {'generate_all': True},
            'output': {'directory': 'default', 'verbose': True}
        }
        print("📋 Using default configuration (no config file specified)")
    
    # Command-line arguments override config file
    if args.files:
        config['files'] = args.files if len(args.files) > 1 else args.files[0]
    if args.targets:
        config['target_variables'] = args.targets
    if args.output_dir:
        config['output']['directory'] = args.output_dir
    if args.plots:
        if 'all' in args.plots:
            config['plots']['generate_all'] = True
        else:
            config['plots']['generate_all'] = False
            config['plots']['kde'] = 'kde' in args.plots
            config['plots']['parcoords'] = 'parcoords' in args.plots
            config['plots']['pdf'] = 'pdf' in args.plots
    
    # Parse command-line filter expressions (override config if provided)
    if args.input_filter:
        cli_input_filters = parse_filter_expression(args.input_filter)
        if 'input_filters' not in config:
            config['input_filters'] = {}
        config['input_filters'].update(cli_input_filters)
    
    if args.output_filter:
        cli_output_filters = parse_filter_expression(args.output_filter)
        if 'output_filters' not in config:
            config['output_filters'] = {}
        config['output_filters'].update(cli_output_filters)
    
    # ============================================================================
    # RESOLVE FILE PATHS
    # ============================================================================
    
    files_config = config.get('files', 'latest')
    
    if isinstance(files_config, str) and files_config == 'latest':
        # Find latest folder and files in outputs directory
        outputs_dir = base_dir / 'outputs'
        if not outputs_dir.exists():
            print(f"❌ Error: Outputs directory not found: {outputs_dir}")
            sys.exit(1)
        
        latest_folder, h5_files = find_latest_output_folder(outputs_dir)
        
        if latest_folder and h5_files:
            file_paths = h5_files
            print(f"📂 Using latest folder: {latest_folder.name}")
            print(f"   Found {len(h5_files)} HDF5 file(s)")
            # Store the latest folder for output directory
            config['_latest_folder'] = latest_folder
        else:
            # Fallback to root outputs directory
            latest_file = find_latest_h5_file(outputs_dir)
            if latest_file is None:
                print(f"❌ Error: No HDF5 files found in {outputs_dir}")
                sys.exit(1)
            file_paths = [latest_file]
            print(f"📂 Using latest file: {latest_file.name} (in root outputs/)")
            config['_latest_folder'] = outputs_dir
    else:
        # User provided specific file(s) or folder(s) in config
        if isinstance(files_config, str):
            files_list = [files_config]
        else:
            files_list = files_config
        
        file_paths = []
        for f in files_list:
            p = Path(f)
            if not p.is_absolute():
                # Try relative to current directory first
                if not p.exists():
                    # Try relative to base directory
                    p_alt = base_dir / f
                    if p_alt.exists():
                        p = p_alt
                    # Try relative to outputs directory
                    elif (base_dir / 'outputs' / f).exists():
                        p = base_dir / 'outputs' / f
            
            if not p.exists():
                print(f"❌ Error: Path not found: {f}")
                sys.exit(1)
            
            # If it's a directory, find HDF5 files inside
            if p.is_dir():
                h5_files = list(p.glob("*.h5"))
                if not h5_files:
                    print(f"❌ Error: No HDF5 files found in directory: {p}")
                    sys.exit(1)
                file_paths.extend(h5_files)
                print(f"📁 Found {len(h5_files)} file(s) in: {p.name}")
                # Store folder for output if not set
                if '_latest_folder' not in config:
                    config['_latest_folder'] = p
            else:
                # It's a file
                file_paths.append(p.resolve())
    
    # ============================================================================
    # PARSE FILTERS
    # ============================================================================
    
    input_filters = clean_filters(config.get('input_filters', {}))
    output_filters = clean_filters(config.get('output_filters', {}))
    
    if input_filters:
        print(f"\n🔍 Input filters:")
        for var, conds in input_filters.items():
            if conds['min'] is not None and conds['max'] is not None:
                print(f"   {var}: {conds['min']} ≤ {var} ≤ {conds['max']}")
            elif conds['min'] is not None:
                print(f"   {var}: {var} ≥ {conds['min']}")
            elif conds['max'] is not None:
                print(f"   {var}: {var} ≤ {conds['max']}")
    
    if output_filters:
        print(f"\n🔍 Output filters:")
        for var, conds in output_filters.items():
            if conds['min'] is not None and conds['max'] is not None:
                print(f"   {var}: {conds['min']} ≤ {var} ≤ {conds['max']}")
            elif conds['min'] is not None:
                print(f"   {var}: {var} ≥ {conds['min']}")
            elif conds['max'] is not None:
                print(f"   {var}: {var} ≤ {conds['max']}")
    
    # ============================================================================
    # GET TARGET VARIABLES
    # ============================================================================
    
    targets = config.get('target_variables', ['unrealized_profits', 't_startup'])
    print(f"\n🎯 Target variables: {', '.join(targets)}")
    
    # ============================================================================
    # DETERMINE PLOT TYPES
    # ============================================================================
    
    plots_config = config.get('plots', {})
    if plots_config.get('generate_all', True):
        plot_types = ['kde', 'parcoords', 'pdf']
    else:
        plot_types = []
        if plots_config.get('kde', False):
            plot_types.append('kde')
        if plots_config.get('parcoords', False):
            plot_types.append('parcoords')
        if plots_config.get('pdf', False):
            plot_types.append('pdf')
    
    print(f"📊 Plot types: {', '.join(plot_types)}")
    
    # ============================================================================
    # DETERMINE OUTPUT DIRECTORY
    # ============================================================================
    
    output_config = config.get('output', {})
    output_dir_setting = output_config.get('directory', 'default')
    
    if output_dir_setting == 'default':
        # Use the latest folder if available, otherwise postprocess subfolder
        if '_latest_folder' in config:
            output_dir = config['_latest_folder']
            print(f"💾 Output directory: {output_dir} (latest folder)")
        else:
            output_dir = Path(__file__).parent / 'postprocess'
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"💾 Output directory: {output_dir}")
    else:
        output_dir = Path(output_dir_setting)
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Output directory: {output_dir} (custom)")
    
    # ============================================================================
    # GENERATE PLOTS
    # ============================================================================
    
    generate_plots(file_paths, targets, input_filters, output_filters, 
                   plot_types, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ POSTPROCESSING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
