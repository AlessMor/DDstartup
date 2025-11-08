"""
DD Startup Analysis Tool - Main Entry Point

This script provides a command-line interface for running DD startup analysis
with various parameter configurations and methods.
"""

import sys
import warnings
from pathlib import Path

# Local imports
from ddstartup.utils.io_functions import (
    resolve_file_path,
    load_config,
    load_parameter_fields,
    prepare_input_data,
    print_configuration,
    generate_output_path
)
from ddstartup.utils.system_profiler import get_optimal_parameters, override_with_config
from ddstartup.methods.parametric_computation import run_parametric_analysis, print_parametric_summary
from ddstartup.methods.sobol_computation import run_sobol_analysis, print_sobol_summary
from ddstartup.utils.tools import parse_arguments

# Suppress warnings
warnings.filterwarnings("ignore", module="scipy.integrate")


def main():
    """Main execution function."""
    
    #############################################################################################
    #                                           INITIALIZATION
    #############################################################################################

    # ============================================================================
    # COMMAND-LINE ARGUMENT PARSING
    # ============================================================================
    # Parse command-line arguments to get parameter file, config file, and flags
    args = parse_arguments()
    
    # ============================================================================
    # FILE PATH RESOLUTION
    # ============================================================================
    # Resolve parameter and configuration file paths
    # - Parameter file: contains physics parameters (e.g., V_plasma, T_i, n_tot)
    #   Supports both Python (.py) and YAML (.yaml, .yml) formats
    # - Config file: contains analysis settings (e.g., method, n_jobs, verbose)
    try:
        param_file = resolve_file_path(args.params, 'inputs', ['.yaml', '.yml', '.py'])
        config_file = resolve_file_path(args.config, 'inputs', ['.yaml', '.yml'])
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # ============================================================================
    # CONFIGURATION LOADING
    # ============================================================================
    # Load YAML configuration file containing analysis settings
    try:
        config = load_config(config_file)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1
    
    # Override verbose setting if specified on command line
    if args.verbose:
        config['verbose'] = True
    
    verbose = config['verbose']
    
    # ============================================================================
    # DRY RUN CHECK (EARLY EXIT)
    # ============================================================================
    # If dry-run flag is set, exit here after validating configuration files
    # No need to prepare input data or profile system for dry-run
    if args.dry_run:
        print("\n✅ Dry run completed. Configuration files validated successfully.")
        print(f"✅ Parameter file: {param_file}")
        print(f"✅ Config file: {config_file}")
        print(f"✅ Analysis type: {config['analysis_type']}")
        print(f"✅ Method: {config['method']}")
        return 0
    
    # ============================================================================
    # PARAMETER LOADING AND INPUT DATA PREPARATION
    # ============================================================================
    # Load parameter fields from Python parameter file
    try:
        param_fields = load_parameter_fields(param_file)
    except ImportError as e:
        print(f"Error loading parameter fields: {e}", file=sys.stderr)
        return 1
    
    # Override max_simulation_time from parameter config if it exists
    if 'max_simulation_time' in param_fields and param_fields['max_simulation_time'] is not None:
        config['max_simulation_time'] = param_fields['max_simulation_time']
    
    # Prepare input data arrays for analysis
    # This converts parameter fields into proper format for computation
    try:
        input_data = prepare_input_data(param_fields, config['analysis_type'])
    except (ValueError, AttributeError) as e:
        print(f"Error preparing input data: {e}", file=sys.stderr)
        return 1
    
    # ============================================================================
    # SYSTEM PROFILING AND OPTIMIZATION
    # ============================================================================
    # Only profile system if performance parameters are not specified in config
    # This avoids unnecessary overhead when user provides explicit values
    try:
        needs_profiling = (
            config.get('n_jobs') is None or 
            config.get('chunk_size') is None or 
            config.get('batch_size') is None or
            (config['method'] == 'sobol' and config.get('N_SAMPLES') is None)
        )
        
        if needs_profiling:
            # Profile system hardware and determine optimal parameters
            optimal_params = get_optimal_parameters(
                analysis_method=config['method'],
                verbose=verbose
            )
            # Use config values if provided, otherwise use profiled optimal values
            optimal_params = override_with_config(optimal_params, config)
        else:
            # Use user-provided values directly (no profiling needed)
            optimal_params = {
                'n_jobs': config['n_jobs'],
                'chunk_size': config['chunk_size'],
                'batch_size': config['batch_size'],
            }
            if config['method'] == 'sobol':
                optimal_params['N_SAMPLES'] = config.get('N_SAMPLES')
                optimal_params['order'] = config.get('order')
        
        # Update config with final parallelization parameters
        config.update({
            'n_jobs': optimal_params['n_jobs'],
            'chunk_size': optimal_params['chunk_size'],
            'batch_size': optimal_params['batch_size'],
        })
        # Add Sobol-specific parameters if using Sobol analysis
        if config['method'] == 'sobol':
            config['N_SAMPLES'] = optimal_params['N_SAMPLES']
            config['order'] = optimal_params['order']
            
    except Exception as e:
        print(f"Error during system profiling: {e}", file=sys.stderr)
        return 1
    
    # ============================================================================
    # CONFIGURATION VALIDATION AND DISPLAY
    # ============================================================================
    # Print complete configuration for user review
    if verbose:
        print_configuration(config, param_fields, input_data, param_file, config_file)
    
    # ============================================================================
    # OUTPUT DIRECTORY AND FILE SETUP
    # ============================================================================
    # Generate output directory path and filename for results
    try:
        output_dir, output_file = generate_output_path(
            base_dir=config.get('output_dir', 'outputs'),
            analysis_method=config['method'],
            analysis_type=config['analysis_type']
        )
        
        if verbose:
            print(f"Output directory: {output_dir}")
            print(f"Output file: {output_file}")
            
    except Exception as e:
        print(f"Error creating output directory: {e}", file=sys.stderr)
        return 1
    
    # ============================================================================
    # PHYSICS MODULE IMPORT
    # ============================================================================
    # Import the appropriate compute function based on analysis type
    # The compute_single_combination function is the "work unit" that will be
    # executed in parallel. Each worker process receives:
    #   - linear_index: integer identifying which parameter combination to compute
    #   - input_arrays_flat: flattened parameter arrays (shared read-only data)
    #   - param_shapes_array: grid dimensions for index conversion
    # 
    # This design minimizes data transfer between processes while allowing
    # thousands of combinations to be computed in parallel.
    
    #############################################################################################
    #                                           ANALYSIS EXECUTION
    #############################################################################################

    # Run the analysis based on the selected method (parametric, sobol, or lhs)
    try:
        if config['method'] == 'parametric':
            # ----------------------------------------------------------------
            # PARAMETRIC ANALYSIS - PARALLEL GRID COMPUTATION
            # ----------------------------------------------------------------
            # Performs a full parameter sweep across all combinations
            # The compute function is now created internally by run_parametric_analysis
            # based on the analysis_type (lump or T_seeded).
            # 
            # Uses joblib.Parallel to:
            #   1. Spawn n_jobs worker processes
            #   2. Distribute linear indices (0, 1, 2, ..., n_combinations-1)
            #   3. Each worker calls the dynamically created compute function
            #   4. Results are collected and written to HDF5 in batches
            # 
            # Output: HDF5 file with complete grid of results
            stats = run_parametric_analysis(
                input_data=input_data,
                output_file=output_file,
                config=config,
                verbose=verbose,
                filter_expr=config.get('filter')
            )
            
            # Print analysis summary statistics
            print_parametric_summary(output_file, stats, verbose)
            
        elif config['method'] in ['sobol', 'lhs']:
            # ----------------------------------------------------------------
            # SENSITIVITY ANALYSIS - SOBOL/LHS SAMPLING
            # ----------------------------------------------------------------
            # Performs global sensitivity analysis using Sobol sequences
            # (Saltelli sampling scheme) to efficiently explore parameter space.
            # Computes first-order (S1) and total-order (ST) sensitivity indices.
            # 
            # Total evaluations: N × (2k + 2) where N = n_samples, k = n_params
            # 
            # Output: HDF5 file with Sobol indices for each output metric
            
            stats = run_sobol_analysis(
                input_data=input_data,
                output_file=output_file,
                config=config,
                compute_function=None,  # Not used - internal evaluation
                verbose=verbose
            )
            
            # Print sensitivity analysis summary
            print_sobol_summary(stats, verbose)
            
        elif config['method'] == 'elementary_effects':
            # ----------------------------------------------------------------
            # ELEMENTARY EFFECTS (MORRIS) - ONE-AT-A-TIME SENSITIVITY
            # ----------------------------------------------------------------
            # Performs Elementary Effects sensitivity analysis using random
            # trajectories through parameter space with one-at-a-time (OAT)
            # perturbations. Computes:
            #   - μ (mu): mean effect (with direction)
            #   - μ* (mu_star): mean absolute effect (main sensitivity)
            #   - σ (sigma): standard deviation (interaction effects)
            # 
            # This method is computationally cheaper than Sobol but still
            # provides global sensitivity information.
            # 
            # Output: HDF5 file with sensitivity indices for each parameter
            from ddstartup.methods.elemeffects_computation import (
                run_elementary_effects_analysis,
                print_elementary_effects_summary
            )
            
            stats = run_elementary_effects_analysis(
                input_data=input_data,
                output_file=output_file,
                config=config,
                verbose=verbose
            )
            
            # Print elementary effects summary
            print_elementary_effects_summary(stats, verbose)
            
        else:
            print(f"Error: Unknown analysis method: {config['method']}", file=sys.stderr)
            return 1
            
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # ============================================================================
    # COMPLETION
    # ============================================================================
    if verbose:
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
