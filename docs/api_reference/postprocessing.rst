Postprocessing Module
=====================

The postprocessing module provides tools for analyzing and visualizing HDF5 output from DD startup simulations.

Quick Start
-----------

.. code-block:: bash

   # Auto-detect latest results
   python -m ddstartup.postprocessing
   
   # Use configuration file
   python -m ddstartup.postprocessing inputs/postprocess_config.yaml
   
   # Command-line options
   python -m ddstartup.postprocessing \
       --files outputs/results.h5 \
       --targets t_startup unrealized_gains \
       --plots kde parcoords pdf

Overview and code structure
--------

The module processes HDF5 files containing simulation results and generates different types of plots.

The :func:`main` function is the core of the postprocessing.
1. Step 0: Set directory and parse CLI args
   * The code finds the root directory (parent of parent, and assumes the other inputs and outputs folders are there, if not specified otherwise in the CL). [:func:`base_dir`]
   * An "args" ArgumentParser object is created, to read and save all CL arguments (including the name/location of the postprocessing file). [:func:`build_parser`]
2. Step 1: Build the "args" dictionary
   * A "config" dictionary is created, by reading the info inside the postprocessing config YAML file specified in command line. [:func:`load_config_from_args`]
   * A list of h5 file paths to analyse is saved. [:func:`resolve_file_paths`]
   * Eventual "args" passed by CL overwrite the "config" dictionary. [:func:`apply_cli_overrides`]
3. Step 2:  

Plotting Functions
-----------------------

KDE Plots
~~~~~~~~~

.. py:function:: kde_quartile_plot(df_filtered, target, input_parameters, target_unit, outputs_dir, file_type, plot_name)

   Generate KDE plots for each input parameter, split by target quartiles.
   
   Creates a grid of subplots showing probability distributions across four quartiles of the target variable. Also saves a CSV file with quartile extreme values.
   
   :param df_filtered: Filtered DataFrame
   :param target: Target variable name
   :param input_parameters: List of parameters to plot
   :param target_unit: Unit string for target
   :param outputs_dir: Output directory
   :param file_type: Analysis type for title
   :param plot_name: Output filename

Parallel Coordinates
~~~~~~~~~~~~~~~~~~~~

.. py:function:: generate_parcoords_plot(df_filtered, target, input_parameters, target_unit, file_type, output_path)

   Generate interactive parallel coordinates plot using Plotly.
   
   Shows relationships between multiple parameters as parallel vertical axes. Colors lines by target variable quantiles (6 bins). Automatically samples large datasets (>1M rows).
   
   :param df_filtered: Filtered DataFrame
   :param target: Target variable name
   :param input_parameters: Parameters to include
   :param target_unit: Unit string for target
   :param file_type: Analysis type for title
   :param output_path: Output HTML path

PDF Plots
~~~~~~~~~

.. py:function:: generate_pdf_plot(dataframes_dict, var, label_list, filters, output_path)

   Generate probability density function comparison plot.
   
   Creates log-scale PDF plots for comparing distributions across multiple datasets. Uses 10,000 histogram bins for smooth curves.
   
   :param dataframes_dict: {filename: {var: array}} nested dictionary
   :param var: Variable name to plot
   :param label_list: Labels for each dataset
   :param filters: Optional min/max filters
   :param output_path: Output PNG path

Configuration
-------------

YAML Configuration File
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   files:
     - outputs/run_001/results.h5
     - outputs/run_002/results.h5
   
   target_variables:
     - t_startup
     - unrealized_gains
   
   input_filters:
     V_plasma:
       min: 100
       max: 200
     n_tot:
       min: 1.0e14
       max: 5.0e14
   
   output_filters:
     t_startup:
       min: 0
       max: 1.0e8
     unrealized_gains:
       min: 0
       max: 1.0e10
   
   plot_types:
     - kde
     - parcoords
     - pdf
   
   output_dir: plots

Command-Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   usage: python -m ddstartup.postprocessing [CONFIG_FILE] [OPTIONS]
   
   positional arguments:
     config_file           YAML configuration file (optional)
   
   optional arguments:
     --files FILE [FILE ...]
                          HDF5 files to process
     --targets VAR [VAR ...]
                          Target variables (default: t_startup unrealized_gains)
     --input-filter EXPR  Input filters: "var<max,var>min"
     --output-filter EXPR Output filters: "var<max,var>min"
     --plots TYPE [TYPE ...]
                          Plot types: kde, parcoords, pdf (default: all)
     --output-dir DIR     Output directory (default: same as input)

Filter Syntax
~~~~~~~~~~~~~

Filter expressions use comparison operators:

* ``variable<value`` - Maximum threshold
* ``variable>value`` - Minimum threshold  
* ``variable<=value`` - Maximum threshold (inclusive)
* ``variable>=value`` - Minimum threshold (inclusive)
* ``variable==value`` - Exact value

Multiple filters separated by commas: ``"V_plasma<150,n_tot>1e14"``

Output Files
------------

The module generates:

* **KDE plots** (PNG): ``kde_{target}_{analysis_type}.png``
* **Parallel coordinates** (HTML): ``parcoords_{target}_{analysis_type}.html``
* **PDF plots** (PNG): ``pdf_{variable}_{analysis_type}.png``
* **Quartile data** (CSV): ``quartile_{target}_values_{plot_name}.csv``

CSV files contain lowest and middle values for each quartile with corresponding input parameters.

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   # Process latest results with all plot types
   python -m ddstartup.postprocessing
   
   # Generate only KDE plots
   python -m ddstartup.postprocessing --plots kde

Advanced Filtering
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Filter input parameters and outputs
   python -m ddstartup.postprocessing \
       --input-filter "V_plasma<150,n_tot>1e14,T_i>=50" \
       --output-filter "unrealized_gains<5e9"

Multiple Files
~~~~~~~~~~~~~~

.. code-block:: bash

   # Compare results from multiple runs
   python -m ddstartup.postprocessing \
       --files outputs/baseline.h5 outputs/optimized.h5 \
       --targets unrealized_gains \
       --plots pdf

Python API
~~~~~~~~~~

.. code-block:: python

   from ddstartup.postprocessing import postprocess_functions as ppf
   from pathlib import Path
   
   # Load data
   h5_path = Path("outputs/results.h5")
   df = ppf.load_h5_to_dataframe(h5_path)
   
   # Apply filters
   filters = ppf.parse_filter_expression("V_plasma<150")
   df_filtered = ppf.apply_filters(df, filters, {}, "t_startup")
   
   # Scale target
   df_scaled, unit = ppf.scale_target(df_filtered, "t_startup")
   
   print(f"Filtered: {len(df_filtered)} rows, Unit: {unit}")

Dependencies
------------

* **Required**: numpy, pandas, h5py, matplotlib, pathlib
* **Optional**: pyyaml (YAML configs), seaborn (KDE plots), plotly (parallel coords)

See Also
--------

* :doc:`../api_reference/index` - Full API reference
* :doc:`../examples/index` - Usage examples
* ``POSTPROCESS_QUICKSTART.md`` - Quick start guide (project root)
