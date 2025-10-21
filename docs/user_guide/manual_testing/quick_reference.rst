.. _quick-reference:

Quick Reference Guide
=====================

This page provides quick lookup tables and common code snippets for manual testing.

Key Parameters Reference
------------------------

Contour Plots
~~~~~~~~~~~~~

.. code-block:: python

   plot_2d_cell_mean_heatmap(
       df, x, y, target,
       outputs_dir,
       interpolate=True,  # Smooth vs discrete
       plot_name=None
   )
   
   plot_pairwise_contours(
       df, inputs, target,
       outputs_dir,
       max_pairs=None,    # Limit number of plots
       interpolate=True
   )

Importance Matrix
~~~~~~~~~~~~~~~~~

.. code-block:: python

   plot_effect_size_matrix(
       df, target, inputs,
       outputs_dir,
       plot_name=None,
       save_csv=True      # Save effect sizes
   )
   
   compute_effect_size_matrix(
       df, target, inputs,
       quartiles=4        # Number of quartiles
   )

SHAP-Style Plots
~~~~~~~~~~~~~~~~

.. code-block:: python

   create_shap_style_beeswarm_plot(
       df, input_parameters, target, target_unit,
       outputs_dir,
       plot_name,
       max_display=20,    # Top N features
       max_samples=2000   # Sample limit for speed
   )

K-Means Clustering
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   cluster_and_quartile_bar(
       df, inputs, target,
       outputs_dir,
       n_clusters=5,      # Number of clusters
       plot_name=None,
       save_csv=True      # Save cluster centers
   )

Interpretation Tables
---------------------

Cohen's d (Effect Size)
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Value
     - Interpretation
   * - \|d\| < 0.2
     - Negligible
   * - 0.2 ≤ \|d\| < 0.5
     - Small
   * - 0.5 ≤ \|d\| < 0.8
     - Medium
   * - \|d\| ≥ 0.8
     - Large

Correlation (SHAP Importance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Value
     - Interpretation
   * - \|r\| < 0.3
     - Weak
   * - 0.3 ≤ \|r\| < 0.7
     - Moderate
   * - \|r\| ≥ 0.7
     - Strong

Quartiles
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Quartile
     - Range
     - Meaning
   * - Q1
     - 0-25%
     - Lowest values
   * - Q2
     - 25-50%
     - Below median
   * - Q3
     - 50-75%
     - Above median
   * - Q4
     - 75-100%
     - Highest values

Common Code Snippets
--------------------

Data Loading
~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from ddstartup.postprocessing.postprocess_functions import (
       load_h5_to_dataframe,
       find_latest_h5_file,
       get_input_parameters
   )
   
   # Auto-detect latest
   outputs_dir = Path('../outputs')
   latest_h5_file = find_latest_h5_file(outputs_dir)
   files_to_analyze = [latest_h5_file]
   
   # Or specify directory
   outputs_dir = Path('../outputs/20251009_131829_parametric_lump')
   files_to_analyze = sorted(outputs_dir.glob('*.h5'))
   
   # Load data
   df = load_h5_to_dataframe(files_to_analyze)
   inputs = get_input_parameters(df)

Data Filtering
~~~~~~~~~~~~~~

.. code-block:: python

   from ddstartup.postprocessing.postprocess_functions import apply_filters
   
   # Only successful cases
   filters = {'Q_fusion': {'min': 0}}
   
   # Specific parameter range
   filters = {'Ti_0': {'min': 5e3, 'max': 20e3}}
   
   # Multiple conditions
   filters = {
       'Q_fusion': {'min': 1e6},
       'n_0': {'max': 1e20},
       'tau_E': {'min': 0.1}
   }
   
   df_filtered = apply_filters(df, filters)

Multi-Target Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   targets = ['Q_fusion', 'tau_E', 'Ti_0']
   
   for target in targets:
       # SHAP analysis
       create_shap_style_beeswarm_plot(
           df, inputs, target, 
           target_unit=PARAM_UNITS.get(target, ''),
           outputs_dir=test_outputs_dir,
           plot_name=f'shap_{target}'
       )
       
       # Or importance matrix
       plot_effect_size_matrix(
           df, target, inputs,
           outputs_dir=test_outputs_dir,
           plot_name=f'effects_{target}'
       )

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sample large datasets
   if len(df) > 10000:
       df_sample = df.sample(n=10000, random_state=42)
   else:
       df_sample = df
   
   # Reduce samples in SHAP plots
   create_shap_style_beeswarm_plot(
       df, inputs, target, target_unit,
       outputs_dir, plot_name,
       max_display=10,     # Show top 10 only
       max_samples=1000    # Limit to 1000 samples
   )
   
   # Limit pairwise combinations
   plot_pairwise_contours(
       df, inputs, target,
       outputs_dir,
       max_pairs=6,        # Max 6 plots
       interpolate=False   # Faster discrete heatmap
   )

Finding Optimal Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.cluster import KMeans
   from sklearn.preprocessing import StandardScaler
   
   # Cluster analysis
   kmeans, crosstab = cluster_and_quartile_bar(
       df, inputs, target,
       outputs_dir, n_clusters=5,
       save_csv=True
   )
   
   # Find best cluster (highest Q4 proportion)
   best_cluster = crosstab['Q4'].idxmax()
   
   # Load cluster centers
   centers = pd.read_csv(outputs_dir / 'cluster_centers.csv')
   optimal_params = centers.iloc[best_cluster]
   
   print(f"Optimal parameters (Cluster {best_cluster}):")
   for param in inputs:
       print(f"  {param}: {optimal_params[param]:.3e}")

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   
   # Process all H5 files in outputs
   for h5_file in Path('outputs').glob('*/*.h5'):
       print(f"Processing {h5_file.name}...")
       
       df = load_h5_to_dataframe([h5_file])
       outputs_dir = h5_file.parent / 'plots'
       outputs_dir.mkdir(exist_ok=True)
       
       # Run your analysis
       plot_effect_size_matrix(
           df, 'Q_fusion', inputs,
           outputs_dir=outputs_dir
       )

Debugging Tips
~~~~~~~~~~~~~~

.. code-block:: python

   # Check available files
   files = list(Path('../outputs').glob('*/*.h5'))
   print(f"Found {len(files)} files")
   
   # Check dataframe structure
   print(f"Shape: {df.shape}")
   print(f"Columns: {df.columns.tolist()}")
   
   # Check target statistics
   print(f"\nTarget '{target}' stats:")
   print(df[target].describe())
   
   # Check for constant parameters
   for param in inputs:
       std = df[param].std()
       status = "CONSTANT" if std == 0 else f"varying (std={std:.2e})"
       print(f"{param}: {status}")
   
   # Verify filters
   print(f"\nBefore filter: {len(df)} rows")
   df_filtered = apply_filters(df, filters)
   print(f"After filter: {len(df_filtered)} rows")

Workflow Templates
------------------

Template 1: Quick Exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Load data
   df = load_h5_to_dataframe([latest_h5_file])
   inputs = get_input_parameters(df)
   target = 'Q_fusion'
   
   # 2. Quick importance check
   create_shap_style_beeswarm_plot(
       df, inputs, target, 'W',
       outputs_dir, 'quick_shap'
   )
   
   # 3. Look at top 3 parameters
   # (Based on SHAP results)
   top_3 = ['Ti_0', 'n_0', 'B_0']
   plot_pairwise_contours(
       df, top_3, target, outputs_dir
   )

Template 2: Detailed Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Load and filter data
   df = load_h5_to_dataframe(files_to_analyze)
   df = apply_filters(df, {'Q_fusion': {'min': 0}})
   inputs = get_input_parameters(df)
   
   # 2. Importance analysis
   effects = plot_effect_size_matrix(
       df, target, inputs, outputs_dir
   )
   
   # 3. Clustering
   kmeans, crosstab = cluster_and_quartile_bar(
       df, inputs, target, outputs_dir, n_clusters=5
   )
   
   # 4. Detailed visualization
   plot_pairwise_contours(
       df, inputs, target, outputs_dir,
       max_pairs=10, interpolate=True
   )

Template 3: Multi-Target Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   targets = ['Q_fusion', 'tau_E', 'Ti_0']
   
   for target in targets:
       print(f"\nAnalyzing {target}...")
       
       # Importance
       effects = compute_effect_size_matrix(df, target, inputs)
       top_params = effects.abs().mean(axis=1).nlargest(5)
       
       print(f"Top 5 important parameters:")
       print(top_params)
       
       # Clustering
       kmeans, _ = cluster_and_quartile_bar(
           df, inputs, target,
           outputs_dir, n_clusters=5,
           plot_name=f'kmeans_{target}'
       )

Common Targets
--------------

.. code-block:: python

   # Fusion power
   target = 'Q_fusion'
   target_unit = 'W'
   
   # Confinement time
   target = 'tau_E'
   target_unit = 's'
   
   # Central temperature
   target = 'Ti_0'
   target_unit = 'K'
   
   # Central density
   target = 'n_0'
   target_unit = 'm^-3'

Output File Locations
---------------------

.. code-block:: text

   # Manual test outputs
   outputs/
   ├── manual_test_contour/
   │   ├── heatmap_*.png
   │   ├── pairwise_*.png
   │   └── interactive_*.html
   ├── manual_test_importance/
   │   ├── importance_matrix_*.png
   │   └── *_effects.csv
   ├── manual_test_shap/
   │   └── shap_beeswarm_*.png
   └── manual_test_kmeans/
       ├── kmeans_*.png
       ├── *_cluster_centers.csv
       └── elbow_method_*.png

See Also
--------

* :doc:`plotting_overview` - Overview of all plot types
* :doc:`testing_guide` - Detailed testing procedures
* :doc:`notebooks_index` - Complete notebook index
