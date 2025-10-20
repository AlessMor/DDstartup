.. _testing-guide:

Comprehensive Testing Guide
============================

This guide provides detailed information for using the manual testing notebooks.

Tips for Effective Testing
---------------------------

Start Simple
~~~~~~~~~~~~

* Use auto-detect for latest output
* Test with default parameters first
* Gradually add complexity

Understand Your Data
~~~~~~~~~~~~~~~~~~~~

* Check data shape and columns
* Review target statistics (min, max, distribution)
* Identify constant vs varying parameters

Iterative Testing
~~~~~~~~~~~~~~~~~

* Run once with defaults
* Adjust parameters based on results
* Compare different configurations

Interpretation
~~~~~~~~~~~~~~

* Read the summary section in each notebook
* Understand what the plots show
* Use interpretation guides for Cohen's d, importance scores, etc.

Save Your Work
~~~~~~~~~~~~~~

* Plots are auto-saved to disk
* CSV files contain numerical results
* HTML files for interactive plots

Advanced Features
-----------------

Multi-Target Analysis
~~~~~~~~~~~~~~~~~~~~~

Most notebooks support analyzing multiple targets:

.. code-block:: python

   targets = ['Q_fusion', 'tau_E', 'Ti_0']
   for target in targets:
       # Run analysis for each target
       plot_effect_size_matrix(df, target, inputs, outputs_dir)

Custom Parameter Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Focus on specific input parameters:

.. code-block:: python

   selected_inputs = ['Ti_0', 'n_0', 'B_0']
   # Use selected_inputs instead of all inputs

Parameter Tuning
~~~~~~~~~~~~~~~~

Experiment with different settings:

**Contours:**

* ``interpolate=True/False``
* ``max_pairs`` - Limit number of pairwise plots

**Importance Matrix:**

* ``quartiles=4`` (or 3, 5, etc.)

**SHAP Plots:**

* ``max_display=20`` - Top N features to show
* ``max_samples=2000`` - Sample limit for performance

**K-Means:**

* ``n_clusters=5`` (or use elbow method to determine)

Interpretation Guides
---------------------

Cohen's d (Importance Matrix)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Cohen's d Value
     - Interpretation
   * - |d| < 0.2
     - Negligible effect
   * - 0.2 ≤ |d| < 0.5
     - Small effect
   * - 0.5 ≤ |d| < 0.8
     - Medium effect
   * - |d| ≥ 0.8
     - Large effect

**How to Read the Matrix:**

* Each row = one input parameter
* Each column = one quartile of the target
* Cell value = Cohen's d when comparing:
  
  - Samples in that quartile vs. all other samples
  - For that specific input parameter

Feature Importance (SHAP)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Based on correlation with target
* Higher = more influential parameter
* Color shows feature value (blue=low, red=high)

.. list-table:: Correlation Strength
   :header-rows: 1
   :widths: 40 60

   * - Correlation Value
     - Interpretation
   * - |r| < 0.3
     - Weak correlation
   * - 0.3 ≤ |r| < 0.7
     - Moderate correlation
   * - |r| ≥ 0.7
     - Strong correlation

K-Means Quartiles
~~~~~~~~~~~~~~~~~

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

**Key Use Cases:**

* Look for clusters with high Q4 proportion → good parameter combinations
* Avoid parameter ranges in worst clusters

Common Workflows
----------------

1. Initial Exploration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Start with SHAP to identify important parameters
   create_shap_style_beeswarm_plot(
       df, inputs, target, target_unit,
       outputs_dir, plot_name='initial_shap'
   )
   
   # Then look at top 2-3 parameters with contours
   top_params = ['Ti_0', 'n_0', 'B_0']  # Based on SHAP results
   plot_pairwise_contours(
       df, inputs=top_params, target=target,
       outputs_dir=outputs_dir
   )

2. Optimization
~~~~~~~~~~~~~~~

.. code-block:: python

   # Find optimal regions with K-means
   kmeans, crosstab = cluster_and_quartile_bar(
       df, inputs, target,
       outputs_dir, n_clusters=5
   )
   
   # Examine cluster centers CSV
   centers = pd.read_csv('cluster_centers.csv')
   best_cluster = crosstab['Q4'].idxmax()
   best_params = centers.iloc[best_cluster]

3. Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute effect sizes
   effects = compute_effect_size_matrix(df, target, inputs)
   
   # Identify parameters with large effects
   important = effects.abs().mean(axis=1).sort_values(ascending=False)
   print(f"Most important parameters: {important.head(5).index.tolist()}")

4. Multi-Target Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   targets = ['Q_fusion', 'tau_E', 'Ti_0']
   for target in targets:
       plot_effect_size_matrix(
           df, target, inputs,
           outputs_dir, plot_name=f'effects_{target}'
       )

Performance Tips
----------------

Large Datasets
~~~~~~~~~~~~~~

* Use ``max_samples`` in SHAP plots
* Filter data before plotting
* Limit ``max_pairs`` in contour plots
* Use discrete heatmaps instead of interpolation

Memory Optimization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sample data if too large
   if len(df) > 10000:
       df_sample = df.sample(n=10000, random_state=42)
   else:
       df_sample = df
   
   # Use sample for plotting
   create_shap_style_beeswarm_plot(df_sample, ...)

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process multiple files
   from pathlib import Path
   
   for h5_file in Path('outputs').glob('*/*.h5'):
       df = load_h5_to_dataframe([h5_file])
       # Run analysis
       plot_effect_size_matrix(
           df, target, inputs,
           outputs_dir=h5_file.parent / 'plots'
       )

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**No files found**

* Check outputs directory path
* Ensure HDF5 files exist
* Use absolute paths if relative paths fail

.. code-block:: python

   # Debug: List available files
   from pathlib import Path
   files = list(Path('../outputs').glob('*/*.h5'))
   print(f"Found {len(files)} files: {files}")

**Target not in dataframe**

* List available columns: ``print(df.columns)``
* Check target name spelling
* Verify data was loaded correctly

.. code-block:: python

   # Debug: Show available columns
   print(f"Available columns: {df.columns.tolist()}")

**Constant parameters**

* Some methods exclude zero-variance parameters
* Check parameter variation: ``df[param].std()``
* This is expected for single-point analyses

.. code-block:: python

   # Debug: Check for constant parameters
   for param in inputs:
       std = df[param].std()
       if std == 0:
           print(f"{param} is constant (value={df[param].iloc[0]:.2e})")

**Memory issues**

* Reduce ``max_samples`` in SHAP plots
* Filter data before plotting
* Process fewer parameter pairs

.. code-block:: python

   # Reduce memory usage
   df_filtered = apply_filters(df, {'Q_fusion': {'min': 0}})
   df_sample = df_filtered.sample(n=5000, random_state=42)

**Plotting errors**

* Ensure output directory is writable
* Check for missing dependencies
* Verify data types are correct

Save and Load Results
----------------------

.. code-block:: python

   # Save processed data
   df.to_hdf('processed_results.h5', key='data')
   
   # Save specific analysis
   effects.to_csv('importance_analysis.csv')
   kmeans_centers.to_csv('optimal_regions.csv')
   
   # Load for further analysis
   df = pd.read_hdf('processed_results.h5', key='data')
   effects = pd.read_csv('importance_analysis.csv', index_col=0)

Best Practices
--------------

1. **Start with data exploration**
   
   * Load data and check shape/columns
   * Review target statistics
   * Identify constant vs varying parameters

2. **Use appropriate plot types**
   
   * SHAP/Importance for parameter ranking
   * Contours for 2D relationships
   * K-means for optimal regions
   * Parallel coords for multi-dimensional exploration

3. **Apply filters strategically**
   
   * Remove failed cases
   * Focus on specific parameter ranges
   * Reduce dataset size if needed

4. **Iterate and refine**
   
   * Run with defaults first
   * Adjust based on results
   * Compare different configurations

5. **Document your findings**
   
   * Save plots with descriptive names
   * Export numerical results to CSV
   * Keep notes on important discoveries

See Also
--------

* :doc:`plotting_overview` - Overview of all plot types
* :doc:`quick_reference` - Quick lookup tables
* :doc:`notebooks_index` - Complete notebook index
