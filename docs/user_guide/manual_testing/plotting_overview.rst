.. _plotting-overview:

Plotting Methods Overview
==========================

This page provides a comprehensive overview of all plotting methods available in the ``ddstartup/postprocessing`` module.

Plot Types Summary
------------------

.. list-table:: Available Plot Types
   :header-rows: 1
   :widths: 15 25 25 35

   * - Plot Type
     - Notebook
     - Main Functions
     - Best For
   * - Contour/Heatmap
     - ``manual_contour_plots_verification.ipynb``
     - | ``plot_2d_cell_mean_heatmap()``
       | ``plot_pairwise_contours()``
       | ``plot_interactive_pairwise_contours()``
     - 2D parameter relationships, spatial patterns
   * - Importance Matrix
     - ``manual_importance_matrix_verification.ipynb``
     - | ``compute_effect_size_matrix()``
       | ``plot_effect_size_matrix()``
     - Parameter influence analysis, Cohen's d effect sizes
   * - SHAP-Style
     - ``manual_shap_plots_verification.ipynb``
     - | ``create_shap_style_beeswarm_plot()``
       | ``compute_feature_importance()``
     - Feature importance, correlation analysis
   * - K-Means Clustering
     - ``manual_kmeans_plots_verification.ipynb``
     - ``cluster_and_quartile_bar()``
     - Parameter space clustering, optimal regions
   * - KDE
     - ``manual_kde_plots_verification.ipynb``
     - ``kde_quartile_plot()``
     - Distribution analysis by quartiles
   * - Parallel Coordinates
     - ``manual_parcoords_plots_verification.ipynb``
     - Parallel coords (Plotly)
     - Multi-dimensional visualization
   * - PDF
     - ``manual_pdf_plots_verification.ipynb``
     - PDF plots
     - Probability density functions

Use Case Decision Tree
----------------------

**I want to understand which parameters matter most**
   → Use **SHAP-Style** or **Importance Matrix**
   
   - SHAP: Quick correlation-based importance
   - Importance Matrix: Cohen's d across quartiles

**I want to see relationships between two parameters**
   → Use **Contour/Heatmap**
   
   - Interpolated for smooth surfaces
   - Discrete for sparse data
   - Interactive for exploration

**I want to find optimal parameter combinations**
   → Use **K-Means Clustering**
   
   - Identifies parameter regions
   - Shows quartile distribution
   - Exports cluster centers

**I want to compare parameter distributions**
   → Use **KDE** or **PDF**
   
   - KDE: By target quartiles
   - PDF: Full distribution

**I want to explore high-dimensional data**
   → Use **Parallel Coordinates**
   
   - Interactive filtering
   - Pattern identification
   - Color by target value

Detailed Plot Types
-------------------

.. _manual-contour-plots:

Contour Plots
~~~~~~~~~~~~~

**Source Module:** ``plot_contour_functions.py``

Functions tested:

* ``plot_2d_cell_mean_heatmap()`` - 2D heatmaps with/without interpolation
* ``plot_pairwise_contours()`` - All pairwise combinations
* ``plot_interactive_pairwise_contours()`` - Interactive Plotly version

**Use Cases:**

* Visualizing relationships between two input parameters and target
* Creating smooth contour plots for dense grids
* Generating interactive HTML plots for exploration

**Key Parameters:**

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

.. _manual-importance-matrix:

Importance Matrix
~~~~~~~~~~~~~~~~~

**Source Module:** ``plot_importance_matrix.py``

Functions tested:

* ``cohen_d()`` - Effect size calculation
* ``compute_effect_size_matrix()`` - Matrix computation
* ``plot_effect_size_matrix()`` - Heatmap visualization

**Use Cases:**

* Identifying which inputs most influence the target
* Understanding parameter importance at different output ranges
* Cohen's d effect size analysis across quartiles

**Key Parameters:**

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

**Interpretation:**

.. list-table:: Cohen's d Effect Size
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Interpretation
   * - |d| < 0.2
     - Negligible effect
   * - 0.2 ≤ |d| < 0.5
     - Small effect
   * - 0.5 ≤ |d| < 0.8
     - Medium effect
   * - |d| ≥ 0.8
     - Large effect

.. _manual-shap-plots:

SHAP-Style Plots
~~~~~~~~~~~~~~~~

**Source Module:** ``plot_shap_functions.py``

Functions tested:

* ``compute_feature_importance()`` - Correlation-based importance
* ``normalize_to_range()`` - Value normalization
* ``create_shap_style_beeswarm_plot()`` - Beeswarm visualization

**Use Cases:**

* Feature importance visualization without ML models
* Understanding which parameters affect outputs and how
* Identifying parameter value ranges that increase/decrease target

**Key Parameters:**

.. code-block:: python

   create_shap_style_beeswarm_plot(
       df, input_parameters, target, target_unit,
       outputs_dir,
       plot_name,
       max_display=20,    # Top N features
       max_samples=2000   # Sample limit for speed
   )

**Reading the Plot:**

1. Top features = most important (highest |correlation|)
2. Spread width = impact magnitude
3. Color = feature value (blue→red = low→high)
4. Horizontal position = correlation × normalized value

.. _manual-kmeans-plots:

K-Means Clustering
~~~~~~~~~~~~~~~~~~

**Source Module:** ``plot_kmeans_functions.py``

Functions tested:

* ``cluster_and_quartile_bar()`` - Clustering and quartile distribution

**Use Cases:**

* Grouping similar parameter combinations
* Identifying optimal parameter regions
* Understanding quartile distribution across clusters
* Elbow method for optimal k selection

**Key Parameters:**

.. code-block:: python

   cluster_and_quartile_bar(
       df, inputs, target,
       outputs_dir,
       n_clusters=5,      # Number of clusters
       plot_name=None,
       save_csv=True      # Save cluster centers
   )

**Interpretation:**

* Each cluster represents a distinct region in parameter space
* Bar height shows proportion of samples in each quartile
* Clusters with high Q4 proportion → good parameter combinations
* Clusters with high Q1 proportion → poor parameter combinations

.. _manual-kde-plots:

KDE Plots
~~~~~~~~~

**Source Module:** ``plot_kde_functions.py``

Functions tested:

* ``kde_quartile_plot()`` - Kernel density estimation by quartiles

**Use Cases:**

* Visualizing parameter distributions across target quartiles
* Understanding how input ranges differ for high vs low outputs

.. _manual-parcoords-plots:

Parallel Coordinates
~~~~~~~~~~~~~~~~~~~~

**Source Module:** ``plot_parcoords_functions.py``

Functions tested:

* Parallel coordinates plots (interactive Plotly)

**Use Cases:**

* Multi-dimensional parameter space visualization
* Identifying parameter patterns for high-performing cases
* Interactive filtering and exploration

.. _manual-pdf-plots:

PDF Plots
~~~~~~~~~

**Source Module:** ``plot_pdf_functions.py``

Functions tested:

* PDF (Probability Density Function) plots

**Use Cases:**

* Distribution analysis of parameters and outputs
* Statistical characterization of results

Output Files
------------

.. list-table:: Output File Types
   :header-rows: 1
   :widths: 20 10 10 10 50

   * - Plot Type
     - PNG
     - CSV
     - HTML
     - Content
   * - Contour
     - ✅
     - ❌
     - ✅*
     - Heatmaps, contours (*interactive only)
   * - Importance
     - ✅
     - ✅
     - ❌
     - Heatmap + effect size matrix
   * - SHAP
     - ✅
     - ❌
     - ❌
     - Beeswarm plot
   * - K-Means
     - ✅
     - ✅
     - ❌
     - Bar chart + cluster centers
   * - KDE
     - ✅
     - ❌
     - ❌
     - Density plots
   * - Parcoords
     - ❌
     - ❌
     - ✅
     - Interactive parallel coordinates
   * - PDF
     - ✅
     - ❌
     - ❌
     - Probability density

See Also
--------

* :doc:`testing_guide` - Detailed testing procedures
* :doc:`quick_reference` - Quick lookup and examples
* :doc:`notebooks_index` - Complete notebook index
