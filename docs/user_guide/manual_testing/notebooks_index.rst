.. _notebooks-index:

Notebooks Index
===============

This page provides a complete index of all manual testing notebooks with links and descriptions.

Complete Notebook List
----------------------

Postprocessing Plot Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Contour Plots Verification**

   :Location: ``tests/manual_contour_plots_verification.ipynb``
   :Source Module: ``plot_contour_functions.py``
   
   **Features:**
   
   * 2D heatmaps with/without interpolation
   * Pairwise parameter combinations
   * Interactive Plotly visualizations
   * Custom pair selection
   
   **Test Cases:**
   
   * Single 2D heatmap (interpolated)
   * Single 2D heatmap (discrete)
   * Pairwise contours (all combinations)
   * Interactive pairwise contours (Plotly)
   * Custom pair selection

2. **Importance Matrix Verification**

   :Location: ``tests/manual_importance_matrix_verification.ipynb``
   :Source Module: ``plot_importance_matrix.py``
   
   **Features:**
   
   * Cohen's d effect size calculation
   * Effect size matrices (customizable quartiles)
   * Heatmap visualizations
   * Multi-target comparison
   
   **Test Cases:**
   
   * Cohen's d for sample comparison
   * Effect size matrix (4 quartiles)
   * Effect size matrix heatmap
   * Custom number of quartiles
   * Subset of input parameters
   * Multiple targets analysis

3. **SHAP-Style Plots Verification**

   :Location: ``tests/manual_shap_plots_verification.ipynb``
   :Source Module: ``plot_shap_functions.py``
   
   **Features:**
   
   * Correlation-based feature importance
   * SHAP-style beeswarm plots
   * No ML model required
   * Constant parameter detection
   
   **Test Cases:**
   
   * Feature importance computation
   * Constant parameter detection
   * SHAP-style beeswarm plot (full)
   * SHAP plot with limited display
   * SHAP plot with sample limiting
   * Multiple targets analysis
   * Manual correlation verification

4. **K-Means Clustering Verification**

   :Location: ``tests/manual_kmeans_plots_verification.ipynb``
   :Source Module: ``plot_kmeans_functions.py``
   
   **Features:**
   
   * K-means clustering of parameter space
   * Quartile distribution analysis
   * Cluster center extraction
   * Elbow method for optimal k
   * PCA visualization
   
   **Test Cases:**
   
   * Basic K-means clustering (5 clusters)
   * Cluster centers analysis
   * Different numbers of clusters
   * Elbow method for optimal k
   * Cluster analysis with target statistics
   * PCA 2D visualization
   * Multiple targets comparison

5. **KDE Plots Verification**

   :Location: ``tests/manual_kde_plots_verification.ipynb``
   :Source Module: ``plot_kde_functions.py``
   
   **Features:**
   
   * Kernel density estimation
   * Quartile-based distribution plots
   * Parameter distribution analysis
   
   **Use Cases:**
   
   * Visualizing parameter distributions across target quartiles
   * Understanding input ranges for high vs low outputs

6. **Parallel Coordinates Verification**

   :Location: ``tests/manual_parcoords_plots_verification.ipynb``
   :Source Module: ``plot_parcoords_functions.py``
   
   **Features:**
   
   * Interactive Plotly parallel coordinates
   * Multi-dimensional visualization
   * Pattern identification
   * Interactive filtering
   
   **Use Cases:**
   
   * Multi-dimensional parameter space exploration
   * Identifying patterns in high-performing cases

7. **PDF Plots Verification**

   :Location: ``tests/manual_pdf_plots_verification.ipynb``
   :Source Module: ``plot_pdf_functions.py``
   
   **Features:**
   
   * Probability density function plots
   * Statistical distribution analysis
   
   **Use Cases:**
   
   * Distribution analysis of parameters and outputs
   * Statistical characterization

Physics Model Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~

8. **Lump Model Verification**

   :Location: ``tests/manual_lump_verification.ipynb``
   :Source Module: ``physics/lump_functions.py``
   
   **Purpose:**
   
   * Verification of lump model physics
   * Parameter sensitivity testing
   * Model validation

9. **T-Seeded Model Verification**

   :Location: ``tests/manual_tseeded_verification.ipynb``
   :Source Module: ``physics/Tseeded_functions.py``
   
   **Purpose:**
   
   * Verification of temperature-seeded model
   * ODE integration testing
   * Model validation

Notebook Structure
------------------

All notebooks follow a consistent structure:

1. **Title and Introduction**
   
   * Clear description of what is being tested
   * Links to source modules

2. **Imports and Setup**
   
   * Required library imports
   * Path configuration
   * Success confirmation

3. **Configuration Section**
   
   * Data source selection (auto-detect or manual)
   * Target variable selection
   * Configurable parameters

4. **Data Loading**
   
   * HDF5 file loading
   * Data inspection
   * Input parameter extraction
   * Target statistics

5. **Data Filtering (Optional)**
   
   * Filter configuration
   * Filter application
   * Shape verification

6. **Test Cases (Multiple)**
   
   * Test 1: Basic functionality
   * Test 2: Parameter variations
   * Test 3: Edge cases
   * Test 4+: Advanced features
   * Each with inline outputs and explanations

7. **Summary Section**
   
   * Functions tested checklist
   * Key features summary
   * Interpretation guide
   * Use cases
   * Output file locations

Quick Access by Use Case
-------------------------

Find Optimal Parameters
~~~~~~~~~~~~~~~~~~~~~~~

* :ref:`manual-kmeans-plots` - Cluster analysis
* :ref:`manual-importance-matrix` - Effect size ranking

Understand Parameter Influence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :ref:`manual-shap-plots` - Feature importance
* :ref:`manual-importance-matrix` - Cohen's d analysis

Visualize Relationships
~~~~~~~~~~~~~~~~~~~~~~~

* :ref:`manual-contour-plots` - 2D relationships
* :ref:`manual-parcoords-plots` - Multi-dimensional

Analyze Distributions
~~~~~~~~~~~~~~~~~~~~~

* :ref:`manual-kde-plots` - Quartile distributions
* :ref:`manual-pdf-plots` - Probability densities

How to Use Notebooks
---------------------

1. **Navigate to tests directory:**

   .. code-block:: bash
   
      cd tests/

2. **Open with Jupyter:**

   .. code-block:: bash
   
      jupyter notebook manual_contour_plots_verification.ipynb

3. **Or use JupyterLab:**

   .. code-block:: bash
   
      jupyter lab

4. **Run all cells:**
   
   * Click "Cell" → "Run All"
   * Or use keyboard shortcut: Shift+Enter for each cell

5. **Review outputs:**
   
   * Inline plots and results
   * Saved files in ``outputs/manual_test_*/``

Notebook Dependencies
---------------------

Required Packages
~~~~~~~~~~~~~~~~~

.. code-block:: text

   # Core
   numpy
   pandas
   matplotlib
   seaborn
   
   # Plotting
   plotly
   
   # Analysis
   scikit-learn
   scipy
   
   # I/O
   h5py
   tables

Install all with:

.. code-block:: bash

   pip install -r requirements.txt

Data Requirements
~~~~~~~~~~~~~~~~~

* HDF5 files with simulation results (``*.h5``)
* Located in ``outputs/`` directory
* Containing:
  
  * Input parameters (e.g., ``Ti_0``, ``n_0``, ``B_0``)
  * Output variables (e.g., ``Q_fusion``, ``tau_E``)

Output Structure
----------------

Running notebooks creates organized test outputs:

.. code-block:: text

   outputs/
   ├── manual_test_contour/
   │   ├── heatmap_Ti_0_vs_n_0.png
   │   ├── pairwise_contours_Q_fusion.png
   │   └── interactive_pairwise_Q_fusion.html
   │
   ├── manual_test_importance/
   │   ├── importance_matrix_Q_fusion.png
   │   └── importance_matrix_Q_fusion_effects.csv
   │
   ├── manual_test_shap/
   │   └── shap_beeswarm_Q_fusion.png
   │
   ├── manual_test_kmeans/
   │   ├── kmeans_Q_fusion_k5.png
   │   ├── kmeans_Q_fusion_k5_cluster_centers.csv
   │   ├── elbow_method_Q_fusion.png
   │   └── cluster_pca_2d_Q_fusion.png
   │
   └── (other test directories...)

Tips for Using Notebooks
-------------------------

For Beginners
~~~~~~~~~~~~~

1. Start with simpler notebooks: PDF → KDE → Contour
2. Use auto-detect for data loading
3. Run with default parameters first
4. Read summary sections carefully

For Advanced Users
~~~~~~~~~~~~~~~~~~

1. Start with SHAP or Importance for quick insights
2. Customize parameters based on needs
3. Run multi-target analyses
4. Export results for further processing

For Developers
~~~~~~~~~~~~~~

1. Review source modules referenced in notebooks
2. Check unit tests in ``test_plot_*.py``
3. Compare manual vs automated test results
4. Contribute new test cases

Troubleshooting
---------------

See :doc:`testing_guide` for detailed troubleshooting, including:

* File not found errors
* Missing target columns
* Constant parameter handling
* Memory optimization
* Plotting errors

See Also
--------

* :doc:`index` - Manual testing overview
* :doc:`plotting_overview` - Plot types reference
* :doc:`testing_guide` - Comprehensive guide
* :doc:`quick_reference` - Code snippets
