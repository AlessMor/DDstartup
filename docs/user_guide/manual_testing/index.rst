Manual Testing Guide
====================

This section contains comprehensive documentation for manual verification and testing of all plotting methods in the ``ddstartup/postprocessing`` module.

Overview
--------

Each notebook provides:

* Step-by-step testing of plotting functions
* Multiple test cases with different parameters
* Visual verification of outputs
* Detailed explanations and interpretation guides
* Example configurations and use cases

.. toctree::
   :maxdepth: 2
   :caption: Manual Testing Documentation

   plotting_overview
   testing_guide
   quick_reference
   notebooks_index

Quick Navigation
----------------

Testing Notebooks
~~~~~~~~~~~~~~~~~

**Postprocessing Plots (Complete Coverage)**

1. :ref:`manual-contour-plots` - 2D heatmaps and contour plots
2. :ref:`manual-importance-matrix` - Cohen's d effect size analysis
3. :ref:`manual-shap-plots` - SHAP-style beeswarm plots
4. :ref:`manual-kmeans-plots` - K-means clustering analysis
5. :ref:`manual-kde-plots` - Kernel density estimation
6. :ref:`manual-parcoords-plots` - Parallel coordinates plots
7. :ref:`manual-pdf-plots` - Probability density functions

**Physics Models**

8. ``manual_lump_verification.ipynb`` - Lump model verification
9. ``manual_tseeded_verification.ipynb`` - T-seeded model verification

Coverage Summary
----------------

.. list-table:: Module Coverage
   :header-rows: 1
   :widths: 40 20 20 20

   * - Module
     - Manual Test
     - Unit Test
     - Status
   * - ``plot_contour_functions.py``
     - ✅
     - ❌
     - Complete
   * - ``plot_importance_matrix.py``
     - ✅
     - ❌
     - Complete
   * - ``plot_shap_functions.py``
     - ✅
     - ✅
     - Complete
   * - ``plot_kmeans_functions.py``
     - ✅
     - ❌
     - Complete
   * - ``plot_kde_functions.py``
     - ✅
     - ✅
     - Complete
   * - ``plot_parcoords_functions.py``
     - ✅
     - ✅
     - Complete
   * - ``plot_pdf_functions.py``
     - ✅
     - ✅
     - Complete

**Result: 7/7 plotting modules have manual testing notebooks** ✅

Getting Started
---------------

Quick Start
~~~~~~~~~~~

**Option 1: Auto-detect Latest Output (Recommended)**

.. code-block:: python

   from pathlib import Path
   from ddstartup.postprocessing.postprocess_functions import find_latest_h5_file
   
   outputs_dir = Path('../outputs')
   latest_h5_file = find_latest_h5_file(outputs_dir)
   files_to_analyze = [latest_h5_file]

**Option 2: Specify Directory**

.. code-block:: python

   outputs_dir = Path('../outputs/20251009_131829_parametric_lump')
   files_to_analyze = sorted(outputs_dir.glob('*.h5'))

**Option 3: Specify Files**

.. code-block:: python

   files_to_analyze = [
       Path('../outputs/folder1/results.h5'),
       Path('../outputs/folder2/results.h5')
   ]

Test Output Organization
~~~~~~~~~~~~~~~~~~~~~~~~

Running notebooks creates organized output directories:

.. code-block:: text

   outputs/
   ├── manual_test_contour/       # Contour plot tests
   ├── manual_test_importance/     # Importance matrix tests
   ├── manual_test_shap/          # SHAP-style plot tests
   ├── manual_test_kmeans/        # K-means clustering tests
   └── ...

Common Configuration
~~~~~~~~~~~~~~~~~~~~

All notebooks follow a similar structure:

1. **Imports** - Load required libraries and functions
2. **Configuration** - Set input files and target variable
3. **Data Loading** - Load and inspect HDF5 data
4. **Filtering** (Optional) - Apply data filters
5. **Testing** - Multiple test cases with variations
6. **Summary** - Interpretation guide and best practices

Typical Workflow
----------------

1. **Choose a notebook** based on the visualization you want to test
2. **Configure the data source** (auto-detect or specify directory/files)
3. **Set the target variable** to analyze
4. **Run all cells** to execute tests
5. **Review outputs** - plots are displayed inline and saved to disk
6. **Adjust parameters** - modify n_clusters, max_display, etc. and re-run

Target Variables
~~~~~~~~~~~~~~~~

Common target variables to analyze:

* ``Q_fusion`` - Fusion power output
* ``tau_E`` - Energy confinement time
* ``Ti_0`` - Central ion temperature
* ``n_0`` - Central density
* (Any output column from your simulation)

Data Filtering Examples
~~~~~~~~~~~~~~~~~~~~~~~

Apply filters to focus on specific regions:

.. code-block:: python

   filters = {
       'Q_fusion': {'min': 0},                    # Only positive fusion power
       'Ti_0': {'min': 5e3, 'max': 20e3},        # Temperature range
       'n_0': {'min': 1e19},                      # Minimum density
   }
   
   df_filtered = apply_filters(df, filters)

See Also
--------

* :doc:`/user_guide/postprocessing_workflow` - Main postprocessing guide
* :doc:`/api_reference/postprocessing` - API reference
* :doc:`/examples/index` - Example notebooks
