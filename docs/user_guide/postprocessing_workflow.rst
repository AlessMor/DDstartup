Postprocessing Workflow
=======================

This guide covers the postprocessing workflow for analyzing simulation outputs.

Overview
--------

After running simulations, the postprocessing module:

1. Loads HDF5 output files
2. Applies filters to focus on specific parameter ranges
3. Generates visualizations (KDE, parallel coordinates, PDF plots)
4. Exports quartile statistics to CSV

Workflow Steps
--------------

Step 1: Locate Output Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module automatically finds the latest output folder:

.. code-block:: bash

   outputs/
   ├── 2024-10-01_T_seeded/
   │   └── results.h5
   └── 2024-10-08_T_seeded/  ← Latest (auto-selected)
       └── results.h5

Or specify files explicitly:

.. code-block:: bash

   python -m ddstartup.postprocessing --files outputs/specific_run/results.h5

Step 2: Configure Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option A: YAML Configuration** (recommended for complex analyses)

Create ``inputs/my_config.yaml``:

.. code-block:: yaml

   target_variables:
     - t_startup
   
   input_filters:
     V_plasma:
       max: 150
     n_tot:
       min: 1.0e14
   
   plot_types:
     - kde
     - parcoords

Run:

.. code-block:: bash

   python -m ddstartup.postprocessing inputs/my_config.yaml

**Option B: Command-Line Arguments** (quick analyses)

.. code-block:: bash

   python -m ddstartup.postprocessing \
       --targets t_startup unrealized_gains \
       --input-filter "V_plasma<150,n_tot>1e14" \
       --plots kde parcoords

Step 3: Review Outputs
~~~~~~~~~~~~~~~~~~~~~~~

Outputs are saved in the same directory as input files (or ``--output-dir``):

* ``kde_t_startup_analysis.png`` - KDE plots by quartile
* ``parcoords_t_startup_analysis.html`` - Interactive parallel coordinates
* ``pdf_t_startup_analysis.png`` - Probability density functions
* ``quartile_t_startup_values_kde.csv`` - Quartile statistics

Visualization Types
-------------------

KDE Plots (Kernel Density Estimation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Understand how input parameters differ across performance quartiles.

**Output**: Grid of subplots showing probability distributions for each input parameter, color-coded by target variable quartiles (Q1-Q4).

**Interpretation**:
- Overlapping distributions → parameter has little effect on target
- Separated distributions → parameter strongly influences target
- Quartile CSV shows specific parameter values for best/worst performance

**Example Use Case**: "What plasma volume values lead to fastest startup times?"

Parallel Coordinates
~~~~~~~~~~~~~~~~~~~~

**Purpose**: Explore multi-dimensional relationships interactively.

**Output**: HTML file with interactive Plotly visualization showing all parameters as parallel vertical axes.

**Interpretation**:
- Lines connect parameter values for single simulation
- Color indicates target variable value (6 quantile bins)
- Hover for exact values
- Brush axes to filter data

**Example Use Case**: "Which parameter combinations achieve low startup time AND low cost?"

PDF Plots (Probability Density Function)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Compare distributions across different simulation runs or configurations.

**Output**: Log-scale plots showing probability density for a variable across multiple datasets.

**Interpretation**:
- Peak location → most likely values
- Width → variability
- Multiple curves → comparison across runs

**Example Use Case**: "How does plasma volume distribution differ between baseline and optimized designs?"

Common Workflows
----------------

Workflow 1: Initial Exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examine all results without filters:

.. code-block:: bash

   python -m ddstartup.postprocessing

Review all three plot types to identify:
- Important parameters (KDE plots)
- Parameter correlations (parallel coordinates)
- Distribution shapes (PDF plots)

Workflow 2: Focused Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on initial exploration, apply filters to focus on feasible region:

.. code-block:: bash

   python -m ddstartup.postprocessing \
       --input-filter "V_plasma>=100,V_plasma<=200,n_tot>1e14" \
       --output-filter "t_startup<1e8"

Workflow 3: Compare Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare multiple simulation runs:

.. code-block:: bash

   python -m ddstartup.postprocessing \
       --files outputs/baseline.h5 outputs/optimized.h5 \
       --plots pdf

Workflow 4: Export Best Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate KDE plots to get CSV with best parameter combinations:

.. code-block:: bash

   python -m ddstartup.postprocessing --plots kde

Review ``quartile_t_startup_values_kde.csv`` for optimal parameter sets (Q1 = best performance).

Filtering Best Practices
-------------------------

Input Filters
~~~~~~~~~~~~~

Filter input parameters to explore physically realistic or feasible design space:

.. code-block:: yaml

   input_filters:
     V_plasma:
       min: 100    # Minimum practical volume
       max: 200    # Maximum allowable volume
     T_i:
       min: 50     # Minimum fusion-relevant temperature

Output Filters
~~~~~~~~~~~~~~

Filter output variables to focus on desirable outcomes:

.. code-block:: yaml

   output_filters:
     t_startup:
       max: 3.15e7  # Max 1 year startup
     unrealized_gains:
       max: 5.0e9   # Max $5B cost

**Note**: Output filters also remove non-finite values (NaN, inf).

Filter Strategy
~~~~~~~~~~~~~~~

1. **Start broad**: Run without filters to see full distribution
2. **Identify outliers**: Note any unrealistic or undesirable regions
3. **Apply constraints**: Add filters to focus analysis
4. **Iterate**: Refine filters based on results

Troubleshooting
---------------

No Plots Generated
~~~~~~~~~~~~~~~~~~

**Symptom**: Command completes but no output files.

**Causes**:
- No data after filtering (all rows removed)
- No input parameters found
- Output directory not writable

**Solution**:

.. code-block:: bash

   # Check filtering (look for "Filtered: X → Y rows")
   python -m ddstartup.postprocessing --verbose
   
   # Try without filters
   python -m ddstartup.postprocessing --input-filter "" --output-filter ""

Empty/Sparse Plots
~~~~~~~~~~~~~~~~~~

**Symptom**: Plots generated but show little data.

**Cause**: Filters too restrictive.

**Solution**: Relax filter constraints or check data ranges in HDF5 file.

Memory Issues
~~~~~~~~~~~~~

**Symptom**: Process killed or very slow with large datasets.

**Solution**: Parallel coordinates automatically samples to 1M points. For KDE/PDF:

.. code-block:: bash

   # Pre-filter data more aggressively
   python -m ddstartup.postprocessing \
       --input-filter "V_plasma>=120,V_plasma<=180"

Module Import Errors
~~~~~~~~~~~~~~~~~~~~

**Symptom**: ``ModuleNotFoundError: No module named 'plotly'``

**Solution**: Install optional dependencies:

.. code-block:: bash

   pip install plotly        # For parallel coordinates
   pip install seaborn       # For KDE plots
   pip install pyyaml        # For YAML configs

Performance Tips
----------------

* Use **YAML configs** for complex repeated analyses
* Apply **input filters** early to reduce data volume
* For large datasets (>1M rows), **parallel coordinates** auto-samples
* Generate only needed plot types: ``--plots kde`` (faster than all)
* Use ``--verbose`` to monitor progress

Integration with Main Workflow
-------------------------------

Typical simulation + postprocessing workflow:

.. code-block:: bash

   # 1. Run simulation
   cd ddstartup
   python main.py config parametric_tseeded
   
   # 2. Postprocess latest results
   python -m ddstartup.postprocessing
   
   # 3. Review plots and quartile CSV
   
   # 4. If needed, re-run with filters
   python -m ddstartup.postprocessing \
       --input-filter "V_plasma<150" \
       --plots kde

See Also
--------

* :doc:`../api_reference/postprocessing` - Full API documentation
* :doc:`configuration_files` - YAML configuration details
* ``POSTPROCESS_QUICKSTART.md`` - Quick reference (project root)
