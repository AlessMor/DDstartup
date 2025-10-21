DD Startup Analysis Tool Documentation
=========================================

Welcome to the DD Startup Analysis Tool documentation. This tool provides a command-line interface for running deuterium-deuterium (DD) fusion reactor startup analysis with various parameter configurations and methods.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   user_guide/index
   api_reference/index
   developer_guide/index
   examples/index

Features
--------

* **Command-line interface** with flexible argument parsing
* **YAML-based configuration** for easy parameter management
* **Multiple analysis methods**: Parametric and Sobol sensitivity analysis
* **Postprocessing module** with 7 plotting methods (contour, SHAP, K-means, KDE, parallel coordinates, PDF, importance matrix)
* **Manual testing suite** with comprehensive Jupyter notebooks for all plot types
* **System profiling** for automatic hardware detection and optimization
* **Comprehensive testing** with pytest (160+ tests) and manual verification notebooks
* **Modular architecture** with separate I/O, profiling, physics, and postprocessing modules

Quick Start
-----------

**Run Simulation**

.. code-block:: bash

   cd ddstartup
   python main.py config parametric_tseeded

**Postprocess Results**

.. code-block:: bash

   python -m ddstartup.postprocessing

See :doc:`getting_started` for detailed installation and :doc:`user_guide/postprocessing_workflow` for postprocessing guide.

**Manual Testing & Visualization**

Explore the comprehensive suite of manual testing notebooks:

* :doc:`user_guide/manual_testing/index` - Complete manual testing guide
* 7 plotting method notebooks with step-by-step examples
* Interactive visualization and analysis workflows

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
