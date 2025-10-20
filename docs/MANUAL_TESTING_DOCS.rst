Manual Testing Documentation Summary
=====================================

This document provides an overview of the manual testing documentation that has been integrated into the Sphinx documentation system.

Documentation Structure
-----------------------

The manual testing documentation is now located in:

.. code-block:: text

   docs/user_guide/manual_testing/
   ├── index.rst              # Main entry point
   ├── plotting_overview.rst  # Overview of all plot types
   ├── testing_guide.rst      # Comprehensive testing guide
   ├── quick_reference.rst    # Quick lookup and code snippets
   └── notebooks_index.rst    # Complete notebook index

Notebook Files
--------------

The actual Jupyter notebooks remain in the ``tests/`` directory:

.. code-block:: text

   tests/
   ├── manual_contour_plots_verification.ipynb
   ├── manual_importance_matrix_verification.ipynb
   ├── manual_shap_plots_verification.ipynb
   ├── manual_kmeans_plots_verification.ipynb
   ├── manual_kde_plots_verification.ipynb
   ├── manual_parcoords_plots_verification.ipynb
   ├── manual_pdf_plots_verification.ipynb
   ├── manual_lump_verification.ipynb
   └── manual_tseeded_verification.ipynb

Documentation Pages
-------------------

Main Index (index.rst)
~~~~~~~~~~~~~~~~~~~~~~

* Entry point for manual testing documentation
* Quick navigation to all sections
* Coverage matrix showing test status
* Getting started guide
* Common configuration examples

Plotting Overview (plotting_overview.rst)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Summary table of all plot types
* Use case decision tree
* Detailed descriptions of each plot type
* Parameter reference
* Interpretation guides
* Output file types

Testing Guide (testing_guide.rst)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Tips for effective testing
* Advanced features and workflows
* Interpretation guides (Cohen's d, correlation, quartiles)
* Common workflows (exploration, optimization, sensitivity)
* Performance tips
* Troubleshooting section
* Best practices

Quick Reference (quick_reference.rst)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Key parameters for all plot types
* Interpretation tables
* Common code snippets
* Workflow templates
* Debugging tips
* Common targets and output locations

Notebooks Index (notebooks_index.rst)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Complete list of all notebooks
* Detailed description of each notebook
* Test cases covered
* Quick access by use case
* Notebook structure explanation
* Dependencies and requirements
* Output structure

Integration with Main Documentation
------------------------------------

The manual testing documentation is integrated into the main Sphinx documentation:

1. **Main Index** (``docs/index.rst``):
   
   * Updated features section to highlight 7 plotting methods
   * Added manual testing suite mention
   * Quick link to manual testing guide

2. **User Guide** (``docs/user_guide/index.rst``):
   
   * Added ``manual_testing/index`` to table of contents
   * Appears after postprocessing_workflow

3. **Cross-References**:
   
   * Links from postprocessing workflow to manual testing
   * Links from API reference to testing examples
   * Links from examples to relevant notebooks

Key Features
------------

Sphinx Features Used
~~~~~~~~~~~~~~~~~~~~

* **reStructuredText** (.rst) format for proper Sphinx integration
* **Cross-references** using ``:ref:`` and ``:doc:`` directives
* **Code blocks** with syntax highlighting
* **Tables** for organized information display
* **Table of contents** with ``.. toctree::`` directives
* **List tables** for better formatting control
* **Internal links** for easy navigation

Accessibility
~~~~~~~~~~~~~

* All content searchable via Sphinx search
* Proper heading hierarchy for navigation
* Cross-referenced throughout documentation
* Included in generated HTML and PDF docs

Migration Notes
---------------

What Was Moved
~~~~~~~~~~~~~~

The following markdown files were converted to reStructuredText and moved to docs:

* ``tests/README_MANUAL_TESTS.md`` → ``docs/user_guide/manual_testing/index.rst``
* ``tests/MANUAL_TESTING_GUIDE.md`` → ``docs/user_guide/manual_testing/testing_guide.rst``
* ``tests/PLOTTING_QUICK_REFERENCE.md`` → ``docs/user_guide/manual_testing/quick_reference.rst``

Additional files created:

* ``docs/user_guide/manual_testing/plotting_overview.rst`` (new, extracted from guide)
* ``docs/user_guide/manual_testing/notebooks_index.rst`` (new, detailed notebook index)

What Remained in Tests
~~~~~~~~~~~~~~~~~~~~~~~

The actual notebooks remain in ``tests/`` because:

1. They need to be in tests directory to properly import from parent
2. They contain executable code that references test fixtures
3. They generate outputs in ``outputs/`` relative to tests location
4. Developer workflow expects notebooks in tests directory

However, they are fully documented and referenced from the Sphinx docs.

Building Documentation
----------------------

To build the documentation with manual testing content:

.. code-block:: bash

   cd docs/
   make html

The manual testing documentation will be included in:

* HTML: ``docs/_build/html/user_guide/manual_testing/index.html``
* Under "User Guide" → "Manual Testing Guide" in navigation

Viewing Documentation
---------------------

After building:

.. code-block:: bash

   # Open in browser
   firefox _build/html/index.html
   
   # Or with Python's HTTP server
   cd _build/html
   python -m http.server 8000

Then navigate to "User Guide" → "Manual Testing Guide"

Future Maintenance
------------------

When Updating Notebooks
~~~~~~~~~~~~~~~~~~~~~~~

1. Update the notebook in ``tests/``
2. If major changes, update corresponding RST documentation
3. Ensure cross-references remain valid
4. Rebuild documentation

When Adding New Plot Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create notebook in ``tests/manual_*_verification.ipynb``
2. Add entry to ``plotting_overview.rst``
3. Add entry to ``notebooks_index.rst``
4. Update coverage table in ``index.rst``
5. Update quick reference if needed
6. Rebuild documentation

Best Practices
~~~~~~~~~~~~~~

* Keep RST docs synchronized with notebook content
* Use consistent formatting and structure
* Maintain cross-references
* Update examples when API changes
* Test all links after updates

See Also
--------

* :doc:`/user_guide/manual_testing/index` - Main manual testing documentation
* :doc:`/user_guide/postprocessing_workflow` - Postprocessing guide
* :doc:`/api_reference/postprocessing` - API reference
