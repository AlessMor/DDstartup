Building the Documentation
==========================

This guide explains how to build the Sphinx documentation for the DD Startup Analysis Tool.

Prerequisites
-------------

Install Sphinx and dependencies:

.. code-block:: bash

   cd dd_startup/docs
   pip install -r requirements.txt

Building HTML Documentation
---------------------------

Basic Build
~~~~~~~~~~~

.. code-block:: bash

   cd dd_startup/docs
   make html

The generated HTML will be in ``_build/html/``.

View Documentation
~~~~~~~~~~~~~~~~~~

Open in browser:

.. code-block:: bash

   # Linux/Mac
   open _build/html/index.html
   
   # Or use Python's HTTP server
   cd _build/html
   python -m http.server 8000
   # Then visit http://localhost:8000

Clean Build
~~~~~~~~~~~

Remove all build artifacts and rebuild:

.. code-block:: bash

   make clean
   make html

Building Other Formats
----------------------

PDF Documentation
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   make latexpdf

The PDF will be in ``_build/latex/ddstartupanalysistool.pdf``.

EPUB
~~~~

.. code-block:: bash

   make epub

The EPUB will be in ``_build/epub/``.

Man Pages
~~~~~~~~~

.. code-block:: bash

   make man

The man pages will be in ``_build/man/``.

Available Make Targets
----------------------

.. code-block:: text

   html        - Build HTML documentation
   latex       - Build LaTeX source
   latexpdf    - Build PDF via LaTeX
   epub        - Build EPUB
   man         - Build man pages
   text        - Build plain text
   json        - Build JSON
   clean       - Remove build artifacts
   help        - Show all targets

Continuous Build
----------------

Auto-rebuild on file changes:

.. code-block:: bash

   pip install sphinx-autobuild
   sphinx-autobuild . _build/html

Visit http://localhost:8000 - the page will auto-refresh on changes.

Documentation Structure
-----------------------

.. code-block:: text

   docs/
   ├── conf.py                      # Sphinx configuration
   ├── index.rst                    # Main index
   ├── getting_started.rst          # Installation guide
   ├── requirements.txt             # Sphinx dependencies
   ├── user_guide/                  # User documentation
   │   ├── index.rst
   │   ├── command_line_interface.rst
   │   ├── configuration_files.rst
   │   ├── parameter_definitions.rst
   │   ├── analysis_methods.rst
   │   ├── output_files.rst
   │   └── troubleshooting.rst
   ├── api_reference/               # API documentation
   │   ├── index.rst
   │   ├── io_functions.rst
   │   ├── system_profiler.rst
   │   ├── custom_classes.rst
   │   ├── units_and_constants.rst
   │   └── tools.rst
   ├── developer_guide/             # Developer docs
   │   ├── index.rst
   │   ├── architecture.rst
   │   ├── testing.rst
   │   ├── contributing.rst
   │   └── refactoring_history.rst
   ├── examples/                    # Examples/tutorials
   │   ├── index.rst
   │   ├── basic_usage.rst
   │   ├── parametric_analysis.rst
   │   ├── sobol_analysis.rst
   │   ├── batch_processing.rst
   │   └── custom_parameters.rst
   └── _build/                      # Generated documentation
       ├── html/
       ├── latex/
       └── ...

Writing Documentation
---------------------

reStructuredText Basics
~~~~~~~~~~~~~~~~~~~~~~~~

**Headings:**

.. code-block:: rst

   =========
   Chapter 1
   =========
   
   Section 1.1
   ===========
   
   Subsection 1.1.1
   ----------------
   
   Subsubsection 1.1.1.1
   ~~~~~~~~~~~~~~~~~~~~~

**Code blocks:**

.. code-block:: rst

   .. code-block:: python
   
      def hello():
          print("Hello, world!")

**Cross-references:**

.. code-block:: rst

   See :doc:`getting_started` for installation.
   See :func:`get_system_info` for details.
   See :class:`ParameterField` for the class.

**Links:**

.. code-block:: rst

   `Python <https://www.python.org>`_
   https://www.python.org

**Lists:**

.. code-block:: rst

   * Item 1
   * Item 2
     
     * Subitem 2.1
     * Subitem 2.2
   
   1. Numbered item 1
   2. Numbered item 2

**Admonitions:**

.. code-block:: rst

   .. note::
      This is a note.
   
   .. warning::
      This is a warning.
   
   .. seealso::
      Related information.

Autodoc
~~~~~~~

Document Python modules automatically:

.. code-block:: rst

   .. automodule:: ddstartup.utils.system_profiler
      :members:
      :undoc-members:
      :show-inheritance:

Document specific functions:

.. code-block:: rst

   .. autofunction:: ddstartup.utils.system_profiler.get_system_info

Document classes:

.. code-block:: rst

   .. autoclass:: ddstartup.utils.custom_classes.ParameterField
      :members:
      :special-members: __init__

Best Practices
--------------

1. **Build regularly**: Test documentation builds frequently
2. **Check warnings**: Fix all Sphinx warnings
3. **Cross-reference**: Use ``:doc:``, ``:func:``, ``:class:`` for internal links
4. **Code examples**: Include working code examples
5. **Keep updated**: Update docs when code changes
6. **Review output**: Check rendered HTML/PDF output
7. **Use autodoc**: Auto-generate API docs from docstrings

Publishing Documentation
------------------------

GitHub Pages
~~~~~~~~~~~~

.. code-block:: bash

   # Build docs
   make html
   
   # Copy to gh-pages branch
   git checkout gh-pages
   cp -r _build/html/* .
   git add .
   git commit -m "Update documentation"
   git push origin gh-pages

Read the Docs
~~~~~~~~~~~~~

1. Create account at https://readthedocs.org
2. Import GitHub repository
3. Configure to build from ``docs/`` directory
4. Documentation will auto-build on commits

Local Documentation Server
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd _build/html
   python -m http.server 8080
   # Access at http://localhost:8080

Troubleshooting
---------------

Missing Modules
~~~~~~~~~~~~~~~

If autodoc can't import modules:

.. code-block:: python

   # In conf.py, add to sys.path
   import os
   import sys
   sys.path.insert(0, os.path.abspath('..'))

Missing Dependencies
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt
   pip install -r ../requirements.txt  # Project requirements

Build Warnings
~~~~~~~~~~~~~~

Fix all warnings for clean builds:

.. code-block:: bash

   make clean html 2>&1 | grep WARNING

Cross-reference Issues
~~~~~~~~~~~~~~~~~~~~~~

Use correct syntax:

.. code-block:: rst

   :doc:`getting_started`          # Document
   :func:`get_system_info`         # Function
   :class:`ParameterField`         # Class
   :meth:`ParameterField.__init__` # Method
   :mod:`ddstartup.utils`          # Module
