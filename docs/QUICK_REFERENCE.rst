Sphinx Documentation Quick Reference
====================================

Build Commands
--------------

::

    cd dd_startup/docs
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Build HTML
    make html
    
    # Build PDF
    make latexpdf
    
    # Build all formats
    make html latexpdf epub
    
    # Clean build
    make clean html
    
    # Auto-rebuild on changes
    pip install sphinx-autobuild
    sphinx-autobuild . _build/html

View Documentation
------------------

::

    # Open HTML in browser
    open _build/html/index.html
    
    # Or use Python HTTP server
    cd _build/html
    python -m http.server 8000
    # Visit http://localhost:8000

File Locations
--------------

Configuration
~~~~~~~~~~~~~

- ``conf.py`` - Sphinx configuration
- ``Makefile`` - Build system
- ``requirements.txt`` - Sphinx dependencies

Main Sections
~~~~~~~~~~~~~

- ``index.rst`` - Main documentation index
- ``getting_started.rst`` - Installation guide
- ``user_guide/`` - User documentation (4 files)
- ``api_reference/`` - API docs (2 files)
- ``developer_guide/`` - Developer docs (1 file)
- ``examples/`` - Tutorials (1 file)

Meta Documentation
~~~~~~~~~~~~~~~~~~

- ``README.md`` - Quick overview
- ``README.rst`` - Detailed build guide
- ``DOCUMENTATION_SUMMARY.rst`` - Complete status
- ``SPHINX_SETUP_SUMMARY.md`` - Setup summary
- ``QUICK_REFERENCE.rst`` - This file

Status Summary
--------------

**Completed:** 13 RST files + 3 config files = 16 total

**Progress:** ~46% complete

**Ready to build:** Yes ✅

Completed Sections
~~~~~~~~~~~~~~~~~~

✅ Getting Started
✅ CLI Documentation
✅ Configuration Files Guide
✅ Parameter Definitions Guide
✅ System Profiler API Reference
✅ Postprocessing Module API Reference
✅ Postprocessing Workflow Guide
✅ All index files
✅ Build system
✅ Sphinx configuration

TODO Sections
~~~~~~~~~~~~~

Priority 1: User Guide
- Analysis methods
- Output files
- Troubleshooting

Priority 2: API Reference
- I/O functions
- Custom classes
- Units and constants
- Tools

Priority 3: Examples
- Basic usage
- Parametric analysis
- Sobol analysis
- Batch processing
- Custom parameters

Priority 4: Developer Guide
- Architecture
- Testing
- Contributing
- Refactoring history

Common Tasks
------------

Add New Documentation File
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create ``new_file.rst`` in appropriate directory
2. Add to section index (e.g., ``user_guide/index.rst``)
3. Update ``DOCUMENTATION_SUMMARY.rst``
4. Build: ``make html``
5. Review: ``open _build/html/new_file.html``

Fix Build Warnings
~~~~~~~~~~~~~~~~~~

::

    make html 2>&1 | grep WARNING

Common warnings:
- Missing cross-references
- Undefined labels
- Duplicate headings

Convert Markdown to RST
~~~~~~~~~~~~~~~~~~~~~~~

::

    pandoc -f markdown -t rst input.md -o output.rst

Then manually fix:
- Code block languages
- Cross-references
- Sphinx directives

Test Documentation
~~~~~~~~~~~~~~~~~~

::

    # Build all formats
    make clean html latexpdf epub
    
    # Check for errors
    echo $?  # Should be 0
    
    # Review output
    ls -lh _build/html/index.html
    ls -lh _build/latex/*.pdf

Useful Sphinx Roles
-------------------

Cross-References
~~~~~~~~~~~~~~~~

::

    :doc:`getting_started`              # Link to document
    :ref:`section-label`                # Link to section
    :func:`get_system_info`             # Link to function
    :class:`ParameterField`             # Link to class
    :meth:`ParameterField.__init__`     # Link to method
    :mod:`ddstartup.utils`              # Link to module

External Links
~~~~~~~~~~~~~~

::

    `Python <https://www.python.org>`_
    https://www.python.org

Code Blocks
~~~~~~~~~~~

::

    .. code-block:: python
    
       def hello():
           print("Hello!")

Admonitions
~~~~~~~~~~~

::

    .. note::
       Important information
    
    .. warning::
       Be careful!
    
    .. seealso::
       Related content

Tables
~~~~~~

::

    ========  ========
    Column 1  Column 2
    ========  ========
    Value 1   Value 2
    Value 3   Value 4
    ========  ========

Documentation Standards
-----------------------

Headings
~~~~~~~~

::

    Chapter Title (rarely used)
    ===========================
    
    Section Title
    =============
    
    Subsection Title
    ----------------
    
    Subsubsection Title
    ~~~~~~~~~~~~~~~~~~~

Line Length
~~~~~~~~~~~

Aim for 80-100 characters per line for readability.

Code Examples
~~~~~~~~~~~~~

- Always include language specifier
- Test all examples before documenting
- Show both input and output where relevant

Cross-References
~~~~~~~~~~~~~~~~

- Use proper Sphinx roles (``:doc:``, ``:func:``, etc.)
- Link to related sections
- Build navigation paths for users

Extensions Used
---------------

- ``sphinx.ext.autodoc`` - Auto API docs
- ``sphinx.ext.napoleon`` - Docstring parsing
- ``sphinx.ext.viewcode`` - Source links
- ``sphinx.ext.intersphinx`` - External refs
- ``sphinx.ext.todo`` - TODO directives
- ``sphinx.ext.mathjax`` - Math rendering
- ``sphinx_rtd_theme`` - RTD theme

Publishing Options
------------------

GitHub Pages
~~~~~~~~~~~~

::

    git checkout gh-pages
    cp -r _build/html/* .
    git add .
    git commit -m "Update docs"
    git push

Read the Docs
~~~~~~~~~~~~~

1. Connect GitHub repo at readthedocs.org
2. Configure to build from ``docs/`` directory
3. Auto-builds on each commit

Local Server
~~~~~~~~~~~~

::

    cd _build/html
    python -m http.server 8080

Help Resources
--------------

- Sphinx docs: https://www.sphinx-doc.org/
- RST reference: https://docutils.sourceforge.io/rst.html
- RTD theme: https://sphinx-rtd-theme.readthedocs.io/
- Build guide: ``README.rst``
- Status: ``DOCUMENTATION_SUMMARY.rst``
- Setup: ``SPHINX_SETUP_SUMMARY.md``

Troubleshooting
---------------

Build Fails
~~~~~~~~~~~

::

    # Clean and rebuild
    make clean
    make html
    
    # Check Python path
    # In conf.py: sys.path.insert(0, os.path.abspath('..'))

Missing Dependencies
~~~~~~~~~~~~~~~~~~~~

::

    pip install -r requirements.txt
    pip install -r ../requirements.txt

Import Errors
~~~~~~~~~~~~~

Ensure parent directory is in sys.path (conf.py).

No Module Warnings
~~~~~~~~~~~~~~~~~~

Check that module imports work from docs/ directory.

Contact
-------

For questions about documentation:

1. Check ``README.rst`` for build help
2. See ``DOCUMENTATION_SUMMARY.rst`` for status
3. Review ``SPHINX_SETUP_SUMMARY.md`` for setup info
4. Refer to Sphinx documentation for technical questions
