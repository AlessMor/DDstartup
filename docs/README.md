DD Startup Analysis Documentation
===================================

Welcome to the DD Startup Analysis Tool documentation source!

This directory contains all the source files for building the comprehensive documentation using Sphinx.

Quick Start
-----------

Build HTML documentation::

    pip install -r requirements.txt
    make html
    open _build/html/index.html

Documentation Structure
-----------------------

Current files::

    docs/
    ├── conf.py                          # Sphinx configuration
    ├── Makefile                         # Build commands
    ├── requirements.txt                 # Sphinx dependencies
    ├── README.md                        # This file
    ├── README.rst                       # Build instructions (detailed)
    ├── DOCUMENTATION_SUMMARY.rst        # Complete status report
    ├── index.rst                        # Main documentation index
    ├── getting_started.rst              # Installation guide
    │
    ├── user_guide/                      # User documentation
    │   ├── index.rst
    │   ├── command_line_interface.rst   # ✅ Complete
    │   ├── configuration_files.rst      # ✅ Complete
    │   ├── parameter_definitions.rst    # ✅ Complete
    │   ├── analysis_methods.rst         # TODO
    │   ├── output_files.rst             # TODO
    │   └── troubleshooting.rst          # TODO
    │
    ├── api_reference/                   # API documentation
    │   ├── index.rst
    │   ├── system_profiler.rst          # ✅ Complete
    │   ├── io_functions.rst             # TODO
    │   ├── custom_classes.rst           # TODO
    │   ├── units_and_constants.rst      # TODO
    │   └── tools.rst                    # TODO
    │
    ├── developer_guide/                 # Developer docs
    │   ├── index.rst
    │   ├── architecture.rst             # TODO
    │   ├── testing.rst                  # TODO
    │   ├── contributing.rst             # TODO
    │   └── refactoring_history.rst      # TODO
    │
    └── examples/                        # Tutorials
        ├── index.rst
        ├── basic_usage.rst              # TODO
        ├── parametric_analysis.rst      # TODO
        ├── sobol_analysis.rst           # TODO
        ├── batch_processing.rst         # TODO
        └── custom_parameters.rst        # TODO

Status
------

**Completed:** 13 files including core structure, configuration, and key user guide sections

**Remaining:** 15 TODO files for complete documentation

**Current Progress:** ~46% complete

Building Documentation
----------------------

HTML (most common)::

    make html

PDF::

    make latexpdf

All formats::

    make html latexpdf epub man

Clean build::

    make clean html

Auto-rebuild::

    pip install sphinx-autobuild
    sphinx-autobuild . _build/html

Key Completed Sections
----------------------

1. **Getting Started** - Complete installation and setup guide
2. **CLI Documentation** - Full command-line interface reference
3. **Configuration Files** - Complete YAML configuration guide
4. **Parameter Definitions** - ParameterField usage and examples
5. **System Profiler API** - Complete API documentation with examples

Next Steps
----------

See ``DOCUMENTATION_SUMMARY.rst`` for:

- Detailed status of all documentation files
- Priority order for completing remaining sections
- Migration guide for existing markdown files
- Quality checklist

Contributing
------------

To add or update documentation:

1. Edit or create ``.rst`` files in appropriate directories
2. Update ``DOCUMENTATION_SUMMARY.rst`` status
3. Build and review: ``make html``
4. Fix any Sphinx warnings
5. Commit changes

Documentation Standards
-----------------------

- Use reStructuredText (.rst) format
- Follow existing heading hierarchy
- Include working code examples
- Add cross-references with ``:doc:``, ``:func:``, ``:class:``
- Test builds before committing
- Update summary document when complete

Requirements
------------

See ``requirements.txt`` for Sphinx dependencies::

    sphinx>=7.0.0
    sphinx_rtd_theme>=2.0.0
    sphinx-autodoc-typehints>=1.25.0

Help
----

- Build instructions: See ``README.rst``
- Complete status: See ``DOCUMENTATION_SUMMARY.rst``
- Sphinx docs: https://www.sphinx-doc.org/
- RST reference: https://docutils.sourceforge.io/rst.html

Issues
------

If you encounter build errors:

1. Check Python path in ``conf.py``
2. Ensure all dependencies installed
3. Run ``make clean html``
4. Review Sphinx warnings in output

For documentation questions or contributions, see the developer guide (when complete).
