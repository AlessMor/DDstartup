# DD Startup Analysis - Main Commands & Directory Structure

## Directory Structure

```
root/                           # Root directory
├── README.md                         # Main project documentation
├── LICENSE                           # Project license
├── requirements.txt                  # Python dependencies
├── pytest.ini                        # Pytest configuration
├── .gitignore                        # Git ignore rules
│
├── ddstartup/                        # Main package directory
│   ├── __init__.py                   # Package initialization
│   ├── __main__.py                   # Entry point for python -m ddstartup
│   ├── main.py                       # Main execution script
│   │
│   ├── physics/                      # Physics models and calculations
│   │   ├── __init__.py
│   │   ├── lump_functions.py         # Lump method (target inventory)
│   │   ├── Tseeded_functions.py      # T-seeded method (time-dependent ODE)
│   │   ├── reactionrates_functions.py # Fusion reaction rate calculations
│   │   └── sobol_functions.py        # Sobol sensitivity analysis
│   │
│   ├── postprocessing/               # Data analysis and visualization
│   │   ├── __init__.py
│   │   ├── __main__.py               # CLI entry point
│   │   ├── cli.py                    # Command-line interface
│   │   ├── postprocess_functions.py  # Data loading and filtering
│   │   ├── plot_kde_functions.py     # KDE quartile plots
│   │   ├── plot_parcoords_functions.py # Parallel coordinates plots
│   │   └── plot_pdf_functions.py     # PDF comparison plots
│   │
│   ├── utils/                        # Utility functions
│   │   ├── __init__.py
│   │   ├── custom_classes.py         # ParameterField class
│   │   ├── io_functions.py           # File I/O operations
│   │   ├── parametric_computation.py # Parametric sweep execution
│   │   ├── sobol_computation.py      # Sobol analysis execution
│   │   ├── system_profiler.py        # Hardware profiling & optimization
│   │   ├── tools.py                  # Helper functions & argument parsing
│   │   └── units_and_constants.py    # Physical constants and units
│   │
│   └── economics/                    # Economic analysis (future)
│       └── __init__.py
│
├── inputs/                           # Configuration files
│   ├── __init__.py
│   ├── params.py                     # Parameter definitions (production)
│   ├── params_test.py                # Parameter definitions (testing)
│   ├── parametric_lump.yaml          # Config: Parametric lump method
│   ├── parametric_tseeded.yaml       # Config: Parametric T-seeded method
│   ├── sobol_lump.yaml               # Config: Sobol lump method
│   ├── sobol_tseeded.yaml            # Config: Sobol T-seeded method
│   ├── postprocess_config.yaml       # Config: Postprocessing (full)
│   └── postprocess_quick.yaml        # Config: Postprocessing (quick)
│
├── outputs/                          # Analysis results (HDF5 files)
│   ├── YYYYMMDD_HHMMSS_parametric_lump/
│   │   ├── h5 files and plots
│
├── tests/                            # Test suite (270 tests)
│   ├── ...
│
└── docs/                             # Documentation (Sphinx)
    ├── ...
```

---
# 1. Running Analysis
Place yourself in the root directory in order to execute

## Using Configuration Files
```bash
python -m ddstartup.main param_file_name.py config_file_name.yaml
```
For example, to run a parameteric analysis - lump method:
```bash
python -m ddstartup.main param.py parametric_lump.yaml
```
or
```bash
python -m ddstartup.main param parametric_lump
```

## Using Command Line Arguments
python -m ddstartup.main config_file_name \
    --parameter min_val max_val \
    ...
#### Command Line Options

- Parameter format: `--param_name min max n_points`
- All parameters are optional (defaults from system profiler will be used)
- Outputs saved to `outputs/YYYYMMDD_HHMMSS_<method>/`

# 2. Postprocessing

All postprocessing functions are in `ddstartup/postprocessing/`.

## Command-Line Interface (CLI)

The postprocessing CLI provides flexible data analysis and visualization options.

### Using YAML Configuration (Recommended)

```bash
# Use configuration file from inputs/ directory
python -m ddstartup.postprocessing postprocess_config.yaml

# Or with full path
python -m ddstartup.postprocessing inputs/postprocess_config.yaml
```

**Configuration file examples:**
- `inputs/postprocess_config.yaml` - Full configuration with all options
- `inputs/postprocess_quick.yaml` - Quick configuration for fast processing

### Command-Line Options Reference

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `config` | - | YAML configuration file | `postprocess_config.yaml` |
| `--files` | `-f` | HDF5 file(s) or folder(s) | `--files file1.h5 file2.h5` |
| `--targets` | `-t` | Target variables to analyze | `--targets t_startup E_lost` |
| `--input-filter` | `-if` | Filter input parameters | `--input-filter "V_plasma<150"` |
| `--output-filter` | `-of` | Filter output variables | `--output-filter "t_startup<1e8"` |
| `--plots` | `-p` | Plot types to generate | `--plots kde parcoords pdf` |
| `--output-dir` | `-o` | Output directory for plots | `--output-dir plots/` |
