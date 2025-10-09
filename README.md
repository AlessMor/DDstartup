# DD Startup Analysis Toolbox

## Aim of the project
The aim of this project is to create a **device-agnostic, integrated analysis toolbox** to **evaluate the operational regime that could enable a D-D startup** built upon open-source tools. 

The model couples plasma performance (density and temperature profiles), neutronics for D–D and D–T neutron spectra, transient fuel-cycle inventory dynamics, tritium processing times, and economic drivers. 

Two complementary approaches are used: (1) a **target-accumulation estimate**, computing the time to produce and store a startup inventory under the assumption of steady-state D-D operation, and (2) a **time-dependent model  of the  fuel-cycle inventories** to capture transient coupling between production, burn, and processing in the fuel cycle as the plasma is started in D-D and progressively transitioned up to a 50-50 D-T mixture by injecting the tritium produced on-site.

Rather than claiming a single best route, our aim is to **map the design space and quantify sensitivities across technical and financial levers**.

**These open, reproducible, and extensible analyses are intended to help designers, modelers, and decision-makers assess the feasibility, timelines, and economic requirements of tritium-free or tritium-lean startup strategies.**

## Physical background - Generating trtium in a fusion reactor
The main reactions inside a D-D mixture are:  
D + D -> T + p  
D + D -> He3 + n  
D + T -> He4 + n  
D + He3 -> He4 + p  

The main pathways leading to trtium production are then:  

            |----> He3 + n (2.45 MeV)
            |            |--------------------> **Tdot_breedingDD**: tritium production due to DD neutrons 
            |                                   interacting with the Li6 in the breeding blanket
    D + D ->|
      |     |----> T + p
      |            |
      |------------|---> D + T --> He4 + n (14.1 MeV)
                   |                     |----> **Tdot_breedingDT**: tritium production due to DT neutrons 
                   |                            interacting with the breeding blanket
                   |
                   |--------------------------> **Tdot_fusion**: tritium production due to DDp fusions, 
                                                considering the losses due to DT fusions (tritium inside the plasma diffuses) 

## Main parameters considered (PRELIMINARY VALUES)
The following parameters have been selected for the analysis (suggested maximum ranges are also reported):

| Parameter | Symbol | Unit | Range |
| --------- | ------ | ---- | ----- |
| <center>Reactor parameters</center> |
| Plasma volume | V_p | m<sup>3</sup> | 10 - 200 |
| Total ion density | n_i_tot | m<sup>-3</sup> | 10<sup>20</sup> - 10<sup>21</sup> |
| Ion temperature | T_i | keV | 10-100 |
| Tritium confinement time | tau<sub>p,T</sub> | s | 0.1 - 5 |
| Helium-3 confinement time | tau<sub>p,He3</sub> | s | 0.1 - 5 |
| Auxiliary power (D-D operation) | P<sub>aux</sub> | MW | 20 - 60 |
| Auxiliary power (D-T operation) | P<sub>aux,DTeq</sub> | MW | 20 - 60 |
| <center>Fuel cycle parameters</center> |
| Tritium Breeding Ratio for D-T neutrons | TBR <sub>DT</sub> | - | 1.05 - 1.2 |
| Tritium Breeding Ratio for D-D neutrons | TBR <sub>DD</sub> | - | 0.5 - 1 |
| Inner fuel cycle residence time | $\tau$<sub>ifc</sub> | h | 1 - 24 |
| Inner fuel cycle residence time | $\tau$<sub>ifc</sub> | h | 12 - 48 |
| Target inventory | I<sub>target</sub> | kg | 0.1 - 5 |
| <center>Plant and economic parameters</center> |
| Thermal efficiency | $\eta_{th}$ | - | 0.3 - 0.5 |
| Capacity factor | C<sub>f</sub> | - | 0.5 - 0.9 |
| Cost of electricity | C<sub>kWh</sub> | $/kWh | 0.1 - 0.4 |

## Tritium production model
This section details the steps necessary to evaluate the tritium produced. Since two different methods have been considered, some steps may be explained for both methods.

1. Define the ranges for the parameters used in the selected method, using the custom class [ParameterField](utils/custom_classes).  
The code will then build the iterator element by creating all possible combinations
2. The code will then evaluate the reactivities ($<\sigma v >$) for each reaction type, by using the correlations defined by Bosch and Hale and implemented in the cfspopcon Python package
3. Depending on the method selected a different approach will be taken:

| Case 1: Analysis up to $I_{target}$ | Case 2: Analysis up to D-T operation |
| --- | --- |
| The code will calculate the tritium production rates due to D - D and D - T neutrons breeding, as well as tritium diffusion. The startup time (time needed to reach the target inventory) will then be calculated taking into account trtium decay, with the formula: $t_{st} = -\frac{1}{\lambda} ln \left( 1-\frac{\lambda N_{target} }{ \dot{}_{tot} } \right) $  | The code will solve a system of equations using *solve_ivp* until a 50D-50T mixture is reached. The time value at which this mixture is achieved will be $t_{st}$|

4. Once the time needed to reach the inventory target (for case 1) or to reach D-T operation (case 2) is known, the code will evaluate the net electric energy produced during operation, taking into account auxiliary heating, thermal efficiency and availability.
5. The resulting energy will be compared with the power that the same reactor would have produced if it was operated using a 50D-50T mixture from the beginning of operation. In this case different valeus for the power lossess due to radiation and auxiliary heating will be used.
6. The economic losses are calculated by multiplying the cost of electricity by the energy lost by operating with a D-D startup rather than D-T.

---

### Basic Usage

```bash
# Run parametric analysis with config file
python -m ddstartup.main config inputs/config_parametric_lump.yaml

# Run with direct parameter specification
python -m ddstartup.main parametric_lump --V_plasma 50 100 5 --n_tot 5e20 1e21 3
```

---

## Command Reference

### 1. Running Analysis

#### Using Configuration Files

**Parametric Analysis (Lump Method):**
```bash
python -m ddstartup.main config inputs/config_parametric_lump.yaml
```

**Parametric Analysis (T-seeded Method):**
```bash
python -m ddstartup.main config inputs/config_parametric_tseeded.yaml
```

**Sobol Sensitivity Analysis (Lump):**
```bash
python -m ddstartup.main config inputs/config_sobol_lump.yaml
```

**Sobol Sensitivity Analysis (T-seeded):**
```bash
python -m ddstartup.main config inputs/config_sobol_tseeded.yaml
```

#### Using Command Line Arguments

**Parametric Lump Method:**
```bash
python -m ddstartup.main parametric_lump \
    --V_plasma 50 100 5 \
    --n_tot 5e20 1e21 3 \
    --tau_p_T 0.5 2.0 4 \
    --tau_p_He3 0.5 2.0 4 \
    --P_aux 30e6 50e6 3 \
    --P_aux_DT_eq 40e6 60e6 3 \
    --TBR_DT 1.05 1.15 3 \
    --TBR_DDn 0.7 0.9 3 \
    --I_target 1.0 3.0 3 \
    --eta_th 0.35 0.45 3 \
    --capacity_factor 0.7 0.9 3 \
    --cost_of_electricity 0.10 0.20 3
```

**Parametric T-seeded Method:**
```bash
python -m ddstartup.main parametric_tseeded \
    --V_plasma 50 100 5 \
    --n_tot 5e20 1e21 3 \
    --tau_p_T 0.5 2.0 4 \
    --P_aux 30e6 50e6 3 \
    --P_aux_DT_eq 40e6 60e6 3 \
    --TBR_DT 1.05 1.15 3 \
    --TBR_DDn 0.7 0.9 3 \
    --tau_ifc 3600 86400 4 \
    --tau_ofc 43200 172800 4 \
    --eta_th 0.35 0.45 3 \
    --capacity_factor 0.7 0.9 3 \
    --cost_of_electricity 0.10 0.20 3
```

**Sobol Sensitivity Analysis:**
```bash
# Lump method
python -m ddstartup.main sobol_lump \
    --V_plasma 50 150 \
    --n_tot 5e20 1e21 \
    --N_samples 1024

# T-seeded method
python -m ddstartup.main sobol_tseeded \
    --V_plasma 50 150 \
    --n_tot 5e20 1e21 \
    --N_samples 1024
```

#### Command Line Options

- Parameter format: `--param_name min max n_points`
- All parameters are optional (defaults from system profiler will be used)
- Outputs saved to `outputs/YYYYMMDD_HHMMSS_<method>/`

---

### 2. Postprocessing

All postprocessing functions are in `ddstartup/postprocessing/`.

#### Interactive Notebooks

Use the manual verification notebooks in `tests/`:
- `manual_kde_plots_verification.ipynb` - KDE quartile plots
- `manual_parcoords_plots_verification.ipynb` - Parallel coordinates
- `manual_pdf_plots_verification.ipynb` - PDF comparison plots

```bash
# Launch Jupyter
jupyter notebook tests/
```

---

### 3. Running Tests

#### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=ddstartup --cov-report=html

# Run with coverage (terminal report)
pytest --cov=ddstartup --cov-report=term-missing
```

#### Run Specific Test Files

```bash
# Test postprocessing functions
pytest tests/test_postprocess_functions.py -v

# Test KDE plotting
pytest tests/test_plot_kde_functions.py -v

# Test parallel coordinates plotting
pytest tests/test_plot_parcoords_functions.py -v

# Test PDF plotting
pytest tests/test_plot_pdf_functions.py -v
```

#### Run Specific Tests

```bash
# Run specific test by name
pytest tests/test_postprocess_functions.py::test_find_latest_output_folder -v

# Run tests matching pattern
pytest -k "test_kde" -v

# Run tests with markers (if defined)
pytest -m "slow" -v
```

#### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=ddstartup --cov-report=html
# View report: open htmlcov/index.html

# Generate XML coverage report (for CI/CD)
pytest --cov=ddstartup --cov-report=xml

# Show missing lines in terminal
pytest --cov=ddstartup --cov-report=term-missing

# Coverage for specific module
pytest tests/test_postprocess_functions.py \
    --cov=ddstartup.postprocessing.postprocess_functions \
    --cov-report=html
```

#### Test Options

```bash
# Stop at first failure
pytest -x

# Show local variables on failure
pytest -l

# Run last failed tests only
pytest --lf

# Parallel execution (requires pytest-xdist)
pytest -n auto

# Disable warnings
pytest --disable-warnings
```

---

### 4. Documentation

#### Build Documentation

```bash
cd docs

# Build HTML documentation
make html

# Build PDF documentation (requires LaTeX)
make latexpdf

# Clean build artifacts
make clean

# View documentation
python -m http.server 8000 --directory _build/html
# Open browser to http://localhost:8000
```

#### Documentation Structure

```
docs/
├── index.rst              # Main documentation page
├── user_guide/            # User guides
│   ├── installation.rst
│   ├── quickstart.rst
│   └── postprocessing_workflow.rst
├── api_reference/         # API documentation
│   ├── physics.rst
│   ├── postprocessing.rst
│   └── utils.rst
└── conf.py               # Sphinx configuration
```

---

### 5. Development Workflow

#### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov sphinx sphinx-rtd-theme

# Install package in editable mode
pip install -e .
```

#### Code Quality Checks

```bash
# Run tests before committing
pytest

# Check code coverage
pytest --cov=ddstartup --cov-report=term-missing

# Format code (if using black)
black ddstartup/

# Lint code (if using flake8)
flake8 ddstartup/
```

#### Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes and commit
git add .
git commit -m "Add new feature"

# Run tests
pytest

# Push changes
git push origin feature/my-new-feature
```

---

### 6. File Structure

```
dd_startup/
├── ddstartup/                 # Main package
│   ├── __init__.py
│   ├── main.py               # Entry point
│   ├── physics/              # Physics models
│   │   ├── lump_functions.py
│   │   ├── Tseeded_functions.py
│   │   └── reactionrates_functions.py
│   ├── postprocessing/       # Data analysis
│   │   ├── postprocess_functions.py
│   │   ├── plot_kde_functions.py
│   │   ├── plot_parcoords_functions.py
│   │   └── plot_pdf_functions.py
│   └── utils/                # Utilities
│       ├── io_functions.py
│       ├── parametric_computation.py
│       ├── sobol_computation.py
│       └── tools.py
├── inputs/                   # Configuration files
│   ├── config_parametric_lump.yaml
│   ├── config_parametric_tseeded.yaml
│   ├── config_sobol_lump.yaml
│   └── config_sobol_tseeded.yaml
├── outputs/                  # Analysis results (HDF5)
├── tests/                    # Test suite
│   ├── test_postprocess_functions.py
│   ├── test_plot_kde_functions.py
│   ├── test_plot_parcoords_functions.py
│   ├── test_plot_pdf_functions.py
│   └── conftest.py
├── docs/                     # Documentation
│   ├── conf.py
│   ├── index.rst
│   ├── user_guide/
│   └── api_reference/
├── requirements.txt          # Dependencies
└── README.md                # This file
```

---

### 7. Output Files

Analysis results are saved to `outputs/YYYYMMDD_HHMMSS_<method>/`:

```
outputs/20241008_143015_parametric_T_seeded/
├── parametric_T_seeded.h5           # Main data file (HDF5)
├── config_used.yaml                  # Configuration snapshot
└── runtime_info.txt                  # Execution metadata
```

**HDF5 File Structure:**
- Input parameters: `V_plasma`, `n_tot`, `tau_p_T`, etc.
- Output variables: `t_startup`, `unrealized_profits`, `E_lost`, etc.
- Time series (T-seeded): `N_ofc`, `N_ifc`, `N_stor`, `n_T`, `n_D`
- Metadata: `sol_success`, `linear_index`

---

### 8. Common Workflows

#### Workflow 1: Parameter Sweep

```bash
# 1. Run parametric analysis
python -m ddstartup.main config inputs/config_parametric_lump.yaml

# 2. Analyze results
jupyter notebook tests/manual_kde_plots_verification.ipynb

# 3. Generate publication plots
python -m ddstartup.postprocessing.plot_kde_functions \
    outputs/latest/parametric_lump.h5 \
    --target t_startup \
    --output plots/
```

#### Workflow 2: Sensitivity Analysis

```bash
# 1. Run Sobol analysis
python -m ddstartup.main config inputs/config_sobol_lump.yaml

# 2. Calculate Sobol indices
python -m ddstartup.postprocessing.sobol_indices \
    outputs/latest/sobol_lump.h5

# 3. Plot sensitivity results
python -m ddstartup.postprocessing.plot_sobol \
    outputs/latest/sobol_lump.h5 \
    --output plots/sobol_indices.png
```

#### Workflow 3: Comparing Methods

```bash
# 1. Run both methods
python -m ddstartup.main config inputs/config_parametric_lump.yaml
python -m ddstartup.main config inputs/config_parametric_tseeded.yaml

# 2. Compare results
jupyter notebook tests/manual_pdf_plots_verification.ipynb
# Select both output files for comparison
```

---

### 9. Troubleshooting

#### Common Issues

**Import Errors:**
```bash
# Ensure package is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/dd_startup"
```

**Memory Issues:**
```bash
# Reduce number of parameter points
python -m ddstartup.main parametric_lump --V_plasma 50 100 3  # fewer points

# Use Sobol instead of parametric for high dimensions
python -m ddstartup.main sobol_lump --N_samples 512
```

**Slow Execution:**
```bash
# Check available cores
python -c "import os; print(os.cpu_count())"

# Set manual thread count
export OMP_NUM_THREADS=8
python -m ddstartup.main config inputs/config_parametric_lump.yaml
```

**Test Failures:**
```bash
# Run with verbose output
pytest -v --tb=short

# Debug specific test
pytest tests/test_postprocess_functions.py::test_name -vv --tb=long
```

---

### 10. Contact & Contributing

[WIP]

---

### 11. License

[WIP]

---

### 12. References

[WIP]

