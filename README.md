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
| Price of electricity | C<sub>kWh</sub> | $/kWh | 0.1 - 0.4 |

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
6. The economic losses are calculated by multiplying the price of electricity by the energy lost by operating with a D-D startup rather than D-T.

---

### Basic Usage

```bash
# Run parametric analysis
python -m ddstartup params_test parametric_tseeded

# Run with parameter filtering
python -m ddstartup params_test parametric_tseeded_filtered

# Verify configuration without running
python -m ddstartup params_test parametric_tseeded --dry-run

# Run Sobol sensitivity analysis
python -m ddstartup params_sobol sobol_tseeded
```


## Command Reference

### 1. Running Analysis

**Quick start:**
```bash
# Parametric T-seeded method
python -m ddstartup params_test parametric_tseeded

# Parametric lump method  
python -m ddstartup params_test parametric_lump

# Sobol sensitivity analysis
python -m ddstartup params_sobol sobol_tseeded
```

**Command structure:**
```bash
python -m ddstartup <parameter_file> <config_file> [--verbose] [--dry-run]
```

**Files:**
- Parameter files (`.yaml`): `inputs/params_test.yaml`, `inputs/params.yaml`
- Config files (`.yaml`): `inputs/parametric_tseeded.yaml`, `inputs/sobol_tseeded.yaml`

---

### 2. Output Files

Results saved to `outputs/YYYYMMDD_HHMMSS_<method>/`:
- `ddstartup_*.h5`: HDF5 data file with inputs, outputs, and time series
- `config_used.yaml`: Configuration snapshot
- `runtime_info.txt`: Execution metadata

**Key variables in HDF5:**
- Inputs: `V_plasma`, `T_i`, `n_tot`, `tau_p_T`, `P_aux`, `TBR_DT`, etc.
- Outputs: `t_startup`, `E_lost`, `unrealized_profits`, `Q_DD`, `Q_DT_eq`
- Time series: `N_ofc`, `N_ifc`, `N_stor`, `n_T`, `n_D`, `P_DDn`, `P_DT`

---

### 3. Postprocessing

**Load and analyze results:**
```python
import h5py
import numpy as np

with h5py.File('outputs/latest/ddstartup_*.h5', 'r') as f:
    t_startup = f['t_startup'][:]
    success = f['sol_success'][:]
    print(f"Success rate: {np.sum(success)/len(success)*100:.1f}%")
```

**Visualization tools** (in `ddstartup/postprocessing/`):
- `plot_kde_functions.py`: Kernel density estimation
- `plot_parcoords_functions.py`: Parallel coordinates
- `plot_shap_functions.py`: SHAP analysis
- `plot_contour_functions.py`: 2D parameter contours

---

### 4. Tests

```bash
pytest                                    # Run all tests
pytest --cov=ddstartup --cov-report=html  # With coverage
pytest tests/test_*.py -v                 # Specific tests
```

---

### 5. Common Workflows

**Parameter sweep:**
```bash
python -m ddstartup params_test parametric_tseeded
# Analyze with Python/Jupyter using HDF5 output
```

**Sensitivity analysis:**
```bash
python -m ddstartup params_sobol sobol_tseeded
# Calculate Sobol indices from output
```

**Compare methods:**
```bash
python -m ddstartup params_test parametric_lump
python -m ddstartup params_test parametric_tseeded
# Compare outputs/*/ddstartup_*.h5 files
```

---

### 6. File Structure

```
dd_startup/
├── ddstartup/                    # Main package
│   ├── __init__.py
│   ├── __main__.py              # Entry point (python -m ddstartup)
│   ├── main.py                  # CLI argument parsing and workflow
│   ├── physics/                 # Physics models
│   │   ├── lump_functions.py       # Lump method (steady-state)
│   │   ├── Tseeded_functions.py    # T-seeded method (time-dependent ODE)
│   │   ├── reactivity_functions.py # Fusion reactivity calculations
│   │   ├── radiation.py            # Radiation losses
│   │   ├── power_balance.py        # Power balance calculations
│   │   └── sobol_functions.py      # Sobol wrappers
│   ├── methods/                 # Analysis methods
│   │   ├── parametric_computation.py   # Parametric sweeps
│   │   ├── sobol_computation.py        # Sobol sensitivity
│   │   └── elemeffects_computation.py  # Elementary effects
│   ├── economics/               # Economic models
│   │   └── economics_functions.py  # Cost calculations
│   ├── postprocessing/          # Data analysis and visualization
│   │   ├── postprocess_functions.py    # Data loading/filtering
│   │   ├── plot_kde_functions.py       # KDE plots
│   │   ├── plot_parcoords_functions.py # Parallel coordinates
│   │   ├── plot_pdf_functions.py       # PDF comparisons
│   │   ├── plot_contour_functions.py   # Contour plots
│   │   ├── plot_shap_functions.py      # SHAP analysis
│   │   ├── plot_kmeans_functions.py    # K-means clustering
│   │   ├── plot_importance_matrix.py   # Feature importance
│   │   ├── plot_elementary_effects.py  # EE plots
│   │   ├── fit_ML_method.py            # ML surrogate models
│   │   └── cli.py                      # Postprocessing CLI
│   └── utils/                   # Utilities
│       ├── io_functions.py             # File I/O and config loading
│       ├── parameter_loader.py         # YAML parameter parsing
│       ├── parameter_registry.py       # Parameter validation
│       ├── reactivity_lookup.py        # Reactivity caching
│       ├── physics_cache.py            # Physics result caching
│       ├── filters.py                  # Parameter filtering
│       ├── system_profiler.py          # Hardware optimization
│       ├── profiling.py                # Performance profiling
│       ├── tools.py                    # CLI tools
│       └── units_and_constants.py      # Physical constants
├── inputs/                      # Configuration files
│   ├── params_test.yaml            # Test parameters (2 points each)
│   ├── params.yaml                 # Full parameters
│   ├── parametric_tseeded.yaml     # T-seeded config
│   ├── parametric_lump.yaml        # Lump method config
│   ├── sobol_tseeded.yaml          # Sobol config
│   └── postprocess_config.yaml     # Postprocessing config
├── outputs/                     # Analysis results (HDF5 files)
├── tests/                       # Test suite
└── docs/                        # Documentation
```

---

### 7. Troubleshooting

**Import errors:**
```bash
pip install -e .  # Install package in editable mode
```

**Memory issues:**
```bash
# Reduce parameter points in YAML config
# Or use Sobol sampling instead of full parametric sweep
```

**Slow execution:**
```bash
# Check system profile output for recommended n_jobs
# Reduce vector_length in config for faster (less accurate) results
```

---

### 8. References

See `docs/` for detailed documentation on:
- Parameter filtering (PARAMETER_FILTERING_GUIDE.md)
- YAML configuration (YAML_PARAMETERS_GUIDE.md)
- Physics models (POWER_BALANCE_IMPLEMENTATION.md)
- Postprocessing (POSTPROCESSING_DOCS_UPDATE.md)

---

