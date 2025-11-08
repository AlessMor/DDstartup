# Test Migration Checklist

Use this checklist to systematically migrate and improve your test suite.

## ✅ Phase 1: Core Physics (Start Here)

### Reaction Rates
- [x] `test_reaction_rates.py` - **DONE** ✅ (22 tests passing)
- [ ] Add test for temperature array vectorization
- [ ] Add test for extreme temperatures (>1000 keV)
- [ ] Add benchmark tests for performance

### Lump Functions
- [ ] Create `unit/physics/test_lump_functions.py`
  - [ ] Test `calculate_P_Bremsstrahlung_lump`
  - [ ] Test `calculate_P_line_lump`
  - [ ] Test `calculate_P_charged_lump`
  - [ ] Test `calculate_P_aux_lump`
  - [ ] Test `lump_numba` core function
  - [ ] Test `lump_solver` wrapper
  - [ ] Test `compute_single_combination`
  - [ ] Test with impossible parameters (e.g., zero breeding)
  - [ ] Test energy conservation
  - [ ] Test Q factor relationships

### T-Seeded Functions
- [ ] Create `unit/physics/test_tseeded_functions.py`
  - [ ] Test `ode_system` evaluation
  - [ ] Test `solve_ode_system` integration
  - [ ] Test `postprocess_fusion_results_Tseeded`
  - [ ] Test `trapz_numba` integration
  - [ ] Test `get_cached_reaction_rates`
  - [ ] Test `compute_single_combination`
  - [ ] Test `compute_batch_combinations`
  - [ ] Test ODE termination events
  - [ ] Test with different vector lengths
  - [ ] Test interpolation accuracy

### Sobol Functions
- [ ] Create `unit/physics/test_sobol_functions.py`
  - [ ] Test Sobol sequence generation
  - [ ] Test sensitivity calculation
  - [ ] Test variance decomposition

## 📦 Phase 2: I/O Operations

### File I/O
- [ ] Create `unit/io/test_io_functions.py`
  - [ ] Test HDF5 file creation
  - [ ] Test HDF5 file reading
  - [ ] Test CSV export
  - [ ] Test file path handling
  - [ ] Test error handling for missing files
  - [ ] Test large file handling

### Parameter Loading
- [ ] Create `unit/io/test_parameter_loader.py`
  - [ ] Test YAML parameter loading
  - [ ] Test Python module parameter loading
  - [ ] Test parameter validation
  - [ ] Test default value handling
  - [ ] Test error handling for invalid parameters

## 🛠️ Phase 3: Utilities

### Custom Classes
- [ ] Create `unit/utils/test_custom_classes.py`
  - [ ] Test ParameterField initialization
  - [ ] Test parameter validation
  - [ ] Test unit handling
  - [ ] Test parametrization types

### Filters
- [ ] Create `unit/utils/test_filters.py`
  - [ ] Test filter creation
  - [ ] Test filter application
  - [ ] Test filter composition
  - [ ] Test edge cases (all filtered, none filtered)

### Tools
- [ ] Create `unit/utils/test_tools.py`
  - [ ] Test `index_to_params`
  - [ ] Test `make_input_dict`
  - [ ] Test `make_output_dict`
  - [ ] Test `fix_vector_length`

## 🔢 Phase 4: Computational Core

### Parametric Computation
- [ ] Create `unit/test_parametric_computation.py`
  - [ ] Test parameter grid generation
  - [ ] Test linear index conversion
  - [ ] Test parallel computation
  - [ ] Test filtering integration
  - [ ] Test progress reporting

### Sobol Computation
- [ ] Create `unit/test_sobol_computation.py`
  - [ ] Test Sobol sample generation
  - [ ] Test sensitivity analysis
  - [ ] Test variance decomposition
  - [ ] Test index conversion

## 📊 Phase 5: Postprocessing

### Data Processing
- [ ] Create `unit/test_postprocess_functions.py`
  - [ ] Test data aggregation
  - [ ] Test statistical calculations
  - [ ] Test filtering

### Plotting Functions
- [ ] Create `unit/test_plot_kde_functions.py`
  - [ ] Test KDE computation
  - [ ] Test quartile calculations

- [ ] Create `unit/test_plot_pdf_functions.py`
  - [ ] Test PDF generation
  - [ ] Test histogram creation

- [ ] Create `unit/test_plot_shap_functions.py`
  - [ ] Test SHAP value computation
  - [ ] Test importance matrix

- [ ] Create `unit/test_plot_parcoords_functions.py`
  - [ ] Test parallel coordinates data preparation

- [ ] Create `unit/test_plot_contour_functions.py`
  - [ ] Test contour data generation
  - [ ] Test interpolation

## 🔗 Phase 6: Integration Tests

### Full Workflows
- [ ] Create `integration/test_lump_workflow.py`
  - [ ] Test complete lump analysis workflow
  - [ ] Test with different parameter files

- [ ] Create `integration/test_tseeded_workflow.py`
  - [ ] Test complete T-seeded workflow
  - [ ] Test with different parameter files

- [ ] Create `integration/test_sobol_workflow.py`
  - [ ] Test complete Sobol analysis workflow

### CLI Tests
- [ ] Create `integration/test_cli.py`
  - [ ] Test main CLI entry point
  - [ ] Test postprocessing CLI
  - [ ] Test help messages
  - [ ] Test error handling

## ⚡ Phase 7: Power Balance

### Power Calculations
- [ ] Review existing `test_power_balance.py`
- [ ] Review existing `test_power_balance_comprehensive.py`
- [ ] Migrate to new structure if needed
- [ ] Add missing edge cases

## 🎯 Quick Actions

### Today
1. [x] Set up test structure ✅
2. [x] Create conftest.py with fixtures ✅
3. [x] Create template ✅
4. [x] Create example tests ✅
5. [ ] **Next: Copy template and create `test_lump_functions.py`**

### This Week
- [ ] Complete Phase 1 (Core Physics)
- [ ] Start Phase 2 (I/O)

### This Month
- [ ] Complete all unit tests
- [ ] Add integration tests
- [ ] Achieve >80% code coverage

## 📝 Notes

### Test Priorities (High to Low)
1. **Core physics** - These are fundamental
2. **I/O functions** - Critical for reliability
3. **Computational core** - Ensures correctness
4. **Utilities** - Support functions
5. **Postprocessing** - Visualization and analysis
6. **Integration** - End-to-end validation

### Coverage Goals
- Physics modules: >90%
- I/O modules: >85%
- Utils: >80%
- Overall: >80%

### Style Guidelines
- One test, one assertion (mostly)
- Use descriptive test names
- Group related tests in classes
- Use parametrize for multiple inputs
- Mark tests appropriately (slow, integration, etc.)

## 🔄 Migration Strategy

For each old test file in `tests/`:
1. Review existing tests
2. Copy relevant tests to new structure
3. Improve test quality (add edge cases, better assertions)
4. Add new tests for uncovered functionality
5. Mark old test as migrated (add comment)
6. Don't delete old tests until all are migrated

## ✨ Quick Wins

Easy tests to write that provide high value:
- [ ] Reaction rate range checks (done!)
- [ ] Function return type tests
- [ ] Positive value checks for physical quantities
- [ ] Finite value checks (no NaN/Inf)
- [ ] Basic input validation

---

**Legend:**
- [ ] Not started
- [x] Complete ✅
- [~] In progress

**Last updated:** 2025-11-06
