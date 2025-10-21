# Summary: Parameter Symbol Formatting Implementation

## What Was Implemented

All plotting functions in the postprocessing module now display **LaTeX-formatted mathematical symbols** instead of raw parameter names, providing publication-quality plots with professional scientific notation.

## Changes Made

### 1. Core Symbol System
**Created:** `ddstartup/utils/parameter_symbols.py`
- Defined `PARAM_SYMBOLS` dictionary with 32 parameter mappings
- Implemented `get_param_symbol(param_name)` - returns LaTeX symbol
- Implemented `get_param_label(param_name, unit, use_symbol)` - returns formatted label

### 2. Updated Plotting Functions

All postprocessing plot functions now use symbols:

#### ✅ `plot_shap_functions.py`
- Y-axis labels: Parameter symbols with correlation values
- X-axis label: "Impact on {symbol}"
- Plot title: "Feature Importance: {symbol}"
- Subtitle: Top feature shown with symbol

#### ✅ `plot_pdf_functions.py`
- X-axis label: Symbol with unit
- Plot title: "PDF of {symbol}"

#### ✅ `plot_kde_functions.py`
- Subplot titles: Symbol with unit for each parameter
- Legend title: Target symbol with unit
- Main title: "KDE of Inputs by {symbol} quartile"

#### ✅ `plot_parcoords_functions.py`
- Axis labels: Symbols with units (HTML format: `symbol<br>[unit]`)
- Colorbar label: Target symbol with unit
- Plot title: "Parallel Coordinates Plot ({symbol})"

#### ✅ `plot_contour_functions.py`
- X/Y axis labels: Symbols with units
- Colorbar label: Target symbol with unit
- Plot titles: "{target_symbol} over {x_symbol} vs {y_symbol}"
- Error case titles: "{x_symbol} vs {y_symbol}"

#### ✅ `plot_kmeans_functions.py`
- Plot title: "Quartile Distribution per Cluster — {symbol}"

#### ✅ `plot_importance_matrix.py`
- Plot title: "Effect Size (Cohen's d) per Quartile — {symbol}"

### 3. Documentation
**Created:**
- `docs/PLOT_SYMBOL_FORMATTING.md` - Complete documentation with examples
- `PLOT_SYMBOLS_QUICK_REF.md` - Quick reference guide
- `test_plot_symbols.py` - Test script for symbol system

## Symbol Examples

### Common Parameter Transformations

```
Parameter Name    →  LaTeX Symbol
──────────────────────────────────────────────────────
V_plasma          →  $V_{\mathrm{plasma}}$
T_i               →  $T_i$
n_tot             →  $n_{\mathrm{tot}}$
tau_p_T           →  $\tau_{p,T}$
P_aux             →  $P_{\mathrm{aux}}$
TBR_DT            →  $\mathrm{TBR}_{\mathrm{DT}}$
t_startup         →  $t_{\mathrm{startup}}$
Q_DD              →  $Q_{\mathrm{DD}}$
E_lost            →  $E_{\mathrm{lost}}$
unrealized_gains  →  $E_{\mathrm{unrealized}}$
eta_th            →  $\eta_{\mathrm{th}}$
capacity_factor   →  $f_{\mathrm{cap}}$
```

### Full Label Examples (with units)

```
Before:  V_plasma [m³]
After:   $V_{\mathrm{plasma}}$ [m³]

Before:  tau_p_T [s]
After:   $\tau_{p,T}$ [s]

Before:  unrealized_gains [J]
After:   $E_{\mathrm{unrealized}}$ [J]
```

## API Usage

### For End Users
No changes required! Just run postprocessing as usual:

```bash
# Activate environment
conda activate ddstartupenv

# Run postprocessing (symbols applied automatically)
python -m ddstartup.postprocessing outputs/your_output_directory/

# Or with specific plot types
python -m ddstartup.postprocessing outputs/your_output_directory/ --shap --kde --pdf
```

### For Developers

```python
from ddstartup.utils.parameter_symbols import get_param_symbol, get_param_label

# Get LaTeX symbol for a parameter
symbol = get_param_symbol('V_plasma')
# Returns: '$V_{\\mathrm{plasma}}$'

# Get formatted label with unit
label = get_param_label('V_plasma', 'm³')
# Returns: '$V_{\\mathrm{plasma}}$ [m³]'

# Get label without symbol (fallback to name)
label = get_param_label('V_plasma', 'm³', use_symbol=False)
# Returns: 'V_plasma [m³]'
```

## Testing

Run the test script:
```bash
conda activate ddstartupenv
python test_plot_symbols.py
```

**Test Results:**
- ✅ All 32 parameter symbols defined
- ✅ All PARAM_UNITS parameters have symbol mappings (100% coverage)
- ✅ Symbol formatting working correctly
- ✅ Label formatting with/without symbols working
- ✅ Label formatting with/without units working

## Files Modified

### Created (3 files)
1. `ddstartup/utils/parameter_symbols.py` - Symbol definitions and API
2. `docs/PLOT_SYMBOL_FORMATTING.md` - Complete documentation
3. `test_plot_symbols.py` - Test/verification script

### Modified (7 files)
1. `ddstartup/postprocessing/plot_shap_functions.py`
2. `ddstartup/postprocessing/plot_pdf_functions.py`
3. `ddstartup/postprocessing/plot_kde_functions.py`
4. `ddstartup/postprocessing/plot_parcoords_functions.py`
5. `ddstartup/postprocessing/plot_contour_functions.py`
6. `ddstartup/postprocessing/plot_kmeans_functions.py`
7. `ddstartup/postprocessing/plot_importance_matrix.py`

### Documentation (2 files)
1. `docs/PLOT_SYMBOL_FORMATTING.md` - Full guide
2. `PLOT_SYMBOLS_QUICK_REF.md` - Quick reference

## Benefits

1. **Professional Appearance** - Publication-ready plots
2. **Standard Notation** - Consistent with fusion physics literature
3. **Improved Readability** - Mathematical symbols more familiar than code names
4. **Clear Units** - Units clearly separated from variable names
5. **Automatic** - No user configuration needed
6. **Flexible** - Easy to add new symbols or customize
7. **Backward Compatible** - Parameter names still work everywhere

## Implementation Details

### Design Pattern
```
All plot functions:
├── Import: from ..utils.parameter_symbols import get_param_symbol, get_param_label
├── Import: from ..utils.tools import PARAM_UNITS
└── Replace all labels:
    - ax.set_xlabel(param) → ax.set_xlabel(get_param_label(param, unit))
    - ax.set_title(target) → ax.set_title(get_param_symbol(target))
    - etc.
```

### Symbol Notation Guidelines
- Text subscripts: `$V_{\mathrm{plasma}}$` (use `\mathrm{}`)
- Variable subscripts: `$T_i$`, `$n_D$` (raw)
- Greek letters: `$\tau$`, `$\eta$`, `$\alpha$`
- Ratios/factors: `$\mathrm{TBR}_{\mathrm{DT}}$`

## Performance Impact

**Negligible:**
- Symbol lookup: O(1) dictionary access
- No matplotlib rendering changes
- No additional dependencies
- Same LaTeX backend

## Validation

### Code Quality
- ✅ No syntax errors
- ✅ No import errors
- ✅ All functions updated consistently
- ✅ Test script passes

### Coverage
- ✅ All 7 plot function types updated
- ✅ All 32 parameters in PARAM_UNITS have symbols
- ✅ All label types covered (axes, titles, legends, colorbars)

## Next Steps

### To Use
1. Run your parametric or Sobol analysis as usual
2. Run postprocessing: `python -m ddstartup.postprocessing outputs/your_dir/`
3. View plots - all labels now use symbols!

### To Extend
1. Add new parameters to `PARAM_SYMBOLS` in `parameter_symbols.py`
2. Follow LaTeX notation guidelines
3. Test with `python test_plot_symbols.py`

## Example Output

When you generate plots, you'll see improvements like:

**SHAP Plot:**
- Old: "Feature Importance: unrealized_gains"
- New: "Feature Importance: $E_{\mathrm{unrealized}}$ [J]"

**KDE Plot:**
- Old: "V_plasma [m³]"
- New: "$V_{\mathrm{plasma}}$ [m³]"

**Parallel Coordinates:**
- Old: "tau_p_T [s]"
- New: "$\tau_{p,T}$ [s]"

**Contour Plot:**
- Old: "t_startup over V_plasma vs T_i"
- New: "$t_{\mathrm{startup}}$ over $V_{\mathrm{plasma}}$ vs $T_i$"

---

## Summary

✅ **Complete implementation of parameter symbol formatting across all postprocessing plots**
- 7 plot types updated
- 32 symbols defined
- 100% parameter coverage
- Fully tested and documented
- Zero configuration required
- Publication-quality output

**Status:** Ready for production use! 🎨📊
