# Parameter Symbol Formatting for Plots

## Overview

All plotting functions in the `ddstartup.postprocessing` module now use **LaTeX-formatted mathematical symbols** instead of parameter names for improved readability and scientific presentation.

## What Changed

### Before
Plot labels showed raw parameter names:
- `V_plasma [m³]`
- `tau_p_T [s]`
- `unrealized_gains [J]`

### After
Plot labels now show proper mathematical symbols:
- $V_{\mathrm{plasma}}$ [m³]
- $\tau_{p,T}$ [s]
- $E_{\mathrm{unrealized}}$ [J]

## Symbol Mappings

The complete symbol mapping is defined in `ddstartup/utils/parameter_symbols.py`:

### Plasma Parameters
| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `V_plasma` | $V_{\mathrm{plasma}}$ | Plasma volume |
| `T_i` | $T_i$ | Ion temperature |
| `n_tot` | $n_{\mathrm{tot}}$ | Total density |
| `n_T` | $n_T$ | Tritium density |
| `n_D` | $n_D$ | Deuterium density |
| `n_He3` | $n_{\mathrm{He3}}$ | Helium-3 density |

### Confinement Times
| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `tau_p_T` | $\tau_{p,T}$ | Tritium particle confinement time |
| `tau_p_He3` | $\tau_{p,\mathrm{He3}}$ | Helium-3 particle confinement time |
| `tau_ifc` | $\tau_{\mathrm{ifc}}$ | In-fuel-cycle confinement time |
| `tau_ofc` | $\tau_{\mathrm{ofc}}$ | Out-of-fuel-cycle confinement time |

### Powers
| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `P_aux` | $P_{\mathrm{aux}}$ | Auxiliary power |
| `P_aux_DT_eq` | $P_{\mathrm{aux,DT}}$ | Auxiliary power at DT equilibrium |
| `P_DDn` | $P_{\mathrm{DD}_n}$ | DD neutron fusion power |
| `P_DDp` | $P_{\mathrm{DD}_p}$ | DD proton fusion power |
| `P_DT` | $P_{\mathrm{DT}}$ | DT fusion power |
| `P_DT_eq` | $P_{\mathrm{DT,eq}}$ | DT equilibrium fusion power |

### Tritium Breeding Ratios
| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `TBR_DT` | $\mathrm{TBR}_{\mathrm{DT}}$ | Tritium breeding ratio for DT |
| `TBR_DDn` | $\mathrm{TBR}_{\mathrm{DD}_n}$ | Tritium breeding ratio for DD-n |

### Inventory
| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `I_target` | $I_{\mathrm{target}}$ | Target inventory |
| `N_ofc` | $N_{\mathrm{ofc}}$ | Out-of-fuel-cycle inventory |
| `N_ifc` | $N_{\mathrm{ifc}}$ | In-fuel-cycle inventory |
| `N_stor` | $N_{\mathrm{stor}}$ | Storage inventory |

### Performance Metrics
| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `Q_DD` | $Q_{\mathrm{DD}}$ | DD Q-factor |
| `Q_DT_eq` | $Q_{\mathrm{DT,eq}}$ | DT equilibrium Q-factor |
| `TBE` | $\mathrm{TBE}$ | Tritium burn-up efficiency |

### Efficiencies and Economics
| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `eta_th` | $\eta_{\mathrm{th}}$ | Thermal efficiency |
| `capacity_factor` | $f_{\mathrm{cap}}$ | Capacity factor |
| `cost_of_electricity` | $C_{\mathrm{elec}}$ | Cost of electricity |

### Time and Energy
| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `t_startup` | $t_{\mathrm{startup}}$ | Startup time |
| `E_lost` | $E_{\mathrm{lost}}$ | Lost energy |
| `unrealized_profits` | $E_{\mathrm{unrealized}}$ | Unrealized energy gains |
| `unrealized_gains` | $E_{\mathrm{unrealized}}$ | Unrealized energy gains |

## Affected Plot Types

All plot types now use symbolic notation:

1. **SHAP Plots** (`plot_shap_functions.py`)
   - Y-axis labels: Parameter symbols with correlation
   - X-axis label: Impact on target (with symbol)
   - Plot title: Feature importance with symbol

2. **PDF Plots** (`plot_pdf_functions.py`)
   - X-axis label: Variable symbol with unit
   - Plot title: PDF of variable symbol

3. **KDE Plots** (`plot_kde_functions.py`)
   - Subplot titles: Parameter symbols with units
   - Legend title: Target symbol with unit
   - Main title: Target symbol

4. **Parallel Coordinates** (`plot_parcoords_functions.py`)
   - Axis labels: Parameter symbols with units
   - Colorbar label: Target symbol with unit
   - Plot title: Target symbol

5. **Contour Plots** (`plot_contour_functions.py`)
   - X/Y axis labels: Parameter symbols with units
   - Colorbar label: Target symbol with unit
   - Plot title: Symbols for all variables

6. **K-means Plots** (`plot_kmeans_functions.py`)
   - Plot title: Target symbol

## Usage

### For End Users

No action required! Symbols are automatically applied to all plots generated through:

```bash
# Command line
python -m ddstartup.postprocessing outputs/your_output_dir/

# With specific plot types
python -m ddstartup.postprocessing outputs/your_output_dir/ --shap --kde --pdf
```

### For Developers

#### Using the Symbol API

```python
from ddstartup.utils.parameter_symbols import get_param_symbol, get_param_label

# Get just the symbol
symbol = get_param_symbol('V_plasma')
# Returns: '$V_{\\mathrm{plasma}}$'

# Get symbol with unit
label = get_param_label('V_plasma', 'm³')
# Returns: '$V_{\\mathrm{plasma}}$ [m³]'

# Get label without symbol (fallback to parameter name)
label = get_param_label('V_plasma', 'm³', use_symbol=False)
# Returns: 'V_plasma [m³]'
```

#### Adding New Symbols

To add a new parameter symbol, edit `ddstartup/utils/parameter_symbols.py`:

```python
PARAM_SYMBOLS = {
    # ... existing symbols ...
    'my_new_param': r'$\alpha_{\mathrm{new}}$',
}
```

**LaTeX Symbol Guidelines:**
- Use `\mathrm{}` for text subscripts: `$V_{\mathrm{plasma}}$`
- Use raw subscripts for variables: `$T_i$`, `$n_D$`
- Use proper Greek letters: `$\tau$`, `$\eta$`, `$\alpha$`
- Keep symbols consistent with standard fusion notation

## Testing

Run the test script to verify symbol mappings:

```bash
python test_plot_symbols.py
```

This will:
- Display all symbol mappings
- Show formatting examples
- Check coverage (all PARAM_UNITS have symbols)

## Benefits

1. **Professional Appearance**: Plots look publication-ready
2. **Standard Notation**: Consistent with fusion physics literature
3. **Readability**: Mathematical symbols are more familiar than code names
4. **Clarity**: Units are clearly separated from variable names
5. **Flexibility**: Easy to switch between symbols and parameter names

## Backward Compatibility

The system maintains full backward compatibility:
- Parameter names work everywhere symbols work
- Functions accept both parameter names and return proper symbols
- CSV files and data still use original parameter names
- Only visual presentation is affected

## Implementation Details

### Architecture

```
parameter_symbols.py
├── PARAM_SYMBOLS dict      # Parameter name → LaTeX symbol mapping
├── get_param_symbol()      # Get symbol for parameter
└── get_param_label()       # Get formatted label (symbol + unit)

All plot functions
├── Import parameter_symbols
├── Import PARAM_UNITS from tools
└── Use get_param_label() for axes, titles, legends
```

### Performance

Negligible performance impact:
- Symbol lookup is O(1) dictionary access
- No matplotlib rendering changes (same LaTeX backend)
- No additional dependencies

## Examples

### Before and After Comparison

**SHAP Plot:**
```
Before: Feature Importance: unrealized_gains
After:  Feature Importance: $E_{\mathrm{unrealized}}$ [J]

Before: Impact on t_startup (correlation × standardized value)
After:  Impact on $t_{\mathrm{startup}}$ (correlation × standardized value)
```

**KDE Plot:**
```
Before: KDE of Inputs by t_startup quartile
After:  KDE of Inputs by $t_{\mathrm{startup}}$ quartile

Before: V_plasma [m³]
After:  $V_{\mathrm{plasma}}$ [m³]
```

**Parallel Coordinates:**
```
Before: Parallel Coordinates Plot (t_startup)
After:  Parallel Coordinates Plot ($t_{\mathrm{startup}}$)
```

## Future Enhancements

Potential future improvements:
1. User configuration to toggle symbols on/off
2. Alternative symbol sets (e.g., ITER notation, textbook notation)
3. HTML rendering for interactive plots
4. Custom symbol definitions per project

## Summary

All plots now use professional mathematical notation automatically. No changes required to your workflow - just run postprocessing as usual and enjoy publication-quality plots! 🎨📊
