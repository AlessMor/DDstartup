# Quick Reference: Parameter Symbols

## Common Symbols

### Quick Lookup Table

```
Parameter Name         →  Symbol                       Unit
─────────────────────────────────────────────────────────────
V_plasma              →  $V_{\mathrm{plasma}}$        m³
T_i                   →  $T_i$                        keV
n_tot                 →  $n_{\mathrm{tot}}$           m⁻³
n_T                   →  $n_T$                        m⁻³
n_D                   →  $n_D$                        m⁻³

tau_p_T               →  $\tau_{p,T}$                 s
tau_p_He3             →  $\tau_{p,\mathrm{He3}}$      s
tau_ifc               →  $\tau_{\mathrm{ifc}}$        s
tau_ofc               →  $\tau_{\mathrm{ofc}}$        s

P_aux                 →  $P_{\mathrm{aux}}$           W
P_aux_DT_eq           →  $P_{\mathrm{aux,DT}}$        W
P_DDn                 →  $P_{\mathrm{DD}_n}$          W
P_DDp                 →  $P_{\mathrm{DD}_p}$          W
P_DT                  →  $P_{\mathrm{DT}}$            W
P_DT_eq               →  $P_{\mathrm{DT,eq}}$         W

TBR_DT                →  $\mathrm{TBR}_{\mathrm{DT}}$ -
TBR_DDn               →  $\mathrm{TBR}_{\mathrm{DD}_n}$ -

I_target              →  $I_{\mathrm{target}}$        kg
N_ofc                 →  $N_{\mathrm{ofc}}$           kg
N_ifc                 →  $N_{\mathrm{ifc}}$           kg
N_stor                →  $N_{\mathrm{stor}}$          kg

Q_DD                  →  $Q_{\mathrm{DD}}$            -
Q_DT_eq               →  $Q_{\mathrm{DT,eq}}$         -
TBE                   →  $\mathrm{TBE}$               -

eta_th                →  $\eta_{\mathrm{th}}$         -
capacity_factor       →  $f_{\mathrm{cap}}$           -
cost_of_electricity   →  $C_{\mathrm{elec}}$          1/J

t_startup             →  $t_{\mathrm{startup}}$       s
E_lost                →  $E_{\mathrm{lost}}$          J
unrealized_gains      →  $E_{\mathrm{unrealized}}$    J
```

## Usage Examples

### Python API
```python
from ddstartup.utils.parameter_symbols import get_param_symbol, get_param_label

# Get symbol only
get_param_symbol('V_plasma')
# → '$V_{\\mathrm{plasma}}$'

# Get symbol with unit
get_param_label('V_plasma', 'm³')
# → '$V_{\\mathrm{plasma}}$ [m³]'

# Get parameter name (no symbol)
get_param_label('V_plasma', 'm³', use_symbol=False)
# → 'V_plasma [m³]'
```

### In Plots
All plot functions automatically use symbols:
- SHAP plots
- PDF plots
- KDE plots
- Parallel coordinates
- Contour/heatmap plots
- K-means clustering plots

No configuration needed - just generate plots normally!

## Adding New Symbols

Edit `ddstartup/utils/parameter_symbols.py`:

```python
PARAM_SYMBOLS = {
    'my_param': r'$\alpha_{\mathrm{custom}}$',
}
```

**LaTeX Tips:**
- Use `\mathrm{}` for text: `$V_{\mathrm{plasma}}$`
- Use raw letters for physics variables: `$T_i$`, `$n_D$`
- Greek letters: `$\tau$`, `$\eta$`, `$\alpha$`, `$\beta$`
- Subscripts: `_{text}` or `_{i}`
- Superscripts: `^{text}` or `^{2}`

## Files Modified

- **Created:**
  - `ddstartup/utils/parameter_symbols.py` - Symbol definitions
  - `docs/PLOT_SYMBOL_FORMATTING.md` - Full documentation
  - `test_plot_symbols.py` - Testing script

- **Updated:**
  - `ddstartup/postprocessing/plot_shap_functions.py`
  - `ddstartup/postprocessing/plot_pdf_functions.py`
  - `ddstartup/postprocessing/plot_kde_functions.py`
  - `ddstartup/postprocessing/plot_parcoords_functions.py`
  - `ddstartup/postprocessing/plot_contour_functions.py`
  - `ddstartup/postprocessing/plot_kmeans_functions.py`

## Testing

```bash
# Test symbol system
python test_plot_symbols.py

# Generate plots with symbols
python -m ddstartup.postprocessing outputs/your_data/
```

---

**✨ All plots now use publication-quality mathematical notation automatically!**
