# Parameter Filtering - Quick Reference

## Add Filter to Config

```yaml
# your_config.yaml
analysis_type: T_seeded
method: parametric
filter: "YOUR_FILTER_HERE"  # ← Add this line
```

## Filter Syntax

### Operators

| Type | Operators | Example |
|------|-----------|---------|
| Comparison | `<` `<=` `>` `>=` `==` `!=` | `P_aux < 1e6` |
| Logical | `and` `or` `not` | `T_i > 10 and T_i < 25` |
| Arithmetic | `+` `-` `*` `/` `**` | `P_aux * 2 > 1e6` |
| Grouping | `( )` | `(A and B) or C` |

### Common Patterns

```yaml
# Single constraint
filter: "P_aux < 1e6"

# Range
filter: "T_i > 10 and T_i < 25"

# Compare parameters
filter: "P_aux_DT_eq < P_aux"

# Multiple conditions (AND)
filter: "P_aux < 1e6 and T_i > 10 and n_tot < 5e20"

# Multiple conditions (OR)
filter: "T_i < 10 or T_i > 25"

# Complex
filter: "(P_aux > 1e6 and T_i > 15) or (n_tot > 5e20)"

# Exclude range
filter: "not (T_i > 10 and T_i < 20)"
```

## Available Parameters

### T_seeded
`V_plasma`, `T_i`, `n_tot`, `tau_p_T`, `P_aux`, `P_aux_DT_eq`, `TBR_DT`, `TBR_DDn`, `tau_ifc`, `tau_ofc`, `eta_th`, `capacity_factor`, `price_of_electricity`

### Lump
Same as T_seeded, plus: `tau_p_He3`, `I_target`

## Usage

```bash
# With filter
python -m ddstartup params_test filtered_config

# No filter (default)
python -m ddstartup params_test standard_config

# Test filter without running
python -m ddstartup params_test filtered_config --dry-run
```

## Output

Console shows:
```
Filter expression: P_aux_DT_eq < P_aux
Total combinations before filtering: 1,000,000
Valid combinations after filtering: 234,567
Filter efficiency: 23.46%
```

HDF5 metadata includes:
- `filter_expression`
- `original_combinations`
- `filtered_combinations`
- `filter_efficiency`

## Tips

✅ Start simple, then refine
✅ Aim for 10-50% efficiency (50-90% reduction)
✅ Use `--dry-run` to test filters
✅ Check units match your parameter definitions
✅ Parameter names are case-sensitive

❌ Avoid: `filter: "P_Aux < 1e6"` (wrong case)
✅ Correct: `filter: "P_aux < 1e6"`

## Examples

### Focus on high-Q regime
```yaml
filter: "P_aux > 5e6 and T_i > 15"
```

### Economically viable cases
```yaml
filter: "eta_th > 0.35 and capacity_factor > 0.6"
```

### Physical constraint
```yaml
filter: "P_aux_DT_eq < P_aux"  # DD power exceeds DT equilibrium
```

### Operating window
```yaml
filter: "T_i > 10 and T_i < 25 and n_tot > 1e20 and n_tot < 5e20"
```

---

📖 Full documentation: `docs/PARAMETER_FILTERING_GUIDE.md`
🧪 Tests: `tests/test_filters.py`
💡 Example: `inputs/parametric_tseeded_filtered.yaml`
