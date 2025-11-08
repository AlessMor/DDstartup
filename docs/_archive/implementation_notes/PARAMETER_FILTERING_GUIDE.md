# Parameter Filtering Guide

## Overview

The DD Startup analysis tool now supports **parameter filtering** to reduce the number of combinations computed. This allows you to apply constraints BEFORE running expensive computations, saving time and resources.

## Key Benefits

- **Faster Analysis**: Skip invalid or uninteresting parameter combinations
- **Focused Results**: Only compute physically meaningful cases
- **Resource Efficient**: Reduce computation time and storage requirements
- **Flexible Constraints**: Support for complex logical expressions

## How It Works

1. **Define Filter**: Add a `filter` expression to your YAML configuration
2. **Evaluation**: The filter is evaluated for all parameter combinations BEFORE computation
3. **Selection**: Only combinations that pass the filter are computed
4. **Output**: Results include filter metadata (expression, efficiency, etc.)

## Filter Syntax

### Basic Comparisons

```yaml
filter: "P_aux < 1e6"              # Auxiliary power less than 1 MW
filter: "T_i >= 10"                # Ion temperature at least 10 keV
filter: "n_tot == 1e20"            # Exact density match
filter: "tau_p_T != 0.1"           # Confinement time not equal to 0.1s
```

### Comparing Parameters

```yaml
filter: "P_aux_DT_eq < P_aux"      # DT equilibrium power less than DD power
filter: "tau_ifc > tau_ofc"        # In-fuel-cycle time > out-fuel-cycle time
filter: "V_plasma >= 100"          # Minimum plasma volume
```

### Logical Operators

#### AND - All conditions must be true
```yaml
filter: "P_aux < 1e6 and T_i > 10"
filter: "V_plasma > 100 and n_tot < 1e20 and tau_p_T > 0.5"
```

#### OR - At least one condition must be true
```yaml
filter: "T_i < 5 or T_i > 25"
filter: "V_plasma < 50 or V_plasma > 200"
```

#### NOT - Negation
```yaml
filter: "not (T_i > 10 and T_i < 20)"  # Exclude 10-20 keV range
```

### Complex Expressions

Use parentheses for grouping:

```yaml
# High power OR high density scenarios
filter: "(P_aux > 1e6 and T_i > 15) or (n_tot > 5e20)"

# Exclude specific ranges
filter: "not ((T_i > 10 and T_i < 15) or (n_tot > 1e21))"

# Multiple constraints
filter: "(V_plasma > 100 and n_tot < 1e20) or (T_i > 15 and T_i < 25 and P_aux < 1e6)"
```

### Range Constraints

```yaml
# Temperature range
filter: "T_i > 10 and T_i < 25"

# Multiple ranges
filter: "(T_i > 5 and T_i < 10) or (T_i > 20 and T_i < 30)"
```

## Available Parameters

### T_seeded Analysis
- `V_plasma` - Plasma volume (m³)
- `T_i` - Ion temperature (keV)
- `n_tot` - Total particle density (m⁻³)
- `tau_p_T` - Tritium confinement time (s)
- `P_aux` - Auxiliary power (W)
- `P_aux_DT_eq` - DT equilibrium auxiliary power (W)
- `TBR_DT` - D-T tritium breeding ratio
- `TBR_DDn` - DD neutron tritium breeding ratio
- `tau_ifc` - In-fuel-cycle time (s)
- `tau_ofc` - Out-fuel-cycle time (s)
- `eta_th` - Thermal efficiency
- `capacity_factor` - Plant capacity factor
- `price_of_electricity` - Cost per Joule ($/J)

### Lump Analysis
Same as T_seeded, but with:
- `tau_p_He3` instead of fuel cycle times
- `I_target` - Target inventory (kg)

## Examples

### Example 1: Power Constraint

Only analyze cases where DD power exceeds DT equilibrium power:

```yaml
# parametric_tseeded_filtered.yaml
analysis_type: T_seeded
method: parametric
filter: "P_aux > P_aux_DT_eq"
```

### Example 2: Operating Window

Focus on a specific operating regime:

```yaml
filter: "T_i > 10 and T_i < 20 and n_tot > 1e20 and n_tot < 5e20"
```

### Example 3: High Performance Cases

Only high-Q scenarios:

```yaml
filter: "P_aux > 5e6 and T_i > 15 and tau_p_T > 1.0"
```

### Example 4: Exclude Unphysical Cases

Skip combinations that don't make physical sense:

```yaml
filter: "(V_plasma > 50 and n_tot < 1e21) and (tau_p_T > 0.1) and (P_aux < 1e7)"
```

### Example 5: Economic Constraints

Focus on economically viable cases:

```yaml
filter: "eta_th > 0.3 and capacity_factor > 0.5 and price_of_electricity < 2e-7"
```

## Usage

### Command Line

```bash
# Use a pre-configured filter
python -m ddstartup params_test parametric_tseeded_filtered

# Standard configuration (no filtering)
python -m ddstartup params_test parametric_tseeded
```

### Creating Your Own Filtered Config

1. Copy an existing YAML config:
```bash
cp inputs/parametric_tseeded.yaml inputs/my_filtered_analysis.yaml
```

2. Add the `filter` line:
```yaml
analysis_type: T_seeded
method: parametric
filter: "YOUR_FILTER_EXPRESSION_HERE"
# ... rest of config
```

3. Run the analysis:
```bash
python -m ddstartup params_test my_filtered_analysis
```

## Output

### Console Output

When filtering is applied, you'll see:

```
==============================================================
APPLYING PARAMETER FILTER
==============================================================
Filter expression: P_aux_DT_eq < P_aux
Total combinations before filtering: 1,000,000
Valid combinations after filtering: 234,567
Filter efficiency: 23.46%
Combinations excluded: 765,433
==============================================================

==============================================================
STARTING PARAMETRIC ANALYSIS
==============================================================
Original combinations: 1,000,000
Filtered combinations: 234,567
Filter reduction: 76.5%
...
```

### HDF5 Metadata

The output file includes filter information:

```python
import h5py

with h5py.File('output.h5', 'r') as f:
    print(f"Filter: {f.attrs['filter_expression']}")
    print(f"Original combinations: {f.attrs['original_combinations']}")
    print(f"Filtered combinations: {f.attrs['filtered_combinations']}")
    print(f"Efficiency: {f.attrs['filter_efficiency']:.2f}%")
```

## Performance Tips

### 1. Filter Early
Apply restrictive filters to reduce computation time significantly:
```yaml
# 90% reduction - huge time savings!
filter: "T_i > 15 and tau_p_T > 1.0"
```

### 2. Use Simple Expressions
Simple filters evaluate faster:
```yaml
# Fast
filter: "P_aux < 1e6"

# Slower (but still efficient)
filter: "(P_aux < 1e6 and T_i > 10) or (n_tot > 1e20 and V_plasma < 100)"
```

### 3. Combine with Parameter Ranges
Start with focused parameter ranges AND filters:
```python
# params_test.py
V_plasma_field = ParameterField(
    np.linspace(100, 200, 50),  # Focused range
    ureg('m^3')
)
```

```yaml
# config.yaml
filter: "T_i > 15"  # Additional constraint
```

## Validation

The filter system includes built-in validation:

### Syntax Errors
```yaml
filter: "P_aux < 1e6 and"  # ERROR: Incomplete expression
```

### Unknown Parameters
```yaml
filter: "P_fusion > 1e6"  # ERROR: 'P_fusion' not in parameter list
```

### Type Errors
```yaml
filter: "P_aux + 'string'"  # ERROR: Invalid operation
```

## Best Practices

1. **Test Filters First**: Use `--dry-run` to validate your filter:
   ```bash
   python -m ddstartup params_test my_filtered_config --dry-run
   ```

2. **Start Conservative**: Begin with simple filters and refine:
   ```yaml
   # Start simple
   filter: "T_i > 10"
   
   # Refine as needed
   filter: "T_i > 10 and T_i < 25 and P_aux < 1e6"
   ```

3. **Document Your Filters**: Add comments explaining the rationale:
   ```yaml
   # Only consider high-temperature regime for ignition studies
   filter: "T_i > 15 and P_aux > 5e6"
   ```

4. **Check Efficiency**: Aim for 10-50% efficiency (50-90% reduction):
   - Too high (>90%): Filter might not be restrictive enough
   - Too low (<1%): Filter might be too restrictive

5. **Validate Results**: Compare filtered vs. unfiltered results to ensure filter doesn't exclude interesting physics

## Troubleshooting

### Problem: Filter excludes all combinations
**Solution**: Relax constraints or check parameter units/scales

### Problem: Filter is too slow
**Solution**: Simplify expression or reduce parameter grid size first

### Problem: Unexpected results
**Solution**: Check parameter names match exactly (case-sensitive!)

## Advanced: Multiple Analysis Workflows

### Workflow 1: Exploratory (No Filter)
```bash
# Broad parameter sweep
python -m ddstartup params_broad parametric_tseeded
```

### Workflow 2: Refined (With Filter)
```bash
# Focus on interesting regime found in step 1
python -m ddstartup params_focused parametric_tseeded_filtered
```

### Workflow 3: Sensitivity (Filtered Sobol)
```bash
# Note: Filtering not yet implemented for Sobol analysis
# Coming in future update!
```

## Implementation Details

For developers interested in the internal implementation:

- Filter expressions are parsed as Python AST (Abstract Syntax Tree)
- Evaluation uses safe operators (no `eval()` or arbitrary code execution)
- Filtering happens before parameter meshgrid creation
- Filtered arrays are flattened and aligned by index
- Special indexing mode handles filtered vs. grid-based data transparently

## Future Enhancements

Planned improvements:
- [ ] Filter support for Sobol/LHS analysis
- [ ] Filter function library (common patterns)
- [ ] Runtime filter modification
- [ ] Filter combination/chaining
- [ ] Statistical filter summary in postprocessing

---

For questions or issues with filtering, please open an issue on GitHub or consult the main documentation.
