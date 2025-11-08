# YAML Parameter Configuration Guide

## Overview

The DD Startup simulation tool now supports YAML-based parameter configuration files as the **recommended approach** for defining simulation parameters. YAML provides better separation of configuration from code, improved readability, and easier maintenance.

## Why YAML?

**Advantages over Python parameter files:**
- **Clean separation**: Configuration data separate from executable code
- **Security**: No code execution risks from parameter files
- **Readability**: More intuitive syntax for non-programmers
- **Version control**: Cleaner diffs in git
- **Validation**: Easier to validate structure programmatically
- **Portability**: Can be used by other tools without Python interpreter

**Legacy Python files (.py) are still supported** for backward compatibility.

## File Structure

```yaml
parameters:
  <parameter_name>:
    type: <scalar|linear|normal|vector>
    # Type-specific fields (see below)
    unit: <unit_string>
    description: <optional description>
  
  total_time:
    value: <simulation_time_in_seconds>
    unit: s
    description: Total simulation time
```

## Parameter Types

### 1. Scalar (Constant Value)

Use when a parameter has a single fixed value:

```yaml
eta_th_field:
  type: scalar
  value: 0.35
  unit: dimensionless
  description: Thermal conversion efficiency
```

**Special value for automatic calculation:**
```yaml
P_aux_field:
  type: scalar
  value: .nan  # Will be calculated from power balance
  unit: MW
  description: Auxiliary heating power
```

### 2. Linear (Linearly Spaced Range)

Use for uniform sampling between min and max:

```yaml
T_i_field:
  type: linear
  min: 14
  max: 20
  points: 5
  unit: keV
  description: Ion temperature
```

Generates: `[14.0, 15.5, 17.0, 18.5, 20.0]`

### 3. Normal (Normal Distribution Sampling)

Use for statistical parameter variation:

```yaml
V_plasma_field:
  type: normal
  mean: 150
  std: 15
  points: 5
  unit: m^3
  description: Plasma volume
```

Generates 5 points sampled from normal distribution with mean=150, std=15.

### 4. Vector (Custom Values)

Use when you need specific custom values:

```yaml
tau_p_T_field:
  type: vector
  values: [0.1, 0.5, 1.0, 2.5, 5.0]
  unit: s
  description: Tritium particle confinement time
```

## Supported Units

### Physical Units
- **Volume**: `m^3`, `m3`
- **Temperature**: `keV`, `eV`
- **Density**: `1/m^3`, `1/m3`, `m^-3`, `m-3`
- **Time**: `s`, `second`, `seconds`
- **Power**: `W`, `watt`, `MW`, `GW`
- **Mass**: `kg`

### Economic Units
- **Price of electricity**: `1/J`, `$/J`

### Dimensionless
- `dimensionless`, `1`, `none`

### Scientific Notation

YAML supports scientific notation, but use strings for safety:

```yaml
n_tot_field:
  type: linear
  min: 1.3e20
  max: 2.1e20
  unit: 1/m^3
```

## Required Parameters

### For T_seeded Analysis
- `V_plasma_field`, `T_i_field`, `n_tot_field`, `tau_p_T_field`
- `P_aux_field`, `P_aux_DT_eq_field`
- `TBR_DT_field`, `TBR_DDn_field`
- `tau_ifc_field`, `tau_ofc_field`
- `eta_th_field`, `capacity_factor_field`, `price_of_electricity_field`

### For Lump Analysis
- Same as T_seeded, plus:
- `tau_p_He3_field`
- `I_target_field`

## Usage

### Command Line

```bash
# Using YAML parameter file
python -m ddstartup params.yaml parametric_tseeded.yaml

# Extension is optional (will be detected automatically)
python -m ddstartup params parametric_tseeded
```

### File Priority

When resolving parameter files, the system searches in this order:
1. `.yaml` extension
2. `.yml` extension  
3. `.py` extension (legacy)

So if you have both `params.yaml` and `params.py`, the YAML file takes precedence.

## Examples

### Example 1: Basic Parametric Study

**File: `inputs/params_basic.yaml`**
```yaml
parameters:
  V_plasma_field:
    type: linear
    min: 100
    max: 200
    points: 3
    unit: m^3
    description: Plasma volume sweep
  
  T_i_field:
    type: linear
    min: 15
    max: 20
    points: 3
    unit: keV
    description: Ion temperature sweep
  
  # ... other parameters with type: scalar for constants
  
  eta_th_field:
    type: scalar
    value: 0.35
    unit: dimensionless
```

This creates 3×3 = 9 combinations for V_plasma and T_i, with other parameters constant.

### Example 2: Automatic P_aux Calculation

**File: `inputs/params_noPaux.yaml`**
```yaml
parameters:
  # ... plasma parameters ...
  
  P_aux_field:
    type: scalar
    value: .nan  # Calculate from power balance
    unit: MW
  
  P_aux_DT_eq_field:
    type: scalar
    value: .nan  # Calculate from power balance
    unit: MW
```

When `P_aux` is set to `.nan`, the code automatically calculates it using:
```
P_aux = P_rad + 3*n_tot*T_i*V_plasma - P_charged
```

### Example 3: Statistical Uncertainty Study

**File: `inputs/params_uncertainty.yaml`**
```yaml
parameters:
  V_plasma_field:
    type: normal
    mean: 150
    std: 15
    points: 10
    unit: m^3
    description: Plasma volume with uncertainty
  
  T_i_field:
    type: normal
    mean: 17
    std: 2
    points: 10
    unit: keV
    description: Ion temperature with uncertainty
  
  price_of_electricity_field:
    type: normal
    mean: 6.944e-08  # 0.25 $/kWh
    std: 2.778e-08   # 0.10 $/kWh
    points: 10
    unit: 1/J
    description: Price with market uncertainty
```

### Example 4: Custom Time Points

**File: `inputs/params_custom.yaml`**
```yaml
parameters:
  tau_ifc_field:
    type: vector
    values: [1800, 3600, 7200, 14400, 28800]  # 0.5h, 1h, 2h, 4h, 8h
    unit: s
    description: Custom in-fuel cycle times
  
  tau_ofc_field:
    type: vector
    values: [86400, 172800, 259200]  # 1, 2, 3 days
    unit: s
    description: Custom out-of-fuel cycle times
```

## Converting Python Files to YAML

To convert an existing `.py` parameter file to YAML:

1. **Scalar parameters** (`parametrization_type="scalar"`):
   ```python
   # Python
   field = ParameterField(parametrization_type="scalar", mean=0.35, ...)
   ```
   ```yaml
   # YAML
   field:
     type: scalar
     value: 0.35
   ```

2. **Linear parameters** (`parametrization_type="linear"`):
   ```python
   # Python
   field = ParameterField(parametrization_type="linear", min_val=10, max_val=20, param_points=5, ...)
   ```
   ```yaml
   # YAML
   field:
     type: linear
     min: 10
     max: 20
     points: 5
   ```

3. **Normal parameters** (`parametrization_type="normal"`):
   ```python
   # Python
   field = ParameterField(parametrization_type="normal", mean=150, std=15, param_points=5, ...)
   ```
   ```yaml
   # YAML
   field:
     type: normal
     mean: 150
     std: 15
     points: 5
   ```

4. **Vector parameters** (`parametrization_type="vector"`):
   ```python
   # Python
   field = ParameterField(parametrization_type="vector", vector=[1, 2, 3], ...)
   ```
   ```yaml
   # YAML
   field:
     type: vector
     values: [1, 2, 3]
   ```

5. **Units**:
   - `u.m**3` → `m^3`
   - `u.keV` → `keV`
   - `1/u.m**3` → `1/m^3`
   - `1/u.kWh` → Convert to `1/J` (multiply by 2.778e-7)

## Best Practices

1. **Use descriptive names**: Include units and context in descriptions
2. **Start simple**: Begin with scalar parameters, add sweeps as needed
3. **Document units**: Always specify units clearly
4. **Version control**: Commit YAML files to track parameter changes
5. **Validate first**: Run with `--dry-run` flag to check configuration
6. **Group related parameters**: Use comments to organize sections

## Troubleshooting

### Error: "Unknown unit 'X'"
- Check spelling against supported units list
- Try using pint syntax (e.g., `meter ** 3` instead of `m^3`)

### Error: "min_val must be a float"
- Scientific notation in YAML might be parsed as string
- Should work with notation like `1.3e20`

### Parameter not found
- Ensure parameter name ends with `_field`
- Check for typos in parameter names
- Verify indentation (YAML is whitespace-sensitive)

### Values not as expected
- Check `points` parameter matches desired sample count
- For vector type, `points` is automatically set to length of `values` array
- Normal distribution samples from percentiles, not random

## Migration Path

1. **Phase 1** (Current): Both .py and .yaml files supported
2. **Phase 2** (Recommended): Move all parameter files to YAML
3. **Phase 3** (Future): Consider deprecating .py parameter files

For now, you can use both formats simultaneously in your project.

## See Also

- `docs/QUICK_REFERENCE.rst` - General tool usage
- `inputs/params.yaml` - Main parameter file example
- `inputs/params_test.yaml` - Small test configuration
- `inputs/params_noPaux.yaml` - Automatic P_aux calculation example
