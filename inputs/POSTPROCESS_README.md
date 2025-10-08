# DD Startup Postprocessing Guide

## Overview

The DD Startup postprocessing tool provides a flexible way to analyze and visualize results from parametric or Sobol analysis runs. It supports YAML configuration files for complex workflows and command-line arguments for quick tasks.

## Quick Start

### 1. Using YAML Configuration (Recommended)

```bash
# Process with default configuration
python -m ddstartup.postprocessing inputs/postprocess_config.yaml

# Process with quick config
python -m ddstartup.postprocessing inputs/postprocess_quick.yaml
```

### 2. Using Command-Line Only

```bash
# Process latest file with defaults
python -m ddstartup.postprocessing

# Process specific file
python -m ddstartup.postprocessing --files outputs/my_results.h5

# With filters
python -m ddstartup.postprocessing --input-filter "V_plasma<150" --output-filter "unrealized_gains>2e6"
```

## YAML Configuration Files

### Available Configuration Files

- **`postprocess_config.yaml`**: Complete configuration with all options and documentation
- **`postprocess_quick.yaml`**: Simplified configuration for quick tasks

### Configuration Structure

```yaml
# File selection
files: "latest"  # or specific path(s)

# Target variables
target_variables:
  - unrealized_gains
  - t_startup

# Filters
input_filters:
  V_plasma:
    min: 100
    max: 150
  
output_filters:
  unrealized_gains:
    min: 2.0e6
    max: null

# Plot types
plots:
  generate_all: true
  kde: true
  parcoords: true
  pdf: true

# Output settings
output:
  directory: "default"
  verbose: true
```

## Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `config` | - | YAML configuration file (positional) |
| `--files` | `-f` | HDF5 file(s) to process |
| `--targets` | `-t` | Target variables to analyze |
| `--input-filter` | `-if` | Filter input parameters |
| `--output-filter` | `-of` | Filter output variables |
| `--plots` | `-p` | Plot types (kde/parcoords/pdf/all) |
| `--output-dir` | `-o` | Output directory |

## Filter Syntax

### YAML Format
```yaml
input_filters:
  parameter_name:
    min: value  # minimum value (or null)
    max: value  # maximum value (or null)
```

### Command-Line Format
```bash
--input-filter "param1<value1,param2>value2,param3>=value3"
--output-filter "output1>value1,output2<=value2"
```

**Supported operators**: `<`, `>`, `<=`, `>=`, `==`

## Plot Types

### 1. KDE (Kernel Density Estimation)
- Quartile-based distribution plots for each input parameter
- Shows how parameter values are distributed across different output quartiles
- Saved as PNG files
- Also generates CSV files with quartile extremes

### 2. Parallel Coordinates
- Interactive Plotly visualization
- Shows relationships between multiple parameters and outputs
- Color-coded by target variable quartiles
- Saved as HTML files (open in web browser)

### 3. PDF (Probability Density Function)
- Statistical distribution of target variables
- Useful for comparing different runs
- Saved as PNG files with logarithmic x-axis

## Examples

### Example 1: High-Cost Scenarios Only

**YAML** (`inputs/high_cost.yaml`):
```yaml
files: "latest"
target_variables: [unrealized_gains]
output_filters:
  unrealized_gains:
    min: 5.0e6
    max: null
plots:
  generate_all: false
  kde: true
  parcoords: true
```

**Command:**
```bash
python -m ddstartup.postprocessing inputs/high_cost.yaml
```

### Example 2: Quick Startup Scenarios

**Command-line only:**
```bash
python -m ddstartup.postprocessing \
  --targets t_startup \
  --output-filter "t_startup<3.154e7" \
  --input-filter "V_plasma<100" \
  --plots kde
```

### Example 3: Compare Multiple Files

**YAML** (`inputs/compare_methods.yaml`):
```yaml
files:
  - "outputs/lump_results.h5"
  - "outputs/tseeded_results.h5"
target_variables: [unrealized_gains, t_startup]
plots:
  generate_all: false
  pdf: true
output:
  directory: "comparison_plots"
```

**Command:**
```bash
python -m ddstartup.postprocessing inputs/compare_methods.yaml
```

### Example 4: Override YAML with Command-Line

```bash
# Use config but override plot types
python -m ddstartup.postprocessing postprocess_config.yaml --plots kde

# Use config but change output directory
python -m ddstartup.postprocessing postprocess_config.yaml --output-dir custom_plots/
```

## Output Files

### File Naming Convention

```
{plot_type}_{filename}_{target}.{extension}

Examples:
- kde_quartiles_dd_startup_20250927_parametric_lump_unrealized_gains.png
- parcoords_dd_startup_20250927_parametric_tseeded_t_startup.html
- pdf_dd_startup_20250927_parametric_lump_t_startup.png
- quartile_unrealized_gains_values_dd_startup_20250927_parametric_lump.csv
```

### Default Output Location

- **Default**: `ddstartup/postprocess/`
- **Custom**: Set in YAML (`output.directory`) or command-line (`--output-dir`)

## Tips and Best Practices

1. **Use YAML for complex workflows**: Easier to manage filters and settings
2. **Start with `postprocess_quick.yaml`**: Modify for your needs
3. **Use command-line for quick tasks**: Override specific settings
4. **Filter early**: Apply filters to reduce data size and processing time
5. **Check CSV files**: Quartile CSVs show extreme parameter combinations
6. **Interactive plots**: Parallel coordinates HTML files are interactive in browser

## Troubleshooting

### PyYAML not found
```bash
pip install pyyaml
```

### No HDF5 files found
- Check that `outputs/` directory exists
- Verify HDF5 files have `.h5` extension
- Use `--files` to specify exact path

### Memory issues with large files
- Apply stricter filters to reduce data
- Process one target variable at a time
- Reduce `parcoords_settings.max_samples` in YAML

### Plots not generated
- Check that matplotlib and plotly are installed
- Verify output directory is writable
- Check console output for specific error messages

## Advanced Usage

### Custom Parameter Order

Edit `utils/postprocess.py` and modify `DESIRED_ORDER` in `get_input_parameters()`:

```python
DESIRED_ORDER = [
    'V_plasma', 'n_tot', 'T_i', ...
]
```

### Custom Color Scales

Modify `get_discrete_colorscale()` in `postprocess/postprocess_functions.py`:

```python
base_colors = [
    "#00FF00",  # green
    "#FFFF00",  # yellow
    "#FF0000",  # red
]
```

### Batch Processing

Create a shell script to process multiple configurations:

```bash
#!/bin/bash
for config in inputs/postprocess_*.yaml; do
    echo "Processing $config..."
    python -m ddstartup.postprocessing "$config"
done
```

## See Also

- **Main analysis tool**: `python -m ddstartup.main`
- **Configuration examples**: `inputs/` directory
- **Plot customization**: `ddstartup/postprocess/` folder
