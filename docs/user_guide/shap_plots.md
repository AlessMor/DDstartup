# SHAP Plot Guide

## Overview

SHAP (SHapley Additive exPlanations) plots provide a unified way to understand how each input parameter affects the output predictions. Based on cooperative game theory, SHAP values show:

- **Feature Importance**: Which inputs have the strongest effect on the output
- **Direction of Effect**: Whether increasing an input increases or decreases the output
- **Interaction Effects**: How inputs interact with each other
- **Individual Predictions**: Why specific parameter combinations produce certain outputs

## What is SHAP?

SHAP values decompose a prediction into contributions from each feature. For a prediction $f(x)$:

$$f(x) = \phi_0 + \sum_{i=1}^{n} \phi_i$$

Where:
- $\phi_0$ is the base value (average prediction)
- $\phi_i$ is the SHAP value for feature $i$
- Higher $|\phi_i|$ means feature $i$ has more impact

## Installation

```bash
conda activate ddstartupenv
pip install shap
```

## Usage

### Command Line

```bash
# Generate all plot types including SHAP
python -m ddstartup.postprocessing postprocess_config.yaml

# Generate only SHAP plots
python -m ddstartup.postprocessing --plots shap

# Generate SHAP for specific target
python -m ddstartup.postprocessing --targets t_startup --plots shap
```

### YAML Configuration

```yaml
plots:
  generate_all: true  # Includes SHAP
  
  # Or specify individually
  shap: true
  
  shap_settings:
    model_type: tree        # 'tree', 'linear', or 'kernel'
    max_samples: 1000       # Performance vs accuracy tradeoff
    plot_types:
      - summary             # Main SHAP plot (beeswarm)
      - summary_bar         # Feature importance bars
      - dependence          # Feature effect curves
      - csv                 # Export SHAP values
```

## Plot Types

### 1. SHAP Summary Plot (Beeswarm)
**File**: `shap_<filename>_<target>_summary_dot.png`

Shows the distribution of SHAP values for each feature:
- **Y-axis**: Features ranked by importance (top = most important)
- **X-axis**: SHAP value (impact on output)
- **Color**: Feature value (pink = high, blue = low)
- **Dots**: Individual data points

**Interpretation**:
- Wide spread = feature has variable impact
- Positive values = feature increases output
- Negative values = feature decreases output
- Color patterns reveal interactions

**Example**: If pink dots (high V_plasma) are on the right, increasing V_plasma increases the output.

### 2. SHAP Summary Bar Plot
**File**: `shap_<filename>_<target>_summary_bar.png`

Shows mean absolute SHAP values:
- **Y-axis**: Features ranked by importance
- **X-axis**: Mean |SHAP value|
- **Interpretation**: Longer bar = more important feature overall

### 3. SHAP Dependence Plots
**Files**: `shap_<filename>_<target>_dependence_<feature>.png`

Created for top 3 most important features:
- **X-axis**: Feature value
- **Y-axis**: SHAP value for that feature
- **Color**: Interaction feature (automatically detected)

**Interpretation**:
- Linear trend = linear relationship
- Non-linear curve = complex relationship
- Color patterns = interaction effects

### 4. SHAP Values CSV
**Files**: 
- `shap_<filename>_<target>_shap_values.csv`: Full SHAP values for each sample
- `shap_<filename>_<target>_shap_importance.csv`: Feature importance ranking

**Columns in shap_values.csv**:
- `shap_<param>`: SHAP value for each parameter
- `value_<param>`: Actual parameter value
- `shap_total`: Sum of all SHAP contributions

**Columns in shap_importance.csv**:
- `feature`: Parameter name
- `mean_abs_shap`: Average absolute SHAP value
- `importance_rank`: Ranking (1 = most important)

## Surrogate Models

SHAP requires a model to explain. We train a "surrogate" model on your simulation data:

### Tree-Based (Default)
```yaml
model_type: tree
```
- **Model**: Gradient Boosting Regressor
- **Pros**: Fast, accurate for complex nonlinear relationships
- **Cons**: Can overfit on small datasets
- **Best for**: Most cases, especially with >500 samples

### Linear
```yaml
model_type: linear
```
- **Model**: Ridge Regression
- **Pros**: Very fast, interpretable
- **Cons**: Assumes linear relationships
- **Best for**: Quick analysis, linear physics relationships

### Kernel (Experimental)
```yaml
model_type: kernel
```
- **Model**: Model-agnostic KernelExplainer
- **Pros**: Most general, no assumptions
- **Cons**: Very slow (minutes to hours)
- **Best for**: When tree/linear models have poor R² score

## Interpreting Results

### Surrogate Model Quality

The script reports R² score:
```
Surrogate model R² score: 0.8523
```

- **R² > 0.9**: Excellent - SHAP values highly reliable
- **R² > 0.7**: Good - SHAP values reasonably accurate
- **R² < 0.7**: Warning - consider different model_type or check data
- **R² < 0.5**: Poor - SHAP interpretation may be misleading

### Common Patterns

1. **Linear Positive Effect**: 
   - Dependence plot shows upward slope
   - High values → positive SHAP → increases output

2. **Threshold Effect**:
   - Dependence plot shows step/jump
   - Feature has no effect until threshold crossed

3. **Interaction**:
   - Dependence plot colored by interaction feature
   - Different colors show different slopes
   - Features work together

4. **Diminishing Returns**:
   - Dependence plot levels off
   - Feature important initially but saturates

## Performance Tips

### Speed vs Accuracy

```yaml
shap_settings:
  max_samples: 100    # Fast but less accurate
  max_samples: 1000   # Balanced (default)
  max_samples: 5000   # Slow but comprehensive
```

Computation time scales as O(n_samples × n_features)

### Large Datasets

For >10,000 samples:
1. Use `max_samples: 1000` or less
2. Use `model_type: tree` (fastest explainer)
3. Consider filtering data first

### Memory Issues

If encountering memory errors:
```yaml
shap_settings:
  max_samples: 500     # Reduce samples
  model_type: linear   # Use simpler model
```

## Examples

### Identify Most Important Parameter

1. Generate SHAP plots:
   ```bash
   python -m ddstartup.postprocessing --targets unrealized_profits --plots shap
   ```

2. Open `shap_..._summary_bar.png`

3. Top feature = most important parameter

### Understand Parameter Effect

1. Generate SHAP plots

2. Open `shap_..._summary_dot.png`

3. Find your parameter of interest:
   - Pink dots on right = high value increases output
   - Blue dots on right = low value increases output
   - Pink dots on left = high value decreases output

### Find Optimal Parameter Range

1. Generate SHAP plots with dependence

2. Open `shap_..._dependence_<param>.png`

3. Identify where SHAP value is most negative (minimizes output) or most positive (maximizes output)

### Compare Multiple Scenarios

1. Apply filters to isolate scenarios:
   ```yaml
   input_filters:
     V_plasma: {min: 50, max: 100}
   ```

2. Generate SHAP plots for filtered data

3. Compare importance rankings

## Comparison with Other Methods

| Method | What It Shows | When to Use |
|--------|--------------|-------------|
| **SHAP** | Exact feature contributions | Understanding individual predictions, interactions |
| **Importance Matrix** | Effect size per quartile | Comparing groups, finding thresholds |
| **KDE Plots** | Distribution of inputs by output | Visual exploration, identifying patterns |
| **Parallel Coordinates** | Multi-dimensional relationships | Finding optimal parameter combinations |
| **Dependence Plots** | Feature vs output relationship | Same as SHAP dependence but model-free |

**Use SHAP when**:
- You need quantitative feature importance
- You want to understand interactions
- You need to explain specific predictions
- You have enough data (>100 samples)

**Use other plots when**:
- You want model-free analysis (KDE, parallel coords)
- You want quartile-based comparisons (importance matrix)
- SHAP computation is too slow

## Troubleshooting

### SHAP Not Installed
```
❌ SHAP library not installed. Install it with: pip install shap
```
**Solution**: `conda activate ddstartupenv && pip install shap`

### Low R² Score Warning
```
⚠️  Warning: Low R² score (0.45). SHAP values may not accurately represent...
```
**Solutions**:
1. Try different `model_type`: tree → linear or vice versa
2. Check for outliers in data
3. Apply filters to clean data
4. Consider if relationships are too complex

### Memory Error
```
MemoryError: Unable to allocate array
```
**Solutions**:
1. Reduce `max_samples` in config
2. Filter data to reduce total samples
3. Use `model_type: linear` (lower memory)

### Slow Computation
SHAP taking >5 minutes:
**Solutions**:
1. Reduce `max_samples` to 500 or less
2. Use `model_type: tree` (fastest)
3. Consider running on specific targets only

## References

- SHAP Documentation: https://shap.readthedocs.io/
- Original Paper: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (2017)
- Tree SHAP: Lundberg et al., "Consistent Individualized Feature Attribution for Tree Ensembles" (2018)
