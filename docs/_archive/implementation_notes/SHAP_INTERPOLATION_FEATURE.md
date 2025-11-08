# SHAP Plot Interpolation Feature

## Overview

The SHAP plotting function now supports **interpolated (smooth) density visualization** as an alternative to traditional scatter plots. This creates continuous, publication-ready plots similar to the official SHAP library.

## Feature Details

### Two Visualization Modes

1. **Scatter Mode** (default, `interpolate=False`)
   - Traditional beeswarm scatter plot
   - Each point represents a single data sample
   - Better for showing individual data points
   - Best for smaller datasets or when exact point locations matter

2. **Interpolated Mode** (`interpolate=True`)
   - Smooth continuous density visualization
   - Uses 2D histogram with Gaussian smoothing
   - More visually appealing and publication-ready
   - Better for large datasets
   - Shows density patterns more clearly

## Usage

### 1. Command Line Interface

```bash
# Enable interpolation via CLI flag
python -m ddstartup.postprocessing --shap-interpolate

# Or with config file override
python -m ddstartup.postprocessing postprocess_config.yaml --shap-interpolate
```

### 2. YAML Configuration File

Edit your `postprocess_config.yaml`:

```yaml
plots:
  shap: true
  
  shap_settings:
    interpolate: true  # Enable smooth interpolated plots
    max_samples: 100000000
    # ... other settings
```

### 3. Python API

```python
from ddstartup.postprocessing.plot_shap_functions import generate_shap_plots

# Generate interpolated SHAP plots
results = generate_shap_plots(
    df_filtered=df,
    target='unrealized_profits',
    input_parameters=['V_plasma', 'n_tot', 'tau_p_T'],
    target_unit='$',
    outputs_dir=Path('outputs'),
    file_type='lump',
    plot_name='my_shap_plot',
    interpolate=True  # Enable interpolation
)
```

## Technical Implementation

### Algorithm

1. **Density Computation**: Creates a 2D histogram (100 bins horizontal, 30 bins vertical)
2. **Smoothing**: Applies Gaussian filter with sigma=[2.0, 1.5] for smooth transitions
3. **Color Mapping**: Computes average feature value per horizontal bin
4. **Rendering**: Draws filled rectangles with alpha blending based on density

### Key Parameters

- **Horizontal bins** (`n_bins_x`): 100 - Controls effect resolution
- **Vertical bins** (`n_bins_y`): 30 - Controls vertical smoothness
- **Gaussian sigma**: [2.0, 1.5] - Controls smoothing intensity
- **Vertical spread** (`y_spread`): 0.5 - Height of density bands

## Interpretation Guide

### Color Coding (Same for Both Modes)

- 🔵 **Blue**: Low parameter values
- 🔴 **Red**: High parameter values
- **Gradient**: Intermediate values

### Position Interpretation

- **Right side (positive)**: Parameter increases the output
- **Left side (negative)**: Parameter decreases the output

### Example: `tau_p_T` and Unrealized Profits

If increasing `tau_p_T` **decreases** unrealized profits (negative correlation):
- ✅ **Correct**: Red dots/density on the LEFT (negative impact)
- ✅ **Correct**: Blue dots/density on the RIGHT (positive impact = less profit loss)

## Comparison: Scatter vs Interpolated

| Feature | Scatter | Interpolated |
|---------|---------|--------------|
| **Visual clarity** | Individual points visible | Smooth continuous patterns |
| **Publication quality** | Good | Excellent |
| **Large datasets** | Can become cluttered | Handles well |
| **Exact point locations** | Preserved | Smoothed (approximate) |
| **Density perception** | Requires inference | Immediately clear |
| **File size** | Larger (many points) | Smaller (rectangles) |
| **Computation time** | Faster | Slightly slower |

## Bug Fix (Included)

This update also **fixes a critical bug** where all parameters showed the same color-position pattern:
- **Old bug**: Used absolute correlation for effects → all parameters had same pattern
- **Fix**: Now uses **signed correlation** for effects → correctly shows direction

### Before Fix
All parameters: Blue left, Red right (incorrect for negative correlations)

### After Fix
- Positive correlation: Blue left, Red right ✅
- Negative correlation: **Red left, Blue right** ✅

## Configuration Priority

Settings are applied in this order (later overrides earlier):

1. Default values in code (`interpolate=False`)
2. YAML configuration file (`shap_settings.interpolate`)
3. Command-line flags (`--shap-interpolate`)

## Examples

### Quick Test

```bash
# Create test script
python test_shap_interpolate.py

# This generates both scatter and interpolated versions for comparison
```

### Production Use

```bash
# Process latest results with interpolation
python -m ddstartup.postprocessing postprocess_config.yaml --shap-interpolate

# Process specific file
python -m ddstartup.postprocessing --files outputs/my_results.h5 --plots shap --shap-interpolate
```

## Recommendations

### When to Use Scatter Mode
- Datasets < 500 samples
- Need to see exact point locations
- Debugging or exploratory analysis
- Discrete parameters with few unique values

### When to Use Interpolated Mode
- Datasets > 1000 samples
- Creating publication figures
- Presentations and reports
- Continuous parameters
- Emphasizing density patterns

## Files Modified

1. `ddstartup/postprocessing/plot_shap_functions.py`
   - Added `interpolate` parameter to functions
   - Implemented 2D histogram + Gaussian smoothing
   - Fixed signed correlation bug

2. `ddstartup/postprocessing/cli.py`
   - Added `--shap-interpolate` CLI flag
   - Added YAML config parsing for `shap_settings.interpolate`
   - Updated plot generation to pass parameter

3. `inputs/postprocess_config.yaml`
   - Added `interpolate: false` option under `shap_settings`
   - Added documentation comments

## Dependencies

No new dependencies required! Uses existing packages:
- `numpy` - Array operations
- `scipy.ndimage` - Gaussian filtering (already imported)
- `matplotlib` - Plotting

## Future Enhancements

Potential improvements for future versions:
1. Adjustable smoothing parameters via config
2. Different interpolation methods (KDE, spline)
3. Interactive HTML version with Plotly
4. Automatic mode selection based on dataset size
5. Side-by-side comparison plots

## References

- Original SHAP library: https://github.com/slundberg/shap
- Correlation-based feature importance documentation
- Gaussian smoothing: `scipy.ndimage.gaussian_filter`
