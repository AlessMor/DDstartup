"""
SHAP-style Feature Importance Plot Functions

This module provides functions for generating SHAP-style beeswarm plots to show
the effect of all input parameters on a chosen output variable.

Unlike traditional SHAP which requires a surrogate model, this implementation
computes feature importance directly from the HDF5 data using correlation-based
analysis, creating visualizations that show:
- How each input parameter affects the output
- The magnitude and direction of each parameter's effect  
- Distribution of feature values colored by their contribution
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import warnings

from ..utils.parameter_symbols import get_param_label, get_param_symbol

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def compute_feature_importance(df, input_parameters, target):
    """
    Compute feature importance using correlation with target.
    
    This gives us a measure of how much each input affects the output.
    Parameters with zero variance (constant values) get zero importance.
    
    Args:
        df: DataFrame with input and output data
        input_parameters: List of input parameter names
        target: Target output variable name
    
    Returns:
        Tuple of (importance_array, correlation_array)
        - importance: Array of importance scores (absolute correlation with target)
        - correlation: Array of signed correlation values (preserves direction)
    """
    importance = np.zeros(len(input_parameters))
    correlation = np.zeros(len(input_parameters))
    
    for i, param in enumerate(input_parameters):
        # Skip parameters with zero variance (constant values)
        if df[param].std() == 0:
            importance[i] = 0.0
            correlation[i] = 0.0
            continue
        
        # Compute correlation between input and output
        corr = np.corrcoef(df[param].values, df[target].values)[0, 1]
        
        # Handle NaN (can occur if target also has zero variance)
        if np.isnan(corr):
            importance[i] = 0.0
            correlation[i] = 0.0
        else:
            # Store both absolute value (for ranking) and signed value (for direction)
            importance[i] = np.abs(corr)
            correlation[i] = corr
    
    return importance, correlation


def normalize_to_range(values):
    """
    Normalize values to [0, 1] range for coloring.
    
    Args:
        values: Array of values to normalize
    
    Returns:
        Normalized values in [0, 1] range
    """
    if values.std() > 0:
        return (values - values.min()) / (values.max() - values.min())
    else:
        return np.ones_like(values) * 0.5


def create_shap_style_beeswarm_plot(df, input_parameters, target, target_unit,
                                    outputs_dir, plot_name, max_display=20, 
                                    max_samples=2000, interpolate=False):
    """
    Create a SHAP-style beeswarm plot showing feature importance.
    
    This creates a visualization similar to SHAP's summary_plot but computed
    directly from data without requiring a trained model. It shows:
    - Features ordered by importance (correlation with target)
    - Each dot is a sample, colored by feature value (blue=low, red=high)
    - Horizontal position shows impact: correlation × standardized(value)
    
    The horizontal spread IS standardized (divided by std dev), making features
    comparable per unit of variation. This ensures that features with different
    scales or discrete vs continuous distributions are fairly compared.
    
    Note: Constant features (zero variance) are automatically excluded.
    
    Args:
        df: DataFrame with input and output data
        input_parameters: List of input parameter names  
        target: Target output variable name
        target_unit: Unit string for target variable
        outputs_dir: Directory to save plot
        plot_name: Base name for saved plot
        max_display: Maximum number of features to display (default: 20)
        max_samples: Maximum samples to plot per feature (default: 2000)
        interpolate: If True, create smooth density-based plot instead of scatter (default: False)
    
    Returns:
        Dictionary with feature importance statistics
    """
    if len(input_parameters) == 0:
        print(f"   No input parameters found. Skipping SHAP-style plot.")
        return None
    
    print(f"   Computing feature importance for {target}...")
    
    # Filter out constant parameters (zero variance)
    varying_params = []
    for param in input_parameters:
        if df[param].std() > 0:
            varying_params.append(param)
        else:
            print(f"   Excluding constant parameter: {param} (std=0)")
    
    if len(varying_params) == 0:
        print(f"   No varying input parameters found. Skipping SHAP plot.")
        return None
    
    # Compute feature importance (correlation-based)
    importance, correlations = compute_feature_importance(df, varying_params, target)
    
    # Sort features by importance
    sorted_indices = np.argsort(importance)[::-1][:max_display]
    sorted_params = [varying_params[i] for i in sorted_indices]
    sorted_importance = importance[sorted_indices]
    sorted_correlations = correlations[sorted_indices]  # Keep signed correlations
    
    print(f"   Top {min(5, len(sorted_params))} features by importance:")
    for param, imp in zip(sorted_params[:5], sorted_importance[:5]):
        print(f"      {param}: {imp:.4f}")
    
    # Create plot
    n_features = len(sorted_params)
    fig, ax = plt.subplots(figsize=(10, max(6, n_features * 0.4)))
    
    # Color map for feature values (blue=low, red=high)
    cmap = cm.get_cmap('coolwarm')
    
    # Standardize target for computing effects
    target_values = df[target].values
    target_mean = target_values.mean()
    target_std = target_values.std()
    if target_std > 0:
        target_standardized = (target_values - target_mean) / target_std
    else:
        target_standardized = target_values - target_mean
    
    # Compute max effect range for scaling (to emphasize high-importance features)
    max_effect_range = 0.0
    feature_effect_ranges = []
    
    for plot_idx, feat_idx in enumerate(sorted_indices):
        param = varying_params[feat_idx]
        feature_values = df[param].values
        feature_centered = feature_values - feature_values.mean()
        feature_std = feature_values.std()
        # Use SIGNED correlation to preserve direction (not absolute importance)
        if feature_std > 0:
            effects = sorted_correlations[plot_idx] * (feature_centered / feature_std)
        else:
            effects = sorted_correlations[plot_idx] * feature_centered
        effect_range = effects.max() - effects.min()
        feature_effect_ranges.append(effect_range)
        max_effect_range = max(max_effect_range, effect_range)
    
    print(f"   Effect range (max): {max_effect_range:.3f}")
    
    # Plot each feature
    for plot_idx, feat_idx in enumerate(sorted_indices):
        param = varying_params[feat_idx]
        
        # Get feature values
        feature_values = df[param].values
        
        # Compute effect: importance × standardized feature values
        # Standardization makes features comparable per unit of variation
        feature_mean = feature_values.mean()
        feature_std = feature_values.std()
        feature_centered = feature_values - feature_mean
        
        # Effect = signed_correlation × standardized_value (divided by std)
        # This shows impact per standard deviation, making features comparable
        # even when they have different scales or are discrete vs continuous
        # CRITICAL: Use signed correlation (not absolute importance) to show direction!
        if feature_std > 0:
            effects = sorted_correlations[plot_idx] * (feature_centered / feature_std)
        else:
            effects = sorted_correlations[plot_idx] * feature_centered
        
        # Normalize feature values for coloring (0 to 1)
        norm_values = normalize_to_range(feature_values)
        
        # Subsample if too many points (for performance and clarity)
        if len(effects) > max_samples:
            sample_indices = np.random.choice(len(effects), max_samples, replace=False)
            effects_plot = effects[sample_indices]
            norm_values_plot = norm_values[sample_indices]
        else:
            effects_plot = effects
            norm_values_plot = norm_values
        
        # Y position for this feature
        y_pos = n_features - plot_idx - 1
        
        if interpolate:
            # Continuous beeswarm using violin-style KDE density with gradient fill
            from scipy.stats import gaussian_kde
            
            # Skip if too few points
            if len(effects_plot) < 5:
                continue
            
            # Compute KDE for density estimation
            try:
                kde = gaussian_kde(effects_plot, bw_method=0.2)
            except (np.linalg.LinAlgError, ValueError):
                continue
            
            # Create dense x-grid for smooth curve
            effect_min, effect_max = effects_plot.min(), effects_plot.max()
            effect_range = effect_max - effect_min
            if effect_range < 1e-10:
                continue
                
            pad = 0.15 * effect_range
            x_grid = np.linspace(effect_min - pad, effect_max + pad, 500)
            
            # Evaluate KDE density at grid points
            density = kde(x_grid)
            
            # Normalize density for y-axis scaling (violin width)
            max_density = density.max()
            if max_density > 0:
                density_scaled = density / max_density * 0.4  # Max width ±0.4
            else:
                continue
            
            # Compute average feature value at each x position for coloring
            # Use weighted average based on proximity
            from scipy.ndimage import gaussian_filter1d
            
            # Create fine bins for color mapping
            n_bins = len(x_grid)
            x_bins = x_grid
            
            # For each x position, compute weighted average of nearby feature values
            color_values = np.zeros(n_bins)
            bandwidth = kde.factor * np.std(effects_plot)
            
            for i, x_center in enumerate(x_bins):
                # Gaussian weights for nearby points
                weights = np.exp(-0.5 * ((effects_plot - x_center) / bandwidth) ** 2)
                weights = weights / (weights.sum() + 1e-10)
                # Weighted average of normalized feature values
                color_values[i] = np.sum(weights * norm_values_plot)
            
            # Smooth the color transitions
            color_values = gaussian_filter1d(color_values, sigma=3.0)
            
            # Clip to valid range
            color_values = np.clip(color_values, 0, 1)
            
            # Create the violin plot with gradient coloring
            # Split into many thin vertical slices, each with its own color
            y_upper = y_pos + density_scaled
            y_lower = y_pos - density_scaled
            
            # Use LineCollection for efficient rendering with color gradients
            from matplotlib.collections import LineCollection
            
            # Create segments for upper and lower contours
            points_upper = np.array([x_grid, y_upper]).T.reshape(-1, 1, 2)
            points_lower = np.array([x_grid, y_lower]).T.reshape(-1, 1, 2)
            
            segments_upper = np.concatenate([points_upper[:-1], points_upper[1:]], axis=1)
            segments_lower = np.concatenate([points_lower[:-1], points_lower[1:]], axis=1)
            
            # Map colors
            colors = cmap(color_values)
            
            # Draw the filled violin using multiple vertical bars
            for i in range(len(x_grid) - 1):
                # Only draw if density is significant
                avg_density = (density_scaled[i] + density_scaled[i+1]) / 2
                if avg_density > 0.01:
                    x_vals = [x_grid[i], x_grid[i+1]]
                    y_lower_vals = [y_lower[i], y_lower[i+1]]
                    y_upper_vals = [y_upper[i], y_upper[i+1]]
                    
                    color_rgb = colors[i]
                    ax.fill_between(x_vals, y_lower_vals, y_upper_vals,
                                   color=color_rgb, alpha=0.8, 
                                   edgecolor='none', zorder=2)
            
            # Add subtle outline
            ax.plot(x_grid, y_upper, color='gray', linewidth=0.3, alpha=0.4, zorder=3)
            ax.plot(x_grid, y_lower, color='gray', linewidth=0.3, alpha=0.4, zorder=3)
            
        else:
            # Original scatter plot
            # Add jitter to y-axis for visibility
            # Use importance-based jitter: more important = more spread = easier to see
            jitter_amount = 0.12 * (1.0 + sorted_importance[plot_idx])  # Use sorted_importance
            y_jitter = np.random.normal(0, jitter_amount, len(effects_plot))
            
            # Plot dots with size proportional to importance
            dot_size = 8 + 20 * sorted_importance[plot_idx]  # Use sorted_importance
            
            scatter = ax.scatter( 
                effects_plot,
                np.ones(len(effects_plot)) * y_pos + y_jitter,
                c=norm_values_plot,
                cmap=cmap,
                s=dot_size,
                alpha=0.6,
                edgecolors='none',
                vmin=0,
                vmax=1,
                rasterized=True  # For better performance with many points
            )
    
    # Set y-axis labels with importance values
    # Labels need to be reversed because y_pos = n_features - plot_idx - 1
    # so the first feature (plot_idx=0) is at the TOP (highest y_pos)
    ax.set_yticks(range(n_features))
    y_labels = [f"{get_param_symbol(param)} (|r|={sorted_importance[i]:.3f})" 
                for i, param in enumerate(sorted_params)]
    y_labels_reversed = y_labels[::-1]  # Reverse the labels!
    ax.set_yticklabels(y_labels_reversed, fontsize=10)
    ax.set_ylim(-0.8, n_features - 0.2)
    
    # Set x-axis label
    target_label = get_param_symbol(target)
    ax.set_xlabel(f'Impact on {target_label} (correlation × standardized value)', fontsize=12)
    ax.axvline(x=0, color='#888888', linestyle='-', linewidth=1.0, alpha=0.8, zorder=0)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Title with explanation
    target_symbol = get_param_symbol(target)
    title_text = f'Feature Importance: {target_symbol}'
    if target_unit:
        title_text += f' [{target_unit}]'
    
    # Add subtitle with correlation info
    top_feature_corr = sorted_importance[0] if len(sorted_importance) > 0 else 0
    top_feature_symbol = get_param_symbol(sorted_params[0]) if len(sorted_params) > 0 else ''
    subtitle = f'(Top feature: {top_feature_symbol} with |correlation|={top_feature_corr:.3f})'
    
    ax.set_title(title_text + '\n' + subtitle,
                 fontsize=14, pad=20, fontweight='bold')
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Feature Value', fontsize=10)
    cbar.ax.set_yticks([0, 0.5, 1.0])
    cbar.ax.set_yticklabels(['Low', 'Mid', 'High'], fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    output_path = outputs_dir / f"{plot_name}_shap_beeswarm.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {output_path.name}")
    
    # Return statistics
    results = {
        'n_features': len(sorted_params),
        'n_samples': len(df),
        'feature_importance': dict(zip(sorted_params, sorted_importance.tolist())),
        'top_feature': sorted_params[0] if len(sorted_params) > 0 else None,
        'top_importance': float(sorted_importance[0]) if len(sorted_importance) > 0 else 0.0
    }
    
    return results


def save_importance_to_csv(df, input_parameters, target, outputs_dir, plot_name):
    """
    Save feature importance rankings to CSV file.
    
    Only includes varying parameters (excludes constants).
    
    Args:
        df: DataFrame with input and output data
        input_parameters: List of input parameter names
        target: Target output variable name
        outputs_dir: Directory to save CSV
        plot_name: Base name for saved file
    """
    # Filter out constant parameters
    varying_params = [param for param in input_parameters if df[param].std() > 0]
    
    if len(varying_params) == 0:
        print(f"   ⚠️  No varying parameters to save to CSV")
        return
    
    # Compute importance and correlations
    importance, correlations = compute_feature_importance(df, varying_params, target)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': varying_params,
        'importance': importance,
        'rank': np.argsort(np.argsort(importance)[::-1]) + 1
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Add correlation values (signed, not absolute) - already computed
    importance_df['correlation'] = correlations
    
    # Save to CSV
    csv_path = outputs_dir / f"{plot_name}_shap_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    
    print(f"   ✅ Saved: {csv_path.name}")


def generate_shap_plots(df_filtered, target, input_parameters, target_unit, 
                       outputs_dir, file_type, plot_name,
                       max_display=20, max_samples=2000, save_csv=True, 
                       interpolate=False):
    """
    Main function to generate SHAP-style plots for a target variable.
    
    This function follows the same pattern as other plot_*_functions.py modules:
    - Takes a filtered DataFrame from HDF5
    - Generates visualization directly from data
    - Saves plot and optional CSV files
    
    Args:
        df_filtered: Filtered DataFrame with data
        target: Target variable name
        input_parameters: List of input parameter names
        target_unit: Unit string for target variable
        outputs_dir: Directory to save plots
        file_type: Type of file (for plot title)
        plot_name: Base name for saved plots
        max_display: Maximum number of features to display (default: 20)
        max_samples: Maximum samples to plot per feature (default: 2000)
        save_csv: Whether to save importance rankings to CSV (default: True)
        interpolate: If True, create smooth density plot instead of scatter (default: False)
    
    Returns:
        Dictionary with results and statistics
    """
    if len(input_parameters) == 0:
        print(f"   No input parameters found for target '{target}'. Skipping SHAP plot.")
        return None
    
    try:
        # Generate beeswarm plot
        results = create_shap_style_beeswarm_plot(
            df_filtered, input_parameters, target, target_unit,
            outputs_dir, plot_name, max_display, max_samples, interpolate
        )
        
        # Save importance to CSV
        if save_csv:
            save_importance_to_csv(
                df_filtered, input_parameters, target, outputs_dir, plot_name
            )
        
        return results
    
    except Exception as e:
        print(f"   ❌ Error generating SHAP plots: {e}")
        import traceback
        traceback.print_exc()
        return None
