"""
Elementary Effects plotting functions.

This module provides visualization tools for Elementary Effects (Morris method)
sensitivity analysis results, including:
- Error bar plots with confidence intervals
- Morris method scatter plots (μ* vs σ)
- Box plots for combined analyses
- 2x2 grid comparisons for different models and metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from matplotlib.gridspec import GridSpec


# LaTeX-style parameter names for consistent plotting
LATEX_NAMES = {
    "V_plasma": r"$V_{\mathrm{p}}$",
    "T_i": r"$T_{\mathrm{i}}$",
    "n_tot": r"$n_{\mathrm{tot}}$",
    "tau_p_T": r"$\tau_{\mathrm{p,T}}$",
    "tau_p_He3": r"$\tau_{\mathrm{p,He3}}$",
    "P_aux": r"$P_{\mathrm{aux}}$",
    "P_aux_DT_eq": r"$P_{\mathrm{aux,DT}}$",
    "TBR_DT": r"$\mathrm{TBR_{DT}}$",
    "TBR_DDn": r"$\mathrm{TBR_{DDn}}$",
    "tau_ifc": r"$\tau_{\mathrm{ifc}}$",
    "tau_ofc": r"$\tau_{\mathrm{ofc}}$",
    "eta_th": r"$\eta_{\mathrm{th}}$",
    "capacity_factor": r"$C_{\mathrm{f}}$",
    "price_of_electricity": r"$c_{\mathrm{e}}$",
    "I_target": r"$I_{\mathrm{target}}$"
}


def plot_confidence_intervals(
    sensitivity_data: Dict[str, Any],
    metric_name: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Create error bar plot with 95% confidence intervals.
    
    Args:
        sensitivity_data: Dictionary with mu, mu_star, sigma, raw_effects
        metric_name: Name of output metric (e.g., 't_startup')
        output_dir: Directory to save plot
        figsize: Figure size tuple
    """
    param_names = list(sensitivity_data['mu'].keys())
    mu_values = [sensitivity_data['mu'][name] for name in param_names]
    mu_star_values = [sensitivity_data['mu_star'][name] for name in param_names]
    sigma_values = [sensitivity_data['sigma'][name] for name in param_names]
    sigma_star_values = [sensitivity_data['sigma_star'][name] for name in param_names]
    
    # Calculate 95% confidence intervals
    sem_values = []
    ci95_values = []
    sem_star_values = []
    ci95_star_values = []
    sample_sizes = []
    
    for name in param_names:
        effects = np.array(sensitivity_data['raw_effects'][name])
        sample_sizes.append(len(effects))
        
        if len(effects) > 0:
            sem = sigma_values[param_names.index(name)] / np.sqrt(len(effects))
            sem_star = sigma_star_values[param_names.index(name)] / np.sqrt(len(effects))
            
            ci95 = 1.96 * sem
            ci95_star = 1.96 * sem_star
            
            sem_values.append(sem)
            sem_star_values.append(sem_star)
            ci95_values.append(ci95)
            ci95_star_values.append(ci95_star)
        else:
            sem_values.append(np.nan)
            sem_star_values.append(np.nan)
            ci95_values.append(np.nan)
            ci95_star_values.append(np.nan)
    
    # Filter valid data
    valid_indices = [i for i, (mu, mu_star, ci, ci_star) in 
                     enumerate(zip(mu_values, mu_star_values, ci95_values, ci95_star_values)) 
                     if np.isfinite(mu) and np.isfinite(mu_star) and 
                     np.isfinite(ci) and np.isfinite(ci_star)]
    
    if len(valid_indices) == 0:
        print(f"No valid data to plot for {metric_name}")
        return
    
    valid_names = [param_names[i] for i in valid_indices]
    valid_mu = [mu_values[i] for i in valid_indices]
    valid_mu_star = [mu_star_values[i] for i in valid_indices]
    valid_ci95 = [ci95_values[i] for i in valid_indices]
    valid_ci95_star = [ci95_star_values[i] for i in valid_indices]
    
    # Sort by mu_star
    sorted_indices = np.argsort(valid_mu_star)[::-1]
    sorted_names = [valid_names[i] for i in sorted_indices]
    sorted_latex_names = [LATEX_NAMES.get(name, name) for name in sorted_names]
    sorted_mu = [valid_mu[i] for i in sorted_indices]
    sorted_mu_star = [valid_mu_star[i] for i in sorted_indices]
    sorted_ci95 = [valid_ci95[i] for i in sorted_indices]
    sorted_ci95_star = [valid_ci95_star[i] for i in sorted_indices]
    
    # Create plot
    fig, ax = plt.figure(figsize=figsize), plt.gca()
    y_pos = np.arange(len(sorted_names))
    
    # Plot μ* with error bars
    ax.errorbar(sorted_mu_star, y_pos, xerr=sorted_ci95_star, fmt='o', 
                color='blue', label=r'$\mu^*$ (Mean of |EE|)', 
                capsize=5, markersize=8)
    
    # Plot μ with error bars
    ax.errorbar(sorted_mu, y_pos, xerr=sorted_ci95, fmt='s', 
                color='green', label=r'$\mu$ (Mean of EE)',
                capsize=5, markersize=8)
    
    # Add zero line
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_latex_names)
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    ax.set_xlabel('Effect Magnitude (Normalized Elementary Effects)')
    ax.set_title(f'Parameter Sensitivity - {metric_name.replace("_", " ").title()}')
    ax.legend(loc='best')
    
    plt.tight_layout()
    output_path = output_dir / f'ee_confidence_intervals_{metric_name}.png'
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confidence interval plot: {output_path}")


def plot_morris_scatter(
    sensitivity_data: Dict[str, Any],
    metric_name: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (8, 8)
):
    """
    Create Morris method scatter plot (μ* vs σ).
    
    Args:
        sensitivity_data: Dictionary with mu, mu_star, sigma, sigma_star
        metric_name: Name of output metric
        output_dir: Directory to save plot
        figsize: Figure size tuple
    """
    param_names = list(sensitivity_data['mu'].keys())
    mu_values = [sensitivity_data['mu'][name] for name in param_names]
    mu_star_values = [sensitivity_data['mu_star'][name] for name in param_names]
    sigma_values = [sensitivity_data['sigma'][name] for name in param_names]
    sigma_star_values = [sensitivity_data['sigma_star'][name] for name in param_names]
    
    # Filter valid data
    valid_indices = [i for i, (mu, mu_star, sigma, sigma_star) in 
                     enumerate(zip(mu_values, mu_star_values, sigma_values, sigma_star_values)) 
                     if np.isfinite(mu) and np.isfinite(mu_star) and 
                     np.isfinite(sigma) and np.isfinite(sigma_star)]
    
    if len(valid_indices) == 0:
        print(f"No valid data for Morris scatter plot: {metric_name}")
        return
    
    valid_names = [param_names[i] for i in valid_indices]
    valid_mu = [mu_values[i] for i in valid_indices]
    valid_mu_star = [mu_star_values[i] for i in valid_indices]
    valid_sigma = [sigma_values[i] for i in valid_indices]
    valid_sigma_star = [sigma_star_values[i] for i in valid_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot μ* vs σ*
    ax.scatter(valid_mu_star, valid_sigma_star, s=100, color='blue', 
               marker='o', label=r'$\mu^*$ vs $\sigma^*$', alpha=0.7)
    
    # Add parameter labels
    for i, name in enumerate(valid_names):
        latex_name = LATEX_NAMES.get(name, name)
        ax.annotate(latex_name, (valid_mu_star[i], valid_sigma_star[i]), 
                   textcoords="offset points", xytext=(0, 10), ha='center',
                   fontsize=10)
    
    # Plot μ vs σ for comparison
    ax.scatter(valid_mu, valid_sigma, s=80, color='green', 
               marker='s', label=r'$\mu$ vs $\sigma$', alpha=0.5)
    
    ax.set_xlabel(r'$\mu^*$ / $\mu$ (Mean Effect)')
    ax.set_ylabel(r'$\sigma^*$ / $\sigma$ (Standard Deviation)')
    ax.set_title(f'Morris Method Plot - {metric_name.replace("_", " ").title()}')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    plt.tight_layout()
    output_path = output_dir / f'ee_morris_scatter_{metric_name}.png'
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Morris scatter plot: {output_path}")


def plot_box_plots(
    sensitivity_data: Dict[str, Any],
    metric_name: str,
    output_dir: Path,
    remove_outliers: bool = True,
    min_effect_threshold: float = 1e-6,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Create box plots of elementary effects distributions.
    
    Args:
        sensitivity_data: Dictionary with raw_effects, mu_star
        metric_name: Name of output metric
        output_dir: Directory to save plot
        remove_outliers: Whether to remove outliers from plot
        min_effect_threshold: Minimum μ* to include parameter
        figsize: Figure size tuple
    """
    ee_values = sensitivity_data['raw_effects']
    mu_star = sensitivity_data['mu_star']
    
    # Filter parameters with significant effects
    param_names = [param for param in ee_values.keys() 
                   if mu_star.get(param, 0) > min_effect_threshold]
    
    if len(param_names) == 0:
        print(f"No parameters with effects above threshold for {metric_name}")
        return
    
    # Prepare data
    df_list = []
    for param in param_names:
        values = np.array(ee_values[param])
        if len(values) > 0:
            # Normalize if values are very large
            if np.mean(np.abs(values)) > 100000:
                output_range = sensitivity_data.get('output_range', 1.0)
                values = values / output_range
            
            # Remove outliers if requested
            if remove_outliers:
                q1, q3 = np.percentile(values, [2.5, 97.5])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                values = values[(values >= lower_bound) & (values <= upper_bound)]
            
            df_list.append(pd.DataFrame({
                'Parameter': [LATEX_NAMES.get(param, param)] * len(values),
                'Elementary Effect': values
            }))
    
    if not df_list:
        print(f"No valid data for box plot: {metric_name}")
        return
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Sort by μ*
    sort_values = {LATEX_NAMES.get(param, param): mu_star.get(param, 0) 
                   for param in param_names}
    param_order = sorted(sort_values.keys(), key=lambda p: sort_values[p], reverse=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.boxplot(x='Elementary Effect', y='Parameter', data=df,
                order=param_order, orient='h', 
                showfliers=not remove_outliers,
                whis=[2.5, 97.5], ax=ax)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    ax.set_xlabel('Normalized Elementary Effect')
    ax.set_ylabel('')
    ax.set_title(f'Elementary Effects Distribution - {metric_name.replace("_", " ").title()}')
    
    plt.tight_layout()
    output_path = output_dir / f'ee_box_plot_{metric_name}.png'
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved box plot: {output_path}")


def create_all_plots(
    stats: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True
):
    """
    Create all standard Elementary Effects plots.
    
    Args:
        stats: Statistics dictionary from elementary effects analysis
        output_dir: Directory to save plots
        verbose: Whether to print progress
    """
    if verbose:
        print(f"\n{'='*60}")
        print("GENERATING ELEMENTARY EFFECTS PLOTS")
        print(f"{'='*60}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sensitivity_results = stats.get('sensitivity_results', {})
    
    for metric, sens_data in sensitivity_results.items():
        if verbose:
            print(f"\nGenerating plots for metric: {metric}")
        
        # Confidence interval plot
        plot_confidence_intervals(sens_data, metric, output_dir)
        
        # Morris scatter plot
        plot_morris_scatter(sens_data, metric, output_dir)
        
        # Box plot
        plot_box_plots(sens_data, metric, output_dir)
    
    if verbose:
        print(f"\n✅ All plots saved to: {output_dir}")
        print(f"{'='*60}\n")
