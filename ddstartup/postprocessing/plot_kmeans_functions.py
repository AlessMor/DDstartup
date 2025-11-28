"""
KMeans clustering and quartile stacked bar plot utilities.

Clusters the input parameter space and plots the distribution of target quartiles per cluster.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ddstartup.postprocessing.postprocess_functions import get_discrete_colorscale
from ddstartup.utils.parameter_registry import get_registry


def _select_scalar_numeric(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Filter to scalar, numeric columns using df.attrs['_inner_dims'] when present."""
    inner = (getattr(df, "attrs", {}) or {}).get("_inner_dims", {})
    usable = []
    for c in cols:
        if c not in df.columns:
            continue
        if int(inner.get(c, 1)) != 1:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        usable.append(c)
    return usable


def cluster_and_quartile_bar(
    df,
    inputs,
    target,
    output_dir=None,
    outputs_dir=None,
    n_clusters=5,
    plot_name=None,
    plot_name_prefix=None,
    save_csv=True,
    registry=None,
    **_,
):
    """
    Cluster data using KMeans and visualize distribution across target quartiles.
    
    Args:
        df: DataFrame with data
        inputs: List of input parameter names
        target: Target variable name
        outputs_dir: Directory to save outputs
        n_clusters: Number of clusters (default: 5)
        plot_name: Optional plot name prefix
        save_csv: Whether to save cluster centers to CSV
        registry: ParameterRegistry instance (optional, will create if not provided)
    """
    outdir = Path(output_dir if output_dir is not None else (outputs_dir if outputs_dir is not None else "."))
    outdir.mkdir(parents=True, exist_ok=True)
    stem = plot_name_prefix or plot_name or f'kmeans_{target}'
    
    # Get registry if not provided
    if registry is None:
        from ddstartup.utils.parameter_registry import get_registry
        registry = get_registry()

    usable_inputs = _select_scalar_numeric(df, list(inputs or []))
    if not usable_inputs:
        print("   No scalar numeric inputs for k-means. Skipping.")
        return None, None

    X = df[usable_inputs].copy()
    if X.shape[0] == 0:
        raise ValueError('No input columns available for clustering')

    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.fillna(0))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(Xs)
    df['cluster'] = labels

    # Quartiles with consistent color scheme (green to red, matching other plots)
    df['quartile'] = pd.qcut(df[target], 4, labels=[f'Q{i+1}' for i in range(4)])

    ctab = pd.crosstab(df['cluster'], df['quartile'], normalize='index')
    
    # Get the same color scheme as KDE and other plots (green = Q1/best, red = Q4/worst)
    colorscale = get_discrete_colorscale(4)
    quartile_colors = [colorscale[i*2][1] for i in range(4)]  # Extract colors for Q1-Q4
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ctab.plot.bar(stacked=True, ax=ax, color=quartile_colors)
    ax.set_ylabel('Proportion')
    target_symbol = registry.get_symbol(target)
    ax.set_title(f'Quartile Distribution per Cluster (k={n_clusters}) — {target_symbol}')
    plt.tight_layout()
    png_name = outdir / f'{stem}.png'
    plt.savefig(png_name, dpi=150)
    plt.close()

    if save_csv:
        # Save cluster centers (mean values)
        centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)
        centers['cluster'] = range(n_clusters)
        centers.to_csv(outdir / f'{stem}_cluster_centers.csv', index=False)
        
        # Save cluster ranges (min, max, mean, std for each parameter)
        ranges_data = []
        for cluster_id in range(n_clusters):
            cluster_mask = df['cluster'] == cluster_id
            cluster_df = df[cluster_mask]
            n_samples = cluster_mask.sum()
            
            for param in X.columns:
                try:
                    # Skip non-scalar columns
                    if cluster_df[param].dtype == 'object':
                        continue
                    ranges_data.append({
                        'cluster': cluster_id,
                        'parameter': param,
                        'min': cluster_df[param].min(),
                        'max': cluster_df[param].max(),
                        'mean': cluster_df[param].mean(),
                        'std': cluster_df[param].std(),
                        'n_samples': n_samples
                    })
                except (TypeError, ValueError):
                    # Skip parameters that cannot have statistics computed
                    continue
        
        ranges_df = pd.DataFrame(ranges_data)
        ranges_df.to_csv(outdir / f'{stem}_cluster_ranges.csv', index=False)

    return kmeans, ctab
