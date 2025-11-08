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


def cluster_and_quartile_bar(df, inputs, target, outputs_dir, n_clusters=5, plot_name=None, save_csv=True):
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    registry = get_registry()

    X = df[inputs].copy()
    # Drop columns not present
    X = X.loc[:, [c for c in inputs if c in X.columns]]
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
    png_name = outputs_dir / (plot_name + '.png' if plot_name else f'kmeans_{target}.png')
    plt.savefig(png_name, dpi=150)
    plt.close()

    if save_csv:
        # Save cluster centers (mean values)
        centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)
        centers['cluster'] = range(n_clusters)
        centers.to_csv(outputs_dir / (plot_name + '_cluster_centers.csv' if plot_name else f'kmeans_centers_{target}.csv'), index=False)
        
        # Save cluster ranges (min, max, mean, std for each parameter)
        ranges_data = []
        for cluster_id in range(n_clusters):
            cluster_mask = df['cluster'] == cluster_id
            cluster_df = df[cluster_mask]
            n_samples = cluster_mask.sum()
            
            for param in X.columns:
                ranges_data.append({
                    'cluster': cluster_id,
                    'parameter': param,
                    'min': cluster_df[param].min(),
                    'max': cluster_df[param].max(),
                    'mean': cluster_df[param].mean(),
                    'std': cluster_df[param].std(),
                    'n_samples': n_samples
                })
        
        ranges_df = pd.DataFrame(ranges_data)
        ranges_df.to_csv(outputs_dir / (plot_name + '_cluster_ranges.csv' if plot_name else f'kmeans_ranges_{target}.csv'), index=False)

    return kmeans, ctab
