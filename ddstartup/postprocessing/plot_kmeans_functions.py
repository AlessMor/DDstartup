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


def cluster_and_quartile_bar(df, inputs, target, outputs_dir, n_clusters=5, plot_name=None, save_csv=True):
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

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

    # Quartiles
    df['quartile'] = pd.qcut(df[target], 4, labels=[f'Q{i+1}' for i in range(4)])

    ctab = pd.crosstab(df['cluster'], df['quartile'], normalize='index')
    fig, ax = plt.subplots(figsize=(8, 4))
    ctab.plot.bar(stacked=True, ax=ax, colormap='Spectral')
    ax.set_ylabel('Proportion')
    ax.set_title(f'Quartile Distribution per Cluster (k={n_clusters}) — {target}')
    plt.tight_layout()
    png_name = outputs_dir / (plot_name + '.png' if plot_name else f'kmeans_{target}.png')
    plt.savefig(png_name, dpi=150)
    plt.close()

    if save_csv:
        centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)
        centers['cluster'] = range(n_clusters)
        centers.to_csv(outputs_dir / (plot_name + '_cluster_centers.csv' if plot_name else f'kmeans_centers_{target}.csv'), index=False)

    return kmeans, ctab
