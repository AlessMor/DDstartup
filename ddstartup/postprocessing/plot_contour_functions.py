"""
2D contour / heatmap plotting utilities for parametric grid data.

If the grid is dense and regular, produce interpolated contour. Otherwise produce
a discrete cell-mean heatmap.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scipy.interpolate import griddata
import itertools, math


def plot_2d_cell_mean_heatmap(df, x, y, target, outputs_dir, plot_name=None, interpolate=True, ax=None):
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if x not in df.columns or y not in df.columns or target not in df.columns:
        raise ValueError('Required columns missing')

    # Determine grid
    xi = np.unique(df[x])
    yi = np.unique(df[y])

    # If grid is regular and reasonably dense, interpolate
    regular = (len(xi) * len(yi) == len(df))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True

    if interpolate and regular and len(xi) > 5 and len(yi) > 5:
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((df[x].values, df[y].values), df[target].values, (X, Y), method='cubic')
        cp = ax.contourf(X, Y, Z, cmap='viridis')
        plt.colorbar(cp, ax=ax)
        ax.set_title(f'{target} over {x} vs {y}')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    else:
        # Build pivot table of means
        pivot = df.pivot_table(index=y, columns=x, values=target, aggfunc='mean')
        sns.heatmap(pivot, cmap='viridis', cbar_kws={'label': target}, ax=ax)
        ax.set_title(f'{target} over {x} vs {y} (cell means)')
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    if created_fig:
        png_name = outputs_dir / (plot_name + '.png' if plot_name else f'contour_{target}_{x}_{y}.png')
        plt.tight_layout()
        plt.savefig(png_name, dpi=150)
        plt.close()
        return png_name
    return None


def plot_pairwise_contours(df, inputs, target, outputs_dir, max_pairs=12, interpolate=True, plot_name=None):
    """Plot pairwise contour/heatmap subplots for combinations of input parameters.

    - inputs: list of input parameter names
    - max_pairs: maximum number of pairs to plot (keeps figures readable)
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Filter inputs to those present in df
    inputs_present = [p for p in inputs if p in df.columns]
    pairs = list(itertools.combinations(inputs_present, 2))
    if not pairs:
        raise ValueError('Need at least two input parameters present in data')

    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    n_pairs = len(pairs)
    ncols = int(math.ceil(math.sqrt(n_pairs)))
    nrows = int(math.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (x, y) in zip(axes, pairs):
        try:
            plot_2d_cell_mean_heatmap(df, x, y, target, outputs_dir, interpolate=interpolate, ax=ax)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center')
            ax.set_title(f'{x} vs {y}')

    # Hide unused axes
    for ax in axes[n_pairs:]:
        ax.set_visible(False)

    png_name = outputs_dir / (plot_name + '.png' if plot_name else f'contour_pairwise_{target}.png')
    plt.tight_layout()
    plt.savefig(png_name, dpi=150)
    plt.close()
    return png_name
