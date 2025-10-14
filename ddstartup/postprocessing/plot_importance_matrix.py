"""
Effect-size / Importance Matrix plotting utilities.

Computes Cohen's d per quartile for each input parameter and saves a heatmap + CSV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def cohen_d(a, b):
    """Compute Cohen's d between two samples."""
    a = np.asarray(a)
    b = np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sa2 = a.std(ddof=1) ** 2
    sb2 = b.std(ddof=1) ** 2
    pooled = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled


def compute_effect_size_matrix(df, target, inputs, quartiles=4):
    """Compute Cohen's d for each input across quartiles of the target.

    Returns a DataFrame with index=inputs and columns=Q1..Qn.
    """
    labels = [f"Q{i+1}" for i in range(quartiles)]
    qcuts = pd.qcut(df[target], quartiles, labels=labels)
    effects = pd.DataFrame(index=inputs, columns=labels, dtype=float)

    for label in labels:
        mask = qcuts == label
        group = df.loc[mask]
        complement = df.loc[~mask]
        for inp in inputs:
            if inp not in df.columns:
                effects.loc[inp, label] = np.nan
                continue
            effects.loc[inp, label] = cohen_d(group[inp].dropna(), complement[inp].dropna())
    return effects


def plot_effect_size_matrix(df, target, inputs, outputs_dir, plot_name=None, save_csv=True):
    """Compute effect-size matrix and plot heatmap. Saves CSV and PNG to outputs_dir.

    Returns the effects DataFrame.
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    effects = compute_effect_size_matrix(df, target, inputs)

    if save_csv:
        csv_name = outputs_dir / (plot_name + '_effects.csv' if plot_name else f'{target}_effects.csv')
        effects.to_csv(csv_name)

    plt.figure(figsize=(max(6, len(inputs)*0.4), 6))
    sns.heatmap(effects.astype(float), cmap='vlag', center=0, annot=True, fmt='.2f')
    plt.title(f"Effect Size (Cohen's d) per Quartile — {target}")
    plt.tight_layout()
    png_name = outputs_dir / (plot_name + '.png' if plot_name else f'{target}_effects.png')
    plt.savefig(png_name, dpi=150)
    plt.close()
    return effects
