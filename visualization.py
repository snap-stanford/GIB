import os
import sys

import numpy as np
import torch
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname("__file__"), ".."))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize(
    z,
    color,
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([])
    ax.set_yticks([])

    sns.scatterplot(z[:, 0], z[:, 1], s=70, hue=color, ax=ax, palette="tab10")
    if ax is None:
        plt.show()


def plot_tsne(out, y_colors, f_suffix="", pca=True):

    z = out
    if pca:
        z = PCA(n_components=5).fit_transform(z)
    z = TSNE(n_components=2).fit_transform(z)

    ncols = 1
    f, axs = plt.subplots(ncols=ncols, figsize=(15, 8))
    visualize(z, color=y_colors, ax=axs)

    plt.savefig(f"TSNE_{f_suffix}.png", dpi=300, transparent=True)
