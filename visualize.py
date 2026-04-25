import matplotlib.pyplot as plt
from numpy._typing import NDArray
from sklearn.manifold import TSNE
import umap.umap_ as umap


import wandb
from proc import Logger
from typing import Any, cast


def plot_and_log_tsne(features: NDArray, labels: NDArray, logger: Logger):

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        learning_rate="auto",
    )

    tsne_features = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        tsne_features[:, 0],
        tsne_features[:, 1],
        c=labels,
        cmap="tab20",
        s=15,
        alpha=0.7,
    )

    ax.set_title("t-SNE Visualization of Embeddings")
    fig.colorbar(scatter, ax=ax, label="Class Label")
    ax.axis("off")

    logger.log({"t-SNE Clustering": wandb.Image(fig)})
    plt.close(fig)

    pass


def plot_umap(features: NDArray, labels: NDArray, logger: Logger):
    umap_model = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
    )

    umap_features = cast(
        Any, umap_model.fit_transform(features)
    )  # umap's model is very ugly

    fig, ax = plt.subplots(figsize=(15, 12))
    scatter = ax.scatter(
        umap_features[:, 0],
        umap_features[:, 1],
        c=labels,
        cmap="tab20",
        s=15,
        alpha=0.7,
    )

    ax.set_title("UMAP Visualization of Embeddings")
    fig.colorbar(scatter, ax=ax, label="Class Label")
    ax.axis("off")

    logger.log({"t-SNE Clustering": wandb.Image(fig)})
    plt.close(fig)

    pass
