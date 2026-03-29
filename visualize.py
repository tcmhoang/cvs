from numpy._typing import NDArray
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from proc import Logger
import wandb


def plot_and_log_tsne(features: NDArray, labels: NDArray, logger: Logger):

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_features = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        tsne_features[:, 0],
        tsne_features[:, 1],
        c=labels,
        cmap="tab20",
        s=15,
        alpha=0.8,
    )

    plt.title("t-SNE Visualization of Embeddings")
    plt.colorbar(scatter)

    logger.log({"t-SNE Clustering": wandb.Image(fig)})
    plt.close(fig)

    pass
