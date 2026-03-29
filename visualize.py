from numpy._typing import NDArray
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from proc import Logger
import wandb


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
