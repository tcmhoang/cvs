import numpy as np
from numpy._typing import NDArray
from torch import device
import torch
from torch.utils.data import DataLoader
from typing import cast, Tuple
from dataset import Tensor
import faiss

from model import RetrievalNet


def extract_features(
    m: RetrievalNet, dataloader: DataLoader[Tensor], dev: device
) -> Tuple[NDArray, NDArray]:
    m.eval()

    with torch.no_grad():
        features, labels = cast(
            Tuple[Tuple[NDArray, ...], Tuple[NDArray, ...]],
            tuple(
                zip(
                    *[
                        (
                            m.forward(cast(Tensor, img).to(dev)).cpu().numpy(),
                            cast(Tensor, label).numpy(),
                        )
                        for img, label in dataloader
                    ]
                )
            ),
        )

        return np.vstack(features), np.array(labels)


def _io_add_and_search(
    index: faiss.Index, query_feats: NDArray, top_k: int
) -> Tuple[NDArray, NDArray]:
    index.add(query_feats)  # type: ignore
    distances, indices = index.search(query_feats, top_k)  # type: ignore

    return distances, indices


def _get_search_index(feats: NDArray, k=5) -> NDArray:
    dim = feats.shape[1]

    faiss.normalize_L2(feats)
    index = faiss.IndexFlatIP(dim)

    _, Index = _io_add_and_search(index, feats, k)

    return Index


def eval_rank(feats: NDArray, labels: NDArray, k=5) -> Tuple[float, float]:

    Index = _get_search_index(feats, k)

    r1 = 0
    rk = 0
    n = len(labels)

    for i in range(n):
        retrieved = Index[i][1:]  # skip itself
        rlabels = labels[retrieved]

        if rlabels[0] == labels[i]:
            r1 += 1
        if labels[i] in rlabels[:k]:
            rk += 1
        pass

    return r1 / n, rk / n


def compute_map(feats: NDArray, labels: NDArray) -> np.floating:

    Index = _get_search_index(feats, len(feats))

    APs = []

    for i in range(len(labels)):
        retrieved = Index[i][1:]
        rlabels = labels[retrieved]
        corrects = rlabels == labels[i]

        precisions = []
        correct_count = 0

        for j, val in enumerate(corrects):
            if val:
                correct_count += 1
                precisions.append(correct_count / (j + 1))

        APs.append(np.mean(precisions) if len(precisions) > 0 else 0)

        pass

    return np.mean(APs)


def save():
    pass


def load():
    pass
