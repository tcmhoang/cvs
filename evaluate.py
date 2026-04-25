import io
from typing import Any, Tuple, cast

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zstandard as zstd
from numpy._typing import NDArray
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

import dio
from dataset import Tensor
from model import RetrievalNet
from proc import Dict


def apply_aqe(features: NDArray, k_aqe=3, a=0.7) -> NDArray:
    n = features.shape[0]

    Index = _get_search_index(features, k_aqe + 1)
    super_features = np.zeros_like(features)

    for i in range(n):
        top_k_indices = Index[i][1:]
        top_k_features = features[top_k_indices]
        top_k_mean = np.mean(top_k_features, axis=0)

        super_features[i] = (a * features[i]) + ((1.0 - a) * top_k_mean)
        pass

    faiss.normalize_L2(super_features)

    return super_features


def extract_features(
    m: RetrievalNet, dataloader: DataLoader[Tensor], dev: torch.device
) -> Tuple[NDArray, NDArray]:
    m.eval()

    with torch.no_grad():
        features, labels = cast(
            Tuple[Tuple[NDArray, ...], Tuple[NDArray, ...]],
            tuple(
                zip(
                    *[
                        (
                            F.normalize(
                                (
                                    m.forward(cast(Tensor, img).to(dev))
                                    + m.forward(torch.flip(img, dims=[3]).to(dev))
                                )
                                / 2,
                                p=2,
                                dim=1,
                            )
                            .cpu()
                            .numpy(),
                            cast(Tensor, label).numpy(),
                        )
                        for img, label in dataloader
                    ]
                )
            ),
        )

        return np.vstack(features), np.concatenate(labels)


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


def rank(feats: NDArray, labels: NDArray, k=5) -> Tuple[float, float]:

    k += 1

    Index = _get_search_index(feats, k)

    r1 = 0
    rk = 0
    n = len(labels)

    for i in range(n):
        retrieved = Index[i][1:]  # skip itself
        rlabels = labels[retrieved]

        if rlabels[0] == labels[i]:
            r1 += 1
            pass
        if labels[i] in rlabels[:k]:
            rk += 1
            pass
        pass

    return r1 / n, rk / n


def silhouette(embeddings: NDArray, labels: NDArray) -> float:
    return silhouette_score(embeddings, labels, metric="cosine")


def knn(embeddings: NDArray, labels: NDArray, k=5) -> float:
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42, stratify=labels
    )

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy


def map(feats: NDArray, labels: NDArray) -> np.floating:

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
                pass
            pass
        pass

        APs.append(np.mean(precisions) if len(precisions) > 0 else 0)

        pass

    return np.mean(APs)


def io_report_csv(
    feats: NDArray, labels: NDArray, path: str, outdir: str, k=5
) -> pd.DataFrame:

    dio.create_dir(outdir)

    k += 1

    Index = _get_search_index(feats, k)

    def go(i: int) -> Dict[str, Any]:
        query_label = labels[i]
        retrieved = Index[i][1:]
        retrieved_labels = labels[retrieved]

        row = {"query_id": i, "query_label": query_label}

        actual_k = min(k, len(retrieved))
        for i in range(actual_k):
            if retrieved[i] == -1:
                break

            row[f"rank{i + 1}_id"] = int(retrieved[i])
            row[f"rank{i + 1}_label"] = int(retrieved_labels[i])
            pass
        pass

        return row

    rows = [go(i) for i in range(len(labels))]

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    return df


def io_save(feats: NDArray, labels: NDArray, path: str, outdir: str) -> None:
    dio.create_dir(outdir)

    buffer = io.BytesIO()
    np.savez(buffer, features=feats, labels=labels)
    buffer.seek(0)

    cctx = zstd.ZstdCompressor(level=10)
    compressed_data = cctx.compress(buffer.read())

    with open(path, "wb") as f:
        f.write(compressed_data)
        pass
    pass


def io_load(path: str) -> Tuple[NDArray, NDArray]:
    with open(path, "rb") as f:
        compressed_data = f.read()
        pass

    dctx = zstd.ZstdDecompressor()
    decompressed_data = dctx.decompress(compressed_data)

    buffer = io.BytesIO(decompressed_data)
    data = np.load(buffer)
    return data["features"], data["labels"]
