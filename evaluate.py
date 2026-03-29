import numpy as np
from numpy._typing import NDArray
from torch import device
import torch
from torch.utils.data import DataLoader
from typing import cast, Tuple
from dataset import Tensor

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


def eval_rank():
    pass


def save():
    pass


def load():
    pass
