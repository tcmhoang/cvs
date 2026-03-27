from itertools import groupby
import PIL
from timm.models.vision_transformer import VisionTransformer
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch import (
    Tensor,
    cosine_similarity,
    hub,
    nn,
    device,
    no_grad,
    ones,
    optim,
    stack,
)
from typing import Dict, cast, Protocol, Any, List, Tuple
import random

from torchvision.transforms.functional import PILImage

from dataset import ImageFolder
from model import RetrievalNet


def train_retrieval_model(
    m: RetrievalNet,
    train_dataset: ImageFolder,
    device: device,
    epochs=20,
    batch_size=32,
    lr=1e-4,
    margin=0.5,
):
    m.train()

    def get_optimizer_params(weight_decay=0.05) -> List[dict]:
        blocks = cast(nn.ModuleList, m.model.blocks)
        backbone = [blocks[-1], m.model.norm]
        head = [m.gem, m.fc]

        m_w_lrc = [(backbone, 0.1), (head, 1)]

        def go_nwlr(data: Tuple[List[nn.Module], float]) -> List[dict]:
            ms, coeff = data
            return [
                {
                    "params": name_w_param[1],
                    "lr": lr * coeff,
                    "weight_decay": 0.0
                    if name_w_param[0].endswith(".bias")
                    or len(name_w_param[1].shape) == 1
                    else weight_decay,
                }
                for m in ms
                for name_w_param in m.named_parameters()
                if name_w_param[1].requires_grad
            ]

        return [param for e in m_w_lrc for params in go_nwlr(e) for param in params]

    optimizer = optim.AdamW(get_optimizer_params())

    tmloss = nn.TripletMarginLoss(margin=margin, p=2)
    interation_per_epoch = max(1, len(train_dataset) // batch_size)

    for e in range(epochs):
        loss = 0.0

        for i in range(interation_per_epoch):
            pass


def gen_class_to_idxes(ds: ImageFolder) -> Dict[int, List[int]]:
    return {
        k: v
        for k, v in {
            k: list(map(lambda idx_w_label: idx_w_label[0], k_w_vs))
            for k, k_w_vs in groupby(
                enumerate(ds.targets), key=lambda idx_w_label: idx_w_label[1]
            )
        }.items()
        if len(v) >= 2
    }


def gen_triplet_batch(
    ds: ImageFolder,
    class_to_indices: Dict[int, List[int]],
    batch_classes: List[int],
    m: RetrievalNet,
    device: device,
) -> Tuple[Tensor, Tensor, Tensor]:

    labels = class_to_indices.keys()

    a_pos_negs = [
        (
            cast(int, x),
            cast(int, y),
            random.sample(
                cast(
                    List[int],
                    class_to_indices.get(
                        random.choice([label for label in labels if label != c])
                    ),
                ),
                5,
            ),
        )
        for c in batch_classes
        for x, y in random.sample(cast(List, class_to_indices.get(c)) or [], 2)
    ]

    def mine_hard_neg(
        anchor: int,
        negs: List[int],
    ) -> int:
        with no_grad():
            a_emb = m(ds[anchor][0].unsqueeze(0).to(device))

        best_neg = None
        best_dist = -1

        for neg_idx in negs:
            with no_grad():
                neg_emb = m(ds[neg_idx][0].unsqueeze(0).to(device))

            dist = cosine_similarity(a_emb, neg_emb).item()
            if dist > best_dist:
                best_dist = dist
                best_neg = neg_idx

        assert best_neg is not None

        return best_neg

    return cast(
        Tuple[Tensor, Tensor, Tensor],
        tuple(
            map(
                stack,
                zip(
                    *map(
                        lambda a_p_ns: (
                            a_p_ns[0],
                            a_p_ns[1],
                            mine_hard_neg(a_p_ns[0], a_p_ns[2]),
                        ),
                        a_pos_negs,
                    )
                ),
            )
        ),
    )
