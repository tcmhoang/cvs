from itertools import groupby
from torch import (
    Tensor,
    bmm,
    cosine_similarity,
    nn,
    device,
    no_grad,
    optim,
    stack,
)
from typing import Dict, cast, List, Tuple
import random

import torch


from dataset import ImageFolder
from main import Logger
from model import RetrievalNet


def retrieval_model(
    m: RetrievalNet,
    train_dataset: ImageFolder,
    device: device,
    logger: Logger,
    epochs=20,
    batch_size=16,
    lr=1e-4,
    margin=0.2,
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tmloss = nn.TripletMarginLoss(margin=margin, p=2)

    class_to_indices = _gen_class_to_idces(train_dataset)
    clsses = list(class_to_indices.keys())

    for e in range(epochs):
        eloss = 0.0

        random.shuffle(clsses)
        batches = [
            clsses[i : i + batch_size] for i in range(0, len(clsses), batch_size)
        ]

        for bcs in batches:
            embs = (
                m(t.to(device))
                for t in _gen_triplet_batch(
                    train_dataset, class_to_indices, bcs, m, device
                )
            )

            loss = cast(Tensor, tmloss(*embs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            eloss += loss.item()

            pass

        scheduler.step()

        avg_loss = eloss / len(batches)
        print(f"e {e + 1}/{epochs} aloss: {avg_loss:.4f}")

        logger.log({"epoch": e + 1, "train_loss": avg_loss})
        pass

    return m


def _gen_class_to_idces(ds: ImageFolder) -> Dict[int, List[int]]:
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


def _gen_triplet_batch(
    ds: ImageFolder,
    class_to_indices: Dict[int, List[int]],
    batch_classes: List[int],
    m: RetrievalNet,
    device: device,
) -> Tuple[Tensor, Tensor, Tensor]:

    labels = class_to_indices.keys()

    anchors, poss, negss = cast(
        Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[List[int], ...]],
        tuple(
            zip(
                *[
                    (
                        cast(int, x),
                        cast(int, y),
                        random.sample(
                            cast(
                                List[int],
                                class_to_indices.get(
                                    random.choice(
                                        [label for label in labels if label != c]
                                    )
                                ),
                            ),
                            5,
                        ),
                    )
                    for c in batch_classes
                    for x, y in random.sample(
                        cast(List, class_to_indices.get(c)) or [], 2
                    )
                ]
            )
        ),
    )

    with torch.no_grad():
        a_imgs = torch.stack([ds[i][0] for i in anchors]).to(device)
        negs = torch.stack([ds[i][0] for fneg in negss for i in fneg]).to(
            device
        )  # (B*5, C, H, W)

        a_embs = m.forward(a_imgs)  # (B, dim)
        n_embs = m.forward(negs).view(len(anchors), 5, -1)  # (B, 5, dim)

        a_embs = torch.nn.functional.normalize(a_embs, p=2, dim=1)
        n_embs = torch.nn.functional.normalize(n_embs, p=2, dim=2)

        pass

    # (B, 1, dim) @ (B, dim, 5) -> (B, 1, 5)
    scores = torch.bmm(a_embs.unsqueeze(1), n_embs.transpose(1, 2)).squeeze(1)
    hard_idx = scores.argmax(dim=1)  # (B,)

    negs = [negss[i][hard_idx[i]] for i in range(len(anchors))]

    return cast(
        Tuple[Tensor, Tensor, Tensor],
        tuple(
            map(
                lambda idcs: torch.stack([ds[i][0] for i in idcs]),
                (anchors, poss, negs),
            )
        ),
    )
