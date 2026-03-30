import random
from itertools import groupby
from typing import Dict, List, Tuple, cast

import torch
from torch import (
    Tensor,
    device,
    nn,
    optim,
)
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

import config
from dataset import ImageFolder
from model import RetrievalNet
from proc import Logger


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

        return [params for e in m_w_lrc for params in go_nwlr(e)]

    optimizer = optim.AdamW(get_optimizer_params())
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.WARM_UP_FACTOR,
        total_iters=int(config.WARM_UP_PERC * epochs),
    )
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[5]
    )

    tmloss = nn.TripletMarginLoss(margin=margin, p=2)

    class_to_indices = _gen_class_to_idces(train_dataset)
    clsses = list(class_to_indices.keys())

    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    for e in range(epochs):
        eloss = 0.0

        random.shuffle(clsses)
        batches = [
            clsses[i : i + batch_size] for i in range(0, len(clsses), batch_size)
        ]

        for bcs in batches:
            imgs = torch.cat(
                [
                    t
                    for t in _gen_triplet_batch(
                        train_dataset, class_to_indices, bcs, m, device, e
                    )
                ],
                dim=0,
            ).to(device)

            m.train()
            optimizer.zero_grad()

            with autocast(device_type=device.type, dtype=torch.float16):
                all_embs = m(imgs)
                embs = torch.chunk(all_embs, 3, dim=0)
                loss = cast(Tensor, tmloss(*embs))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Gradient before clipping
            nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

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
                sorted(enumerate(ds.targets), key=lambda x: x[1]),
                key=lambda idx_w_label: idx_w_label[1],
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
    epoch: int,
) -> Tuple[Tensor, Tensor, Tensor]:

    sample_num = min(epoch + 4, len(batch_classes) // 2)
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
                            sample_num,
                        ),
                    )
                    for c in batch_classes
                    for x, y in [
                        random.sample(cast(List, class_to_indices.get(c)) or [], 2)
                    ]
                ]
            )
        ),
    )

    with torch.no_grad():
        m.eval()
        a_imgs = torch.stack([ds[i][0] for i in anchors]).to(device)
        negs = torch.stack([ds[i][0] for fneg in negss for i in fneg]).to(
            device
        )  # (B*5, C, H, W)

        with autocast(device_type=device.type, dtype=torch.float16):
            a_embs = m.forward(a_imgs)  # (B, dim)
            n_embs = m.forward(negs).view(len(anchors), sample_num, -1)  # (B, SN, dim)

        # back to 32
        a_embs = a_embs.float()
        n_embs = n_embs.float()

        a_embs = torch.nn.functional.normalize(a_embs, p=2, dim=1)
        n_embs = torch.nn.functional.normalize(n_embs, p=2, dim=2)

        pass

    # (B, 1, dim) @ (B, dim, SN) -> (B, 1, SN)
    scores = torch.bmm(a_embs.unsqueeze(1), n_embs.transpose(1, 2)).squeeze(1)

    K = min(3, scores.size(1))
    topk_indices = scores.topk(K, dim=1).indices  # (B, K)

    rand_cols = torch.randint(0, K, (scores.size(0),), device=device)

    hard_idx = topk_indices[torch.arange(scores.size(0)), rand_cols]  # (B,)

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
