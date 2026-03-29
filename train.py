from itertools import groupby
from torch import (
    Tensor,
    cosine_similarity,
    nn,
    device,
    no_grad,
    optim,
    stack,
)
from typing import Dict, cast, List, Tuple
import random


from dataset import ImageFolder
from model import RetrievalNet


def retrieval_model(
    m: RetrievalNet,
    train_dataset: ImageFolder,
    device: device,
    epochs=20,
    batch_size=16,
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

        print(f"e {e} aloss: {eloss / len(batches):.4f}")
        # TODO: use wanb
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
                pass
            pass

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
