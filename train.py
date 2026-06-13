from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Tuple,
    cast,
    Protocol,
)

import torch
from pytorch_metric_learning import losses, distances
from torch import (
    device,
    nn,
    optim,
)
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.optimizer import ParamsT
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
from dataset import ImageFolder
from model import RetrievalNet
from proc import Logger

import numpy as np


class LossFunction(Protocol):
    def parameters(self) -> Iterator[nn.Parameter]: ...
    def to(self, device: device) -> "LossFunction": ...
    def forward(
        self,
        embeddings: Any,
        labels: Any | None = None,
    ) -> Any: ...

    pass


class LossPack(NamedTuple):
    fn: LossFunction
    optim_params: Callable[[nn.Module, float], ParamsT]
    amp_clip: bool

    pass


def pa_loss(numclsses: int) -> LossPack:
    loss_fn = losses.ProxyAnchorLoss(
        num_classes=numclsses,
        embedding_size=config.EMBEDDING_DIM,
        margin=config.PA_MARGIN,
        alpha=config.PA_A,
    )

    return LossPack(
        loss_fn,
        lambda m, lr: get_opt_params(
            m,
            lr,
            [{"params": loss_fn.parameters(), "lr": lr * 100, "weight_decay": 1e-4}],
        ),
        True,
    )


def coscons_loss() -> LossPack:
    return LossPack(
        losses.ContrastiveLoss(
            pos_margin=1, neg_margin=0, distance=distances.CosineSimilarity()
        ),
        lambda m, _: [
            {"params": m.head.parameters(), "lr": 1e-3, "weight_decay": 1e-4}
        ],
        False,
    )


def get_opt_params(m: RetrievalNet, lr: float, params: List[Dict[Any, Any]]) -> ParamsT:
    def go(weight_decay=0.05) -> List[dict]:
        blocks = cast(nn.ModuleList, m.model.blocks)
        backbone = [blocks[-1], m.model.norm]
        head = [m.gem, m.fc]

        m_w_lrc = [(backbone, 0.1), (head, 5)]

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

    ps = go()

    for p in params:
        ps.append(p)
        pass

    return ps


def retrieval_model(
    m: nn.Module,
    train_dataset_supplier: Callable[[], ImageFolder],
    device: device,
    logger: Logger,
    fetch_losspck: Callable[[int], LossPack] = pa_loss,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SZ,
    lr=config.LEARNING_RATE,
):

    m.train()

    train_dataset = train_dataset_supplier()

    targets = train_dataset.targets
    sample_weights_np = 1.0 / np.bincount(targets)
    sample_weights = [float(sample_weights_np[t]) for t in targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    num_classes = len(train_dataset.classes)

    losspck = fetch_losspck(num_classes)
    loss_fn = losspck.fn.to(device)

    optimizer = optim.AdamW(losspck.optim_params(m, lr))

    wepochs = int(config.WARM_UP_PERC * epochs)

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.WARM_UP_FACTOR,
        total_iters=wepochs,
    )
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - wepochs
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[wepochs]
    )

    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    for e in range(epochs):
        eloss = 0.0

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type, dtype=torch.float16):
                embs = m.forward(imgs)
                loss = loss_fn.forward(embs.float(), labels)
                pass

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Gradient before clipping

            if losspck.amp_clip:
                torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), max_norm=1.0)
                pass

            nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            eloss += loss.item()

            pass

        scheduler.step()

        avg_loss = eloss / len(loader)
        clr = optimizer.param_groups[0]["lr"]
        print(f"e {e + 1}/{epochs} aloss: {avg_loss:.4f} base_lr: {clr:.6f}")

        logger.log({"epoch": e + 1, "train_loss": avg_loss, "base_lr": clr})
        pass

    return m
