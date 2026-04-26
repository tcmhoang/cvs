from typing import Callable, List, Tuple, cast

import torch
from pytorch_metric_learning import losses
from torch import (
    device,
    nn,
    optim,
)
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

import config
from dataset import ImageFolder
from model import RetrievalNet
from proc import Logger


def retrieval_model(
    m: RetrievalNet,
    train_dataset_supplier: Callable[[], ImageFolder],
    device: device,
    logger: Logger,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SZ,
    lr=config.LEARNING_RATE,
):

    m.train()

    train_dataset = train_dataset_supplier()

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    def get_optimizer_params(weight_decay=0.05) -> List[dict]:
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

    loss_fn = losses.ProxyAnchorLoss(
        num_classes=len(train_dataset.classes),
        embedding_size=config.EMBEDDING_DIM,
        margin=config.PA_MARGIN,
        alpha=config.PA_A,
    ).to(device)

    params = get_optimizer_params()
    params.append(
        {"params": loss_fn.parameters(), "lr": lr * 100, "weight_decay": 1e-4}
    )

    optimizer = optim.AdamW(params)

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

            torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), max_norm=1.0)
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
