from typing import List, Tuple, cast

import torch
from pytorch_metric_learning import losses, samplers
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
    train_dataset: ImageFolder,
    device: device,
    logger: Logger,
    epochs=20,
    batch_size=16,
    lr=1e-4,
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

    msloss = losses.MultiSimilarityLoss(
        alpha=config.MS_ALPHA, beta=config.MS_BETA, base=config.MS_BASE
    )

    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    for e in range(epochs):
        eloss = 0.0

        sampler = samplers.MPerClassSampler(
            labels=train_dataset.targets,
            m=config.MS_K,
            batch_size=batch_size,
            length_before_new_iter=len(train_dataset),
        )

        loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            m.train()
            optimizer.zero_grad()

            with autocast(device_type=device.type, dtype=torch.float16):
                embs = m.forward(imgs)
                pass

            loss = msloss.forward(embs.float(), labels)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)  # Gradient before clipping
            nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            eloss += loss.item()

            pass

        scheduler.step()

        avg_loss = eloss / len(loader)
        print(f"e {e + 1}/{epochs} aloss: {avg_loss:.4f}")

        logger.log({"epoch": e + 1, "train_loss": avg_loss})
        pass

    return m
