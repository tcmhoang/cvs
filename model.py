from typing import Any, Callable, Protocol, cast

import torch
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, device, hub, nn

import config


class DINOv2ViT(Protocol):
    blocks: nn.ModuleList
    norm: nn.LayerNorm

    def parameters(self) -> Any: ...
    def to(self, device: device) -> "DINOv2ViT": ...


def model_freeze_backbone(base: DINOv2ViT) -> None:
    for param in base.parameters():
        param.requires_grad = False
        pass
    pass


def model_pft(base: DINOv2ViT) -> None:
    model_freeze_backbone(base)

    for param in base.blocks[-1].parameters():
        param.requires_grad = True
        pass

    for param in base.norm.parameters():
        param.requires_grad = True
        pass

    pass


def io_get_model(
    dev: device, mod: Callable[[DINOv2ViT], None] = model_pft
) -> VisionTransformer:
    dinov2_model = cast(
        DINOv2ViT, hub.load("facebookresearch/dinov2", "dinov2_vits14")
    ).to(dev)

    mod(dinov2_model)

    return cast(VisionTransformer, dinov2_model)


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6) -> None:
        super(GeM, self).__init__()

        self.p = nn.Parameter(torch.full((1,), float(p)))
        self.eps = eps
        pass

    def forward(self, x: Tensor) -> Tensor:  # [Batch, Num_Patches, Embedding_Dim]
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)


class RetrievalNet(nn.Module):
    def __init__(
        self,
        model: VisionTransformer,
        gemp=config.GEM_P,
        embeding_dim=config.EMBEDDING_DIM,
    ) -> None:
        super().__init__()

        self.model = model
        self.gem = GeM(p=gemp)

        self.fc = nn.Sequential(
            nn.Linear(384, config.MLP_HEAD_DIM),
            nn.ReLU(),
            nn.Linear(config.MLP_HEAD_DIM, embeding_dim),
        )
        pass

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            fdicts = self.model.forward_features(x)

            if isinstance(fdicts, dict):
                patches = fdicts.get("x_norm_patchtokens")
                if patches is None:
                    features = fdicts.get("x_norm")
                    patches = cast(Tensor, features)[:, 1:, :]  # [Batch, 256, 384]
                    pass
            else:
                features = fdicts
                patches = features[:, 1:, :]  # [Batch, 256, 384]
                pass

            pass

        pooled = self.gem(patches)
        embeddings = self.fc(pooled)

        return nn.functional.normalize(pooled + embeddings, p=2, dim=1)


class LinearProjectionNet(nn.Module):
    def __init__(self, model: VisionTransformer, embedding_dim=config.EMBEDDING_DIM):
        super().__init__()
        self.model = model
        self.head = nn.Linear(384, embedding_dim)
        pass

    def forward(self, x):
        with torch.no_grad():
            fdicts = self.model.forward_features(x)
            clstokens = fdicts["x_norm_clstoken"]  # type: ignore

        projected_features = self.head(clstokens)

        return nn.functional.normalize(projected_features, p=2, dim=1)
