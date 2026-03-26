from torch import Tensor, hub, nn, device, ones
from typing import cast, Protocol, Any
from timm.models.vision_transformer import VisionTransformer


class DINOv2ViT(Protocol):
    blocks: nn.ModuleList
    norm: nn.LayerNorm

    def parameters(self) -> Any: ...
    def to(self, device: device) -> "DINOv2ViT": ...


def io_get_model(dev: device) -> VisionTransformer:
    dinov2_model = cast(
        DINOv2ViT, hub.load("facebookresearch/dinov2", "dinov2_vits14")
    ).to(dev)

    for param in dinov2_model.parameters():
        param.requires_grad = False

    for param in dinov2_model.blocks[-1].parameters():
        param.requires_grad = True

    for param in dinov2_model.norm.parameters():
        param.requires_grad = True

    return cast(VisionTransformer, dinov2_model)


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6) -> None:
        super(GeM, self).__init__()

        self.p = nn.Parameter(ones(1) * p)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # [Batch, Num_Patches, Embedding_Dim]
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)


class RetrievalNet(nn.Module):
    def __init__(self, model: VisionTransformer, embeding_dim=384) -> None:
        super().__init__()

        self.model = model
        self.gem = GeM()

        self.fc = nn.Sequential(
            nn.Linear(384, 512), nn.ReLU(), nn.Linear(512, embeding_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        fdicts = self.model.forward_features(x)
        patches = fdicts[:, 1:, :]  # [Batch, 256, 384]
        pooled = self.gem(patches)
        embeddings = self.fc(pooled)

        return nn.functional.normalize(embeddings, p=2, dim=1)
