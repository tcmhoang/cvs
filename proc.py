from typing import Dict, Protocol, Any

from torch import nn, device


class Logger(Protocol):
    def log(self, data: Dict[str, Any]) -> None: ...

    pass


class DINOv2ViT(Protocol):
    blocks: nn.ModuleList
    norm: nn.LayerNorm

    def parameters(self) -> Any: ...
    def to(self, device: device) -> "DINOv2ViT": ...
