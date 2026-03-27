from timm.models.vision_transformer import VisionTransformer
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch import Tensor, hub, nn, device, ones, optim
from typing import cast, Protocol, Any, List

from dio import Tuple
from model import RetrievalNet


def train_retrieval_model(
    m: RetrievalNet,
    train_dataset: DataLoader,
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
            # TODO: triplet batch
            pass
