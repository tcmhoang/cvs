from typing import List, cast

import faiss
import torch
from numpy._typing import NDArray
from PIL import Image
from torch import Tensor, device
from torch.amp.autocast_mode import autocast
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

from model import RetrievalNet


class Inference:
    def __init__(
        self,
        wmodel: RetrievalNet,
        device: device,
        testdb: ImageFolder,
        trans: Compose,
        feats: NDArray,
    ) -> None:
        self.device = device
        self.database = testdb
        self.transform = trans

        self.model = wmodel.to(device)
        self.model.eval()

        faiss.normalize_L2(feats)
        self.index = faiss.IndexFlatIP(feats.shape[1])
        self.index.add(feats)  # type: ignore
        pass

    def retrieve(self, image_path: str, top_k: int = 5) -> List:
        img = Image.open(image_path).convert("RGB")
        img_tensor = cast(Tensor, self.transform(img)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with autocast(device_type=self.device.type, dtype=torch.float16):
                emb = self.model.forward(img_tensor).float()
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                pass
            pass

        emb_np = emb.cpu().numpy()
        distances, indices = self.index.search(emb_np, top_k)  # type: ignore

        results = []
        for i in range(top_k):
            idx = indices[0][i]
            score = distances[0][i]

            # .samples is a PyTorch list containing tuples of (file_path, class_index)
            matched_img_path, matched_label_idx = self.database.samples[idx]
            matched_class_name = self.database.classes[matched_label_idx]

            results.append(
                {
                    "path": matched_img_path,
                    "class": matched_class_name,
                    "confidence": float(score),
                }
            )

        return results
