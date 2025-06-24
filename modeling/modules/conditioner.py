import kornia
import open_clip
import torch
from torch import nn


class CLIPConditioner(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self):
        super().__init__()
        self.module = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )[0]
        self.module.eval().requires_grad_(False)  # type: ignore
        self.register_buffer(
            "mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        self.register_buffer(
            "std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=True,
        )
        x = (x + 1.0) / 2.0
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.module.encode_image(x)
        return x
