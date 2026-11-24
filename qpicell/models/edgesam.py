"""EdgeSAM segmentation network (architecture of the submitted manuscript).

Encoder: the official EdgeSAM RepViT-m1 image encoder (third_party/edge_sam, img_size 1024, bicubic upsampling, fused
         neck), pretrained weights = the image_encoder.* tensors of edge_sam_3x.pth, loaded strictly. The first
         3-channel convolution is replaced by a 1-channel one whose weight is the mean of the RGB filters.
Head:    4 x [ConvTranspose2d(k=2, s=2) -> BatchNorm2d -> ReLU6] (256 -> 128 -> 64 -> 32 -> 16 channels) and a 1 x 1
         convolution to the class logits. The encoder downsamples by 16, the head upsamples by 16.
Input (B, 1, H, W) is resized to image_size (256) if needed; logits are resized back to (H, W).
LoRA: strategy encoder_only, inserted into the encoder only; the head is trained fully.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qpicell.models import lora

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from third_party.edge_sam import RepViT  # noqa: E402

EMBED_CHANNELS = 256


def _single_channel_stem(encoder: nn.Module) -> None:
    for name, m in encoder.named_modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            new = nn.Conv2d(1, m.out_channels, m.kernel_size, m.stride, m.padding, bias=m.bias is not None)
            new.weight.data = m.weight.data.mean(dim=1, keepdim=True)
            if m.bias is not None:
                new.bias.data = m.bias.data.clone()
            lora._set(encoder, name, new)
            return
    raise AttributeError("no 3-channel stem convolution in the EdgeSAM encoder")


def build_encoder(weights: Optional[Path]) -> nn.Module:
    enc = RepViT(arch="m1", img_size=1024, upsample_mode="bicubic", fuse=True)
    if weights is not None:
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"EdgeSAM weights not found: {weights}")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state and not any(k.startswith("image_encoder.") for k in state):
            state = state["model"]
        enc_state = {k[len("image_encoder."):]: v for k, v in state.items() if k.startswith("image_encoder.")}
        if not enc_state:
            raise RuntimeError(f"{weights} contains no image_encoder.* tensors")
        enc.load_state_dict(enc_state, strict=True)
    _single_channel_stem(enc)
    return enc


class EdgeSAMSegmenter(nn.Module):
    def __init__(self, num_classes: int = 5, image_size: int = 256, weights: Optional[Path] = None):
        super().__init__()
        self.image_size = int(image_size)
        self.encoder = build_encoder(weights)
        c = EMBED_CHANNELS
        self.simple_decoder = nn.Sequential(
            nn.ConvTranspose2d(c, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        if tuple(size) != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        y = self.simple_decoder(self.encoder(x))
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False) if y.shape[-2:] != size else y

    def add_lora(self, r: int, alpha: float, dropout: float = 0.0, strategy: str = "encoder_only") -> int:
        return lora.inject(self.encoder, r, alpha, dropout, strategy)
