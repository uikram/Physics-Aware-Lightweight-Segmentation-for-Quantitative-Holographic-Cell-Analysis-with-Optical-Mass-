"""EdgeSAM-based segmentation network for single-channel quantitative phase images.

Encoder: the official EdgeSAM RepViT image encoder (vendored, see models/vendor/edge_sam), pretrained weights
loaded from the official checkpoint, first convolution averaged to 1 input channel.
Decoder: four transposed-convolution stages followed by a 1x1 classifier.

There is deliberately no silent fallback: if the pretrained weights are requested but cannot be loaded the
constructor raises, so a run can never train a different architecture than the one configured.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qpi.models.lora_utils import get_lora_parameters, inject_lora_into_model, merge_lora_weights
from qpi.models.vendor.edge_sam import RepViT


def build_edge_encoder(weights_path: Optional[Path]) -> nn.Module:
    encoder = RepViT(arch="m1", img_size=1024, upsample_mode="bicubic", fuse=True)
    if weights_path is not None:
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"EdgeSAM checkpoint not found: {weights_path}")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state and not any(k.startswith("image_encoder.") for k in state):
            state = state["model"]
        enc_state = {k[len("image_encoder."):]: v for k, v in state.items() if k.startswith("image_encoder.")}
        if not enc_state:
            raise RuntimeError(f"{weights_path} contains no 'image_encoder.' tensors")
        missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"EdgeSAM encoder weights do not match: missing={missing[:3]} unexpected={unexpected[:3]}")
    # adapt the 3-channel stem to 1 input channel (average of the RGB filters)
    parent, name, conv = None, None, None
    for mod_name, module in encoder.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            parts = mod_name.split(".")
            parent = encoder
            for p in parts[:-1]:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            name, conv = parts[-1], module
            break
    if conv is None:
        raise AttributeError("could not find the 3-channel stem convolution of the EdgeSAM encoder")
    new = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    new.weight.data = conv.weight.data.mean(dim=1, keepdim=True)
    if conv.bias is not None:
        new.bias.data = conv.bias.data.clone()
    if name.isdigit():
        parent[int(name)] = new
    else:
        setattr(parent, name, new)
    return encoder


class EdgeSAMSeg(nn.Module):
    """Input (B, 1, H, W) phase -> logits (B, num_classes, H, W)."""

    ENCODER_CHANNELS = 256

    def __init__(self, num_classes: int = 5, image_size: int = 256, weights_path: Optional[Path] = None):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.encoder = build_edge_encoder(weights_path)
        c = self.ENCODER_CHANNELS
        self.simple_decoder = nn.Sequential(
            nn.ConvTranspose2d(c, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )
        self._lora_injected = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        logits = self.simple_decoder(self.encoder(x))
        if logits.shape[2:] != size:
            logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return logits

    def inject_lora(self, r: int, lora_alpha: float, lora_dropout: float = 0.0, strategy: str = "encoder_only"):
        inject_lora_into_model(self.encoder, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, strategy=strategy)
        self._lora_injected = True
        return self

    def merge_lora(self):
        merge_lora_weights(self.encoder)
        return self

    def get_lora_params(self):
        return get_lora_parameters(self)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
