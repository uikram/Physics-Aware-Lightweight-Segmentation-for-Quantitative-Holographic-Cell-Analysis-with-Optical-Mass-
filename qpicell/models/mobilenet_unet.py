"""MobileNet-UNet (architecture of the submitted manuscript).

Encoder: torchvision MobileNetV2 (ImageNet weights from weights/mobilenet_v2-b0353104.pth), first convolution
         1-channel with the mean of the RGB filters; stages enc0 = features[0] (/2), enc1 = features[1] (/2, 16 ch),
         enc2 = features[2:4] (/4, 24), enc3 = features[4:7] (/8, 32), enc4 = features[7:14] (/16, 96),
         enc5 = features[14:] (/32, 1280).
Decoder: dec4..dec1, each ConvTranspose2d(k=2, s=2, channels / 2) -> concat skip -> Conv3x3 (no bias) -> BN -> ReLU6,
         with 256, 128, 64, 32 output channels and skips enc4, enc3, enc2, enc1; final_up ConvTranspose2d(32, 32, 2, 2);
         bilinear resize to the input size; final_conv 1 x 1 to the class logits.
LoRA: strategy encoder_only on the whole network; afterwards every parameter whose name contains dec, up or final is
      trained (decoder, upsampling and classifier).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qpicell.models import lora


class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, kernel=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=padding, bias=False), nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin // 2, kernel_size=2, stride=2)
        self.conv = ConvBNReLU(cin // 2 + cskip, cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MobileNetUNet(nn.Module):
    def __init__(self, num_classes: int = 5, weights: Optional[Path] = None, channels=(256, 128, 64, 32)):
        super().__init__()
        from torchvision.models import mobilenet_v2
        bb = mobilenet_v2(weights=None)
        if weights is not None:
            if not Path(weights).is_file():
                raise FileNotFoundError(f"MobileNetV2 ImageNet weights not found: {weights}")
            bb.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        c0 = bb.features[0][0]
        stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        stem.weight.data = c0.weight.data.mean(dim=1, keepdim=True)
        bb.features[0][0] = stem
        f = bb.features
        self.enc0, self.enc1 = f[0], f[1]
        self.enc2, self.enc3 = nn.Sequential(*f[2:4]), nn.Sequential(*f[4:7])
        self.enc4, self.enc5 = nn.Sequential(*f[7:14]), nn.Sequential(*f[14:])
        self.dec4 = UpBlock(1280, 96, channels[0])
        self.dec3 = UpBlock(channels[0], 32, channels[1])
        self.dec2 = UpBlock(channels[1], 24, channels[2])
        self.dec1 = UpBlock(channels[2], 16, channels[3])
        self.final_up = nn.ConvTranspose2d(channels[3], channels[3], kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(channels[3], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x); e1 = self.enc1(e0); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3); e5 = self.enc5(e4)
        d = self.dec1(self.dec2(self.dec3(self.dec4(e5, e4), e3), e2), e1)
        d = F.interpolate(self.final_up(d), size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.final_conv(d)

    def add_lora(self, r: int, alpha: float, dropout: float = 0.0, strategy: str = "encoder_only") -> int:
        n = lora.inject(self, r, alpha, dropout, strategy)
        for name, p in self.named_parameters():
            if any(s in name for s in ("dec", "up", "final")):
                p.requires_grad = True
        return n
