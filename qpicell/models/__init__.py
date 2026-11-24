"""Model factory: build(cfg, arch, rank) -> network with LoRA of the given rank (alpha = rank), or rank=None -> full
fine-tuning (no adapters, every parameter trainable)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch.nn as nn

ARCHITECTURES = ("edge_sam", "mobile_sam", "mobilenet_unet")


def build(cfg: Dict, arch: str, rank: Optional[int], pretrained: bool = True) -> nn.Module:
    m = cfg["models"][arch]; n = cfg["data"]["num_classes"]
    w = Path(cfg["paths_abs"]["weights"]) / m["weights"] if pretrained else None
    if arch == "edge_sam":
        from qpicell.models.edgesam import EdgeSAMSegmenter
        net = EdgeSAMSegmenter(n, m["image_size"], w)
    elif arch == "mobile_sam":
        from qpicell.models.mobilesam import MobileSAMSegmenter
        net = MobileSAMSegmenter(n, w)
    elif arch == "mobilenet_unet":
        from qpicell.models.mobilenet_unet import MobileNetUNet
        net = MobileNetUNet(n, w)
    else:
        raise ValueError(f"unknown architecture {arch!r}; known: {ARCHITECTURES}")
    if rank is not None:                       # LoRA of this rank (alpha = rank); rank None = full fine-tuning
        net.n_lora_layers = net.add_lora(int(rank), float(rank), float(cfg["conditions"]["lora_dropout"]), m["lora_strategy"])
    else:
        for p in net.parameters():             # full fine-tuning: every parameter is trained
            p.requires_grad = True
        net.n_lora_layers = 0
    return net
