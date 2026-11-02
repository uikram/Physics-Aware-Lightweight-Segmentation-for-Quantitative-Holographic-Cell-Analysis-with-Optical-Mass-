"""Model registry."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from qpi.models.edge_sam import EdgeSAMSeg
from qpi.models.lora_utils import inject_lora_into_model, merge_lora_weights
from qpi.models.mobile_sam import MobileSAMSeg
from qpi.models.mobilenet_unet import MobileNetUNet


def build_model(run_cfg: Dict[str, Any], weights_dir: Optional[Path], pretrained: bool = True):
    """Instantiate the architecture of ``run_cfg`` (+ LoRA when configured). Never falls back silently."""
    arch, n = run_cfg["arch"], run_cfg["num_classes"]
    w = run_cfg.get("pretrained_weights")
    wp = (Path(weights_dir) / w) if (pretrained and w and weights_dir) else None
    if pretrained and w and wp is None:
        raise ValueError("pretrained weights requested but no weights_dir given")
    if arch == "edge_sam":
        model = EdgeSAMSeg(num_classes=n, image_size=run_cfg["image_size"], weights_path=wp)
    elif arch == "mobile_sam":
        model = MobileSAMSeg(num_classes=n, weights_path=wp)
    elif arch == "mobilenet_unet":
        model = MobileNetUNet(num_classes=n, pretrained=pretrained)
    else:
        raise ValueError(f"unknown architecture '{arch}'")
    lora = run_cfg.get("lora")
    if lora is not None:
        model.inject_lora(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                          strategy=lora["insertion_strategy"])
    return model


__all__ = ["EdgeSAMSeg", "MobileSAMSeg", "MobileNetUNet", "build_model", "inject_lora_into_model",
           "merge_lora_weights"]
