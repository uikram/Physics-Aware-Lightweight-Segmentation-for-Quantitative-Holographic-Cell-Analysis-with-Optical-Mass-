"""Low-rank adaptation (LoRA) with the injection and freezing rules of the submitted experiments.

Adapters (alpha = rank, so the scaling alpha / r = 1):
  LoRALinear   y = x W^T + b + (x A^T B^T) * alpha / r,          A: r x in (Kaiming), B: out x r (zeros)
  LoRAConv2d   y = conv(x, W) + B(A(x)) * alpha / r,             A: 1x1 conv in -> r, B: k x k conv r -> out with the
               original stride and padding (grouped convolutions are never adapted)
The base weights are frozen copies of the pretrained layer; LoRALinear always has a (frozen) bias vector, zero if the
original layer had none.

inject(module, strategy) on the part that is adapted (the encoder for EdgeSAM and MobileSAM, the whole network for
MobileNet-UNet):
  1. freeze every parameter of the part;
  2. replace the eligible layers (groups == 1):
       encoder_only     nn.Linear and nn.Conv2d whose name contains enc, features, blocks, neck, stage or patch_embed
       attention_blocks nn.Linear whose name contains q_proj, v_proj, query, value, attn.qkv, self_attn or qkv
  3. unfreeze: parameters whose name contains final_conv, mask_decoder, simple_decoder, prompt, outc or final_up;
     weight and bias of every BatchNorm2d / LayerNorm / GroupNorm; every remaining nn.Conv2d with one input channel.
Parameters outside the adapted part are not touched (the heads of EdgeSAM and MobileSAM stay trainable).
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_NAMES = ("final_conv", "mask_decoder", "simple_decoder", "prompt", "outc", "final_up")
ENCODER_NAMES = ("enc", "features", "blocks", "neck", "stage", "patch_embed")
ATTENTION_NAMES = ("q_proj", "v_proj", "query", "value", "attn.qkv", "self_attn", "qkv")
STRATEGIES = ("encoder_only", "attention_blocks")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.in_features, self.out_features, self.r = base.in_features, base.out_features, int(r)
        self.scaling = float(alpha) / self.r
        self.merged = False
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(base.bias.detach().clone() if base.bias is not None else torch.zeros(base.out_features),
                                 requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.merged:
            return y
        return y + (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merge(self) -> None:
        if not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if base.groups != 1:
            raise ValueError("grouped convolutions are not adapted")
        if tuple(base.dilation) != (1, 1):
            raise ValueError("dilated convolutions are not supported")
        k, s, p = base.kernel_size[0], base.stride[0], base.padding[0]
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding = base.in_channels, base.out_channels, k, s, p
        self.groups, self.r = 1, int(r)
        self.scaling = float(alpha) / self.r
        self.merged = False
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False) if base.bias is not None else None
        self.lora_A = nn.Conv2d(self.in_channels, self.r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(self.r, self.out_channels, kernel_size=k, stride=s, padding=p, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
        if self.merged:
            return y
        return y + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

    def merge(self) -> None:
        if not self.merged:
            a = self.lora_A.weight.data[:, :, 0, 0]                    # r x in
            self.weight.data += torch.einsum("orhw,ri->oihw", self.lora_B.weight.data, a) * self.scaling
            self.merged = True


def _eligible(name: str, module: nn.Module, strategy: str) -> bool:
    if isinstance(module, nn.Conv2d) and module.groups > 1:
        return False
    if strategy == "encoder_only":
        return isinstance(module, (nn.Linear, nn.Conv2d)) and any(k in name for k in ENCODER_NAMES)
    if strategy == "attention_blocks":
        return isinstance(module, nn.Linear) and any(k in name for k in ATTENTION_NAMES)
    raise ValueError(f"unknown LoRA strategy {strategy!r}; known: {STRATEGIES}")


def _set(root: nn.Module, name: str, new: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    if parts[-1].isdigit():
        parent[int(parts[-1])] = new
    else:
        setattr(parent, parts[-1], new)


def inject(part: nn.Module, r: int, alpha: float, dropout: float = 0.0, strategy: str = "encoder_only") -> int:
    """Insert adapters into `part` (in place) and apply the freezing rules. Returns the number of adapted layers."""
    for p in part.parameters():
        p.requires_grad = False
    targets = [(n, m) for n, m in part.named_modules() if n and _eligible(n, m, strategy)]
    for n, m in targets:
        _set(part, n, LoRALinear(m, r, alpha, dropout) if isinstance(m, nn.Linear) else LoRAConv2d(m, r, alpha, dropout))
    for n, p in part.named_parameters():
        if any(h in n for h in HEAD_NAMES):
            p.requires_grad = True
    for m in part.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for p in (m.weight, m.bias):
                if p is not None:
                    p.requires_grad = True
        if isinstance(m, nn.Conv2d) and m.in_channels == 1:
            for p in (m.weight, m.bias):
                if p is not None:
                    p.requires_grad = True
    return len(targets)


def merge(model: nn.Module) -> nn.Module:
    """Fold every adapter into its base weight (inference / benchmark) and drop the adapter modules."""
    for m in model.modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            m.merge()
            for attr in ("lora_A", "lora_B", "lora_dropout"):
                if hasattr(m, attr):
                    delattr(m, attr)
    return model


def count_parameters(model: nn.Module) -> dict:
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def parameter_report(model: nn.Module, rank=None) -> dict:
    """What is trained: total / trainable parameters, adapter (LoRA A and B) parameters, number of adapted layers and the
    adapted layer names. With LoRA, the adapter count grows linearly with the rank: r x (in + out) per Linear layer and
    r x (in + out x k x k) per Conv2d layer; the other trainable parameters (heads, norms, the 1-channel stem, and for
    MobileNet-UNet the whole decoder) do not depend on the rank. Full fine-tuning: every parameter trainable, no adapter."""
    named = list(model.named_parameters())
    total = sum(p.numel() for _, p in named); trainable = sum(p.numel() for _, p in named if p.requires_grad)
    lora_p = sum(p.numel() for n, p in named if ("lora_A" in n or "lora_B" in n) and p.requires_grad)
    layers = [n for n, m in model.named_modules() if isinstance(m, (LoRALinear, LoRAConv2d))]
    return {"rank": rank, "full_finetune": rank is None, "total": total, "trainable": trainable,
            "trainable_pct": 100.0 * trainable / max(total, 1), "lora_parameters": lora_p, "trainable_non_lora": trainable - lora_p,
            "lora_layers": len(layers), "lora_layer_names": layers}
