"""MobileSAM segmentation network (architecture of the submitted manuscript).

Encoder: the TinyViT image encoder of the official `mobile_sam` package (sam_model_registry["vit_t"]), pretrained
         weights mobile_sam.pt; its first 3-channel convolution (patch_embed.seq[0].c) becomes 1-channel with the mean
         of the RGB filters. It always runs at 1024 x 1024 (TinyViT requirement) and returns a (B, 256, 64, 64) embedding.
Prompt:  one learned 256-d embedding (no point / box / text prompts).
Head:    one mask token per class plus the prompt embedding, one nn.TransformerDecoderLayer (d_model 256, 8 heads,
         feed-forward 128, no dropout, batch_first) attending to the flattened embedding; the embedding is bilinearly
         upsampled x 4 (256 x 256) and every class logit map is the dot product of its token with the upsampled
         embedding. An IoU token (unused, as in the submitted model) is kept so that the parameter set is identical.
Input (B, 1, H, W) is resized to 1024; logits (256 x 256) are resized back to (H, W).
LoRA: strategy attention_blocks, inserted into the encoder only; prompt and head are trained fully.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qpicell.models import lora

EMBED_DIM = 256
ENCODER_INPUT = 1024


class PromptEmbedding(nn.Module):
    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.no_mask_embed = nn.Embedding(1, dim)

    def forward(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.no_mask_embed.weight.unsqueeze(0).expand(batch, -1, -1).to(device)


class TokenMaskDecoder(nn.Module):
    def __init__(self, dim: int = EMBED_DIM, num_classes: int = 5, hidden: int = 128):
        super().__init__()
        self.iou_token = nn.Embedding(1, dim)
        self.mask_tokens = nn.Embedding(num_classes, dim)
        self.transformer = nn.TransformerDecoderLayer(d_model=dim, nhead=8, dim_feedforward=hidden, dropout=0.0, batch_first=True)

    def forward(self, emb: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        B = emb.shape[0]
        seq = emb.flatten(2).permute(0, 2, 1)                                   # (B, HW, C)
        tokens = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1) + prompt
        tokens = self.transformer(tokens, seq)                                  # (B, classes, C)
        up = F.interpolate(emb, scale_factor=4, mode="bilinear", align_corners=False)
        return torch.einsum("bnc,bchw->bnhw", tokens, up)


def build_encoder(weights: Optional[Path]) -> nn.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from mobile_sam import sam_model_registry
    if weights is not None and not Path(weights).is_file():
        raise FileNotFoundError(f"MobileSAM weights not found: {weights}")
    sam = sam_model_registry["vit_t"](checkpoint=str(weights) if weights is not None else None)
    enc = sam.image_encoder
    conv = enc.patch_embed.seq[0].c
    new = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    new.weight.data = conv.weight.data.mean(dim=1, keepdim=True)
    if conv.bias is not None:
        new.bias.data = conv.bias.data.clone()
    enc.patch_embed.seq[0].c = new
    return enc


class MobileSAMSegmenter(nn.Module):
    def __init__(self, num_classes: int = 5, weights: Optional[Path] = None):
        super().__init__()
        self.encoder = build_encoder(weights)
        self.prompt_encoder = PromptEmbedding(EMBED_DIM)
        self.mask_decoder = TokenMaskDecoder(EMBED_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        if tuple(size) != (ENCODER_INPUT, ENCODER_INPUT):
            x = F.interpolate(x, size=(ENCODER_INPUT, ENCODER_INPUT), mode="bilinear", align_corners=False)
        emb = self.encoder(x)
        if emb.dim() == 4 and emb.shape[-1] == EMBED_DIM and emb.shape[1] != EMBED_DIM:
            emb = emb.permute(0, 3, 1, 2)
        y = self.mask_decoder(emb, self.prompt_encoder(x.shape[0], x.device))
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)

    def add_lora(self, r: int, alpha: float, dropout: float = 0.0, strategy: str = "attention_blocks") -> int:
        return lora.inject(self.encoder, r, alpha, dropout, strategy)
