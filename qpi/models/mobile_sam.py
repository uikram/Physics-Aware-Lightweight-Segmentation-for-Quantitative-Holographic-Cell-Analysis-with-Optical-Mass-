"""
MobileSAM: Lightweight adaptation of Segment Anything Model for QPI.
Uses TinyViT as image encoder (pretrained), adapted for single-channel input.
LoRA is applied to attention q/v projections in the transformer blocks.

Requires the ``mobile_sam`` package (see requirements.txt)
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from qpi.models.lora_utils import inject_lora_into_model, merge_lora_weights, get_lora_parameters


class QPIPromptEncoder(nn.Module):
    """
    Simple prompt encoder for QPI: no text/point prompts needed.
    Returns a fixed learned embedding as the prompt.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.no_mask_embed = nn.Embedding(1, embed_dim)
        self.embed_dim     = embed_dim

    def forward(self, batch_size: int, device: torch.device):
        return self.no_mask_embed.weight.unsqueeze(0).expand(batch_size, -1, -1).to(device)


class LightweightMaskDecoder(nn.Module):
    """
    Lightweight mask decoder replacing SAM's transformer decoder.
    Input: image embeddings (B, C, H, W) + prompt embedding (B, 1, C)
    Output: mask logits (B, num_classes, H, W)
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 1,
                 hidden_dim: int = 128):
        super().__init__()

        self.iou_token    = nn.Embedding(1, in_channels)
        self.mask_tokens  = nn.Embedding(num_classes, in_channels)

        self.transformer  = nn.TransformerDecoderLayer(
            d_model=in_channels, nhead=8, dim_feedforward=hidden_dim,
            dropout=0.0, batch_first=True
        )

    def forward(self, image_embeddings: torch.Tensor,
                prompt_embeddings: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image_embeddings.shape
        
        # Flatten spatial dims for transformer
        img_seq = image_embeddings.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        mask_tok = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Integrate the QPI learned prompt into the mask tokens
        mask_tok = mask_tok + prompt_embeddings 
        
        out = self.transformer(mask_tok, img_seq)  # (B, num_classes, C)
        
        upscaled = F.interpolate(image_embeddings, scale_factor=4,
                                 mode="bilinear", align_corners=False)  # (B, C, 4H, 4W)
        
        # Project each spatial position using class token weights
        logits = torch.einsum('bnc,bchw->bnhw', out, upscaled)  # (B, num_classes, 4H, 4W)
        return logits


class MobileSAMSeg(nn.Module):
    """MobileSAM (TinyViT) encoder + lightweight transformer mask decoder for single-channel QPI.

    The encoder always runs at 1024 x 1024 (TinyViT requirement); logits are resized back to the input size.
    There is no silent fallback: unloadable weights raise.
    """

    EMBED_DIM = 256
    ENCODER_INPUT_SIZE = 1024

    def __init__(self, num_classes: int = 5, weights_path: Optional[Path] = None):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_input_size = self.ENCODER_INPUT_SIZE
        self.encoder = self._build_encoder(weights_path)
        self.prompt_encoder = QPIPromptEncoder(embed_dim=self.EMBED_DIM)
        self.mask_decoder = LightweightMaskDecoder(in_channels=self.EMBED_DIM, num_classes=num_classes)
        self._lora_injected = False

    @staticmethod
    def _build_encoder(weights_path: Optional[Path]) -> nn.Module:
        import warnings
        with warnings.catch_warnings():     # the mobile_sam package itself still imports the deprecated timm.models.layers
            warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated", category=FutureWarning)
            warnings.filterwarnings("ignore", message="Overwriting tiny_vit", category=UserWarning)
            from mobile_sam import sam_model_registry
        if weights_path is not None and not Path(weights_path).is_file():
            raise FileNotFoundError(f"MobileSAM checkpoint not found: {weights_path}")
        sam = sam_model_registry["vit_t"](checkpoint=str(weights_path) if weights_path is not None else None)
        encoder = sam.image_encoder
        if not hasattr(encoder.patch_embed, "seq"):
            raise AttributeError("unexpected MobileSAM patch embedding; TinyViT stem not found")
        orig = encoder.patch_embed.seq[0].c
        new = nn.Conv2d(1, orig.out_channels, kernel_size=orig.kernel_size, stride=orig.stride,
                        padding=orig.padding, bias=orig.bias is not None)
        new.weight.data = orig.weight.data.mean(dim=1, keepdim=True)
        if orig.bias is not None:
            new.bias.data = orig.bias.data.clone()
        encoder.patch_embed.seq[0].c = new
        return encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, orig_size = x.shape[0], x.shape[2:]
        if x.shape[-1] != self.encoder_input_size or x.shape[-2] != self.encoder_input_size:
            x = F.interpolate(x, size=(self.encoder_input_size,) * 2, mode="bilinear", align_corners=False)
        emb = self.encoder(x)
        if emb.dim() == 4 and emb.shape[-1] == self.EMBED_DIM and emb.shape[1] != self.EMBED_DIM:
            emb = emb.permute(0, 3, 1, 2)
        decoded = self.mask_decoder(emb, self.prompt_encoder(B, x.device))
        return F.interpolate(decoded, size=orig_size, mode="bilinear", align_corners=False)

    def inject_lora(self, r: int, lora_alpha: float, lora_dropout: float = 0.0, strategy: str = "attention_blocks"):
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
