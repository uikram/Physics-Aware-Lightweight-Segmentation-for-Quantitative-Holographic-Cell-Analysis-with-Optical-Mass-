r"""Loss terms of the submitted manuscript (Eqs. 2-5), re-implemented. All terms act on the predicted foreground
probability p = 1 - softmax(logits)_background and on the referenced phase phi in RADIANS (never on the z-scored
network input). Pixels outside the annotated field of view (ignore label) are excluded from every term.

  Dice  class-weighted, Laplace-smoothed (+1) multi-class soft Dice; per image, averaged over the batch, then over
        classes with the class weights:  L = sum_c w_c (1 - (2 sum p_c g_c + 1) / (sum p_c + sum g_c + 1)) / sum_c w_c
  PMC   phase-mask contrast:  relu(mean_bg(phi) - mean_cell(phi) + margin), means weighted by 1 - p and p
  BGA   boundary-gradient alignment: L1 between the max-normalised gradient magnitudes of p and of phi (inside the
        eroded field of view)
  IPP   GLOBAL integrated-phase preservation:  |sum p phi - sum g phi| / (|sum g phi| + eps), one value per image
PMC, BGA and IPP are averaged over the images of the batch that contain annotated cells.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedDice(nn.Module):
    def __init__(self, num_classes: int, class_weights: Sequence[float], ignore_index: int):
        super().__init__()
        self.n, self.ignore = num_classes, ignore_index
        self.register_buffer("w", torch.tensor(list(class_weights), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = (target != self.ignore).float()
        prob = torch.softmax(logits.float(), dim=1)
        total = logits.new_zeros((), dtype=torch.float32)
        for c in range(self.n):
            p = prob[:, c] * valid
            g = (target == c).float() * valid
            dice = (2 * (p * g).sum(dim=(1, 2)) + 1.0) / (p.sum(dim=(1, 2)) + g.sum(dim=(1, 2)) + 1.0)
            total = total + self.w[c] * (1.0 - dice).mean()
        return total / (float(self.w.sum()) + 1e-4)


def _grad_mag(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1), value=0.0)
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1), value=0.0)
    return torch.sqrt(dx ** 2 + dy ** 2 + 1e-4)


def phase_mask_contrast(p: torch.Tensor, phi: torch.Tensor, valid: torch.Tensor, margin: float) -> torch.Tensor:
    """p, phi, valid: (B, H, W). Returns (B,)."""
    a_cell = (p * valid).sum(dim=(1, 2)).clamp(min=1.0)
    a_bg = ((1 - p) * valid).sum(dim=(1, 2)).clamp(min=1.0)
    mu_cell = (p * phi * valid).sum(dim=(1, 2)) / a_cell
    mu_bg = ((1 - p) * phi * valid).sum(dim=(1, 2)) / a_bg
    return F.relu(mu_bg - mu_cell + margin)


def boundary_gradient_alignment(p: torch.Tensor, phi: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    v = valid.unsqueeze(1)
    v_er = -F.max_pool2d(-v, kernel_size=3, stride=1, padding=1)          # gradients need valid neighbours
    gm, gp = _grad_mag(p.unsqueeze(1)) * v_er, _grad_mag(phi.unsqueeze(1)) * v_er
    gm = gm / (gm.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
    gp = gp / (gp.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
    return ((gm - gp).abs() * v_er).sum(dim=(1, 2, 3)) / v_er.sum(dim=(1, 2, 3)).clamp(min=1.0)


def global_ipp(p: torch.Tensor, g: torch.Tensor, phi: torch.Tensor, valid: torch.Tensor, eps: float) -> torch.Tensor:
    s_pred = (p * phi * valid).sum(dim=(1, 2)); s_gt = (g * phi * valid).sum(dim=(1, 2))
    return (s_pred - s_gt).abs() / (s_gt.abs() + eps)
