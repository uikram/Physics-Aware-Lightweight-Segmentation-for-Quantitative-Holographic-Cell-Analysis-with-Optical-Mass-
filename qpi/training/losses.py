"""Physics-Aware Phase Consistency Loss (manuscript Eqs. 2-6) with support for unannotated pixels.

    L = L_Dice + lambda_pmc * L_PMC + lambda_bga * L_BGA + lambda_pv * L_IPP

All physics terms act on the binary foreground probability 1 - p(background) and on the RAW phase map
(radians, never the z-scored network input). Pixels labelled ``ignore_index`` (outside the annotated field
of view) are excluded from every term. Losses are evaluated in float32 irrespective of autocast.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid(target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return (target != ignore_index).float()


class MultiClassDiceLoss(nn.Module):
    """Class-weighted, Laplace-smoothed (1.0) multi-class Dice loss; per-sample Dice averaged over the batch."""

    def __init__(self, num_classes: int, weights: List[float], ignore_index: int):
        super().__init__()
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.register_buffer("weight", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = _valid(target, self.ignore_index)
        prob = torch.softmax(logits.float(), dim=1)
        total, wsum = logits.new_zeros(()).float(), 0.0
        for c in range(self.num_classes):
            p = prob[:, c] * valid
            g = ((target == c).float()) * valid
            inter = (p * g).sum(dim=(1, 2))
            loss_c = 1.0 - (2.0 * inter + 1.0) / (p.sum(dim=(1, 2)) + g.sum(dim=(1, 2)) + 1.0)
            total = total + self.weight[c] * loss_c.mean()
            wsum += float(self.weight[c])
        return total / (wsum + 1e-4)


def _spatial_gradient(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1), value=0.0)
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1), value=0.0)
    return torch.sqrt(dx ** 2 + dy ** 2 + 1e-4)


def phase_mask_contrast(fg: torch.Tensor, phase: torch.Tensor, valid: torch.Tensor, margin: float) -> torch.Tensor:
    """Eq. 3: hinge on (mean background phase - mean cell phase + margin). Returns (B,)."""
    fg, ph = fg.squeeze(1), phase.squeeze(1)
    a_cell = (fg * valid).sum(dim=(1, 2)).clamp(min=1.0)
    a_bg = ((1 - fg) * valid).sum(dim=(1, 2)).clamp(min=1.0)
    mu_cell = (fg * ph * valid).sum(dim=(1, 2)) / a_cell
    mu_bg = ((1 - fg) * ph * valid).sum(dim=(1, 2)) / a_bg
    return F.relu(mu_bg - mu_cell + margin)


def boundary_gradient_alignment(fg: torch.Tensor, phase: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Eq. 4: L1 between per-sample max-normalised gradient magnitudes of the mask and of the phase. Returns (B,)."""
    v = valid.unsqueeze(1)
    v_g = -F.max_pool2d(-v, kernel_size=3, stride=1, padding=1)          # erosion: gradient needs valid neighbours
    gm, gp = _spatial_gradient(fg) * v_g, _spatial_gradient(phase) * v_g
    gm = gm / (gm.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
    gp = gp / (gp.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
    return ((gm - gp).abs() * v_g).sum(dim=(1, 2, 3)) / v_g.sum(dim=(1, 2, 3)).clamp(min=1.0)


def integrated_phase_preservation(fg: torch.Tensor, target_fg: torch.Tensor, phase: torch.Tensor,
                                  valid: torch.Tensor, eps: float) -> torch.Tensor:
    """Eq. 5: |S_pred - S_gt| / (|S_gt| + eps) with S = sum(mask * raw phase). Returns (B,)."""
    fg, ph = fg.squeeze(1), phase.squeeze(1)
    s_pred = (fg * ph * valid).sum(dim=(1, 2))
    s_gt = (target_fg * ph * valid).sum(dim=(1, 2))
    return (s_pred - s_gt).abs() / (s_gt.abs() + eps)


class PhysicsAwarePhaseLoss(nn.Module):
    def __init__(self, num_classes: int, class_weights: List[float], ignore_index: int,
                 lambda_pmc: float, lambda_bga: float, lambda_pv: float, pmc_margin: float, pv_epsilon: float):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambdas = (lambda_pmc, lambda_bga, lambda_pv)
        self.pmc_margin, self.pv_epsilon = pmc_margin, pv_epsilon
        self.dice = MultiClassDiceLoss(num_classes, class_weights, ignore_index)

    def components(self, logits: torch.Tensor, target: torch.Tensor, phase: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits, phase = logits.float(), phase.float()
        valid = _valid(target, self.ignore_index)
        fg = 1.0 - torch.softmax(logits, dim=1)[:, 0:1]
        target_fg = ((target > 0) & (target != self.ignore_index)).float()
        has_cells = (target_fg.sum(dim=(1, 2)) > 0).float()
        denom = has_cells.sum().clamp(min=1.0)

        def reduce(per_sample: torch.Tensor) -> torch.Tensor:
            return (per_sample * has_cells).sum() / denom

        l_pmc = reduce(phase_mask_contrast(fg, phase, valid, self.pmc_margin))
        l_bga = reduce(boundary_gradient_alignment(fg, phase, valid))
        l_pv = reduce(integrated_phase_preservation(fg, target_fg, phase, valid, self.pv_epsilon))
        return {"L_dice": self.dice(logits, target), "L_pmc": l_pmc, "L_bga": l_bga, "L_pv": l_pv}

    def forward(self, logits: torch.Tensor, target: torch.Tensor, phase: torch.Tensor,
                return_components: bool = False):
        c = self.components(logits, target, phase)
        total = c["L_dice"] + self.lambdas[0] * c["L_pmc"] + self.lambdas[1] * c["L_bga"] + self.lambdas[2] * c["L_pv"]
        if return_components:
            return total, {k: v.detach() for k, v in c.items()}
        return total


class DiceOnlyLoss(nn.Module):
    """Ablation baseline: the class-weighted Dice term alone (physics terms are still *logged*)."""

    def __init__(self, num_classes: int, class_weights: List[float], ignore_index: int):
        super().__init__()
        self.ignore_index = ignore_index
        self.dice = MultiClassDiceLoss(num_classes, class_weights, ignore_index)

    def forward(self, logits, target, phase=None, return_components: bool = False):
        loss = self.dice(logits, target)
        if return_components:
            z = loss.new_zeros(())
            return loss, {"L_dice": loss.detach(), "L_pmc": z, "L_bga": z, "L_pv": z}
        return loss


def build_loss(run_cfg: Dict) -> nn.Module:
    l = run_cfg["loss"]
    n, ign = run_cfg["num_classes"], run_cfg["ignore_index"]
    if l["type"] == "physics_aware":
        return PhysicsAwarePhaseLoss(n, l["class_weights"], ign, l["lambda_pmc"], l["lambda_bga"], l["lambda_pv"],
                                     l["pmc_margin"], l["pv_epsilon"])
    if l["type"] == "dice_only":
        return DiceOnlyLoss(n, l["class_weights"], ign)
    raise ValueError(f"unknown loss type '{l['type']}'")
