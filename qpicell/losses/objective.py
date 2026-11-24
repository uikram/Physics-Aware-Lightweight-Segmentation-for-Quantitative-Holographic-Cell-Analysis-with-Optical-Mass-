"""The seven training objectives (configs/config.yaml, arms):

  dice               Dice
  dice_pmc           Dice + 0.1 PMC
  dice_pmc_bga       Dice + 0.1 PMC + 0.05 BGA
  dice_pmc_bga_ipp   Dice + 0.1 PMC + 0.05 BGA + 0.1 global IPP        (old Full, submitted manuscript)
  dice_cipp          Dice + 0.1 cIPP (eta = 0)
  dice_pmc_cipp      Dice + 0.1 PMC + 0.1 cIPP (eta = 0)
  dice_pmc_bga_cipp  Dice + 0.1 PMC + 0.05 BGA + 0.1 cIPP (eta = 0)  (New Full: cIPP replaces global IPP)
Global IPP and cIPP are never combined; cIPP never has its separation term (eta = 0, asserted). Every term is computed
in float32 on the referenced phase in radians (the network input is a separate, z-scored copy).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from qpicell.losses.cell_ipp import CellIPP
from qpicell.losses.terms import WeightedDice, boundary_gradient_alignment, global_ipp, phase_mask_contrast

COMPONENTS = ("dice", "pmc", "bga", "ipp", "cipp", "cipp_phase", "cipp_overlap", "cipp_far", "cipp_separation")


class Objective(nn.Module):
    def __init__(self, cfg: Dict, arm: str):
        super().__init__()
        if arm not in cfg["arms"]:
            raise ValueError(f"unknown arm {arm!r}; known: {list(cfg['arms'])}")
        a, L = cfg["arms"][arm], cfg["loss"]
        if a["ipp"] and a["cipp"]:
            raise ValueError("global IPP and cIPP are never combined")
        self.arm = arm
        self.lam = {"pmc": L["lambda_pmc"] if a["pmc"] else 0.0, "bga": L["lambda_bga"] if a["bga"] else 0.0,
                    "ipp": L["lambda_ipp"] if a["ipp"] else 0.0, "cipp": L["lambda_cipp"] if a["cipp"] else 0.0}
        self.ignore = int(cfg["data"]["ignore_index"])
        self.margin, self.eps = float(L["pmc_margin_rad"]), float(L["ipp_epsilon"])
        self.dice = WeightedDice(cfg["data"]["num_classes"], L["class_weights"], self.ignore)
        c = L["cipp"]
        if float(c["eta"]) != 0.0:
            raise ValueError(f"loss.cipp.eta = {c['eta']}: every cIPP arm of this pipeline uses eta = 0 (no separation term)")
        self.cipp = CellIPP(c["beta"], c["gamma"], c["eta"], c["kappa"], c["delta"])

    def forward(self, logits: torch.Tensor, target: torch.Tensor, phase: torch.Tensor, cells: Optional[torch.Tensor] = None,
                terr: Optional[torch.Tensor] = None, contacts: Optional[torch.Tensor] = None):
        """logits (B, K, S, S); target (B, S, S); phase (B, 1, S, S) radians; cell maps (B, C, C). -> (total, components)"""
        if phase.dtype != torch.float32:
            raise TypeError(f"physics losses need the float32 phase in radians, got {phase.dtype}")
        logits = logits.float(); phi = phase[:, 0]
        valid = (target != self.ignore).float()
        p = 1.0 - torch.softmax(logits, dim=1)[:, 0]
        g = ((target > 0) & (target != self.ignore)).float()
        has = (g.sum(dim=(1, 2)) > 0).float(); den = has.sum().clamp(min=1.0)
        mean_cells = lambda v: (v * has).sum() / den
        z = logits.new_zeros(())
        c = {k: z for k in COMPONENTS}
        c["dice"] = self.dice(logits, target)
        if self.lam["pmc"]:
            c["pmc"] = mean_cells(phase_mask_contrast(p, phi, valid, self.margin))
        if self.lam["bga"]:
            c["bga"] = mean_cells(boundary_gradient_alignment(p, phi, valid))
        if self.lam["ipp"]:
            c["ipp"] = mean_cells(global_ipp(p, g, phi, valid, self.eps))
        if self.lam["cipp"]:
            if cells is None or terr is None:
                raise ValueError("the cell-level loss needs the cell maps")
            c.update(self.cipp(p, phi, valid, cells, terr, contacts))
        total = c["dice"] + sum(self.lam[k] * c[k] for k in ("pmc", "bga", "ipp", "cipp"))
        return total, {k: v.detach() for k, v in c.items()}
