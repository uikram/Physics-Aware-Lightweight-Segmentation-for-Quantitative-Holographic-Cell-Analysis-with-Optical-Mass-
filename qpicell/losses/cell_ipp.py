r"""Cell-level integrated-phase preservation (cIPP) - the new loss of the revision.

Per image: p = predicted foreground probability, phi = referenced phase (rad), v = 1 inside the annotated field of
view; G_c = pixels of annotated cell c (watershed instance); T_c = its territory (pixels whose nearest annotated cell
is c, at most r = 9 native px away; G_c is inside T_c); T_0 = pixels in no territory; B = contact lines between
touching cells (cell pixels with a 4-neighbour in another cell).

  S_c^gt   = sum_{G_c} phi v                                   annotated integrated phase of cell c
  S_c^pred = sum_{T_c} p phi v                                  predicted integrated phase in c's neighbourhood
  d_c      = max(|S_c^gt|, kappa * median_c' |S_c'^gt|)         floored denominator (kappa = 0.25)
  e_c      = (S_c^pred - S_c^gt) / d_c                          relative integrated-phase error of cell c
  rho(e)   = sqrt(e^2 + delta^2) - delta                        smooth absolute value (delta = 0.05)
  O_c      = 1 - (2 sum_{G_c} p v + 1) / (sum_{T_c} p v + |G_c| + 1)   per-cell soft Dice inside the territory
  F        = sum_{T_0} p |phi| v / sum_c |S_c^gt|               phase predicted far from every annotated cell
  P_sep    = sum_B p v / sum_B v                                foreground predicted on contact lines (merging)

  L_cIPP = mean_{cells of the batch} rho(e_c) + beta mean_cells O_c + gamma mean_images F + eta P_sep
           (beta = gamma = eta = 1)

Difference to the global IPP (terms.global_ipp): (1) no cancellation - a missed cell cannot be compensated by an extra
region elsewhere, because every cell is compared with its own neighbourhood and extra regions are penalised
separately (F); (2) equal weight per cell - in the global term large cells dominate; (3) per-cell shape (O_c).

P_sep is NOT one of these differences in this pipeline. Every arm that uses this loss sets eta = 0, which
losses.objective.Objective asserts, so the separation term is inactive in every run of the experiment matrix and
contributes nothing to any reported result. Touching cells are separated at inference by the marker-controlled
watershed (evaluation.instances.a2), not by the loss. P_sep is kept here only so the term can be enabled in
future work; do not describe the loss as enforcing separation.

Resolution: the cell maps (G, T, B) are given at C x C (256); when the loss maps are larger (MobileSAM: 1024), every
pixel product (p phi v, p v, phi v, v, p |phi| v) is computed at full resolution and averaged over the f x f blocks,
so every per-cell sum is exactly the full-resolution sum divided by f^2 (all terms are ratios).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def per_cell_sums(values: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """(B, H, W) values and integer labels 0..K -> (B, K + 1) sums per label (differentiable in values)."""
    B = values.shape[0]
    ids = (labels + (torch.arange(B, device=labels.device) * (K + 1)).view(B, 1, 1)).reshape(-1)
    return values.new_zeros(B * (K + 1)).index_add_(0, ids, values.reshape(-1)).view(B, K + 1)


class CellIPP(nn.Module):
    def __init__(self, beta: float = 1.0, gamma: float = 1.0, eta: float = 1.0, kappa: float = 0.25, delta: float = 0.05):
        super().__init__()
        self.beta, self.gamma, self.eta, self.kappa, self.delta = beta, gamma, eta, kappa, delta

    def forward(self, p: torch.Tensor, phi: torch.Tensor, valid: torch.Tensor, cells: torch.Tensor,
                terr: torch.Tensor, contacts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        maps = {"pq": p * phi * valid, "p": p * valid, "q": phi * valid, "v": valid, "pabsq": p * phi.abs() * valid}
        f = p.shape[-1] // cells.shape[-1]
        if f > 1:
            if p.shape[-1] != f * cells.shape[-1] or p.shape[-2] != f * cells.shape[-2]:
                raise ValueError(f"loss maps {tuple(p.shape[-2:])} are not a multiple of the cell maps {tuple(cells.shape[-2:])}")
            maps = {k: F.avg_pool2d(x[:, None], f)[:, 0] for k, x in maps.items()}
        K = int(max(int(cells.max()), int(terr.max()), 1))
        s_gt = per_cell_sums(maps["q"], cells, K)[:, 1:]
        n_px = per_cell_sums(maps["v"], cells, K)[:, 1:]
        present = n_px > 0
        z = p.new_zeros(())
        if not bool(present.any()):
            return {"cipp_phase": z, "cipp_overlap": z, "cipp_far": z, "cipp_separation": z, "cipp": z}
        s_pred = per_cell_sums(maps["pq"], terr, K)[:, 1:]
        inter = per_cell_sums(maps["p"], cells, K)[:, 1:]
        p_terr = per_cell_sums(maps["p"], terr, K)[:, 1:]
        a = s_gt.abs()
        med = torch.nan_to_num(torch.nanmedian(torch.where(present, a, torch.full_like(a, float("nan"))), dim=1).values, nan=1.0)
        e = (s_pred - s_gt) / torch.maximum(a, self.kappa * med.view(-1, 1)).clamp(min=1e-6)
        rho = torch.sqrt(e * e + self.delta ** 2) - self.delta
        overlap = 1.0 - (2.0 * inter + 1.0) / (p_terr + n_px + 1.0)
        l_phase, l_overlap = rho[present].mean(), overlap[present].mean()
        has = present.any(dim=1)
        far = (maps["pabsq"] * (terr == 0).float()).sum(dim=(1, 2))
        l_far = (far[has] / a.sum(dim=1)[has].clamp(min=1e-6)).mean()
        l_sep = z
        if self.eta and contacts is not None:
            den = (contacts * maps["v"]).sum()
            l_sep = torch.where(den > 0, (maps["p"] * contacts).sum() / den.clamp(min=1e-6), z)
        total = l_phase + self.beta * l_overlap + self.gamma * l_far + self.eta * l_sep
        return {"cipp_phase": l_phase, "cipp_overlap": l_overlap, "cipp_far": l_far, "cipp_separation": l_sep, "cipp": total}
