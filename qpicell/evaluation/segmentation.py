"""Segmentation scores (definitions of the submitted manuscript), all on the native grid unless stated.

Dice / IoU per class: pooled over images (micro), macro mean over the 4 cell classes; a class absent from prediction
  and annotation has an undefined score (NaN, left out of the mean). A class counts as LOST when its Dice < 0.05.
AJI: aggregated Jaccard index on connected components of the binary foreground, pooled over images.
Boundary F1: pooled, 2-px tolerance; boundary pixels within tolerance + 1 of the unannotated area are dropped.
Global IPE: per image |S_pred - S_gt| / |S_gt|, S = sum of the referenced phase over the predicted / annotated
  foreground; mean over images. Class IPE: the same per class, over the images containing the class.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import ndimage as ndi


def aji_sums(pred_lab: np.ndarray, gt_lab: np.ndarray):
    p_ids, g_ids = np.unique(pred_lab[pred_lab > 0]), np.unique(gt_lab[gt_lab > 0])
    if len(g_ids) == 0 and len(p_ids) == 0:
        return 1.0, 1.0
    if len(g_ids) == 0:
        return 0.0, float((pred_lab > 0).sum())
    if len(p_ids) == 0:
        return 0.0, float((gt_lab > 0).sum())
    mp, mg = int(pred_lab.max()) + 1, int(gt_lab.max()) + 1
    pa, ga = np.bincount(pred_lab.ravel(), minlength=mp), np.bincount(gt_lab.ravel(), minlength=mg)
    both = (pred_lab > 0) & (gt_lab > 0)
    inter = np.bincount(gt_lab[both] * mp + pred_lab[both], minlength=mg * mp).reshape(mg, mp)
    tot_i = tot_u = 0.0; used = set()
    for g in g_ids:
        row = inter[g]; cand = np.nonzero(row)[0]
        if len(cand) == 0:
            tot_u += float(ga[g]); continue
        union = ga[g] + pa[cand] - row[cand]; iou = row[cand] / union
        j = int(np.argmax(iou)); p = int(cand[j])
        tot_i += float(row[p]); tot_u += float(union[j]); used.add(p)
    for p in p_ids:
        if p not in used:
            tot_u += float(pa[p])
    return tot_i, tot_u


def boundary_counts(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, tol: int):
    edge = lambda m: m & ~ndi.binary_erosion(m)
    pb, gb = edge(pred), edge(gt)
    if (~valid).any():
        near = ndi.binary_dilation(~valid, iterations=tol + 1); pb, gb = pb & ~near, gb & ~near
    if not pb.any() and not gb.any():
        return 1.0, 0.0, 0.0
    pd_, gd = ndi.binary_dilation(pb, iterations=tol), ndi.binary_dilation(gb, iterations=tol)
    return float((pb & gd).sum()), float((pb & ~gd).sum()), float((gb & ~pd_).sum())


def _ipe(sp: float, sg: float) -> float:
    if abs(sg) < 1e-8:
        return 0.0 if abs(sp) < 1e-8 else 1.0
    return abs(sp - sg) / abs(sg)


class SegmentationScores:
    def __init__(self, cfg: Dict):
        self.nc, self.ign = int(cfg["data"]["num_classes"]), int(cfg["data"]["ignore_index"])
        self.names, self.tol = list(cfg["data"]["class_names"]), int(cfg["evaluation"]["boundary_tolerance_px"])
        self.lost = float(cfg["evaluation"]["lost_class_dice"])
        self.tp, self.fp, self.fn = np.zeros(self.nc), np.zeros(self.nc), np.zeros(self.nc)
        self.ai = self.au = self.btp = self.bfp = self.bfn = 0.0
        self.rows: List[Dict] = []

    def add(self, pred: np.ndarray, target: np.ndarray, phase: np.ndarray, stem: str = "", day: int = -1) -> Dict:
        valid = target != self.ign
        p, t = np.where(valid, pred, 0).astype(np.int64), np.where(valid, target, 0).astype(np.int64)
        row = {"stem": stem, "storage_day": day}
        ph = np.where(valid, phase, 0.0)
        for c in range(1, self.nc):
            pc, tc = p == c, t == c
            tp, fp, fn = (pc & tc).sum(), (pc & ~tc & valid).sum(), (~pc & tc & valid).sum()
            self.tp[c] += tp; self.fp[c] += fp; self.fn[c] += fn
            row[f"dice_{self.names[c]}"] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan
            row[f"ipe_{self.names[c]}"] = _ipe(float(ph[pc].sum()), float(ph[tc].sum())) if tc.any() else np.nan
        pb, tb = p > 0, t > 0
        i, u = aji_sums(ndi.label(pb)[0], ndi.label(tb)[0]); self.ai += i; self.au += u
        a, b, c_ = boundary_counts(pb, tb, valid, self.tol); self.btp += a; self.bfp += b; self.bfn += c_
        sp, sg = float(ph[pb].sum()), float(ph[tb].sum())
        row.update(S_pred=sp, S_gt=sg, ipe_global=_ipe(sp, sg), aji=i / u if u else np.nan)
        self.rows.append(row)
        return row

    def summary(self) -> Dict:
        eps = 1e-8
        den = 2 * self.tp + self.fp + self.fn; dice = np.where(den > 0, 2 * self.tp / np.maximum(den, eps), np.nan)
        den_i = self.tp + self.fp + self.fn; iou = np.where(den_i > 0, self.tp / np.maximum(den_i, eps), np.nan)
        prec = (self.btp + eps) / (self.btp + self.bfp + eps); rec = (self.btp + eps) / (self.btp + self.bfn + eps)
        ipe = np.array([r["ipe_global"] for r in self.rows], float)
        out = {"mean_dice": float(np.nanmean(dice[1:])) if np.isfinite(dice[1:]).any() else float("nan"),
               "mean_iou": float(np.nanmean(iou[1:])) if np.isfinite(iou[1:]).any() else float("nan"),
               "aji": float((self.ai + eps) / (self.au + eps)), "bf1": float(2 * prec * rec / (prec + rec + eps)),
               "ipe_global": float(ipe.mean()) if len(ipe) else float("nan"),
               "ipe_global_sd": float(ipe.std()) if len(ipe) else float("nan"), "n_images": len(self.rows)}
        lost = []
        for c in range(1, self.nc):
            n = self.names[c]
            out[f"dice_{n}"] = float(dice[c]); out[f"iou_{n}"] = float(iou[c])
            v = [r[f"ipe_{n}"] for r in self.rows if np.isfinite(r[f"ipe_{n}"])]
            out[f"ipe_{n}"] = float(np.mean(v)) if v else float("nan"); out[f"n_images_with_{n}"] = len(v)
            if np.isfinite(dice[c]) and dice[c] < self.lost:
                lost.append(f"{n} ({c})")
        out["lost_classes"] = lost; out["n_lost_classes"] = len(lost)
        return out
