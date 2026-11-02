"""Segmentation and phase-preservation metrics (manuscript Sec. 3.3).

All functions work on NumPy arrays and ignore pixels labelled ``ignore_index`` (no annotation).

* Dice / IoU: pooled (micro) over all images per class, then macro-averaged over the 4 foreground classes;
  a class that is never predicted contributes 0 (never excluded).
* AJI: aggregated Jaccard index on connected components of the binary foreground, pooled over images.
* Boundary F1: pooled, 2-px tolerance; boundary pixels within tolerance+1 of an unannotated region are dropped.
* IPE (Eq. 11): per image |S_pred - S_gt| / |S_gt| on RAW phase (radians), global = mean over images,
  class-specific = mean over the images containing the class.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi


def aji_components(pred_lab: np.ndarray, gt_lab: np.ndarray):
    """Intersection and union sums of the Aggregated Jaccard Index for one image (vectorised)."""
    pred_ids, gt_ids = np.unique(pred_lab[pred_lab > 0]), np.unique(gt_lab[gt_lab > 0])
    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0, 1.0
    if len(gt_ids) == 0:
        return 0.0, float((pred_lab > 0).sum())
    if len(pred_ids) == 0:
        return 0.0, float((gt_lab > 0).sum())
    max_p, max_t = int(pred_lab.max()) + 1, int(gt_lab.max()) + 1
    p_areas, t_areas = np.bincount(pred_lab.ravel(), minlength=max_p), np.bincount(gt_lab.ravel(), minlength=max_t)
    both = (pred_lab > 0) & (gt_lab > 0)
    inter_tab = np.bincount(gt_lab[both] * max_p + pred_lab[both], minlength=max_t * max_p)
    tot_i = tot_u = 0.0
    used = set()
    for t in gt_ids:
        best_iou, best_p, best_i, best_u = 0.0, -1, 0.0, 0.0
        for p in pred_ids:
            inter = float(inter_tab[t * max_p + p])
            if inter > 0:
                union = float(t_areas[t] + p_areas[p]) - inter
                iou = inter / union
                if iou > best_iou:
                    best_iou, best_p, best_i, best_u = iou, p, inter, union
        if best_p >= 0:
            tot_i += best_i
            tot_u += best_u
            used.add(best_p)
        else:
            tot_u += float(t_areas[t])
    for p in pred_ids:
        if p not in used:
            tot_u += float(p_areas[p])
    return tot_i, tot_u


def boundary_stats(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, tol: int):
    def edge(m):
        return m.astype(bool) & ~ndi.binary_erosion(m)

    pb, gb = edge(pred), edge(gt)
    if (~valid).any():
        near_invalid = ndi.binary_dilation(~valid, iterations=tol + 1)
        pb, gb = pb & ~near_invalid, gb & ~near_invalid
    if not pb.any() and not gb.any():
        return 1.0, 0.0, 0.0
    pd_, gd = ndi.binary_dilation(pb, iterations=tol), ndi.binary_dilation(gb, iterations=tol)
    return float((pb & gd).sum()), float((pb & ~gd).sum()), float((gb & ~pd_).sum())


def integrated_phase_error(pred_mask: np.ndarray, gt_mask: np.ndarray, phase: np.ndarray) -> Dict[str, float]:
    sp, sg = float((pred_mask * phase).sum()), float((gt_mask * phase).sum())
    if abs(sg) < 1e-8:
        return {"S_pred": sp, "S_gt": sg, "ipe": 0.0 if abs(sp) < 1e-8 else 1.0}
    return {"S_pred": sp, "S_gt": sg, "ipe": abs(sp - sg) / abs(sg)}


class SegmentationMetrics:
    def __init__(self, num_classes: int = 5, class_names: Optional[List[str]] = None, boundary_tolerance: int = 2,
                 ignore_index: int = 255):
        self.nc, self.tol, self.ignore = num_classes, boundary_tolerance, ignore_index
        self.class_names = class_names or [f"class{c}" for c in range(num_classes)]
        self.reset()

    def reset(self):
        self.tp, self.fp, self.fn = (np.zeros(self.nc) for _ in range(3))
        self.aji_i = self.aji_u = 0.0
        self.b_tp = self.b_fp = self.b_fn = 0.0
        self.ipe_global: List[float] = []
        self.ipe_class: Dict[int, List[float]] = {c: [] for c in range(1, self.nc)}
        self.per_image: List[Dict] = []

    def update(self, pred: np.ndarray, target: np.ndarray, phase: np.ndarray, name: str = "", day: int = -1):
        valid = target != self.ignore
        p, t = np.where(valid, pred, 0).astype(np.int64), np.where(valid, target, 0).astype(np.int64)
        for c in range(1, self.nc):
            pc, tc = p == c, t == c
            self.tp[c] += (pc & tc).sum()
            self.fp[c] += (pc & ~tc & valid).sum()
            self.fn[c] += (~pc & tc & valid).sum()
        pb, tb = p > 0, t > 0
        i, u = aji_components(ndi.label(pb)[0], ndi.label(tb)[0])
        self.aji_i += i
        self.aji_u += u
        a, b, c_ = boundary_stats(pb, tb, valid, self.tol)
        self.b_tp += a
        self.b_fp += b
        self.b_fn += c_
        ph = np.where(valid, phase, 0.0)
        g = integrated_phase_error(pb, tb, ph)
        self.ipe_global.append(g["ipe"])
        row = {"stem": name, "storage_day": day, "S_gt": g["S_gt"], "S_pred": g["S_pred"], "ipe_global": g["ipe"]}
        for c in range(1, self.nc):
            tc = t == c
            if tc.any():
                e = integrated_phase_error(p == c, tc, ph)["ipe"]
                self.ipe_class[c].append(e)
                row[f"ipe_{self.class_names[c]}"] = e
            else:
                row[f"ipe_{self.class_names[c]}"] = float("nan")
        self.per_image.append(row)

    def compute(self) -> Dict[str, float]:
        eps = 1e-8
        # a class that is absent from both ground truth and prediction has an undefined score (NaN, excluded
        # from the macro mean) instead of a spurious 1.0
        den = 2 * self.tp + self.fp + self.fn
        dice = np.where(den > 0, 2 * self.tp / np.maximum(den, eps), np.nan)
        den_i = self.tp + self.fp + self.fn
        iou = np.where(den_i > 0, self.tp / np.maximum(den_i, eps), np.nan)
        prec = (self.b_tp + eps) / (self.b_tp + self.b_fp + eps)
        rec = (self.b_tp + eps) / (self.b_tp + self.b_fn + eps)
        out = {
            "mean_dice": float(np.nanmean(dice[1:])), "mean_iou": float(np.nanmean(iou[1:])),
            "aji": float((self.aji_i + eps) / (self.aji_u + eps)),
            "bf1": float(2 * prec * rec / (prec + rec + eps)),
            "ipe_global": float(np.mean(self.ipe_global)) if self.ipe_global else float("nan"),
            "ipe_global_std": float(np.std(self.ipe_global)) if self.ipe_global else float("nan"),
            "n_images": len(self.ipe_global),
        }
        for c in range(1, self.nc):
            n = self.class_names[c]
            out[f"dice_{n}"] = float(dice[c])
            out[f"iou_{n}"] = float(iou[c])
            out[f"ipe_{n}"] = float(np.mean(self.ipe_class[c])) if self.ipe_class[c] else float("nan")
            out[f"n_images_with_{n}"] = len(self.ipe_class[c])
        return out
