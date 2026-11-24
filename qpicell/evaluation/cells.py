"""Cell-level evaluation on the native 768 grid (referenced phase in radians).

Annotated cell = watershed instance inside the annotated field of view. It is SCORED when its area is within
cells.area_range_um2, it does not touch the unannotated area and it is not near phase wrapping; all instances take
part in the matching. Predicted cell = an instance of the supplied instance map (A2 watershed, primary; CC = 4-connected components,
diagnostic), clipped to the field of view.

Per scored annotated cell c:
  matched     a predicted component with IoU >= 0.5 (greedy one-to-one)  -> e_matched = (S_pred - S_gt) / |S_gt|
  assigned    S over the predicted foreground inside the territory of c (9 px) -> e_assigned (no matching needed;
              a missed cell gives about -100 %)
  missed      the predicted foreground covers < 50 % of c
  merged      c is one of >= 2 annotated cells that each have >= 50 % of their area inside ONE predicted component
  fragmented  >= 2 predicted components each cover >= 10 % of c
Per predicted component: false positive when < 50 % of its area lies on annotated foreground; a component is
used for the all-cell dry mass when its area is in range, it does not touch the unannotated area and is not near
wrapping.
Shape: projected area A = N dx^2 and circularity C = 4 pi A / P^2 with P the Crofton perimeter (skimage, 4
directions), as in the submitted manuscript. C = 1 is a perfect disc; it falls as the outline becomes irregular.
Dry mass m = k S (k = lambda / (2 pi alpha) dx^2). Matched-cell dry mass: one point per matched scored cell.
All-cell dry mass: per image, median over all usable predicted components vs median over all scored annotated cells
(independent of matching; this is the population-level comparison of the submitted manuscript).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from qpicell.data.cells import territories, touches_unannotated
from qpicell.data.wrapping import flagged_labels
from qpicell.evaluation.agreement import agreement
from qpicell.evaluation.shape import circularity_per_label


def match(gl: np.ndarray, ng: int, pl: np.ndarray, npd: int, thr: float) -> Dict[int, Tuple[int, float]]:
    if ng == 0 or npd == 0:
        return {}
    m = (gl > 0) & (pl > 0)
    inter = np.bincount(gl[m] * (npd + 1) + pl[m], minlength=(ng + 1) * (npd + 1)).reshape(ng + 1, npd + 1)
    ga, pa = np.bincount(gl.ravel(), minlength=ng + 1), np.bincount(pl.ravel(), minlength=npd + 1)
    iou = inter / (ga[:, None] + pa[None, :] - inter + 1e-9); iou[0] = 0; iou[:, 0] = 0
    out, used = {}, set()
    for gi, pi in sorted(zip(*np.nonzero(iou >= thr)), key=lambda t: -iou[t]):
        if gi in out or pi in used:
            continue
        out[int(gi)] = (int(pi), float(iou[gi, pi])); used.add(pi)
    return out


def evaluate_image(cfg: Dict, pred: np.ndarray, phase: np.ndarray, mask: np.ndarray, cells: np.ndarray,
                   wrapped: np.ndarray, stem: str = "", day: int = -1, labels: Optional[np.ndarray] = None):
    """-> (rows per annotated cell, rows per predicted instance, image row).
    labels: predicted instance map (A2 or CC, built WITHOUT ground truth); None = 4-connected components (CC)."""
    ign = int(cfg["data"]["ignore_index"]); valid = mask != ign
    k_pg = cfg["_k_pg"]; px_area = cfg["_px_area"]; k_ov = cfg["_k_ov"]
    lo, hi = cfg["cells"]["area_range_um2"]; dil = int(cfg["cells"]["wrapping"]["dilation_px"])
    names = cfg["data"]["class_names"]
    p = (pred > 0) & valid
    gl = np.where(valid, cells, 0).astype(np.int64); ng = int(gl.max())
    if labels is None:
        pl, npd = ndi.label(p)
    else:
        pl = np.where(valid, labels, 0).astype(np.int64)          # clipped to the annotated field of view for scoring
        u = np.unique(pl); u = u[u > 0]; lut = np.zeros(int(pl.max()) + 1, np.int64)
        lut[u] = np.arange(1, len(u) + 1); pl = lut[pl]                # 1..N, 0 = background (also without background pixels)
        npd = int(pl.max())
    ig, ip = np.arange(ng + 1), np.arange(npd + 1)
    Sg, Ag = ndi.sum(phase, gl, ig), np.bincount(gl.ravel(), minlength=ng + 1)
    Sp, Ap = ndi.sum(phase, pl, ip), np.bincount(pl.ravel(), minlength=npd + 1)
    terr = territories(gl, float(cfg["cells"]["territory_radius_native_px"]))
    Sa = ndi.sum(np.where(p, phase, 0.0), terr, ig)
    cover = ndi.sum(p.astype(float), gl, ig)
    Pg, Cg = circularity_per_label(gl, ng, cfg["physics"]["pixel_size_um"])
    Pp, Cp = circularity_per_label(pl, npd, cfg["physics"]["pixel_size_um"])
    cut_g, cut_p = touches_unannotated(gl, valid), touches_unannotated(pl, valid)
    wr_g, wr_p = flagged_labels(gl, wrapped, dil), flagged_labels(pl, wrapped, dil)
    both = (gl > 0) & (pl > 0)
    inter = np.bincount(gl[both] * (npd + 1) + pl[both], minlength=(ng + 1) * (npd + 1)).reshape(ng + 1, npd + 1)
    frac = inter / np.maximum(Ag[:, None], 1)
    pairs = match(gl, ng, pl, npd, float(cfg["evaluation"]["match_iou"]))
    merged = np.zeros(ng + 1, bool)
    for j in range(1, npd + 1):
        own = np.nonzero(frac[1:, j] >= 0.5)[0] + 1
        if len(own) >= 2:
            merged[own] = True
    frag = (frac[:, 1:] >= 0.1).sum(axis=1) >= 2
    gt_cls = ndi.labeled_comprehension(np.where(valid, mask, 0), gl, ig, lambda v: np.bincount(v, minlength=5)[1:].argmax() + 1, int, 0)
    pr_cls = ndi.labeled_comprehension(np.where(valid, pred, 0), pl, ip, lambda v: np.bincount(v, minlength=5)[1:].argmax() + 1, int, 0) if npd else np.zeros(1, int)
    crow = []
    for c in range(1, ng + 1):
        if Ag[c] == 0:
            continue
        area = Ag[c] * px_area
        scored = bool(lo <= area <= hi and not cut_g[c] and not wr_g[c])
        r = dict(stem=stem, storage_day=day, cell=c, gt_class=names[int(gt_cls[c])], area_um2=area,
                 perimeter_um=float(Pg[c]), circularity=float(Cg[c]), scored=scored,
                 touches_unannotated=bool(cut_g[c]), wrapped=bool(wr_g[c]), S_gt=float(Sg[c]), ov_gt_um3=k_ov * float(Sg[c]),
                 dm_gt_pg=k_pg * float(Sg[c]),
                 matched=c in pairs, missed=bool(cover[c] < 0.5 * Ag[c]), merged=bool(merged[c]), fragmented=bool(frag[c]),
                 S_assigned=float(Sa[c]), e_assigned=float((Sa[c] - Sg[c]) / abs(Sg[c])) if Sg[c] else np.nan)
        if c in pairs:
            j, iou = pairs[c]
            r.update(pred_component=j, iou=iou, S_pred=float(Sp[j]), ov_pred_um3=k_ov * float(Sp[j]), dm_pred_pg=k_pg * float(Sp[j]),
                     area_pred_um2=Ap[j] * px_area, perimeter_pred_um=float(Pp[j]), circularity_pred=float(Cp[j]),
                     pred_class=names[int(pr_cls[j])], pred_wrapped=bool(wr_p[j]),
                     e_matched=float((Sp[j] - Sg[c]) / abs(Sg[c])) if Sg[c] else np.nan)
        crow.append(r)
    on_fg = ndi.sum((gl > 0).astype(float), pl, ip)
    prow = []
    for j in range(1, npd + 1):
        area = Ap[j] * px_area
        prow.append(dict(stem=stem, storage_day=day, component=j, pred_class=names[int(pr_cls[j])], area_um2=area,
                         perimeter_um=float(Pp[j]), circularity=float(Cp[j]),
                         S=float(Sp[j]), ov_um3=k_ov * float(Sp[j]), dm_pg=k_pg * float(Sp[j]), false_positive=bool(on_fg[j] < 0.5 * Ap[j]),
                         usable=bool(lo <= area <= hi and not cut_p[j] and not wr_p[j]), matched=any(v[0] == j for v in pairs.values())))
    g = (gl > 0)
    sgt, spr = float(phase[g].sum()), float(phase[p].sum())
    irow = dict(stem=stem, storage_day=day, n_cells=len(crow), n_components=int(npd), S_gt=sgt, S_pred=spr,
                ipe_global=abs(spr - sgt) / abs(sgt) if sgt else np.nan)
    return crow, prow, irow


def summarise(cells: List[Dict], comps: List[Dict], images: List[Dict]) -> Dict:
    C = pd.DataFrame(cells); P = pd.DataFrame(comps); I = pd.DataFrame(images)
    S = C[C.scored]
    M = S[S.matched & ~S.get("pred_wrapped", pd.Series(False, index=S.index)).fillna(False).astype(bool)]
    out = {"n_cells_scored": int(len(S)), "n_cells_all": int(len(C)),
           "match_rate_pct": 100 * float(S.matched.mean()), "detection_rate_pct": 100 * float((~S.missed).mean()),
           "missed_pct": 100 * float(S.missed.mean()), "merged_pct": 100 * float(S.merged.mean()),
           "fragmented_pct": 100 * float(S.fragmented.mean()),
           "n_false_positive_components": int(P.false_positive.sum()) if len(P) else 0,
           "n_predicted_components": int(len(P)),
           "cell_ipe_matched_median_pct": 100 * float(M.e_matched.abs().median()) if len(M) else np.nan,
           "cell_ipe_matched_mean_pct": 100 * float(M.e_matched.abs().mean()) if len(M) else np.nan,
           "cell_ipe_matched_p90_pct": 100 * float(M.e_matched.abs().quantile(.9)) if len(M) else np.nan,
           "cell_ipe_matched_signed_median_pct": 100 * float(M.e_matched.median()) if len(M) else np.nan,
           "cell_ipe_assigned_median_pct": 100 * float(S.e_assigned.abs().median()),
           "cell_ipe_assigned_mean_pct": 100 * float(S.e_assigned.abs().mean()),
           "cell_ipe_assigned_over_10pct": 100 * float((S.e_assigned.abs() > 0.1).mean()),
           "ipe_global_mean_pct": 100 * float(I.ipe_global.mean())}
    out["drymass_matched"] = agreement(M.dm_pred_pg, M.dm_gt_pg) if len(M) else {"n": 0}
    U = P[P.usable] if len(P) else P
    if len(U) and len(S):
        med = pd.concat([U.groupby("stem").dm_pg.median(), S.groupby("stem").dm_gt_pg.median()], axis=1, keys=["p", "g"]).dropna()
        out["drymass_all_cells_image_median"] = agreement(med.p, med.g)
        out["drymass_all_cells_population"] = {
            "n_pred_components": int(len(U)), "n_annotated_cells": int(len(S)),
            "median_pred_pg": float(U.dm_pg.median()), "median_annotated_pg": float(S.dm_gt_pg.median()),
            "mean_pred_pg": float(U.dm_pg.mean()), "mean_annotated_pg": float(S.dm_gt_pg.mean()),
            "relative_difference_of_medians_pct": 100 * float((U.dm_pg.median() - S.dm_gt_pg.median()) / S.dm_gt_pg.median())}
    return out
