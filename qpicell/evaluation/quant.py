"""Quantitative cell-level analyses on top of evaluation/cells.py (ground truth used here, for scoring only).

touching cell   a reference cell with a contact line to another reference cell; isolated = single-cell component
cluster         connected component of the annotated foreground (4-connectivity); multi-cell cluster = >= 2 cells
phase accounting (per image; network terms do not depend on the instance step):
  retained = phase of annotated pixels predicted as foreground / annotated phase (all, touching, isolated)
  lost by the network = annotated phase predicted as background; extra = predicted phase outside the annotation
  unassigned by the instance step = predicted-foreground pixels without an instance (0 by construction, counted)
combined vs individual: E_network (instance-free, reference territories), E_combined (sum over the matched instances of
  a fully matched cluster), individual e_c; partition residual = e_c - E_combined; concealed cluster =
  |E_combined| <= 2 % but some |e_c| > 5 %
over-splitting: per cluster, number of instances with >= 50 % of their area inside it vs number of reference cells
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from qpicell.data.cells import contact_lines
from qpicell.evaluation.agreement import agreement


def touching_and_clusters(cells: np.ndarray, valid: np.ndarray):
    c = np.where(valid, cells, 0).astype(np.int64)
    B = contact_lines(c)
    ids = set(np.unique(c[B & (c > 0)]).tolist())
    K, nK = ndi.label(c > 0)
    n = int(c.max())
    clus = ndi.maximum(K, c, np.arange(n + 1)).astype(np.int64) if n else np.zeros(1, np.int64)
    return {i: (i in ids) for i in range(1, n + 1)}, {i: int(clus[i]) for i in range(1, n + 1)}, K, nK


def phase_accounting(phase, pred, mask, cells, valid, touch, lab_valid) -> Dict:
    gt = (mask > 0) & valid
    p = (pred > 0) & valid
    T = np.isin(cells, [i for i, t in touch.items() if t]) & gt
    I = gt & ~T
    s = lambda m: float(phase[m].sum())  # noqa: E731
    un = p & (lab_valid == 0)
    return {"S_annotated": s(gt), "S_predicted_fg": s(p), "S_true_positive": s(gt & p),
            "S_annotated_touching": s(T), "S_retained_touching": s(T & p),
            "S_annotated_isolated": s(I), "S_retained_isolated": s(I & p),
            "px_annotated": int(gt.sum()), "px_predicted_fg": int(p.sum()), "px_fg_missed_by_network": int((gt & ~p).sum()),
            "px_unassigned_by_instances": int(un.sum()), "S_unassigned_by_instances": s(un)}


def oversplit_rows(lab_valid, K, nK, cells_valid, stem, lo_px) -> List[Dict]:
    rows = []
    if nK == 0:
        return rows
    n_cells = ndi.labeled_comprehension(cells_valid, K, np.arange(1, nK + 1), lambda v: len(np.unique(v[v > 0])), int, 0)
    nl = int(lab_valid.max())
    if nl:
        area = np.bincount(lab_valid.ravel(), minlength=nl + 1)
        m = (lab_valid > 0) & (K > 0)
        inter = np.bincount(lab_valid[m] * (nK + 1) + K[m], minlength=(nl + 1) * (nK + 1)).reshape(nl + 1, nK + 1)
        best = inter.argmax(1); frac = inter.max(1) / np.maximum(area, 1)
        own = (frac >= 0.5) & (best > 0); own[0] = False
    for k in range(1, nK + 1):
        if nl:
            js = np.nonzero(own & (best == k))[0]; n_inst = len(js); small = int((area[js] < lo_px).sum())
        else:
            n_inst = small = 0
        rows.append({"stem": stem, "cluster": k, "n_cells": int(n_cells[k - 1]), "n_instances": n_inst, "n_small_instances": small})
    return rows


def cluster_table(C: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stem, k), g in C.groupby(["stem", "cluster"]):
        if len(g) < 2:
            continue
        Sg = g.S_gt.sum()
        r = {"stem": stem, "cluster": k, "n_cells": len(g), "S_gt": Sg,
             "E_network": (g.S_assigned.sum() - Sg) / abs(Sg) if Sg else np.nan, "fully_matched": bool(g.matched.all())}
        if r["fully_matched"]:
            r["E_combined"] = (g.S_pred.sum() - Sg) / abs(Sg) if Sg else np.nan
            sc = g[g.scored]
            r["max_abs_individual"] = float(sc.e_matched.abs().max()) if len(sc) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def exclusion_reasons(row, lo: float, hi: float) -> List[str]:
    r = []
    if row["area_um2"] < lo:
        r.append("area_below_min")
    if row["area_um2"] > hi:
        r.append("area_above_max")
    if row["touches_unannotated"]:
        r.append("touches_unannotated_area")
    if row["wrapped"]:
        r.append("near_phase_wrapping")
    return r


def group_metrics(g: pd.DataFrame, within: float = 0.05) -> Dict:
    r = {"n": len(g)}
    if not len(g):
        return r
    mm = g[g.matched]
    r.update({"match_pct": 100 * g.matched.mean(), "missed_pct": 100 * g.missed.mean(), "merged_pct": 100 * g.merged.mean(),
              "fragmented_pct": 100 * g.fragmented.mean(),
              "cell_ipe_signed_median_pct": 100 * mm.e_matched.median() if len(mm) else None,
              "cell_ipe_abs_median_pct": 100 * mm.e_matched.abs().median() if len(mm) else None,
              "cell_ipe_signed_mean_pct": 100 * mm.e_matched.mean() if len(mm) else None,
              "cell_ipe_abs_p90_pct": 100 * mm.e_matched.abs().quantile(0.9) if len(mm) else None,
              f"within_{within * 100:g}pct_of_all_cells_pct": 100 * (g.matched & (g.e_matched.abs() <= within)).mean(),
              "network_cell_ipe_signed_median_pct": 100 * g.e_assigned.median(),
              "network_cell_ipe_abs_median_pct": 100 * g.e_assigned.abs().median()})
    if len(mm) > 2:
        dm = agreement(mm.dm_pred_pg, mm.dm_gt_pg); ov = agreement(mm.ov_pred_um3, mm.ov_gt_um3)
        r["dry_mass_matched"] = {**dm, "sum_diff_pct": 100 * (mm.dm_pred_pg.sum() / mm.dm_gt_pg.sum() - 1)}
        r["optical_volume_matched"] = {**ov, "sum_diff_pct": 100 * (mm.ov_pred_um3.sum() / mm.ov_gt_um3.sum() - 1)}
    return r


def cluster_summary(C: pd.DataFrame) -> Dict:
    K = cluster_table(C)
    out = {"multi_cell_clusters": len(K)}
    if not len(K):
        return out
    out.update(network_E_signed_median_pct=100 * K.E_network.median(), network_E_abs_median_pct=100 * K.E_network.abs().median())
    FM = K[K.fully_matched]
    out.update(fully_matched=len(FM), fully_matched_pct=100 * len(FM) / len(K))
    if len(FM):
        cm = C.merge(FM[["stem", "cluster", "E_combined"]], on=["stem", "cluster"]); cm = cm[cm.scored & cm.matched]
        res = cm.e_matched - cm.E_combined
        out.update(combined_E_signed_median_pct=100 * FM.E_combined.median(), combined_E_abs_median_pct=100 * FM.E_combined.abs().median(),
                   cells=len(cm), individual_abs_median_pct=100 * cm.e_matched.abs().median(),
                   individual_signed_median_pct=100 * cm.e_matched.median(),
                   partition_residual_abs_median_pct=100 * res.abs().median(), partition_residual_abs_p90_pct=100 * res.abs().quantile(0.9),
                   concealed_clusters=int(((FM.E_combined.abs() <= 0.02) & (FM.max_abs_individual > 0.05)).sum()),
                   combined_within_2pct_clusters=int((FM.E_combined.abs() <= 0.02).sum()))
    return out


def oversplit_summary(O: pd.DataFrame) -> Dict:
    multi, single = O[O.n_cells >= 2], O[O.n_cells == 1]
    f = lambda m: float(m.mean() * 100) if len(m) else None  # noqa: E731
    return {"multi_clusters_more_instances_than_cells_pct": f(multi.n_instances > multi.n_cells),
            "multi_clusters_exact_count_pct": f(multi.n_instances == multi.n_cells),
            "multi_clusters_fewer_instances_pct": f(multi.n_instances < multi.n_cells),
            "single_cell_components_split_pct": f(single.n_instances >= 2),
            "small_instances_inside_annotation": int(O.n_small_instances.sum()) if len(O) else 0}
