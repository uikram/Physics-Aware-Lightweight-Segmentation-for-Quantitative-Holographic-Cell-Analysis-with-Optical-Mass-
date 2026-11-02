"""Aggregate vs per-cell integrated-phase error, class-specific IPE and phase/boundary-stratified errors.

Per image i (binary foreground, annotated region only):
    S_P = sum(P * phi), S_G = sum(G * phi);   IPE_abs = |S_P - S_G| / |S_G|
    global IPE = mean_i IPE_abs on RAW phase (what the loss and Table 3/5 use);
    the same quantity on plane-corrected phase is reported as the background-corrected variant.
Per cell: GT cells = connected components with area in the configured range (um^2) that do not touch an
unannotated region; each is matched one-to-one (IoU >= match_iou) to a predicted component; the error is
(S_pred - S_gt)/|S_gt| on plane-corrected phase. Unmatched GT cells are counted as missed.
Strata: pixel error rate in distance bands to the GT boundary; false-negative rate inside GT cells by
per-image quartile of corrected phase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from qpi.analysis import wrapping as wrp
from qpi.analysis.common import (cluster_bootstrap_mean, components, load_predictions, match_cells,
                                 pixel_area_um2, plane_correct, touches_unannotated)

BANDS = ((0, 2), (2, 4), (4, 8))


def analyse_run(cfg: Dict, run_id: str, run_dir: Path, tag: str, val: Dict[str, Dict], out_dir: Path) -> Dict:
    ev, names = cfg["evaluation"], cfg["data"]["class_names"]
    area_lo, area_hi = ev["cell_area_range_um2"]
    px_area = pixel_area_um2(cfg)
    preds = load_predictions(run_dir, tag)
    per_image, per_cell, by_class = [], [], []
    band = {b: [0, 0] for b in BANDS}
    quart = {q: [0, 0] for q in range(4)}
    n_gt = n_match = n_unmatched_pred = 0
    for stem, d in val.items():
        phase, gt, valid = d["phase"], d["gt"], d["valid"]
        pc = np.where(valid, preds[stem], 0)
        p, g = pc > 0, gt > 0
        corr = plane_correct(phase, g, valid, ev["background_exclusion_px"])
        row = {"run_id": run_id, "stem": stem, "day": d["day"]}
        for tagp, ph in (("raw", phase), ("bgcorr", corr)):
            sp, sg = float((p * ph).sum()), float((g * ph).sum())
            row[f"S_gt_{tagp}"], row[f"S_pred_{tagp}"] = sg, sp
            row[f"ipe_signed_{tagp}"] = (sp - sg) / abs(sg) if sg else np.nan
            row[f"ipe_abs_{tagp}"] = abs(sp - sg) / abs(sg) if sg else np.nan
        # same quantity on the phase exactly as provided (processed phase + the offset removed by prepare_dataset)
        sp_ap, sg_ap = float((p * (phase + d["offset"])).sum()), float((g * (phase + d["offset"])).sum())
        row["ipe_abs_asprovided"] = abs(sp_ap - sg_ap) / abs(sg_ap) if sg_ap else np.nan
        per_image.append(row)
        for c in range(1, cfg["data"]["num_classes"]):
            gc, pcm = gt == c, pc == c
            if gc.any():
                sg, sp = float((gc * corr).sum()), float((pcm * corr).sum())
                by_class.append({"run_id": run_id, "stem": stem, "class": names[c], "present_in_pred": bool(pcm.any()),
                                 "ipe_signed": (sp - sg) / abs(sg), "ipe_abs": abs(sp - sg) / abs(sg)})
        gl, ng = components(gt, valid)
        pl, npd = components(pc, valid)
        wr = wrp.rule(cfg)
        gt_wrapped = wrp.flag_components(gl, ng, wrp.wrapped_pixels(phase + d["offset"], wr["ceiling"], wr["tolerance"],
                                                                      wr["jump"]), wr["dilation_px"])
        per_image[-1]["n_wrapped_gt_components"] = int(gt_wrapped.sum())
        sizes = np.bincount(gl.ravel(), minlength=ng + 1)
        cut = touches_unannotated(gl, ng, valid)
        keep = [i for i in range(1, ng + 1)
                if area_lo <= sizes[i] * px_area <= area_hi and not cut[i]]
        pairs = {gi: (pi, iou) for gi, pi, iou in match_cells(gl, ng, pl, npd, ev["match_iou"])}
        sg_c = ndi.sum(corr, gl, np.arange(ng + 1))
        sp_c = ndi.sum(corr, pl, np.arange(npd + 1))
        n_gt += len(keep)
        n_unmatched_pred += npd - len(pairs)
        for gi in keep:
            lbl = int(np.bincount(gt[gl == gi], minlength=cfg["data"]["num_classes"])[1:].argmax() + 1)
            rec = {"run_id": run_id, "stem": stem, "day": d["day"], "gt_cell": gi, "gt_class": names[lbl],
                   "area_um2_gt": sizes[gi] * px_area, "matched": gi in pairs, "wrapped": bool(gt_wrapped[gi])}
            if gi in pairs:
                pi, iou = pairs[gi]
                n_match += 1
                rec.update(iou=iou, S_gt=float(sg_c[gi]), S_pred=float(sp_c[pi]),
                           err_signed=float((sp_c[pi] - sg_c[gi]) / abs(sg_c[gi])),
                           err_abs=float(abs(sp_c[pi] - sg_c[gi]) / abs(sg_c[gi])))
            per_cell.append(rec)
        # strata (annotated region only)
        err = (p != g) & valid
        dist = np.where(g, ndi.distance_transform_edt(g), ndi.distance_transform_edt(~g))
        for b, (lo, hi) in zip(band, BANDS):
            sel = (dist > lo) & (dist <= hi) & valid
            band[b][0] += int(err[sel].sum())
            band[b][1] += int(sel.sum())
        if g.any():
            qi = np.digitize(corr, np.quantile(corr[g], [0.25, 0.5, 0.75]))
            for q in range(4):
                sel = g & (qi == q)
                quart[q][0] += int((~p)[sel].sum())
                quart[q][1] += int(sel.sum())
    pi_df, pc_df = pd.DataFrame(per_image), pd.DataFrame(per_cell)
    mt = pc_df[pc_df.matched] if not pc_df.empty else pc_df
    mt_nw = mt[~mt.wrapped] if len(mt) else mt
    nowrap_img = pi_df[pi_df.n_wrapped_gt_components == 0]
    summary = {
        "run_id": run_id, "checkpoint": tag, "n_images": len(pi_df),
        "ipe_global_raw_mean_pct": 100 * pi_df.ipe_abs_raw.mean(),
        "ipe_global_asprovided_mean_pct": 100 * pi_df.ipe_abs_asprovided.mean(),
        "ipe_global_bgcorr_mean_pct": 100 * pi_df.ipe_abs_bgcorr.mean(),
        "ipe_global_bgcorr_median_pct": 100 * pi_df.ipe_abs_bgcorr.median(),
        "ipe_global_bgcorr_signed_mean_pct": 100 * pi_df.ipe_signed_bgcorr.mean(),
        "ipe_pooled_bgcorr_pct": 100 * abs((pi_df.S_pred_bgcorr - pi_df.S_gt_bgcorr).sum()) / pi_df.S_gt_bgcorr.abs().sum(),
        "gt_cells": n_gt, "matched_cells": n_match,
        "matched_cells_pct": 100 * n_match / n_gt if n_gt else None,
        "unmatched_predicted_components": n_unmatched_pred,
        # wrapping (D-21): images without any wrapped annotated component, and per-cell errors without wrapped cells
        "n_images_without_wrapped_components": int(len(nowrap_img)),
        "ipe_global_raw_mean_pct_images_without_wrapped": 100 * nowrap_img.ipe_abs_raw.mean() if len(nowrap_img) else None,
        "gt_cells_wrapped": int(pc_df.wrapped.sum()) if len(pc_df) else 0,
    }
    if len(mt):
        summary.update(percell_abs_median_pct=100 * mt.err_abs.median(), percell_abs_p90_pct=100 * mt.err_abs.quantile(0.9),
                       percell_signed_median_pct=100 * mt.err_signed.median(),
                       percell_frac_over_10pct=100 * (mt.err_abs > 0.10).mean())
    if len(mt_nw):
        summary.update(percell_abs_median_pct_nowrap=100 * mt_nw.err_abs.median(),
                       percell_abs_p90_pct_nowrap=100 * mt_nw.err_abs.quantile(0.9),
                       percell_signed_median_pct_nowrap=100 * mt_nw.err_signed.median(),
                       percell_frac_over_10pct_nowrap=100 * (mt_nw.err_abs > 0.10).mean(),
                       matched_cells_nowrap=int(len(mt_nw)))
    strata = [{"run_id": run_id, "stratum": f"boundary band {lo}-{hi} px", "error_rate_pct": 100 * band[(lo, hi)][0] / max(band[(lo, hi)][1], 1)}
              for lo, hi in BANDS]
    strata += [{"run_id": run_id, "stratum": f"false negatives in phase quartile Q{q + 1}",
                "error_rate_pct": 100 * quart[q][0] / max(quart[q][1], 1)} for q in range(4)]
    out_dir.mkdir(parents=True, exist_ok=True)
    pi_df.to_csv(out_dir / f"{run_id}__per_image.csv", index=False)
    pc_df.to_csv(out_dir / f"{run_id}__per_cell.csv", index=False)
    bc = pd.DataFrame(by_class)
    class_summary = {}
    if not bc.empty:
        for cn, g in bc.groupby("class"):
            m_c = mt[mt.gt_class == cn] if len(mt) else mt
            class_summary[cn] = {"images": int(len(g)), "absent_in_prediction": int((~g.present_in_pred).sum()),
                                 "ipe_abs_mean_pct": 100 * g.ipe_abs.mean(), "ipe_abs_median_pct": 100 * g.ipe_abs.median(),
                                 "percell_abs_median_pct": 100 * m_c.err_abs.median() if len(m_c) else None}
    return {"summary": summary, "strata": strata, "by_class": class_summary}


def paired_comparison(cfg: Dict, per_image: Dict[str, pd.DataFrame], pairs: List[tuple]) -> List[Dict]:
    """Per-image differences (B - A) of IPE (percentage points) with a day-cluster bootstrap CI."""
    ev, out = cfg["evaluation"], []
    for a, b in pairs:
        if a not in per_image or b not in per_image:
            continue
        da, db = per_image[a].set_index("stem"), per_image[b].set_index("stem")
        stems = sorted(set(da.index) & set(db.index))
        days = da.loc[stems, "day"].to_numpy()
        for metric in ("ipe_abs_raw", "ipe_abs_bgcorr"):
            diff = 100 * (db.loc[stems, metric] - da.loc[stems, metric]).to_numpy()
            m, lo, hi = cluster_bootstrap_mean(diff, days, ev["bootstrap_resamples"], ev["bootstrap_seed"])
            out.append({"A": a, "B": b, "metric": metric, "mean_diff_pct_points": m, "ci95_lo": lo, "ci95_hi": hi,
                        "n_images": len(stems)})
    return out
