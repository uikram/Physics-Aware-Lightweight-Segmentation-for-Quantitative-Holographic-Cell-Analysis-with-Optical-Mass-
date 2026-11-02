"""Per-cell morphology and phase-derived dry mass; agreement between predicted and annotation masks; storage statistics.

Cells flagged as phase-wrapped (column ``wrapped``, qpi/analysis/wrapping.py, D-21) are kept in the table; run_analysis
reports every statistic for all cells and for the not-wrapped cells.

Dry mass (manuscript Eq. 10): m = lambda / (2 pi alpha) * sum_cell(phi - phi_bg) * dx^2, with the imaging constants of
the configuration and phi in radians. The background level phi_bg is the median phase of the non-cellular region of
each image (valid pixels farther than ``background_exclusion_px`` from the foreground of the mask under analysis).
No calibration to a reference MCH is applied: the MCH value is reported only as a plausibility check.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats
from skimage.measure import regionprops

from qpi.analysis import wrapping as wrp
from qpi.analysis.common import (background_level, components, dry_mass_factor_pg, pixel_area_um2,
                                 touches_unannotated)


def measure_cells(cfg: Dict, stem: str, day: int, class_map: np.ndarray, phase: np.ndarray, valid: np.ndarray,
                  offset: float = 0.0) -> List[Dict]:
    """``phase`` is the processed (background-referenced) phase; ``offset`` is the value prepare_dataset removed, so that
    phase + offset is the phase as provided, on which the wrapping rule (D-21) acts."""
    ev = cfg["evaluation"]
    lo, hi = ev["cell_area_range_um2"]
    px_area, k_pg = pixel_area_um2(cfg), dry_mass_factor_pg(cfg)
    lab, n = components(class_map, valid)
    if n == 0:
        return []
    bg = background_level(phase, lab > 0, valid, ev["background_exclusion_px"])
    if not np.isfinite(bg):
        return []
    idx = np.arange(1, n + 1)
    npix = ndi.sum(np.ones_like(phase), lab, idx)
    s_phi = ndi.sum(phase - bg, lab, idx)
    cut = touches_unannotated(lab, n, valid)
    wr = wrp.rule(cfg)
    wrapped = wrp.flag_components(lab, n, wrp.wrapped_pixels(phase + offset, wr["ceiling"], wr["tolerance"], wr["jump"]),
                                  wr["dilation_px"])
    perim = {r.label: r.perimeter_crofton for r in regionprops(lab)}
    rows = []
    for i, npx, s in zip(idx, npix, s_phi):
        area = npx * px_area
        if not (lo <= area <= hi) or cut[i]:
            continue
        cls_counts = np.bincount(class_map[lab == i], minlength=cfg["data"]["num_classes"])[1:]
        cls = int(cls_counts.argmax() + 1)
        per = perim[i]
        rows.append({"stem": stem, "storage_day": day, "cell": int(i), "class_idx": cls,
                     "class_name": cfg["data"]["class_names"][cls], "area_um2": float(area),
                     "perimeter_px": float(per), "circularity": float(4 * math.pi * npx / per ** 2) if per > 0 else np.nan,
                     "integrated_phase_rad_px": float(s), "dry_mass_pg": float(s * k_pg), "background_rad": bg,
                     "max_phase_rad": float(phase[lab == i].max()), "wrapped": bool(wrapped[i])})
    return rows


def cells_table(cfg: Dict, val: Dict[str, Dict], maps: Optional[Dict[str, np.ndarray]]) -> pd.DataFrame:
    """Cells measured on ``maps`` (predicted class maps), or on the annotation when ``maps`` is None."""
    rows: List[Dict] = []
    for stem, d in val.items():
        cm = d["gt"] if maps is None else np.where(d["valid"], maps[stem], 0)
        rows += measure_cells(cfg, stem, d["day"], cm, d["phase"], d["valid"], d["offset"])
    cols = ["stem", "storage_day", "cell", "class_idx", "class_name", "area_um2", "perimeter_px", "circularity",
            "integrated_phase_rad_px", "dry_mass_pg", "background_rad", "max_phase_rad", "wrapped"]
    return pd.DataFrame(rows, columns=cols)


def _ccc(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(2 * np.cov(a, b, ddof=0)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def agreement(pred: pd.DataFrame, gt: pd.DataFrame) -> Dict:
    """Agreement of per-image median dry mass (image = unit of analysis): Pearson r, Lin CCC, MdARE, Bland-Altman."""
    j = pd.concat([pred.groupby("stem").dry_mass_pg.median(), gt.groupby("stem").dry_mass_pg.median()],
                  axis=1, keys=["pred", "gt"]).dropna()
    if len(j) < 3:
        return {"n_images": int(len(j)), "note": "fewer than 3 paired images; agreement statistics not computed"}
    # j["gt"] not j.gt: DataFrame.gt is the pandas "greater than" method, not the column
    p_, g_ = j["pred"], j["gt"]
    d = p_ - g_
    rel = d / g_
    return {"n_images": int(len(j)), "pearson_r": float(stats.pearsonr(p_, g_)[0]), "lin_ccc": _ccc(p_, g_),
            "bland_altman_mean_pg": float(d.mean()),
            "bland_altman_loa_low_pg": float(d.mean() - 1.96 * d.std(ddof=1)),
            "bland_altman_loa_high_pg": float(d.mean() + 1.96 * d.std(ddof=1)),
            "median_paired_rel_diff_pct": float(100 * rel.median()), "mdare_pct": float(100 * rel.abs().median()),
            "mean_abs_rel_pct": float(100 * rel.abs().mean()), "max_abs_rel_pct": float(100 * rel.abs().max())}


def storage_statistics(cfg: Dict, cells: pd.DataFrame, area_max: Optional[float] = None) -> Dict:
    """Spearman vs storage day, early-vs-late Mann-Whitney and partial Spearman, on per-image medians."""
    a = cfg["analysis"]
    d = cells if area_max is None else cells[cells.area_um2 <= area_max]
    img = d.groupby("stem").agg(day=("storage_day", "first"), area=("area_um2", "median"),
                                dm=("dry_mass_pg", "median"), circ=("circularity", "median")).reset_index()
    out: Dict = {"n_images": int(len(img)), "area_filter_um2": area_max}
    if len(img) < 4:
        out["note"] = "fewer than 4 images; storage statistics not computed"
        return out
    for key, col in (("area", "area"), ("dry_mass", "dm"), ("circularity", "circ")):
        r = stats.spearmanr(img.day, img[col])
        e, l = img[img.day <= a["storage_early_max_day"]][col], img[img.day >= a["storage_late_min_day"]][col]
        entry = {"spearman_rho": float(r.statistic), "spearman_p": float(r.pvalue)}
        if len(e) and len(l):
            entry.update(early_median=float(e.median()), late_median=float(l.median()),
                         late_vs_early_pct=float(100 * (l.median() / e.median() - 1)),
                         mannwhitney_p=float(stats.mannwhitneyu(e, l, alternative="two-sided").pvalue),
                         n_early=int(len(e)), n_late=int(len(l)))
        out[key] = entry
    ranks = img[["day", "dm", "area"]].rank()

    def resid(y, x):
        return y - np.polyval(np.polyfit(x, y, 1), x)

    pr = stats.pearsonr(resid(ranks.dm, ranks.area), resid(ranks.day, ranks.area))
    out["dry_mass_given_area_partial_spearman"] = {"rho": float(pr.statistic), "p": float(pr.pvalue)}
    return out


def per_day_table(cfg: Dict, cells: pd.DataFrame) -> List[Dict]:
    names = cfg["data"]["class_names"]
    rows = []
    for day, g in cells.groupby("storage_day"):
        comp = {names[c]: float((g.class_idx == c).mean() * 100) for c in range(1, cfg["data"]["num_classes"])}
        rows.append({"storage_day": int(day), "n_cells": int(len(g)), "n_images": int(g.stem.nunique()),
                     "area_um2_mean": float(g.area_um2.mean()), "area_um2_sd": float(g.area_um2.std(ddof=1)) if len(g) > 1 else None,
                     "circularity_mean": float(g.circularity.mean()),
                     "circularity_sd": float(g.circularity.std(ddof=1)) if len(g) > 1 else None,
                     "dry_mass_pg_mean": float(g.dry_mass_pg.mean()),
                     "dry_mass_pg_sd": float(g.dry_mass_pg.std(ddof=1)) if len(g) > 1 else None,
                     "composition_pct": comp})
    return rows


def consistency_checks(cfg: Dict, cells: pd.DataFrame) -> Dict:
    """Plausibility checks of pixel pitch and phase scale (not calibrations)."""
    ref = cfg["physics"]["mch_reference_pg"]
    out = {"mch_reference_pg": ref, "n_cells": int(len(cells)),
           "fraction_cells_max_phase_above_2pi": float((cells.max_phase_rad > 2 * math.pi).mean()) if len(cells) else None}
    d0 = cells[cells.storage_day == 0]
    if len(d0):
        out.update(day0_n_cells=int(len(d0)), day0_median_dry_mass_pg=float(d0.dry_mass_pg.median()),
                   day0_median_over_mch=float(d0.dry_mass_pg.median() / ref),
                   day0_median_area_um2=float(d0.area_um2.median()))
    return out
