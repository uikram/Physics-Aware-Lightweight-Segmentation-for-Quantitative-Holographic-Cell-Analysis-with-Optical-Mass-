"""Storage-time analysis of cell area and dry mass on the 66 test images (11 storage days x 6 images).

Unit of analysis = the image (per-image median over its cells): the cells of one image, and the images of one day, are
not independent. Sources:
  annotation   scored reference cells (area, dry mass from the referenced float32 phase)
  every run    usable predicted cells (A2 instances of the saved predictions; usable = 15-200 um^2, not cut by the
               unannotated area, not near phase wrapping), same phase reference

Per source and metric (area um^2, dry mass pg):
  Spearman rho vs storage day (asymptotic p)
  exact date-block permutation p: all 11! = 39,916,800 assignments of the 11 storage days to the 11 blocks of images
     (the images of one day stay together), two-sided on |rho|; Monte Carlo (200,000) when the blocks are unequal
  date bootstrap: storage days resampled with replacement -> 95 % interval of rho
  leave-one-day-out: min / max of rho
  early (day <= 8) vs late (day >= 37): medians, % change, Mann-Whitney p
  prediction - annotation per image vs storage day (is a predicted trend a segmentation bias?): rho and permutation p
These are associations with storage time in one sample series; they are not evidence of a biological mechanism.
"""
from __future__ import annotations

import math
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tifffile
from scipy import stats

_PERM8 = None


def exact_block_permutation_p(days: np.ndarray, values: np.ndarray) -> float:
    """Exact two-sided p of |Spearman rho| under permutation of whole day blocks; NaN if blocks are unequal."""
    global _PERM8
    blocks = np.unique(days)
    sizes = np.array([(days == b).sum() for b in blocks])
    nb = len(blocks)
    if nb != 11 or not (sizes == sizes[0]).all():
        return float("nan")
    s, n = int(sizes[0]), len(values)
    vr = stats.rankdata(values)
    V = np.array([vr[days == b].sum() for b in blocks], float)        # rank sum of each block
    pos = np.arange(nb, dtype=float)
    mean = (n + 1) / 2.0
    day_rank = s * pos + (s + 1) / 2.0                                  # tied rank of the images at position p
    sd_d = math.sqrt(np.sum(s * (day_rank - mean) ** 2) / n); sd_v = vr.std()
    rho = lambda T: (s * T + (s + 1) / 2.0 * V.sum() - n * mean * mean) / (n * sd_d * sd_v)   # T = sum_p p V[block at p]
    r_obs = rho(float(pos @ V))
    if _PERM8 is None:
        _PERM8 = np.array(list(permutations(range(8))), dtype=np.int8)
    tail_w = np.arange(3, 11, dtype=float)
    hits = total = 0; thr = abs(r_obs) - 1e-12
    for a in range(nb):                                                 # positions 0, 1, 2 fixed, 8! for the rest
        for b in range(nb):
            if b == a:
                continue
            for c in range(nb):
                if c in (a, b):
                    continue
                rest = np.array([x for x in range(nb) if x not in (a, b, c)])
                T = 1.0 * V[b] + 2.0 * V[c] + V[rest][_PERM8] @ tail_w
                hits += int(np.sum(np.abs(rho(T)) >= thr)); total += len(T)
    return hits / total


def _row_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rx, ry = stats.rankdata(x, axis=1), stats.rankdata(y, axis=1)
    rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


def monte_carlo_block_permutation_p(days: np.ndarray, values: np.ndarray, n: int = 200000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    blocks = np.unique(days); idx = np.searchsorted(blocks, days)
    r_obs = abs(stats.spearmanr(days, values).statistic)
    hits = tot = 0
    for start in range(0, n, 20000):
        m = min(20000, n - start)
        perm = np.argsort(rng.random((m, len(blocks))), axis=1)
        x = blocks[perm][:, idx].astype(float)
        hits += int(np.sum(np.abs(_row_spearman(x, np.broadcast_to(values, x.shape).astype(float))) >= r_obs - 1e-12)); tot += m
    return (hits + 1) / (tot + 1)


def date_bootstrap(days: np.ndarray, values: np.ndarray, n: int, seed: int = 0):
    """Resample whole storage days with replacement; resamples with < 3 distinct days are skipped."""
    rng = np.random.default_rng(seed)
    u = np.unique(days); groups = [np.nonzero(days == d)[0] for d in u]
    out = []
    if len({len(g) for g in groups}) == 1:                       # equal blocks: vectorised
        D = np.stack([days[g] for g in groups]).astype(float); V = np.stack([values[g] for g in groups]).astype(float)
        for start in range(0, n, 4000):
            m = min(4000, n - start)
            pick = rng.integers(0, len(u), (m, len(u)))
            keep = np.array([len(np.unique(p)) >= 3 for p in pick])
            out.append(_row_spearman(D[pick].reshape(m, -1), V[pick].reshape(m, -1))[keep])
        out = np.concatenate(out)
    else:
        for _ in range(n):
            pick = rng.integers(0, len(u), len(u))
            if len(np.unique(pick)) < 3:
                continue
            ix = np.concatenate([groups[i] for i in pick])
            out.append(stats.spearmanr(days[ix], values[ix]).statistic)
        out = np.asarray(out)
    out = out[np.isfinite(out)]
    if out.size == 0:                                            # e.g. constant values: rho undefined
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def trend(img: pd.DataFrame, col: str, cfg: Dict, exact: bool = True) -> Dict:
    d = img[["storage_day", col]].dropna()
    days, v = d.storage_day.to_numpy(), d[col].to_numpy(float)
    if len(np.unique(days)) < 3 or np.ptp(v) == 0:
        return {"n_images": int(len(v)), "n_days": int(len(np.unique(days))), "rho": float("nan"), "p_asymptotic": float("nan"),
                "p_block_permutation": float("nan"), "permutation_method": "not computed (fewer than 3 days or constant values)",
                "rho_ci95_date_bootstrap": [float("nan"), float("nan")], "rho_leave_one_day_out": [float("nan"), float("nan")],
                "early_median": None, "late_median": None, "early_to_late_change_pct": None, "mann_whitney_p": float("nan")}
    r, p = stats.spearmanr(days, v)
    pe = exact_block_permutation_p(days, v) if exact else float("nan")
    method = "exact_11!"
    if not np.isfinite(pe):
        pe, method = monte_carlo_block_permutation_p(days, v), "monte_carlo_200k"
    lo, hi = date_bootstrap(days, v, int(cfg["storage"]["date_bootstrap"]))
    lodo = [stats.spearmanr(days[days != x], v[days != x]).statistic for x in np.unique(days)]
    e, l = v[days <= cfg["storage"]["early_max_day"]], v[days >= cfg["storage"]["late_min_day"]]
    mw = stats.mannwhitneyu(e, l, alternative="two-sided").pvalue if len(e) and len(l) else float("nan")
    return {"n_images": int(len(v)), "n_days": int(len(np.unique(days))), "rho": float(r), "p_asymptotic": float(p),
            "p_block_permutation": float(pe), "permutation_method": method, "rho_ci95_date_bootstrap": [lo, hi],
            "rho_leave_one_day_out": [float(np.min(lodo)), float(np.max(lodo))],
            "early_median": float(np.median(e)) if len(e) else None, "late_median": float(np.median(l)) if len(l) else None,
            "early_to_late_change_pct": float(100 * (np.median(l) - np.median(e)) / np.median(e)) if len(e) and len(l) else None,
            "mann_whitney_p": float(mw)}


def per_image(cells: pd.DataFrame, area: str, dm: str) -> pd.DataFrame:
    return cells.groupby("stem").agg(storage_day=("storage_day", "first"), area=(area, "median"), dry_mass=(dm, "median"),
                                     n=(dm, "size")).reset_index()


@lru_cache(maxsize=1)
def _ground_truth(prepared: str, stems: tuple, ign: int):
    from qpicell.physics import assert_phase
    out = {}
    for s in stems:
        r = lambda sub: tifffile.imread(Path(prepared) / "val" / sub / f"{s}.tif")  # noqa: E731
        ph = r("phase"); assert_phase(ph, s)
        out[s] = (ph.astype(np.float64), r("mask"), r("cells").astype(np.int64), r("wrapped").astype(bool))
    return out


def predicted_cells(cfg: Dict, predictions: Path, day_of: Dict[str, int], instances: str = "A2") -> pd.DataFrame:
    """Usable predicted cells of one run: instances rebuilt from the saved class maps (no ground truth for the
    instances), scored with the production evaluate_image (ground truth only for the usable flags)."""
    from qpicell.evaluation.cells import evaluate_image
    from qpicell.evaluation.evaluate import prepare_cfg
    from qpicell.evaluation.instances import a2, cc
    c = prepare_cfg(cfg); z = np.load(predictions); stems = tuple(sorted(z.keys()))
    gt = _ground_truth(str(cfg["paths_abs"]["prepared"]), stems, int(cfg["data"]["ignore_index"]))
    rows = []
    for s in stems:
        ph, mask, cells, wr = gt[s]; pred = z[s]
        lab = a2(pred, int(cfg["cells"]["watershed_h"])) if instances == "A2" else cc(pred)
        _, pr, _ = evaluate_image(c, pred, ph, mask, cells, wr, s, day_of[s], labels=lab)
        rows += [r for r in pr if r["usable"]]
    return pd.DataFrame(rows)


def run_trends(cfg: Dict, P: pd.DataFrame, ann: pd.DataFrame, exact: bool = True) -> Dict:
    pi = per_image(P, "area_um2", "dm_pg")
    out = {"n_cells": int(len(P)), "n_images": int(pi.stem.nunique())}
    for m in ("area", "dry_mass"):
        out[m] = trend(pi, m, cfg, exact)
        j = pi.merge(ann, on=["stem", "storage_day"], suffixes=("_p", "_a")); j["bias"] = j[f"{m}_p"] - j[f"{m}_a"]
        bt = trend(j, "bias", cfg, exact)
        out[m]["bias_vs_annotation"] = {"rho": bt["rho"], "p_block_permutation": bt["p_block_permutation"],
                                        "median_bias": float(j.bias.median())}
    out["per_day_median"] = {m: pi.groupby("storage_day")[m].median().round(6).to_dict() for m in ("area", "dry_mass")}
    return out


def annotation_cells(cfg: Dict, stems, day_of: Dict[str, int]) -> pd.DataFrame:
    """Scored reference cells of the test images (area, dry mass, class) from the prepared data alone."""
    from qpicell.evaluation.cells import evaluate_image
    from qpicell.evaluation.evaluate import prepare_cfg
    c = prepare_cfg(cfg); gt = _ground_truth(str(cfg["paths_abs"]["prepared"]), tuple(sorted(stems)), int(cfg["data"]["ignore_index"]))
    rows = []
    for s in sorted(stems):
        ph, mask, cells, wr = gt[s]
        cr, _, _ = evaluate_image(c, mask, ph, mask, cells, wr, s, day_of[s], labels=np.where(mask != int(cfg["data"]["ignore_index"]), cells, 0))
        rows += [r for r in cr if r["scored"]]
    return pd.DataFrame(rows)
