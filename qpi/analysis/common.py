"""Shared helpers for the post-training analyses (all work at the native 768 x 768 grid, raw phase in radians)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
from scipy import ndimage as ndi

from qpi.data.dataset import read_manifest


def load_validation(cfg: Dict) -> Dict[str, Dict]:
    """Native-resolution validation data keyed by stem: phase (float32), gt (uint8, ignore=255), valid, day."""
    root = Path(cfg["paths_abs"]["processed_root"])
    rows = read_manifest(root, "val")
    n = cfg["data"]["max_val_samples"]
    if n is not None:
        rows = rows[: int(n)]
    ign = cfg["data"]["ignore_index"]
    out: Dict[str, Dict] = {}
    for r in rows:
        ph = tifffile.imread(root / "val" / "phase" / f"{r['stem']}.tif").astype(np.float64)
        gt = tifffile.imread(root / "val" / "mask" / f"{r['stem']}.tif")
        out[r["stem"]] = {"phase": ph, "gt": np.where(gt == ign, 0, gt).astype(np.int64), "valid": gt != ign,
                          "day": int(r["storage_day"]), "offset": float(r["phase_offset_rad"])}
    return out


def load_predictions(run_dir: Path, tag: str) -> Dict[str, np.ndarray]:
    with np.load(Path(run_dir) / f"predictions_{tag}.npz") as z:
        return {k: z[k].astype(np.int64) for k in z.files}


def pixel_area_um2(cfg: Dict) -> float:
    return float(cfg["physics"]["pixel_size_um"]) ** 2


def dry_mass_factor_pg(cfg: Dict) -> float:
    """pg per (rad * px^2):  dx^2 * lambda / (2 pi alpha)   (manuscript Eq. 10; alpha in mL/g)."""
    p = cfg["physics"]
    return p["pixel_size_um"] ** 2 * p["wavelength_um"] / (2 * math.pi * p["refractive_increment_ml_per_g"])


def plane_correct(phase: np.ndarray, gt_fg: np.ndarray, valid: np.ndarray, exclusion_px: int) -> np.ndarray:
    """Subtract a least-squares plane fitted to background pixels (valid, > exclusion_px from any GT cell)."""
    bg = valid & ~ndi.binary_dilation(gt_fg, iterations=exclusion_px)
    if bg.sum() < 100:
        raise ValueError("too few background pixels for the plane fit")
    yy, xx = np.mgrid[0: phase.shape[0], 0: phase.shape[1]]
    A = np.stack([np.ones(int(bg.sum())), xx[bg], yy[bg]], 1)
    c, *_ = np.linalg.lstsq(A, phase[bg], rcond=None)
    return phase - (c[0] + c[1] * xx + c[2] * yy)


def background_level(phase: np.ndarray, fg: np.ndarray, valid: np.ndarray, exclusion_px: int) -> float:
    """Median phase of the non-cellular region (valid pixels farther than exclusion_px from any foreground)."""
    bg = valid & ~ndi.binary_dilation(fg, iterations=exclusion_px)
    if bg.sum() < 100:
        return float("nan")      # caller must skip the image (no measurable background)
    return float(np.median(phase[bg]))


def match_cells(gl: np.ndarray, ng: int, pl: np.ndarray, npd: int, min_iou: float) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching of labelled GT and predicted components by IoU >= min_iou."""
    if ng == 0 or npd == 0:
        return []
    m = (gl > 0) & (pl > 0)
    inter = np.bincount(gl[m] * (npd + 1) + pl[m], minlength=(ng + 1) * (npd + 1)).reshape(ng + 1, npd + 1)
    ga, pa = np.bincount(gl.ravel(), minlength=ng + 1), np.bincount(pl.ravel(), minlength=npd + 1)
    iou = inter / (ga[:, None] + pa[None, :] - inter + 1e-9)
    iou[0, :] = 0
    iou[:, 0] = 0
    pairs, ug, up = [], set(), set()
    for gi, pi in sorted(zip(*np.nonzero(iou >= min_iou)), key=lambda t: -iou[t]):
        if gi in ug or pi in up:
            continue
        pairs.append((int(gi), int(pi), float(iou[gi, pi])))
        ug.add(gi)
        up.add(pi)
    return pairs


def components(class_map: np.ndarray, valid: np.ndarray):
    """4-connected components of the binary foreground inside the annotated region."""
    return ndi.label((class_map > 0) & valid)


def touches_unannotated(lab: np.ndarray, n: int, valid: np.ndarray) -> np.ndarray:
    """Boolean array (index = label) marking components in contact with an unannotated region.

    Cells cut by the artificial boundary of the annotated field of view are excluded from per-cell statistics;
    contact with the *image border* is allowed (as in the original analysis).
    """
    flag = np.zeros(n + 1, bool)
    if valid.all() or n == 0:
        return flag
    near = ndi.binary_dilation(~valid, iterations=1)
    flag[np.unique(lab[near & (lab > 0)])] = True
    return flag


def cluster_bootstrap_mean(diff: np.ndarray, groups: np.ndarray, n: int, seed: int) -> Tuple[float, float, float]:
    """Mean and 95% percentile CI of per-image differences, resampling storage days (clusters) with replacement."""
    rng = np.random.default_rng(seed)
    days = np.unique(groups)
    by = {d: diff[groups == d] for d in days}
    stats = [np.mean(np.concatenate([by[d] for d in rng.choice(days, len(days), replace=True)])) for _ in range(n)]
    return float(np.mean(diff)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
