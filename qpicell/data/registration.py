"""Registration of the 8-bit annotation frame onto the 32-bit float phase map.

The masks were drawn on 8-bit exports of the phase maps. The exports are cropped / zoomed (isotropically or with
different x/y scales), shifted, and stretched with their own grey-level mapping, so they are not in the frame of the
float maps. The affine transform  x_float = A x_8bit + t  is estimated by intensity-based ECC registration
(invariant to a linear grey-level mapping) and the mask is carried onto the float grid with nearest-neighbour
sampling. Float pixels the export does not cover get the ignore label (no annotation exists there).

Only the 8-bit image is used for the geometry; the phase values themselves are never taken from the 8-bit export:
training, the losses and dry mass use the 32-bit float maps in radians.

Search: coarse multi-start ECC at 1/4 resolution over a grid of scales and shifts, refined at 1/2 and full
resolution; if the result is below the ECC threshold, a wide multi-start (+-240 px) is tried and the better result
kept. Quality gate: ECC >= min_ecc and Pearson >= min_pearson; an ECC in [min_ecc_borderline, min_ecc) is accepted
only with geometric evidence that the deficit is a monotone non-linear grey export, not misalignment (correlation
maximum within +-3 px at zero shift, Spearman >= min_spearman and Spearman - Pearson >= min_rank_gain).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

STANDARD_GRID = dict(sx=(0.6, 0.8, 0.9, 1.0, 1.2), sy=(0.6, 0.7, 0.9, 1.0, 1.2), dx=(-10, 0, 10), dy=(-10, 0, 10))
WIDE_GRID = dict(sx=(0.6, 0.8, 1.0, 1.2), sy=(0.6, 0.8, 1.0, 1.2), dx=tuple(range(-60, 61, 15)), dy=tuple(range(-60, 61, 15)))


@dataclass
class Registration:
    warp: np.ndarray                 # 2 x 3, x_float = warp[:, :2] @ x_8bit + warp[:, 2]
    ecc: float
    pearson: float                   # float-vs-8-bit correlation in the overlap
    spearman: float                  # rank correlation in the overlap
    peak_shift_px: float             # distance of the correlation maximum from zero shift (+-3 px search)
    scale_x: float
    scale_y: float
    rotation_deg: float
    shear: float                     # cosine of the angle between the transformed axes (0 = no shear)
    search: str                      # standard | wide
    accepted: bool = False
    basis: str = "rejected"          # primary | borderline_nonlinear_export | rejected

    def summary(self) -> Dict:
        d = asdict(self)
        d["warp"] = np.round(self.warp, 6).tolist()
        return d


def _normalise(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _ecc(template, image, init, iters, eps, gauss) -> Tuple[float, np.ndarray]:
    try:
        cc, w = cv2.findTransformECC(template, image, init.astype(np.float32), cv2.MOTION_AFFINE,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps), None, gauss)
        return float(cc), w
    except cv2.error:
        return -1.0, init


def _multistart(a: np.ndarray, b: np.ndarray, factor: int, grid: dict) -> Tuple[float, np.ndarray]:
    small = lambda x, f: cv2.resize(x, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
    a_s, b_s = small(a, factor), small(b, factor)
    best_cc, best_w = -1.0, None
    for sx in grid["sx"]:
        for sy in grid["sy"]:
            for dx in grid["dx"]:
                for dy in grid["dy"]:
                    cc, w = _ecc(a_s, b_s, np.array([[sx, 0, dx], [0, sy, dy]], np.float32), 100, 1e-4, 1)
                    if cc > best_cc:
                        best_cc, best_w = cc, w
    if best_w is None:
        raise RuntimeError("registration could not be initialised")
    w = best_w.copy(); w[:, 2] *= factor
    cc = best_cc
    for f in (2, 1):                                   # refinement at 1/2 and full resolution
        w_f = w.copy(); w_f[:, 2] /= f
        cc, w_f = _ecc(small(a, f), small(b, f), w_f, 300, 1e-6, 3)
        w = w_f.copy(); w[:, 2] *= f
    return cc, w


def sample_float_in_8bit_frame(phase: np.ndarray, warp: np.ndarray, shape) -> np.ndarray:
    """The float map resampled into the 8-bit frame (bilinear); NaN where the float map does not reach."""
    h, w = shape
    s = cv2.warpAffine(phase.astype(np.float32), warp.astype(np.float32), (w, h),
                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
    inside = cv2.warpAffine(np.ones_like(phase, np.float32), warp.astype(np.float32), (w, h),
                            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0.5
    return np.where(np.isfinite(s) & inside, s, np.nan)


def _geometry_evidence(phase: np.ndarray, grey8: np.ndarray, warp: np.ndarray, max_shift: int = 3):
    """(Spearman rho in the overlap, distance of the correlation maximum from zero shift over +-max_shift px)."""
    from scipy.stats import rankdata
    base = sample_float_in_8bit_frame(phase, warp, grey8.shape)
    ok0 = np.isfinite(base)
    ok0 = cv2.erode(ok0.astype(np.uint8), np.ones((2 * max_shift + 3, 2 * max_shift + 3), np.uint8)) > 0
    if ok0.sum() < 1000:
        return float("nan"), float("nan")
    g = grey8.astype(np.float64)
    rho = float(np.corrcoef(rankdata(base[ok0]), rankdata(g[ok0]))[0, 1])
    A = warp[:, :2]
    best, best_d = -2.0, (0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            ws = warp.copy()
            ws[:, 2] = warp[:, 2] - A @ np.array([dx, dy], np.float64)
            sm = sample_float_in_8bit_frame(phase, ws, grey8.shape)
            m = ok0 & np.isfinite(sm)
            if m.sum() < 1000:
                continue
            c = float(np.corrcoef(sm[m], g[m])[0, 1])
            if c > best:
                best, best_d = c, (dy, dx)
    return rho, float(np.hypot(*best_d))


def register(phase: np.ndarray, grey8: np.ndarray, gate: Dict, pyramid_factor: int = 4) -> Registration:
    """Estimate the 8-bit -> float affine transform and apply the quality gate."""
    a, b = _normalise(grey8), _normalise(phase)
    cc, w = _multistart(a, b, pyramid_factor, STANDARD_GRID)
    search = "standard"
    if cc < gate["min_ecc"]:
        cc2, w2 = _multistart(a, b, pyramid_factor, WIDE_GRID)
        if cc2 > cc:
            cc, w, search = cc2, w2, "wide"
    A = w[:, :2]
    sx, sy = float(np.hypot(A[0, 0], A[1, 0])), float(np.hypot(A[0, 1], A[1, 1]))
    rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    shear = float((A[0, 0] * A[0, 1] + A[1, 0] * A[1, 1]) / max(sx * sy, 1e-9))
    sampled = sample_float_in_8bit_frame(phase, w, grey8.shape)
    ok = np.isfinite(sampled)
    pearson = float(np.corrcoef(sampled[ok], grey8[ok].astype(np.float64))[0, 1]) if ok.sum() >= 1000 else float("nan")
    rho, shift = _geometry_evidence(phase, grey8, w) if ok.sum() >= 1000 else (float("nan"), float("nan"))
    reg = Registration(w, float(cc), pearson, rho, shift, sx, sy, rot, shear, search)
    primary = reg.ecc >= gate["min_ecc"]
    geometric = bool(reg.peak_shift_px <= gate["max_peak_shift_px"] and reg.spearman >= gate["min_spearman"]
                     and (reg.spearman - reg.pearson) >= gate["min_rank_gain"])
    borderline = (not primary) and reg.ecc >= gate["min_ecc_borderline"] and geometric
    reg.accepted = bool((primary or borderline) and reg.pearson >= gate["min_pearson"])
    reg.basis = ("primary" if primary else "borderline_nonlinear_export") if reg.accepted else "rejected"
    return reg


def warp_mask(mask8: np.ndarray, warp: np.ndarray, out_shape, ignore_index: int = 255) -> np.ndarray:
    """Carry the 8-bit-frame label mask onto the float grid (nearest neighbour); uncovered pixels -> ignore_index."""
    if (mask8 == ignore_index).any():
        raise ValueError("the 8-bit mask already contains the ignore label")
    h, w = out_shape
    return cv2.warpAffine(mask8.astype(np.uint8), warp.astype(np.float32), (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=int(ignore_index))
