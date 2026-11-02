"""Registration of a float32 phase map (radians) onto the 8-bit export frame that carries the masks.

Background
    The annotation masks were drawn on 8-bit exports of the phase maps. Those exports are NOT in the
    frame of the float32 maps: they are cropped/zoomed (isotropically or with different x/y scales)
    and shifted, and each was stretched with its own linear grey-level mapping. The masks are therefore
    carried onto the float32 grid by an affine transform estimated with intensity-based registration
    (ECC, invariant to the per-image linear grey-level mapping).

Convention
    ``estimate_transform`` returns W (2x3) with   x_float = W[:, :2] @ x_8bit + W[:, 2]
    (x, y pixel coordinates). ``warp_mask_to_float`` produces, for every float pixel, the label of the
    8-bit mask pixel it falls on (nearest neighbour). Float pixels that fall outside the 8-bit image
    receive ``ignore_index`` (no annotation exists there).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _ecc(template: np.ndarray, image: np.ndarray, init: np.ndarray, iters: int, eps: float, gauss: int):
    try:
        cc, warp = cv2.findTransformECC(
            template, image, init.astype(np.float32), cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps), None, gauss)
        return float(cc), warp
    except cv2.error:
        return -1.0, init


@dataclass
class Registration:
    warp: np.ndarray            # 2x3, x_float = A x_8bit + t
    ecc_correlation: float
    pearson_r: float            # float-vs-8-bit correlation inside the overlap after warping
    scale_x: float              # sqrt(A00^2 + A10^2)
    scale_y: float
    rotation_deg: float
    shear: float
    grey_per_rad: float         # slope of the linear fit  grey = a * phase + b  inside the overlap
    grey_offset: float
    spearman_r: float = float("nan")      # rank correlation float-vs-8-bit in the overlap (robust to a non-linear grey export)
    peak_shift_px: float = float("nan")   # distance (8-bit frame px) of the correlation maximum from zero shift, searched in +-3 px
    search: str = "standard"              # "standard" or "extended" (wide multi-start used when the standard search did not converge)


_STD_GRID = dict(sx=(0.6, 0.8, 0.9, 1.0, 1.2), sy=(0.6, 0.7, 0.9, 1.0, 1.2), dx=(-10, 0, 10), dy=(-10, 0, 10))
# Extended multi-start (translations up to +-240 px in the full-resolution frame): used only when the standard search
# does not reach the quality threshold, so images that already register well are never affected by it.
_EXT_GRID = dict(sx=(0.6, 0.8, 1.0, 1.2), sy=(0.6, 0.8, 1.0, 1.2), dx=tuple(range(-60, 61, 15)), dy=tuple(range(-60, 61, 15)))


def _search(a: np.ndarray, b: np.ndarray, f: int, grid: dict):
    """Coarse multi-start ECC followed by two refinements. Returns (final ECC correlation, warp 2x3 at full resolution)."""
    a_s = cv2.resize(a, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
    b_s = cv2.resize(b, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
    best_cc, best_w = -1.0, None
    for sx in grid["sx"]:
        for sy in grid["sy"]:
            for dx in grid["dx"]:
                for dy in grid["dy"]:
                    init = np.array([[sx, 0, dx], [0, sy, dy]], np.float32)
                    cc, w = _ecc(a_s, b_s, init, 100, 1e-4, 1)
                    if cc > best_cc:
                        best_cc, best_w = cc, w
    if best_w is None:
        raise RuntimeError("registration failed to initialise")
    w = best_w.copy()
    w[:, 2] *= f
    cc = best_cc
    for sc in (2, 1):
        a_l = cv2.resize(a, None, fx=1 / sc, fy=1 / sc, interpolation=cv2.INTER_AREA)
        b_l = cv2.resize(b, None, fx=1 / sc, fy=1 / sc, interpolation=cv2.INTER_AREA)
        w2 = w.copy()
        w2[:, 2] /= sc
        cc, w2 = _ecc(a_l, b_l, w2, 300, 1e-6, 3)
        w = w2.copy()
        w[:, 2] *= sc
    return cc, w


def _sample_float_in_8bit_frame(phase: np.ndarray, w: np.ndarray, shape) -> np.ndarray:
    h, wd = shape
    sampled = cv2.warpAffine(phase.astype(np.float32), w.astype(np.float32), (wd, h),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
    ok = np.isfinite(sampled)
    ok &= cv2.warpAffine(np.ones_like(phase, np.float32), w.astype(np.float32), (wd, h),
                         flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0.5
    return np.where(ok, sampled, np.nan)


def geometry_check(phase: np.ndarray, grey8: np.ndarray, w: np.ndarray, max_shift: int = 3):
    """Alignment evidence that does not assume a linear grey-level export.

    Returns (Spearman rank correlation in the overlap, distance in px of the correlation maximum from zero shift).
    The float map is re-sampled with the warp displaced by every integer shift in [-max_shift, max_shift]^2 (8-bit
    frame); a correctly registered pair has its maximum at zero shift."""
    from scipy.stats import rankdata
    base = _sample_float_in_8bit_frame(phase, w, grey8.shape)
    ok0 = np.isfinite(base)
    ok0 = cv2.erode(ok0.astype(np.uint8), np.ones((2 * max_shift + 3, 2 * max_shift + 3), np.uint8)) > 0
    if ok0.sum() < 1000:
        return float("nan"), float("nan")
    g = grey8.astype(np.float64)
    rho = float(np.corrcoef(rankdata(base[ok0]), rankdata(g[ok0]))[0, 1])
    A = w[:, :2]
    best, best_d = -2.0, (0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            ws = w.copy()
            ws[:, 2] = w[:, 2] - A @ np.array([dx, dy], np.float64)       # sample the float map at x8 - d
            sm = _sample_float_in_8bit_frame(phase, ws, grey8.shape)
            m = ok0 & np.isfinite(sm)
            if m.sum() < 1000:
                continue
            c = float(np.corrcoef(sm[m], g[m])[0, 1])
            if c > best:
                best, best_d = c, (dy, dx)
    return rho, float(np.hypot(*best_d))


def estimate_transform(phase: np.ndarray, grey8: np.ndarray, pyramid_factor: int = 4,
                       retry_below: float = 0.98) -> Registration:
    """Estimate the affine map 8-bit frame -> float frame.

    Standard multi-start search first; if its final ECC correlation is below ``retry_below`` an extended wide-translation
    multi-start is run and the better of the two is kept. The quality gate is applied by the caller on the result."""
    a, b = _norm(grey8), _norm(phase)
    cc, w = _search(a, b, pyramid_factor, _STD_GRID)
    search = "standard"
    if cc < retry_below:
        cc2, w2 = _search(a, b, pyramid_factor, _EXT_GRID)
        if cc2 > cc:
            cc, w, search = cc2, w2, "extended"

    A = w[:, :2]
    sx_, sy_ = float(np.hypot(A[0, 0], A[1, 0])), float(np.hypot(A[0, 1], A[1, 1]))
    rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    shear = float((A[0, 0] * A[0, 1] + A[1, 0] * A[1, 1]) / max(sx_ * sy_, 1e-9))   # cosine of the axis angle (0 = no shear)
    # correlation and grey-level mapping inside the overlap (8-bit frame sampled from the float map)
    sampled = _sample_float_in_8bit_frame(phase, w, grey8.shape)
    ok = np.isfinite(sampled)
    if ok.sum() < 1000:
        return Registration(w, float(cc), float("nan"), sx_, sy_, rot, shear, float("nan"), float("nan"),
                            search=search)
    p = np.polyfit(sampled[ok], grey8[ok].astype(np.float64), 1)
    r = float(np.corrcoef(sampled[ok], grey8[ok].astype(np.float64))[0, 1])
    rho, shift = geometry_check(phase, grey8, w)
    return Registration(w, float(cc), r, sx_, sy_, rot, shear, float(p[0]), float(p[1]),
                        spearman_r=rho, peak_shift_px=shift, search=search)


def warp_mask_to_float(mask8: np.ndarray, warp: np.ndarray, out_shape: Tuple[int, int],
                       ignore_index: int = 255) -> np.ndarray:
    """Carry the 8-bit-frame label mask onto the float grid. Uncovered pixels -> ``ignore_index``."""
    h, w = out_shape
    # label 255 must not collide with a real class
    if (mask8 == ignore_index).any():
        raise ValueError("input mask already contains the ignore label")
    # `warp` maps 8-bit coordinates -> float coordinates, i.e. it is the forward (src -> dst) map of
    # cv2.warpAffine (which inverts it internally for sampling).
    out = cv2.warpAffine(mask8.astype(np.uint8), warp.astype(np.float32), (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=int(ignore_index))
    return out
