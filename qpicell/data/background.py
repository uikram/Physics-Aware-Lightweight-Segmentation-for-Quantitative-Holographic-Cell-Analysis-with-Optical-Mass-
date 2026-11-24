"""Annotation-free background phase reference (decision D-3).

The background of every phase map is a smooth surface b(x, y) = polynomial of order p (p = 2: offset, tilt and
curvature) fitted to the pixels that are not part of any object. It is estimated from the phase map alone, without
annotation or field-of-view information, so the same function serves training, evaluation and new data. The phase
used everywhere (network input, physics-aware losses, integrated phase, dry mass) is  phi = phase - b, in radians.

  1. start: smoothed phase below its Otsu threshold (cells are the bright class) and with low local texture
     (7 x 7 standard deviation below the 70th percentile of those pixels), minus a `grow`-px margin around the rest
  2. residual r = phase - b; noise s = 1.4826 * median |r| over the background residuals r <= 0 (cells only add phase,
     so the negative side of the residual is pure background noise)
  3. object = |r| > k s, grown by `grow` px (cells, halos, debris); background = all other pixels
  4. least-squares fit of the polynomial on the background (order 0: median); repeat 2-4 until < 0.1 % of the pixels
     change their class (at most max_iter iterations)
The estimate is equivariant to a constant: background(phase + c) = background(phase) + c (tested), so a constant
offset in the provided maps never matters.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi


def design_matrix(shape, order: int) -> np.ndarray:
    """Polynomial terms on coordinates normalised to [-1, 1]: 1 | x, y | x^2, x y, y^2."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = ((xx - w / 2) / (w / 2)).ravel(); y = ((yy - h / 2) / (h / 2)).ravel()
    cols = [np.ones_like(x)]
    if order >= 1:
        cols += [x, y]
    if order >= 2:
        cols += [x * x, x * y, y * y]
    return np.stack(cols, 1)


def fit_surface(phase: np.ndarray, mask: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """(surface, coefficients) of the order-`order` polynomial fitted to phase[mask]; order 0 uses the median."""
    if order == 0:
        c = np.array([float(np.median(phase[mask]))])
        return np.full(phase.shape, c[0]), c
    A = design_matrix(phase.shape, order); m = mask.ravel()
    c, *_ = np.linalg.lstsq(A[m], phase.ravel()[m], rcond=None)
    return (A @ c).reshape(phase.shape), c


def surface_from_coefficients(coeffs, shape, order: int) -> np.ndarray:
    return (design_matrix(shape, order) @ np.asarray(coeffs, float)).reshape(shape)


def estimate_background(phase: np.ndarray, order: int = 2, k: float = 3.0, grow: int = 5, max_iter: int = 30,
                        region: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """Annotation-free background surface. `region` optionally restricts the pixels allowed in the background set."""
    from skimage.filters import threshold_otsu
    phase = np.asarray(phase, np.float64)
    region = np.ones(phase.shape, bool) if region is None else region
    smooth = ndi.gaussian_filter(phase, 2)
    t = threshold_otsu(smooth[region])
    local_sd = np.sqrt(np.maximum(ndi.uniform_filter(phase ** 2, 7) - ndi.uniform_filter(phase, 7) ** 2, 0))
    low = region & (smooth < t)
    bg = low & (local_sd < np.percentile(local_sd[low], 70))
    bg &= ~ndi.binary_dilation(region & (smooth >= t), iterations=grow)
    if bg.sum() < 0.01 * region.sum():
        bg = low
    surf, coeffs = fit_surface(phase, bg, order)
    noise, it = float("nan"), 0
    for it in range(1, max_iter + 1):
        r = phase - surf; rb = r[bg]; neg = rb[rb <= 0]
        noise = 1.4826 * float(np.median(np.abs(neg)) if neg.size > 100 else np.median(np.abs(rb - np.median(rb))))
        new = region & ~ndi.binary_dilation(np.abs(r) > k * noise, iterations=grow)
        if new.sum() < 0.02 * region.sum():
            raise ValueError("background set below 2 % of the image")
        surf, coeffs = fit_surface(phase, new, order)
        changed = float((new ^ bg).mean()); bg = new
        if it > 1 and changed < 1e-3:
            break
    return surf, {"coefficients": coeffs, "iterations": it, "noise_sd_rad": noise, "background_fraction": float(bg.mean()),
                  "background_mask": bg}


def annotation_background(phase: np.ndarray, annotated_fg: np.ndarray, valid: np.ndarray, order: int,
                          exclusion_px: int = 5) -> np.ndarray:
    """Same-order fit on the ANNOTATED background (> exclusion_px from any annotated cell). Used only to validate the
    annotation-free reference; never part of the pipeline."""
    bg = valid & ~ndi.binary_dilation(annotated_fg, iterations=exclusion_px)
    return fit_surface(phase, bg, order)[0]
