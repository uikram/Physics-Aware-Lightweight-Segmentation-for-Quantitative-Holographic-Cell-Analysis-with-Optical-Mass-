"""Shape of a segmented cell: projected area, perimeter and circularity.

    A = N dx^2                 N = pixels of the instance, dx = effective object-plane pixel pitch (um/px)
    P = Crofton perimeter      estimated from the number of boundary crossings along 4 directions, which is much less
                               sensitive to pixelation than counting boundary pixels (skimage.measure.perimeter_crofton)
    C = 4 pi A / P^2           1 for a perfect disc, smaller for an irregular or elongated outline

Same definition as the submitted manuscript (Section 2.6). Every instance is measured on the native grid, on its own,
so touching cells separated by the watershed are measured separately.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import perimeter_crofton


def circularity_per_label(labels: np.ndarray, n: int, pixel_size_um: float) -> Tuple[np.ndarray, np.ndarray]:
    """Perimeter (um) and circularity of every label 1..n of an instance map. Index 0 is unused (background).

    Each instance is cropped with a one-pixel margin and measured alone, so a neighbouring instance never contributes
    to its outline. Labels without pixels give NaN."""
    P = np.full(n + 1, np.nan); C = np.full(n + 1, np.nan)
    if n < 1:
        return P, C
    boxes = ndi.find_objects(labels, max_label=n)
    for i, sl in enumerate(boxes, start=1):
        if sl is None:
            continue
        pad = (slice(max(sl[0].start - 1, 0), sl[0].stop + 1), slice(max(sl[1].start - 1, 0), sl[1].stop + 1))
        m = labels[pad] == i
        area_px = int(m.sum())
        if area_px == 0:
            continue
        per_px = float(perimeter_crofton(m, directions=4))
        P[i] = per_px * pixel_size_um
        C[i] = 4 * np.pi * (area_px * pixel_size_um ** 2) / P[i] ** 2 if P[i] > 0 else np.nan
    return P, C
