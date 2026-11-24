"""Cell instances, territories and contact lines (decision D-4).

The annotation marks cell pixels and classes but draws touching cells as one connected blob (about 14 % of the
blobs are clusters of 2-8 cells). One cell = one region of a marker-controlled watershed of the annotated foreground:
markers = h-maxima (h = 3 px) of the Gaussian-smoothed distance to the background, flooded on the negative distance.

territories(): every pixel within `radius` px of a cell is assigned to its nearest cell (truncated Voronoi region);
  a cell's predicted integrated phase is collected in its territory.
contact_lines(): cell pixels that have a 4-neighbour belonging to ANOTHER cell (the line where two cells touch).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def split_cells(foreground: np.ndarray, h: int = 3) -> np.ndarray:
    """int32 instance map 1..N of the boolean foreground."""
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed
    fg = np.asarray(foreground, bool)
    if not fg.any():
        return np.zeros(fg.shape, np.int32)
    dist = ndi.distance_transform_edt(fg)
    markers = ndi.label(h_maxima(ndi.gaussian_filter(dist, 1), h))[0]
    lab = watershed(-dist, markers, mask=fg).astype(np.int32)
    rest = fg & (lab == 0)                       # foreground without a marker keeps its own component
    if rest.any():
        r, _ = ndi.label(rest)
        lab[rest] = r[rest] + lab.max()
    ids = np.unique(lab)
    ids = ids[ids > 0]                           # relabel 1..N; 0 stays background (also when there is no background pixel)
    lut = np.zeros(int(lab.max()) + 1, np.int32); lut[ids] = np.arange(1, len(ids) + 1)
    return lut[lab]


def territories(cells: np.ndarray, radius: float) -> np.ndarray:
    if cells.max() == 0:
        return np.zeros_like(cells)
    dist, (iy, ix) = ndi.distance_transform_edt(cells == 0, return_indices=True)
    t = cells[iy, ix]
    t[dist > radius] = 0
    return t.astype(cells.dtype)


def contact_lines(cells: np.ndarray) -> np.ndarray:
    t = np.zeros(cells.shape, bool)
    a, b = cells[:, :-1], cells[:, 1:]
    m = (a > 0) & (b > 0) & (a != b); t[:, :-1] |= m; t[:, 1:] |= m
    a, b = cells[:-1, :], cells[1:, :]
    m = (a > 0) & (b > 0) & (a != b); t[:-1, :] |= m; t[1:, :] |= m
    return t


def touches_unannotated(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Boolean per label (index 0..max) - True if the region touches the unannotated area (cut by the field of view)."""
    n = int(labels.max())
    flag = np.zeros(n + 1, bool)
    if n and not valid.all():
        near = ndi.binary_dilation(~valid, iterations=1)
        flag[np.unique(labels[near & (labels > 0)])] = True
    flag[0] = False
    return flag
