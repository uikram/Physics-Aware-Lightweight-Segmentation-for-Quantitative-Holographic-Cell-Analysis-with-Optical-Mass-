"""Pixels near phase wrapping in the PROVIDED float map (radians, as delivered, before the background reference):
within ceiling_tolerance of the 2 pi ceiling, or at a jump larger than jump_rad between neighbours. Cells within
dilation_px of such a pixel are left out of every integrated-phase and dry-mass result (segmentation is still trained
and scored on them)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage as ndi


def wrapped_pixels(raw_phase: np.ndarray, rule: Dict) -> np.ndarray:
    flag = raw_phase >= (rule["ceiling_rad"] - rule["ceiling_tolerance_rad"])
    dy = np.abs(np.diff(raw_phase, axis=0)) > rule["jump_rad"]
    dx = np.abs(np.diff(raw_phase, axis=1)) > rule["jump_rad"]
    flag[:-1] |= dy; flag[1:] |= dy; flag[:, :-1] |= dx; flag[:, 1:] |= dx
    return flag


def flagged_labels(labels: np.ndarray, wrapped: np.ndarray, dilation_px: int) -> np.ndarray:
    n = int(labels.max())
    out = np.zeros(n + 1, bool)
    if n == 0 or not wrapped.any():
        return out
    near = ndi.binary_dilation(wrapped, iterations=int(dilation_px)) if dilation_px > 0 else wrapped
    out[np.unique(labels[near & (labels > 0)])] = True
    out[0] = False
    return out
