"""Instance reconstruction from the PREDICTED foreground only (no ground truth is used).

A2 (primary): qpicell.data.cells.split_cells(pred > 0, h) with h = cells.watershed_h = 3 - the same marker-controlled
   watershed that defines the reference cells: markers = h-maxima of the Gaussian-smoothed distance transform of the
   BINARY foreground, flooded on the negative distance inside the foreground; foreground without a marker keeps its
   own connected component. Every foreground pixel receives exactly one label; no phase value is used.
CC (diagnostic): 4-connected components of the predicted foreground.
Both run on the whole 768 x 768 prediction; the phase map is never touched.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi

from qpicell.data.cells import split_cells


def a2(pred: np.ndarray, h: int) -> np.ndarray:
    fg = np.asarray(pred) > 0
    lab = split_cells(fg, int(h)).astype(np.int32)
    return lab


def cc(pred: np.ndarray) -> np.ndarray:
    return ndi.label(np.asarray(pred) > 0)[0].astype(np.int32)


def check_assignment(pred: np.ndarray, lab: np.ndarray) -> Dict:
    """Every predicted-foreground pixel must carry a label and no background pixel may carry one."""
    fg = np.asarray(pred) > 0
    out = {"fg_px": int(fg.sum()), "unassigned_fg_px": int((fg & (lab == 0)).sum()),
           "labels_outside_fg_px": int(((lab > 0) & ~fg).sum()), "n_instances": int(len(np.unique(lab[lab > 0])))}
    if out["unassigned_fg_px"] or out["labels_outside_fg_px"]:
        raise AssertionError(f"instance map does not tile the predicted foreground: {out}")
    return out


def reconstruct(pred: np.ndarray, h: int) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    labs = {"A2": a2(pred, h), "CC": cc(pred)}
    return labs, {k: check_assignment(pred, v) for k, v in labs.items()}
