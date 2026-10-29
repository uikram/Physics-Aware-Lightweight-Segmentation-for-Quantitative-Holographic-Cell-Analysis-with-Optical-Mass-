"""Detection of phase-wrapped cells (DECISIONS.md D-21).

The date-named float maps are confined to [0, 2*pi]; cells whose true phase exceeds 2*pi are wrapped back to small
values, which shows up as (i) pixels at the 2*pi ceiling and (ii) jumps of about 2*pi between neighbouring pixels.
Automatic 2-D unwrapping was tried and is not reliable on these maps (it leaves jumps behind and produces implausible
values), so affected cells are excluded from quantitative phase results instead.

The rule is applied to the RAW phase (processed phase + the offset removed by prepare_dataset), because the ceiling is a
property of the file as provided.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import ndimage as ndi


def wrapped_pixels(raw: np.ndarray, ceiling: float, tolerance: float, jump: float) -> np.ndarray:
    """Pixels at the ceiling (raw >= ceiling - tolerance) or at either end of a neighbour jump larger than ``jump``."""
    flag = raw >= (ceiling - tolerance)
    dy = np.abs(np.diff(raw, axis=0)) > jump
    dx = np.abs(np.diff(raw, axis=1)) > jump
    flag[:-1] |= dy
    flag[1:] |= dy
    flag[:, :-1] |= dx
    flag[:, 1:] |= dx
    return flag


def flag_components(lab: np.ndarray, n: int, flagged_px: np.ndarray, dilation_px: int) -> np.ndarray:
    """Boolean array (index = label): component has a flagged pixel within ``dilation_px`` of it."""
    out = np.zeros(n + 1, bool)
    if n == 0 or not flagged_px.any():
        return out
    near = ndi.binary_dilation(flagged_px, iterations=int(dilation_px)) if dilation_px > 0 else flagged_px
    out[np.unique(lab[near & (lab > 0)])] = True
    return out


def rule(cfg: Dict) -> Dict:
    w = cfg["analysis"]["wrapping"]
    return {"ceiling": float(w["ceiling_rad"]), "tolerance": float(w["ceiling_tolerance_rad"]),
            "jump": float(w["jump_rad"]), "dilation_px": int(w["dilation_px"])}


def summarise(cells, class_names: List[str]) -> Dict:
    """Counts of wrapped cells overall, per class, per storage day and the number of images with at least one."""
    if len(cells) == 0:
        return {"n_cells": 0, "n_wrapped": 0, "pct_wrapped": None}
    w = cells["wrapped"].astype(bool)
    by_class = {class_names[int(c)]: {"n_cells": int(len(g)), "n_wrapped": int(g["wrapped"].sum()),
                                      "pct_wrapped": float(100 * g["wrapped"].mean())}
                for c, g in cells.groupby("class_idx")}
    by_day = [{"storage_day": int(d), "n_cells": int(len(g)), "n_wrapped": int(g["wrapped"].sum()),
               "pct_wrapped": float(100 * g["wrapped"].mean())} for d, g in cells.groupby("storage_day")]
    per_img = cells.groupby("stem")["wrapped"].sum()
    return {"n_cells": int(len(cells)), "n_wrapped": int(w.sum()), "pct_wrapped": float(100 * w.mean()),
            "n_images": int(cells["stem"].nunique()), "n_images_with_wrapped": int((per_img > 0).sum()),
            "by_class": by_class, "by_storage_day": by_day}
