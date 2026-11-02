"""Screen the annotated validation cells for cells superposed along the optical axis (Reviewer 2, comment 4).

Two RBCs stacked on top of each other occupy the footprint of one cell but delay the light roughly twice as much,
so their mean phase per pixel is about twice that of a single cell of the same type. This script finds every
annotated cell (4-connected component of the annotation, 15-200 um^2, not touching an unannotated region; the same
definition as qpi/analysis/drymass.py) and compares its mean background-referenced phase with the median of the
cells of the same class on the same storage day. A cell is a candidate when

    mean phase >= RATIO_MIN x median mean phase of its class and day, and
    area       <= AREA_MAX  x median area of its class and day (single-cell footprint).

Cells flagged as phase-wrapped in cells_gt.csv are listed separately, because a superposed pair could exceed 2 pi.
Candidates are written to a CSV and shown in a contact sheet for visual inspection; the screen does not decide by
itself whether a cell is a superposed pair.

Usage:
    python artifacts/code/reviewer_r2_4_check.py            (run from the repository root)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

RATIO_MIN = 1.6
AREA_MAX = 1.25
PX_UM = 0.1441
AREA_RANGE = (15.0, 200.0)


def cells_of(stem, gt, phase, day):
    valid = gt != 255
    g = np.where(valid, gt, 0)
    lab, n = ndi.label((g > 0) & valid)              # 4-connected, as in qpi/analysis/common.components
    if n == 0:
        return []
    near = ndi.binary_dilation(~valid, iterations=1) if not valid.all() else np.zeros_like(valid)
    cut = np.zeros(n + 1, bool)
    cut[np.unique(lab[near & (lab > 0)])] = True
    idx = np.arange(1, n + 1)
    npx = ndi.sum(np.ones_like(phase), lab, idx)
    s = ndi.sum(phase, lab, idx)
    mx = ndi.maximum(phase, lab, idx)
    rows = []
    for i, k, si, m in zip(idx, npx, s, mx):
        area = k * PX_UM ** 2
        if not (AREA_RANGE[0] <= area <= AREA_RANGE[1]) or cut[i]:
            continue
        cls = int(np.bincount(g[lab == i], minlength=5)[1:].argmax() + 1)
        ys, xs = np.nonzero(lab == i)
        rows.append(dict(stem=stem, storage_day=day, cell=int(i), class_idx=cls, area_um2=area,
                         mean_phase_rad=si / k, max_phase_rad=float(m), cy=int(ys.mean()), cx=int(xs.mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repository root (contains data/processed and outputs/)")
    ap.add_argument("--out", default="artifacts/audit")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    proc = root / "data" / "processed"
    man = pd.read_csv(proc / "manifest.csv")
    val = man[man.split == "val"]
    rows = []
    for stem, day in zip(val.stem, val.storage_day):
        gt = tifffile.imread(proc / "val" / "mask" / f"{stem}.tif")
        ph = tifffile.imread(proc / "val" / "phase" / f"{stem}.tif").astype(np.float64)
        rows += cells_of(stem, gt, ph, int(day))
    c = pd.DataFrame(rows)
    gtc = pd.read_csv(root / "outputs" / "cell_analysis" / "annotation" / "cells_gt.csv")[["stem", "cell", "wrapped"]]
    c = c.merge(gtc, on=["stem", "cell"], how="left")
    c["wrapped"] = c["wrapped"].fillna(False).astype(bool)

    ref = c[~c.wrapped].groupby(["storage_day", "class_idx"]).agg(ref_phase=("mean_phase_rad", "median"),
                                                                  ref_area=("area_um2", "median")).reset_index()
    c = c.merge(ref, on=["storage_day", "class_idx"], how="left")
    c["phase_ratio"] = c.mean_phase_rad / c.ref_phase
    c["area_ratio"] = c.area_um2 / c.ref_area
    c["candidate"] = (c.phase_ratio >= RATIO_MIN) & (c.area_ratio <= AREA_MAX)
    c.sort_values("phase_ratio", ascending=False).to_csv(out / "reviewer_r2_4_screen.csv", index=False)

    cand = c[c.candidate].sort_values("phase_ratio", ascending=False)
    print(f"annotated cells screened: {len(c)} (wrapped: {int(c.wrapped.sum())})")
    print(f"phase ratio to class-and-day median: median {c.phase_ratio.median():.2f}, "
          f"99th percentile {c.phase_ratio.quantile(.99):.2f}, max {c.phase_ratio.max():.2f}")
    print(f"candidates (ratio >= {RATIO_MIN}, area <= {AREA_MAX} x median): {len(cand)}, "
          f"of which wrapped: {int(cand.wrapped.sum())}")
    print(cand[["stem", "storage_day", "cell", "class_idx", "area_um2", "mean_phase_rad", "max_phase_rad",
                "phase_ratio", "area_ratio", "wrapped"]].round(2).to_string(index=False))

    try:                                              # contact sheet of the 24 cells with the highest phase ratio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = c.sort_values("phase_ratio", ascending=False).head(24)
        fig, axs = plt.subplots(4, 6, figsize=(12, 8.5))
        for ax, (_, r) in zip(axs.ravel(), top.iterrows()):
            ph = tifffile.imread(proc / "val" / "phase" / f"{r.stem}.tif")
            y0, x0 = max(0, r.cy - 40), max(0, r.cx - 40)
            ax.imshow(ph[y0:y0 + 80, x0:x0 + 80], cmap="gray", vmin=0, vmax=max(3.0, float(r.max_phase_rad)))
            ax.set_title(f"{r.stem} d{r.storage_day} c{r.class_idx}\nx{r.phase_ratio:.2f} A{r.area_um2:.0f}"
                         + (" W" if r.wrapped else ""), fontsize=7)
            ax.axis("off")
        fig.suptitle("Highest mean phase relative to class-and-day median (x: ratio, A: area um^2, W: wrapped)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "reviewer_r2_4_top24.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
