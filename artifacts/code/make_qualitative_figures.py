#!/usr/bin/env python
"""Qualitative figures that need the phase maps: run on the machine that has data/processed (NOT in the repository).

    python artifacts/code/make_qualitative_figures.py [--processed-root data/processed] [--seed 42]

Writes artifacts/figures/{main,si}/:
    FQ1_qualitative_segmentation   replaces old Fig. 6: per class, the 256 x 256 window with most annotated pixels of that class
                                   (over all validation images, raster tie-break): phase, annotation, MobileSAM r8, EdgeSAM r32,
                                   MobileNet-UNet r8 (seed 42, last-epoch predictions)
    FQS1_gradient_alignment        replaces old Fig. S-4: phase, |grad phase|, contours of both preferred models on |grad phase|
    FQS2_storage_examples          replaces old Fig. S-5: first validation image (alphabetical) of five evenly spaced storage days

The selection rules are deterministic and identical to manuscript/figures.py (fig6, figS4, figS5); only the models shown change
to the preferred configurations. Mask values: 1 discocyte, 2 spherocyte, 3 echinocyte, 4 stomatocyte (D-25).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from skimage.measure import find_contours  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from make_figures import ARCH_COLOR, CLASS_COLOR, save  # noqa: E402
from qpi.analysis.common import load_predictions, load_validation  # noqa: E402
from qpi.config import load_config  # noqa: E402

SHOW = C.PREFERRED + [C.BASELINE]
LABEL_CMAP = ListedColormap(["#00000000"] + [CLASS_COLOR[k] for k in C.CLASSES])


def best_window(mask: np.ndarray, valid: np.ndarray, size: int = 256, stride: int = 32):
    best, pos = -1, (0, 0)
    H, W = mask.shape
    m = mask & valid
    for y in range(0, H - size + 1, stride):
        for x in range(0, W - size + 1, stride):
            n = int(m[y:y + size, x:x + size].sum())
            if n > best:
                best, pos = n, (y, x)
    return pos


def overlay(ax, phase, labels, valid, lo, hi):
    ax.imshow(phase, cmap="gray", vmin=lo, vmax=hi)
    ax.imshow(np.where(valid, labels, 0), cmap=LABEL_CMAP, vmin=0, vmax=4, alpha=0.45, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def fq1(val, preds):
    fig, axes = plt.subplots(4, 2 + len(SHOW), figsize=(7.4, 6.2))
    for r, cl in enumerate(C.CLASSES):
        cls = r + 1
        stems = sorted(val)
        best = max(stems, key=lambda s: (int(((val[s]["gt"] == cls) & val[s]["valid"]).sum()), -stems.index(s)))
        v = val[best]
        y, x = best_window(v["gt"] == cls, v["valid"])
        sl = (slice(y, y + 256), slice(x, x + 256))
        lo, hi = np.percentile(v["phase"][sl], [1, 99])
        axes[r, 0].imshow(v["phase"][sl], cmap="gray", vmin=lo, vmax=hi)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        axes[r, 0].grid(False)
        axes[r, 0].set_ylabel(f"{C.CLASS_DISPLAY[cl]}\n{best} (day {v['day']})", fontsize=6.5)
        panels = [("Annotation", v["gt"])] + [(f"{C.ARCH_LABEL[a]} r{rk}", preds[(a, rk)][best]) for a, rk in SHOW]
        for j, (t, lab) in enumerate(panels, start=1):
            overlay(axes[r, j], v["phase"][sl], lab[sl], v["valid"][sl], lo, hi)
            if r == 0:
                axes[r, j].set_title(t, fontsize=7)
        if r == 0:
            axes[r, 0].set_title("Phase (rad)", fontsize=7)
    fig.legend(handles=[Patch(color=CLASS_COLOR[k], label=C.CLASS_DISPLAY[k]) for k in C.CLASSES], loc="lower center", ncol=4)
    save(fig, "main", "FQ1_qualitative_segmentation")


def fqs1(val, preds):
    stems = sorted(val)
    stem = max(stems, key=lambda s: (int(((val[s]["gt"] > 0) & val[s]["valid"]).sum()), -stems.index(s)))
    v = val[stem]
    y, x = best_window(v["gt"] > 0, v["valid"])
    sl = (slice(y, y + 256), slice(x, x + 256))
    gy, gx = np.gradient(v["phase"])
    g = np.hypot(gx, gy)
    lo, hi = np.percentile(v["phase"][sl], [1, 99])
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5))
    ax[0].imshow(v["phase"][sl], cmap="gray", vmin=lo, vmax=hi)
    ax[0].set_title("Phase (rad)")
    ax[1].imshow(g[sl], cmap="magma")
    ax[1].set_title("|∇φ|")
    ax[2].imshow(g[sl], cmap="magma")
    for a, rk in C.PREFERRED:
        for cont in find_contours((preds[(a, rk)][stem][sl] > 0).astype(float), 0.5):
            ax[2].plot(cont[:, 1], cont[:, 0], color=ARCH_COLOR[a], lw=0.8)
    ax[2].set_title(f"Contours on |∇φ| ({stem})")
    ax[2].legend(handles=[Patch(color=ARCH_COLOR[a], label=f"{C.ARCH_LABEL[a]} r{rk}") for a, rk in C.PREFERRED], fontsize=6, loc="lower right")
    for a_ in ax:
        a_.set_xticks([])
        a_.set_yticks([])
        a_.grid(False)
    save(fig, "si", "FQS1_gradient_alignment")


def fqs2(val):
    days = sorted({v["day"] for v in val.values()})
    pick = [days[int(round(i))] for i in np.linspace(0, len(days) - 1, min(5, len(days)))]
    fig, axes = plt.subplots(1, len(pick), figsize=(1.9 * len(pick), 2.1), squeeze=False)
    for ax, d in zip(axes[0], pick):
        stem = sorted(s for s in val if val[s]["day"] == d)[0]
        ph = val[stem]["phase"]
        lo, hi = np.percentile(ph[val[stem]["valid"]], [1, 99.5])
        ax.imshow(ph, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"Day {d}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    save(fig, "si", "FQS2_storage_examples")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", default=None, help="default: paths.processed_root of configs/base.yaml")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = load_config()
    if a.processed_root:
        cfg["paths_abs"]["processed_root"] = Path(a.processed_root).resolve()
    man = Path(cfg["paths_abs"]["processed_root"]) / "manifest.csv"
    if not man.exists():
        print(f"make_qualitative_figures.py: {man} not found. These figures need the processed phase maps "
              "(run scripts/prepare_dataset.py, or run this on the server that has data/processed).", file=sys.stderr)
        return 1
    val = load_validation(cfg)
    preds = {(ar, rk): load_predictions(C.run_dir(ar, rk, "full", a.seed), "last") for ar, rk in SHOW}
    for k, p in preds.items():
        missing = sorted(set(val) - set(p))
        if missing:
            raise RuntimeError(f"{k}: no prediction for {missing[:3]} ...")
    fq1(val, preds)
    fqs1(val, preds)
    fqs2(val)
    print("[qualitative] FQ1, FQS1, FQS2 written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
