#!/usr/bin/env python
"""Figures and numbers that need the processed phase maps and masks (data/processed, NOT in the repository).

Run on the machine that has data/processed, from the repository root:

    python artifacts/code/make_data_figures.py [--processed-root data/processed] [--seed 42]

Writes to artifacts/figures/manuscript/ (same names as the placeholders in the Word files, so replacing the picture in Word
is a one-to-one swap):
    Fig6_qualitative.png             one 256 x 256 window per class (the window with most annotated pixels of that class, raster
                                     tie-break): phase, annotation, MobileNet-UNet (r=8), MobileSAM (r=8), EdgeSAM (r=8),
                                     EdgeSAM (r=32); all four predictions from the SAME training seed (--seed)
    FigS4_gradient_alignment.png     phase, |grad phase| and EdgeSAM (r=32) / MobileSAM (r=8) contours on |grad phase|
    FigS5_storage_images.png         phase maps of five storage days (0, 8, 19, 27, 47; first image of each day, alphabetical)
                                     with EdgeSAM (r=32) contours in the second row
    FigS6_component_fate.png         (a) kept predicted components / annotated cells per day (each seed); (b) fate of the
                                     predicted components inside the annotated field (mean of seeds), EdgeSAM (r=32)
and artifacts/audit/server_numbers.json with the numbers quoted in the manuscript that need the masks:
    unannotated fraction of the grid, day-15 foreground fraction, component fate on day 15, predicted / annotated foreground
    area on day 15, bridged annotated components, pixel confusion shares.
Definitions follow qpi/analysis/common.py: components = 4-connected foreground inside the annotated region; kept = 15-200 um2
and not touching the unannotated region (dilation by 1 px). Mask values: 1 discocyte, 2 spherocyte, 3 echinocyte, 4 stomatocyte.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage.measure import find_contours  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from make_manuscript_figures import ARCH_COL, CLS_COL, CLS_LAB, FIG, INK, label, save  # noqa: E402
from qpi.analysis.common import components, load_predictions, load_validation, touches_unannotated  # noqa: E402
from qpi.config import load_config  # noqa: E402

SHOW = [("mobilenet_unet", 8), ("mobile_sam", 8), ("edge_sam", 8), ("edge_sam", 32)]
LABEL_CMAP = ListedColormap(["#00000000"] + [CLS_COL[k] for k in C.CLASSES])
PX_UM2 = 0.1441 ** 2
LO, HI = 15.0, 200.0


def best_window(mask, valid, size=256, stride=32):
    best, pos = -1, (0, 0)
    m = mask & valid
    for y in range(0, m.shape[0] - size + 1, stride):
        for x in range(0, m.shape[1] - size + 1, stride):
            n = int(m[y:y + size, x:x + size].sum())
            if n > best:
                best, pos = n, (y, x)
    return pos


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def overlay(ax, phase, labels, valid, lo, hi):
    ax.imshow(phase, cmap="gray", vmin=lo, vmax=hi)
    ax.imshow(np.where(valid, labels, 0), cmap=LABEL_CMAP, vmin=0, vmax=4, alpha=0.5, interpolation="nearest")
    ax.imshow(np.where(valid, np.nan, 1.0), cmap="Greys", vmin=0, vmax=2, alpha=0.5)
    clean(ax)


def fig6(val, preds, seed):
    fig, axes = plt.subplots(4, 2 + len(SHOW), figsize=(7.0, 4.9))
    stems = sorted(val)
    for r, cl in enumerate(C.CLASSES):
        cls = r + 1
        best = max(stems, key=lambda s: (int(((val[s]["gt"] == cls) & val[s]["valid"]).sum()), -stems.index(s)))
        v = val[best]
        y, x = best_window(v["gt"] == cls, v["valid"])
        sl = (slice(y, y + 256), slice(x, x + 256))
        lo, hi = np.percentile(v["phase"][sl], [1, 99])
        axes[r, 0].imshow(v["phase"][sl], cmap="gray", vmin=lo, vmax=hi)
        clean(axes[r, 0])
        axes[r, 0].set_ylabel(f"{CLS_LAB[cl]}\nday {v['day']}", fontsize=6.5)
        panels = [("Annotation", v["gt"])] + [(label(a, rk), preds[(a, rk)][best]) for a, rk in SHOW]
        for j, (t, lab) in enumerate(panels, start=1):
            overlay(axes[r, j], v["phase"][sl], lab[sl], v["valid"][sl], lo, hi)
            if r == 0:
                axes[r, j].set_title(t, fontsize=6.5)
        if r == 0:
            axes[r, 0].set_title("Phase", fontsize=6.5)
    fig.legend(handles=[Patch(color=CLS_COL[k], label=CLS_LAB[k]) for k in C.CLASSES]
               + [Patch(color="#bdbdbd", label="outside annotated field")], loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "Fig6_qualitative")


def figS4(val, preds):
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
    ax[1].imshow(g[sl], cmap="gray")
    ax[1].set_title("Phase-gradient magnitude")
    ax[2].imshow(g[sl], cmap="gray")
    for a, rk in [("edge_sam", 32), ("mobile_sam", 8)]:
        for cont in find_contours((preds[(a, rk)][stem][sl] > 0).astype(float), 0.5):
            ax[2].plot(cont[:, 1], cont[:, 0], color=ARCH_COL[a], lw=0.8)
    ax[2].set_title("Predicted contours")
    ax[2].legend(handles=[Patch(color=ARCH_COL[a], label=label(a, rk)) for a, rk in [("edge_sam", 32), ("mobile_sam", 8)]],
                 fontsize=6, loc="lower right", frameon=True)
    for a_ in ax:
        clean(a_)
    fig.tight_layout()
    save(fig, "FigS4_gradient_alignment")
    return stem


def figS5(val, preds):
    days = [0, 8, 19, 27, 47]
    fig, axes = plt.subplots(2, len(days), figsize=(7.0, 3.0))
    for j, d in enumerate(days):
        stem = sorted(s for s in val if val[s]["day"] == d)[0]
        ph = val[stem]["phase"]
        lo, hi = np.percentile(ph[val[stem]["valid"]], [1, 99.5])
        for i in range(2):
            axes[i, j].imshow(ph, cmap="gray", vmin=lo, vmax=hi)
            clean(axes[i, j])
        for cont in find_contours((preds[("edge_sam", 32)][stem] > 0).astype(float), 0.5):
            axes[1, j].plot(cont[:, 1], cont[:, 0], color=ARCH_COL["edge_sam"], lw=0.5)
        axes[0, j].set_title(f"Day {d}")
    axes[0, 0].set_ylabel("Phase")
    axes[1, 0].set_ylabel("EdgeSAM (r=32)")
    fig.tight_layout()
    save(fig, "FigS5_storage_images")


def fate(val, pred):
    """Per day: kept, touching (admissible size), small (< 15 um2), large (> 200 um2) predicted components; annotated kept."""
    out = {}
    for s, v in val.items():
        d = out.setdefault(v["day"], {"kept": 0, "touching": 0, "small": 0, "large": 0, "ann_kept": 0,
                                       "pred_fg_px": 0, "ann_fg_px": 0})
        for name, cmap in (("pred", pred[s]), ("ann", v["gt"])):
            lab, n = components(cmap, v["valid"])
            if n == 0:
                continue
            area = ndi.sum(np.ones_like(lab), lab, np.arange(1, n + 1)) * PX_UM2
            cut = touches_unannotated(lab, n, v["valid"])[1:]
            ok = (area >= LO) & (area <= HI)
            d[f"{name}_fg_px"] += int(((cmap > 0) & v["valid"]).sum())
            if name == "pred":
                d["kept"] += int((ok & ~cut).sum())
                d["touching"] += int((ok & cut).sum())
                d["small"] += int((area < LO).sum())
                d["large"] += int((area > HI).sum())
            else:
                d["ann_kept"] += int((ok & ~cut).sum())
    return out


def figS6(fates):
    days = sorted(fates[C.SEEDS[0]])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.25]})
    for s, mk in zip(C.SEEDS, ["o", "^", "s"]):
        r = [fates[s][d]["kept"] / fates[s][d]["ann_kept"] for d in days]
        ax[0].plot(days, r, mk, color=ARCH_COL["edge_sam"], mec=INK, mew=0.3, ms=4, label=f"seed {s}")
    m = [np.mean([fates[s][d]["kept"] / fates[s][d]["ann_kept"] for s in C.SEEDS]) for d in days]
    ax[0].plot(days, m, color=ARCH_COL["edge_sam"], lw=1.1)
    ax[0].axhline(0.6, color="#6b6b66", lw=0.6, ls=":")
    ax[0].set_ylim(0, 1.1)
    ax[0].set_xlabel("Storage day")
    ax[0].set_ylabel("Kept components / annotated cells")
    ax[0].set_title("(a)", loc="left")
    ax[0].legend(frameon=False, fontsize=6.5)
    x = np.arange(len(days))
    bottom = np.zeros(len(days))
    for key, col, lab in [("kept", ARCH_COL["edge_sam"], "kept (15-200 µm², inside field)"),
                          ("touching", "#9cc3ee", "15-200 µm², touches unannotated region"),
                          ("small", "#bdbdb8", "< 15 µm²"), ("large", "#4d4d4d", "> 200 µm²")]:
        v = np.array([np.mean([fates[s][d][key] for s in C.SEEDS]) for d in days])
        ax[1].bar(x, v, 0.75, bottom=bottom, color=col, edgecolor="white", linewidth=0.4, label=lab)
        bottom += v
    ann = [fates[C.SEEDS[0]][d]["ann_kept"] for d in days]
    ax[1].scatter(x, ann, marker="_", s=120, color=INK, linewidths=1.6, zorder=5, label="annotated cells (same filters)")
    ax[1].set_ylim(0, 1.08 * max(bottom.max(), max(ann)))      # the whole stack of every day stays visible
    ax[1].set_xticks(x, [str(d) for d in days])
    ax[1].set_xlabel("Storage day")
    ax[1].set_ylabel("Predicted components (mean of 3 seeds)")
    ax[1].set_title("(b)", loc="left")
    ax[1].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, "FigS6_component_fate")


def bridged_and_confusion(val, pred):
    """Annotated kept components bridged to another annotated component by one predicted component (>= 30% of both);
    pixel confusion: share of annotated class-c pixels predicted as class k (inside the annotated field)."""
    n_ann = n_bridged = 0
    conf = np.zeros((5, 5), np.int64)
    for s, v in val.items():
        gl, gn = components(v["gt"], v["valid"])
        pl, pn = components(pred[s], v["valid"])
        if gn:
            area = ndi.sum(np.ones_like(gl), gl, np.arange(1, gn + 1)) * PX_UM2
            keep = (area >= LO) & (area <= HI) & ~touches_unannotated(gl, gn, v["valid"])[1:]
            for gi in np.nonzero(keep)[0] + 1:
                m = gl == gi
                n_ann += 1
                for pi in np.unique(pl[m]):
                    if pi == 0 or (pl[m] == pi).mean() < 0.3:
                        continue
                    others = [g for g in np.unique(gl[pl == pi]) if g not in (0, gi)]
                    if any((pl[gl == g] == pi).mean() >= 0.3 for g in others):
                        n_bridged += 1
                        break
        vv = v["valid"]
        conf += np.bincount(v["gt"][vv] * 5 + pred[s][vv], minlength=25).reshape(5, 5)
    share = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    return {"annotated_components": n_ann, "bridged_pct": 100 * n_bridged / max(n_ann, 1),
            "confusion_pct_rows_annotated": {C.CLASSES[c - 1]: {(["background"] + list(C.CLASSES))[k]: round(100 * share[c, k], 1)
                                                               for k in range(5)} for c in range(1, 5)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", default=None)
    ap.add_argument("--seed", type=int, default=42, help="training seed shown in Fig. 6 (same seed for every model)")
    ap.add_argument("--only-figS6", action="store_true", help="redraw only Fig. S-6 (fast; skips the other figures and numbers)")
    a = ap.parse_args()
    cfg = load_config()
    if a.processed_root:
        cfg["paths_abs"]["processed_root"] = Path(a.processed_root).resolve()
    if not (Path(cfg["paths_abs"]["processed_root"]) / "manifest.csv").exists():
        print("make_data_figures.py: data/processed not found; run on the machine with the processed phase maps.", file=sys.stderr)
        return 1
    val = load_validation(cfg)
    preds = {(ar, rk): load_predictions(C.run_dir(ar, rk, "full", a.seed), "last") for ar, rk in SHOW}
    if a.only_figS6:
        figS6({s: fate(val, load_predictions(C.run_dir("edge_sam", 32, "full", s), "last")) for s in C.SEEDS})
        return 0
    fig6(val, preds, a.seed)
    figS4(val, preds)
    figS5(val, preds)
    fates = {s: fate(val, load_predictions(C.run_dir("edge_sam", 32, "full", s), "last")) for s in C.SEEDS}
    figS6(fates)
    numbers = {
        "unannotated_fraction_mean_pct": 100 * float(np.mean([1 - v["valid"].mean() for v in val.values()])),
        "day15_foreground_fraction": [float(((v["gt"] > 0) & v["valid"]).sum() / v["valid"].sum())
                                      for v in val.values() if v["day"] == 15],
        "day15_fate_edge_sam_r32_mean_of_seeds": {k: float(np.mean([fates[s][15][k] for s in C.SEEDS]))
                                                 for k in fates[C.SEEDS[0]][15]},
        "note": "validation images only; the 14% quoted for the whole dataset needs the training masks as well",
    }
    numbers["day15_pred_over_ann_foreground_area"] = (numbers["day15_fate_edge_sam_r32_mean_of_seeds"]["pred_fg_px"]
                                                       / numbers["day15_fate_edge_sam_r32_mean_of_seeds"]["ann_fg_px"])
    numbers["bridging_and_confusion"] = {}
    for ar, rk in SHOW:
        for s in C.SEEDS:
            p = load_predictions(C.run_dir(ar, rk, "full", s), "last")
            numbers["bridging_and_confusion"][f"{ar}_r{rk}_s{s}"] = bridged_and_confusion(val, p)
    out = C.ART / "audit" / "server_numbers.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(numbers, indent=1))
    print(f"[data figures] Fig6, FigS4, FigS5, FigS6 and {out} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
