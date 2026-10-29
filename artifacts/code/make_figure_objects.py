#!/usr/bin/env python
"""Image objects for the editable PowerPoint figures (artifacts/figures/pptx/Figures_revised.pptx).

Run on the machine that has data/processed, from the repository root:

    python artifacts/code/make_figure_objects.py [--processed-root data/processed] [--seed 42]

Every object is a bare image (no axes, titles, scale bars or text): labels, legends, colour-bar numbers and 20 µm scale bars
are editable shapes on the slides. In PowerPoint, right-click the placeholder with the same file name > Change Picture.

    artifacts/figures/objects/
      Fig1/   p1_phase.png  p1_phase_colorbar.png  p2_norm.png  p2_norm_colorbar.png  p3_mask.png
              p4_volume.png  p4_volume_colorbar.png
      FigS4/  p1_phase.png  p2_gradient.png  p3_contours.png
      Fig6/   r1_discocyte_c1_phase.png ... r4_stomatocyte_c6_edgesam_r32.png   (4 rows x 6 columns)
      FigS5/  day00_phase.png  day00_contours.png ... day47_contours.png
      objects.json   colour-bar limits, chosen images/windows, storage days, scale-bar lengths

Selection rules (deterministic, annotation only):
  Fig. 1 and Fig. S-4: the validation image with the most annotated foreground pixels. Fig. 1 shows the 256 x 256 px window
                       with the most foreground in it (32 px grid); Fig. S-4 shows the full 768 x 768 px field.
  Fig. 6: per class, the image with the most pixels of that class and, in it, the 256 x 256 px window with the most pixels
          of that class (32 px grid); all predictions from training seed --seed.
  Fig. S-5: the first validation image (alphabetical) of storage days 0, 8, 19, 27 and 47; contours of EdgeSAM (r=32).
Pixel pitch 0.1441 µm: a full field is 110.67 µm wide (20 µm = 18.07% of the panel width), a 256 px window 36.89 µm
(20 µm = 54.21%).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from qpi.analysis.common import load_predictions, load_validation  # noqa: E402
from qpi.config import load_config  # noqa: E402

OUT = C.ART / "figures" / "objects"
PX_UM = 0.1441
CLASS_RGB = {1: (0, 114, 178), 2: (0, 158, 115), 3: (213, 94, 0), 4: (204, 121, 167)}   # same colours as Figs. 6, 7
OUTSIDE_RGB = (189, 189, 189)
EDGE_RGB = (230, 40, 40)
SHOW = [("mobilenet_unet", 8, "mobilenet_unet_r8"), ("mobile_sam", 8, "mobilesam_r8"), ("edge_sam", 8, "edgesam_r8"),
        ("edge_sam", 32, "edgesam_r32")]


# ------------------------------------------------------------------------------------------------ helpers
def best_window(mask, valid, size=256, stride=32):
    best, pos = -1, (0, 0)
    m = mask & valid
    for y in range(0, m.shape[0] - size + 1, stride):
        for x in range(0, m.shape[1] - size + 1, stride):
            n = int(m[y:y + size, x:x + size].sum())
            if n > best:
                best, pos = n, (y, x)
    return pos


def to_gray(a, lo, hi):
    g = np.clip((a - lo) / (hi - lo), 0, 1)
    return np.repeat((255 * g)[..., None], 3, axis=2)


def to_cmap(a, lo, hi, name):
    g = np.clip((a - lo) / (hi - lo), 0, 1)
    return 255 * colormaps[name](g)[..., :3]


def upscale(rgb, k):
    return np.kron(rgb, np.ones((k, k, 1))) if k > 1 else rgb


def save(rgb, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path, dpi=(300, 300))


def overlay(gray_rgb, labels, valid, alpha=0.5):
    out = gray_rgb.copy()
    for c, col in CLASS_RGB.items():
        m = (labels == c) & valid
        out[m] = (1 - alpha) * out[m] + alpha * np.array(col)
    out[~valid] = 0.5 * out[~valid] + 0.5 * np.array(OUTSIDE_RGB)
    return out


def contours(rgb, fg, k, colour, width=3):
    """Draw the outline of the foreground on an image already upscaled by k (outline computed at the upscaled size)."""
    big = np.kron(fg.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    edge = big & ~ndi.binary_erosion(big, iterations=width)
    out = rgb.copy()
    out[edge] = colour
    return out


def colorbar(path: Path, cmap: str, h=600, w=40):
    g = np.linspace(1, 0, h)[:, None].repeat(w, axis=1)
    rgb = 255 * colormaps[cmap](g)[..., :3]
    rgb[[0, -1], :] = 0
    rgb[:, [0, -1]] = 0
    save(rgb, path)


def background(phase, gt_fg, valid, excl=5):
    bg = valid & ~ndi.binary_dilation(gt_fg, iterations=excl)
    return float(np.median(phase[bg])) if bg.any() else 0.0


# ------------------------------------------------------------------------------------------------ figures
def fig1(val, pred_e32, meta):
    stems = sorted(val)
    stem = max(stems, key=lambda s: (int(((val[s]["gt"] > 0) & val[s]["valid"]).sum()), -stems.index(s)))
    v = val[stem]
    y, x = best_window(v["gt"] > 0, v["valid"])
    sl = (slice(y, y + 256), slice(x, x + 256))
    ph = v["phase"]
    lo, hi = np.percentile(ph[sl], [1, 99.5])
    k = 4
    save(upscale(to_gray(ph[sl], lo, hi), k), OUT / "Fig1" / "p1_phase.png")
    colorbar(OUT / "Fig1" / "p1_phase_colorbar.png", "gray")
    z = (ph - ph.mean()) / ph.std()                       # the network input is the per-image z-score of the map
    save(upscale(to_gray(z[sl], -3, 3), k), OUT / "Fig1" / "p2_norm.png")
    colorbar(OUT / "Fig1" / "p2_norm_colorbar.png", "gray")
    lab = pred_e32[stem][sl]
    base = upscale(overlay(to_gray(ph[sl], lo, hi), lab, v["valid"][sl], alpha=0.55), k)
    save(contours(base, lab > 0, k, (40, 40, 40)), OUT / "Fig1" / "p3_mask.png")
    bgv = background(ph, v["gt"] > 0, v["valid"])
    rel = ph[sl] - bgv
    fg = lab > 0
    vlo, vhi = 0.0, float(np.percentile(rel[fg], 99)) if fg.any() else 1.0
    vol = to_cmap(rel, vlo, vhi, "viridis")
    vol[~fg] = 255
    save(upscale(vol, k), OUT / "Fig1" / "p4_volume.png")
    colorbar(OUT / "Fig1" / "p4_volume_colorbar.png", "viridis")
    meta["Fig1"] = {"image": stem, "storage_day": v["day"], "window_yx": [y, x], "window_px": 256,
                    "p1_phase_colorbar_rad": [round(float(lo), 2), round(float(hi), 2)],
                    "p2_norm_colorbar": [-3, 3], "p4_volume_colorbar_rad": [vlo, round(vhi, 2)],
                    "prediction": "EdgeSAM (r=32)", "scale_bar_20um_fraction_of_width": round(20 / (256 * PX_UM), 4)}


def figS4(val, pred_e32, meta):
    stems = sorted(val)
    stem = max(stems, key=lambda s: (int(((val[s]["gt"] > 0) & val[s]["valid"]).sum()), -stems.index(s)))
    v = val[stem]
    ph = v["phase"]
    k = 2
    lo, hi = np.percentile(ph[v["valid"]], [1, 99.5])
    save(upscale(to_cmap(ph, lo, hi, "inferno"), k), OUT / "FigS4" / "p1_phase.png")
    gy, gx = np.gradient(ph)
    g = np.hypot(gx, gy)
    ghi = float(np.percentile(g, 99.5))
    grad = to_cmap(g, 0, ghi, "viridis")
    save(upscale(grad, k), OUT / "FigS4" / "p2_gradient.png")
    gray = upscale(to_gray(g, 0, ghi), k)
    save(contours(gray, pred_e32[stem] > 0, k, EDGE_RGB), OUT / "FigS4" / "p3_contours.png")
    meta["FigS4"] = {"image": stem, "storage_day": v["day"], "field_px": 768, "upsampling": k,
                     "phase_range_rad": [round(float(lo), 2), round(float(hi), 2)], "contours": "EdgeSAM (r=32)",
                     "scale_bar_20um_fraction_of_width": round(20 / (768 * PX_UM), 4)}


def fig6(val, preds, seed, meta):
    stems = sorted(val)
    rows = []
    k = 4
    for r, cl in enumerate(C.CLASSES, start=1):
        best = max(stems, key=lambda s: (int(((val[s]["gt"] == r) & val[s]["valid"]).sum()), -stems.index(s)))
        v = val[best]
        y, x = best_window(v["gt"] == r, v["valid"])
        sl = (slice(y, y + 256), slice(x, x + 256))
        lo, hi = np.percentile(v["phase"][sl], [1, 99])
        gray = to_gray(v["phase"][sl], lo, hi)
        name = f"r{r}_{cl}"
        save(upscale(gray, k), OUT / "Fig6" / f"{name}_c1_phase.png")
        save(upscale(overlay(gray, v["gt"][sl], v["valid"][sl]), k), OUT / "Fig6" / f"{name}_c2_annotation.png")
        for c, (a, rk, tag) in enumerate(SHOW, start=3):
            save(upscale(overlay(gray, preds[(a, rk)][best][sl], v["valid"][sl]), k), OUT / "Fig6" / f"{name}_c{c}_{tag}.png")
        rows.append({"row": r, "class": cl, "image": best, "storage_day": v["day"], "window_yx": [y, x]})
    meta["Fig6"] = {"rows": rows, "training_seed": seed, "window_px": 256, "upsampling": k,
                    "scale_bar_20um_fraction_of_width": round(20 / (256 * PX_UM), 4)}


def figS5(val, pred_e32, meta):
    days = [0, 8, 19, 27, 47]
    k = 2
    out = []
    for d in days:
        stem = sorted(s for s in val if val[s]["day"] == d)[0]
        v = val[stem]
        lo, hi = np.percentile(v["phase"][v["valid"]], [1, 99.5])
        img = upscale(to_cmap(v["phase"], lo, hi, "inferno"), k)
        save(img, OUT / "FigS5" / f"day{d:02d}_phase.png")
        save(contours(img, pred_e32[stem] > 0, k, (80, 220, 255)), OUT / "FigS5" / f"day{d:02d}_contours.png")
        out.append({"storage_day": d, "image": stem})
    meta["FigS5"] = {"panels": out, "contours": "EdgeSAM (r=32)", "field_px": 768, "upsampling": k,
                     "scale_bar_20um_fraction_of_width": round(20 / (768 * PX_UM), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", default=None)
    ap.add_argument("--seed", type=int, default=42, help="training seed of every prediction shown")
    a = ap.parse_args()
    cfg = load_config()
    if a.processed_root:
        cfg["paths_abs"]["processed_root"] = Path(a.processed_root).resolve()
    if not (Path(cfg["paths_abs"]["processed_root"]) / "manifest.csv").exists():
        print("make_figure_objects.py: data/processed not found; run on the machine with the processed phase maps.",
              file=sys.stderr)
        return 1
    val = load_validation(cfg)
    preds = {(ar, rk): load_predictions(C.run_dir(ar, rk, "full", a.seed), "last") for ar, rk, _ in SHOW}
    meta = {"pixel_um": PX_UM, "training_seed": a.seed}
    fig1(val, preds[("edge_sam", 32)], meta)
    figS4(val, preds[("edge_sam", 32)], meta)
    fig6(val, preds, a.seed, meta)
    figS5(val, preds[("edge_sam", 32)], meta)
    (OUT / "objects.json").write_text(json.dumps(meta, indent=1))
    n = len(list(OUT.rglob("*.png")))
    print(f"[figure objects] {n} images and objects.json written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
