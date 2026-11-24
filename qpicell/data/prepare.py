"""Build data/prepared from data/raw. No annotation is used for the phase values.

Inputs (data/raw):
  dataset/X_train, X_val     8-bit exports (geometry only; base images, the 'rotation*' copies are ignored)
  dataset/Y_train, Y_val     annotation masks in the 8-bit frame (0 background, 1-4 cell classes)
  raw_data/, X_train_raw/    32-bit float phase maps in radians, found by the same file stem

Per image (outputs in data/prepared/<split>/):
  phase/<stem>.tif   float32 radians: provided float map minus its annotation-free background (order 2)
  mask/<stem>.tif    uint8 labels on the float grid (nearest-neighbour registration), 255 = not annotated
  cells/<stem>.tif   uint16 cell instances (watershed split of the annotated foreground)
  wrapped/<stem>.tif uint8, 1 = pixel near phase wrapping in the provided map
manifest.csv: one row per image (registration result, background coefficients, counts, storage day, checksums);
missing_phase.csv: base images without a float map (not used); report.json: totals.
The float maps stay 32-bit radians: they are never converted to 8 bit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tifffile
from scipy import ndimage as ndi

from qpicell.data.background import estimate_background
from qpicell.data.cells import split_cells
from qpicell.data.registration import register, warp_mask
from qpicell.physics import assert_mask, assert_phase
from qpicell.data.wrapping import wrapped_pixels
from qpicell.io import find_image, list_stems, write_json

DATE_STEM = re.compile(r"^(\d{8})_\d+$")


def storage_day(stem: str, day0: str) -> int:
    m = DATE_STEM.match(stem)
    if not m:
        return -1
    return (datetime.strptime(m.group(1), "%Y%m%d") - datetime.strptime(day0, "%Y-%m-%d")).days


def discover(cfg: Dict) -> List[Dict]:
    ds = Path(cfg["paths_abs"]["raw_dataset"]); out = []
    for split in ("train", "val"):
        xdir, ydir = ds / f"X_{split}", ds / f"Y_{split}"
        for stem in list_stems(xdir):
            phase = next((p for d in cfg["paths_abs"]["raw_phase_dirs"] if (p := find_image(d, stem)) is not None), None)
            out.append(dict(split=split, stem=stem, phase=phase, grey8=find_image(xdir, stem), mask8=find_image(ydir, stem)))
    return out


def _one(job) -> Dict:
    src, cfg, out_root = job
    stem, split = src["stem"], src["split"]
    raw = tifffile.imread(src["phase"])
    if raw.dtype != np.float32 or raw.ndim != 2:
        raise ValueError(f"{stem}: expected a 2-D float32 phase map, got {raw.dtype} {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{stem}: phase map contains NaN / Inf")
    assert_phase(raw, f"raw {stem}")                                     # float32 radians, not 8-bit
    grey8, mask8 = tifffile.imread(src["grey8"]), tifffile.imread(src["mask8"])
    if mask8.shape != grey8.shape:
        raise ValueError(f"{stem}: 8-bit image {grey8.shape} and mask {mask8.shape} differ")
    if mask8.max() > cfg["data"]["num_classes"] - 1:
        raise ValueError(f"{stem}: unexpected mask labels {np.unique(mask8).tolist()}")
    ign = cfg["data"]["ignore_index"]
    reg = register(raw.astype(np.float64), grey8, cfg["registration"], cfg["registration"]["pyramid_factor"])
    row = dict(split=split, stem=stem, storage_day=storage_day(stem, cfg["data"]["day0"]),
               group="date" if DATE_STEM.match(stem) else stem.rsplit("_", 1)[0],
               phase_file=str(Path(src["phase"]).relative_to(cfg["project_root"])),
               registration_ok=reg.accepted, registration_basis=reg.basis, registration_search=reg.search,
               ecc=round(reg.ecc, 5), pearson=round(reg.pearson, 5), spearman=round(reg.spearman, 5),
               peak_shift_px=reg.peak_shift_px, scale_x=round(reg.scale_x, 5), scale_y=round(reg.scale_y, 5),
               rotation_deg=round(reg.rotation_deg, 4), shear=round(reg.shear, 5), warp=json.dumps(np.round(reg.warp, 6).tolist()),
               raw_min=float(raw.min()), raw_max=float(raw.max()), raw_mean=float(raw.mean()))
    if not reg.accepted:
        return row
    mask = warp_mask(mask8, reg.warp, raw.shape, ign)
    valid = mask != ign
    bcfg = cfg["background"]
    surf, info = estimate_background(raw.astype(np.float64), bcfg["order"], bcfg["k_sigma"], bcfg["grow_px"], bcfg["max_iter"])
    phase = (raw.astype(np.float64) - surf).astype(np.float32)          # radians, float32 - the ONLY background subtraction
    assert_phase(phase, f"prepared {stem}")
    assert_mask(mask.astype(np.uint8), cfg["data"]["num_classes"], ign, f"mask {stem}")
    fg = (mask > 0) & valid
    cells = split_cells(fg, cfg["cells"]["watershed_h"])
    if cells.max() > 65535:
        raise ValueError(f"{stem}: more than 65535 cells")
    wrap = wrapped_pixels(raw.astype(np.float64), cfg["cells"]["wrapping"])
    for sub, arr in (("phase", phase), ("mask", mask.astype(np.uint8)), ("cells", cells.astype(np.uint16)), ("wrapped", wrap.astype(np.uint8))):
        (out_root / split / sub).mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(out_root / split / sub / f"{stem}.tif", arr)
    row.update(valid_fraction=round(float(valid.mean()), 5),
               background_coefficients=json.dumps([round(float(c), 7) for c in info["coefficients"]]),
               background_iterations=info["iterations"], background_noise_sd_rad=round(info["noise_sd_rad"], 6),
               background_fraction=round(info["background_fraction"], 5),
               background_leak_pct=round(100 * float((info["background_mask"] & fg).sum() / max(fg.sum(), 1)), 4),
               phase_min=float(phase.min()), phase_max=float(phase.max()),
               n_components=int(ndi.label(fg)[1]), n_cells=int(cells.max()), wrapped_pixels=int(wrap.sum()),
               foreground_fraction=round(float(fg.sum() / max(valid.sum(), 1)), 5),
               **{f"px_class{c}": int(((mask == c) & valid).sum()) for c in range(cfg["data"]["num_classes"])},
               phase_sha256=hashlib.sha256(phase.tobytes()).hexdigest())
    return row


def prepare(cfg: Dict, workers: int = 4, limit: Optional[int] = None, out_root: Optional[Path] = None) -> Dict:
    out_root = Path(out_root or cfg["paths_abs"]["prepared"])
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"{out_root} exists and is not empty; delete it first (nothing is overwritten)")
    out_root.mkdir(parents=True, exist_ok=True)
    sources = discover(cfg)
    incomplete = [s["stem"] for s in sources if s["phase"] is not None and (s["grey8"] is None or s["mask8"] is None)]
    if incomplete:
        raise FileNotFoundError(f"8-bit image or mask missing for {incomplete}")
    missing = [s for s in sources if s["phase"] is None]
    todo = [s for s in sources if s["phase"] is not None]
    if limit:
        todo = [s for sp in ("train", "val") for s in [t for t in todo if t["split"] == sp][:limit]]
    print(f"prepare: {len(todo)} images with a float map "
          f"({sum(s['split'] == 'train' for s in todo)} train, {sum(s['split'] == 'val' for s in todo)} val), "
          f"{len(missing)} base images without one -> {out_root}", flush=True)
    jobs = [(s, cfg, out_root) for s in todo]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            rows = []
            for i, r in enumerate(ex.map(_one, jobs, chunksize=1), 1):
                rows.append(r)
                if i % 20 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} images", flush=True)
    else:
        rows = [_one(j) for j in jobs]
    rows.sort(key=lambda r: (r["split"], r["stem"]))
    keys = list(dict.fromkeys(k for r in rows for k in r))         # union of the columns, first-seen order
    with open(out_root / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(out_root / "missing_phase.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["split", "stem"]); [w.writerow([s["split"], s["stem"]]) for s in missing]
    ok = [r for r in rows if r["registration_ok"]]
    rep = {"n_images": len(rows), "n_accepted": {sp: sum(r["split"] == sp for r in ok) for sp in ("train", "val")},
           "rejected": [(r["split"], r["stem"], r["ecc"], r["pearson"]) for r in rows if not r["registration_ok"]],
           "borderline_accepted": [r["stem"] for r in ok if r["registration_basis"] != "primary"],
           "wide_search": [r["stem"] for r in rows if r["registration_search"] == "wide"],
           "missing_phase": {sp: sum(s["split"] == sp for s in missing) for sp in ("train", "val")},
           "cells": {sp: sum(r["n_cells"] for r in ok if r["split"] == sp) for sp in ("train", "val")},
           "components": {sp: sum(r["n_components"] for r in ok if r["split"] == sp) for sp in ("train", "val")},
           "valid_fraction": {"min": min(r["valid_fraction"] for r in ok), "mean": float(np.mean([r["valid_fraction"] for r in ok])),
                              "max": max(r["valid_fraction"] for r in ok)},
           "max_background_leak_pct": max(r["background_leak_pct"] for r in ok),
           "phase_dtype": "float32 radians (provided map minus annotation-free order-2 background)"}
    write_json(out_root / "report.json", rep)
    print(json.dumps(rep, indent=2), flush=True)
    if rep["rejected"]:
        raise RuntimeError(f"{len(rep['rejected'])} image(s) failed the registration gate: {rep['rejected']}")
    return rep
