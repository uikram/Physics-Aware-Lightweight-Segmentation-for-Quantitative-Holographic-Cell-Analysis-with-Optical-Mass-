"""Build the processed (float-frame) dataset from the raw float32 phase maps and the annotation masks."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tifffile

from qpi.data.registration import estimate_transform, warp_mask_to_float
from qpi.utils.io import find_image, write_json

DATE_STEM = re.compile(r"^(\d{8})_\d+$")


def storage_day(stem: str, day0: str) -> int:
    """Storage day of a date-named stem (YYYYMMDD_n); -1 for anonymous stems (X_train_#####)."""
    m = DATE_STEM.match(stem)
    if not m:
        return -1
    d = datetime.strptime(m.group(1), "%Y%m%d") - datetime.strptime(day0, "%Y-%m-%d")
    return d.days


@dataclass
class Source:
    split: str            # train | val
    stem: str
    phase: Optional[Path]  # float32 map (None if unavailable)
    grey8: Optional[Path]
    mask8: Optional[Path]


def discover_sources(raw_root: Path) -> List[Source]:
    """List every base image (rotation files excluded) of the original 8-bit dataset and locate its float map."""
    ds = raw_root / "dataset"
    float_dirs = [raw_root / "raw_data", raw_root / "X_train_raw"]
    out: List[Source] = []
    for split, sub in (("train", "train"), ("val", "val")):
        xdir, ydir = ds / f"X_{sub}", ds / f"Y_{sub}"
        if not xdir.is_dir() or not ydir.is_dir():
            raise FileNotFoundError(f"missing {xdir} or {ydir}")
        stems = sorted({p.stem for p in xdir.iterdir()
                        if p.suffix.lower() in (".tif", ".tiff") and not p.stem.startswith("rotation")})
        for stem in stems:
            phase = next((q for d in float_dirs if (q := find_image(d, stem)) is not None), None)
            out.append(Source(split, stem, phase, find_image(xdir, stem), find_image(ydir, stem)))
    return out


def registration_gate(reg, reg_cfg: dict) -> Tuple[bool, str]:
    """Registration quality gate -> (accepted, basis).

    primary    ECC >= min_ecc_correlation and Pearson >= min_pearson_r.
    borderline ECC in [min_ecc_borderline, min_ecc_correlation) is accepted only if the deficit is explained by a
               non-linear (but monotone) grey-level export rather than by misalignment, i.e. all of:
               the correlation maximum within +-3 px sits at zero shift (<= max_peak_shift_px), the rank correlation
               is >= min_spearman_r and exceeds the linear Pearson r by >= min_rank_gain. A misregistered pair has
               Spearman ~ Pearson (both low) and fails. Pearson >= min_pearson_r is required in every case.
    """
    primary = reg.ecc_correlation >= reg_cfg["min_ecc_correlation"]
    geom_ok = bool(reg.peak_shift_px <= reg_cfg["max_peak_shift_px"]
                   and reg.spearman_r >= reg_cfg["min_spearman_r"]
                   and (reg.spearman_r - reg.pearson_r) >= reg_cfg["min_rank_gain"])
    borderline = (not primary) and reg.ecc_correlation >= reg_cfg["min_ecc_borderline"] and geom_ok
    ok = bool((primary or borderline) and reg.pearson_r >= reg_cfg["min_pearson_r"])
    basis = "primary" if primary else ("borderline_nonlinear_export_verified" if borderline else "rejected")
    return ok, (basis if ok else "rejected")


def _process_one(args: Tuple[Source, Path, dict, dict]) -> dict:
    src, out_root, reg_cfg, data_cfg = args
    ignore_index = data_cfg["ignore_index"]
    row: dict = {"split": src.split, "stem": src.stem}
    phase = tifffile.imread(src.phase).astype(np.float32)
    grey8 = tifffile.imread(src.grey8)
    mask8 = tifffile.imread(src.mask8)
    if phase.ndim != 2 or grey8.ndim != 2 or mask8.ndim != 2:
        raise ValueError(f"{src.stem}: expected 2-D arrays, got {phase.shape}, {grey8.shape}, {mask8.shape}")
    if mask8.shape != grey8.shape:
        raise ValueError(f"{src.stem}: mask {mask8.shape} and 8-bit image {grey8.shape} differ")
    if not np.isfinite(phase).all():
        raise ValueError(f"{src.stem}: float phase map contains NaN/Inf")
    labels = np.unique(mask8)
    if labels.max() > 4:
        raise ValueError(f"{src.stem}: unexpected labels {labels.tolist()}")
    reg = estimate_transform(phase, grey8, reg_cfg["pyramid_factor"], retry_below=reg_cfg["min_ecc_correlation"])
    mask = warp_mask_to_float(mask8, reg.warp, phase.shape, ignore_index)
    valid = mask != ignore_index
    # Phase reference. The provided float maps use inconsistent offsets (date-named maps lie in [0, 2*pi] after the
    # 24 Oct 2026 correction by the data provider (they were in [0, pi] before), the X_train_*/X_val_* maps are zero-mean). With phase_reference == "background" every map is referenced to its own
    # background (median phase of annotated pixels farther than `background_exclusion_px` from any cell) so that the
    # physics terms and the IPE see the same convention in training and evaluation. Units stay radians.
    offset = 0.0
    if data_cfg["phase_reference"] == "background":
        from scipy import ndimage as ndi
        fg = (mask > 0) & valid
        bgm = valid & ~ndi.binary_dilation(fg, iterations=int(data_cfg["background_exclusion_px"]))
        if bgm.sum() < 1000:
            raise ValueError(f"{src.stem}: too few background pixels ({int(bgm.sum())}) to reference the phase")
        offset = float(np.median(phase[bgm]))
    elif data_cfg["phase_reference"] != "none":
        raise ValueError(f"unknown phase_reference '{data_cfg['phase_reference']}'")
    raw_min, raw_max, raw_mean = float(phase.min()), float(phase.max()), float(phase.mean())
    phase = (phase - np.float32(offset)).astype(np.float32)
    ok, basis = registration_gate(reg, reg_cfg)
    row.update(
        ecc_correlation=round(reg.ecc_correlation, 5), pearson_r=round(reg.pearson_r, 5),
        scale_x=round(reg.scale_x, 5), scale_y=round(reg.scale_y, 5), rotation_deg=round(reg.rotation_deg, 4),
        shear=round(reg.shear, 5), grey_per_rad=round(reg.grey_per_rad, 3), grey_offset=round(reg.grey_offset, 3),
        spearman_r=round(reg.spearman_r, 5), peak_shift_px=round(reg.peak_shift_px, 3),
        registration_search=reg.search, registration_basis=basis,
        warp=json.dumps(np.round(reg.warp, 6).tolist()), registration_ok=bool(ok),
        valid_fraction=round(float(valid.mean()), 5),
        phase_offset_rad=round(offset, 6), phase_min_asprovided=raw_min, phase_max_asprovided=raw_max,
        phase_mean_asprovided=raw_mean, phase_min=float(phase.min()), phase_max=float(phase.max()),
        phase_mean=float(phase.mean()),
        foreground_fraction_valid=round(float(((mask > 0) & valid).sum() / max(valid.sum(), 1)), 5),
    )
    for c in range(5):
        row[f"px_class{c}"] = int(((mask == c) & valid).sum())
    if ok:
        (out_root / src.split / "phase").mkdir(parents=True, exist_ok=True)
        (out_root / src.split / "mask").mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(out_root / src.split / "phase" / f"{src.stem}.tif", phase)
        tifffile.imwrite(out_root / src.split / "mask" / f"{src.stem}.tif", mask.astype(np.uint8))
        row["phase_sha256"] = hashlib.sha256(phase.tobytes()).hexdigest()
    return row


def prepare_dataset(cfg: dict, raw_root: Path, out_root: Path, workers: int = 4, limit: Optional[int] = None,
                    allow_failures: bool = False) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(raw_root)
    missing = [s for s in sources if s.phase is None]
    incomplete = [s for s in sources if s.phase is not None and (s.grey8 is None or s.mask8 is None)]
    if incomplete:
        raise FileNotFoundError(f"8-bit image or mask missing for: {[s.stem for s in incomplete]}")
    todo = [s for s in sources if s.phase is not None]
    if limit:  # evenly spaced subset of at most ``limit`` images per split (deterministic; smoke tests)
        sub = []
        for sp in ("train", "val"):
            sel = sorted((s for s in todo if s.split == sp), key=lambda s: s.stem)
            if len(sel) > limit:
                sel = [sel[int(i)] for i in np.linspace(0, len(sel) - 1, limit).round()]
            sub += sel
        todo = sub
    jobs = [(s, out_root, cfg["registration"], cfg["data"]) for s in todo]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_process_one, jobs, chunksize=1))
    else:
        rows = [_process_one(j) for j in jobs]
    day0 = cfg["data"]["day0_date"]
    for r in rows:
        r["storage_day"] = storage_day(r["stem"], day0)
        r["group"] = ("date" if DATE_STEM.match(r["stem"]) else r["stem"].rsplit("_", 1)[0])
    rows.sort(key=lambda r: (r["split"], r["stem"]))
    with open(out_root / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    failed = [r for r in rows if not r["registration_ok"]]
    with open(out_root / "missing_float_data.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "stem"])
        for s in missing:
            w.writerow([s.split, s.stem])
    report = {
        "raw_root": str(raw_root),
        "n_base_images_in_8bit_dataset": {sp: sum(1 for s in sources if s.split == sp) for sp in ("train", "val")},
        "n_with_float_data": {sp: sum(1 for s in sources if s.split == sp and s.phase is not None) for sp in ("train", "val")},
        "n_missing_float_data": {sp: sum(1 for s in missing if s.split == sp) for sp in ("train", "val")},
        "n_processed_ok": {sp: sum(1 for r in rows if r["split"] == sp and r["registration_ok"]) for sp in ("train", "val")},
        "n_registration_failed": [{"split": r["split"], "stem": r["stem"], "ecc": r["ecc_correlation"],
                                   "pearson_r": r["pearson_r"]} for r in failed],
        "registration_thresholds": cfg["registration"],
        "n_registration_extended_search": [r["stem"] for r in rows if r["registration_search"] == "extended"],
        "n_registration_borderline_accepted": [r["stem"] for r in rows if r["registration_basis"] == "borderline_nonlinear_export_verified"],
        "valid_fraction": {"min": float(min(r["valid_fraction"] for r in rows)),
                           "mean": float(np.mean([r["valid_fraction"] for r in rows])),
                           "max": float(max(r["valid_fraction"] for r in rows))},
        "phase_reference": cfg["data"]["phase_reference"],
        "phase_range_processed": {"min": float(min(r["phase_min"] for r in rows)), "max": float(max(r["phase_max"] for r in rows))},
        "phase_range_asprovided": {"min": float(min(r["phase_min_asprovided"] for r in rows)),
                                   "max": float(max(r["phase_max_asprovided"] for r in rows))},
        "phase_offset_rad_by_group": {g: {"min": float(min(r["phase_offset_rad"] for r in rows if r["group"] == g)),
                                          "max": float(max(r["phase_offset_rad"] for r in rows if r["group"] == g)),
                                          "n": sum(1 for r in rows if r["group"] == g)}
                                      for g in sorted({r["group"] for r in rows})},
        "ignore_index": cfg["data"]["ignore_index"],
    }
    write_json(out_root / "dataset_report.json", report)
    if failed and not allow_failures:
        raise RuntimeError(f"{len(failed)} image(s) failed the registration quality gate: "
                           f"{[r['stem'] for r in failed]} (see {out_root/'manifest.csv'})")
    return report
