"""Dataset verification: raw structure, rotation consistency, labels, leakage and processed-data integrity."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import tifffile

from qpi.utils.io import find_image, write_json


def _read_manifest(processed: Path) -> List[Dict]:
    with open(processed / "manifest.csv") as fh:
        return list(csv.DictReader(fh))


def verify_dataset(cfg: Dict, raw_root: Path, processed: Path) -> Dict:
    checks: Dict[str, Dict] = {}

    def record(name: str, ok: bool, **info):
        checks[name] = {"ok": bool(ok), **info}

    ds = raw_root / "dataset"
    ign = cfg["data"]["ignore_index"]
    names = {sp: {k: sorted(p.name for p in (ds / f"{k}_{sp}").iterdir() if p.suffix.lower() in (".tif", ".tiff"))
                  for k in ("X", "Y")} for sp in ("train", "val")}

    # 1. image/mask pairing and counts of the original 8-bit dataset
    for sp in ("train", "val"):
        x, y = {Path(n).stem for n in names[sp]["X"]}, {Path(n).stem for n in names[sp]["Y"]}
        record(f"pairing_{sp}", x == y, n_images=len(x), n_masks=len(y), only_images=sorted(x - y)[:5],
               only_masks=sorted(y - x)[:5])
    base_train = sorted(Path(n).stem for n in names["train"]["X"] if not n.startswith("rotation"))
    rot_train = sorted(Path(n).stem for n in names["train"]["X"] if n.startswith("rotation"))
    record("train_composition", len(rot_train) == 3 * len(base_train), n_base=len(base_train), n_rotation_files=len(rot_train),
           expected_views=4 * len(base_train))

    # 2. rotation files are exact rot90(base, -k) for image and mask
    bad = []
    for stem in base_train:
        for sub in ("X", "Y"):
            b = tifffile.imread(find_image(ds / f"{sub}_train", stem))
            for k in (1, 2, 3):
                rp = find_image(ds / f"{sub}_train", f"rotation{stem}_{k}")
                if rp is None or not np.array_equal(tifffile.imread(rp), np.rot90(b, -k)):
                    bad.append(f"{sub}:{stem}:{k}")
    record("rotation_convention_np_rot90_minus_k", not bad, n_checked=6 * len(base_train), n_bad=len(bad), examples=bad[:5])

    # 3. label values in all masks
    labels = set()
    for sp in ("train", "val"):
        for n in names[sp]["Y"]:
            labels |= set(np.unique(tifffile.imread(ds / f"Y_{sp}" / n)).tolist())
    record("mask_labels_subset_of_0_to_4", labels <= {0, 1, 2, 3, 4}, labels=sorted(labels))

    # 4. processed manifest
    rows = _read_manifest(processed)
    by_split = {sp: [r for r in rows if r["split"] == sp] for sp in ("train", "val")}
    record("val_complete", len(by_split["val"]) == len(names["val"]["X"]), n_processed=len(by_split["val"]),
           n_expected=len(names["val"]["X"]))
    miss = processed / "missing_float_data.csv"
    n_missing = sum(1 for _ in open(miss)) - 1 if miss.exists() else None
    record("train_float_coverage", True, n_processed=len(by_split["train"]), n_base_total=len(base_train),
           n_missing_float=n_missing, training_views=len(by_split["train"]) * len(cfg["data"]["train_rotations"]),
           warning=(f"{n_missing} base training images have no float32 map and are excluded (missing_float_data.csv)"
                    if n_missing else None))
    record("registration_quality", all(r["registration_ok"] == "True" for r in rows),
           n_failed=sum(r["registration_ok"] != "True" for r in rows),
           min_ecc=min(float(r["ecc_correlation"]) for r in rows), min_pearson=min(float(r["pearson_r"]) for r in rows),
           max_abs_rotation_deg=max(abs(float(r["rotation_deg"])) for r in rows), max_abs_shear=max(abs(float(r["shear"])) for r in rows),
           max_peak_shift_px=max(float(r["peak_shift_px"]) for r in rows),
           extended_search=[r["stem"] for r in rows if r.get("registration_search") == "extended"],
           borderline_accepted=[r["stem"] for r in rows if r.get("registration_basis") == "borderline_nonlinear_export_verified"])
    vf = [float(r["valid_fraction"]) for r in rows]
    record("annotated_coverage", True, valid_fraction_min=min(vf), valid_fraction_mean=float(np.mean(vf)), valid_fraction_max=max(vf))

    # 5. integrity of the processed files + leakage between splits
    hashes: Dict[str, Dict[str, str]] = {"train": {}, "val": {}}
    problems = []
    px_classes = {sp: np.zeros(5) for sp in ("train", "val")}
    for r in rows:
        sp, stem = r["split"], r["stem"]
        pp, mp = processed / sp / "phase" / f"{stem}.tif", processed / sp / "mask" / f"{stem}.tif"
        if not pp.exists() or not mp.exists():
            problems.append(f"missing:{sp}:{stem}")
            continue
        ph, mk = tifffile.imread(pp), tifffile.imread(mp)
        if ph.dtype != np.float32 or ph.shape != (cfg["data"]["native_size"],) * 2 or not np.isfinite(ph).all():
            problems.append(f"phase:{sp}:{stem}")
        if mk.shape != ph.shape or not set(np.unique(mk).tolist()) <= {0, 1, 2, 3, 4, ign}:
            problems.append(f"mask:{sp}:{stem}")
        if hashlib.sha256(ph.tobytes()).hexdigest() != r["phase_sha256"]:
            problems.append(f"hash:{sp}:{stem}")
        # canonical hash over the 4 rotations -> detects the same field of view stored under two names
        canon = min(hashlib.sha256(np.ascontiguousarray(np.rot90(ph, -k)).tobytes()).hexdigest() for k in range(4))
        hashes[sp][stem] = canon
        for c in range(5):
            px_classes[sp][c] += (mk == c).sum()
    record("processed_files_integrity", not problems, n_problems=len(problems), examples=problems[:5])
    overlap_names = sorted(set(by_split["train"][i]["stem"] for i in range(len(by_split["train"]))) &
                           set(r["stem"] for r in by_split["val"]))
    overlap_hash = sorted(set(hashes["train"].values()) & set(hashes["val"].values()))
    record("no_leakage_between_splits", not overlap_names and not overlap_hash, shared_stems=overlap_names[:5],
           n_shared_content_hashes=len(overlap_hash))

    # 6. validation design (six patches per storage time point) and class presence
    days: Dict[int, int] = {}
    for r in by_split["val"]:
        days[int(r["storage_day"])] = days.get(int(r["storage_day"]), 0) + 1
    record("validation_design", set(days.values()) == {6} and len(days) == 11, images_per_day=dict(sorted(days.items())))
    for sp in ("train", "val"):
        record(f"all_classes_present_{sp}", bool((px_classes[sp][1:] > 0).all()),
               pixels_per_class=[int(v) for v in px_classes[sp]])

    ok = all(c["ok"] for c in checks.values())
    report = {"ok": ok, "checks": checks}
    write_json(processed / "verification_report.json", report)
    return report
