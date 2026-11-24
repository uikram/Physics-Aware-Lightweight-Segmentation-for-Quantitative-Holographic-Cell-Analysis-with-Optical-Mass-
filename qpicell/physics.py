"""Physical constants, quantitative conversions and the guards that protect the 32-bit phase data.

Phase phi: float32 radians, referenced to the annotation-free background (subtracted ONCE, in data preparation).
  integrated phase   S = sum_{x in cell} phi(x)                                   [rad px]
  optical volume     OV = lambda / (2 pi) * S * dx^2                              [um^3]  (integrated optical path difference)
  dry mass           m  = OV / alpha = lambda / (2 pi alpha) * S * dx^2           [pg]    (alpha in um^3/pg = ml/g)
with lambda = 0.666 um, alpha = 0.2 ml/g = 0.2 um^3/pg, dx = 0.1441 um. These three numbers are asserted.
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

REQUIRED = {"wavelength_um": 0.666, "refractive_increment_ml_per_g": 0.2, "pixel_size_um": 0.1441}
CLASS_NAMES = ["background", "discocyte", "spherocyte", "echinocyte", "stomatocyte"]
PHASE_ABS_MAX_RAD = 40.0           # plausibility bound for referenced phase maps (radians, not degrees / not 8-bit)


class PhaseDataError(ValueError):
    pass


def assert_physics(cfg: Dict) -> None:
    p = cfg["physics"]
    for k, v in REQUIRED.items():
        if not math.isclose(float(p[k]), v, rel_tol=0, abs_tol=1e-12):
            raise PhaseDataError(f"physics.{k} = {p[k]}, required {v}")
    if list(cfg["data"]["class_names"]) != CLASS_NAMES:
        raise PhaseDataError(f"class mapping {cfg['data']['class_names']} != required {CLASS_NAMES}")


def optical_volume_factor_um3(cfg: Dict) -> float:
    """um^3 per (rad px)."""
    p = cfg["physics"]
    return p["wavelength_um"] / (2 * math.pi) * p["pixel_size_um"] ** 2


def dry_mass_factor_pg(cfg: Dict) -> float:
    """pg per (rad px)."""
    return optical_volume_factor_um3(cfg) / cfg["physics"]["refractive_increment_ml_per_g"]


def assert_phase(arr: np.ndarray, name: str = "phase") -> None:
    """A referenced phase map must be float32 radians: catches 8-bit conversion, integer or rescaled data and NaNs."""
    if not isinstance(arr, np.ndarray):
        raise PhaseDataError(f"{name}: expected a numpy array, got {type(arr)}")
    if arr.dtype != np.float32:
        raise PhaseDataError(f"{name}: dtype {arr.dtype}, required float32 radians (never 8-bit)")
    if arr.ndim != 2:
        raise PhaseDataError(f"{name}: expected a 2-D map, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise PhaseDataError(f"{name}: contains NaN or inf")
    a = float(np.abs(arr).max())
    if a > PHASE_ABS_MAX_RAD:
        raise PhaseDataError(f"{name}: |phase| up to {a:.1f} - not radians (degrees or 8-bit grey values?)")
    if arr.size > 100 and float(np.mean(arr == np.round(arr))) > 0.99:
        raise PhaseDataError(f"{name}: >99 % integer values - looks like quantised (8-bit) data")


def assert_mask(arr: np.ndarray, num_classes: int, ignore: int, name: str = "mask") -> None:
    if arr.dtype != np.uint8:
        raise PhaseDataError(f"{name}: dtype {arr.dtype}, required uint8 labels")
    bad = set(np.unique(arr).tolist()) - set(range(num_classes)) - {ignore}
    if bad:
        raise PhaseDataError(f"{name}: unexpected labels {sorted(bad)}")


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def verify_prepared(cfg: Dict, splits=("train", "val")) -> Dict:
    """Every prepared phase map must be float32 radians and byte-identical to the checksum written when it was prepared
    (data/prepared/manifest.csv -> phase_sha256): catches modification after preparation, a second background
    subtraction or a re-export. Returns counts; raises PhaseDataError on the first problem."""
    import tifffile
    root = Path(cfg["paths_abs"]["prepared"])
    with open(root / "manifest.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] in splits and r["registration_ok"] == "True"]
    if not rows:
        raise PhaseDataError(f"no usable rows in {root / 'manifest.csv'}")
    for r in rows:
        if "phase_sha256" not in r or not r["phase_sha256"]:
            raise PhaseDataError("manifest without phase_sha256: prepare the data with this pipeline")
        ph = tifffile.imread(root / r["split"] / "phase" / f"{r['stem']}.tif")
        assert_phase(ph, f"{r['split']}/{r['stem']}")
        if sha256_array(ph) != r["phase_sha256"]:
            raise PhaseDataError(f"{r['split']}/{r['stem']}: phase map differs from the one written at preparation")
        if not r.get("background_coefficients"):
            raise PhaseDataError(f"{r['stem']}: no background record - background subtraction not done by this pipeline")
    return {"images": len(rows), "by_split": {s: sum(r["split"] == s for r in rows) for s in splits}}


def split_stems(cfg: Dict) -> Dict[str, List[str]]:
    """train / validation / test stems (see configs/config.yaml -> split)."""
    from qpicell.data.dataset import read_manifest, tuning_fold
    sp = cfg["split"]; root = cfg["paths_abs"]["prepared"]
    tr = [r["stem"] for r in read_manifest(root, "train")]
    out = {"train": sorted(s for s in tr if tuning_fold(s, sp["fold_modulus"]) != sp["validation_fold"]),
           "validation": sorted(s for s in tr if tuning_fold(s, sp["fold_modulus"]) == sp["validation_fold"]),
           "test": sorted(r["stem"] for r in read_manifest(root, "val"))}
    a, b, c = set(out["train"]), set(out["validation"]), set(out["test"])
    if a & b or a & c or b & c:
        raise PhaseDataError(f"split overlap: train/val {len(a & b)}, train/test {len(a & c)}, val/test {len(b & c)}")
    return out
