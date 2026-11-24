"""32-bit phase data and physical calculations: constants, unit factors, guards against 8-bit / wrong-unit data,
detection of modified prepared maps, disjoint split."""
import copy
import csv
import math

import numpy as np
import pytest
import tifffile

from qpicell.config import load_config
from qpicell.physics import (PhaseDataError, assert_mask, assert_phase, assert_physics, dry_mass_factor_pg,
                             optical_volume_factor_um3, sha256_array, split_stems, verify_prepared)


def test_constants_and_class_mapping(cfg):
    assert_physics(cfg)
    assert cfg["physics"] == {"wavelength_um": 0.666, "refractive_increment_ml_per_g": 0.2, "pixel_size_um": 0.1441}
    assert cfg["data"]["class_names"] == ["background", "discocyte", "spherocyte", "echinocyte", "stomatocyte"]
    for k, bad in (("wavelength_um", 0.633), ("refractive_increment_ml_per_g", 0.19), ("pixel_size_um", 0.144)):
        c = copy.deepcopy(cfg); c["physics"][k] = bad
        with pytest.raises(PhaseDataError):
            assert_physics(c)
    c = copy.deepcopy(cfg); c["data"]["class_names"] = ["background", "discocyte", "echinocyte", "spherocyte", "stomatocyte"]
    with pytest.raises(PhaseDataError):
        assert_physics(c)


def test_unit_factors(cfg):
    ov = optical_volume_factor_um3(cfg); dm = dry_mass_factor_pg(cfg)
    assert ov == pytest.approx(0.666 / (2 * math.pi) * 0.1441 ** 2, rel=1e-12)
    assert dm == pytest.approx(ov / 0.2, rel=1e-12)
    assert dm == pytest.approx(0.0110050578, rel=1e-6)            # pg per (rad px)
    # a uniform 1 rad disc of 50 um^2 -> optical volume 0.106 um^3 x ... check against the closed form
    area_px = 50.0 / 0.1441 ** 2
    assert dm * area_px == pytest.approx(0.666 / (2 * math.pi * 0.2) * 50.0, rel=1e-12)


def test_phase_guard():
    rng = np.random.default_rng(0)
    ok = rng.normal(0, 0.5, (64, 64)).astype(np.float32)
    assert_phase(ok)
    with pytest.raises(PhaseDataError):
        assert_phase(ok.astype(np.float64))                         # wrong dtype
    with pytest.raises(PhaseDataError):
        assert_phase((ok * 40 + 128).clip(0, 255).astype(np.uint8))  # 8-bit
    with pytest.raises(PhaseDataError):
        assert_phase(np.round(ok * 40 + 128).astype(np.float32))     # 8-bit values stored as float
    with pytest.raises(PhaseDataError):
        assert_phase(np.degrees(ok * 10).astype(np.float32))         # degrees, not radians
    bad = ok.copy(); bad[0, 0] = np.nan
    with pytest.raises(PhaseDataError):
        assert_phase(bad)


def test_mask_guard():
    m = np.zeros((8, 8), np.uint8); m[0, 0] = 255; m[1, 1] = 4
    assert_mask(m, 5, 255)
    m[2, 2] = 7
    with pytest.raises(PhaseDataError):
        assert_mask(m, 5, 255)


def _fake_prepared(tmp_path, n_train=10, n_val=3):
    rng = np.random.default_rng(1); rows = []
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            stem = f"{split}_{i:02d}"
            ph = rng.normal(0, 0.3, (32, 32)).astype(np.float32)
            (tmp_path / split / "phase").mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(tmp_path / split / "phase" / f"{stem}.tif", ph)
            rows.append({"split": split, "stem": stem, "registration_ok": "True", "phase_sha256": sha256_array(ph),
                         "background_coefficients": "[0,0,0,0,0,0]", "storage_day": "0"})
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def test_verify_prepared_detects_modification(tmp_path):
    cfg = load_config(overrides={"paths": {"prepared": str(tmp_path)}})
    rows = _fake_prepared(tmp_path)
    assert verify_prepared(cfg)["images"] == len(rows)
    f = tmp_path / "train" / "phase" / "train_03.tif"
    ph = tifffile.imread(f); tifffile.imwrite(f, (ph - 0.01).astype(np.float32))       # e.g. a second background subtraction
    with pytest.raises(PhaseDataError):
        verify_prepared(cfg)


def test_split_disjoint_and_rule(tmp_path):
    cfg = load_config(overrides={"paths": {"prepared": str(tmp_path)}})
    _fake_prepared(tmp_path, n_train=40, n_val=5)
    sp = split_stems(cfg)
    from qpicell.data.dataset import tuning_fold
    assert all(tuning_fold(s) != 0 for s in sp["train"]) and all(tuning_fold(s) == 0 for s in sp["validation"])
    assert not (set(sp["train"]) & set(sp["validation"])) and len(sp["test"]) == 5
    assert len(sp["train"]) + len(sp["validation"]) == 40
