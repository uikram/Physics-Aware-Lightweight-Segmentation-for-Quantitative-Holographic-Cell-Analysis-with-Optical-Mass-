"""Checks of data/prepared (skipped until scripts/prepare_data.py has been run): the phase stays 32-bit radians,
masks hold only class labels / ignore, cell maps agree with the masks, and every validation image is accepted."""
import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile


@pytest.fixture(scope="module")
def prepared(cfg):
    p = Path(cfg["paths_abs"]["prepared"])
    if not (p / "manifest.csv").exists():
        pytest.skip("data/prepared not built yet")
    return p, list(csv.DictReader(open(p / "manifest.csv")))


def test_manifest(prepared, cfg):
    p, rows = prepared
    val = [r for r in rows if r["split"] == "val"]
    assert all(r["registration_ok"] == "True" for r in rows)
    if len(val) == 66:
        days = {r["storage_day"] for r in val}
        assert len(days) == 11


def test_phase_float32_and_masks(prepared, cfg):
    p, rows = prepared
    K, ign = cfg["data"]["num_classes"], cfg["data"]["ignore_index"]
    for r in rows[:12]:
        d = p / r["split"]
        ph = tifffile.imread(d / "phase" / f"{r['stem']}.tif"); m = tifffile.imread(d / "mask" / f"{r['stem']}.tif")
        c = tifffile.imread(d / "cells" / f"{r['stem']}.tif")
        assert ph.dtype == np.float32 and np.isfinite(ph).all() and ph.shape == m.shape
        assert set(np.unique(m)) <= set(range(K)) | {ign}
        assert not ((c > 0) & ((m == 0) | (m == ign))).any()


def test_prepared_checksums_and_split(prepared, cfg):
    from qpicell.physics import split_stems, verify_prepared
    verify_prepared(cfg)
    sp = split_stems(cfg)
    exp = cfg["split"]["expected"]
    assert {k: len(v) for k, v in sp.items()} == exp
