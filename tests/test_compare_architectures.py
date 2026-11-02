"""scripts/compare_architectures.py: composition error, class count ratio and the per-architecture summary on synthetic cell tables."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qpi.config import load_config  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("compare_architectures", ROOT / "scripts" / "compare_architectures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cells(rng, scale=1.0, shift_class=False):
    rows = []
    for i, day in enumerate([0, 0, 5, 5, 12, 12, 30, 30, 47, 47]):
        for k in range(8):
            cls = 1 if k < 5 else (2 if not shift_class else 1)
            rows.append({"stem": f"img{i}", "storage_day": day, "class_idx": cls, "area_um2": (60 - day * 0.6) * scale + k,
                         "dry_mass_pg": (30 - day * 0.1) * scale + rng.normal(0, 0.5), "circularity": 0.8, "wrapped": False})
    return pd.DataFrame(rows)


def test_composition_error_zero_for_identical_and_positive_otherwise():
    ca = _load()
    rng = np.random.default_rng(0)
    gt = _cells(rng)
    assert ca.composition_error(gt, gt, 5)["mean_abs_pp"] == 0.0
    off = ca.composition_error(_cells(rng, shift_class=True), gt, 5)
    assert off["mean_abs_pp"] > 0 and off["max_abs_pp"] > 0


def test_summary_and_wrapped_exclusion():
    ca = _load()
    cfg = load_config()
    rng = np.random.default_rng(1)
    gt, pred = _cells(rng), _cells(rng, scale=1.02)
    pred.loc[0, "wrapped"] = True
    res = ca.summarise_architecture(cfg, pred, gt)
    assert res["cells_predicted"] == len(pred) - 1 and res["cells_annotated"] == len(gt)
    assert 0.9 < res["agreement"]["pearson_r"] <= 1.0
    assert abs(res["class_counts"]["discocyte"]["ratio"] - 1.0) < 0.05
    row = ca._flat(res)
    assert row["drymass_rho_gt"] < 0 and "composition_mean_abs_pp" in row
