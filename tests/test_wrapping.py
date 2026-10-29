"""Phase-wrap exclusion (D-21) and its wiring into the cell table."""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpi.analysis import wrapping as wrp  # noqa: E402
from qpi.analysis.drymass import cells_table, measure_cells  # noqa: E402
from qpi.config import load_config  # noqa: E402

TWO_PI = 2 * math.pi


def _scene(n=96):
    """Background 1.0 rad (as provided), two disc cells: one ordinary (peak 3), one wrapped (true peak 7.5 -> wraps)."""
    yy, xx = np.mgrid[:n, :n]
    raw = np.full((n, n), 1.0)
    lab = np.zeros((n, n), np.uint8)
    for (cy, cx, peak, c) in ((25, 25, 3.0, 1), (65, 65, 7.5, 2)):
        r = np.hypot(yy - cy, xx - cx)
        cell = r < 14
        raw = np.where(cell, 1.0 + (peak - 1.0) * (1 - (r / 14) ** 2), raw)
        lab[cell] = c
    raw = np.mod(raw, TWO_PI)                      # what the camera chain stores: wrapped into [0, 2*pi)
    return raw, lab


def test_wrapped_pixels_detect_ceiling_and_jumps():
    raw, _ = _scene()
    px = wrp.wrapped_pixels(raw, TWO_PI, 0.3, math.pi)
    assert px[65, 65 - 3:65 + 3].any() or px[60:70, 60:70].any()     # inside the wrapped cell
    assert not px[20:30, 20:30].any()                                # ordinary cell untouched
    assert not px[:5, :5].any()                                      # flat background untouched


def test_flag_components_marks_only_the_wrapped_cell():
    from scipy import ndimage as ndi
    raw, lab = _scene()
    labl, n = ndi.label(lab > 0)
    flag = wrp.flag_components(labl, n, wrp.wrapped_pixels(raw, TWO_PI, 0.3, math.pi), 2)
    assert n == 2 and flag[1:].tolist() == [False, True]


def test_measure_cells_adds_wrapped_column_and_uses_raw_phase():
    cfg = load_config()
    cfg["evaluation"]["cell_area_range_um2"] = [0.0, 1e6]
    raw, lab = _scene()
    valid = np.ones_like(lab, bool)
    offset = 1.0                                   # prepare_dataset subtracted the background level
    rows = measure_cells(cfg, "s", 0, lab.astype(np.int64), raw - offset, valid, offset)
    assert [r["wrapped"] for r in sorted(rows, key=lambda r: r["cell"])] == [False, True]
    # without the offset the ceiling test would miss it: the stored (background-referenced) phase is 1 rad lower
    px_ref = wrp.wrapped_pixels(raw - offset, TWO_PI, 0.3, math.pi)
    px_raw = wrp.wrapped_pixels(raw, TWO_PI, 0.3, math.pi)
    assert px_raw.sum() >= px_ref.sum()


def test_cells_table_and_summary():
    cfg = load_config()
    cfg["evaluation"]["cell_area_range_um2"] = [0.0, 1e6]
    raw, lab = _scene()
    val = {"s": {"phase": raw - 1.0, "gt": lab.astype(np.int64), "valid": np.ones_like(lab, bool), "day": 3, "offset": 1.0}}
    df = cells_table(cfg, val, None)
    assert "wrapped" in df.columns and int(df.wrapped.sum()) == 1
    s = wrp.summarise(df, cfg["data"]["class_names"])
    assert s["n_cells"] == 2 and s["n_wrapped"] == 1 and s["by_class"]["echinocyte"]["n_wrapped"] == 1
    assert s["by_class"]["discocyte"]["n_wrapped"] == 0 and s["by_storage_day"][0]["pct_wrapped"] == 50.0
    assert s["n_images_with_wrapped"] == 1


def test_config_rule_is_the_documented_one():
    r = wrp.rule(load_config())
    assert r == {"ceiling": TWO_PI, "tolerance": 0.3, "jump": math.pi, "dilation_px": 2}


def test_phase_integral_reports_wrapped_cells_and_nowrap_statistics(tmp_path):
    from qpi.analysis.phase_integral import analyse_run
    cfg = load_config()
    cfg["evaluation"]["cell_area_range_um2"] = [0.0, 1e6]
    raw, lab = _scene()
    gt = lab.astype(np.int64)
    val = {"s": {"phase": raw - 1.0, "gt": gt, "valid": np.ones_like(lab, bool), "day": 3, "offset": 1.0}}
    run = tmp_path / "run"
    run.mkdir()
    pred = gt.copy()
    pred[24:27, 24:27] = 0                                   # small error in the ordinary cell
    np.savez_compressed(run / "predictions_last.npz", s=pred.astype(np.uint8))
    res = analyse_run(cfg, "r", run, "last", val, tmp_path / "out")
    s = res["summary"]
    assert s["gt_cells"] == 2 and s["gt_cells_wrapped"] == 1 and s["n_images_without_wrapped_components"] == 0
    assert s["ipe_global_raw_mean_pct_images_without_wrapped"] is None
    assert s["matched_cells_nowrap"] == 1 and "percell_abs_median_pct_nowrap" in s
    import pandas as pd
    assert pd.read_csv(tmp_path / "out" / "r__per_cell.csv")["wrapped"].tolist().count(True) == 1
