"""Evaluation on controlled inputs: semantic metrics, A2 reconstruction (no ground truth, every foreground pixel
assigned, phase untouched), cell-level phase / optical volume / dry mass, touching vs isolated, JSON export."""
import json

import numpy as np
import pytest

from qpicell.data.cells import split_cells
from qpicell.evaluation.agreement import agreement, ccc
from qpicell.evaluation.evaluate import prepare_cfg, score_predictions
from qpicell.evaluation.instances import a2, check_assignment, reconstruct
from qpicell.evaluation.segmentation import SegmentationScores
from qpicell.io import write_json
from qpicell.physics import sha256_array


def _scene(N=128):
    """Two touching discs (one cluster) + one isolated disc; phase 1 rad on the cells, small noise outside."""
    yy, xx = np.mgrid[0:N, 0:N]
    d1 = (yy - 40) ** 2 + (xx - 35) ** 2 < 14 ** 2; d2 = (yy - 40) ** 2 + (xx - 60) ** 2 < 14 ** 2
    d3 = (yy - 95) ** 2 + (xx - 90) ** 2 < 14 ** 2
    mask = np.zeros((N, N), np.uint8); mask[d1 | d2] = 1; mask[d3] = 2
    rng = np.random.default_rng(0)
    phase = (rng.normal(0, 0.01, (N, N)) + 1.0 * (mask > 0)).astype(np.float32)
    cells = split_cells(mask > 0, 3).astype(np.int64)
    return phase, mask, cells


def test_perfect_segmentation(cfg):
    m = np.zeros((64, 64), np.uint8); m[10:30, 10:30] = 1; m[35:55, 35:55] = 3
    s = SegmentationScores(cfg); s.add(m, m, np.ones((64, 64)), "a", 0); out = s.summary()
    assert out["dice_discocyte"] == pytest.approx(1) and out["aji"] == pytest.approx(1) and out["ipe_global"] == pytest.approx(0)
    assert np.isnan(out["dice_spherocyte"])


def test_ccc_and_agreement():
    a = np.linspace(1, 10, 50)
    assert ccc(a, a) == pytest.approx(1) and ccc(a, a + 5) < 0.6
    r = agreement(a * 1.1, a); assert r["pearson_r"] == pytest.approx(1) and r["mdare_pct"] == pytest.approx(10)


def test_a2_assigns_every_foreground_pixel():
    phase, mask, cells = _scene()
    lab = a2(mask, 3)
    chk = check_assignment(mask, lab)
    assert chk["unassigned_fg_px"] == 0 and chk["labels_outside_fg_px"] == 0 and chk["n_instances"] == 3
    labs, checks = reconstruct(mask, 3)
    assert checks["CC"]["n_instances"] == 2 and checks["A2"]["n_instances"] == 3      # CC merges the touching pair


def test_score_predictions_perfect_prediction(cfg):
    """Prediction = annotation: A2 matches all 3 cells with zero phase error, CC merges the touching pair; the phase map
    is unchanged; optical volume and dry mass follow the physical factors."""
    phase, mask, cells = _scene(); before = sha256_array(phase)
    preds = {"img": dict(pred=mask.copy(), phase=phase, mask=mask, cells=cells, wrapped=np.zeros_like(mask), day=0)}
    c = dict(cfg); c["cells"] = dict(cfg["cells"], area_range_um2=[1.0, 1000.0])
    res = score_predictions(c, preds, {"img": before})
    assert sha256_array(phase) == before
    A, CC = res["instances"]["A2"], res["instances"]["CC"]
    assert A["cells"]["scored"]["match_pct"] == pytest.approx(100) and abs(A["cells"]["scored"]["cell_ipe_abs_median_pct"]) < 1e-9
    assert A["cells"]["touching"]["n"] == 2 and A["cells"]["isolated"]["n"] == 1
    assert CC["cells"]["touching"]["merged_pct"] == pytest.approx(100)
    assert A["assignment"]["unassigned_fg_px"] == 0 and A["phase"]["retained_all_pct"] == pytest.approx(100)
    assert A["clusters"]["multi_cell_clusters"] == 1 and A["clusters"]["partition_residual_abs_median_pct"] == pytest.approx(0, abs=1e-9)
    T = res["cells_tables"]["A2"]; k = prepare_cfg(c)
    row = T.iloc[0]
    assert row.dm_gt_pg == pytest.approx(k["_k_pg"] * row.S_gt) and row.ov_gt_um3 == pytest.approx(k["_k_ov"] * row.S_gt)
    assert res["reference_cells"]["all"] == 3 and res["reference_cells"]["scored"] == 3


def test_partial_prediction_counts_network_loss(cfg):
    """Removing a contact band from the prediction lowers the touching cells' phase; A2 still assigns every predicted
    pixel (the loss is attributed to the network, not to the instance step)."""
    phase, mask, cells = _scene()
    pred = mask.copy(); pred[:, 47:49] = 0
    c = dict(cfg); c["cells"] = dict(cfg["cells"], area_range_um2=[1.0, 1000.0])
    res = score_predictions(c, {"img": dict(pred=pred, phase=phase, mask=mask, cells=cells, wrapped=np.zeros_like(mask), day=0)}, {})
    A = res["instances"]["A2"]
    assert A["phase"]["lost_by_network_pct"] > 0 and A["phase"]["fg_px_unassigned_by_instance_step"] == 0
    assert A["cells"]["touching"]["cell_ipe_signed_median_pct"] < 0


def test_json_export_roundtrip(cfg, tmp_path):
    phase, mask, cells = _scene()
    res = score_predictions(cfg, {"img": dict(pred=mask, phase=phase, mask=mask, cells=cells, wrapped=np.zeros_like(mask), day=0)}, {})
    res.pop("cells_tables")
    write_json(tmp_path / "m.json", res)
    back = json.loads((tmp_path / "m.json").read_text())
    assert back["semantic"]["mean_dice"] == pytest.approx(res["semantic"]["mean_dice"])
    assert set(back["instances"]) == {"A2", "CC"}


def test_all_foreground_prediction(cfg):
    """A degenerate prediction (everything foreground, as from an untrained model) must still give one instance per
    connected region and be scored, not crash."""
    phase, mask, cells = _scene()
    full = np.ones_like(mask)
    lab = a2(full, 3)
    assert check_assignment(full, lab)["n_instances"] >= 1 and lab.min() >= 1
    res = score_predictions(cfg, {"img": dict(pred=full, phase=phase, mask=mask, cells=cells, wrapped=np.zeros_like(mask), day=0)}, {})
    assert res["instances"]["A2"]["assignment"]["unassigned_fg_px"] == 0
    assert res["instances"]["CC"]["n_predicted_instances"] == 1
