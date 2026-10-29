"""Fast unit tests (no pretrained weights, no GPU, no real data)."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpi.config import load_config  # noqa: E402
from qpi.data.dataset import QPIDataset  # noqa: E402
from qpi.data.registration import warp_mask_to_float  # noqa: E402
from qpi.evaluation.metrics import SegmentationMetrics  # noqa: E402
from qpi.experiments import build_grid, trainable_specs  # noqa: E402
from qpi.training.losses import PhysicsAwarePhaseLoss, integrated_phase_preservation  # noqa: E402
from qpi.utils.env import enforce_gpu_policy  # noqa: E402
from qpi.utils.io import sanitize  # noqa: E402
from qpi.utils.outputs import claim_output_root  # noqa: E402


def _fake_processed(tmp: Path, n=64):
    rng = np.random.default_rng(0)
    for sp in ("train", "val"):
        (tmp / sp / "phase").mkdir(parents=True)
        (tmp / sp / "mask").mkdir(parents=True)
    ph = rng.normal(size=(n, n)).astype(np.float32)
    mk = rng.integers(0, 5, size=(n, n)).astype(np.uint8)
    for sp in ("train", "val"):
        tifffile.imwrite(tmp / sp / "phase" / "a.tif", ph)
        tifffile.imwrite(tmp / sp / "mask" / "a.tif", mk)
    with open(tmp / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "stem", "registration_ok", "storage_day"])
        w.writeheader()
        for sp in ("train", "val"):
            w.writerow({"split": sp, "stem": "a", "registration_ok": "True", "storage_day": 0})
    return ph, mk


def test_rotation_views_match_rot90_minus_k(tmp_path):
    ph, mk = _fake_processed(tmp_path)
    ds = QPIDataset(tmp_path, "train", 64, [0, 1, 2, 3])
    assert len(ds) == 4
    for i, k in enumerate([0, 1, 2, 3]):
        it = ds[i]
        assert np.array_equal(it["phase_raw"][0].numpy(), np.rot90(ph, -k))
        assert np.array_equal(it["mask"].numpy(), np.rot90(mk, -k))
    assert ds.sample_id(0) == "a" and ds.sample_id(1) == "rotationa_1"


def test_input_is_zscored_but_raw_phase_is_kept(tmp_path):
    ph, _ = _fake_processed(tmp_path)
    it = QPIDataset(tmp_path, "val", 64, [0])[0]
    assert abs(float(it["phase"].mean())) < 1e-5 and abs(float(it["phase"].std()) - 1) < 1e-3
    assert np.allclose(it["phase_raw"][0].numpy(), ph)


def test_mask_warp_is_forward_and_marks_uncovered_as_ignore():
    m = np.zeros((10, 10), np.uint8)
    m[2:4, 2:4] = 1
    warp = np.array([[1, 0, 3], [0, 1, 1]], np.float32)       # x_float = x_8bit + (3, 1)
    out = warp_mask_to_float(m, warp, (12, 12))
    assert (out[3:5, 5:7] == 1).all() and (out == 1).sum() == 4
    assert out[0, 0] == 255 and out[11, 0] == 255             # not covered by the source image


def test_loss_ignores_unannotated_pixels():
    loss = PhysicsAwarePhaseLoss(5, [0.5, 1, 1.5, 2, 2], 255, 0.1, 0.05, 0.1, 0.1, 1e-6)
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 16, 16)
    target = torch.randint(0, 5, (2, 16, 16))
    target[:, :, 8:] = 255
    phase = torch.randn(2, 1, 16, 16)
    a = loss(logits, target, phase)
    logits2 = logits.clone()
    logits2[:, :, :, 8:] += 100.0 * torch.randn_like(logits2[:, :, :, 8:])
    phase2 = phase.clone()
    phase2[:, :, :, 8:] = 1e3
    # the boundary-gradient term looks at 1-pixel neighbours of the border column; compare away from it
    c1 = loss.components(logits, target, phase)
    c2 = loss.components(logits2, target, phase2)
    assert torch.isfinite(a)
    assert torch.allclose(c1["L_dice"], c2["L_dice"], atol=1e-5)
    assert torch.allclose(c1["L_pv"], c2["L_pv"], atol=1e-5)
    assert torch.allclose(c1["L_pmc"], c2["L_pmc"], atol=1e-5)


def test_ipp_zero_for_identical_foreground():
    fg = (torch.rand(2, 1, 8, 8) > 0.5).float()
    phase = torch.rand(2, 1, 8, 8)
    out = integrated_phase_preservation(fg, fg.squeeze(1), phase, torch.ones(2, 8, 8), 1e-6)
    assert torch.allclose(out, torch.zeros(2), atol=1e-6)


def test_metrics_perfect_prediction_and_absent_class_is_nan():
    t = np.zeros((20, 20), int)
    t[2:8, 2:8] = 1
    t[10:15, 10:15] = 3
    m = SegmentationMetrics(5, ["bg", "a", "b", "c", "d"])
    m.update(t.copy(), t, np.random.default_rng(0).random((20, 20)))
    r = m.compute()
    assert r["mean_dice"] == pytest.approx(1.0) and r["ipe_global"] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(r["dice_b"]) and np.isnan(r["dice_d"])


def test_ignored_pixels_do_not_count_as_errors():
    t = np.zeros((20, 20), int)
    t[2:8, 2:8] = 1
    p = t.copy()
    t[:, 12:] = 255
    p[:, 12:] = 1                                              # wrong, but unannotated
    m = SegmentationMetrics(5, ["bg", "a", "b", "c", "d"])
    m.update(p, t, np.ones((20, 20)))
    assert m.compute()["dice_a"] == pytest.approx(1.0)


def test_grid_sizes():
    full = build_grid(load_config())
    assert len(full) == 30 and len(trainable_specs(full)) == 27
    smoke = build_grid(load_config("smoke.yaml"))
    assert len(smoke) == 18 and len(trainable_specs(smoke)) == 15
    aliases = {s.run_id: s.alias_of for s in full if s.alias_of}
    assert aliases == {f"{a}_abl_full": f"{a}_lora_r8" for a in ("edge_sam", "mobile_sam", "mobilenet_unet")}


def test_gpu_policy_refuses_other_gpus(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError):
        enforce_gpu_policy([2, 3])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    enforce_gpu_policy([2, 3])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(RuntimeError):
        enforce_gpu_policy([2, 3])


def test_output_mode_guard(tmp_path):
    cfg = load_config("smoke.yaml")
    cfg["paths_abs"]["output_root"] = tmp_path / "full"
    with pytest.raises(RuntimeError):
        claim_output_root(cfg)
    full = load_config()
    full["paths_abs"]["output_root"] = tmp_path / "smoke"
    with pytest.raises(RuntimeError):
        claim_output_root(full)


def test_sanitize_makes_json_strict():
    assert sanitize({"a": float("nan"), "b": [float("inf"), 1.0], "c": np.float32(2.0)}) == {"a": None, "b": [None, 1.0], "c": 2.0}


# ----------------------------------------------------------------------------------------------------------
# image-size / spatial-alignment tests (training grid, evaluation upsampling)
# ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("size", [256, 1024])
def test_mask_and_phase_stay_aligned_after_resize_and_eval_upsampling(tmp_path, size):
    """A square cell is resized with the dataset code, turned into logits, upsampled with the evaluation code and
    must land on the original pixels (no half-block shift between mask, phase and prediction)."""
    import torch.nn.functional as F
    n = 768
    for sp in ("train", "val"):
        (tmp_path / sp / "phase").mkdir(parents=True)
        (tmp_path / sp / "mask").mkdir(parents=True)
    yy, xx = np.mgrid[0:n, 0:n]
    cell = (np.abs(yy - 300) < 90) & (np.abs(xx - 450) < 120)
    mask = cell.astype(np.uint8) * 1
    phase = np.where(cell, 1.5, 0.0).astype(np.float32)
    for sp in ("train", "val"):
        tifffile.imwrite(tmp_path / sp / "phase" / "a.tif", phase)
        tifffile.imwrite(tmp_path / sp / "mask" / "a.tif", mask)
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "stem", "registration_ok", "storage_day"])
        w.writeheader()
        for sp in ("train", "val"):
            w.writerow({"split": sp, "stem": "a", "registration_ok": "True", "storage_day": 0})
    it = QPIDataset(tmp_path, "val", size, [0], return_native=True)[0]
    m, ph = it["mask"].numpy(), it["phase_raw"][0].numpy()
    ys, xs = np.nonzero(m > 0)
    ysp, xsp = np.nonzero(ph > 0.75)
    assert abs(ys.mean() - ysp.mean()) < 0.35 and abs(xs.mean() - xsp.mean()) < 0.35      # mask vs phase on the model grid
    oh = F.one_hot(it["mask"][None], 5).permute(0, 3, 1, 2).float() * 20                 # oracle logits
    up = F.interpolate(oh, size=(n, n), mode="bilinear", align_corners=False).argmax(1)[0].numpy()
    ys0, xs0 = np.nonzero(mask > 0)
    ys1, xs1 = np.nonzero(up > 0)
    assert abs(ys0.mean() - ys1.mean()) < 0.6 and abs(xs0.mean() - xs1.mean()) < 0.6     # model grid -> native grid (quantisation at 256 is <= 0.5 px; the old top-left sampling gave 1.0)
    assert 2 * ((up > 0) & (mask > 0)).sum() / ((up > 0).sum() + (mask > 0).sum()) > 0.98


@pytest.mark.parametrize("arch", ["edge_sam", "mobile_sam", "mobilenet_unet"])
def test_models_return_logits_at_input_size(arch):
    from qpi.experiments import resolve, run_config
    from qpi.models import build_model
    if arch == "mobile_sam":
        pytest.importorskip("mobile_sam")
    cfg = load_config()
    rc = run_config(cfg, resolve(build_grid(cfg), f"{arch}_lora_r8"))
    model = build_model(rc, None, pretrained=False).eval()
    for size in (rc["image_size"], 512):
        with torch.no_grad():
            out = model(torch.randn(1, 1, size, size))
        assert tuple(out.shape) == (1, rc["num_classes"], size, size)


# ---------------------------------------------------------------------------------------------------------------------
# Registration: wide-offset recovery and the geometric borderline evidence
# ---------------------------------------------------------------------------------------------------------------------
def _synthetic_phase(n=768, seed=3):
    import cv2
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n), np.float32)
    for _ in range(140):
        cx, cy, r = rng.integers(0, n), rng.integers(0, n), rng.integers(12, 26)
        cv2.circle(img, (int(cx), int(cy)), int(r), float(rng.uniform(0.5, 2.5)), -1)
    img = cv2.GaussianBlur(img, (0, 0), 3) + 0.05 * cv2.GaussianBlur(rng.standard_normal((n, n)).astype(np.float32), (0, 0), 6)
    return img.astype(np.float32)


def _to_8bit_frame(phase, w, gamma=1.0):
    import cv2
    g = cv2.warpAffine(phase, np.asarray(w, np.float32), phase.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    g = (g - g.min()) / (g.max() - g.min())
    return np.clip(255 * g ** gamma, 0, 255).astype(np.uint8)


def test_registration_recovers_large_offset_that_standard_search_misses():
    from qpi.data.registration import estimate_transform
    phase = _synthetic_phase()
    w_true = np.array([[0.6147, 0.0, 225.0], [0.0, 0.778, 105.0]])
    grey = _to_8bit_frame(phase, w_true)
    reg = estimate_transform(phase, grey, 4)
    assert reg.ecc_correlation > 0.98 and reg.pearson_r > 0.95
    assert np.abs(reg.warp - w_true).max() < 1.0
    assert reg.peak_shift_px == 0 and reg.search == "extended"


def test_geometry_check_peak_at_zero_for_aligned_and_displaced_for_shifted_pair():
    from qpi.data.registration import geometry_check
    phase = _synthetic_phase()
    eye = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    grey = _to_8bit_frame(phase, eye)
    rho, shift = geometry_check(phase, grey, eye)
    assert rho > 0.9 and shift == 0
    _, shift2 = geometry_check(phase, np.roll(grey, 3, axis=1), eye)       # same image displaced by 3 px
    assert shift2 >= 2


def test_registration_gate_accepts_nonlinear_export_but_not_misalignment():
    from types import SimpleNamespace as R
    from qpi.data.prepare import registration_gate
    cfg = load_config()["registration"]
    good = R(ecc_correlation=0.995, pearson_r=0.99, spearman_r=0.99, peak_shift_px=0.0)
    nonlinear = R(ecc_correlation=0.962, pearson_r=0.947, spearman_r=0.978, peak_shift_px=0.0)     # X_train_00159
    failed = R(ecc_correlation=0.218, pearson_r=0.211, spearman_r=0.20, peak_shift_px=3.0)         # 20110526_1 before fix
    shifted = R(ecc_correlation=0.95, pearson_r=0.93, spearman_r=0.94, peak_shift_px=2.0)
    flat = R(ecc_correlation=0.95, pearson_r=0.93, spearman_r=0.94, peak_shift_px=0.0)             # no rank gain
    assert registration_gate(good, cfg) == (True, "primary")
    assert registration_gate(nonlinear, cfg) == (True, "borderline_nonlinear_export_verified")
    for bad in (failed, shifted, flat):
        assert registration_gate(bad, cfg) == (False, "rejected")


def test_dry_mass_agreement_runs_and_is_exact_for_identical_inputs():
    import pandas as pd
    from qpi.analysis.drymass import agreement
    stems = [f"s{i}" for i in range(6)]
    gt = pd.DataFrame({"stem": np.repeat(stems, 3), "dry_mass_pg": np.tile([10.0, 11.0, 12.0], 6) + np.repeat(np.arange(6), 3)})
    out = agreement(gt.copy(), gt)
    assert out["n_images"] == 6
    assert out["mdare_pct"] == 0.0 and abs(out["bland_altman_mean_pg"]) < 1e-12
    assert out["pearson_r"] == pytest.approx(1.0) and out["lin_ccc"] == pytest.approx(1.0)
