"""Correctness and quality gate of the benchmark: ONNX Runtime FP32 against PyTorch FP32 (D-26).

Trained checkpoints (best.pt of the reference rank, LoRA merged) are loaded for inference only. Nothing here is timed.

1. ``logit_comparison``: logits of the ONNX model and the PyTorch model on 16 real validation maps: maximum and mean absolute
   difference, argmax agreement, and whether every disagreeing pixel is a near-tie (margin between the two top PyTorch logits
   within 2 x the largest logit difference).
2. ``quality``: on every validation image (rotation 0) the segmentation of both models is scored with the evaluation code of the
   paper (mean Dice against the annotation) and the per-image total dry mass of the predicted cells is computed with the analysis
   code. The ONNX model is accepted when it stays within ``ACCEPTANCE`` of the PyTorch model (thresholds fixed before the first
   trial, unchanged).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from qpi.benchmark import (N_DEVICE_INPUTS, _export_onnx, _first_tensor, _make_session, _release, apply_tf32_policy,
                           benchmark_architectures, reference_rank, warm_libraries)
from qpi.experiments import build_grid, resolve, run_config, run_dir
from qpi.utils.io import write_json

# Acceptance gate for the ONNX model against the PyTorch FP32 model of the same checkpoint on the validation images.
ACCEPTANCE = {"argmax_agreement_pct_min": 99.9,          # pixels where both models give the same class
              "mean_dice_abs_change_max": 0.001,         # |mean Dice (ONNX) - mean Dice (PyTorch)|, both against the annotation
              "dry_mass_mean_abs_rel_error_max": 1e-3,   # per-image total dry mass relative to the PyTorch prediction, mean over images
              "dry_mass_p99_rel_error_max": 5e-3}        # and its 99th percentile over images


def load_trained(cfg: Dict, arch: str, rank: int):
    """Model with the LoRA of ``rank`` injected and the best.pt weights loaded, LoRA merged (deployment form)."""
    from qpi.models import build_model
    spec = resolve(build_grid(cfg), f"{arch}_lora_r{rank}")
    rc = run_config(cfg, spec)
    ckpt = run_dir(cfg, spec.run_id) / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    model = build_model(rc, cfg["paths_abs"]["weights_dir"], pretrained=False)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model_state"], strict=True)
    if hasattr(model, "merge_lora"):
        model.merge_lora()
    return model.eval().float(), rc, str(ckpt), state.get("epoch")


def compare_logits(ref: np.ndarray, got: np.ndarray) -> Dict[str, Any]:
    """Logit differences and an explanation of every arg-max disagreement by near-ties.

    ``explained_by_near_ties`` is True only when the disagreeing pixels are a small share (< 1 %) and the PyTorch margin between the
    two top logits at every one of them is within twice the largest logit difference."""
    ref = ref.astype(np.float32)
    got = got.astype(np.float32)
    nf_ref, nf_got = int((~np.isfinite(ref)).sum()), int((~np.isfinite(got)).sum())
    ref_c, got_c = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(got, nan=0.0, posinf=0.0, neginf=0.0)
    diff = np.abs(ref_c - got_c)
    max_diff = float(diff.max())
    dis = ref_c.argmax(1) != got_c.argmax(1)
    srt = np.sort(ref_c, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    dis_pct = float(100.0 * dis.mean())
    out = {"max_abs_logit_diff": max_diff, "mean_abs_logit_diff": float(diff.mean()),
           "nonfinite_reference": nf_ref, "nonfinite_other": nf_got,
           "argmax_agreement_pct": float(100.0 - dis_pct), "disagree_pct": dis_pct, "n_pixels": int(dis.size),
           "n_disagree": int(dis.sum())}
    within = True
    if dis.any():
        within = bool(margin[dis].max() <= 2.0 * max_diff + 1e-6)
        out["max_margin_at_disagreeing_pixels"] = float(margin[dis].max())
    out["all_disagreements_within_2x_max_logit_diff"] = within
    out["explained_by_near_ties"] = bool(within and dis_pct < 1.0 and nf_ref == 0 and nf_got == 0)
    return out


def _inputs(cfg: Dict, rc: Dict, seed: int = 42):
    from qpi.data.dataset import QPIDataset
    ds = QPIDataset(cfg["paths_abs"]["processed_root"], "val", rc["image_size"], [0], cfg["data"]["ignore_index"],
                    cfg["data"]["max_val_samples"])
    idx = np.random.default_rng(seed).permutation(len(ds))[: min(N_DEVICE_INPUTS, len(ds))]
    return [ds[int(i)]["phase"][None] for i in idx]


def _onnx_runner(path: Path, dev):
    """Callable x (torch, on ``dev``) -> logits (torch, on ``dev``) through an ONNX Runtime session with GPU-bound IO."""
    import onnxruntime as ort
    sess = _make_session(path, dev)
    in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    def run(x):
        b = sess.io_binding()
        b.bind_ortvalue_input(in_name, ort.OrtValue.ortvalue_from_numpy(x.detach().cpu().numpy(), "cuda", dev.index or 0))
        b.bind_output(out_name, "cuda", dev.index or 0)
        sess.run_with_iobinding(b)
        return torch.from_numpy(b.copy_outputs_to_cpu()[0]).to(dev).float()
    return run, sess


def verdict_for(row: Dict[str, Any], n_dry_mass: int) -> str:
    ok = (row["argmax_agreement_pct"] >= ACCEPTANCE["argmax_agreement_pct_min"]
          and abs(row["mean_dice_change"]) <= ACCEPTANCE["mean_dice_abs_change_max"]
          and n_dry_mass > 0 and row["dry_mass_mean_abs_rel_error"] <= ACCEPTANCE["dry_mass_mean_abs_rel_error_max"]
          and row["dry_mass_p99_rel_error"] <= ACCEPTANCE["dry_mass_p99_rel_error_max"])
    return "accepted" if ok else "rejected"


def check_architecture(cfg: Dict, arch: str, rank: int, dev, scratch: Path) -> Dict[str, Any]:
    """ONNX FP32 against PyTorch FP32 for one architecture: logits on 16 maps, then segmentation and dry mass on all validation images."""
    import torch.nn.functional as F
    from qpi.analysis.common import load_validation
    from qpi.analysis.drymass import measure_cells
    from qpi.data.dataset import QPIDataset
    from qpi.evaluation.metrics import SegmentationMetrics
    d = cfg["data"]
    model, rc, ckpt, epoch = load_trained(cfg, arch, rank)
    model = model.to(dev).eval()
    S = int(rc["image_size"])
    path = scratch / f"eq_{arch}_fp32.onnx"
    export_model = load_trained(cfg, arch, rank)[0].to(dev).eval()     # the export changes grad flags: export a separate copy
    _export_onnx(export_model, torch.zeros(1, 1, S, S, device=dev), path)
    del export_model
    _release(dev)
    run_onnx, sess = _onnx_runner(path, dev)
    runners = {"pytorch_fp32": lambda x: _first_tensor(model(x)).float(), "onnx_fp32": run_onnx}

    with torch.inference_mode():
        xs = [x.to(dev).float() for x in _inputs(cfg, rc)]
        ref = np.concatenate([runners["pytorch_fp32"](x).cpu().numpy() for x in xs], axis=0)
        got = np.concatenate([runners["onnx_fp32"](x).cpu().numpy() for x in xs], axis=0)
    res: Dict[str, Any] = {"architecture": arch, "rank": rank, "checkpoint": ckpt, "checkpoint_epoch": epoch, "input_size": S,
                           "n_inputs": len(xs), "logit_comparison": compare_logits(ref, got)}

    ds = QPIDataset(cfg["paths_abs"]["processed_root"], "val", S, [0], d["ignore_index"], d["max_val_samples"], return_native=True)
    val = load_validation(cfg)
    metrics = {k: SegmentationMetrics(rc["num_classes"], d["class_names"], cfg["evaluation"]["boundary_tolerance_px"], d["ignore_index"])
               for k in runners}
    agree = {k: [0, 0] for k in runners}
    mass: Dict[str, Dict[str, float]] = {k: {} for k in runners}
    with torch.inference_mode():
        for i in range(len(ds)):
            item = ds[i]
            stem = item["stem"]
            x = item["phase"][None].to(dev).float()
            native = item["phase_native"].shape[-1]
            mask = item["mask_native"].numpy()
            phase_n = item["phase_native"][0].numpy()
            valid = mask != d["ignore_index"]
            ref_pred = None
            for k, run in runners.items():
                lg = run(x)
                if lg.shape[-1] != native:
                    lg = F.interpolate(lg, size=(native, native), mode="bilinear", align_corners=False)
                pred = lg.argmax(1)[0].cpu().numpy().astype(np.uint8)
                if k == "pytorch_fp32":
                    ref_pred = pred
                metrics[k].update(pred, mask, phase_n, stem, int(item["storage_day"]))
                agree[k][0] += int(((pred == ref_pred) & valid).sum())
                agree[k][1] += int(valid.sum())
                v = val[stem]
                cells = measure_cells(cfg, stem, v["day"], np.where(v["valid"], pred, 0).astype(np.int64), v["phase"], v["valid"], v["offset"])
                mass[k][stem] = float(sum(c["dry_mass_pg"] for c in cells))
    ref_dice = metrics["pytorch_fp32"].compute()["mean_dice"]
    ref_mass = mass["pytorch_fp32"]
    pos = {s: m for s, m in ref_mass.items() if m > 0}
    candidates: List[Dict[str, Any]] = []
    for k in runners:
        mc = metrics[k].compute()
        rel = np.asarray([abs(mass[k][s] - ref_mass[s]) / ref_mass[s] for s in pos])
        worst = max(pos, key=lambda s: abs(mass[k][s] - ref_mass[s]) / ref_mass[s]) if pos else None
        row = {"candidate": k, "n_images": len(mass[k]), "n_images_in_dry_mass_error": int(rel.size),
               "n_images_zero_reference_mass": len(ref_mass) - len(pos),
               "reference_dry_mass_min_pg": float(min(pos.values())) if pos else None,
               "dry_mass_worst_image": worst,
               "argmax_agreement_pct": 100.0 * agree[k][0] / max(agree[k][1], 1),
               "mean_dice": mc["mean_dice"], "mean_dice_change": mc["mean_dice"] - ref_dice,
               "dry_mass_mean_abs_rel_error": float(rel.mean()) if rel.size else None,
               "dry_mass_p99_rel_error": float(np.percentile(rel, 99)) if rel.size else None,
               "dry_mass_max_rel_error": float(rel.max()) if rel.size else None}
        row["verdict"] = "reference" if k == "pytorch_fp32" else verdict_for(row, int(rel.size))
        candidates.append(row)
    res["quality"] = {"rotation": 0, "n_images": len(ds), "candidates": candidates}
    del sess, runners, model
    path.unlink(missing_ok=True)
    _release(dev)
    return res


def run_equivalence(cfg: Dict, dev, out_path: Path, scratch: Path) -> Dict[str, Any]:
    """Check every benchmarked architecture; any failure stops the run (a missing check is not a pass)."""
    scratch.mkdir(parents=True, exist_ok=True)
    tf32 = apply_tf32_policy()
    warm_libraries(dev, scratch)
    rank = reference_rank(cfg)
    res: Dict[str, Any] = {"rank": rank, "precision": "fp32", "tf32": tf32, "acceptance": ACCEPTANCE, "architectures": []}
    for arch in benchmark_architectures(cfg):
        print(f"[equivalence] {arch}", flush=True)
        res["architectures"].append(check_architecture(cfg, arch, rank, dev, scratch))
    res["all_accepted"] = all(c["verdict"] in ("reference", "accepted") for a in res["architectures"] for c in a["quality"]["candidates"])
    write_json(out_path, res)
    return res
