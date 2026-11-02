"""Evaluation of a trained checkpoint at the native 768 x 768 resolution.

Logits are bilinearly resized to the native grid before the arg-max, so every metric (Dice, AJI, BF1, IPE) and
every downstream measurement (per-cell IPE, dry mass) is computed on the original pixel grid and raw phase.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qpi.data.dataset import QPIDataset
from qpi.evaluation.metrics import SegmentationMetrics


@torch.no_grad()
def evaluate_checkpoint(model: torch.nn.Module, ckpt_path: Path, run_cfg: Dict, cfg: Dict, device: torch.device,
                        out_dir: Path, tag: str) -> Dict:
    """Load ``ckpt_path``, predict the validation images and write predictions_<tag>.npz / per_image_<tag>.csv."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).eval()
    d = cfg["data"]
    ds = QPIDataset(cfg["paths_abs"]["processed_root"], "val", run_cfg["image_size"], d["val_rotations"],
                    d["ignore_index"], d["max_val_samples"], return_native=True)
    loader = DataLoader(ds, batch_size=run_cfg["batch_size"], shuffle=False, num_workers=0)
    metrics = SegmentationMetrics(run_cfg["num_classes"], d["class_names"], cfg["evaluation"]["boundary_tolerance_px"],
                                  d["ignore_index"])
    preds: Dict[str, np.ndarray] = {}
    for batch in loader:
        x = batch["phase"].to(device).float()
        logits = model(x).float()
        native = batch["phase_native"].shape[-1]
        if logits.shape[-1] != native:
            logits = F.interpolate(logits, size=(native, native), mode="bilinear", align_corners=False)
        pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
        for i in range(pred.shape[0]):
            sid = batch["sample_id"][i]
            preds[sid] = pred[i]
            metrics.update(pred[i], batch["mask_native"][i].numpy(), batch["phase_native"][i, 0].numpy(), sid,
                           int(batch["storage_day"][i]))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"predictions_{tag}.npz", **preds)
    rows = metrics.per_image
    with open(out_dir / f"per_image_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = metrics.compute()
    res["checkpoint_epoch"] = int(state.get("epoch", -1))
    return res
