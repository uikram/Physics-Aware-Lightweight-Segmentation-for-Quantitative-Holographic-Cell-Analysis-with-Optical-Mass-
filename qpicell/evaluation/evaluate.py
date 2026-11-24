"""Evaluate one trained run on the TEST images (validation split, 66 images), for each checkpoint (last = primary,
best = max validation Dice on fold 0 of the training split).

Inference: the network sees only its z-scored input; logits at the model size -> bilinear to the native 768 grid ->
arg-max. Instances are reconstructed from the predicted class map alone (A2 primary, CC diagnostic). Ground truth is
read only for scoring. Every phase sum uses the float32 referenced phase in radians (checked against the checksum of
data preparation before and after scoring).

results/<Arch>/<condition>/<loss>/seed<seed>/
  metrics_<tag>.json        metadata + semantic + per-instance-step quantitative results
  cells_<tag>.json          one record per reference cell and instance step (A2, CC)
  per_image_<tag>.json      one record per test image
  predictions_<tag>.npz     predicted class maps (uint8, 768 x 768)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qpicell.config import config_hash
from qpicell.data.dataset import PhaseDataset, read_manifest
from qpicell.evaluation.cells import evaluate_image, summarise
from qpicell.evaluation.instances import reconstruct
from qpicell.evaluation.quant import (cluster_summary, exclusion_reasons, group_metrics, oversplit_rows, oversplit_summary,
                                      phase_accounting, touching_and_clusters)
from qpicell.evaluation.segmentation import SegmentationScores, aji_sums
from qpicell.io import write_json
from qpicell.physics import assert_phase, dry_mass_factor_pg, optical_volume_factor_um3, sha256_array, split_stems

MODES = ("A2", "CC")


def prepare_cfg(cfg: Dict) -> Dict:
    cfg = dict(cfg)
    cfg["_k_pg"] = dry_mass_factor_pg(cfg); cfg["_k_ov"] = optical_volume_factor_um3(cfg); cfg["_px_area"] = cfg["physics"]["pixel_size_um"] ** 2
    return cfg


@torch.no_grad()
def predict(model, dataset: PhaseDataset, batch_size: int, device) -> Dict[str, Dict]:
    model.eval().to(device)
    out = {}
    for b in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        logits = model(b["input"].to(device).float()).float()            # the network sees only its input
        N = b["phase_native"].shape[-1]
        if logits.shape[-1] != N:
            logits = F.interpolate(logits, size=(N, N), mode="bilinear", align_corners=False)
        pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
        for i, sid in enumerate(b["sample_id"]):
            ph = b["phase_native"][i, 0].numpy()
            out[sid] = dict(pred=pred[i], phase=ph, mask=b["mask_native"][i].numpy().astype(np.uint8), cells=b["cells_native"][i].numpy(),
                            wrapped=b["wrapped_native"][i].numpy(), day=int(b["storage_day"][i]))
    return out


def score_predictions(cfg: Dict, preds: Dict[str, Dict], manifest_sha: Dict[str, str]) -> Dict:
    cfg = prepare_cfg(cfg)
    h = int(cfg["cells"]["watershed_h"]); ign = int(cfg["data"]["ignore_index"])
    lo, hi = cfg["cells"]["area_range_um2"]; lo_px = lo / cfg["_px_area"]
    seg = SegmentationScores(cfg)
    acc = {m: {"cells": [], "comps": [], "imgs": [], "over": [], "aji": [0.0, 0.0], "assign": []} for m in MODES}
    per_image = []
    for sid, d in preds.items():
        ph32 = d["phase"]; assert_phase(ph32, sid)
        if manifest_sha.get(sid) and sha256_array(ph32) != manifest_sha[sid]:
            raise AssertionError(f"{sid}: phase map differs from data preparation (checksum)")
        before = sha256_array(ph32)
        phase = ph32.astype(np.float64); mask, cells, wr, pred = d["mask"], d["cells"].astype(np.int64), d["wrapped"].astype(bool), d["pred"]
        valid = mask != ign
        row = seg.add(pred, mask, phase, sid, d["day"])
        labs, checks = reconstruct(pred, h)                                 # from the prediction only
        touch, clus, K, nK = touching_and_clusters(cells, valid)
        ref = np.where(valid, cells, 0)
        for m in MODES:
            cr, pr, ir = evaluate_image(cfg, pred, phase, mask, cells, wr, sid, d["day"], labels=labs[m])
            for r in cr:
                r["touching"] = touch.get(r["cell"], False); r["cluster"] = clus.get(r["cell"], 0)
            lab_v = np.where(valid, labs[m], 0)
            acc[m]["cells"] += cr; acc[m]["comps"] += pr
            acc[m]["imgs"].append({**ir, **phase_accounting(phase, pred, mask, cells, valid, touch, lab_v)})
            acc[m]["over"] += oversplit_rows(lab_v, K, nK, ref, sid, lo_px)
            i, u = aji_sums(lab_v, ref); acc[m]["aji"][0] += i; acc[m]["aji"][1] += u
            acc[m]["assign"].append(checks[m])
        per_image.append({**row, "a2_instances": checks["A2"]["n_instances"], "cc_instances": checks["CC"]["n_instances"]})
        if sha256_array(ph32) != before:
            raise AssertionError(f"{sid}: phase map modified during evaluation")
    semantic = seg.summary()
    out = {"semantic": semantic, "instances": {}, "per_image": per_image, "cells_tables": {}}
    for m in MODES:
        C = pd.DataFrame(acc[m]["cells"]); I = pd.DataFrame(acc[m]["imgs"]); O = pd.DataFrame(acc[m]["over"])
        for col in ("e_matched", "dm_pred_pg", "ov_pred_um3", "S_pred"):
            if col not in C:
                C[col] = np.nan
        prod = summarise(acc[m]["cells"], acc[m]["comps"], acc[m]["imgs"])
        S = C[C.scored]; within = float(cfg["evaluation"]["within_pct"]) / 100
        Sa = I.S_annotated.sum()
        res = {
            "assignment": {"fg_px": int(sum(a["fg_px"] for a in acc[m]["assign"])),
                           "unassigned_fg_px": int(sum(a["unassigned_fg_px"] for a in acc[m]["assign"])),
                           "labels_outside_fg_px": int(sum(a["labels_outside_fg_px"] for a in acc[m]["assign"])),
                           "n_instances": int(sum(a["n_instances"] for a in acc[m]["assign"]))},
            "aji_instances_vs_reference_cells": acc[m]["aji"][0] / acc[m]["aji"][1] if acc[m]["aji"][1] else None,
            "detection_all_reference_cells": {"n": len(C), "match_pct": 100 * C.matched.mean(), "missed_pct": 100 * C.missed.mean(),
                                              "merged_pct": 100 * C.merged.mean(), "fragmented_pct": 100 * C.fragmented.mean()},
            "cells": {"scored": group_metrics(S, within), "touching": group_metrics(S[S.touching], within),
                      "isolated": group_metrics(S[~S.touching], within)},
            "clusters": cluster_summary(C), "oversplit": oversplit_summary(O),
            "n_predicted_instances": prod["n_predicted_components"], "n_false_positive_instances": prod["n_false_positive_components"],
            "dry_mass": {"matched": prod.get("drymass_matched"), "per_image_median": prod.get("drymass_all_cells_image_median"),
                         "population": prod.get("drymass_all_cells_population")},
            "phase": {"global_ipe_pooled_signed_pct": 100 * (I.S_pred.sum() - I.S_gt.sum()) / abs(I.S_gt.sum()),
                      "retained_all_pct": 100 * I.S_true_positive.sum() / Sa,
                      "retained_touching_pct": 100 * I.S_retained_touching.sum() / I.S_annotated_touching.sum() if I.S_annotated_touching.sum() else None,
                      "retained_isolated_pct": 100 * I.S_retained_isolated.sum() / I.S_annotated_isolated.sum() if I.S_annotated_isolated.sum() else None,
                      "lost_by_network_pct": 100 * (Sa - I.S_true_positive.sum()) / Sa,
                      "extra_outside_annotation_pct": 100 * (I.S_predicted_fg.sum() - I.S_true_positive.sum()) / Sa,
                      "fg_px_missed_by_network": int(I.px_fg_missed_by_network.sum()),
                      "fg_px_unassigned_by_instance_step": int(I.px_unassigned_by_instances.sum()),
                      "unassigned_by_instance_step_pct": 100 * I.S_unassigned_by_instances.sum() / Sa}}
        out["instances"][m] = res
        out["cells_tables"][m] = C
    # reference cells: all / scored / excluded with reasons (independent of the instance step)
    C = out["cells_tables"]["A2"]
    exc = C[~C.scored]; reasons: Dict[str, int] = {}
    for _, r in exc.iterrows():
        for x in exclusion_reasons(r, lo, hi):
            reasons[x] = reasons.get(x, 0) + 1
    out["reference_cells"] = {"all": len(C), "scored": int(C.scored.sum()), "scored_touching": int((C.scored & C.touching).sum()),
                              "scored_isolated": int((C.scored & ~C.touching).sum()), "excluded": len(exc),
                              "excluded_touching": int(exc.touching.sum()), "excluded_isolated": int((~exc.touching).sum()),
                              "exclusion_reasons_non_exclusive": reasons}
    return out


def evaluate_checkpoint(cfg: Dict, model, ckpt: Path, arch: str, device, out_dir: Path, tag: str,
                        run_info: Optional[Dict] = None, model_info: Optional[Dict] = None) -> Dict:
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"], strict=True)
    m = cfg["models"][arch]; stems = split_stems(cfg)["test"]
    ds = PhaseDataset(cfg, "val", m["image_size"], cfg["data"]["eval_rotations"], return_native=True, stems=stems)
    sha = {r["stem"]: r.get("phase_sha256") for r in read_manifest(cfg["paths_abs"]["prepared"], "val")}
    t0 = time.time(); preds = predict(model, ds, m["batch_size"], device); t_inf = time.time() - t0
    res = score_predictions(cfg, preds, sha)
    tables = res.pop("cells_tables"); per_image = res.pop("per_image")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"run": run_info or {}, "architecture": arch, "architecture_display": m["display"], "checkpoint": tag,
            "checkpoint_epoch": int(state.get("epoch", -1)), "checkpoint_file": str(ckpt),
            "checkpoint_selection": "last epoch" if tag == "last" else "max validation macro Dice on fold 0 of the training split",
            "evaluation_split": "test (validation split, 66 images; never used for training or checkpoint selection)",
            "n_images": len(preds), "instances_primary": cfg["evaluation"]["instances_primary"],
            "instances_secondary": cfg["evaluation"]["instances_secondary"], "watershed_h": int(cfg["cells"]["watershed_h"]),
            "match_iou": cfg["evaluation"]["match_iou"], "area_range_um2": cfg["cells"]["area_range_um2"],
            "physics": cfg["physics"], "class_names": cfg["data"]["class_names"], "config_hash": config_hash(cfg),
            "model": model_info or {}, "inference_seconds": t_inf, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(out_dir / f"metrics_{tag}.json", {"metadata": meta, **res})
    cells = []
    for mode, C in tables.items():
        cells += [{"instances": mode, **r} for r in C.to_dict("records")]
    write_json(out_dir / f"cells_{tag}.json", {"metadata": meta, "cells": cells})
    write_json(out_dir / f"per_image_{tag}.json", {"metadata": meta, "images": per_image})
    np.savez_compressed(out_dir / f"predictions_{tag}.npz", **{k: v["pred"] for k, v in preds.items()})
    return {"metadata": meta, **res}
