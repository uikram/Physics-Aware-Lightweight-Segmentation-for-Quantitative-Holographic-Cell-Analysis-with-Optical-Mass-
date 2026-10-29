"""Robustness to residual background offsets and low-order phase aberrations (inference only).

A distortion field D(x, y) in radians is added to the native phase map before the model's own preprocessing
(resize to the training resolution, per-image z-score):
    offset   D = c
    tilt     D = A (x - 0.5)
    bowl     D = A (r^2 / 0.5 - 0.5)                    (defocus-like, zero mean over the field)
    astig    D = A ((x - 0.5)^2 - (y - 0.5)^2) / 0.25
Reported against the annotation: macro Dice (4 classes, pooled), foreground IoU, and IPE computed on
(a) the true phase after plane background correction (segmentation robustness) and
(b) the distorted, uncorrected phase (sensitivity of the raw-phase IPE itself).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from qpi.analysis.common import load_validation, plane_correct


def distortion_fields(cfg: Dict, n: int = 768) -> Dict[str, np.ndarray]:
    r = cfg["analysis"]["robustness"]
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    r2 = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
    out = {"none": 0 * xx}
    for c in r["offsets_rad"]:
        out[f"offset{c:+.2f}"] = c + 0 * xx
    for a in r["amplitudes_rad"]:
        out[f"tilt{a:.2f}"] = a * (xx - 0.5)
        out[f"bowl{a:.2f}"] = a * (r2 / 0.5 - 0.5)
        out[f"astig{a:.2f}"] = a * ((xx - 0.5) ** 2 - (yy - 0.5) ** 2) / 0.25
    return out


@torch.no_grad()
def run_robustness(cfg: Dict, run_cfgs: Dict[str, Dict], run_dirs: Dict[str, Path], tag: str, device: torch.device,
                   weights_dir: Path, ctx: Dict | None = None) -> pd.DataFrame:
    """``ctx`` (optional dict) caches the validation data and the plane-corrected phase between calls."""
    from qpi.models import build_model                      # imported here: the result files can be read without the model code
    ctx = ctx if ctx is not None else {}
    if "val" not in ctx:
        ctx["val"] = load_validation(cfg)
        ev0 = cfg["evaluation"]
        ctx["corrected"] = {s: plane_correct(d["phase"], d["gt"] > 0, d["valid"], ev0["background_exclusion_px"])
                            for s, d in ctx["val"].items()}
    val, corrected = ctx["val"], ctx["corrected"]
    fields = distortion_fields(cfg)
    nc = cfg["data"]["num_classes"]
    rows: List[Dict] = []
    for run_id, rc in run_cfgs.items():
        model = build_model(rc, weights_dir, pretrained=False)
        state = torch.load(run_dirs[run_id] / "checkpoints" / f"{tag}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"], strict=True)
        model.to(device).eval()
        S = rc["image_size"]
        for fname, D in fields.items():
            tp, fp, fn = (np.zeros(nc) for _ in range(3))
            ious, ipe_true, ipe_seen = [], [], []
            for stem, d in val.items():
                ph = d["phase"] + D
                x = torch.from_numpy(ph).float()[None, None]
                x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False, antialias=S < x.shape[-1])
                x = ((x - x.mean()) / x.std().clamp(min=1e-8)).to(device)
                lg = F.interpolate(model(x).float(), size=ph.shape, mode="bilinear", align_corners=False)
                pred = np.where(d["valid"], lg.argmax(1)[0].cpu().numpy(), 0)
                gt = d["gt"]
                for c in range(1, nc):
                    tp[c] += ((pred == c) & (gt == c)).sum()
                    fp[c] += ((pred == c) & (gt != c) & d["valid"]).sum()
                    fn[c] += ((pred != c) & (gt == c)).sum()
                pb, gb = pred > 0, gt > 0
                ious.append((pb & gb).sum() / max((pb | gb).sum(), 1))
                ipe_true.append(abs((pb * corrected[stem]).sum() - (gb * corrected[stem]).sum()) / abs((gb * corrected[stem]).sum()))
                seen = np.where(d["valid"], ph, 0.0)
                ipe_seen.append(abs((pb * seen).sum() - (gb * seen).sum()) / abs((gb * seen).sum()))
            dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
            rows.append({"run_id": run_id, "distortion": fname, "macro_dice": float(dice[1:].mean()),
                         "fg_iou": float(np.mean(ious)), "ipe_bgcorr_true_pct": 100 * float(np.mean(ipe_true)),
                         "ipe_uncorrected_seen_pct": 100 * float(np.mean(ipe_seen))})
        del model
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ individual results (D-29)
import time  # noqa: E402

from qpi.utils.io import read_json, sha256_file, write_json  # noqa: E402
from qpi.utils.layout import rel  # noqa: E402
from qpi.utils.results_io import read_unit, write_unit  # noqa: E402

LEGACY_FIELDS = ("run_id", "distortion", "macro_dice", "fg_iou", "ipe_bgcorr_true_pct", "ipe_uncorrected_seen_pct")


def robustness_unit_path(store_dir: Path, run_id: str) -> Path:
    return Path(store_dir) / f"{run_id}.json"


def unit_rows(df: pd.DataFrame) -> List[Dict]:
    """Rows of one run with the change of macro Dice against the undistorted map and the by-construction flag.

    ``invariant_by_construction`` marks the constant-offset rows: the offset is added before the model's per-image z-score,
    which removes any constant, so an unchanged Dice there is a property of the preprocessing, not of the trained weights."""
    base = float(df.loc[df.distortion == "none", "macro_dice"].iloc[0])
    rows = []
    for r in df.round(5).to_dict("records"):
        rows.append({**r, "delta_macro_dice_vs_none": round(r["macro_dice"] - base, 5),
                     "invariant_by_construction": str(r["distortion"]).startswith("offset")})
    return rows


def unit_is_current(unit: Dict | None, tag: str, ckpt: Path, fields: List[str], trust_missing: bool = False):
    if unit is None:
        return False, "no result yet"
    if unit.get("checkpoint") != tag:
        return False, "checkpoint tag changed"
    if sorted(r["distortion"] for r in unit.get("rows", [])) != sorted(fields):
        return False, "distortion set changed"
    stored = unit.get("checkpoint_file", {}).get("sha256")
    if stored is None:                                        # migrated from the older robustness.json: no hash was recorded
        return True, ""
    if not ckpt.exists():
        return (True, "") if trust_missing else (False, f"{ckpt.name} not found (cannot verify)")
    if sha256_file(ckpt) != stored:
        return False, f"{ckpt.name} changed after the analysis"
    return True, ""


def run_robustness_units(cfg: Dict, refs, tag: str, device, weights_dir: Path, store_dir: Path, prog, label,
                         force: bool = False, trust_missing: bool = False) -> Dict[str, Dict]:
    """Robustness of every run in ``refs`` (cell_analysis.RunRef); finished runs are written one by one, current ones skipped."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    fields = list(distortion_fields(cfg))
    ctx: Dict = {}
    out: Dict[str, Dict] = {}
    for ref in refs:
        up = robustness_unit_path(store_dir, ref.run_id)
        ckpt = ref.run_dir / "checkpoints" / f"{tag}.pt"
        unit = None if force else read_unit(up)
        ok, why = unit_is_current(unit, tag, ckpt, fields, trust_missing) if unit is not None else (False, "forced" if force else "no result yet")
        if ok:
            out[ref.run_id] = unit
            prog.skipped(label(ref), "up to date")
            continue
        cj = ref.run_dir / "config.json"
        if not cj.exists() or not ckpt.exists():
            prog.fail(label(ref), f"run folder incomplete: {cj.name if not cj.exists() else ckpt.name} missing in {rel(ref.run_dir)}")
            continue
        prog.start(label(ref))
        t0 = time.time()
        try:
            rc = read_json(cj)["run"]
            df = run_robustness(cfg, {ref.run_id: rc}, {ref.run_id: ref.run_dir}, tag, device, weights_dir, ctx)
        except Exception as exc:                                  # one failing run must not stop the others
            prog.fail(label(ref), exc)
            continue
        unit = {"run_id": ref.run_id, "config": ref.config, "arch": ref.arch, "lora_rank": ref.rank, "loss": ref.loss,
                "seed": ref.seed, "checkpoint": tag, "status": "complete",
                "checkpoint_file": {"path": rel(ckpt), "bytes": int(ckpt.stat().st_size), "sha256": sha256_file(ckpt)},
                "rows": unit_rows(df)}
        write_unit(up, unit, sources=[ref.run_dir / "metrics.json"])
        out[ref.run_id] = read_unit(up)
        prog.done(label(ref), time.time() - t0, f"{len(df)} distortions")
    return out


def write_legacy_robustness(store_dir: Path, ids: List[str], tag: str, analysis_dir: Path) -> bool:
    """Rebuild ``robustness.csv`` / ``robustness.json`` of ``outputs/full/analysis`` (the file the older scripts read) from the
    individual results of ``ids``. Returns False (and writes nothing) when one of them is missing."""
    units = [read_unit(robustness_unit_path(store_dir, i)) for i in ids]
    if any(u is None for u in units):
        return False
    rows = [{k: r[k] for k in LEGACY_FIELDS} for u in units for r in u["rows"]]
    df = pd.DataFrame(rows)
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(analysis_dir / "robustness.csv", index=False)
    write_json(analysis_dir / "robustness.json", {"checkpoint": tag, "rows": df.round(5).to_dict("records")})
    return True
