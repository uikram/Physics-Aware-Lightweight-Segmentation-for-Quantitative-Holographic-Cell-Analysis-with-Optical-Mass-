"""Shared loaders for the revision artifacts (tables and figures). Read only: nothing under outputs/ is written.

Sources (all produced by the pipeline, see EXPERIMENTS.md and DECISIONS.md D-29):

    outputs/compiled/cell_analysis_multiseed.json   per-seed segmentation + cell / dry-mass metrics of all 63 configurations
                                                     (180 LoRA runs + 3 full fine-tuning runs), MANUSCRIPT class names
    outputs/compiled/segmentation_multiseed.json    the same segmentation metrics, LEGACY class names (cross-checked below)
    outputs/compiled/benchmark_multiseed.json       inference benchmark v5 (FP32 PyTorch and ONNX Runtime, 3 seeds)
    outputs/compiled/robustness_multiseed.json      background-perturbation robustness
    outputs/{full,multiseed}/runs/<id>/metrics.json mean IoU and per-class IoU (not in the compiled files), training info
    outputs/{full,multiseed}/runs/<id>/history.csv  per-epoch training curves
    outputs/analysis/cell/<id>.json                 per-run cell analysis (storage statistics, strata, reference sensitivity)
    outputs/cell_analysis/annotation/cells_gt.csv   annotation-derived cells;  outputs/cell_analysis/runs/<id>/cells_pred.csv

Class names (DECISIONS.md D-25): mask value 2 = spherocyte, 3 = echinocyte. The compiled cell file and the cell units already
use these names; metrics.json and the compiled segmentation file use the legacy names and are swapped here (classnames.swap_obj).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qpi.utils.classnames import CLASSES, swap_obj  # noqa: E402  (mask values 1..4: discocyte, spherocyte, echinocyte, stomatocyte)

OUT = ROOT / "outputs"
ART = ROOT / "artifacts"
TABLES = ART / "tables"
FIGURES = ART / "figures"

ARCHS = ["mobile_sam", "edge_sam", "mobilenet_unet"]
ARCH_LABEL = {"mobile_sam": "MobileSAM", "edge_sam": "EdgeSAM", "mobilenet_unet": "MobileNet-UNet"}
RANKS = [2, 4, 8, 16, 32]
SEEDS = [42, 43, 44]
LOSSES = ["dice_only", "pmc", "pmc_bga", "full"]
LOSS_LABEL = {"dice_only": "Dice only", "pmc": "Dice + PMC", "pmc_bga": "Dice + PMC + BGA", "full": "Dice + PMC + BGA + IPP (full)"}
# Class labels (author, confirmed): mask value 1 = discocyte, 2 = SPHEROCYTE, 3 = ECHINOCYTE, 4 = stomatocyte.
# Every artifact writes the mask value next to the name so that no table or figure can be read with the old (swapped) names.
MASK_VALUE = {"discocyte": 1, "spherocyte": 2, "echinocyte": 3, "stomatocyte": 4}
CLASS_SHORT = {c: f"{s}({MASK_VALUE[c]})" for c, s in
               {"discocyte": "disco", "spherocyte": "sphero", "echinocyte": "echino", "stomatocyte": "stomato"}.items()}
CLASS_DISPLAY = {c: f"{c} ({MASK_VALUE[c]})" for c in MASK_VALUE}
assert [c for c, _ in sorted(MASK_VALUE.items(), key=lambda kv: kv[1])] == list(CLASSES), "class order differs from qpi/utils/classnames.py"
SAME_RANK = 8
# Presentation choice of the professor (DECISIONS.md D-27, D-30, configs/views.yaml "preferred"); no rule selects them.
PREFERRED = [("mobile_sam", 8), ("edge_sam", 32)]
BASELINE = ("mobilenet_unet", 8)
COLLAPSE_DICE = 0.05      # a class with Dice below this counts as lost (same threshold as the aggregator)


# ------------------------------------------------------------------------------------------------ identifiers
def base_id(arch: str, rank: Optional[int], loss: str) -> str:
    if loss == "full_finetune":
        return f"{arch}_full_finetune"
    if loss == "full":
        return f"{arch}_lora_r{rank}"
    return f"{arch}_abl_{loss}" + ("" if rank == SAME_RANK else f"_r{rank}")


def config_key(arch: str, rank: Optional[int], loss: str) -> str:
    return f"{arch}|{base_id(arch, rank, loss)[len(arch) + 1:]}"


def in_main_grid(rank: Optional[int], loss: str) -> bool:
    """Configurations trained at seed 42 in outputs/full: the rank sweep, the rank-8 ablation and full fine-tuning."""
    return loss in ("full", "full_finetune") or rank == SAME_RANK


def run_id(arch: str, rank: Optional[int], loss: str, seed: int) -> str:
    b = base_id(arch, rank, loss)
    return b if (seed == 42 and in_main_grid(rank, loss)) else f"{b}__s{seed}"


def run_dir(arch: str, rank: Optional[int], loss: str, seed: int) -> Path:
    """Seed 42 of the main grid lives in outputs/full (the multiseed __s42 folders are hard-link mirrors of the same
    files); the loss ablations at ranks other than 8 exist only in outputs/multiseed, seed 42 included."""
    if seed == 42 and in_main_grid(rank, loss):
        return OUT / "full" / "runs" / base_id(arch, rank, loss)
    return OUT / "multiseed" / "runs" / run_id(arch, rank, loss, seed)


def all_lora_configs() -> List[tuple]:
    return [(a, r, l) for a in ARCHS for r in RANKS for l in LOSSES]


# ------------------------------------------------------------------------------------------------ files
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p: Path) -> Any:
    return json.loads(Path(p).read_text())


@lru_cache(maxsize=None)
def compiled(name: str) -> Dict[str, Any]:
    d = read_json(OUT / "compiled" / f"{name}_multiseed.json")
    if not d.get("complete") or d.get("problems"):
        raise RuntimeError(f"compiled/{name}_multiseed.json is not complete: {d.get('problems')}")
    return d


def cell_configs() -> Dict[str, Any]:
    return compiled("cell_analysis")["configurations"]


def cfg_entry(arch: str, rank: Optional[int], loss: str) -> Dict[str, Any]:
    return cell_configs()[config_key(arch, rank, loss)]


def metric(arch: str, rank: Optional[int], loss: str, name: str) -> Dict[str, Any]:
    """{'mean','sd','min','max','n','per_seed'} of a metric in the compiled cell file (manuscript class names)."""
    return cfg_entry(arch, rank, loss)["metrics"][name]


@lru_cache(maxsize=None)
def metrics_json(arch: str, rank: Optional[int], loss: str, seed: int) -> Dict[str, Any]:
    """metrics.json of one run with the class names swapped to the manuscript order."""
    return swap_obj(read_json(run_dir(arch, rank, loss, seed) / "metrics.json"))


@lru_cache(maxsize=None)
def cell_unit(arch: str, rank: Optional[int], loss: str, seed: int) -> Dict[str, Any]:
    return read_json(OUT / "analysis" / "cell" / f"{run_id(arch, rank, loss, seed)}.json")


def history(arch: str, rank: Optional[int], loss: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(run_dir(arch, rank, loss, seed) / "history.csv")


def cells_gt(nowrap: bool = True) -> pd.DataFrame:
    g = pd.read_csv(OUT / "cell_analysis" / "annotation" / "cells_gt.csv")
    return g[~g.wrapped.astype(bool)] if nowrap else g


def cells_pred(arch: str, rank: Optional[int], loss: str, seed: int, nowrap: bool = True) -> pd.DataFrame:
    p = pd.read_csv(OUT / "cell_analysis" / "runs" / run_id(arch, rank, loss, seed) / "cells_pred.csv")
    return p[~p.wrapped.astype(bool)] if nowrap else p


# ------------------------------------------------------------------------------------------------ per-run table
def runs_frame() -> pd.DataFrame:
    """One row per trained run (180 LoRA + 3 full fine-tuning): segmentation metrics (manuscript names), IoU, lost classes."""
    rows = []
    for key, c in cell_configs().items():
        arch, rank, loss = c["arch"], c["lora_rank"], c["loss"]
        for s in c["seeds_found"]:
            m = c["metrics"]
            ev = metrics_json(arch, rank, loss, s)["evaluation"]["last"]
            r: Dict[str, Any] = {"config": key, "arch": arch, "rank": rank, "loss": loss, "seed": s,
                                 "run_id": run_id(arch, rank, loss, s),
                                 "trainable_params": metrics_json(arch, rank, loss, s)["parameters"]["trainable"],
                                 "total_params": metrics_json(arch, rank, loss, s)["parameters"]["total"],
                                 "epochs": metrics_json(arch, rank, loss, s)["training"]["epochs_completed"],
                                 "train_seconds": metrics_json(arch, rank, loss, s)["training"]["train_seconds"],
                                 "mean_iou": ev["mean_iou"]}
            for name in ("mean_dice", "aji", "bf1", "ipe_global_pct"):
                r[name] = m[name]["per_seed"][str(s)]
            for cl in CLASSES:
                r[f"dice_{cl}"] = m[f"dice_{cl}"]["per_seed"][str(s)]
                r[f"iou_{cl}"] = ev[f"iou_{cl}"]
                v = ev.get(f"ipe_{cl}")
                r[f"ipe_{cl}_pct"] = None if v is None or (isinstance(v, float) and math.isnan(v)) else 100 * v
            # consistency: metrics.json (swapped) must give the same Dice as the compiled file
            for name in ["mean_dice", "aji", "bf1"] + [f"dice_{cl}" for cl in CLASSES]:
                if abs(ev[name] - r[name]) > 1e-9:
                    raise RuntimeError(f"{r['run_id']}: {name} differs between metrics.json and the compiled file")
            lost = [cl for cl in CLASSES if not r[f"dice_{cl}"] >= COLLAPSE_DICE]
            r["lost_classes"] = " ".join(lost)
            r["n_lost"] = len(lost)
            rows.append(r)
    df = pd.DataFrame(rows)
    n_lora = int((df.loss != "full_finetune").sum())
    if n_lora != 180 or int((df.loss == "full_finetune").sum()) != 3:
        raise RuntimeError(f"expected 180 LoRA and 3 full fine-tuning runs, found {n_lora} and {int((df.loss == 'full_finetune').sum())}")
    return df.sort_values(["arch", "loss", "rank", "seed"], na_position="last").reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ statistics
def mean_sd(vals: Iterable[Optional[float]]) -> tuple:
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return (None, None, 0)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else None, len(v))


def paired_diff(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, Any]:
    """b minus a over the common seeds; 95% t-interval of the seed-paired differences (wide with n = 3)."""
    seeds = sorted(set(a) & set(b))
    d = np.array([b[s] - a[s] for s in seeds], float)
    out = {"n": len(d), "mean": float(d.mean()) if len(d) else None, "lo": None, "hi": None,
           "n_b_greater": int((d > 0).sum())}
    if len(d) > 1:
        h = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))
        out["lo"], out["hi"] = out["mean"] - h, out["mean"] + h
    out["excludes_zero"] = None if out["lo"] is None else not (out["lo"] <= 0 <= out["hi"])
    return out


# ------------------------------------------------------------------------------------------------ formatting
def fmt(x: Optional[float], nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def pm(mean: Optional[float], sd: Optional[float], nd: int = 3) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.{nd}f}" if sd is None else f"{mean:.{nd}f} ± {sd:.{nd}f}"


def pm_metric(m: Dict[str, Any], nd: int = 3, scale: float = 1.0) -> str:
    sd = None if m["sd"] is None else m["sd"] * scale
    return pm(None if m["mean"] is None else m["mean"] * scale, sd, nd)


def fmt_diff(p: Dict[str, Any], nd: int = 3) -> str:
    if p["mean"] is None:
        return "n/a"
    ci = "" if p["lo"] is None else f" [{p['lo']:+.{nd}f}, {p['hi']:+.{nd}f}]"
    return f"{p['mean']:+.{nd}f}{ci}"


def params_m(x: int) -> str:
    return f"{x / 1e6:.2f} M"


def lost_text(arch: str, rank: Optional[int], loss: str) -> str:
    """'0/3' or '1/3 (s44: spherocyte)' from the per-seed class Dice of the compiled file."""
    c = cfg_entry(arch, rank, loss)
    parts = []
    for s in c["seeds_found"]:
        lost = [CLASS_SHORT[cl] for cl in CLASSES if not c["metrics"][f"dice_{cl}"]["per_seed"][str(s)] >= COLLAPSE_DICE]
        if lost:
            parts.append(f"s{s}: {', '.join(lost)}")
    k, n = len(parts), len(c["seeds_found"])
    return f"{k}/{n}" + (f" ({'; '.join(parts)})" if parts else "")


def n_lost_runs(arch: str, rank: Optional[int], loss: str) -> int:
    c = cfg_entry(arch, rank, loss)
    return sum(1 for s in c["seeds_found"]
               if any(not c["metrics"][f"dice_{cl}"]["per_seed"][str(s)] >= COLLAPSE_DICE for cl in CLASSES))


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and math.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(out)
