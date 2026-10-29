"""Per-cell analyses of every finished run, from the saved predictions. No GPU, no training.

For each run (architecture x LoRA rank x seed, full objective by default) the saved ``predictions_<checkpoint>.npz`` is turned
into a per-cell table with the code path of the selected-run analysis (``qpi.analysis.drymass``, ``phase_integral``,
``wrapping``), and the following is stored per run:

  segmentation   Dice, AJI, boundary F1, global IPE and per-class Dice from the run's metrics.json (as the aggregator reads them)
  cells          cells found (all / not phase-wrapped), against the annotation, per-class counts, composition error
  dry_mass       per-image agreement (r, CCC, bias, limits of agreement, MdARE), day-0 median, per-cell error and matched fraction
  storage        Spearman trends of area and dry mass with storage day, early versus late, per-day tables
  phase_integral the full ``phase_integral.analyse_run`` summary, per-class summary and error strata
  background     residual background level that dry mass subtracts after the per-image reference (should be close to 0)
  reference_sensitivity  global IPE with the phase referenced to the annotated background, to the predicted background, and not at all

The annotation itself is analysed once (``annotation``), together with the per-image background offsets removed in
preprocessing (``phase_offsets``). Class names are written in the MANUSCRIPT order (DECISIONS.md D-25: mask value 2 =
spherocyte, 3 = echinocyte); nothing read from a file with legacy names is swapped here, the names are applied directly.

Run layout read (never written): seed 42 of a main-grid run in ``<main_root>/runs/<id>``, other seeds in
``<multiseed_root>/runs/<id>__s<seed>``; the loss ablation at ranks other than the selected one only exists in the multi-seed root.
"""
from __future__ import annotations

import copy
import math
import multiprocessing as mp
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from qpi.analysis import drymass as dm
from qpi.analysis import phase_integral as pi_mod
from qpi.analysis import wrapping as wrp
from qpi.analysis.common import background_level, load_predictions
from qpi.utils.classnames import CLASSES, MANUSCRIPT_NAMES, legacy_name
from qpi.utils.io import read_json, sha256_file
from qpi.utils.layout import ROOT, rel
from qpi.utils.results_io import read_unit, write_unit

SCHEMA_VERSION = 1
LOSSES = ("full", "dice_only", "pmc", "pmc_bga")
COLLAPSE_DICE = 0.05                      # same threshold as qpi.analysis.multiseed
DATE_STEM = re.compile(r"^\d{8}_\d+$")


# ------------------------------------------------------------------------------------------------ which runs
@dataclass(frozen=True)
class RunRef:
    run_id: str
    config: str            # e.g. "edge_sam|lora_r32" (same keys as the multi-seed aggregator)
    arch: str
    rank: Optional[int]    # None -> full fine-tuning
    loss: str              # full | dice_only | pmc | pmc_bga | full_finetune
    seed: int
    run_dir: Path


def manuscript_cfg(cfg: Dict) -> Dict:
    """Copy of the configuration whose class names are the manuscript order (D-25); the original is not touched."""
    c = copy.deepcopy(cfg)
    c["data"]["class_names"] = list(MANUSCRIPT_NAMES)
    return c


def locate_run(cfg: Dict, main_root: Path, ms_root: Path, arch: str, rank: Optional[int], loss: str, seed: int) -> Optional[RunRef]:
    """Where the run of (architecture, rank, loss, seed) lives, or None when no such run exists by design."""
    sel, main_seed = int(cfg["lora"]["selected_rank"]), int(cfg["seed"])
    main_possible = True
    if loss == "full_finetune":
        if seed != main_seed:
            return None                                   # full fine-tuning is a single seed-42 run (D-B)
        base, label, rank = f"{arch}_full_finetune", "full_finetune", None
    elif loss == "full":
        base, label = f"{arch}_lora_r{rank}", f"lora_r{rank}"
    elif loss in LOSSES:
        if rank == sel:
            base, label = f"{arch}_abl_{loss}", f"abl_{loss}"
        else:
            base, label, main_possible = f"{arch}_abl_{loss}_r{rank}", f"abl_{loss}_r{rank}", False
    else:
        raise ValueError(f"unknown loss {loss!r}")
    if main_possible and seed == main_seed:
        rid, root = base, main_root
    else:
        rid, root = f"{base}__s{seed}", ms_root
    return RunRef(rid, f"{arch}|{label}", arch, rank, loss, int(seed), Path(root) / "runs" / rid)


def expected_runs(cfg: Dict, main_root: Path, ms_root: Path, archs: Sequence[str], ranks: Sequence[int],
                  seeds: Sequence[int], losses: Sequence[str] = ("full",), include_full_finetune: bool = False) -> List[RunRef]:
    refs: List[RunRef] = []
    for a in archs:
        for loss in losses:
            for r in ranks:
                for s in seeds:
                    ref = locate_run(cfg, main_root, ms_root, a, int(r), loss, int(s))
                    if ref is not None:
                        refs.append(ref)
        if include_full_finetune:
            for s in seeds:
                ref = locate_run(cfg, main_root, ms_root, a, None, "full_finetune", int(s))
                if ref is not None:
                    refs.append(ref)
    ids = [r.run_id for r in refs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run ids in the requested scope")
    return refs


# ------------------------------------------------------------------------------------------------ small helpers
def fingerprint(p: Path, deep: bool = True) -> Dict[str, Any]:
    st = p.stat()
    out = {"file": p.name, "bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns),
           "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
    if deep:
        out["sha256"] = sha256_file(p)
    return out


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pct(x: Any) -> Optional[float]:
    v = _f(x)
    return None if v is None else 100.0 * v


def composition_error(pred: pd.DataFrame, gt: pd.DataFrame, num_classes: int) -> Dict:
    """Mean and maximum absolute difference (percentage points) between predicted and annotated class composition per day.

    Same definition as scripts/compare_architectures.py (a unit test keeps the two identical)."""
    diffs: List[float] = []
    per_day: Dict[int, float] = {}
    for day in sorted(set(gt.storage_day) | set(pred.storage_day)):
        p, g = pred[pred.storage_day == day], gt[gt.storage_day == day]
        if not len(p) or not len(g):
            continue
        d = [abs(100 * (p.class_idx == c).mean() - 100 * (g.class_idx == c).mean()) for c in range(1, num_classes)]
        per_day[int(day)] = float(np.mean(d))
        diffs += d
    return {"mean_abs_pp": float(np.mean(diffs)) if diffs else None, "max_abs_pp": float(np.max(diffs)) if diffs else None,
            "per_day_mean_abs_pp": per_day}


def class_count_ratio(pred: pd.DataFrame, gt: pd.DataFrame, names: List[str]) -> Dict:
    out = {}
    for c in range(1, len(names)):
        n_p, n_g = int((pred.class_idx == c).sum()), int((gt.class_idx == c).sum())
        out[names[c]] = {"predicted": n_p, "annotated": n_g, "ratio": (n_p / n_g) if n_g else None}
    return out


def _storage(cfg: Dict, cells: pd.DataFrame) -> Optional[Dict]:
    return dm.storage_statistics(cfg, cells) if len(cells) else None


def _per_day(cfg: Dict, cells: pd.DataFrame) -> List[Dict]:
    return dm.per_day_table(cfg, cells) if len(cells) else []


def background_residual(cells: pd.DataFrame) -> Dict[str, Any]:
    """Background level that the dry-mass step still subtracts, per image (one value per image; radians)."""
    if not len(cells):
        return {"n_images": 0}
    r = cells.groupby("stem")["background_rad"].first().abs()
    return {"n_images": int(len(r)), "median_abs_rad": float(r.median()), "p90_abs_rad": float(r.quantile(0.9)),
            "max_abs_rad": float(r.max())}


# ------------------------------------------------------------------------------------------------ the annotation, once
def phase_offsets(val: Dict[str, Dict]) -> Dict[str, Any]:
    """Per-image offset (radians) that preprocessing subtracted to reference each map to its own background (D-03)."""
    groups: Dict[str, List[float]] = {"date_named": [], "X_prefixed": [], "other": []}
    per_image = {}
    for stem, d in val.items():
        g = "date_named" if DATE_STEM.match(stem) else ("X_prefixed" if stem.startswith("X_") else "other")
        groups[g].append(float(d["offset"]))
        per_image[stem] = {"storage_day": int(d["day"]), "group": g, "offset_rad": float(d["offset"])}
    summ = {}
    for g, v in groups.items():
        if v:
            a = np.asarray(v)
            summ[g] = {"n_images": int(len(a)), "min_rad": float(a.min()), "median_rad": float(np.median(a)),
                       "mean_rad": float(a.mean()), "max_rad": float(a.max())}
    return {"note": "validation images; offset = median phase of annotated background (more than background_exclusion_px from any "
                    "cell) of the map as provided, subtracted once in prepare_dataset (D-03)",
            "by_group": summ, "per_image": per_image}


def annotation_block(cfg: Dict, cells_gt: pd.DataFrame, val: Dict[str, Dict]) -> Dict[str, Any]:
    names = cfg["data"]["class_names"]
    nw = cells_gt[~cells_gt.wrapped]
    return {
        "n_cells_all": int(len(cells_gt)), "n_cells_nowrap": int(len(nw)),
        "wrapping": wrp.summarise(cells_gt, names),
        "per_day_all": _per_day(cfg, cells_gt), "per_day_nowrap": _per_day(cfg, nw),
        "storage_all": _storage(cfg, cells_gt), "storage_nowrap": _storage(cfg, nw),
        "consistency_nowrap": dm.consistency_checks(cfg, nw), "consistency_all": dm.consistency_checks(cfg, cells_gt),
        "background_residual": background_residual(cells_gt),
        "phase_offsets": phase_offsets(val),
    }


def reference_sensitivity(cfg: Dict, val: Dict[str, Dict], preds: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Global IPE under three phase references (Q-2, DECISIONS.md D-03). No retraining is involved, only the reference changes.

    annotation_background   the reference of the experiments: median phase of the annotated background (equals the processed map)
    prediction_background   the same rule with the PREDICTED foreground, i.e. what is available when no annotation exists
    as_provided             no reference at all: the map exactly as supplied, offset included
    Also the per-image difference of the two background levels (prediction minus annotation, radians)."""
    excl = int(cfg["evaluation"]["background_exclusion_px"])
    ipe: Dict[str, List[float]] = {"annotation_background": [], "prediction_background": [], "as_provided": []}
    diff: List[float] = []
    for stem, d in val.items():
        valid = d["valid"]
        g = d["gt"] > 0
        p = np.where(valid, preds[stem], 0) > 0
        raw = d["phase"] + d["offset"]                                     # the map as supplied
        bg_g, bg_p = background_level(raw, g, valid, excl), background_level(raw, p, valid, excl)
        if not (np.isfinite(bg_g) and np.isfinite(bg_p)):
            continue
        diff.append(bg_p - bg_g)
        for name, ph in (("annotation_background", raw - bg_g), ("prediction_background", raw - bg_p), ("as_provided", raw)):
            sg, sp = float((g * ph).sum()), float((p * ph).sum())
            if abs(sg) > 1e-8:
                ipe[name].append(abs(sp - sg) / abs(sg))
    a = np.abs(np.asarray(diff)) if diff else np.asarray([])
    return {"n_images": len(diff),
            "ipe_global_pct": {k: (100.0 * float(np.mean(v)) if v else None) for k, v in ipe.items()},
            "background_difference_prediction_minus_annotation_rad": {
                "median_signed": float(np.median(diff)) if diff else None,
                "median_abs": float(np.median(a)) if diff else None, "max_abs": float(a.max()) if diff else None}}


# ------------------------------------------------------------------------------------------------ one run
def segmentation_block(m: Dict, ckpt: str) -> Dict[str, Any]:
    ev = m["evaluation"][ckpt]
    dice = {c: _f(ev.get(f"dice_{legacy_name(c)}")) for c in CLASSES}       # metrics.json carries the legacy names
    ipe = {c: _pct(ev.get(f"ipe_{legacy_name(c)}")) for c in CLASSES}
    collapsed = [c for c in CLASSES if not (dice[c] is not None and dice[c] >= COLLAPSE_DICE)]
    return {"mean_dice": _f(ev["mean_dice"]), "aji": _f(ev["aji"]), "bf1": _f(ev["bf1"]), "ipe_global_pct": _pct(ev["ipe_global"]),
            "dice_by_class": dice, "ipe_by_class_pct": ipe, "collapsed_classes": collapsed, "n_collapsed": len(collapsed),
            "trainable_params": m.get("parameters", {}).get("trainable"), "total_params": m.get("parameters", {}).get("total"),
            "epochs_completed": m.get("training", {}).get("epochs_completed")}


def analyse_run(cfg: Dict, val: Dict[str, Dict], cells_gt: pd.DataFrame, ref: RunRef, tag: str,
                pi_dir: Optional[Path] = None, csv_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Analysis of one run. ``cfg`` must already carry the manuscript class names (``manuscript_cfg``)."""
    base = {"run_id": ref.run_id, "config": ref.config, "arch": ref.arch, "lora_rank": ref.rank, "loss": ref.loss,
            "seed": ref.seed, "run_dir": str(ref.run_dir)}
    mp_, pp = ref.run_dir / "metrics.json", ref.run_dir / f"predictions_{tag}.npz"
    if not mp_.exists():
        return {**base, "status": "missing", "reason": f"no metrics.json in {ref.run_dir}"}
    m = read_json(mp_)
    if m.get("status") != "complete":
        return {**base, "status": str(m.get("status")), "reason": f"run status is {m.get('status')}"}
    if not pp.exists():
        return {**base, "status": "missing", "reason": f"no {pp.name} in {ref.run_dir}"}
    names, nc = cfg["data"]["class_names"], cfg["data"]["num_classes"]
    preds = load_predictions(ref.run_dir, tag)
    cells_pred = dm.cells_table(cfg, val, preds)
    p_nw, g_nw = cells_pred[~cells_pred.wrapped], cells_gt[~cells_gt.wrapped]
    d0 = p_nw[p_nw.storage_day == 0]
    entry: Dict[str, Any] = {
        **base, "status": "complete", "checkpoint": tag,
        "metrics_json": fingerprint(mp_, deep=False), "predictions": fingerprint(pp, deep=True),
        "segmentation": segmentation_block(m, tag),
        "cells": {
            "predicted_all": int(len(cells_pred)), "predicted_nowrap": int(len(p_nw)), "annotated_nowrap": int(len(g_nw)),
            "cells_ratio_nowrap": float(len(p_nw) / len(g_nw)) if len(g_nw) else None,
            "wrapping": wrp.summarise(cells_pred, names),
            "class_counts_nowrap": class_count_ratio(p_nw, g_nw, names),
            "class_counts_all": class_count_ratio(cells_pred, cells_gt, names),
            "composition_error_nowrap": composition_error(p_nw, g_nw, nc),
            "composition_error_all": composition_error(cells_pred, cells_gt, nc),
        },
        "dry_mass": {
            "agreement_nowrap": dm.agreement(p_nw, g_nw) if len(p_nw) and len(g_nw) else None,
            "agreement_all": dm.agreement(cells_pred, cells_gt) if len(cells_pred) and len(cells_gt) else None,
            "day0_median_dry_mass_pg_nowrap": float(d0.dry_mass_pg.median()) if len(d0) else None,
            "consistency_nowrap": dm.consistency_checks(cfg, p_nw), "consistency_all": dm.consistency_checks(cfg, cells_pred),
        },
        "storage": {
            "pred_nowrap": _storage(cfg, p_nw), "pred_all": _storage(cfg, cells_pred),
            "per_day_pred_nowrap": _per_day(cfg, p_nw), "per_day_pred_all": _per_day(cfg, cells_pred),
        },
        "background": background_residual(cells_pred),
        "reference_sensitivity": reference_sensitivity(cfg, val, preds),
    }
    pi = pi_mod.analyse_run(cfg, ref.run_id, ref.run_dir, tag, val, pi_dir if pi_dir is not None else Path("."))
    entry["phase_integral"] = {"summary": pi["summary"], "by_class": pi["by_class"], "strata": pi["strata"]}
    if csv_dir is not None:
        d = Path(csv_dir) / ref.run_id
        d.mkdir(parents=True, exist_ok=True)
        cells_pred.to_csv(d / "cells_pred.csv", index=False)
        entry["outputs"] = {"cells_pred_csv": str(Path("runs") / ref.run_id / "cells_pred.csv")}
    return entry


# ------------------------------------------------------------------------------------------------ flat tables
def _g(d: Optional[Dict], *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def flatten(e: Dict[str, Any]) -> Dict[str, Any]:
    """One row of scalars per complete run (used for the CSV and for the mean +- SD over seeds)."""
    seg, cells, dmass = e["segmentation"], e["cells"], e["dry_mass"]
    ag = dmass.get("agreement_nowrap") or {}
    pis = e["phase_integral"]["summary"]
    sp = _g(e, "storage", "pred_nowrap") or {}
    row: Dict[str, Any] = {
        "run_id": e["run_id"], "config": e["config"], "arch": e["arch"], "lora_rank": e["lora_rank"], "loss": e["loss"],
        "seed": e["seed"],
        "mean_dice": seg["mean_dice"], "aji": seg["aji"], "bf1": seg["bf1"], "ipe_global_pct": seg["ipe_global_pct"],
        "n_collapsed_classes": seg["n_collapsed"],
        "cells_predicted_nowrap": cells["predicted_nowrap"], "cells_annotated_nowrap": cells["annotated_nowrap"],
        "cells_ratio_nowrap": cells["cells_ratio_nowrap"], "cells_predicted_all": cells["predicted_all"],
        "pct_predicted_cells_wrapped": _g(cells, "wrapping", "pct_wrapped"),
        "composition_mean_abs_pp_nowrap": _g(cells, "composition_error_nowrap", "mean_abs_pp"),
        "composition_max_abs_pp_nowrap": _g(cells, "composition_error_nowrap", "max_abs_pp"),
        "pearson_r": ag.get("pearson_r"), "lin_ccc": ag.get("lin_ccc"), "bias_pg": ag.get("bland_altman_mean_pg"),
        "loa_low_pg": ag.get("bland_altman_loa_low_pg"), "loa_high_pg": ag.get("bland_altman_loa_high_pg"),
        "mdare_pct": ag.get("mdare_pct"), "mean_abs_rel_pct": ag.get("mean_abs_rel_pct"),
        "day0_median_dry_mass_pg": dmass.get("day0_median_dry_mass_pg_nowrap"),
        "gt_cells_matched_pct": pis.get("matched_cells_pct"),
        "percell_abs_median_pct": pis.get("percell_abs_median_pct"), "percell_abs_p90_pct": pis.get("percell_abs_p90_pct"),
        "percell_signed_median_pct": pis.get("percell_signed_median_pct"),
        "percell_frac_over_10pct": pis.get("percell_frac_over_10pct"),
        "percell_abs_median_pct_nowrap": pis.get("percell_abs_median_pct_nowrap"),
        "percell_abs_p90_pct_nowrap": pis.get("percell_abs_p90_pct_nowrap"),
        "percell_frac_over_10pct_nowrap": pis.get("percell_frac_over_10pct_nowrap"),
        "ipe_global_raw_mean_pct": pis.get("ipe_global_raw_mean_pct"),
        "ipe_global_asprovided_mean_pct": pis.get("ipe_global_asprovided_mean_pct"),
        "ipe_global_bgcorr_mean_pct": pis.get("ipe_global_bgcorr_mean_pct"),
        "area_rho_pred": _g(sp, "area", "spearman_rho"), "dry_mass_rho_pred": _g(sp, "dry_mass", "spearman_rho"),
        "dry_mass_late_vs_early_pct_pred": _g(sp, "dry_mass", "late_vs_early_pct"),
        "dry_mass_given_area_partial_rho_pred": _g(sp, "dry_mass_given_area_partial_spearman", "rho"),
        "background_residual_median_abs_rad": _g(e, "background", "median_abs_rad"),
        "background_residual_max_abs_rad": _g(e, "background", "max_abs_rad"),
        "ipe_pct_reference_annotation": _g(e, "reference_sensitivity", "ipe_global_pct", "annotation_background"),
        "ipe_pct_reference_prediction": _g(e, "reference_sensitivity", "ipe_global_pct", "prediction_background"),
        "ipe_pct_no_reference_as_provided": _g(e, "reference_sensitivity", "ipe_global_pct", "as_provided"),
    }
    gtn, gtw = pis.get("gt_cells"), pis.get("gt_cells_wrapped")
    mnw = pis.get("matched_cells_nowrap")
    row["gt_cells_matched_pct_nowrap"] = (100.0 * mnw / (gtn - gtw)) if (mnw is not None and gtn is not None and gtw is not None
                                                                     and gtn - gtw > 0) else None
    for c in CLASSES:
        row[f"dice_{c}"] = seg["dice_by_class"][c]
        row[f"count_ratio_{c}"] = _g(cells, "class_counts_nowrap", c, "ratio")
    return {k: (bool(v) if isinstance(v, (np.bool_, bool)) else v) for k, v in row.items()}


_ID_COLS = ("run_id", "config", "arch", "lora_rank", "loss", "seed")


def aggregate(entries: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Mean, sample SD (n - 1), min, max and the per-seed values of every scalar, per configuration (over seeds)."""
    rows = [flatten(e) for e in entries if e.get("status") == "complete"]
    out: Dict[str, Dict[str, Any]] = {}
    by_cfg: Dict[str, List[Dict]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)
    for cfg_key, rs in sorted(by_cfg.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        stats: Dict[str, Any] = {}
        for k in rs[0]:
            if k in _ID_COLS:
                continue
            vals = [(r["seed"], _f(r[k])) for r in rs]
            fin = [v for _, v in vals if v is not None]
            stats[k] = {"mean": float(np.mean(fin)) if fin else None,
                        "sd": float(np.std(fin, ddof=1)) if len(fin) > 1 else None,
                        "min": float(np.min(fin)) if fin else None, "max": float(np.max(fin)) if fin else None,
                        "n": len(fin), "per_seed": {str(s): v for s, v in vals}}
        out[cfg_key] = {"arch": rs[0]["arch"], "lora_rank": rs[0]["lora_rank"], "loss": rs[0]["loss"],
                        "seeds": [r["seed"] for r in rs], "n_seeds": len(rs), "metrics": stats}
    return out


# ------------------------------------------------------------------------------------------------ regression check
def _close(a: Any, b: Any) -> bool:
    fa, fb = _f(a), _f(b)
    if fa is None or fb is None:
        return fa is None and fb is None
    return math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-9)


def reference_checks(main_root: Path, entries: Sequence[Dict[str, Any]], main_seed: int, sel_arch: str, sel_rank: int) -> List[Dict]:
    """Compare the numbers computed here with the files written earlier by scripts/run_analysis.py and compare_architectures.py.

    Only seed-42 runs of the main grid exist there. A difference means the earlier file is stale or the code path changed."""
    checks: List[Dict] = []
    by_id = {e["run_id"]: e for e in entries if e.get("status") == "complete"}

    def cmp(source: str, run_id: str, key: str, expected: Any, got: Any) -> None:
        checks.append({"source": source, "run_id": run_id, "key": key, "expected": expected, "got": got,
                       "ok": bool(_close(expected, got))})

    dj = main_root / "analysis" / "drymass.json"
    rid = f"{sel_arch}_lora_r{sel_rank}"
    if dj.exists() and rid in by_id:
        d = read_json(dj)
        if d.get("selected_run") == rid:
            e = by_id[rid]
            for k, v in (d.get("agreement_nowrap") or {}).items():
                cmp("analysis/drymass.json", rid, f"agreement_nowrap.{k}", v, (e["dry_mass"]["agreement_nowrap"] or {}).get(k))
            for k, v in (d.get("agreement") or {}).items():
                cmp("analysis/drymass.json", rid, f"agreement.{k}", v, (e["dry_mass"]["agreement_all"] or {}).get(k))
            if d.get("wrapping_pred"):
                for k in ("n_cells", "n_wrapped"):
                    cmp("analysis/drymass.json", rid, f"wrapping_pred.{k}", d["wrapping_pred"].get(k), _g(e, "cells", "wrapping", k))
    pj = main_root / "analysis" / "phase_integral.json"
    if pj.exists():
        per_run = read_json(pj).get("per_run", {})
        for rid2, e in by_id.items():
            if e["seed"] == main_seed and rid2 in per_run:
                for k, v in per_run[rid2].items():
                    if k in ("run_id", "checkpoint") or not isinstance(v, (int, float)):
                        continue
                    cmp("analysis/phase_integral.json", rid2, f"summary.{k}", v, e["phase_integral"]["summary"].get(k))
    aj = main_root / "analysis" / "architecture_comparison.json"
    if aj.exists():
        arch_res = read_json(aj).get("architectures", {})
        rk = read_json(aj).get("rank")
        for arch, r in arch_res.items():
            rid3 = f"{arch}_lora_r{rk}"
            if rid3 in by_id:
                e = by_id[rid3]
                cmp("analysis/architecture_comparison.json", rid3, "cells_predicted", r["cells_predicted"], e["cells"]["predicted_nowrap"])
                cmp("analysis/architecture_comparison.json", rid3, "agreement.mdare_pct", _g(r, "agreement", "mdare_pct"),
                    _g(e, "dry_mass", "agreement_nowrap", "mdare_pct"))
                cmp("analysis/architecture_comparison.json", rid3, "composition_error.mean_abs_pp",
                    _g(r, "composition_error", "mean_abs_pp"), _g(e, "cells", "composition_error_nowrap", "mean_abs_pp"))
    return checks


# ------------------------------------------------------------------------------------------------ the whole analysis
_CTX: Dict[str, Any] = {}


def _worker(ref: RunRef) -> Dict[str, Any]:
    return analyse_run(_CTX["cfg"], _CTX["val"], _CTX["cells_gt"], ref, _CTX["tag"], _CTX["pi_dir"], _CTX["csv_dir"])


def run_all(cfg_m: Dict, val: Dict[str, Dict], refs: Sequence[RunRef], tag: str, out_dir: Path, jobs: int = 1,
            write_csv: bool = True, log: Callable[[str], None] = print) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Analyse every run in ``refs``. Returns the annotation cells table and the list of run entries (in the order of refs)."""
    out_dir = Path(out_dir)
    cells_gt = dm.cells_table(cfg_m, val, None)
    _CTX.update(cfg=cfg_m, val=val, cells_gt=cells_gt, tag=tag, pi_dir=out_dir / "phase_integral",
                csv_dir=(out_dir / "runs") if write_csv else None)
    entries: List[Dict[str, Any]] = []
    if jobs and jobs > 1 and len(refs) > 1:
        with mp.get_context("fork").Pool(min(jobs, len(refs))) as pool:
            for i, e in enumerate(pool.imap(_worker, refs), 1):
                entries.append(e)
                log(f"[cell_analysis] {i}/{len(refs)} {e['run_id']}: {e['status']}")
    else:
        for i, ref in enumerate(refs, 1):
            e = _worker(ref)
            entries.append(e)
            log(f"[cell_analysis] {i}/{len(refs)} {e['run_id']}: {e['status']}"
                + (f" ({e['cells']['predicted_nowrap']} predicted cells, MdARE "
                   f"{_g(e, 'dry_mass', 'agreement_nowrap', 'mdare_pct') or float('nan'):.2f}%)" if e["status"] == "complete" else
                   f" ({e.get('reason')})"))
    return cells_gt, entries


# ------------------------------------------------------------------------------------------------ individual result files (D-29)
def unit_label(ref: RunRef) -> str:
    return f"{ref.arch} | {'full-ft' if ref.rank is None else 'r' + str(ref.rank)} | {ref.loss} | s{ref.seed}"


def unit_path(store_dir: Path, run_id: str) -> Path:
    return Path(store_dir) / f"{run_id}.json"


def to_unit(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Entry as stored in an individual file: same fields, ``run_dir`` relative to the project root."""
    d = dict(entry)
    d["run_dir"] = rel(entry["run_dir"])
    d["schema_version"] = SCHEMA_VERSION
    return d


def from_unit(unit: Dict[str, Any], ref: RunRef) -> Dict[str, Any]:
    """Back to the entry the older code reads (absolute ``run_dir`` of this machine, no provenance block)."""
    e = {k: v for k, v in unit.items() if k not in ("unit_schema", "provenance", "schema_version")}
    e["run_dir"] = str(ref.run_dir)
    return e


def unit_status(unit: Optional[Dict[str, Any]], ref: RunRef, tag: str, manifest_sha: Optional[str],
                trust_missing_sources: bool = False) -> Tuple[bool, str]:
    """Is the stored individual result still valid for this run? (True, '') or (False, reason)."""
    if unit is None:
        return False, "no result yet"
    if unit.get("status") != "complete":
        return False, f"stored status {unit.get('status')}"
    if int(unit.get("schema_version", 0)) != SCHEMA_VERSION:
        return False, "schema version changed"
    if unit.get("checkpoint") != tag:
        return False, f"checkpoint {unit.get('checkpoint')} != {tag}"
    if manifest_sha and unit.get("provenance", {}).get("manifest_sha256") not in (None, manifest_sha):
        return False, "dataset manifest changed"
    pp = ref.run_dir / f"predictions_{tag}.npz"
    if not pp.exists():
        return (True, "") if trust_missing_sources else (False, f"{pp.name} not found (cannot verify)")
    if sha256_file(pp) != unit.get("predictions", {}).get("sha256"):
        return False, f"{pp.name} changed after the analysis"
    return True, ""


def _timed_worker(ref: RunRef) -> Tuple[Dict[str, Any], float]:
    t0 = time.time()
    try:
        e = _worker(ref)
    except Exception as exc:                                   # one failing run must not stop the others
        e = {"run_id": ref.run_id, "config": ref.config, "arch": ref.arch, "lora_rank": ref.rank, "loss": ref.loss,
             "seed": ref.seed, "run_dir": str(ref.run_dir), "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    return e, time.time() - t0


def run_units(cfg_m: Dict, refs: Sequence[RunRef], tag: str, out_dir: Path, store_dir: Path, manifest_sha: Optional[str],
              load_val: Callable[[], Dict[str, Dict]], prog, jobs: int = 1, write_csv: bool = True, force: bool = False,
              allow_missing: bool = False, trust_missing_sources: bool = False
              ) -> Tuple[List[Dict[str, Any]], Optional[pd.DataFrame], Optional[Dict[str, Dict]]]:
    """Analyse the runs whose individual result is missing or out of date; every finished run is written at once.

    Returns the entries in the order of ``refs`` (runs that could not be analysed carry status ``missing``/``error``),
    the annotation cells table and the validation data (both None when nothing had to be computed).
    """
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    entries: Dict[str, Dict[str, Any]] = {}
    todo: List[RunRef] = []
    for ref in refs:
        unit = None if force else read_unit(unit_path(store_dir, ref.run_id))
        ok, why = unit_status(unit, ref, tag, manifest_sha, trust_missing_sources) if unit is not None else (False, "forced" if force else "no result yet")
        if ok:
            entries[ref.run_id] = from_unit(unit, ref)
            prog.skipped(unit_label(ref), "up to date")
        else:
            todo.append(ref)

    val = cells_gt = None
    if todo:
        prog.stage(f"analyse {len(todo)} run(s)" + (f", {jobs} parallel" if jobs and jobs > 1 else ""))
        val = load_val()
        cells_gt = dm.cells_table(cfg_m, val, None)
        _CTX.update(cfg=cfg_m, val=val, cells_gt=cells_gt, tag=tag, pi_dir=Path(out_dir) / "phase_integral",
                    csv_dir=(Path(out_dir) / "runs") if write_csv else None)
        by_ref = {r.run_id: r for r in todo}

        def finish(e: Dict[str, Any], secs: float) -> None:
            entries[e["run_id"]] = e
            ref = by_ref[e["run_id"]]
            if e["status"] == "complete":
                write_unit(unit_path(store_dir, ref.run_id), to_unit(e),
                           sources=[ref.run_dir / "metrics.json", ref.run_dir / f"predictions_{tag}.npz"],
                           manifest_sha256=manifest_sha)
                mdare = _g(e, "dry_mass", "agreement_nowrap", "mdare_pct")
                prog.done(unit_label(ref), secs, f"{e['cells']['predicted_nowrap']} predicted cells"
                          + (f", MdARE {mdare:.2f}%" if mdare is not None else ""))
            elif allow_missing and e["status"] in ("missing",):
                prog.skipped(unit_label(ref), e.get("reason", "not available"))
            else:
                prog.fail(unit_label(ref), e.get("reason") or e["status"])

        if jobs and jobs > 1 and len(todo) > 1:
            with mp.get_context("fork").Pool(min(jobs, len(todo))) as pool:
                for e, secs in pool.imap_unordered(_timed_worker, todo):
                    finish(e, secs)
        else:
            for ref in todo:
                prog.start(unit_label(ref))
                e, secs = _timed_worker(ref)
                finish(e, secs)
    return [entries[r.run_id] for r in refs], cells_gt, val


# ------------------------------------------------------------------------------------------------ freshness of the stored file
def freshness_problems(doc: Dict[str, Any], main_root: Path, ms_root: Path, cfg: Dict) -> List[str]:
    """Why ``cell_analysis.json`` no longer matches the runs on disk (empty list = up to date)."""
    problems: List[str] = []
    sc = doc.get("meta", {}).get("scope", {})
    tag = doc["meta"]["checkpoint"]
    done = {e["run_id"]: e for e in doc.get("runs", {}).values() if e.get("status") == "complete"}
    for rid, e in done.items():
        pp = Path(e["run_dir"]) / f"predictions_{tag}.npz"
        if not pp.exists():
            problems.append(f"{rid}: {pp.name} no longer exists")
            continue
        st = pp.stat()
        if int(st.st_size) != e["predictions"]["bytes"] or int(st.st_mtime_ns) != e["predictions"]["mtime_ns"]:
            problems.append(f"{rid}: {pp.name} changed after the analysis")
    refs = expected_runs(cfg, main_root, ms_root, sc.get("archs", []), sc.get("ranks", []), sc.get("seeds", []),
                         sc.get("losses", ["full"]), bool(sc.get("include_full_finetune", False)))
    for ref in refs:
        if ref.run_id in done:
            continue
        m = ref.run_dir / "metrics.json"
        if m.exists() and read_json(m).get("status") == "complete" and (ref.run_dir / f"predictions_{tag}.npz").exists():
            problems.append(f"{ref.run_id}: finished but not in the analysis")
    if int(doc.get("meta", {}).get("schema_version", 0)) != SCHEMA_VERSION:
        problems.append("schema version differs from the code; rerun scripts/run_cell_analysis.py")
    return problems


def verification_errors(doc: Dict[str, Any], cfg: Dict, main_root: Path, ms_root: Path,
                        results_tables: Optional[Dict[str, Any]] = None) -> List[str]:
    """Everything scripts/verify_outputs.py requires of ``cell_analysis.json`` (empty list = fine)."""
    errors: List[str] = []
    meta = doc["meta"]
    if not meta.get("complete"):
        errors.append(f"cell_analysis.json is incomplete: {[p['run_id'] for p in meta.get('problems', [])][:5]}")
    ref = meta.get("reference_check", {})
    if ref.get("n_differences"):
        errors.append(f"cell_analysis.json differs from the earlier analysis files in outputs/full/analysis "
                      f"({ref['n_differences']} values); rerun scripts/run_analysis.py --skip-robustness and "
                      "scripts/compare_architectures.py, then scripts/run_cell_analysis.py")
    if meta.get("class_names_by_mask_value") != MANUSCRIPT_NAMES:
        errors.append("cell_analysis.json does not carry the manuscript class names (D-25)")
    for p in freshness_problems(doc, main_root, ms_root, cfg):
        errors.append(f"cell_analysis.json is out of date: {p}; rerun ./ops/run_cell_analysis.sh")
    if results_tables is not None:
        n_res = len(results_tables.get("cell_analysis_runs", []))
        if n_res != meta.get("n_runs_analysed"):
            errors.append(f"results.json holds {n_res} cell-analysis runs, cell_analysis.json {meta.get('n_runs_analysed')}; "
                          "rerun ./ops/make_manuscript_outputs.sh")
    return errors
