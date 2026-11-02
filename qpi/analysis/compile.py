"""Compilers: individual result files -> multi-seed JSONs (DECISIONS.md D-29). No science is computed here, only aggregation.

Inputs (one JSON per run / timing pass, see ``qpi.utils.layout``) -> outputs in ``outputs/compiled/``:

    segmentation_multiseed.json   from outputs/multiseed/summary/runs_index.csv (built from every run's metrics.json)
    cell_analysis_multiseed.json  from outputs/analysis/cell/*.json
    robustness_multiseed.json     from outputs/analysis/robustness/*.json
    benchmark_multiseed.json      from outputs/analysis/benchmark/*.json  (Peak VRAM headline = maximum over seeds)

Aggregation rules (written into every file under ``aggregation``): statistics over seeds, mean and sample SD (n - 1), min, max,
the per-seed values; collapsed runs are included; a missing or failed seed is excluded and listed under ``problems`` (never
counted as zero). Every file lists its inputs with size and sha256. ``complete`` is true only when ``problems`` is empty.
Nothing here selects a model, a rank or a seed: choosing is done at the end with ``apply_view`` and ``configs/views.yaml``.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from qpi.utils.io import read_json, sha256_file, write_json
from qpi.utils.layout import KINDS, ROOT, analysis_dir, compiled_path, rel, unit_files
from qpi.utils.results_io import read_unit

SCHEMA_VERSION = 1
DEFAULT_SEEDS = (42, 43, 44)
AGGREGATION = {
    "mean_over": "seeds", "sd": "sample SD (n - 1); null when fewer than two seeds", "min_max": "over seeds",
    "per_seed": "stored for every statistic", "collapsed_runs": "included (a class with Dice < 0.05 counts as collapsed)",
    "missing_or_failed": "excluded and listed under problems; never counted as zero",
}


# ------------------------------------------------------------------------------------------------ statistics
def _num(x: Any) -> Optional[float]:
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def stats(per_seed: Dict[Any, Any]) -> Dict[str, Any]:
    """Mean, sample SD, min, max and the per-seed values of ``{seed: value}`` (None values are excluded from the statistics)."""
    vals = [v for v in (_num(x) for x in per_seed.values()) if v is not None]
    return {"mean": float(np.mean(vals)) if vals else None,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "min": float(np.min(vals)) if vals else None, "max": float(np.max(vals)) if vals else None,
            "n": len(vals), "per_seed": {str(k): _num(v) for k, v in per_seed.items()}}


def leaves(obj: Any, prefix: str = "") -> Dict[str, float]:
    """Numeric leaves of a nested dict as ``{'a.b.c': value}`` (lists, strings and booleans are skipped)."""
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        v = _num(obj)
        if v is not None:
            out[prefix] = v
    return out


def stats_of_leaves(per_seed: Dict[Any, Any]) -> Dict[str, Dict[str, Any]]:
    keys: List[str] = []
    flat = {s: leaves(o) for s, o in per_seed.items()}
    for f in flat.values():
        keys += [k for k in f if k not in keys]
    return {k: stats({s: f.get(k) for s, f in flat.items()}) for k in keys}


def _inputs(paths: Iterable[Path], root: Path) -> List[Dict[str, Any]]:
    return [{"path": rel(p, root), "bytes": int(Path(p).stat().st_size), "sha256": sha256_file(p)} for p in paths if Path(p).exists()]


def _document(kind: str, inputs: List[Dict], problems: List[str], body: Dict[str, Any]) -> Dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "kind": kind, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "aggregation": AGGREGATION, "inputs": inputs, "n_inputs": len(inputs), "problems": problems,
            "complete": not problems, **body}


RUN_ID = re.compile(r"^(edge_sam|mobile_sam|mobilenet_unet)_(?:lora_r(\d+)|abl_(dice_only|pmc_bga|pmc|full)(?:_r(\d+))?|(full_finetune))(?:__s(\d+))?$")


def parse_run_id(run_id: str, reference_rank: int = 8, main_seed: int = 42) -> Dict[str, Any]:
    """arch, rank, loss, seed and configuration key of a run id such as 'edge_sam_abl_pmc_r4__s43' (None if not a run id)."""
    m = RUN_ID.match(run_id)
    if not m:
        return {}
    arch, r_lora, loss_abl, r_abl, ft, seed = m.groups()
    if ft:
        rank, loss, label = None, "full_finetune", "full_finetune"
    elif r_lora:
        rank, loss, label = int(r_lora), "full", f"lora_r{r_lora}"
    else:
        rank, loss = (int(r_abl) if r_abl else reference_rank), loss_abl
        label = f"abl_{loss_abl}" + (f"_r{r_abl}" if r_abl else "")
    return {"arch": arch, "lora_rank": rank, "loss": loss, "seed": int(seed) if seed else main_seed, "config": f"{arch}|{label}"}


def config_fields(config: str, reference_rank: int = 8) -> Dict[str, Any]:
    """arch, LoRA rank and loss of a configuration key such as 'edge_sam|lora_r16', 'edge_sam|abl_pmc', 'edge_sam|abl_pmc_r4'."""
    arch, label = config.split("|")
    m = re.fullmatch(r"lora_r(\d+)", label)
    if m:
        return {"arch": arch, "lora_rank": int(m.group(1)), "loss": "full"}
    if label == "full_finetune":
        return {"arch": arch, "lora_rank": None, "loss": "full_finetune"}
    m = re.fullmatch(r"abl_(dice_only|pmc_bga|pmc|full)(?:_r(\d+))?", label)
    if m:
        return {"arch": arch, "lora_rank": int(m.group(2)) if m.group(2) else reference_rank, "loss": m.group(1)}
    return {"arch": arch, "lora_rank": None, "loss": label}


# ------------------------------------------------------------------------------------------------ segmentation
def compile_segmentation(root: Path = ROOT, runs_index: Optional[Path] = None, expected_seeds: Sequence[int] = DEFAULT_SEEDS
                         ) -> Dict[str, Any]:
    import pandas as pd
    from qpi.analysis.multiseed import METRICS

    root = Path(root)
    ri = Path(runs_index) if runs_index else root / "outputs" / "multiseed" / "summary" / "runs_index.csv"
    df = pd.read_csv(ri)
    problems: List[str] = []
    configs: Dict[str, Any] = {}
    runs: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        runs.append({k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.to_dict().items()
                     if k in ("config", "seed", "run_id", "origin", "metrics_path", "metrics_sha256", "status")})
    for key, g in df.groupby("config"):
        exp = [42] if key.endswith("|full_finetune") else list(expected_seeds)
        done = g[g.status == "complete"].sort_values("seed")
        found = [int(s) for s in done.seed]
        missing = [s for s in exp if s not in found]
        bad = g[g.status != "complete"]
        for _, r in bad.iterrows():
            problems.append(f"{r['run_id']}: status {r['status']}")
        for s in missing:
            problems.append(f"{key}: seed {s} missing")
        metrics = {m: stats({int(s): v for s, v in zip(done.seed, done[m])}) for m in METRICS if m in done}
        coll = done[done.n_collapsed > 0]
        configs[key] = {**config_fields(key), "seeds_expected": exp, "seeds_found": found, "seeds_missing": missing,
                        "n_seeds": len(found), "trainable_params": int(done.trainable_params.iloc[0]) if len(done) else None,
                        "n_runs_with_a_collapsed_class": int(len(coll)), "collapsed_seeds": [int(s) for s in coll.seed],
                        "collapsed_classes_by_seed": {str(int(s)): str(c).split() for s, c in zip(coll.seed, coll.collapsed_classes)},
                        "metrics": metrics}
    extra: Dict[str, Any] = {}
    ms_json = ri.parent / "multiseed_results.json"
    inputs = [ri] + ([ms_json] if ms_json.exists() else [])
    if ms_json.exists():
        ms = read_json(ms_json)
        extra["collapse_dice_threshold"] = ms.get("collapse_dice_threshold")
        extra["paired"] = ms.get("paired", [])
        # descriptive facts about the rank sweep; the configured default rank is NOT a selection (D-27)
        extra["rank_facts"] = {a: {"arch": v["arch"], "configured_default_rank": v["selected_rank"], "per_metric": v["per_metric"]}
                               for a, v in ms.get("rank_selection", {}).items()}
    return _document("segmentation_multiseed", _inputs(inputs, root) +
                     [{"path": p["metrics_path"], "sha256": p["metrics_sha256"], "listed_in": rel(ri, root)}
                      for p in runs if p.get("metrics_sha256")], problems,
                     {"checkpoint": "last", "n_runs": len(runs), "configurations": configs, "runs": runs, **extra})


# ------------------------------------------------------------------------------------------------ cell analysis
def _rows_by_key(per_seed_rows: Dict[Any, List[Dict]], key: str) -> Dict[Any, Dict[Any, Dict]]:
    out: Dict[Any, Dict[Any, Dict]] = {}
    for s, rows in per_seed_rows.items():
        for r in rows or []:
            out.setdefault(r.get(key), {})[s] = r
    return out


def compile_cell(root: Path = ROOT, expected_seeds: Sequence[int] = DEFAULT_SEEDS, expect: str = "present") -> Dict[str, Any]:
    """``expect``: 'present' checks the seeds of every configuration found; 'all' also requires the full grid
    (3 architectures x ranks x 4 losses + full fine-tuning) to be present."""
    from qpi.analysis import cell_analysis as ca

    root = Path(root)
    files = unit_files("cell", root)
    problems: List[str] = []
    entries: List[Dict[str, Any]] = []
    for f in files:
        u = read_unit(f)
        if u is None:
            problems.append(f"{rel(f, root)}: unreadable or from another unit schema")
            continue
        if u.get("status") != "complete":
            problems.append(f"{u.get('run_id')}: status {u.get('status')}")
            continue
        entries.append({k: v for k, v in u.items() if k not in ("unit_schema", "provenance")})
    agg = ca.aggregate(entries)
    by_cfg: Dict[str, List[Dict]] = {}
    for e in entries:
        by_cfg.setdefault(e["config"], []).append(e)
    configs: Dict[str, Any] = {}
    for key, v in agg.items():
        es = sorted(by_cfg[key], key=lambda e: e["seed"])
        exp = [42] if v["loss"] == "full_finetune" else list(expected_seeds)
        missing = [s for s in exp if s not in v["seeds"]]
        problems += [f"{key}: seed {s} missing" for s in missing]
        per_day = _rows_by_key({e["seed"]: e["storage"]["per_day_pred_nowrap"] for e in es}, "storage_day")
        strata = _rows_by_key({e["seed"]: e["phase_integral"]["strata"] for e in es}, "stratum")
        configs[key] = {
            "arch": v["arch"], "lora_rank": v["lora_rank"], "loss": v["loss"], "seeds_expected": exp, "seeds_found": v["seeds"],
            "seeds_missing": missing, "n_seeds": v["n_seeds"], "metrics": v["metrics"],
            "per_day_pred_nowrap": [{"storage_day": d, "n_seeds": len(rows),
                                     "metrics": stats_of_leaves({s: {k: v for k, v in r.items() if k != "storage_day"}
                                                                 for s, r in rows.items()})}
                                    for d, rows in sorted(per_day.items(), key=lambda kv: (kv[0] is None, kv[0]))],
            "error_strata": {name: stats({s: r.get("error_rate_pct") for s, r in rows.items()}) for name, rows in strata.items()},
            "phase_integral_by_class": stats_of_leaves({e["seed"]: e["phase_integral"]["by_class"] for e in es}),
        }
    if expect == "all":
        got = {(e["arch"], e["lora_rank"], e["loss"], e["seed"]) for e in entries}
        for arch in sorted({e["arch"] for e in entries}):
            for loss in ca.LOSSES:
                for rk in sorted({e["lora_rank"] for e in entries if e["lora_rank"] is not None}):
                    for s in expected_seeds:
                        if (arch, rk, loss, s) not in got and ca.locate_run({"lora": {"selected_rank": 8}, "seed": 42}, Path("."), Path("."), arch, rk, loss, s):
                            problems.append(f"{arch} r{rk} {loss} seed {s}: no individual result")
    ann_path = analysis_dir("cell", root) / "_annotation.json"
    body: Dict[str, Any] = {"n_runs": len(entries), "configurations": configs}
    inputs = list(files)
    if ann_path.exists():
        body["annotation"] = read_json(ann_path)["annotation"]
        inputs.append(ann_path)
    return _document("cell_analysis_multiseed", _inputs(inputs, root), sorted(set(problems)), body)


# ------------------------------------------------------------------------------------------------ robustness
def compile_robustness(root: Path = ROOT, expected_seeds: Sequence[int] = DEFAULT_SEEDS) -> Dict[str, Any]:
    root = Path(root)
    files = unit_files("robustness", root)
    problems: List[str] = []
    units: List[Dict] = []
    for f in files:
        u = read_unit(f)
        if u is None or u.get("status") != "complete":
            problems.append(f"{rel(f, root)}: unreadable, incomplete or from another unit schema")
            continue
        units.append(u)
    by_cfg: Dict[str, List[Dict]] = {}
    for u in units:
        by_cfg.setdefault(u["config"], []).append(u)
    configs: Dict[str, Any] = {}
    fields = ("macro_dice", "fg_iou", "ipe_bgcorr_true_pct", "ipe_uncorrected_seen_pct", "delta_macro_dice_vs_none")
    for key, us in sorted(by_cfg.items()):
        us = sorted(us, key=lambda u: u["seed"])
        seeds = [u["seed"] for u in us]
        exp = [42] if us[0]["loss"] == "full_finetune" else list(expected_seeds)
        # a configuration that only exists for seed 42 by design (the seed list given) is checked against ``exp``
        missing = [s for s in exp if s not in seeds]
        problems += [f"{key}: seed {s} missing" for s in missing]
        names = [r["distortion"] for r in us[0]["rows"]]
        dist = {}
        for n in names:
            rows = {u["seed"]: next(r for r in u["rows"] if r["distortion"] == n) for u in us}
            dist[n] = {**{f: stats({s: r[f] for s, r in rows.items()}) for f in fields},
                       "invariant_by_construction": bool(next(iter(rows.values()))["invariant_by_construction"])}
        configs[key] = {"arch": us[0]["arch"], "lora_rank": us[0]["lora_rank"], "loss": us[0]["loss"], "seeds_expected": exp,
                        "seeds_found": seeds, "seeds_missing": missing, "n_seeds": len(seeds), "distortions": dist}
    return _document("robustness_multiseed", _inputs(files, root), sorted(set(problems)),
                     {"checkpoint": units[0]["checkpoint"] if units else None, "n_runs": len(units), "configurations": configs,
                      "note": "rows flagged invariant_by_construction: the offset is added before the per-image z-score of the network "
                              "input, which removes any constant; an unchanged Dice there is a property of the preprocessing"})


# ------------------------------------------------------------------------------------------------ benchmark
def compile_benchmark(root: Path = ROOT, expected_seeds: Sequence[int] = DEFAULT_SEEDS) -> Dict[str, Any]:
    root = Path(root)
    d = analysis_dir("benchmark", root)
    files = unit_files("benchmark", root)
    proto_path = d / "_protocol.json"
    problems: List[str] = []
    units = []
    for f in files:
        u = read_unit(f)
        if u is None:
            problems.append(f"{rel(f, root)}: unreadable or from another unit schema")
        else:
            units.append(u)
    groups: Dict[tuple, List[Dict]] = {}
    for u in units:
        groups.setdefault((u["architecture"], u["backend"]), []).append(u)
    rows = []
    for (arch, backend), us in sorted(groups.items()):
        us = sorted(us, key=lambda u: u["seed"])
        seeds = [u["seed"] for u in us]
        missing = [s for s in expected_seeds if s not in seeds]
        problems += [f"{arch} {backend}: seed {s} missing" for s in missing]
        m = {k: stats({u["seed"]: u[k] for u in us}) for k in ("latency_p50_ms", "latency_p99_ms", "fps", "peak_vram_mb")}
        m["peak_vram_mb_max"] = max(u["peak_vram_mb"] for u in us)         # headline (provisioning number): worst case over seeds
        rows.append({"architecture": arch, "backend": backend, "precision": us[0].get("precision"), "input_size": us[0].get("input_size"),
                     "batch_size": us[0].get("batch_size"), "weights": us[0].get("weights"),
                     "checkpoint_sha256_16": us[0].get("checkpoint_sha256_16"), "seeds_expected": list(expected_seeds),
                     "seeds_found": seeds, "seeds_missing": missing, "n_seeds": len(seeds), **m})
    body: Dict[str, Any] = {"rows": rows}
    inputs = list(files)
    if proto_path.exists():
        p = read_json(proto_path)
        body.update({k: p[k] for k in p if k in ("protocol", "gpu_name", "host_details", "parameters_by_rank")})
        inputs.append(proto_path)
    eq = root / "outputs" / "full" / "benchmark" / "equivalence.json"
    if eq.exists():
        body["equivalence"] = read_json(eq)
        inputs.append(eq)
    return _document("benchmark_multiseed", _inputs(inputs, root), sorted(set(problems)), body)


# ------------------------------------------------------------------------------------------------ writing and validation
COMPILERS = {"segmentation": compile_segmentation, "cell": compile_cell, "robustness": compile_robustness,
             "benchmark": compile_benchmark}
NAMES = {"segmentation": "segmentation", "cell": "cell_analysis", "robustness": "robustness", "benchmark": "benchmark"}


def write_compiled(kind: str, root: Path = ROOT, **kw: Any) -> Dict[str, Any]:
    """Compile ``kind``, write it atomically, read it back and check it. Returns {'path', 'doc', 'ok', 'errors'}."""
    doc = COMPILERS[kind](Path(root), **kw)
    path = compiled_path(NAMES[kind], Path(root))
    write_json(path, doc)
    back = json.loads(path.read_text())
    errors: List[str] = []
    if back.get("kind") != doc["kind"] or back.get("n_inputs") != doc["n_inputs"]:
        errors.append("the file read back differs from the compiled document")
    if kind != "benchmark" and not back.get("configurations"):
        errors.append("no configurations were compiled")
    if kind == "benchmark" and not back.get("rows"):
        errors.append("no benchmark rows were compiled")
    return {"path": path, "doc": back, "ok": not errors, "errors": errors}


def stale_problems(kind: str, root: Path = ROOT) -> List[str]:
    """Why the compiled file of ``kind`` no longer matches its inputs (empty list: it is current). Uses the recorded sha256 of every
    input and the set of individual files now present, so it does not depend on file times."""
    root = Path(root)
    path = compiled_path(NAMES[kind], root)
    if not path.exists():
        return ["not compiled"]
    doc = json.loads(path.read_text())
    out: List[str] = []
    recorded = {i["path"]: i["sha256"] for i in doc.get("inputs", [])}
    for pth, sha in recorded.items():
        f = root / pth
        if not f.exists():
            if "/runs/" not in pth:                       # run folders may be absent on the PC; the individual files are what counts
                out.append(f"input removed: {pth}")
        elif sha256_file(f) != sha:
            out.append(f"input changed: {pth}")
    if kind in KINDS:
        out += [f"new individual file not compiled: {rel(f, root)}" for f in unit_files(kind, root) if rel(f, root) not in recorded]
    return out


# ------------------------------------------------------------------------------------------------ views (choosing at the end)
def load_views(path: Optional[Path] = None) -> Dict[str, Any]:
    import yaml
    p = Path(path) if path else ROOT / "configs" / "views.yaml"
    return yaml.safe_load(p.read_text())["views"]


def apply_view(compiled: Dict[str, Any], view: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Configurations of a compiled file that belong to a view. A view is either ``{configs: [{arch, rank, loss}, ...]}`` (in that
    order) or a filter ``{arch: [...], rank: r, loss: l}`` (missing keys match everything). Nothing in the compiled file changes."""
    confs = [{"config": k, **v} for k, v in compiled["configurations"].items()]

    def match(c: Dict[str, Any], spec: Dict[str, Any]) -> bool:
        archs = spec.get("arch")
        archs = [archs] if isinstance(archs, str) else archs
        return ((archs is None or c["arch"] in archs) and (spec.get("rank") is None or c["lora_rank"] == spec["rank"])
                and (spec.get("loss") is None or c["loss"] == spec["loss"]))

    if "configs" in view:
        out = []
        for spec in view["configs"]:
            out += [c for c in confs if match(c, spec)]
        return out
    return [c for c in confs if match(c, view)]
