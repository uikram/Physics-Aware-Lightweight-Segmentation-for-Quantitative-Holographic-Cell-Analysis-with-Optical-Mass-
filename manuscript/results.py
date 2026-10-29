#!/usr/bin/env python
"""Consolidate every number used in the manuscript into ONE JSON file.

Reads only artefacts produced by the pipeline (per-run ``metrics.json``, ``analysis/*.json``,
``benchmark/benchmark.json``, ``data/processed/dataset_report.json`` and ``verification_report.json``,
``grid.json``) and the verified configuration. Nothing is typed in or estimated here; values are copied
or derived by the explicit formulas below (``x100`` for percentages). Missing inputs abort with an
explicit list instead of producing a partial file.

    python manuscript/results.py [--config-overlay smoke.yaml] [--output PATH]

Output: ``<paths.manuscript_outputs>/results.json`` (smoke runs write to ``outputs_smoke`` and are marked).

Class names (DECISIONS.md D-25): every file read here carries the legacy names (mask value 2 = "echinocyte", 3 = "spherocyte").
They are exchanged on reading (``qpi/utils/classnames.py``), so results.json uses the manuscript order: mask value 2 =
spherocyte, 3 = echinocyte. No number changes.

Multi-seed tables (``ms_*``) are built from ``outputs/multiseed/summary/runs_index.csv``. The rank and the architecture of the
single-run tables and figures default to configs/base.yaml (``analysis.selected_rank`` / ``selected_arch``); ``--rank`` and
``--arch`` override them for this run only (nothing in configs/ changes). Dry-mass and morphology analyses exist only for the run
named in ``meta.drymass_run``.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qpi.config import load_config  # noqa: E402
from qpi.experiments import build_grid, trainable_specs  # noqa: E402
from qpi.utils.io import read_json, sha256_file, write_json  # noqa: E402
from qpi.utils.outputs import mode_of  # noqa: E402
from qpi.utils.classnames import CLASSES, LEGACY_NAMES, MANUSCRIPT_NAMES, swap_obj  # noqa: E402
import ms_tables  # noqa: E402   (manuscript/ms_tables.py; the script directory is on sys.path)

ARCH_LABEL = {"edge_sam": "EdgeSAM", "mobile_sam": "MobileSAM", "mobilenet_unet": "MobileNet-UNet"}
LOSS_ROWS = [("dice_only", "Dice only"), ("pmc", "Dice + PMC"), ("pmc_bga", "Dice + PMC + BGA"),
             ("full", "Dice + PMC + BGA + IPP (full)")]


class Missing(Exception):
    pass


class Collector:
    """Reads input files, remembers their hashes (provenance) and the problems found."""

    def __init__(self):
        self.inputs: Dict[str, Dict[str, Any]] = {}
        self.missing: List[str] = []

    def load(self, path: Path, required: bool = True, swap: bool = True) -> Optional[Any]:
        """``swap=False`` only for files that already carry the manuscript class names (outputs/cell_analysis)."""
        if not path.exists():
            if required:
                self.missing.append(str(path))
            return None
        self.inputs[str(path)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        d = read_json(path)
        return swap_obj(d) if swap else d         # legacy -> manuscript class names (D-25)


def pct(x: Optional[float]) -> Optional[float]:
    return None if x is None else 100.0 * x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-overlay", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--rank", type=int, default=None, help="LoRA rank of the single-run tables/figures (default: configs/base.yaml)")
    ap.add_argument("--arch", default=None, help="architecture of the single-run analyses (default: configs/base.yaml)")
    ap.add_argument("--multiseed-summary", default="outputs/multiseed/summary",
                    help="folder with runs_index.csv (relative to the project root)")
    args = ap.parse_args()
    cfg = load_config(args.config_overlay)
    mode = mode_of(cfg)
    if list(cfg["data"]["class_names"]) != LEGACY_NAMES:
        print("results.py: configs/base.yaml class_names are not the legacy pipeline names "
              f"{LEGACY_NAMES}; the swap to the manuscript order (D-25) would be applied twice or to the wrong files. "
              "Read qpi/utils/classnames.py before changing them.", file=sys.stderr)
        return 1
    out_root = Path(cfg["paths_abs"]["output_root"])
    man_root = Path(cfg["paths_abs"]["manuscript_outputs"])
    processed = Path(cfg["paths_abs"]["processed_root"])
    sel_rank = int(args.rank if args.rank is not None else cfg["lora"]["selected_rank"])
    sel_arch = args.arch or cfg["analysis"]["selected_arch"]
    if sel_rank not in [int(r) for r in cfg["lora"]["ranks"]]:
        print(f"results.py: rank {sel_rank} is not in the rank sweep {cfg['lora']['ranks']}", file=sys.stderr)
        return 1
    ckpt = cfg["evaluation"]["checkpoint"]
    col = Collector()

    specs = build_grid(cfg)
    by_id = {s.run_id: s for s in specs}
    archs = list(dict.fromkeys(s.arch for s in specs))

    # ---------------------------------------------------------------- per-run metrics
    runs: Dict[str, Dict] = {}
    for s in trainable_specs(specs):
        m = col.load(out_root / "runs" / s.run_id / "metrics.json")
        if m is not None:
            runs[s.run_id] = m

    def metrics(run_id: str) -> Optional[Dict]:
        """Metrics of a run id, following aliases (e.g. ``<arch>_abl_full`` -> ``<arch>_lora_r8``)."""
        sp = by_id.get(run_id)
        rid = sp.alias_of if sp is not None and sp.alias_of else run_id
        return runs.get(rid)

    def ev(run_id: str) -> Optional[Dict]:
        m = metrics(run_id)
        return None if m is None else m["evaluation"][ckpt]

    pi = col.load(out_root / "analysis" / "phase_integral.json")
    dm = col.load(out_root / "analysis" / "drymass.json")
    rb = col.load(out_root / "analysis" / "robustness.json")
    bm = col.load(out_root / "benchmark" / "benchmark.json")
    eq = col.load(out_root / "benchmark" / "equivalence.json")
    ds_report = col.load(processed / "dataset_report.json")
    ds_verify = col.load(processed / "verification_report.json")
    grid = col.load(out_root / "grid.json", required=False)
    cell_doc = col.load(Path(cfg["project_root"]) / "outputs" / "cell_analysis" / "cell_analysis.json", required=False, swap=False)
    if cell_doc is not None and cell_doc["meta"]["class_names_by_mask_value"] != MANUSCRIPT_NAMES:
        print("results.py: outputs/cell_analysis/cell_analysis.json does not carry the manuscript class names; rerun "
              "scripts/run_cell_analysis.py (its names are written directly and must not be swapped again)", file=sys.stderr)
        return 1
    ms_index = Path(cfg["project_root"]) / args.multiseed_summary / "runs_index.csv"
    ms_json = Path(cfg["project_root"]) / args.multiseed_summary / "multiseed_results.json"
    for pth in (ms_index, ms_json):
        if not pth.exists():
            col.missing.append(str(pth))

    if col.missing:
        print("results.py: required inputs are missing:", file=sys.stderr)
        for p in col.missing:
            print(f"  - {p}", file=sys.stderr)
        return 1

    # ---------------------------------------------------------------- run status / integrity
    status = {rid: m["status"] for rid, m in runs.items()}
    bad = {rid: st for rid, st in status.items() if st not in ("complete", "diverged")}
    if bad:
        print(f"results.py: runs not finished: {bad}", file=sys.stderr)
        return 1
    primary = {m.get("primary_checkpoint") for m in runs.values()}
    if primary != {ckpt}:
        print(f"results.py: primary checkpoint mismatch: config={ckpt}, runs={primary}", file=sys.stderr)
        return 1
    wrong_mode = sorted(rid for rid, m in runs.items() if m.get("mode") != mode)
    if wrong_mode:
        print(f"results.py: runs from a different mode ({mode}) found: {wrong_mode}", file=sys.stderr)
        return 1

    def seg_row(run_id: str) -> Dict[str, Any]:
        e, m = ev(run_id), metrics(run_id)
        sp = by_id[run_id]
        return {
            "run_id": sp.alias_of or run_id,
            "architecture": ARCH_LABEL.get(sp.arch, sp.arch),
            "lora_rank": sp.lora_r if sp.lora_r is not None else None,
            "mean_dice": e["mean_dice"], "mean_iou": e["mean_iou"], "aji": e["aji"], "bf1": e["bf1"],
            "ipe_global_pct": pct(e["ipe_global"]),
            **{f"dice_{c}": e[f"dice_{c}"] for c in CLASSES},
            **{f"iou_{c}": e[f"iou_{c}"] for c in CLASSES},
            **{f"ipe_{c}_pct": pct(e[f"ipe_{c}"]) for c in CLASSES},
            "params_total": m["parameters"]["total"], "params_trainable": m["parameters"]["trainable"],
            "trainable_percent": m["parameters"]["trainable_percent"],
            "status": m["status"],
        }

    tables: Dict[str, Any] = {}
    tables["table1_segmentation_selected_rank"] = [seg_row(f"{a}_lora_r{sel_rank}") for a in archs]
    # swap self-check (D-25): the manuscript class that is called 'spherocyte' must carry the numbers that the legacy file
    # stores under 'echinocyte' (mask value 2), and vice versa. Read the raw file again, without the swap.
    for a_ in archs:
        raw = read_json(out_root / "runs" / f"{a_}_lora_r{sel_rank}" / "metrics.json")["evaluation"][ckpt]
        row_ = next(r for r in tables["table1_segmentation_selected_rank"] if r["run_id"] == f"{a_}_lora_r{sel_rank}")
        ok_ = (row_["dice_spherocyte"] == raw["dice_echinocyte"] and row_["dice_echinocyte"] == raw["dice_spherocyte"]
               and row_["dice_discocyte"] == raw["dice_discocyte"] and row_["dice_stomatocyte"] == raw["dice_stomatocyte"])
        if not ok_:
            print(f"results.py: class-name swap self-check failed for {a_}", file=sys.stderr)
            return 1
    tables["table2_lora_vs_full_finetune"] = [
        {**seg_row(rid), "method": "LoRA" if by_id[rid].lora_r is not None else "Full fine-tuning"}
        for a in archs for rid in (f"{a}_lora_r{sel_rank}", f"{a}_full_finetune") if rid in by_id]
    tables["table3_loss_ablation"] = [
        {"loss_configuration": label, "ablation": key, **seg_row(f"{a}_abl_{key}")}
        for a in archs for key, label in LOSS_ROWS if f"{a}_abl_{key}" in by_id]
    tables["table4_per_class_ablation_selected_arch"] = [
        {"loss_configuration": label, "ablation": key, "architecture": ARCH_LABEL.get(sel_arch, sel_arch),
         **{f"ipe_{c}_pct": pct(ev(f"{sel_arch}_abl_{key}")[f"ipe_{c}"]) for c in CLASSES},
         **{f"dice_{c}": ev(f"{sel_arch}_abl_{key}")[f"dice_{c}"] for c in CLASSES}}
        for key, label in LOSS_ROWS if f"{sel_arch}_abl_{key}" in by_id]
    tables["table5_integrated_phase_preservation"] = [
        {"architecture": ARCH_LABEL.get(a, a), "lora_rank": sel_rank,
         "ipe_global_pct": pct(ev(f"{a}_lora_r{sel_rank}")["ipe_global"]),
         **{f"ipe_{c}_pct": pct(ev(f"{a}_lora_r{sel_rank}")[f"ipe_{c}"]) for c in CLASSES},
         "aji": ev(f"{a}_lora_r{sel_rank}")["aji"]} for a in archs]
    tables["tableS1_lora_rank_sweep"] = [
        {**seg_row(f"{a}_lora_r{r}"), "rank": r}
        for a in archs for r in cfg["lora"]["ranks"] if f"{a}_lora_r{r}" in by_id]

    # benchmark (protocol v5, D-26): FP32, PyTorch and ONNX Runtime, one measurement per architecture at its deployment
    # input size, LoRA merged, so the timing is rank independent. Row fields: mean over seeds of the per-seed P50, P99, FPS and Peak
    # VRAM; "<field>_sd" is the sample SD over seeds, "<field>_per_seed" the per-seed values.
    tables["table6_benchmark"] = [{**r, "architecture": ARCH_LABEL.get(r["architecture"], r["architecture"])} for r in bm["rows"]]
    tables["table6_benchmark_per_seed"] = [{**r, "architecture": ARCH_LABEL.get(r["architecture"], r["architecture"])}
                                           for r in bm["per_seed"]]
    # correctness and quality gate: ONNX FP32 against PyTorch FP32 (qpi/equivalence.py)
    tables["table_onnx_equivalence"] = [
        {"architecture": ARCH_LABEL.get(a["architecture"], a["architecture"]), "rank": a["rank"], "n_inputs": a["n_inputs"],
         **a["logit_comparison"],
         **{f"quality_{k}": v for k, v in next(c for c in a["quality"]["candidates"] if c["candidate"] == "onnx_fp32").items()
            if k != "candidate"},
         "mean_dice_pytorch": next(c for c in a["quality"]["candidates"] if c["candidate"] == "pytorch_fp32")["mean_dice"]}
        for a in eq["architectures"]]
    tables["onnx_equivalence_acceptance"] = eq["acceptance"]
    tables["tableS1_trainable_parameters_by_rank"] = [
        {**r, "architecture": ARCH_LABEL.get(r["architecture"], r["architecture"])}
        for r in bm.get("parameters_by_rank", [])]

    # storage statistics (Table 7)
    tables["table7_storage_per_day_predicted"] = dm["per_day_pred"]
    tables["table7_storage_per_day_annotated"] = dm["per_day_gt"]
    # same tables without phase-wrapped cells (D-21): use these for dry mass; the all-cell tables above are for
    # morphology and class composition, which wrapping does not affect
    tables["table7_storage_per_day_predicted_nowrap"] = dm["per_day_pred_nowrap"]
    tables["table7_storage_per_day_annotated_nowrap"] = dm["per_day_gt_nowrap"]

    # ---------------------------------------------------------------- multi-seed tables (3 seeds per configuration)
    ms = ms_tables.build(ms_index, ms_json)
    for pth in (ms_index, ms_json):
        col.inputs[str(pth)] = {"sha256": sha256_file(pth), "bytes": pth.stat().st_size}
    tables["ms_runs"] = ms["runs"]
    tables["ms_summary"] = ms["summary"]
    tables["ms_paired"] = ms["paired"]
    tables["ms_meta"] = ms["meta"]

    # ---------------------------------------------------------------- per-cell analyses of every architecture, rank and seed
    if cell_doc is not None:
        from qpi.analysis.cell_analysis import flatten
        tables["cell_analysis_runs"] = [flatten(e) for e in cell_doc["runs"].values() if e.get("status") == "complete"]
        tables["cell_analysis_by_configuration"] = cell_doc["by_configuration"]
        tables["cell_analysis_annotation"] = cell_doc["annotation"]

    # ---------------------------------------------------------------- analyses (verbatim from the stage outputs)
    analysis = {
        "phase_integral": {
            "checkpoint": pi["checkpoint"],
            "per_run": pi["per_run"], "by_class": pi["by_class"], "strata": pi["strata"],
            "paired_comparisons": pi["paired_comparisons"],
        },
        "dry_mass": {k: dm[k] for k in dm if not k.startswith("per_day_")},
        "robustness": rb,
    }

    # ---------------------------------------------------------------- run integrity and flags
    flags = {
        "diverged_runs": sorted(r for r, s in status.items() if s == "diverged"),
        "primary_checkpoint": ckpt,
        "best_checkpoint_selected_on_evaluation_split": bool(cfg["evaluation"].get("also_evaluate_best")),
        "n_trained_runs": len(runs), "n_expected_trained_runs": len(trainable_specs(specs)),
        "aliases": {s.run_id: s.alias_of for s in specs if s.alias_of},
        "dataset_verification_ok": None if ds_verify is None else ds_verify.get("ok"),
    }
    if flags["n_trained_runs"] != flags["n_expected_trained_runs"]:
        print("results.py: not all runs of the grid are present", file=sys.stderr)
        return 1
    best_view = {rid: m["evaluation"].get("best") for rid, m in runs.items()}

    first = next(iter(runs.values()))
    result = {
        "meta": {
            "mode": mode, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "seed": cfg["seed"], "selected_architecture": sel_arch, "selected_lora_rank": sel_rank,
            "single_run_seed": cfg["seed"], "multiseed_seeds": ms["meta"]["seeds"],
            "rank_is_placeholder": "rank and architecture of the single-run tables are those requested on the command line "
                                   "or configs/base.yaml; the manuscript rank is decided by the author (not by this script)",
            "drymass_run": dm.get("selected_run"),
            "cell_analysis": None if cell_doc is None else {k: cell_doc["meta"][k] for k in (
                "generated_utc", "scope", "n_runs_requested", "n_runs_analysed", "complete", "checkpoint", "reference_check")},
            "class_names_by_mask_value": MANUSCRIPT_NAMES,
            "class_name_note": "mask value 2 = spherocyte, 3 = echinocyte (D-25); the pipeline files use the opposite names for 2 and 3",
            "units": {"*_pct": "percent = 100 x fraction produced by the evaluation code", "ipe": "integrated phase error",
                      "dry_mass": "pg, physical constants only (no MCH calibration)", "area": "um^2"},
            "n_validation_images": first.get("n_val_images"),
        },
        "configuration": swap_obj({k: cfg[k] for k in ("seed", "physics", "data", "registration", "training", "loss",
                                                       "evaluation", "analysis", "benchmark", "lora") if k in cfg}),
        "dataset": {"report": ds_report, "verification": ds_verify},
        "flags": flags,
        "run_status": {rid: {"status": m["status"], "parameters": m["parameters"], "training": m["training"],
                              "n_train_views": m.get("n_train_views"), "wall_seconds": m.get("wall_seconds")}
                       for rid, m in runs.items()},
        "tables": tables,
        "benchmark": {"protocol": bm.get("protocol"), "gpu_name": bm.get("gpu_name"), "host_details": bm.get("host_details"),
                      "gpu_at_start_of_seeds": bm.get("gpu_at_start_of_seeds"), "equivalence": eq},
        "analysis": analysis,
        "evaluation_best_checkpoint_selected_on_evaluation_split": best_view,
        "environment": first["environment"],
        "provenance": {"inputs": col.inputs, "grid": grid},
    }
    target = Path(args.output) if args.output else man_root / "results.json"
    write_json(target, result)
    print(f"wrote {target} ({len(col.inputs)} inputs, {len(runs)} runs, mode={mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
