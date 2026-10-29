#!/usr/bin/env python
"""Post-training analyses. Reads finished runs under <output_root>/runs and writes <output_root>/analysis.

    python scripts/run_analysis.py [--config-overlay smoke.yaml] [--gpu 0]

Stages: phase_integral (all runs), drymass (selected run + annotation), robustness (selected configurations).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from qpi.config import load_config  # noqa: E402
from qpi.experiments import build_grid, resolve, run_config, run_dir  # noqa: E402
from qpi.utils.io import read_json, write_json  # noqa: E402
from qpi.utils.outputs import claim_output_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-overlay", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip-robustness", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config_overlay)
    root = claim_output_root(cfg)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    specs = build_grid(cfg)
    primary = cfg["evaluation"]["checkpoint"]

    from qpi.analysis.common import load_predictions, load_validation
    from qpi.analysis import drymass as dm
    from qpi.analysis import phase_integral as pi_mod

    val = load_validation(cfg)
    finished = {}
    for s in specs:
        t = resolve(specs, s.run_id)
        m = run_dir(cfg, t.run_id) / "metrics.json"
        if not m.exists():
            raise FileNotFoundError(f"run {t.run_id} has no metrics.json; run training first")
        if read_json(m)["status"] != "complete":
            print(f"[analysis] skipping {s.run_id}: status {read_json(m)['status']}")
            continue
        finished[s.run_id] = run_dir(cfg, t.run_id)

    # ---- integrated-phase analyses (all finished runs) --------------------------------------------------------
    summaries, strata, by_class, per_image = {}, [], {}, {}
    for rid, rdir in finished.items():
        res = pi_mod.analyse_run(cfg, rid, rdir, primary, val, out / "phase_integral")
        summaries[rid] = res["summary"]
        strata += res["strata"]
        by_class[rid] = res["by_class"]
        per_image[rid] = pd.read_csv(out / "phase_integral" / f"{rid}__per_image.csv")
    arch_pairs = [(f"{x}_abl_dice_only", f"{x}_abl_full") for x in cfg["architectures"]]
    arch_pairs += [(f"{x}_full_finetune", f"{x}_lora_r{cfg['lora']['selected_rank']}") for x in cfg["architectures"]]
    write_json(out / "phase_integral.json", {
        "checkpoint": primary, "per_run": summaries, "strata": strata, "by_class": by_class,
        "paired_comparisons": pi_mod.paired_comparison(cfg, per_image, arch_pairs)})
    print(f"[analysis] phase_integral: {len(summaries)} runs")

    # ---- dry mass / morphology for the selected configuration -----------------------------------------------
    sel = f"{cfg['analysis']['selected_arch']}_lora_r{cfg['analysis']['selected_rank']}"
    if sel in finished:
        preds = load_predictions(finished[sel], primary)
        cells_pred, cells_gt = dm.cells_table(cfg, val, preds), dm.cells_table(cfg, val, None)
        (out / "drymass").mkdir(exist_ok=True)
        cells_pred.to_csv(out / "drymass" / "cells_pred.csv", index=False)
        cells_gt.to_csv(out / "drymass" / "cells_gt.csv", index=False)
        thr = cfg["analysis"]["doublet_area_um2"]
        from qpi.analysis import wrapping as wrp
        names = cfg["data"]["class_names"]
        nw_pred, nw_gt = cells_pred[~cells_pred.wrapped], cells_gt[~cells_gt.wrapped]
        both = len(nw_pred) and len(nw_gt)
        write_json(out / "drymass.json", {
            "selected_run": sel, "checkpoint": primary,
            "note": "keys ending in _nowrap exclude cells flagged as phase-wrapped (DECISIONS.md D-21); the other keys use all cells",
            "wrapping_rule": wrp.rule(cfg),
            "wrapping_pred": wrp.summarise(cells_pred, names), "wrapping_gt": wrp.summarise(cells_gt, names),
            "per_day_pred_nowrap": dm.per_day_table(cfg, nw_pred) if len(nw_pred) else [],
            "per_day_gt_nowrap": dm.per_day_table(cfg, nw_gt) if len(nw_gt) else [],
            "agreement_nowrap": dm.agreement(nw_pred, nw_gt) if both else None,
            "storage_pred_nowrap": dm.storage_statistics(cfg, nw_pred) if len(nw_pred) else None,
            "storage_pred_excluding_large_nowrap": dm.storage_statistics(cfg, nw_pred, thr) if len(nw_pred) else None,
            "storage_gt_nowrap": dm.storage_statistics(cfg, nw_gt) if len(nw_gt) else None,
            "consistency_pred_nowrap": dm.consistency_checks(cfg, nw_pred),
            "consistency_gt_nowrap": dm.consistency_checks(cfg, nw_gt),
            "per_day_pred": dm.per_day_table(cfg, cells_pred) if len(cells_pred) else [],
            "per_day_gt": dm.per_day_table(cfg, cells_gt) if len(cells_gt) else [],
            "agreement": dm.agreement(cells_pred, cells_gt) if len(cells_pred) and len(cells_gt) else None,
            "storage_pred": dm.storage_statistics(cfg, cells_pred) if len(cells_pred) else None,
            "storage_pred_excluding_large": dm.storage_statistics(cfg, cells_pred, thr) if len(cells_pred) else None,
            "storage_gt": dm.storage_statistics(cfg, cells_gt) if len(cells_gt) else None,
            "consistency_pred": dm.consistency_checks(cfg, cells_pred),
            "consistency_gt": dm.consistency_checks(cfg, cells_gt)})
        print(f"[analysis] drymass: {len(cells_pred)} predicted / {len(cells_gt)} annotated cells")
    else:
        print(f"[analysis] drymass skipped: {sel} not finished")

    # ---- robustness ---------------------------------------------------------------------------------------------
    if not a.skip_robustness:
        from qpi.utils.env import enforce_gpu_policy, resolve_device
        enforce_gpu_policy(cfg["hardware"]["allowed_physical_gpus"])
        device = resolve_device(a.gpu)
        from qpi.analysis import cell_analysis as ca
        from qpi.analysis import robustness as rb
        from qpi.utils.layout import ROOT, analysis_dir
        from qpi.utils.progress import Progress
        sel_rank = cfg["lora"]["selected_rank"]
        ids = [f"{x}_lora_r{sel_rank}" for x in cfg["architectures"]] + [f"{x}_abl_dice_only" for x in cfg["architectures"]]
        ids = [i for i in ids if i in finished]
        refs = [r for r in ca.expected_runs(cfg, root, ROOT / "outputs" / "multiseed", list(cfg["architectures"]), [sel_rank],
                                            [int(cfg["seed"])], ["full", "dice_only"]) if r.run_id in ids]
        store = analysis_dir("robustness", ROOT)
        prog = Progress("robustness", len(refs))
        rb.run_robustness_units(cfg, refs, primary, device, cfg["paths_abs"]["weights_dir"], store, prog, ca.unit_label)
        ok = rb.write_legacy_robustness(store, ids, primary, out)
        print(f"[analysis] robustness: {len(ids)} runs, individual results in {store}")
        return prog.finish(extra_ok=ok, extra_message="robustness.json could not be rebuilt (a run is missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
