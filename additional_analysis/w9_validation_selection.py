#!/usr/bin/env python
"""W-9: select the loss configuration and the LoRA rank on the 41 VALIDATION images instead of the test images.

Why: in the manuscript the proposed objective and the representative ranks were chosen after looking at the 66 test
images (Section 3.5), so the reported results at those configurations are exploratory. This script repeats the choice
on the validation images (fold 0 of the training split; never trained on) and then reads, without changing them, the
existing test results at the configurations chosen that way.

Two steps:

  1. --infer   (GPU, inference only)  For every finished run, load checkpoints/last.pt, predict the 41 validation
               images and score them with the same code as the test evaluation (qpicell.evaluation.evaluate.
               score_predictions: watershed cells, scored cells, cells within 5 %, Dice, lost classes).
               One small JSON per run in additional_analysis/results/w9_validation_selection/runs/. Resumable:
               runs that already have a JSON are skipped unless --force.
  2. --select  (CPU)  Apply the selection rules to the validation results and look up the test results of the chosen
               configurations in results/<Arch>/<condition>/<arm>/seed*/metrics_last.json.
               Writes w9_validation_selection.json and w9_summary.md.

Nothing is trained, no configuration is changed, and results/ and training_results/ are only read. The 'best.pt'
checkpoint is NOT used, because it was itself chosen on these validation images.

Selection rules (all on validation, LoRA ranks only):
  current   the rule of Section 3.5: discard ranks at which any seed lost a class (Dice < 0.05), then take the
            highest mean share of scored cells within 5 % (smaller rank on a tie); if every rank lost a class,
            take the highest share and mark the rule as not applied.
  balanced  among ranks at which no seed lost a class, the highest mean of (share within 5 % / 100) and mean Dice;
            same fallback. (Placeholder for the rule Prof. Moon prefers - change BALANCED_SCORE below if needed.)
Objective check: per architecture, the seven loss configurations on validation over the 12 LoRA runs
(mean share within 5 %, and the proposed objective against each other configuration: median paired difference,
pairs better, exact Wilcoxon p, Holm over the six comparisons).

Run from the project root on the server:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python additional_analysis/w9_validation_selection.py --infer --arms dice_cipp
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python additional_analysis/w9_validation_selection.py --infer
    python additional_analysis/w9_validation_selection.py --select
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

from common import ARMS, ARM_LABEL, REPRESENTATIVE, fmt, fmt_p, holm, md_table, out_dir, provenance, wilcoxon_signed_rank_exact, write_json, write_text  # noqa: E402

OUT = out_dir("w9_validation_selection")
RUNS_OUT = OUT / "runs"
DISPLAY = {"edge_sam": "EdgeSAM", "mobile_sam": "MobileSAM", "mobilenet_unet": "MobileNet_UNet"}
RANKS = ["r8", "r16", "r32", "r64"]


def BALANCED_SCORE(within5_pct: float, mean_dice: float) -> float:
    return 0.5 * (within5_pct / 100.0 + mean_dice)


# ----------------------------------------------------------------------------------------------------------- infer
def infer(a) -> None:
    import torch
    from qpicell.cli import add_common, config  # noqa: F401
    from qpicell.data.dataset import PhaseDataset, read_manifest
    from qpicell.evaluation.evaluate import predict, score_predictions
    from qpicell.io import read_json
    from qpicell.models import build
    from qpicell.physics import assert_physics, split_stems, verify_prepared
    from qpicell.training.runs import grid, training_dir

    cfg = config(a); assert_physics(cfg); verify_prepared(cfg)
    dev = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda:0")
    stems = split_stems(cfg)["validation"]
    sha = {r["stem"]: r.get("phase_sha256") for r in read_manifest(cfg["paths_abs"]["prepared"], "train")}
    pl = lambda s, f=str: [f(x) for x in s.split(",")] if s else None
    runs = grid(cfg, pl(a.archs), pl(a.conditions), pl(a.arms), pl(a.seeds, int))
    print(f"[w9] {len(runs)} runs, {len(stems)} validation images, device {dev}", flush=True)
    datasets = {}
    for k, r in enumerate(runs, 1):
        disp = DISPLAY[r.arch]
        dst = RUNS_OUT / disp / r.condition / r.arm / f"seed{r.seed}.json"
        if dst.exists() and not a.force:
            continue
        tdir = training_dir(cfg, r); tj = tdir / "training.json"; ck = tdir / "checkpoints" / "last.pt"
        if not tj.exists() or read_json(tj).get("status") != "complete" or not ck.exists():
            print(f"[w9] skip {r.run_id}: not complete or no last.pt", flush=True); continue
        m = cfg["models"][r.arch]
        if r.arch not in datasets:
            datasets[r.arch] = PhaseDataset(cfg, "train", m["image_size"], [0], return_native=True, stems=stems)
        model = build(cfg, r.arch, r.rank, pretrained=False)
        state = torch.load(ck, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"], strict=True)
        t0 = time.time()
        preds = predict(model, datasets[r.arch], m["batch_size"], dev)
        res = score_predictions(cfg, preds, sha)
        sem = res["semantic"]; A2 = res["instances"]["A2"]
        rec = {"run_id": r.run_id, "arch": disp, "condition": r.condition, "arm": r.arm, "seed": r.seed,
               "checkpoint": "last", "checkpoint_epoch": int(state.get("epoch", -1)), "n_images": len(preds),
               "mean_dice": sem["mean_dice"], "bf1": sem.get("bf1"), "ipe_global": sem.get("ipe_global"),
               "n_lost_classes": sem["n_lost_classes"], "lost_classes": sem["lost_classes"],
               "dice_per_class": {c: sem.get(f"dice_{c}") for c in ("discocyte", "spherocyte", "echinocyte", "stomatocyte")},
               "n_images_with_class": {c: sem.get(f"n_images_with_{c}") for c in ("discocyte", "spherocyte", "echinocyte", "stomatocyte")},
               "cells": A2["cells"], "reference_cells": res["reference_cells"], "seconds": time.time() - t0}
        write_json(dst, rec)
        w5 = A2["cells"]["scored"].get("within_5pct_of_all_cells_pct")
        print(f"[w9] {k}/{len(runs)} {r.run_id}: Dice {sem['mean_dice']:.3f}, within 5 % {w5:.1f}, lost {sem['n_lost_classes']}", flush=True)
        del model, preds
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------- select
def load_val():
    rows = []
    for f in sorted(RUNS_OUT.rglob("seed*.json")):
        d = json.loads(f.read_text())
        rows.append(dict(arch=d["arch"], condition=d["condition"], arm=d["arm"], seed=d["seed"], mean_dice=d["mean_dice"],
                         lost=d["n_lost_classes"], within5=d["cells"]["scored"]["within_5pct_of_all_cells_pct"],
                         n_scored=d["reference_cells"]["scored"]))
    return rows


def test_metrics(arch, cond, arm, seed):
    f = ROOT / "results" / arch / cond / arm / f"seed{seed}" / "metrics_last.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text()); A = m["instances"]["A2"]["cells"]["scored"]
    return dict(within5=A.get("within_5pct_of_all_cells_pct"), matched_ipe=A.get("cell_ipe_abs_median_pct"),
                mean_dice=m["semantic"]["mean_dice"], lost=m["semantic"]["n_lost_classes"])


def choose(rows, arch, arm, rule):
    """Apply one rank rule to the validation results.

    `current`  the earlier rule: the share of cells within 5 % alone, behind a binary stability filter that
               falls back to scoring every rank when no rank kept all classes. Kept so the previously reported
               selection can be reproduced.
    `balanced` the rule now in `configs/config.yaml`: graded stability (keep the ranks at which the FEWEST
               seeds lost a class), then the mean of the share within 5 % / 100 and the mean Dice, smaller rank
               on a tie. This matches `scripts/evaluation/objective_and_rank_selection.py`, so the validation
               selection and the test-set selection can never disagree about what the rule is.
    """
    stats = {}
    for c in RANKS:
        L = [r for r in rows if r["arch"] == arch and r["arm"] == arm and r["condition"] == c]
        if len(L) < 3:
            continue
        stats[c] = dict(within5=float(np.mean([r["within5"] for r in L])), mean_dice=float(np.mean([r["mean_dice"] for r in L])),
                        seeds_lost=sum(1 for r in L if r["lost"] > 0),
                        any_lost=any(r["lost"] > 0 for r in L), n_seeds=len(L))
    if not stats:
        return None, stats, False
    ok = [c for c in stats if not stats[c]["any_lost"]]
    applied = bool(ok)
    if rule == "current":
        pool = ok or list(stats)
        key = lambda c: stats[c]["within5"]                                            # noqa: E731
    else:
        fewest = min(stats[c]["seeds_lost"] for c in stats)
        pool = [c for c in stats if stats[c]["seeds_lost"] == fewest]
        key = lambda c: BALANCED_SCORE(stats[c]["within5"], stats[c]["mean_dice"])     # noqa: E731
    best = max(pool, key=lambda c: (round(key(c), 10), -RANKS.index(c)))   # smaller rank on a tie
    return best, stats, applied


def select(a) -> None:
    rows = load_val()
    if not rows:
        sys.exit("no validation results: run --infer first")
    archs = sorted({r["arch"] for r in rows})
    out = {"meta": provenance("w9_validation_selection.py", [str(RUNS_OUT.relative_to(ROOT))],
                              {"note": "validation = fold 0 of the training split (41 images); last checkpoint; test results only read"}),
           "rank_selection": {}, "objective_on_validation": {}}
    md = ["# W-9 Selection on the validation images (41 images, last checkpoint)\n"]
    # ---- rank selection under the proposed objective
    hdr = ["Architecture", "Rule", "Rank chosen on validation", "Rule applied", "Rank chosen on test (manuscript)",
           "Test: cells within 5 % (%)", "Test: mean Dice", "Test: runs that lost a class"]
    tab = []
    for arch in archs:
        out["rank_selection"][arch] = {}
        for rule in ("current", "balanced"):
            best, stats, applied = choose(rows, arch, "dice_cipp", rule)
            t = [test_metrics(arch, best, "dice_cipp", s) for s in (42, 43, 44)] if best else []
            t = [x for x in t if x]
            tw = float(np.mean([x["within5"] for x in t])) if t else None
            td = float(np.mean([x["mean_dice"] for x in t])) if t else None
            tl = sum(x["lost"] > 0 for x in t) if t else None
            out["rank_selection"][arch][rule] = dict(chosen=best, rule_applied=applied, validation=stats,
                                                     test_at_chosen=dict(within5=tw, mean_dice=td, runs_lost=tl, per_seed=t))
            tab.append([arch, rule, best or "–", "yes" if applied else "no (every rank lost a class)",
                        REPRESENTATIVE.get(arch, "–"), fmt(tw, 1), fmt(td, 3), f"{tl} of {len(t)}" if t else "–"])
    md += ["## Rank under the proposed objective\n", md_table(hdr, tab), "\nValidation values per rank are in the JSON.\n"]
    # ---- objective check on validation
    for arch in archs:
        means = {}
        for arm in ARMS:
            L = [r["within5"] for r in rows if r["arch"] == arch and r["arm"] == arm and r["condition"] in RANKS]
            means[arm] = (float(np.mean(L)), len(L)) if L else (None, 0)
        comps, ps = [], []
        prop = {(r["condition"], r["seed"]): r["within5"] for r in rows if r["arch"] == arch and r["arm"] == "dice_cipp" and r["condition"] in RANKS}
        for arm in ARMS:
            if arm == "dice_cipp":
                continue
            oth = {(r["condition"], r["seed"]): r["within5"] for r in rows if r["arch"] == arch and r["arm"] == arm and r["condition"] in RANKS}
            keys = sorted(set(prop) & set(oth))
            d = np.array([prop[k] - oth[k] for k in keys])
            p = wilcoxon_signed_rank_exact(d) if len(d) else None
            comps.append(dict(compared=arm, n_pairs=len(d), median_diff=float(np.median(d)) if len(d) else None,
                              pairs_better=int((d > 0).sum()) if len(d) else None, p=p)); ps.append(p)
        for c, ph in zip(comps, holm(ps)):
            c["p_holm"] = ph
        best_arm = max((a_ for a_ in means if means[a_][0] is not None), key=lambda a_: means[a_][0])
        out["objective_on_validation"][arch] = dict(mean_within5=means, best_arm_on_validation=best_arm, proposed_vs=comps)
        md += [f"\n## {arch}: loss configurations on validation (12 LoRA runs)\n",
               f"Highest mean share within 5 % on validation: **{ARM_LABEL[best_arm]}**\n",
               md_table(["Compared configuration", "Mean within 5 % (%)", "Median diff. (proposed − compared)", "Pairs better", "p (Holm)"],
                        [[ARM_LABEL[c["compared"]], fmt(means[c["compared"]][0], 1), fmt(c["median_diff"], 2, True),
                          f"{c['pairs_better']}/{c['n_pairs']}" if c["n_pairs"] else "–", fmt_p(c["p_holm"])] for c in comps]),
               f"\nProposed objective: {fmt(means['dice_cipp'][0], 1)} % within 5 % on validation.\n"]
    write_json(OUT / "w9_validation_selection.json", out)
    write_text(OUT / "w9_summary.md", "\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--infer", action="store_true"); ap.add_argument("--select", action="store_true")
    ap.add_argument("--archs"); ap.add_argument("--conditions"); ap.add_argument("--arms"); ap.add_argument("--seeds")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--cpu", action="store_true")
    try:
        from qpicell.cli import add_common
        add_common(ap)
    except Exception:
        pass
    a = ap.parse_args()
    if not (a.infer or a.select):
        ap.error("give --infer and/or --select")
    if a.infer:
        infer(a)
    if a.select:
        select(a)
