#!/usr/bin/env python
"""Storage-time analysis (CPU; reads data/prepared and each run's saved predictions, trains nothing).
    python scripts/evaluation/storage_trends.py --workers 16                  # every run with predictions_last.npz
    python scripts/evaluation/storage_trends.py --seeds 42 --archs edge_sam   # a subset
    python scripts/evaluation/storage_trends.py --no-exact                   # Monte Carlo permutation (fast)

Annotation: area and dry mass of the scored reference cells vs storage day (Spearman, exact 11! date-block
permutation, date bootstrap, leave-one-day-out, early vs late), per-day summary and class composition.
Every run: the same on its usable predicted cells (A2 instances rebuilt from predictions_<tag>.npz with h = 3), plus
the trend of (prediction - annotation) per image, which tests whether a predicted trend is a segmentation bias.
Outputs, results/consolidated/storage/:
  annotation.json            trends of the reference cells + per-day table (cells, images, mean / SD, class %)
  runs_<tag>.json            one record per run: architecture, condition, loss, seed, trends, bias trends, per-day medians
  summary_<tag>.json         per architecture x condition x loss: mean / SD over seeds of rho and early-to-late change
Associations with storage time in one sample series only; no biological mechanism is inferred."""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from qpicell.cli import add_common, config, parse_list  # noqa: E402
from qpicell.io import write_json  # noqa: E402
from qpicell.physics import assert_physics, split_stems  # noqa: E402
from qpicell.training.runs import grid, results_dir  # noqa: E402


def _days(cfg):
    with open(Path(cfg["paths_abs"]["prepared"]) / "manifest.csv") as fh:
        return {r["stem"]: int(r["storage_day"]) for r in csv.DictReader(fh) if r["split"] == "val"}


def _job(args):
    cfg, run, tag, exact, ann_records = args
    import pandas as pd
    from qpicell.evaluation.storage import predicted_cells, run_trends
    f = results_dir(cfg, run) / f"predictions_{tag}.npz"
    P = predicted_cells(cfg, f, _days(cfg), cfg["storage"]["instances"])
    if P.empty:
        return None
    return {**run.to_dict(), "checkpoint": tag, "instances": cfg["storage"]["instances"],
            **run_trends(cfg, P, pd.DataFrame(ann_records), exact)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="last"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--archs"); ap.add_argument("--conditions"); ap.add_argument("--arms"); ap.add_argument("--seeds")
    add_common(ap)
    a = ap.parse_args(); cfg = config(a); assert_physics(cfg)
    from qpicell.evaluation.storage import annotation_cells, per_image, trend
    out = Path(cfg["paths_abs"]["results"]) / "consolidated" / "storage"; out.mkdir(parents=True, exist_ok=True)
    days = _days(cfg); stems = split_stems(cfg)["test"]
    print(f"[storage] {len(stems)} test images, {len(set(days[s] for s in stems))} storage days", flush=True)
    A = annotation_cells(cfg, stems, days)
    ann = per_image(A, "area_um2", "dm_gt_pg")
    per_day = A.groupby("storage_day").agg(n_cells=("cell", "size"), n_images=("stem", "nunique"), area_mean=("area_um2", "mean"),
                                           area_sd=("area_um2", "std"), dry_mass_mean=("dm_gt_pg", "mean"), dry_mass_sd=("dm_gt_pg", "std")).reset_index()
    for cls in cfg["data"]["class_names"][1:]:
        per_day[f"pct_{cls}"] = A.groupby("storage_day").gt_class.apply(lambda s: 100 * (s == cls).mean()).values
    res = {"n_scored_cells": int(len(A)), "trends": {m: trend(ann, m, cfg, not a.no_exact) for m in ("area", "dry_mass")},
           "per_day": per_day.to_dict("records"), "per_image": ann.to_dict("records"),
           "note": "associations with storage time in one sample series; not evidence of a biological mechanism"}
    write_json(out / "annotation.json", res)
    t = res["trends"]["dry_mass"]
    print(f"[storage] annotation: {len(A)} scored cells | dry mass rho {t['rho']:+.3f} (block-permutation p {t['p_block_permutation']:.4f}) | "
          f"area rho {res['trends']['area']['rho']:+.3f}", flush=True)
    runs = [r for r in grid(cfg, parse_list(a.archs), parse_list(a.conditions), parse_list(a.arms), parse_list(a.seeds, int))
            if (results_dir(cfg, r) / f"predictions_{a.checkpoint}.npz").exists()]
    print(f"[storage] {len(runs)} runs with predictions_{a.checkpoint}.npz; workers {a.workers}; "
          f"{'Monte Carlo' if a.no_exact else 'exact 11!'} permutation", flush=True)
    recs = ann.to_dict("records"); rows = []
    with ProcessPoolExecutor(max(1, a.workers)) as ex:
        for i, r in enumerate(ex.map(_job, [(cfg, r, a.checkpoint, not a.no_exact, recs) for r in runs]), 1):
            if r is not None:
                rows.append(r)
                print(f"  {i}/{len(runs)} {r['run_id']}: dry mass rho {r['dry_mass']['rho']:+.3f} "
                      f"(bias rho {r['dry_mass']['bias_vs_annotation']['rho']:+.3f}) | area rho {r['area']['rho']:+.3f}", flush=True)
    write_json(out / f"runs_{a.checkpoint}.json", {"checkpoint": a.checkpoint, "n": len(rows), "runs": rows})
    groups = {}
    for r in rows:
        groups.setdefault((r["arch"], r["condition"], r["arm"]), []).append(r)
    summ = []
    for (arch, cond, arm), xs in sorted(groups.items()):
        row = {"architecture": arch, "condition": cond, "loss": arm, "seeds": sorted(x["seed"] for x in xs)}
        for m in ("area", "dry_mass"):
            for k in ("rho", "early_to_late_change_pct"):
                v = np.array([x[m][k] for x in xs if x[m][k] is not None], float)
                row[f"{m}_{k}_mean"] = float(np.nanmean(v)) if len(v) else None
                row[f"{m}_{k}_sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else None
        summ.append(row)
    write_json(out / f"summary_{a.checkpoint}.json", {"checkpoint": a.checkpoint, "annotation": res["trends"], "groups": summ})
    print(f"[storage] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
