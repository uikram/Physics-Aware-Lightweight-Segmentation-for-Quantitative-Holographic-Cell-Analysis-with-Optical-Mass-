#!/usr/bin/env python
"""Collect every per-run result JSON into consolidated JSON files (CPU, reads only results/).
    python scripts/evaluation/consolidate.py

results/consolidated/
  runs_<tag>.json        one flat record per run (architecture, condition, rank, loss, seed + every key metric)
  seed_summary_<tag>.json   per architecture x condition x loss: mean, SD, min, max and n over the seeds
  matrix_status.json     which of the 315 runs have results (and which are missing)
tag = last (primary) and best. Flat metric names are those written by qpicell/evaluation/.

Nothing here is recomputed: every field is copied out of a run's metrics_<tag>.json. Per-cell shape measurements
(perimeter, circularity) are deliberately NOT summarised here, because they live in cells_<tag>.json and adding
them to the run records would not change metrics_<tag>.json, which scripts/evaluation/rescore.py checks for exact
reproduction."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from qpicell.cli import add_common, config  # noqa: E402
from qpicell.io import write_json  # noqa: E402
from qpicell.training.runs import grid, results_dir  # noqa: E402


def g(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return None
        d = d[k]
    return d


def flatten(M: dict) -> dict:
    md, sem = M["metadata"], M["semantic"]; run = md["run"]; model = md.get("model", {}); par = model.get("parameters", {})
    r = {"run_id": run["run_id"], "architecture": md["architecture"], "architecture_display": md["architecture_display"],
         "condition": run["condition"], "rank": run["rank"], "full_finetune": run["full_finetune"], "loss": run["arm"], "seed": run["seed"],
         "checkpoint": md["checkpoint"], "checkpoint_epoch": md["checkpoint_epoch"], "n_images": md["n_images"],
         "params_total": par.get("total"), "params_trainable": par.get("trainable"), "params_adapter": par.get("lora_parameters"),
         "adapted_layers": par.get("lora_layers"), "learning_rate": model.get("learning_rate"),
         "train_seconds": model.get("train_seconds"), "seconds_per_epoch": model.get("seconds_per_epoch_median"),
         "best_epoch": g(model, "best_epoch", "epoch"),
         "dice_macro": sem["mean_dice"], "iou_macro": sem["mean_iou"], "aji_cc": sem["aji"], "bf1": sem["bf1"],
         "ipe_global": sem["ipe_global"], "n_lost_classes": sem["n_lost_classes"]}
    for c in md["class_names"][1:]:
        r[f"dice_{c}"] = sem.get(f"dice_{c}")
        r[f"iou_{c}"] = sem.get(f"iou_{c}")
        r[f"ipe_{c}"] = sem.get(f"ipe_{c}")
        r[f"n_images_with_{c}"] = sem.get(f"n_images_with_{c}")
        # which classes were lost, not only how many: the manuscript reports this per class
        r[f"lost_{c}"] = bool(any(str(x).startswith(f"{c} (") for x in (sem.get("lost_classes") or [])))
    r["lost_classes"] = ", ".join(str(x) for x in (sem.get("lost_classes") or []))
    for mode in ("A2", "CC"):
        X = M["instances"][mode]; p = mode.lower()
        r[f"{p}_aji_cells"] = X["aji_instances_vs_reference_cells"]
        r[f"{p}_unassigned_px"] = X["assignment"]["unassigned_fg_px"]
        for grp in ("scored", "touching", "isolated"):
            G = X["cells"][grp]
            for k in ("n", "match_pct", "missed_pct", "merged_pct", "fragmented_pct", "cell_ipe_signed_median_pct",
                      "cell_ipe_abs_median_pct", "cell_ipe_abs_p90_pct", "within_5pct_of_all_cells_pct",
                      "network_cell_ipe_signed_median_pct", "network_cell_ipe_abs_median_pct"):
                r[f"{p}_{grp}_{k}"] = G.get(k)
            r[f"{p}_{grp}_drymass_mdare_pct"] = g(G, "dry_mass_matched", "mdare_pct")
            r[f"{p}_{grp}_drymass_ccc"] = g(G, "dry_mass_matched", "ccc")
            r[f"{p}_{grp}_drymass_pearson_r"] = g(G, "dry_mass_matched", "pearson_r")
            r[f"{p}_{grp}_drymass_r_expected_from_sd"] = g(G, "dry_mass_matched", "r_expected_from_sd")
            r[f"{p}_{grp}_drymass_loa_low"] = g(G, "dry_mass_matched", "loa_low")
            r[f"{p}_{grp}_drymass_loa_high"] = g(G, "dry_mass_matched", "loa_high")
            r[f"{p}_{grp}_drymass_bias_pg"] = g(G, "dry_mass_matched", "bias")
            r[f"{p}_{grp}_drymass_sum_diff_pct"] = g(G, "dry_mass_matched", "sum_diff_pct")
            r[f"{p}_{grp}_optvol_mdare_pct"] = g(G, "optical_volume_matched", "mdare_pct")
            r[f"{p}_{grp}_optvol_pearson_r"] = g(G, "optical_volume_matched", "pearson_r")
        for k, v in X["clusters"].items():
            r[f"{p}_cluster_{k}"] = v
        for k, v in X["oversplit"].items():
            r[f"{p}_{k}"] = v
        for k, v in X["phase"].items():
            r[f"{p}_phase_{k}"] = v
        r[f"{p}_drymass_population_diff_pct"] = g(X, "dry_mass", "population", "relative_difference_of_medians_pct")
        r[f"{p}_drymass_image_median_mdare_pct"] = g(X, "dry_mass", "per_image_median", "mdare_pct")
        r[f"{p}_n_instances"] = X["n_predicted_instances"]; r[f"{p}_false_positive_instances"] = X["n_false_positive_instances"]
    return r


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); add_common(ap)
    a = ap.parse_args(); cfg = config(a)
    out = Path(cfg["paths_abs"]["results"]) / "consolidated"
    runs = grid(cfg); status = {"n_runs": len(runs), "with_results": {}, "missing": {}}
    for tag in cfg["evaluation"]["checkpoints"]:
        recs, missing = [], []
        for r in runs:
            f = results_dir(cfg, r) / f"metrics_{tag}.json"
            if f.exists():
                recs.append(flatten(json.loads(f.read_text())))
            else:
                missing.append(r.run_id)
        status["with_results"][tag] = len(recs); status["missing"][tag] = missing
        write_json(out / f"runs_{tag}.json", {"checkpoint": tag, "n": len(recs), "runs": recs})
        groups = {}
        for x in recs:
            groups.setdefault((x["architecture"], x["condition"], x["loss"]), []).append(x)
        summ = []
        for (arch, cond, loss), xs in sorted(groups.items()):
            row = {"architecture": arch, "architecture_display": xs[0]["architecture_display"], "condition": cond, "rank": xs[0]["rank"],
                   "full_finetune": xs[0]["full_finetune"], "loss": loss, "seeds": sorted(x["seed"] for x in xs), "n_seeds": len(xs), "metrics": {}}
            for k, v in xs[0].items():
                vals = [x[k] for x in xs if isinstance(x.get(k), (int, float)) and not isinstance(x.get(k), bool) and x.get(k) is not None]
                if vals and k not in ("seed", "rank"):
                    arr = np.asarray(vals, float)
                    row["metrics"][k] = {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                                         "min": float(arr.min()), "max": float(arr.max()), "n": int(len(arr))}
            summ.append(row)
        write_json(out / f"seed_summary_{tag}.json", {"checkpoint": tag, "groups": summ})
        print(f"[consolidate] {tag}: {len(recs)} / {len(runs)} runs with results -> {out}", flush=True)
    write_json(out / "matrix_status.json", status)


if __name__ == "__main__":
    main()
