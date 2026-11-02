#!/usr/bin/env python
"""Physics and biology comparison of the three architectures at the selected rank (read-only; no training, no GPU).

    python scripts/compare_architectures.py [--rank 8] [--config-overlay smoke.yaml]

For each architecture the saved predictions of ``<arch>_lora_r<rank>`` are turned into per-cell tables (the same code path as
the selected-run dry-mass analysis), wrapped cells are excluded (D-21), and the following are written to
``<output_root>/analysis/architecture_comparison.{json,csv,md}``:

  measurement   cells found, per-image dry-mass agreement (r, CCC, bias, limits of agreement, MdARE), day-0 dry mass
  biology       per-class cell counts against the annotation, composition error per storage day (percentage points),
                storage trends of area and dry mass (Spearman, early vs late) with the annotation as reference
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qpi.config import load_config  # noqa: E402
from qpi.utils.io import read_json, write_json  # noqa: E402
from qpi.utils.outputs import claim_output_root  # noqa: E402


def composition_error(pred: pd.DataFrame, gt: pd.DataFrame, num_classes: int) -> Dict:
    """Mean and maximum absolute difference (percentage points) between predicted and annotated class composition per day."""
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


def summarise_architecture(cfg: Dict, pred: pd.DataFrame, gt: pd.DataFrame) -> Dict:
    from qpi.analysis import drymass as dm
    names = cfg["data"]["class_names"]
    p, g = pred[~pred.wrapped], gt[~gt.wrapped]
    d0 = p[p.storage_day == 0]
    return {
        "cells_predicted": int(len(p)), "cells_annotated": int(len(g)), "cells_ratio": float(len(p) / len(g)),
        "agreement": dm.agreement(p, g),
        "day0_median_dry_mass_pg": float(d0.dry_mass_pg.median()) if len(d0) else None,
        "class_counts": class_count_ratio(p, g, names),
        "composition_error": composition_error(p, g, cfg["data"]["num_classes"]),
        "storage_pred": dm.storage_statistics(cfg, p), "storage_gt": dm.storage_statistics(cfg, g),
    }


def _flat(res: Dict) -> Dict:
    a, sp, sg = res["agreement"], res["storage_pred"], res["storage_gt"]
    row = {"cells_predicted": res["cells_predicted"], "cells_annotated": res["cells_annotated"],
           "pearson_r": a.get("pearson_r"), "ccc": a.get("lin_ccc"), "bias_pg": a.get("bland_altman_mean_pg"),
           "loa_low_pg": a.get("bland_altman_loa_low_pg"), "loa_high_pg": a.get("bland_altman_loa_high_pg"),
           "mdare_pct": a.get("mdare_pct"), "day0_dry_mass_pg": res["day0_median_dry_mass_pg"],
           "composition_mean_abs_pp": res["composition_error"]["mean_abs_pp"],
           "area_rho_pred": sp["area"]["spearman_rho"], "area_rho_gt": sg["area"]["spearman_rho"],
           "drymass_rho_pred": sp["dry_mass"]["spearman_rho"], "drymass_rho_gt": sg["dry_mass"]["spearman_rho"],
           "drymass_late_vs_early_pct_pred": sp["dry_mass"].get("late_vs_early_pct"),
           "drymass_late_vs_early_pct_gt": sg["dry_mass"].get("late_vs_early_pct")}
    for cname, v in res["class_counts"].items():
        row[f"count_ratio_{cname}"] = v["ratio"]
    return row


def _markdown(t: pd.DataFrame) -> str:
    cols = list(t.columns)
    lines = ["| metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in t.iterrows():
        lines.append(f"| {idx} | " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-overlay", default=None)
    ap.add_argument("--rank", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config_overlay)
    root = claim_output_root(cfg)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rank = a.rank or cfg["lora"]["selected_rank"]
    primary = cfg["evaluation"]["checkpoint"]

    from qpi.analysis import drymass as dm
    from qpi.analysis.common import load_predictions, load_validation
    from qpi.experiments import run_dir
    val = load_validation(cfg)
    cells_gt = dm.cells_table(cfg, val, None)
    results: Dict[str, Dict] = {}
    for arch in cfg["architectures"]:
        rid = f"{arch}_lora_r{rank}"
        rdir = run_dir(cfg, rid)
        m = rdir / "metrics.json"
        if not m.exists() or read_json(m)["status"] != "complete":
            print(f"[compare_architectures] skipping {rid}: not finished")
            continue
        cells_pred = dm.cells_table(cfg, val, load_predictions(rdir, primary))
        results[arch] = summarise_architecture(cfg, cells_pred, cells_gt)
        print(f"[compare_architectures] {rid}: {results[arch]['cells_predicted']} predicted cells")
    write_json(out / "architecture_comparison.json", {"checkpoint": primary, "rank": rank,
                                                      "note": "wrapped cells excluded (D-21)", "architectures": results})
    table = pd.DataFrame({k: _flat(v) for k, v in results.items()}).round(3)
    table.to_csv(out / "architecture_comparison.csv")
    (out / "architecture_comparison.md").write_text(
        f"# Architectures at r{rank} (checkpoint {primary}, wrapped cells excluded)\n\n" + _markdown(table) + "\n")
    print(table.to_string())
    print(f"[compare_architectures] -> {out / 'architecture_comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
