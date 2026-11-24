"""W-10  The three quantities the manuscript quotes that no current analysis emits.

Each of these appears in the revised manuscript but could not be reproduced from any file in the
project, so they are computed here from the per-run cell files and written with provenance:

  A  the 90th percentile of the matched-cell integrated-phase error
     ("one matched cell in ten had an integrated-phase error above ...")
  B  the median absolute relative error (MdARE) of the per-image median dry mass, prediction
     against annotation ("the per-image median dry mass differed by an MdARE of ...")

W-7 part B already writes the per-image class confusion (`w7b_class_confusion_per_image.csv`), so this
script does not repeat it.

Read-only: reads results/<Arch>/<cond>/dice_cipp/seed<seed>/cells_last.json and writes only below
additional_analysis/results/w10_claim_checks/.

    python additional_analysis/w10_claim_checks.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPRESENTATIVE, ROOT, SEEDS, fmt, md_table, out_dir, provenance,  # noqa: E402
                    results_dir, write_json, write_text)

ARM = "dice_cipp"


def cells(arch: str, cond: str, seed: int) -> pd.DataFrame:
    p = results_dir(arch, cond, ARM, seed) / "cells_last.json"
    if not p.exists():
        raise SystemExit(f"[w10] {p} is missing.")
    C = pd.DataFrame(json.loads(p.read_text())["cells"])
    need = ["instances", "scored", "matched", "e_matched", "dm_pred_pg", "dm_gt_pg", "stem"]
    miss = [c for c in need if c not in C]
    if miss:
        raise SystemExit(f"[w10] cells_last.json has no column(s) {miss}. Present: {sorted(C.columns)}")
    return C[(C["instances"] == "A2") & (C["scored"] == True)].copy()  # noqa: E712


def main() -> None:
    ap = argparse.ArgumentParser()
    a = ap.parse_args()

    rows_a: List[Dict] = []
    rows_b: List[Dict] = []
    inputs: List[str] = []

    for arch, cond in REPRESENTATIVE.items():
        for seed in SEEDS:
            C = cells(arch, cond, seed)
            inputs.append(str((results_dir(arch, cond, ARM, seed) / "cells_last.json").relative_to(ROOT)))

            # ---- A: 90th percentile of the matched-cell error ----
            e = np.abs(C.loc[C["matched"].fillna(False).astype(bool), "e_matched"].astype(float)) * 100.0
            e = e[np.isfinite(e)]
            rows_a.append({"arch": arch, "rank": cond, "seed": seed, "n_matched": int(len(e)),
                           "median_pct": float(np.median(e)) if len(e) else None,
                           "p90_pct": float(np.percentile(e, 90)) if len(e) else None,
                           "p95_pct": float(np.percentile(e, 95)) if len(e) else None})

            # ---- B: MdARE of the per-image median dry mass ----
            rel = []
            for stem, g in C.groupby("stem"):
                m = g["matched"].fillna(False).astype(bool)
                gp = g.loc[m, "dm_pred_pg"].astype(float); gt = g.loc[m, "dm_gt_pg"].astype(float)
                gp = gp[np.isfinite(gp)]; gt = gt[np.isfinite(gt)]
                if len(gp) < 1 or len(gt) < 1:
                    continue
                a_, b_ = float(np.median(gp)), float(np.median(gt))
                if b_ != 0:
                    rel.append(abs(a_ - b_) / abs(b_) * 100.0)
            rows_b.append({"arch": arch, "rank": cond, "seed": seed, "n_images": len(rel),
                           "mdare_pct": float(np.median(rel)) if rel else None})

            print(f"[w10] {arch} {cond} seed{seed}: done", flush=True)


    d = out_dir("w10_claim_checks")
    write_json(d / "w10_claim_checks.json",
               {"meta": provenance("w10_claim_checks.py", inputs,
                                   {"representative": REPRESENTATIVE, "arm": ARM,
                                    "note": "A: percentiles of |matched-cell IPE|; B: MdARE of the per-image "
                                            "median dry mass; C: per-image class confusion"}),
                "a_matched_cell_percentiles": rows_a, "b_per_image_mdare": rows_b})

    L = ["# W-10  Quantities quoted in the manuscript that no other analysis emits", "",
         f"Representative ranks: {REPRESENTATIVE}. Objective: {ARM}. Last checkpoint, A2 instances.", "",
         "## A. Matched-cell integrated-phase error: median and upper percentiles", ""]
    L += [md_table(["Configuration", "Matched cells", "Median (%)", "90th percentile (%)", "95th percentile (%)"],
                   [[f"{r['arch']} {r['rank']} seed{r['seed']}", f"{r['n_matched']:,}",
                     fmt(r["median_pct"], 2), fmt(r["p90_pct"], 1), fmt(r["p95_pct"], 1)] for r in rows_a]), ""]
    p90 = [r["p90_pct"] for r in rows_a if r["p90_pct"] is not None]
    if p90:
        L += [f"Across the nine runs the 90th percentile ranges from {min(p90):.1f} % to {max(p90):.1f} %.", ""]
    L += ["## B. MdARE of the per-image median dry mass, prediction against annotation", ""]
    L += [md_table(["Configuration", "Images", "MdARE (%)"],
                   [[f"{r['arch']} {r['rank']} seed{r['seed']}", str(r["n_images"]), fmt(r["mdare_pct"], 2)]
                    for r in rows_b]), ""]
    md = [r["mdare_pct"] for r in rows_b if r["mdare_pct"] is not None]
    if md:
        L += [f"Across the nine runs the MdARE ranges from {min(md):.2f} % to {max(md):.2f} %.", ""]
    write_text(d / "w10_summary.md", "\n".join(L))
    print(f"[w10] wrote {d.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
