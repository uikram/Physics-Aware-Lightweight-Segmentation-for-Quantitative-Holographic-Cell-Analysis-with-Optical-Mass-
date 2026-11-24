#!/usr/bin/env python
"""W-5  Optical-axis superposition check and crowding per storage day, on the NEW (watershed) reference cells.

Why: the manuscript (Section 3.1) and the reply to Reviewer 2, Comment 4 report a superposition check (6 of 2049
cells) and the day-15 crowding on the OLD reference cells (4-connected components). The final pipeline defines
reference cells by a marker-controlled watershed of the annotated foreground (3,084 test cells), so both numbers
must be re-derived. Nothing here depends on a trained network: the reference-cell columns are identical in every
run's cells_last.json (checked below).

1. Superposition screen (criterion of the manuscript, unchanged):
   mean phase of a cell = S_gt / N_px (rad), N_px = area / dx^2. A cell is a CANDIDATE when its mean phase is
   >= 1.6 x the median mean phase of the cells of the same class on the same storage day AND its area is
   <= 1.25 x the median area of that group (dense but not large: two cells stacked along the optical axis).
   Sensitivity: thresholds 1.4 / 1.6 / 1.8. Candidates are listed (stem, cell, class, day, ratios) for visual
   inspection; the screen cannot tell an exactly aligned stack from a single thick cell.
2. Crowding per storage day: reference cells, touching cells (share), multi-cell clusters (connected components of
   the annotated foreground with >= 2 watershed cells), share of cells in such clusters, largest cluster,
   scored cells. If data/prepared/val/mask/*.tif is present (server), also the annotated-foreground fraction of the
   annotated field of view per image and day (as in the manuscript: "mean fraction 0.48 at day 15").

Input (read-only): results/<Arch>/<cond>/<loss>/seed<seed>/cells_last.json (reference columns only);
                   optionally data/prepared/manifest.csv and data/prepared/val/mask/*.tif
Usage:   python additional_analysis/w5_reference_cells_checks.py
Outputs: additional_analysis/results/w5_reference_cells/{w5_reference_cells.json, w5_reference_cells_summary.md,
         w5_superposition_candidates.csv}
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PIXEL_SIZE_UM, ROOT, fmt, md_table, out_dir, provenance, write_json, write_text  # noqa: E402

REF_COLS = ["stem", "cell", "storage_day", "gt_class", "area_um2", "S_gt", "scored", "touches_unannotated", "wrapped",
            "touching", "cluster"]
DEFAULT_SOURCE = "MobileSAM/r64/dice_cipp/seed42"
DEFAULT_CHECK = ["EdgeSAM/r8/dice_cipp/seed42", "MobileNet_UNet/r32/dice/seed43", "MobileSAM/r64/dice_pmc_bga_ipp/seed44"]


def ref_cells(rel: str) -> pd.DataFrame:
    C = pd.DataFrame(json.loads((ROOT / "results" / rel / "cells_last.json").read_text())["cells"])
    C = C[C["instances"] == "A2"][REF_COLS].sort_values(["stem", "cell"]).reset_index(drop=True)
    return C


def superposition(C: pd.DataFrame, phase_thr: float, area_thr: float) -> pd.DataFrame:
    C = C.copy()
    C["n_px"] = C["area_um2"] / PIXEL_SIZE_UM ** 2
    C["mean_phase_rad"] = C["S_gt"] / C["n_px"]
    g = C.groupby(["storage_day", "gt_class"])
    C["group_median_mean_phase"] = g["mean_phase_rad"].transform("median")
    C["group_median_area"] = g["area_um2"].transform("median")
    C["group_n"] = g["cell"].transform("count")
    C["phase_ratio"] = C["mean_phase_rad"] / C["group_median_mean_phase"]
    C["area_ratio"] = C["area_um2"] / C["group_median_area"]
    return C[(C["phase_ratio"] >= phase_thr) & (C["area_ratio"] <= area_thr)]


def crowding(C: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, g in C.groupby("storage_day"):
        cl = g.groupby(["stem", "cluster"])["cell"].size()
        multi = cl[cl >= 2]
        in_multi = int(multi.sum())
        rows.append({"storage_day": int(day), "images": int(g["stem"].nunique()), "reference_cells": int(len(g)),
                     "scored_cells": int(g["scored"].sum()), "touching_cells": int(g["touching"].sum()),
                     "touching_pct": 100 * float(g["touching"].mean()),
                     "multi_cell_clusters": int(len(multi)), "cells_in_multi_cell_clusters": in_multi,
                     "cells_in_multi_cell_clusters_pct": 100 * in_multi / len(g),
                     "largest_cluster_cells": int(cl.max()) if len(cl) else 0,
                     "clusters_of_3_or_more": int((cl >= 3).sum())})
    return pd.DataFrame(rows)


def foreground_fraction():
    prep = ROOT / "data" / "prepared"
    mdir = prep / "val" / "mask"
    if not (prep / "manifest.csv").exists() or not mdir.exists():
        return None
    import tifffile
    man = pd.read_csv(prep / "manifest.csv")
    man = man[(man["split"] == "val") & (man["registration_ok"].astype(str) == "True")]
    out = []
    for _, r in man.iterrows():
        m = tifffile.imread(mdir / f"{r['stem']}.tif")
        valid = m != 255; fg = (m > 0) & valid
        out.append({"stem": r["stem"], "storage_day": int(r["storage_day"]), "valid_fraction_of_grid": float(valid.mean()),
                    "foreground_fraction_of_annotated": float(fg.sum() / max(valid.sum(), 1))})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="run whose cells_last.json supplies the reference columns")
    ap.add_argument("--check", default=",".join(DEFAULT_CHECK), help="runs whose reference columns must be identical")
    a = ap.parse_args()
    C = ref_cells(a.source)
    consistency = []
    for rel in [x for x in a.check.split(",") if x]:
        p = ROOT / "results" / rel / "cells_last.json"
        if not p.exists():
            consistency.append({"run": rel, "status": "not found"}); continue
        D = ref_cells(rel)
        same = len(D) == len(C) and all(
            np.allclose(D[c].astype(float), C[c].astype(float), equal_nan=True) if C[c].dtype.kind in "fi" else D[c].equals(C[c])
            for c in REF_COLS)
        consistency.append({"run": rel, "status": "identical" if same else "DIFFERENT"})
    if any(c["status"] == "DIFFERENT" for c in consistency):
        print("[w5] WARNING: reference columns differ between runs", consistency, flush=True)

    sup = {}
    cand_main = None
    for pt in (1.4, 1.6, 1.8):
        s = superposition(C, pt, 1.25)
        sup[f"phase_ratio_ge_{pt}"] = {"phase_ratio_threshold": pt, "area_ratio_threshold": 1.25,
                                       "candidates_all_reference_cells": int(len(s)),
                                       "candidates_scored_cells": int(s["scored"].sum()),
                                       "candidates_touching": int(s["touching"].sum())}
        if pt == 1.6:
            cand_main = s
    cand_cols = ["stem", "cell", "storage_day", "gt_class", "area_um2", "mean_phase_rad", "phase_ratio", "area_ratio",
                 "group_n", "touching", "scored", "wrapped"]
    cand = cand_main[cand_cols].sort_values("phase_ratio", ascending=False)

    crowd = crowding(C)
    ff = foreground_fraction()
    if ff is not None:
        agg = ff.groupby("storage_day").agg(fg_mean=("foreground_fraction_of_annotated", "mean"),
                                           fg_min=("foreground_fraction_of_annotated", "min"),
                                           fg_max=("foreground_fraction_of_annotated", "max"),
                                           valid_mean=("valid_fraction_of_grid", "mean")).reset_index()
        crowd = crowd.merge(agg, on="storage_day", how="left")

    res = {"reference_cells": {"all": int(len(C)), "scored": int(C["scored"].sum()), "touching": int(C["touching"].sum()),
                               "images": int(C["stem"].nunique()), "days": int(C["storage_day"].nunique())},
           "consistency_of_reference_columns": consistency,
           "superposition_screen": sup,
           "superposition_candidates_ratio_1_6": cand.to_dict("records"),
           "crowding_per_day": crowd.to_dict("records"),
           "foreground_fraction_available": ff is not None,
           "foreground_fraction_per_image": None if ff is None else ff.to_dict("records"),
           "mean_unannotated_fraction_of_grid": None if ff is None else float(1 - ff["valid_fraction_of_grid"].mean())}
    meta = provenance("w5_reference_cells_checks.py", [f"results/{a.source}/cells_last.json"] +
                      [f"results/{c['run']}/cells_last.json" for c in consistency] +
                      (["data/prepared/manifest.csv", "data/prepared/val/mask/*.tif"] if ff is not None else []),
                      {"pixel_size_um": PIXEL_SIZE_UM,
                       "criterion": "mean phase >= k x median of same class & day, area <= 1.25 x median (k = 1.6 main; 1.4, 1.8 sensitivity)"})
    od = out_dir("w5_reference_cells")
    write_json(od / "w5_reference_cells.json", {"metadata": meta, "results": res})
    buf = io.StringIO(); cand.to_csv(buf, index=False); write_text(od / "w5_superposition_candidates.csv", buf.getvalue())

    L = ["# W-5 Reference-cell checks on the watershed cells", "",
         f"{res['reference_cells']['all']:,} reference cells ({res['reference_cells']['scored']:,} scored, "
         f"{res['reference_cells']['touching']:,} touching) in {res['reference_cells']['images']} test images, "
         f"{res['reference_cells']['days']} storage days. Source: results/{a.source}/cells_last.json.", "",
         "Reference columns identical in: " + ", ".join(f"{c['run']} ({c['status']})" for c in consistency), "",
         "## Superposition screen", "",
         md_table(["mean-phase ratio ≥", "area ratio ≤", "candidates (all cells)", "of which scored", "of which touching"],
                  [[v["phase_ratio_threshold"], v["area_ratio_threshold"], v["candidates_all_reference_cells"],
                    v["candidates_scored_cells"], v["candidates_touching"]] for v in sup.values()]), "",
         "Candidates at ratio 1.6 (inspect visually; listed in w5_superposition_candidates.csv):", "",
         md_table(["stem", "cell", "day", "class", "area (µm²)", "mean phase (rad)", "phase ratio", "area ratio", "touching"],
                  [[r["stem"], r["cell"], r["storage_day"], r["gt_class"], fmt(r["area_um2"], 1), fmt(r["mean_phase_rad"], 3),
                    fmt(r["phase_ratio"], 2), fmt(r["area_ratio"], 2), r["touching"]] for r in res["superposition_candidates_ratio_1_6"]]), "",
         "## Crowding per storage day", ""]
    hdr = ["day", "images", "reference cells", "scored", "touching (%)", "multi-cell clusters", "cells in clusters (%)",
           "largest cluster", "clusters ≥ 3 cells"]
    rows = [[r["storage_day"], r["images"], r["reference_cells"], r["scored_cells"], fmt(r["touching_pct"], 1),
             r["multi_cell_clusters"], fmt(r["cells_in_multi_cell_clusters_pct"], 1), r["largest_cluster_cells"],
             r["clusters_of_3_or_more"]] for r in res["crowding_per_day"]]
    if ff is not None:
        hdr += ["foreground fraction of annotated field, mean [min, max]"]
        for row, r in zip(rows, res["crowding_per_day"]):
            row.append(f"{r['fg_mean']:.2f} [{r['fg_min']:.2f}, {r['fg_max']:.2f}]")
    L += [md_table(hdr, rows), ""]
    if ff is None:
        L.append("Foreground fractions not computed: data/prepared is not present on this machine (run on the server).")
    else:
        L.append(f"Mean unannotated fraction of the 768 × 768 grid over the test images: {res['mean_unannotated_fraction_of_grid']:.3f}.")
    write_text(od / "w5_reference_cells_summary.md", "\n".join(L))


if __name__ == "__main__":
    main()
