"""W-8  Signed and absolute integrated-phase errors of matched cells, touching against isolated.

Feeds Table S16 and the signed-error sentences of Section 4.1.

Read-only: reads results/<Arch>/<cond>/<arm>/seed<seed>/cells_last.json and writes only
additional_analysis/results/w8_signed_errors/.

The representative rank comes from common.REPRESENTATIVE, which reads
results/consolidated/selection/rank_selection.csv. It is never written here as a literal, so this
analysis cannot report a rank the selection no longer uses.

    python additional_analysis/w8_signed_errors.py
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

from common import (ARM_LABEL, DISPLAY_ARCH, REPRESENTATIVE, ROOT, SEEDS, fmt, md_table, out_dir,
                    provenance, results_dir, write_json, write_text)

ARMS_COMPARED = ["dice", "dice_cipp"]
SUBSETS = ["touching", "isolated", "scored"]
COLS = ["stem", "cell", "storage_day", "touching", "matched", "e_matched"]


def load_cells(arch_display: str, cond: str, arm: str, seed: int) -> pd.DataFrame:
    p = results_dir(arch_display, cond, arm, seed) / "cells_last.json"
    if not p.exists():
        raise SystemExit(f"[w8] {p} is missing.")
    C = pd.DataFrame(json.loads(p.read_text())["cells"])
    missing = [c for c in ("instances", "scored", "e_matched", "matched", "touching") if c not in C]
    if missing:
        raise SystemExit(f"[w8] {p} has no column(s) {missing}. Columns present: {sorted(C.columns)}")
    C = C[(C["instances"] == "A2") & (C["scored"] == True)].copy()  # noqa: E712
    for c in COLS:
        if c not in C:
            C[c] = np.nan
    C = C[COLS].sort_values(["stem", "cell"]).reset_index(drop=True)
    C["matched"] = C["matched"].fillna(False).astype(bool)
    C["touching"] = C["touching"].fillna(False).astype(bool)
    return C


def stats_for(C: pd.DataFrame, subset: str) -> Dict:
    if subset == "touching":
        sel = C["touching"].to_numpy(bool)
    elif subset == "isolated":
        sel = ~C["touching"].to_numpy(bool)
    else:
        sel = np.ones(len(C), bool)
    S = C[sel]
    e = S["e_matched"].astype(float).where(S["matched"])      # signed, fraction
    e = e[np.isfinite(e)]
    n_scored = int(sel.sum())
    if len(e) == 0:
        return {"subset": subset, "n_scored": n_scored, "n_matched": 0, "signed_median_pct": None,
                "abs_median_pct": None, "within5_pct": None, "share_negative_pct": None}
    return {"subset": subset, "n_scored": n_scored, "n_matched": int(len(e)),
            "signed_median_pct": float(np.median(100 * e)),
            "abs_median_pct": float(np.median(np.abs(100 * e))),
            # a missed cell counts as a failure, so the denominator is every scored cell of the subset
            "within5_pct": 100.0 * float(np.sum(np.abs(e) <= 0.05)) / n_scored,
            "share_negative_pct": 100.0 * float(np.mean(e < 0))}


def main() -> None:
    per_run: List[Dict] = []
    inputs: List[str] = []
    for arch_display, cond in REPRESENTATIVE.items():
        for arm in ARMS_COMPARED:
            for seed in SEEDS:
                C = load_cells(arch_display, cond, arm, seed)
                inputs.append(str((results_dir(arch_display, cond, arm, seed) / "cells_last.json").relative_to(ROOT)))
                for subset in SUBSETS:
                    row = {"arch": arch_display, "rank": cond, "arm": arm, "seed": seed}
                    row.update(stats_for(C, subset))
                    per_run.append(row)
                print(f"[w8] {arch_display} {cond} {arm} seed{seed}: done", flush=True)

    # mean and SD over the three seeds
    mean_sd: List[Dict] = []
    df = pd.DataFrame(per_run)
    for (arch, rank, arm, subset), g in df.groupby(["arch", "rank", "arm", "subset"], sort=False):
        rec = {"arch": arch, "rank": rank, "arm": arm, "subset": subset,
               "n_scored": int(g["n_scored"].iloc[0]),
               "n_matched_per_seed": [int(x) for x in g["n_matched"]]}
        for k in ("signed_median_pct", "abs_median_pct", "within5_pct", "share_negative_pct"):
            v = g[k].astype(float).to_numpy()
            rec[k] = [float(np.mean(v)), float(np.std(v, ddof=1))] if np.isfinite(v).all() else [None, None]
        mean_sd.append(rec)

    d = out_dir("w8_signed_errors")
    write_json(d / "w8_signed_errors.json",
               {"meta": provenance("w8_signed_errors.py", inputs,
                                   {"representative": REPRESENTATIVE, "arms": ARMS_COMPARED,
                                    "note": "signed error = 100*(S_pred - S_ref)/S_ref of matched cells; "
                                            "within5 counts a missed cell as a failure"}),
                "per_run": per_run, "mean_sd_over_seeds": mean_sd})

    # the Table S16 layout
    L = ["# W-8  Signed and absolute integrated-phase errors of matched cells", "",
         f"Representative ranks: {REPRESENTATIVE}. Mean ± SD over seeds {SEEDS}; last checkpoint; A2 instances.",
         "A negative signed median means the prediction underestimates the integrated phase.", ""]
    for arch_display in REPRESENTATIVE:
        L += [f"## {arch_display} ({REPRESENTATIVE[arch_display]})", ""]
        rows = []
        for arm in ARMS_COMPARED:
            for subset in ("touching", "isolated"):
                r = [x for x in mean_sd if x["arch"] == arch_display and x["arm"] == arm and x["subset"] == subset][0]
                def pm(k, nd=2, signed=False):
                    m, s = r[k]
                    return "–" if m is None else f"{fmt(m, nd, signed)} ± {fmt(s, nd)}"
                rows.append([ARM_LABEL[arm], subset, f"{r['n_scored']:,}",
                             pm("signed_median_pct", 2, True), pm("abs_median_pct", 2),
                             pm("within5_pct", 1), pm("share_negative_pct", 1)])
        L += [md_table(["Objective", "Cells", "Scored cells", "Signed median error (%)",
                        "Absolute median error (%)", "Within 5 % (%)", "Share with a negative error (%)"], rows), ""]
    write_text(d / "w8_summary.md", "\n".join(L))
    print(f"[w8] wrote {d.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
