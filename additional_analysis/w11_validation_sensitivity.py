"""W-11  The rank the selection rule would choose at each tolerance, on the VALIDATION split.

Why this exists
---------------
`results/consolidated/selection/rank_selection_sensitivity.csv` sweeps the tolerance over the TEST
metrics, because objective_and_rank_selection.py calls apply_rule() without the validation choice in
that loop. With selection.source = validation that sweep answers a different question from the one
the manuscript reports, and at zero tolerance it returns r = 64 for MobileSAM while the manuscript
reports r = 32. This script runs the same rule over the same tolerances on the validation metrics,
which is the split the rule actually reads, so the two agree at tolerance 0.

Read-only: reads additional_analysis/results/w9_validation_selection/w9_validation_selection.json
and configs/config.yaml, and writes only below additional_analysis/results/w11_validation_sensitivity/.

    python additional_analysis/w11_validation_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config, md_table, out_dir, provenance, write_json, write_text

W9_DISPLAY = {"EdgeSAM": "EdgeSAM", "MobileSAM": "MobileSAM", "MobileNet_UNet": "MobileNet-UNet"}


def score(rec: Dict, weights: Dict[str, float]) -> float:
    """The balanced score: each metric scaled to 0..1 and weighted, as configs/config.yaml defines it."""
    s = 0.0
    for name, w in weights.items():
        if name.startswith("a2_scored_within_5pct"):
            s += w * float(rec["within5"]) / 100.0
        elif name == "dice_macro":
            s += w * float(rec["mean_dice"])
        else:
            raise SystemExit(f"[w11] unknown selection weight {name!r}")
    return s


def choose(ranks: Dict[str, Dict], weights: Dict[str, float], tol: float, stability: str) -> Dict:
    """Stability first, then the best balanced score, then the smallest rank within `tol` of it."""
    order = sorted(ranks, key=lambda r: int(str(r).lstrip("r")))
    if stability == "graded":
        fewest = min(int(ranks[r]["seeds_lost"]) for r in order)
        keep = [r for r in order if int(ranks[r]["seeds_lost"]) == fewest]
    elif stability == "binary":
        keep = [r for r in order if not ranks[r]["any_lost"]] or list(order)
        fewest = None
    else:
        keep, fewest = list(order), None
    scores = {r: score(ranks[r], weights) for r in keep}
    best = max(scores.values())
    within = [r for r in keep if scores[r] >= best - tol]
    chosen = min(within, key=lambda r: int(str(r).lstrip("r")))
    return {"chosen": chosen, "kept_by_stability": keep, "fewest_seeds_lost": fewest,
            "best_score": best, "chosen_score": scores[chosen],
            "every_rank_lost_a_class": all(ranks[r]["any_lost"] for r in order)}


def main() -> None:
    cfg = load_config()
    sel = cfg["selection"]
    weights = {k: float(v) for k, v in sel["weights"].items()}
    stability = str(sel.get("stability", "graded"))
    rule = str(sel["rule"])
    path = ROOT / sel["validation_results"]
    if not path.exists():
        raise SystemExit(f"[w11] {path} is missing: run\n"
                         f"  python additional_analysis/w9_validation_selection.py --select\nfirst.")
    data = json.loads(path.read_text(encoding="utf-8"))["rank_selection"]

    tols = [round(float(x), 4) for x in np.arange(0.0, 0.0501, 0.0025)]
    rows: List[Dict] = []
    for arch, disp in W9_DISPLAY.items():
        if arch not in data:
            raise SystemExit(f"[w11] {path} has no entry for {arch}.")
        if rule not in data[arch]:
            raise SystemExit(f"[w11] {path} has no '{rule}' rule for {arch}; it holds {sorted(data[arch])}.")
        ranks = data[arch][rule]["validation"]
        for tol in tols:
            rec = choose(ranks, weights, tol, stability)
            rows.append({"architecture": arch, "architecture_display": disp, "split": "validation",
                         "tolerance_points": tol, **rec})

    d = out_dir("w11_validation_sensitivity")
    write_json(d / "w11_validation_sensitivity.json",
               {"meta": provenance("w11_validation_sensitivity.py",
                                   [str(path.relative_to(ROOT)), "configs/config.yaml"],
                                   {"rule": rule, "stability": stability, "weights": weights,
                                    "note": "the same rule as scripts/evaluation/objective_and_rank_selection.py, "
                                            "applied to the validation metrics instead of the test metrics"}),
                "sensitivity": rows})

    archs = [W9_DISPLAY[a] for a in W9_DISPLAY]
    body = []
    for tol in tols:
        cells = []
        for disp in archs:
            r = [x for x in rows if x["architecture_display"] == disp and x["tolerance_points"] == tol][0]
            note = " (no rank kept every class)" if r["every_rank_lost_a_class"] else ""
            cells.append(f"r = {str(r['chosen']).lstrip('r')}{note}")
        body.append([f"{tol:g}"] + cells)
    write_text(d / "w11_summary.md", "\n".join(
        ["# W-11  The rank the rule chooses at each tolerance, on the validation split", "",
         f"Rule: {rule}; stability: {stability}; weights: {weights}. The score is on a 0 to 1 scale, so a "
         "tolerance is in score points, not percentage points.", "",
         md_table(["Tolerance (score points)"] + archs, body), "",
         "At a tolerance of zero this reproduces the ranks reported in the manuscript, because the rule "
         "and the split are the ones the manuscript uses. Table S9 sweeps the same rule over the test "
         "metrics instead, which is why its tolerance-zero entry differs for MobileSAM.", ""]))
    print(f"[w11] wrote {d.relative_to(ROOT)}", flush=True)
    for disp in archs:
        r = [x for x in rows if x["architecture_display"] == disp and x["tolerance_points"] == 0.0][0]
        print(f"[w11] {disp:16s} tolerance 0 -> {r['chosen']}", flush=True)


if __name__ == "__main__":
    main()
