#!/usr/bin/env python
"""W-4  One-factor comparison: cell-level IPP against global IPP, everything else equal.

Why: Table 2 compares the proposed objective (Dice + cIPP) with the old full objective (Dice + PMC + BGA + global IPP).
That comparison changes two things at once (cIPP replaces IPP AND PMC + BGA are removed). The matrix contains the
clean pair: dice_pmc_bga_cipp (new full) vs dice_pmc_bga_ipp (old full), which differ ONLY in cIPP vs global IPP.
This is the comparison that supports "cell-level, not global, integrated-phase preservation" (Prof. point 1).

Pairs: same architecture, condition and seed. Reported:
  pooled LoRA (36 pairs = 3 architectures x 4 ranks x 3 seeds), in the layout of Table 2, with the bootstrap interval
  of the median difference (10,000 resamples of the pairs, rng 0, as objective_and_rank_selection.py) and the
  Wilcoxon p; this block is cross-checked against results/consolidated/selection/arm_confirmation.csv;
  per architecture (12 pairs, exact Wilcoxon); full fine-tuning (9 pairs pooled, 3 per architecture, descriptive).
Input (read-only): results/consolidated/runs_last.json, results/consolidated/selection/arm_confirmation.csv
Usage:   python additional_analysis/w4_cell_vs_global_ipp.py
Outputs: additional_analysis/results/w4_cell_vs_global_ipp/{w4_cell_vs_global_ipp.json, .md, .csv}
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_LABEL, ROOT, fmt, fmt_p, md_table, out_dir, provenance, write_json, write_text  # noqa: E402
from w2_per_architecture_tests import METRICS, wilcoxon_exact  # noqa: E402

A, B = "dice_pmc_bga_cipp", "dice_pmc_bga_ipp"
LORA = ["r8", "r16", "r32", "r64"]


def boot_ci(d, n=10000, rng=None):
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return [None, None]
    idx = rng.integers(0, len(d), size=(n, len(d)))
    med = np.median(d[idx], axis=1)
    return [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]


def block(runs, archs, conds, rng):
    key = lambda r: (r["architecture_display"], r["condition"], r["seed"])  # noqa: E731
    ra = {key(r): r for r in runs if r["loss"] == A and r["condition"] in conds and r["architecture_display"] in archs}
    rb = {key(r): r for r in runs if r["loss"] == B and r["condition"] in conds and r["architecture_display"] in archs}
    keys = sorted(set(ra) & set(rb)); out = {}
    for field, direction, label, scale in METRICS:
        va = np.array([np.nan if ra[k].get(field) is None else float(ra[k][field]) * scale for k in keys])
        vb = np.array([np.nan if rb[k].get(field) is None else float(rb[k][field]) * scale for k in keys])
        ok = np.isfinite(va) & np.isfinite(vb); d = va[ok] - vb[ok]
        out[field] = {"label": label, "larger_is_better": direction, "n_pairs": int(ok.sum()),
                      "mean_new_full": float(va[ok].mean()) if ok.any() else None,
                      "mean_old_full": float(vb[ok].mean()) if ok.any() else None,
                      "median_difference": float(np.median(d)) if len(d) else None,
                      "pairs_new_full_better": int((d * direction > 0).sum()) if direction else None,
                      "ci95_median_difference": boot_ci(d, rng=rng),
                      "wilcoxon_p": wilcoxon_exact(d)}
    return {"n_pairs": len(keys), "metrics": out}


def _wilcoxon_any(d):
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) < 6:
        return None
    from scipy.stats import wilcoxon
    return float(wilcoxon(d).pvalue)


def main():
    runs = json.loads((ROOT / "results" / "consolidated" / "runs_last.json").read_text())["runs"]
    archs = sorted({r["architecture_display"] for r in runs})
    rng = np.random.default_rng(0)
    res = {"comparison": {"a": A, "b": B, "label_a": ARM_LABEL[A], "label_b": ARM_LABEL[B],
                          "only_difference": "cell-level IPP (cIPP, eta = 0) instead of global IPP; both with PMC and BGA, same lambdas"},
           "lora_pooled": block(runs, archs, LORA, rng),
           "per_architecture_lora": {a: block(runs, [a], LORA, rng) for a in archs},
           "full_finetuning_pooled": block(runs, archs, ["fullft"], rng)}
    # cross-check with the project's own pooled comparison
    ac = ROOT / "results" / "consolidated" / "selection" / "arm_confirmation.csv"
    check = []
    if ac.exists():
        with open(ac, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["loss_a"] == A and r["loss_b"] == B and r["metric"] in res["lora_pooled"]["metrics"]:
                    mine = res["lora_pooled"]["metrics"][r["metric"]]
                    scale = 100 if r["metric"] == "ipe_global" else 1
                    check.append({"metric": r["metric"],
                                  "median_difference_project": float(r["median_difference_a_minus_b"]) * scale,
                                  "median_difference_here": mine["median_difference"],
                                  "wilcoxon_p_project": float(r["wilcoxon_p"]) if r["wilcoxon_p"] else None,
                                  "agree": bool(np.isclose(float(r["median_difference_a_minus_b"]) * scale, mine["median_difference"], atol=1e-9))})
    res["cross_check_arm_confirmation"] = check
    meta = provenance("w4_cell_vs_global_ipp.py", ["results/consolidated/runs_last.json", "results/consolidated/selection/arm_confirmation.csv"])
    od = out_dir("w4_cell_vs_global_ipp")
    write_json(od / "w4_cell_vs_global_ipp.json", {"metadata": meta, "results": res})

    hdr = ["Quantity", "New full (cIPP)", "Old full (global IPP)", "Median difference", "Pairs new full better",
           "95 % interval of the median difference", "p (Wilcoxon)"]
    def rows_of(b):
        rr = []
        for field, *_ in METRICS:
            m = b["metrics"][field]; lo, hi = m["ci95_median_difference"]
            rr.append([m["label"], fmt(m["mean_new_full"], 3), fmt(m["mean_old_full"], 3), fmt(m["median_difference"], 3, True),
                       "–" if m["pairs_new_full_better"] is None else f"{m['pairs_new_full_better']} of {m['n_pairs']}",
                       "–" if lo is None else f"[{lo:+.3f}, {hi:+.3f}]", fmt_p(m["wilcoxon_p"])])
        return rr
    L = ["# W-4 Cell-level IPP against global IPP, everything else equal", "",
         f"{ARM_LABEL[A]} minus {ARM_LABEL[B]}. Same architecture, condition and seed. Checkpoint last.", "",
         f"## LoRA ranks, all architectures ({res['lora_pooled']['n_pairs']} pairs)", "", md_table(hdr, rows_of(res["lora_pooled"])), ""]
    for a in archs:
        L += [f"## {a}, LoRA ranks ({res['per_architecture_lora'][a]['n_pairs']} pairs)", "", md_table(hdr, rows_of(res["per_architecture_lora"][a])), ""]
    L += [f"## Full fine-tuning, all architectures ({res['full_finetuning_pooled']['n_pairs']} pairs)", "",
          md_table(hdr, rows_of(res["full_finetuning_pooled"])), "",
          f"Cross-check against arm_confirmation.csv: {sum(c['agree'] for c in check)} of {len(check)} metrics agree."]
    write_text(od / "w4_cell_vs_global_ipp.md", "\n".join(L))
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n"); w.writerow(["block"] + hdr)
    for name, b in [("LoRA, all architectures", res["lora_pooled"])] + [(f"LoRA, {a}", res["per_architecture_lora"][a]) for a in archs] \
            + [("Full fine-tuning, all architectures", res["full_finetuning_pooled"])]:
        for r in rows_of(b):
            w.writerow([name] + r)
    write_text(od / "w4_cell_vs_global_ipp.csv", buf.getvalue())


if __name__ == "__main__":
    main()
