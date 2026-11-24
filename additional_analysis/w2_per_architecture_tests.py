#!/usr/bin/env python
"""W-2  Per-architecture run-level comparison of the proposed objective with every other loss configuration.

Why: selection_summary.md pools 36 pairs (3 architectures x 4 LoRA ranks x 3 seeds) into one Wilcoxon test, which
hides that the effect differs between architectures (e.g. MobileSAM vs Dice-only). This script repeats the paired
comparison separately for each architecture. It is the source of manuscript Table 2 and of the per-architecture
columns of Supplementary Table S6 (both checkpoints).

Comparison families (revised 20 Nov 2026):

  proposed_vs_others   the proposed objective (dice_cipp) against each of the other six loss configurations.
                       One family per architecture and metric: m = 6 hypotheses, Holm step-down. This is the
                       question of Table 2 / Table S6 ("is the proposed objective different from configuration X?").
                       A comparison whose paired differences are all zero, or too few to reach a small p, still
                       belongs to the family: its exact p is computed (1.0 when every difference is zero), so the
                       family size is always 6 and no hypothesis silently drops out of the adjustment.
  one_factor           Dice + PMC + BGA + cIPP against Dice + PMC + BGA + global IPP (they differ only in the phase
                       term). A separate question (Section 3.5, third comparison; Table S7, whose published values
                       come from W-4). One hypothesis per architecture and metric, so no within-family adjustment:
                       the reported p is the exact p. Kept here as a cross-check of W-4, not merged with the family
                       above.

Blocks per architecture:
  lora_ranks_pooled    12 pairs (4 LoRA ranks x 3 seeds, paired on rank and seed); exact two-sided Wilcoxon
                       signed-rank test (zero differences dropped, mid-ranks for ties, exact conditional
                       distribution); Holm over the family proposed_vs_others.
  one_factor           the same 12 pairs for the one-factor comparison; exact p, family of one.
  full_finetuning      3 pairs per architecture: too few for a test (smallest attainable two-sided p 0.25), so only
                       per-seed differences are given, for both families; no p-values.

Removed on 20 Nov 2026: the former `representative_rank` block (per-seed differences at one rank). It was computed at
the ranks in force before 18 Nov (EdgeSAM r8, MobileSAM r64, MobileNet_UNet r32), and no manuscript table, figure
or other script reads it (consumers checked: manuscript/figtab/tables.py, manuscript/verify.py, w4).

Unchanged for other scripts: `METRICS` and `wilcoxon_exact` keep their previous names and behaviour, because
w4_cell_vs_global_ipp.py imports them.

Input (read-only): results/consolidated/runs_<checkpoint>.json.
Usage (from the project root):
    python additional_analysis/w2_per_architecture_tests.py --checkpoint last
    python additional_analysis/w2_per_architecture_tests.py --checkpoint best
Options: --out-name NAME     output folder below additional_analysis/results/ (default w2_per_architecture_rerun,
                             so the earlier outputs in w2_per_architecture/ are not overwritten)
         --compare-with NAME folder holding the earlier outputs to check against (default w2_per_architecture;
                             'none' to skip). The check fails (exit code 1) if any quantity that the revision does
                             not change (pairs, means, medians, counts, raw p of tests that existed before) differs.
Outputs: additional_analysis/results/<out-name>/w2_per_architecture_<ckpt>.json, ..._summary.md,
         w2_rerun_check_<ckpt>.json, w2_rerun_check_<ckpt>.md
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ARM_LABEL, ARMS, OUT_ROOT, ROOT, fmt, fmt_p, holm, md_table, out_dir,  # noqa: E402
                    provenance, wilcoxon_signed_rank_exact, write_json, write_text)

PROPOSED = "dice_cipp"
FAMILY_PROPOSED = [(PROPOSED, x) for x in ARMS if x != PROPOSED]                 # 6 comparisons
ONE_FACTOR = [("dice_pmc_bga_cipp", "dice_pmc_bga_ipp")]                          # cell-level vs global IPP
LORA = ["r8", "r16", "r32", "r64"]
METRICS = [  # (field, direction (+1 larger is better), label, scale)
    ("a2_scored_within_5pct_of_all_cells_pct", +1, "cells within 5 % (all scored), %", 1),
    ("a2_touching_within_5pct_of_all_cells_pct", +1, "cells within 5 % (touching), %", 1),
    ("a2_isolated_within_5pct_of_all_cells_pct", +1, "cells within 5 % (isolated), %", 1),
    ("a2_scored_cell_ipe_abs_median_pct", -1, "cell IPE, matched cells, median %", 1),
    ("a2_scored_network_cell_ipe_abs_median_pct", -1, "cell IPE without matching, median %", 1),
    ("ipe_global", -1, "global IPE, %", 100),
    ("a2_scored_drymass_ccc", +1, "dry mass CCC (matched cells)", 1),
    ("a2_scored_drymass_bias_pg", 0, "dry mass bias, pg (matched cells)", 1),
    ("a2_scored_match_pct", +1, "cells matched, %", 1),
    ("a2_scored_missed_pct", -1, "cells missed, %", 1),
    ("a2_scored_merged_pct", -1, "cells merged, %", 1),
    ("a2_scored_fragmented_pct", -1, "cells fragmented, %", 1),
    ("dice_macro", +1, "macro Dice", 1),
    ("a2_aji_cells", +1, "AJI, separated cells", 1),
    ("bf1", +1, "boundary F1", 1),
    ("n_lost_classes", -1, "classes lost per run", 1),
]
KEY_METRICS = ["a2_scored_within_5pct_of_all_cells_pct", "a2_scored_cell_ipe_abs_median_pct",
               "a2_scored_network_cell_ipe_abs_median_pct", "ipe_global", "a2_scored_drymass_ccc",
               "a2_scored_missed_pct", "dice_macro"]
TOL = 1e-12                                   # agreement tolerance of the rerun check


def wilcoxon_exact(d: np.ndarray):
    """Unchanged (imported by w4): exact two-sided p, None with fewer than 6 non-zero differences."""
    return wilcoxon_signed_rank_exact(d)


def wilcoxon_exact_any_n(d: np.ndarray):
    """Exact two-sided Wilcoxon signed-rank p for ANY number of non-zero differences, so that every member of a
    comparison family has a p-value. Zero differences are dropped (as in wilcoxon_exact); 1.0 when none remain.
    With 6 or more non-zero differences this is exactly wilcoxon_exact (common.wilcoxon_signed_rank_exact). With
    1-5 the exact conditional distribution of the signs is enumerated (2^n patterns, mid-ranks for ties); such a p
    cannot be small (at least 2 / 2^n), which is the correct outcome for an uninformative comparison."""
    from scipy.stats import rankdata
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return 1.0
    if n >= 6:
        return wilcoxon_signed_rank_exact(d)
    r = rankdata(np.abs(d))
    t_obs = float(r[d > 0].sum()); mean = float(r.sum()) / 2.0
    hits = sum(1 for s in itertools.product((0, 1), repeat=n)
               if abs(float(np.dot(r, s)) - mean) >= abs(t_obs - mean) - 1e-9)
    return min(1.0, hits / 2 ** n)


def value(r, field, scale):
    v = r.get(field)
    return np.nan if v is None or isinstance(v, bool) else float(v) * scale


def compare(runs, arch, conds, a, b, test: bool):
    idx = lambda arm: {(r["condition"], r["seed"]): r for r in runs  # noqa: E731
                       if r["architecture_display"] == arch and r["loss"] == arm and r["condition"] in conds}
    A, B = idx(a), idx(b)
    keys = sorted(set(A) & set(B))
    out = {}
    for field, direction, label, scale in METRICS:
        va = np.array([value(A[k], field, scale) for k in keys]); vb = np.array([value(B[k], field, scale) for k in keys])
        ok = np.isfinite(va) & np.isfinite(vb); d = va[ok] - vb[ok]
        nz = int(np.count_nonzero(d))
        out[field] = {"label": label, "larger_is_better": direction, "n_pairs": int(ok.sum()),
                      "n_nonzero_differences": nz,
                      "mean_a": float(va[ok].mean()) if ok.any() else None, "mean_b": float(vb[ok].mean()) if ok.any() else None,
                      "median_difference_a_minus_b": float(np.median(d)) if len(d) else None,
                      "pairs_a_better": int((d * direction > 0).sum()) if direction else None,
                      "pairs_b_better": int((d * direction < 0).sum()) if direction else None,
                      "pairs_tied": int((d == 0).sum()),
                      "wilcoxon_exact_p": wilcoxon_exact_any_n(d) if test else None,
                      "smallest_attainable_p": (min(1.0, 2.0 / 2 ** nz) if nz else 1.0) if test else None,
                      "per_pair": [{"condition": k[0], "seed": k[1], "a": float(va[i]), "b": float(vb[i]), "a_minus_b": float(va[i] - vb[i])}
                                   for i, k in enumerate(keys) if ok[i]]}
    return {"loss_a": a, "loss_b": b, "label_a": ARM_LABEL[a], "label_b": ARM_LABEL[b], "pairs": [list(k) for k in keys], "metrics": out}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rerun_check(res: dict, old_path: Path, n_arch: int) -> dict:
    """Compare the new LoRA-block values with an earlier W-2 output. Quantities the revision does not change must be
    identical; the Holm-adjusted p is expected to change (family of 7 -> 6) and is listed, not tested."""
    chk = {"compared_with": str(old_path.relative_to(ROOT)) if old_path.exists() else str(old_path),
           "structure": {}, "unchanged_quantities": {"checked": 0, "mismatches": []}, "holm_changes": [],
           "raw_p_newly_defined": [], "passed": False}
    st = chk["structure"]
    st["architectures"] = sorted(res)
    st["n_architectures_ok"] = len(res) == n_arch == 3
    st["comparisons_per_architecture_ok"] = all(len(res[a]["lora_ranks_pooled"]["comparisons"]) == 6 and
                                                len(res[a]["one_factor"]["comparisons"]) == 1 for a in res)
    st["pairs_per_comparison"] = sorted({m["n_pairs"] for a in res for blk in ("lora_ranks_pooled", "one_factor")
                                         for c in res[a][blk]["comparisons"] for m in c["metrics"].values()})
    st["holm_family_size_always_6"] = all(m.get("holm_family_size") == 6 for a in res
                                          for c in res[a]["lora_ranks_pooled"]["comparisons"] for m in c["metrics"].values())
    if not old_path.exists():
        chk["note"] = "earlier output not found; only the structural checks were run"
        chk["passed"] = all(v for k, v in st.items() if k.endswith("_ok") or k.startswith("holm_family"))
        return chk
    old = json.loads(old_path.read_text(encoding="utf-8"))["results"]
    u = chk["unchanged_quantities"]
    for arch in res:
        oc = old.get(arch, {}).get("lora_ranks_pooled", {}).get("comparisons", [])
        new_cmps = res[arch]["lora_ranks_pooled"]["comparisons"] + res[arch]["one_factor"]["comparisons"]
        for c in new_cmps:
            o = next((x for x in oc if x["loss_a"] == c["loss_a"] and x["loss_b"] == c["loss_b"]), None)
            if o is None:
                u["mismatches"].append({"architecture": arch, "comparison": f"{c['loss_a']} - {c['loss_b']}", "what": "missing in earlier output"})
                continue
            for f, m in c["metrics"].items():
                om = o["metrics"].get(f)
                if om is None:
                    continue
                for q in ("n_pairs", "mean_a", "mean_b", "median_difference_a_minus_b", "pairs_a_better", "pairs_b_better"):
                    a_, b_ = m.get(q), om.get(q)
                    u["checked"] += 1
                    if (a_ is None) != (b_ is None) or (a_ is not None and abs(float(a_) - float(b_)) > TOL):
                        u["mismatches"].append({"architecture": arch, "comparison": f"{c['loss_a']} - {c['loss_b']}",
                                                "metric": f, "quantity": q, "new": a_, "earlier": b_})
                p_new, p_old = m.get("wilcoxon_exact_p"), om.get("wilcoxon_exact_p")
                if p_old is None:
                    if p_new is not None:
                        chk["raw_p_newly_defined"].append({"architecture": arch, "comparison": f"{c['loss_a']} - {c['loss_b']}",
                                                           "metric": f, "new_p": p_new, "n_nonzero": m["n_nonzero_differences"]})
                else:
                    u["checked"] += 1
                    if p_new is None or abs(p_new - p_old) > TOL:
                        u["mismatches"].append({"architecture": arch, "comparison": f"{c['loss_a']} - {c['loss_b']}",
                                                "metric": f, "quantity": "wilcoxon_exact_p", "new": p_new, "earlier": p_old})
                h_new, h_old = m.get("wilcoxon_p_holm_within_architecture"), om.get("wilcoxon_p_holm_within_architecture")
                if h_new != h_old:
                    chk["holm_changes"].append({"architecture": arch, "comparison": f"{c['loss_a']} - {c['loss_b']}",
                                                "metric": f, "new": h_new, "earlier": h_old})
    chk["passed"] = (not u["mismatches"]) and all(v for k, v in st.items() if k.endswith("_ok") or k.startswith("holm_family"))
    return chk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="last", choices=["last", "best"])
    ap.add_argument("--out-name", default="w2_per_architecture_rerun")
    ap.add_argument("--compare-with", default="w2_per_architecture")
    a = ap.parse_args()
    src = ROOT / "results" / "consolidated" / f"runs_{a.checkpoint}.json"
    runs = json.loads(src.read_text())["runs"]
    archs = sorted({r["architecture_display"] for r in runs})
    res = {}
    for arch in archs:
        res[arch] = {
            "lora_ranks_pooled": {"conditions": LORA, "family": "proposed_vs_others",
                                  "comparisons": [compare(runs, arch, LORA, x, y, True) for x, y in FAMILY_PROPOSED]},
            "one_factor": {"conditions": LORA, "family": "one_factor",
                           "comparisons": [compare(runs, arch, LORA, x, y, True) for x, y in ONE_FACTOR]},
            "full_finetuning": {"conditions": ["fullft"], "family": "descriptive (3 pairs, no test)",
                                "comparisons": [compare(runs, arch, ["fullft"], x, y, False) for x, y in FAMILY_PROPOSED + ONE_FACTOR]},
        }
        comps = res[arch]["lora_ranks_pooled"]["comparisons"]
        for field, *_ in METRICS:                                  # Holm within architecture and metric, m = 6
            ps = [c["metrics"][field]["wilcoxon_exact_p"] for c in comps]
            adj = holm(ps)
            for c, p in zip(comps, adj):
                c["metrics"][field]["wilcoxon_p_holm_within_architecture"] = p
                c["metrics"][field]["holm_family_size"] = sum(x is not None for x in ps)
        for c in res[arch]["one_factor"]["comparisons"]:          # family of one: adjusted p = exact p
            for field, *_ in METRICS:
                c["metrics"][field]["wilcoxon_p_holm_within_architecture"] = c["metrics"][field]["wilcoxon_exact_p"]
                c["metrics"][field]["holm_family_size"] = 1
    meta = provenance("w2_per_architecture_tests.py", [str(src.relative_to(ROOT))],
                      {"checkpoint": a.checkpoint, "input_sha256": {str(src.relative_to(ROOT)): sha256(src)},
                       "proposed": PROPOSED,
                       "families": {"proposed_vs_others": [list(p) for p in FAMILY_PROPOSED], "one_factor": [list(p) for p in ONE_FACTOR]},
                       "test": "exact two-sided Wilcoxon signed-rank on the run-level paired differences (LoRA: 12 pairs); "
                               "zero differences dropped; exact p for any number of non-zero differences (1.0 if none)",
                       "holm_family": "within each architecture and metric, over the six comparisons of the proposed "
                                      "objective with the other loss configurations (LoRA block); the one-factor "
                                      "comparison is a separate family of one and is not adjusted",
                       "revision": "20 Nov 2026: families separated (6 + 1 instead of 7); every family member tested; "
                                   "representative_rank block removed"})
    od = out_dir(a.out_name)
    write_json(od / f"w2_per_architecture_{a.checkpoint}.json", {"metadata": meta, "results": res})

    L = [f"# W-2 Per-architecture comparison, checkpoint {a.checkpoint}", "",
         "Run-level paired differences (A minus B), LoRA ranks pooled: 12 pairs (4 ranks × 3 seeds), exact Wilcoxon.",
         "Family 'proposed vs others': six comparisons, Holm within architecture and metric (m = 6 always).",
         "Family 'one factor' (new full vs old full): one comparison, p not adjusted.",
         "'better' counts pairs in which A is better. Full fine-tuning: 3 pairs, no test (JSON only).", ""]
    for arch in archs:
        for blk, title in (("lora_ranks_pooled", "Proposed vs others (12 pairs, Holm m = 6)"),
                           ("one_factor", "One factor: cIPP vs global IPP in the full objective (12 pairs, unadjusted)")):
            rows = []
            for c in res[arch][blk]["comparisons"]:
                for f in KEY_METRICS:
                    m = c["metrics"][f]
                    rows.append([f"{c['loss_a']} − {c['loss_b']}", m["label"], fmt(m["mean_a"], 3), fmt(m["mean_b"], 3),
                                 fmt(m["median_difference_a_minus_b"], 3, True),
                                 "–" if m["pairs_a_better"] is None else f"{m['pairs_a_better']}/{m['n_pairs']}",
                                 str(m["n_nonzero_differences"]), fmt_p(m["wilcoxon_exact_p"]),
                                 fmt_p(m.get("wilcoxon_p_holm_within_architecture")), str(m.get("holm_family_size"))])
            L += [f"## {arch}: {title}", "",
                  md_table(["A − B", "metric", "mean A", "mean B", "median Δ", "A better", "non-zero", "p", "p (adjusted)", "m"], rows), ""]
    write_text(od / f"w2_per_architecture_{a.checkpoint}_summary.md", "\n".join(L))

    if a.compare_with.lower() != "none":
        old_path = OUT_ROOT / a.compare_with / f"w2_per_architecture_{a.checkpoint}.json"
        chk = rerun_check(res, old_path, len(archs))
        chk["metadata"] = provenance("w2_per_architecture_tests.py", [str(src.relative_to(ROOT))], {"checkpoint": a.checkpoint})
        write_json(od / f"w2_rerun_check_{a.checkpoint}.json", chk)
        u = chk["unchanged_quantities"]
        M = [f"# W-2 rerun check, checkpoint {a.checkpoint}", "",
             f"**{'PASSED' if chk['passed'] else 'FAILED'}** — compared with `{chk['compared_with']}`.", "",
             f"* Structure: {json.dumps(chk['structure'])}",
             f"* Unchanged quantities checked: {u['checked']}; mismatches: {len(u['mismatches'])}",
             f"* Raw p newly defined (comparisons with fewer than 6 non-zero differences): {len(chk['raw_p_newly_defined'])}",
             f"* Holm-adjusted p changed (expected: family of 7 became 6): {len(chk['holm_changes'])}", ""]
        if u["mismatches"]:
            M += ["## Mismatches (must be explained before the outputs are used)", "",
                  md_table(["architecture", "comparison", "metric", "quantity", "new", "earlier"],
                           [[x.get("architecture"), x.get("comparison"), x.get("metric", "–"), x.get("quantity", x.get("what")),
                             str(x.get("new")), str(x.get("earlier"))] for x in u["mismatches"]]), ""]
        if chk["raw_p_newly_defined"]:
            M += ["## Raw p newly defined", "", md_table(["architecture", "comparison", "metric", "p", "non-zero"],
                  [[x["architecture"], x["comparison"], x["metric"], fmt_p(x["new_p"]), str(x["n_nonzero"])] for x in chk["raw_p_newly_defined"]]), ""]
        if chk["holm_changes"]:
            M += ["## Adjusted p, earlier vs new", "", md_table(["architecture", "comparison", "metric", "earlier", "new"],
                  [[x["architecture"], x["comparison"], x["metric"], fmt_p(x["earlier"]), fmt_p(x["new"])] for x in chk["holm_changes"]]), ""]
        write_text(od / f"w2_rerun_check_{a.checkpoint}.md", "\n".join(M))
        print(f"[w2] rerun check {'PASSED' if chk['passed'] else 'FAILED'} ({u['checked']} quantities, "
              f"{len(u['mismatches'])} mismatches)", flush=True)
        if not chk["passed"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
