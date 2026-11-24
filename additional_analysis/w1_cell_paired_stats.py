#!/usr/bin/env python
"""W-1  Cell-paired statistics: does the proposed objective improve integrated-phase preservation of the SAME cells?

Why: the existing comparison (scripts/evaluation/objective_and_rank_selection.py) uses one number per run and pools
36 runs over architectures and ranks. The manuscript's claim is about individual cells, so this analysis pairs the
same reference cell under two loss configurations (same architecture, adaptation condition and seed) and respects
the dependence structure of the data: cells within images, images within storage days.

Inputs (read-only): results/<Arch>/<condition>/<loss>/seed<seed>/cells_last.json, A2 instances, scored cells only.
The fields used (e_matched, e_assigned, matched, missed, dm_*_pg, touching) are not changed by the 13-Nov rescoring.

Per reference cell c and seed s, for loss A and loss B:
  within5     1 if c is matched (IoU >= 0.5) and |e_matched| <= 5 %, else 0 (missed / unmatched cells count as 0)
  |e_assigned| matching-free integrated-phase error in the cell's territory (a missed cell gives ~100 %)
  |e_matched| integrated-phase error of the matched instance (only where c is matched under BOTH losses)
  |dm error|  |dm_pred - dm_gt| in pg (only where c is matched under BOTH losses)

Statistics (A minus B; cells x seeds pooled, each cell carries its three seeds):
  d_within5_pp           100 * mean(within5_A - within5_B)                         percentage points of all scored cells
  d_median_abs_e_assigned_pp  median|e_assigned|_A - median|e_assigned|_B (percentage points)
  mean_paired_abs_e_assigned_pp  mean(|e_assigned|_A - |e_assigned|_B) (pp; sensitive to missed cells)
  d_median_abs_e_matched_both_pp  conditional on matched under both (pp)
  d_median_abs_drymass_err_both_pg  conditional on matched under both (pg)

Uncertainty:
  95 % percentile interval from a hierarchical bootstrap: storage days resampled, then images within each day,
  then cells within each image (all seeds of a cell move together). Default 10,000 resamples.
  p (day-level label swap): exact permutation test that swaps the A/B labels of ALL cells of a storage day at once,
  over all 2^11 = 2048 patterns; two-sided on |statistic|. The smallest attainable p is 2/2048 = 9.8e-4.
  Holm correction within each architecture over (comparisons x the two primary statistics: d_within5_pp and
  d_median_abs_e_assigned_pp).

Usage (from the project root):
  python additional_analysis/w1_cell_paired_stats.py                       # representative configurations
  python additional_analysis/w1_cell_paired_stats.py --conditions all      # every rank and full fine-tuning too
  python additional_analysis/w1_cell_paired_stats.py --bootstrap 2000      # quicker
Outputs: additional_analysis/results/w1_cell_paired/{w1_cell_paired.json, w1_cell_paired_summary.md}
         (--conditions all -> w1_cell_paired_all.*; any other list -> w1_cell_paired_custom.*)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ARM_LABEL, REPRESENTATIVE, ROOT, SEEDS, fmt, fmt_p, holm, md_table, out_dir,  # noqa: E402
                    provenance, results_dir, write_json, write_text)

COMPARISONS = [("dice_cipp", "dice"), ("dice_cipp", "dice_pmc_bga_ipp"), ("dice_cipp", "dice_pmc_bga_cipp"),
               ("dice_cipp", "dice_pmc_cipp"), ("dice_pmc_bga_cipp", "dice_pmc_bga_ipp")]
PRIMARY = ("d_within5_pp", "d_median_abs_e_assigned_pp")
STAT_DOC = {
    "d_within5_pp": "difference in the share of all scored cells within 5 % (percentage points; >0 favours A)",
    "d_median_abs_e_assigned_pp": "difference of median |matching-free cell integrated-phase error| (pp; <0 favours A)",
    "mean_paired_abs_e_assigned_pp": "mean paired difference of |matching-free error| (pp; <0 favours A; missed cells weigh ~100 pp)",
    "d_median_abs_e_matched_both_pp": "difference of median |matched-cell error|, cells matched under both (pp; <0 favours A)",
    "d_median_abs_drymass_err_both_pg": "difference of median |dry-mass error|, cells matched under both (pg; <0 favours A)",
}
COLS = ["stem", "cell", "storage_day", "touching", "matched", "missed", "e_matched", "e_assigned", "dm_pred_pg", "dm_gt_pg"]


def load_cells(arch: str, cond: str, arm: str, seed: int) -> pd.DataFrame:
    p = results_dir(arch, cond, arm, seed) / "cells_last.json"
    C = pd.DataFrame(json.loads(p.read_text())["cells"])
    C = C[(C["instances"] == "A2") & (C["scored"] == True)].copy()  # noqa: E712
    for c in COLS:
        if c not in C:
            C[c] = np.nan
    C = C[COLS].sort_values(["stem", "cell"]).reset_index(drop=True)
    C["matched"] = C["matched"].fillna(False).astype(bool)
    C["touching"] = C["touching"].fillna(False).astype(bool)
    return C


def per_seed_arrays(C: pd.DataFrame) -> Dict[str, np.ndarray]:
    em = C["e_matched"].astype(float).abs().where(C["matched"])
    return {"w": (C["matched"] & (em <= 0.05)).to_numpy(float),
            "ea": 100 * C["e_assigned"].astype(float).abs().to_numpy(),
            "em": 100 * em.to_numpy(float),
            "dme": (C["dm_pred_pg"].astype(float) - C["dm_gt_pg"].astype(float)).abs().where(C["matched"]).to_numpy(float)}


def stats(A: Dict[str, np.ndarray], B: Dict[str, np.ndarray], rows: np.ndarray) -> Dict[str, float]:
    """A, B: arrays (cells, seeds). rows: cell indices (with repetition)."""
    a = {k: v[rows] for k, v in A.items()}; b = {k: v[rows] for k, v in B.items()}
    both = np.isfinite(a["em"]) & np.isfinite(b["em"])
    okA = np.isfinite(a["ea"]); okB = np.isfinite(b["ea"]); ok = okA & okB
    out = {"d_within5_pp": 100.0 * float(np.mean(a["w"] - b["w"])),
           "d_median_abs_e_assigned_pp": float(np.median(a["ea"][okA]) - np.median(b["ea"][okB])) if okA.any() and okB.any() else np.nan,
           "mean_paired_abs_e_assigned_pp": float(np.mean(a["ea"][ok] - b["ea"][ok])) if ok.any() else np.nan,
           "d_median_abs_e_matched_both_pp": float(np.median(a["em"][both]) - np.median(b["em"][both])) if both.any() else np.nan,
           "d_median_abs_drymass_err_both_pg": float(np.median(a["dme"][both]) - np.median(b["dme"][both])) if both.any() else np.nan}
    return out


def levels(A: Dict[str, np.ndarray], B: Dict[str, np.ndarray], rows: np.ndarray) -> Dict[str, float]:
    a = {k: v[rows] for k, v in A.items()}; b = {k: v[rows] for k, v in B.items()}
    both = np.isfinite(a["em"]) & np.isfinite(b["em"])
    return {"within5_pct_A": 100 * float(np.mean(a["w"])), "within5_pct_B": 100 * float(np.mean(b["w"])),
            "median_abs_e_assigned_pct_A": float(np.nanmedian(a["ea"])), "median_abs_e_assigned_pct_B": float(np.nanmedian(b["ea"])),
            "median_abs_e_matched_both_pct_A": float(np.median(a["em"][both])) if both.any() else np.nan,
            "median_abs_e_matched_both_pct_B": float(np.median(b["em"][both])) if both.any() else np.nan,
            "matched_both_pairs": int(both.sum()),
            "matched_pct_A": 100 * float(np.mean(np.isfinite(a["em"]))), "matched_pct_B": 100 * float(np.mean(np.isfinite(b["em"])))}


class Hierarchy:
    """Day -> image -> cell index structure for the hierarchical bootstrap."""

    def __init__(self, keys: pd.DataFrame):
        self.n = len(keys)
        img_codes, imgs = pd.factorize(keys["stem"])
        order = np.argsort(img_codes, kind="stable")
        self.order = order                                           # cell rows grouped by image
        counts = np.bincount(img_codes, minlength=len(imgs))
        self.img_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.img_count = counts
        day_of_img = keys.groupby("stem")["storage_day"].first().reindex(imgs).to_numpy()
        self.days = np.unique(day_of_img)
        self.day_imgs = [np.nonzero(day_of_img == d)[0] for d in self.days]
        self.day_of_cell = keys["storage_day"].to_numpy()

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        dsel = rng.integers(0, len(self.days), len(self.days))
        imgs = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in (self.day_imgs[d] for d in dsel)])
        cnt = self.img_count[imgs]; start = self.img_start[imgs]
        rep_cnt = np.repeat(cnt, cnt)
        pos = np.repeat(start, cnt) + (rng.random(int(cnt.sum())) * rep_cnt).astype(np.int64)
        return self.order[pos]


def day_swap_p(A, B, H: Hierarchy, rows_mask: np.ndarray, observed: Dict[str, float]) -> Dict[str, float]:
    """Exact two-sided permutation p, swapping A/B labels per storage day (2^n_days patterns)."""
    nd = len(H.days)
    hits = {k: 0 for k in observed}; total = 0
    rows = np.nonzero(rows_mask)[0]
    day_idx = np.searchsorted(H.days, H.day_of_cell[rows])
    for pattern in itertools.product((False, True), repeat=nd):
        flip = np.asarray(pattern)[day_idx][:, None]
        a = {k: np.where(flip, B[k][rows], A[k][rows]) for k in A}
        b = {k: np.where(flip, A[k][rows], B[k][rows]) for k in A}
        s = stats(a, b, np.arange(len(rows)))
        total += 1
        for k, v in observed.items():
            if np.isfinite(v) and np.isfinite(s[k]) and abs(s[k]) >= abs(v) - 1e-12:
                hits[k] += 1
    return {k: (hits[k] / total if np.isfinite(observed[k]) else np.nan) for k in observed}


def analyse(arch: str, cond: str, arm_a: str, arm_b: str, seeds: List[int], B_boot: int, rng) -> Dict:
    per_seed, A, Bv, keys_ref, identical = {}, {k: [] for k in ("w", "ea", "em", "dme")}, {k: [] for k in ("w", "ea", "em", "dme")}, None, True
    for s in seeds:
        ca, cb = load_cells(arch, cond, arm_a, s), load_cells(arch, cond, arm_b, s)
        same = ca[["stem", "cell"]].equals(cb[["stem", "cell"]])
        if not same:
            m = ca[["stem", "cell"]].merge(cb[["stem", "cell"]], how="inner")
            ca = ca.merge(m, how="inner").sort_values(["stem", "cell"]).reset_index(drop=True)
            cb = cb.merge(m, how="inner").sort_values(["stem", "cell"]).reset_index(drop=True)
        if keys_ref is None:
            keys_ref = ca[["stem", "cell", "storage_day", "touching"]].copy()
        elif not ca[["stem", "cell"]].equals(keys_ref[["stem", "cell"]]):
            identical = False
            raise RuntimeError(f"{arch} {cond}: the scored reference cells differ between seeds")
        identical &= same
        pa, pb = per_seed_arrays(ca), per_seed_arrays(cb)
        for k in A:
            A[k].append(pa[k]); Bv[k].append(pb[k])
        one = {k: v[:, None] for k, v in pa.items()}; two = {k: v[:, None] for k, v in pb.items()}
        per_seed[str(s)] = {**stats(one, two, np.arange(len(ca))), **levels(one, two, np.arange(len(ca)))}
    A = {k: np.stack(v, 1) for k, v in A.items()}; Bv = {k: np.stack(v, 1) for k, v in Bv.items()}
    H = Hierarchy(keys_ref)
    groups = {"all": np.ones(H.n, bool), "touching": keys_ref["touching"].to_numpy(bool),
              "isolated": ~keys_ref["touching"].to_numpy(bool)}
    out = {"architecture": arch, "condition": cond, "loss_a": arm_a, "loss_b": arm_b,
           "label_a": ARM_LABEL[arm_a], "label_b": ARM_LABEL[arm_b], "seeds": seeds,
           "n_scored_cells": int(H.n), "n_images": int(len(H.img_count)), "n_days": int(len(H.days)),
           "reference_cells_identical_between_losses": bool(identical), "per_seed": per_seed, "groups": {}}
    boot = {g: {k: np.empty(B_boot) for k in STAT_DOC} for g in groups}
    for i in range(B_boot):
        rows = H.sample(rng)
        for g, mask in groups.items():
            r = rows[mask[rows]]
            s = stats(A, Bv, r) if len(r) else {k: np.nan for k in STAT_DOC}
            for k in STAT_DOC:
                boot[g][k][i] = s[k]
    for g, mask in groups.items():
        rows = np.nonzero(mask)[0]
        obs = stats(A, Bv, rows)
        p = day_swap_p(A, Bv, H, mask, obs) if g == "all" else {k: None for k in obs}
        out["groups"][g] = {"n_cells": int(mask.sum()), "levels": levels(A, Bv, rows), "statistics": {
            k: {"estimate": obs[k], "ci95_hierarchical_bootstrap": [float(np.nanpercentile(boot[g][k], 2.5)),
                                                                   float(np.nanpercentile(boot[g][k], 97.5))],
                "p_day_label_swap": p[k], "description": STAT_DOC[k]} for k in STAT_DOC}}
    return out


def summary_md(res: List[Dict], meta: Dict) -> str:
    L = ["# W-1 Cell-paired comparison of loss configurations", "",
         f"Created {meta['created']}. Checkpoint: last. A2 instances, scored reference cells. Statistics are A minus B, "
         "cells x seeds pooled. Interval: hierarchical bootstrap (days → images → cells, "
         f"{meta['bootstrap']} resamples). p: exact day-level label-swap permutation (2^11 patterns; smallest attainable "
         "p = 9.8e-4). p (Holm): within each architecture over comparisons × the two primary statistics.", "",
         "Reading the signs: for 'within 5 %' a positive difference favours A; for every error a negative difference favours A.", ""]
    for arch in sorted({r["architecture"] for r in res}):
        for cond in sorted({r["condition"] for r in res if r["architecture"] == arch}):
            rr = [r for r in res if r["architecture"] == arch and r["condition"] == cond]
            L += [f"## {arch}, {cond}", "",
                  f"{rr[0]['n_scored_cells']:,} scored reference cells in {rr[0]['n_images']} images, {rr[0]['n_days']} days; seeds {rr[0]['seeds']}.", ""]
            rows = []
            for r in rr:
                G = r["groups"]["all"]; S = G["statistics"]; lv = G["levels"]
                def cell(k, nd=2):
                    s = S[k]; lo, hi = s["ci95_hierarchical_bootstrap"]
                    return f"{fmt(s['estimate'], nd, True)} [{fmt(lo, nd, True)}, {fmt(hi, nd, True)}]"
                rows.append([f"{r['loss_a']} − {r['loss_b']}",
                             f"{lv['within5_pct_A']:.2f} vs {lv['within5_pct_B']:.2f}", cell("d_within5_pp"),
                             fmt_p(S["d_within5_pp"]["p_day_label_swap"]), fmt_p(S["d_within5_pp"].get("p_holm")),
                             f"{lv['median_abs_e_assigned_pct_A']:.3f} vs {lv['median_abs_e_assigned_pct_B']:.3f}",
                             cell("d_median_abs_e_assigned_pp", 3),
                             fmt_p(S["d_median_abs_e_assigned_pp"]["p_day_label_swap"]),
                             fmt_p(S["d_median_abs_e_assigned_pp"].get("p_holm")),
                             cell("d_median_abs_e_matched_both_pp", 3), cell("d_median_abs_drymass_err_both_pg", 3)])
            L.append(md_table(["A − B", "within 5 % (%), A vs B", "Δ within 5 % (pp) [95 % CI]", "p", "p (Holm)",
                               "median abs(e_assigned) (%), A vs B", "Δ median abs(e_assigned) (pp) [95 % CI]", "p", "p (Holm)",
                               "Δ median abs(e_matched), matched in both (pp)", "Δ median abs(dry-mass error), matched in both (pg)"], rows))
            L.append("")
            sub = []
            for r in rr:
                for g in ("touching", "isolated"):
                    s = r["groups"][g]["statistics"]["d_within5_pp"]; lo, hi = s["ci95_hierarchical_bootstrap"]
                    e = r["groups"][g]["statistics"]["d_median_abs_e_assigned_pp"]; elo, ehi = e["ci95_hierarchical_bootstrap"]
                    sub.append([f"{r['loss_a']} − {r['loss_b']}", g, f"{r['groups'][g]['n_cells']:,}",
                                f"{fmt(s['estimate'], 2, True)} [{fmt(lo, 2, True)}, {fmt(hi, 2, True)}]",
                                f"{fmt(e['estimate'], 3, True)} [{fmt(elo, 3, True)}, {fmt(ehi, 3, True)}]"])
            L += ["Touching and isolated cells (intervals only; no test):", "",
                  md_table(["A − B", "cells", "n", "Δ within 5 % (pp)", "Δ median abs(e_assigned) (pp)"], sub), ""]
            seeds = []
            for r in rr:
                for s, v in r["per_seed"].items():
                    seeds.append([f"{r['loss_a']} − {r['loss_b']}", s, fmt(v["d_within5_pp"], 2, True),
                                  fmt(v["d_median_abs_e_assigned_pp"], 3, True), fmt(v["d_median_abs_e_matched_both_pp"], 3, True)])
            L += ["Per seed (point estimates):", "",
                  md_table(["A − B", "seed", "Δ within 5 % (pp)", "Δ median abs(e_assigned) (pp)", "Δ median abs(e_matched) both (pp)"], seeds), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", default="representative",
                    help="'representative' (EdgeSAM r8, MobileSAM r64, MobileNet_UNet r32), 'all', or e.g. 'MobileSAM:r16,EdgeSAM:r8'")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--comparisons", default=None, help="override, e.g. 'dice_cipp:dice,dice_cipp:dice_pmc_bga_ipp'")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    global COMPARISONS
    if a.comparisons:
        COMPARISONS = [tuple(x.split(":")) for x in a.comparisons.split(",")]
    if a.conditions == "representative":
        configs = list(REPRESENTATIVE.items())
    elif a.conditions == "all":
        configs = [(arch, c) for arch in REPRESENTATIVE for c in ("r8", "r16", "r32", "r64", "fullft")]
    else:
        configs = [tuple(x.split(":")) for x in a.conditions.split(",")]
    rng = np.random.default_rng(a.rng_seed)
    res, inputs, t0 = [], [], time.time()
    for arch, cond in configs:
        for arm_a, arm_b in COMPARISONS:
            t1 = time.time()
            r = analyse(arch, cond, arm_a, arm_b, seeds, a.bootstrap, rng)
            res.append(r)
            inputs += [str(results_dir(arch, cond, x, s).relative_to(ROOT) / "cells_last.json") for x in (arm_a, arm_b) for s in seeds]
            st = r["groups"]["all"]["statistics"]
            print(f"[w1] {arch} {cond} {arm_a} - {arm_b}: d_within5 {st['d_within5_pp']['estimate']:+.2f} pp "
                  f"(p {st['d_within5_pp']['p_day_label_swap']:.4f}), d_median|e_assigned| "
                  f"{st['d_median_abs_e_assigned_pp']['estimate']:+.3f} pp  ({time.time() - t1:.0f} s)", flush=True)
    for arch in {r["architecture"] for r in res}:
        cells = [(r, k) for r in res if r["architecture"] == arch for k in PRIMARY]
        adj = holm([r["groups"]["all"]["statistics"][k]["p_day_label_swap"] for r, k in cells])
        for (r, k), p in zip(cells, adj):
            r["groups"]["all"]["statistics"][k]["p_holm"] = p
    meta = provenance("w1_cell_paired_stats.py", inputs, {"bootstrap": a.bootstrap, "rng_seed": a.rng_seed,
                                                          "seconds": time.time() - t0, "checkpoint": "last",
                                                          "instances": "A2", "cells": "scored reference cells",
                                                          "statistics": STAT_DOC, "primary_statistics": list(PRIMARY)})
    od = out_dir("w1_cell_paired")
    stem = "w1_cell_paired" + ("" if a.conditions == "representative" else ("_all" if a.conditions == "all" else "_custom"))
    meta["conditions_argument"] = a.conditions
    write_json(od / f"{stem}.json", {"metadata": meta, "results": res})
    write_text(od / f"{stem}_summary.md", summary_md(res, meta))


if __name__ == "__main__":
    main()
