"""W-7 - three read-only follow-up checks for the manuscript revision.

Reads only results/<Arch>/<rank>/<arm>/seed<s>/cells_last.json (the per-cell files written by rescore.py) and
results/consolidated/selection/rank_selection.csv. Writes only additional_analysis/results/w7_followup/.
No training, no inference, no images, no GPU. Run from the project root:

    python additional_analysis/w7_followup_checks.py
    python additional_analysis/w7_followup_checks.py --threshold 10 --stems 20110627_1,20110607_9,20110530_6,<disc>

Part A  (item 10, Fig. 7 outliers) - for the 9 representative runs (proposed objective, last checkpoint): every
        matched, scored reference cell whose predicted dry mass differs from the reference by more than
        --threshold pg (default 10), with storage day, class, area, IoU and the merged / fragmented / touching
        flags; how often these flags occur among the outliers and among all matched scored cells; and the
        Bland-Altman limits of agreement with and without the outliers (descriptive, no test).
Part B  (item 9, Fig. 6 statements) - cell-level class confusion (reference class x predicted class of matched
        scored cells; missed cells separately) for the 9 representative runs, overall and per image. The rows of
        the --stems images (the images shown in Fig. 6) are also written for seed 42. NOTE: whole images, not the
        192 x 192 px windows of Fig. 6 (the window position is not stored in any result file); a window is part
        of its image, so the image counts bound but do not equal what the window shows.
Part C  (item 13, the offsetting-error hypothesis of the Discussion) - for the previous full objective
        (dice_pmc_bga_ipp) and the new full objective (dice_pmc_bga_cipp), every LoRA rank and seed: per image,
        the signed matching-free error dS_c = S_assigned - S_gt of each scored reference cell (predicted phase in
        the cell's territory minus reference phase), and the cancellation ratio
            C = |sum_c dS_c| / sum_c |dS_c|      (1: all errors have the same sign; near 0: errors cancel)
        together with the net and gross errors relative to sum_c S_gt. Per run: medians over the 66 images; per
        architecture: paired comparison of the two objectives over the 12 rank-seed pairs (exact Wilcoxon,
        UNADJUSTED, exploratory). Also the mean signed e_assigned of merged and of non-merged cells.
        LIMITATION: only reference-cell territories are covered, so predicted phase far from any reference cell
        (false-positive foreground) is not seen; the global integrated-phase term of training also sees it.

Every number is computed from the per-cell files; nothing is estimated or filled in. Files without
metadata.rescored are reported and, unless --allow-unrescored is given, the script stops.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SEEDS, ARM_LABEL, results_dir, out_dir, write_json  # noqa: E402

ARCHES = ["EdgeSAM", "MobileSAM", "MobileNet_UNet"]
ARCH_KEY = {"EdgeSAM": "edge_sam", "MobileSAM": "mobile_sam", "MobileNet_UNet": "mobilenet_unet"}
RANKS = ["r8", "r16", "r32", "r64"]
PROPOSED, OLD_FULL, NEW_FULL = "dice_cipp", "dice_pmc_bga_ipp", "dice_pmc_bga_cipp"
CLASSES = ["discocyte", "spherocyte", "echinocyte", "stomatocyte"]


def representative() -> dict:
    """Representative rank per architecture, read from rank_selection.csv (not hard-coded)."""
    path = ROOT / "results" / "consolidated" / "selection" / "rank_selection.csv"
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    disp = {v: k for k, v in ARCH_KEY.items()}
    rep = {}
    for r in rows:
        if float(r["tolerance_points"]) == 0.0:
            rep[disp[r["architecture"]]] = r["selected_condition"]
    return rep


def load_cells(arch: str, cond: str, arm: str, seed: int, allow_unrescored: bool, problems: list):
    p = results_dir(arch, cond, arm, seed) / "cells_last.json"
    if not p.exists():
        problems.append(f"missing: {p.relative_to(ROOT)}")
        return None
    d = json.loads(p.read_text())
    if not d.get("metadata", {}).get("rescored") and not allow_unrescored:
        problems.append(f"not rescored (no metadata.rescored): {p.relative_to(ROOT)}")
        return None
    return [c for c in d["cells"] if c.get("instances") == "A2"]


def loa(diff: np.ndarray):
    b, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    return {"n": int(diff.size), "bias_pg": b, "loa_low_pg": b - 1.96 * sd, "loa_high_pg": b + 1.96 * sd}


def part_a(rep, thr, allow, problems, out):
    rows, summary = [], {}
    for arch in ARCHES:
        cond = rep[arch]
        for s in SEEDS:
            cells = load_cells(arch, cond, PROPOSED, s, allow, problems)
            if cells is None:
                continue
            m = [c for c in cells if c["scored"] and c["matched"] and c.get("dm_pred_pg") is not None]
            diff = np.array([c["dm_pred_pg"] - c["dm_gt_pg"] for c in m], float)
            mean = np.array([(c["dm_pred_pg"] + c["dm_gt_pg"]) / 2 for c in m], float)
            out_idx = np.where(np.abs(diff) > thr)[0]
            for i in out_idx:
                c = m[i]
                rows.append({"architecture": arch, "rank": cond, "seed": s, "stem": c["stem"],
                             "storage_day": c["storage_day"], "cell": c["cell"], "gt_class": c["gt_class"],
                             "pred_class": c.get("pred_class"), "area_um2": c["area_um2"],
                             "area_pred_um2": c.get("area_pred_um2"), "iou": c.get("iou"),
                             "dm_gt_pg": c["dm_gt_pg"], "dm_pred_pg": c["dm_pred_pg"],
                             "difference_pg": float(diff[i]), "mean_pg": float(mean[i]),
                             "merged": c["merged"], "fragmented": c["fragmented"], "touching": c["touching"],
                             "pred_wrapped": c.get("pred_wrapped")})
            keep = np.abs(diff) <= thr

            def share(flag, idx):
                return float(np.mean([bool(m[i][flag]) for i in idx])) if len(idx) else None
            all_idx = np.arange(len(m))
            summary[f"{arch} {cond} seed{s}"] = {
                "matched_scored_cells": len(m),
                "outliers_above_threshold": int((diff > thr).sum()),
                "outliers_below_minus_threshold": int((diff < -thr).sum()),
                "share_merged": {"outliers": share("merged", out_idx), "all_matched": share("merged", all_idx)},
                "share_touching": {"outliers": share("touching", out_idx), "all_matched": share("touching", all_idx)},
                "share_fragmented": {"outliers": share("fragmented", out_idx),
                                     "all_matched": share("fragmented", all_idx)},
                "outlier_mean_pg_range": ([float(mean[out_idx].min()), float(mean[out_idx].max())]
                                          if len(out_idx) else None),
                "outlier_difference_pg_range": ([float(diff[out_idx].min()), float(diff[out_idx].max())]
                                                if len(out_idx) else None),
                "outliers_by_class": dict(Counter(m[i]["gt_class"] for i in out_idx)),
                "outliers_by_day": dict(Counter(int(m[i]["storage_day"]) for i in out_idx)),
                "limits_of_agreement_all": loa(diff) if diff.size > 1 else None,
                "limits_of_agreement_without_outliers": loa(diff[keep]) if keep.sum() > 1 else None,
            }
    with open(out / "w7a_dry_mass_outliers.csv", "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return summary


def part_b(rep, stems, allow, problems, out):
    overall, per_image = {}, []
    labels = CLASSES + ["(missed)"]
    for arch in ARCHES:
        cond = rep[arch]
        for s in SEEDS:
            cells = load_cells(arch, cond, PROPOSED, s, allow, problems)
            if cells is None:
                continue
            M = defaultdict(lambda: Counter())
            img = defaultdict(lambda: defaultdict(Counter))
            for c in cells:
                if not c["scored"]:
                    continue
                pc = c.get("pred_class") if c["matched"] else ("(missed)" if c["missed"] else "(unmatched)")
                M[c["gt_class"]][pc] += 1
                img[c["stem"]][c["gt_class"]][pc] += 1
            overall[f"{arch} {cond} seed{s}"] = {g: dict(M[g]) for g in CLASSES}
            for stem, G in img.items():
                for g, cnt in G.items():
                    for pc, n in cnt.items():
                        per_image.append({"architecture": arch, "rank": cond, "seed": s, "stem": stem,
                                          "gt_class": g, "pred_class": pc, "cells": n})
    with open(out / "w7b_class_confusion_per_image.csv", "w", newline="") as fh:
        if per_image:
            w = csv.DictWriter(fh, fieldnames=list(per_image[0].keys()))
            w.writeheader(); w.writerows(per_image)
    fig6 = [r for r in per_image if r["stem"] in stems and r["seed"] == 42]
    return overall, fig6, labels


def part_c(allow, problems, out):
    from scipy.stats import wilcoxon
    per_run, img_rows = {}, []
    for arch in ARCHES:
        for cond in RANKS:
            for arm in (OLD_FULL, NEW_FULL):
                for s in SEEDS:
                    cells = load_cells(arch, cond, arm, s, allow, problems)
                    if cells is None:
                        continue
                    by = defaultdict(list)
                    for c in cells:
                        if c["scored"] and c.get("S_assigned") is not None:
                            by[c["stem"]].append(c)
                    C, net, gross = [], [], []
                    for stem, cs in by.items():
                        dS = np.array([c["S_assigned"] - c["S_gt"] for c in cs], float)
                        Sg = float(sum(c["S_gt"] for c in cs))
                        g = float(np.abs(dS).sum())
                        ci = abs(float(dS.sum())) / g if g > 0 else np.nan
                        C.append(ci); net.append(abs(float(dS.sum())) / Sg); gross.append(g / Sg)
                        img_rows.append({"architecture": arch, "rank": cond, "objective": arm, "seed": s,
                                         "stem": stem, "cells": len(cs), "cancellation_ratio": ci,
                                         "net_error": net[-1], "gross_error": gross[-1]})
                    em = [c["e_assigned"] for cs in by.values() for c in cs if c["merged"]]
                    en = [c["e_assigned"] for cs in by.values() for c in cs if not c["merged"]]
                    per_run[(arch, cond, arm, s)] = {
                        "median_cancellation_ratio": float(np.nanmedian(C)),
                        "median_net_error": float(np.median(net)), "median_gross_error": float(np.median(gross)),
                        "mean_signed_e_assigned_merged": float(np.mean(em)) if em else None,
                        "mean_signed_e_assigned_not_merged": float(np.mean(en)) if en else None,
                        "merged_cells": len(em)}
    with open(out / "w7c_cancellation_per_image.csv", "w", newline="") as fh:
        if img_rows:
            w = csv.DictWriter(fh, fieldnames=list(img_rows[0].keys()))
            w.writeheader(); w.writerows(img_rows)
    summary = {}
    for arch in ARCHES:
        pairs = [(per_run.get((arch, r, OLD_FULL, s)), per_run.get((arch, r, NEW_FULL, s)))
                 for r in RANKS for s in SEEDS]
        pairs = [(a, b) for a, b in pairs if a and b]
        if not pairs:
            continue
        res = {"pairs": len(pairs)}
        for k in ("median_cancellation_ratio", "median_net_error", "median_gross_error",
                  "mean_signed_e_assigned_merged"):
            a = np.array([p[0][k] for p in pairs], float); b = np.array([p[1][k] for p in pairs], float)
            ok = ~(np.isnan(a) | np.isnan(b))
            a, b = a[ok], b[ok]
            p = float(wilcoxon(a, b, method="exact").pvalue) if len(a) >= 5 and np.any(a != b) else None
            res[k] = {"old_full_mean": float(a.mean()), "new_full_mean": float(b.mean()),
                      "median_difference_new_minus_old": float(np.median(b - a)),
                      "pairs_new_lower": int((b < a).sum()), "wilcoxon_p_unadjusted_exploratory": p}
        summary[arch] = res
    return summary, {f"{k[0]} {k[1]} {k[2]} seed{k[3]}": v for k, v in per_run.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=10.0, help="|prediction - reference| dry mass, pg")
    ap.add_argument("--stems", default="20110627_1,20110607_9,20110530_6",
                    help="comma-separated images of Fig. 6 (add the discocyte row's image)")
    ap.add_argument("--allow-unrescored", action="store_true", help="testing only")
    a = ap.parse_args()
    out = out_dir("w7_followup")
    problems: list = []
    rep = representative()
    print("representative ranks (rank_selection.csv):", rep, flush=True)

    A = part_a(rep, a.threshold, a.allow_unrescored, problems, out)
    B, fig6, labels = part_b(rep, [s.strip() for s in a.stems.split(",") if s.strip()],
                             a.allow_unrescored, problems, out)
    C, C_runs = part_c(a.allow_unrescored, problems, out)

    meta = {"script": "additional_analysis/w7_followup_checks.py", "threshold_pg": a.threshold,
            "representative": rep, "stems": a.stems, "problems": problems,
            "note_part_b": "whole images, not the Fig. 6 windows",
            "note_part_c": "reference-cell territories only; exact Wilcoxon, unadjusted, exploratory"}
    write_json(out / "w7_followup.json", {"metadata": meta, "A_dry_mass_outliers": A,
                                          "B_class_confusion": B, "B_fig6_images_seed42": fig6,
                                          "C_cancellation_summary": C, "C_cancellation_per_run": C_runs})

    L = ["# W-7 follow-up checks", "", f"Representative ranks: {rep}. Threshold: {a.threshold} pg.", ""]
    if problems:
        L += ["**Problems (files skipped):**", ""] + [f"- {p}" for p in problems] + [""]
    L += ["## A. Dry-mass outliers (matched scored cells, |prediction - reference| > threshold)", "",
          "| Run | matched | > +thr | < -thr | merged (out / all) | touching (out / all) | fragmented (out / all) |"
          " mean pg range of outliers | LoA all | LoA without outliers |", "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "–" if v is None else f"{v:.2f}"
    for k, v in A.items():
        la, lb = v["limits_of_agreement_all"], v["limits_of_agreement_without_outliers"]
        L.append(f"| {k} | {v['matched_scored_cells']} | {v['outliers_above_threshold']} | "
                 f"{v['outliers_below_minus_threshold']} | {f(v['share_merged']['outliers'])} / "
                 f"{f(v['share_merged']['all_matched'])} | {f(v['share_touching']['outliers'])} / "
                 f"{f(v['share_touching']['all_matched'])} | {f(v['share_fragmented']['outliers'])} / "
                 f"{f(v['share_fragmented']['all_matched'])} | {v['outlier_mean_pg_range']} | "
                 f"{f(la['loa_low_pg'])} to {f(la['loa_high_pg'])} | "
                 f"{f(lb['loa_low_pg']) if lb else '–'} to {f(lb['loa_high_pg']) if lb else '–'} |")
    L += ["", "Every outlier cell: `w7a_dry_mass_outliers.csv`.", "",
          "## B. Cell-level class confusion (scored cells; rows: reference class)", ""]
    for k, M in B.items():
        cols = sorted({pc for g in M.values() for pc in g}, key=lambda x: (x.startswith("("), x))
        L += [f"**{k}**", "", "| reference \\ predicted | " + " | ".join(cols) + " |",
              "|---|" + "---|" * len(cols)]
        for g in CLASSES:
            L.append(f"| {g} | " + " | ".join(str(M.get(g, {}).get(c, 0)) for c in cols) + " |")
        L.append("")
    L += ["Fig. 6 images (seed 42, whole image):", "", "| Architecture | Image | Reference class | Predicted | Cells |",
          "|---|---|---|---|---|"]
    for r in sorted(fig6, key=lambda r: (r["stem"], r["architecture"], r["gt_class"], -r["cells"])):
        L.append(f"| {r['architecture']} | {r['stem']} | {r['gt_class']} | {r['pred_class']} | {r['cells']} |")
    L += ["", "## C. Offsetting cell errors: previous full (global IPP) against new full (cIPP), 12 pairs",
          "", "Cancellation ratio C = |sum dS| / sum |dS| per image (1 = no cancellation). Exploratory, unadjusted.",
          "", "| Architecture | quantity | previous full | new full | median diff (new - old) | pairs new lower | p |",
          "|---|---|---|---|---|---|---|"]
    for arch, res in C.items():
        for q, v in res.items():
            if q == "pairs":
                continue
            p = v["wilcoxon_p_unadjusted_exploratory"]
            L.append(f"| {arch} | {q} | {v['old_full_mean']:.4f} | {v['new_full_mean']:.4f} | "
                     f"{v['median_difference_new_minus_old']:+.4f} | {v['pairs_new_lower']} / {res['pairs']} | "
                     f"{'–' if p is None else f'{p:.3g}'} |")
    (out / "w7_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[write] {(out / 'w7_summary.md').relative_to(ROOT)}")
    if problems:
        print(f"{len(problems)} files skipped - see w7_summary.md", flush=True)
        if not a.allow_unrescored and any("not rescored" in p for p in problems):
            sys.exit(2)


if __name__ == "__main__":
    main()
