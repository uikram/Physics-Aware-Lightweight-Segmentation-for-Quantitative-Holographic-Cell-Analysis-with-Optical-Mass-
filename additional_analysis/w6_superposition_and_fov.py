#!/usr/bin/env python
"""W-6  (a) Are the superposition candidates of W-5 stacked cells, dense single cells, or watershed artefacts?
       (b) Which fraction of the 768 x 768 grid is outside the annotated field of view, per split?

Part (a) works on the 11 (default) candidates in results/w5_reference_cells/w5_superposition_candidates.csv. For each
candidate it reads the prepared phase map, annotation mask and reference-cell map of its test image and computes:

  shape of the phase inside the cell (distance-normalised): mean phase in the CENTRE (normalised distance to the
     cell edge >= 0.5) vs the RIM (< 0.5), as a centre/rim ratio. A discocyte is biconcave, so its centre is
     thinner than its rim (ratio < 1). Two cells stacked along the optical axis, or a spherocyte-like dense cell, have
     a centre at least as high as the rim (ratio >= 1). The same ratio is computed for every reference cell of the same
     class in the test set, and the candidate's percentile is reported.
  max and 95th-percentile phase, with their percentiles among the same-class cells.
  neighbours: reference cells sharing a contact line; their mean phases; and the MERGED region (candidate +
     touching neighbours): if the high mean phase disappears on merging, the split may have given the candidate the
     phase-dense part of a cluster.
  watershed sensitivity: the annotated foreground of the image is split again with h = 1, 2, 3, 4, 5 (h = 3 is the
     pipeline value). The region that overlaps the candidate most is taken; its area and mean-phase ratio
     (against the same class/day median as W-5) say whether the candidate survives a different split.
  a verdict per candidate, from fixed rules (stated in the output), plus a figure panel for visual inspection.

Part (b) reads data/prepared/report.json and data/prepared/manifest.csv (valid_fraction per image) and recomputes the
fraction from the masks (pixels with label 255) for every prepared image, per split (train / val) and overall.

Inputs (read-only): data/prepared/{manifest.csv, report.json, train|val/{phase,mask,cells}/*.tif},
                    additional_analysis/results/w5_reference_cells/w5_superposition_candidates.csv
Requires: numpy, pandas, scipy, scikit-image, tifffile; matplotlib for the figures (skipped if absent).
Usage:   python additional_analysis/w6_superposition_and_fov.py
Outputs: additional_analysis/results/w6_superposition_and_fov/{w6_superposition.json, w6_fov.json, w6_summary.md,
         figures/candidate_<stem>_<cell>.png, figures/all_candidates.png}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PIXEL_SIZE_UM, ROOT, fmt, md_table, out_dir, provenance, write_json, write_text  # noqa: E402

sys.path.insert(0, str(ROOT))
CLASS_NAMES = ["background", "discocyte", "spherocyte", "echinocyte", "stomatocyte"]
H_VALUES = [1, 2, 3, 4, 5]
RULES = {
    "biconcave_cell": "centre/rim < 1 (thinner centre than rim): a normal discocyte profile at a high phase level",
    "split_artefact": "touching cell that is no longer a candidate (phase ratio < 1.6 or area ratio > 1.25) at some "
                      "watershed h in 1..5: the high mean phase depends on where the split puts the boundary",
    "boundary_uncertain": "touching cell that stays a candidate at every h, but a touching neighbour has a phase ratio "
                          "< 0.8 (phase may be assigned to the wrong side of the contact line)",
    "possible_stack": "centre/rim >= 1.15 and max phase >= the 99th percentile of its class (cannot be told apart from "
                      "a single optically thick cell)",
    "dense_single_cell": "none of the above: a single cell with a high phase level whose profile is not extreme enough for possible_stack",
}
RULE_ORDER = ["biconcave_cell", "split_artefact", "boundary_uncertain", "possible_stack", "dense_single_cell"]


def read(split, sub, stem):
    import tifffile
    return tifffile.imread(ROOT / "data" / "prepared" / split / sub / f"{stem}.tif")


def centre_rim(phase, region):
    """Mean phase of the centre (normalised edge distance >= 0.5) over the rim (< 0.5)."""
    d = ndi.distance_transform_edt(region)
    if d.max() <= 1:
        return np.nan
    dn = d / d.max()
    c, r = phase[region & (dn >= 0.5)], phase[region & (dn < 0.5)]
    return float(c.mean() / r.mean()) if len(c) and len(r) and r.mean() != 0 else np.nan


def per_cell_shape(phase, cells, valid):
    """centre/rim, max, p95 for every reference cell of one image (keyed by label)."""
    out = {}
    lab = np.where(valid, cells, 0)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        pad = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
        reg = lab[pad] == i
        if reg.sum() < 5:
            continue
        ph = phase[pad]
        out[i] = {"centre_rim": centre_rim(ph, reg), "max": float(ph[reg].max()), "p95": float(np.percentile(ph[reg], 95)),
                  "mean": float(ph[reg].mean()), "n_px": int(reg.sum())}
    return out


def neighbours(cells, valid, c):
    lab = np.where(valid, cells, 0)
    reg = lab == c
    ring = ndi.binary_dilation(reg, structure=ndi.generate_binary_structure(2, 1)) & ~reg
    nb = np.unique(lab[ring]); return [int(x) for x in nb if x > 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=str(ROOT / "additional_analysis" / "results" / "w5_reference_cells" /
                                                "w5_superposition_candidates.csv"))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    from qpicell.data.cells import split_cells          # the pipeline's own watershed (read-only use)
    od = out_dir("w6_superposition_and_fov"); figdir = od / "figures"
    prep = ROOT / "data" / "prepared"
    if not (prep / "manifest.csv").exists():
        sys.exit(f"[w6] {prep / 'manifest.csv'} not found: run this on the machine that has data/prepared")
    man = pd.read_csv(prep / "manifest.csv")
    man["registration_ok"] = man["registration_ok"].astype(str) == "True"
    ok = man[man["registration_ok"]]

    # ---------------- (b) field of view ----------------
    fov_rows = []
    for _, r in ok.iterrows():
        m = read(r["split"], "mask", r["stem"])
        fov_rows.append({"split": r["split"], "stem": r["stem"], "storage_day": int(r["storage_day"]),
                         "valid_fraction_recomputed": float((m != 255).mean()),
                         "valid_fraction_manifest": float(r["valid_fraction"]) if "valid_fraction" in r and pd.notna(r["valid_fraction"]) else None})
    F = pd.DataFrame(fov_rows)
    F["unannotated_fraction"] = 1 - F["valid_fraction_recomputed"]
    def summ(x):
        return {"n_images": int(len(x)), "unannotated_mean": float(x.mean()), "unannotated_median": float(x.median()),
                "unannotated_min": float(x.min()), "unannotated_max": float(x.max()),
                "images_fully_annotated": int((x == 0).sum())}
    fov = {"per_split": {s: summ(g["unannotated_fraction"]) for s, g in F.groupby("split")},
           "all_images": summ(F["unannotated_fraction"]),
           "manifest_matches_recomputed": bool(np.allclose(F["valid_fraction_manifest"].astype(float),
                                                           F["valid_fraction_recomputed"], atol=1e-4))
           if F["valid_fraction_manifest"].notna().all() else None}
    rep = prep / "report.json"
    if rep.exists():
        fov["report_json_valid_fraction"] = json.loads(rep.read_text()).get("valid_fraction")
    write_json(od / "w6_fov.json", {"metadata": provenance("w6_superposition_and_fov.py",
                                                           ["data/prepared/manifest.csv", "data/prepared/report.json",
                                                            "data/prepared/{train,val}/mask/*.tif"]),
                                    "results": fov, "per_image": F.to_dict("records")})

    # ---------------- (a) superposition candidates ----------------
    cand = pd.read_csv(a.candidates)
    val = ok[ok["split"] == "val"].set_index("stem")
    # reference distribution: every reference cell of the test images
    ref, cache = [], {}
    for stem, r in val.iterrows():
        ph = read("val", "phase", stem).astype(np.float64); mk = read("val", "mask", stem); cl = read("val", "cells", stem).astype(np.int64)
        valid = mk != 255
        cache[stem] = (ph, mk, cl, valid)
        cls = ndi.labeled_comprehension(np.where(valid, mk, 0), np.where(valid, cl, 0), np.arange(cl.max() + 1),
                                        lambda v: np.bincount(v, minlength=5)[1:].argmax() + 1, int, 0)
        for i, s in per_cell_shape(ph, cl, valid).items():
            ref.append({"stem": stem, "cell": i, "storage_day": int(r["storage_day"]), "gt_class": CLASS_NAMES[int(cls[i])], **s})
    R = pd.DataFrame(ref)
    R["area_um2"] = R["n_px"] * PIXEL_SIZE_UM ** 2
    grp = R.groupby(["storage_day", "gt_class"]).agg(med_mean=("mean", "median"), med_area=("area_um2", "median")).reset_index()

    def pct(col, cls, v):
        x = R.loc[R["gt_class"] == cls, col].dropna().to_numpy()
        return float(100 * (x < v).mean()) if len(x) and np.isfinite(v) else None

    results = []
    for _, c in cand.iterrows():
        stem, cid, day, cls = c["stem"], int(c["cell"]), int(c["storage_day"]), c["gt_class"]
        ph, mk, cl, valid = cache[stem]
        g = grp[(grp["storage_day"] == day) & (grp["gt_class"] == cls)].iloc[0]
        me = R[(R["stem"] == stem) & (R["cell"] == cid)].iloc[0]
        nbs = neighbours(cl, valid, cid)
        merged = np.isin(np.where(valid, cl, 0), [cid] + nbs)
        merged_mean = float(ph[merged].mean()); merged_area = float(merged.sum() * PIXEL_SIZE_UM ** 2)
        nb_info = []
        for n in nbs:
            rr = R[(R["stem"] == stem) & (R["cell"] == n)]
            if len(rr):
                nb_info.append({"cell": n, "gt_class": rr.iloc[0]["gt_class"], "mean_phase": float(rr.iloc[0]["mean"]),
                                "phase_ratio": float(rr.iloc[0]["mean"] / g["med_mean"]), "area_um2": float(rr.iloc[0]["area_um2"])})
        fg = (mk > 0) & valid
        reg0 = np.where(valid, cl, 0) == cid
        hs = []
        for h in H_VALUES:
            lab = split_cells(fg, h)
            ids, cnt = np.unique(lab[reg0 & (lab > 0)], return_counts=True)
            if not len(ids):
                hs.append({"h": h, "found": False}); continue
            j = int(ids[np.argmax(cnt)]); rg = lab == j
            area = float(rg.sum() * PIXEL_SIZE_UM ** 2); mean = float(ph[rg].mean())
            hs.append({"h": h, "found": True, "area_um2": area, "mean_phase": mean,
                       "phase_ratio": mean / g["med_mean"], "area_ratio": area / g["med_area"],
                       "overlap_with_h3_cell": float((rg & reg0).sum() / max(reg0.sum(), 1)),
                       "still_candidate": bool(mean / g["med_mean"] >= 1.6 and area / g["med_area"] <= 1.25)})
        merged_ratio = merged_mean / g["med_mean"]
        persist_h = all(x.get("still_candidate", False) for x in hs if x["found"])
        cr = me["centre_rim"]
        low_nb = any(n["phase_ratio"] < 0.8 for n in nb_info)
        if np.isfinite(cr) and cr < 1:
            verdict = "biconcave_cell"
        elif c["touching"] and not persist_h:
            verdict = "split_artefact"
        elif c["touching"] and low_nb:
            verdict = "boundary_uncertain"
        elif np.isfinite(cr) and cr >= 1.15 and (pct("max", cls, me["max"]) or 0) >= 99:
            verdict = "possible_stack"
        else:
            verdict = "dense_single_cell"
        results.append({"stem": stem, "cell": cid, "storage_day": day, "gt_class": cls, "touching": bool(c["touching"]),
                        "area_um2": float(c["area_um2"]), "mean_phase_rad": float(c["mean_phase_rad"]),
                        "phase_ratio": float(c["phase_ratio"]), "area_ratio": float(c["area_ratio"]),
                        "max_phase_rad": me["max"], "max_phase_percentile_in_class": pct("max", cls, me["max"]),
                        "p95_phase_rad": me["p95"], "centre_rim_ratio": cr,
                        "centre_rim_percentile_in_class": pct("centre_rim", cls, cr),
                        "class_median_centre_rim": float(R.loc[R["gt_class"] == cls, "centre_rim"].median()),
                        "neighbours": nb_info, "neighbour_with_low_phase_ratio": bool(low_nb), "merged_with_neighbours": {"cells": [cid] + nbs, "area_um2": merged_area,
                                                                          "mean_phase": merged_mean, "phase_ratio": merged_ratio},
                        "watershed_h_sensitivity": hs, "candidate_at_every_h": persist_h, "verdict": verdict})
        if not a.no_figures:
            _figure(figdir, stem, cid, ph, mk, cl, valid, results[-1])
    if not a.no_figures and results:
        _montage(figdir, results, cache)
    counts = {k: sum(r["verdict"] == k for r in results) for k in RULE_ORDER}
    meta = provenance("w6_superposition_and_fov.py", [str(Path(a.candidates).relative_to(ROOT)), "data/prepared/val/{phase,mask,cells}/*.tif"],
                      {"rules": RULES, "h_values": H_VALUES, "centre_definition": "normalised distance to the cell edge >= 0.5",
                       "reference_distribution": f"{len(R)} reference cells of the {len(val)} test images"})
    write_json(od / "w6_superposition.json", {"metadata": meta, "verdict_counts": counts, "candidates": results,
                                             "class_reference": R.groupby("gt_class")[["centre_rim", "max", "mean"]].describe().round(4).to_dict()})

    L = ["# W-6 Superposition candidates and the annotated field of view", "", "## (a) Candidates (W-5, ratio 1.6)", "",
         "Verdict counts: " + ", ".join(f"{k} {v}" for k, v in counts.items()), "", "Rules:", ""]
    L += [f"- **{k}**: {v}" for k, v in RULES.items()] + [""]
    L.append(md_table(["stem", "cell", "day", "touching", "phase ratio", "centre/rim (class pct)", "max phase rad (class pct)",
                       "merged ratio (info)", "neighbour ratio < 0.8", "candidate at every h", "verdict"],
                      [[r["stem"], r["cell"], r["storage_day"], r["touching"], fmt(r["phase_ratio"], 2),
                        f"{fmt(r['centre_rim_ratio'], 2)} ({fmt(r['centre_rim_percentile_in_class'], 0)})",
                        f"{fmt(r['max_phase_rad'], 2)} ({fmt(r['max_phase_percentile_in_class'], 1)})",
                        fmt(r["merged_with_neighbours"]["phase_ratio"], 2), r["neighbour_with_low_phase_ratio"], r["candidate_at_every_h"], r["verdict"]] for r in results]))
    L += ["", f"Class medians of the centre/rim ratio: " +
          ", ".join(f"{k} {v:.2f}" for k, v in R.groupby('gt_class')['centre_rim'].median().items()), "",
          "Figures: figures/candidate_<stem>_<cell>.png (phase crop with reference-cell boundaries; candidate in red) and figures/all_candidates.png.", "",
          "## (b) Fraction of the 768 × 768 grid outside the annotated field of view", ""]
    L.append(md_table(["set", "images", "mean", "median", "min", "max", "fully annotated images"],
                      [[k, v["n_images"], f"{100 * v['unannotated_mean']:.1f} %", f"{100 * v['unannotated_median']:.1f} %",
                        f"{100 * v['unannotated_min']:.1f} %", f"{100 * v['unannotated_max']:.1f} %", v["images_fully_annotated"]]
                       for k, v in list(fov["per_split"].items()) + [("all", fov["all_images"])]]))
    L += ["", f"manifest valid_fraction equals the recomputed value: {fov['manifest_matches_recomputed']}",
          f"report.json valid_fraction: {fov.get('report_json_valid_fraction')}"]
    write_text(od / "w6_summary.md", "\n".join(L))


def _figure(figdir, stem, cid, ph, mk, cl, valid, r):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage.segmentation import find_boundaries
    except Exception:
        return
    figdir.mkdir(parents=True, exist_ok=True)
    reg = (np.where(valid, cl, 0) == cid)
    ys, xs = np.nonzero(reg); m = 40
    sl = (slice(max(ys.min() - m, 0), ys.max() + m), slice(max(xs.min() - m, 0), xs.max() + m))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    im = ax[0].imshow(ph[sl], cmap="viridis", vmin=0, vmax=max(3.0, float(ph[sl].max())))
    b = find_boundaries(np.where(valid, cl, 0)[sl], mode="inner")
    ov = np.zeros(b.shape + (4,)); ov[b] = (1, 1, 1, 0.8); ov[find_boundaries(reg[sl], mode="inner")] = (1, 0, 0, 1)
    ax[0].imshow(ov); ax[0].set_title(f"phase (rad): {stem}, cell {cid}, day {r['storage_day']}", fontsize=9); fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].imshow(np.where(valid, mk, 0)[sl], cmap="tab10", vmin=0, vmax=9, interpolation="nearest"); ax[1].imshow(ov)
    ax[1].set_title("annotation class\n(1 disco, 2 sphero, 3 echino, 4 stomato); red = candidate", fontsize=9)
    d = ndi.distance_transform_edt(reg); dn = d / max(d.max(), 1)
    bins = np.linspace(0, 1, 11); prof = [ph[reg & (dn >= lo) & (dn < hi)].mean() if (reg & (dn >= lo) & (dn < hi)).any() else np.nan
                                          for lo, hi in zip(bins[:-1], bins[1:])]
    ax[2].plot(0.5 * (bins[:-1] + bins[1:]), prof, "o-"); ax[2].set_xlabel("normalised distance from edge (0 = edge, 1 = centre)", fontsize=9)
    ax[2].set_ylabel("mean phase (rad)"); ax[2].set_title(f"radial profile, centre/rim {r['centre_rim_ratio']:.2f}\nverdict: {r['verdict']}", fontsize=9)
    for x in ax[:2]:
        x.set_xticks([]); x.set_yticks([])
    fig.tight_layout(); fig.savefig(figdir / f"candidate_{stem}_{cid}.png", dpi=130); plt.close(fig)


def _montage(figdir, results, cache):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage.segmentation import find_boundaries
    except Exception:
        return
    n = len(results); cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    for ax, r in zip(axes.ravel(), results):
        ph, mk, cl, valid = cache[r["stem"]]
        reg = np.where(valid, cl, 0) == r["cell"]; ys, xs = np.nonzero(reg); m = 40
        sl = (slice(max(ys.min() - m, 0), ys.max() + m), slice(max(xs.min() - m, 0), xs.max() + m))
        ax.imshow(ph[sl], cmap="viridis", vmin=0, vmax=3.5)
        ov = np.zeros(reg[sl].shape + (4,)); ov[find_boundaries(np.where(valid, cl, 0)[sl], mode="inner")] = (1, 1, 1, 0.7)
        ov[find_boundaries(reg[sl], mode="inner")] = (1, 0, 0, 1); ax.imshow(ov)
        ax.set_title(f"{r['stem']} #{r['cell']} d{r['storage_day']}\n{r['verdict']}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(figdir / "all_candidates.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
