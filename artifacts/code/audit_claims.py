"""Recompute every number of the revised manuscript that can be checked from outputs/ (read only).

Writes artifacts/audit/claims_check.json and prints a readable summary. Analyses:
  1. run design (180 LoRA runs + 3 full fine-tuning runs)
  2. seed-paired IPE comparisons of the loss ablation over the full rank sweep (45 per comparison)
  3. lost-class runs per architecture and loss
  4. date-clustered storage statistics (permutation over the 11 dates, date bootstrap, leave-one-date-out)
     for the annotation and for every seed of EdgeSAM (r=32) and MobileSAM (r=8)
  5. a storage-dependent dry-mass bias test: per-image (prediction - annotation) median dry mass vs storage day
  6. class composition differences, kept-component ratios
  7. dry-mass agreement per seed, per-cell IPE, low-phase false-negative strata, robustness, benchmark

Not checkable here (need data/processed, which is not in the repository): annotated field-of-view fraction, training-map
offsets, day-15 foreground fraction, bridged-component rates, pixel confusion shares, component fate (fragments, >200 um2,
touching), predicted/annotated foreground area ratio.
"""
from __future__ import annotations

import json
import math
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from common import (ART, ARCHS, config_key, CLASSES, COLLAPSE_DICE, LOSSES, RANKS, SEEDS, OUT, cell_unit, cells_gt, cells_pred,
                    compiled, metric, paired_diff, runs_frame)

RNG_SEED = 20261029
N_PERM, N_BOOT = 20000, 20000
EARLY, LATE = (0, 5, 8), (37, 47)


def per_image(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.groupby(["stem", "storage_day"])[col].median().reset_index()


def date_clustered(img: pd.DataFrame, col: str, rng: np.random.Generator) -> dict:
    """Image-level Spearman rho vs storage day; days permuted / resampled as whole dates (one date per day)."""
    days = np.array(sorted(img.storage_day.unique()))
    groups = {d: img.loc[img.storage_day == d, col].to_numpy() for d in days}
    rho = stats.spearmanr(img.storage_day, img[col]).statistic
    cnt = 0
    for _ in range(N_PERM):
        perm = rng.permutation(days)
        x = np.concatenate([np.full(len(groups[d]), p) for d, p in zip(days, perm)])
        y = np.concatenate([groups[d] for d in days])
        cnt += abs(stats.spearmanr(x, y).statistic) >= abs(rho) - 1e-12
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(days, len(days), replace=True)
        x = np.concatenate([np.full(len(groups[d]), d) for d in pick])
        y = np.concatenate([groups[d] for d in pick])
        if len(np.unique(x)) > 1:
            boots.append(stats.spearmanr(x, y).statistic)
    lodo = [stats.spearmanr(img.loc[img.storage_day != d, "storage_day"], img.loc[img.storage_day != d, col]).statistic
            for d in days]
    early = img.loc[img.storage_day.isin(EARLY), col].median()
    late = img.loc[img.storage_day.isin(LATE), col].median()
    return {"rho": float(rho), "p_naive": float(stats.spearmanr(img.storage_day, img[col]).pvalue),
            "p_perm_dates": float((cnt + 1) / (N_PERM + 1)),
            "boot95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "lodo_range": [float(min(lodo)), float(max(lodo))],
            "early_late_pct": float(100 * (late - early) / early) if col != "diff" else float(late - early)}


def composition(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["storage_day", "class_name"]).size().unstack(fill_value=0)
    t = t.reindex(columns=CLASSES, fill_value=0)
    return 100 * t.div(t.sum(axis=1), axis=0)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    res: dict = {}

    # 1 ---------------------------------------------------------------- design
    rf = runs_frame()
    lora = rf[rf.loss != "full_finetune"]
    res["design"] = {"lora_runs": int(len(lora)), "full_ft_runs": int((rf.loss == "full_finetune").sum()),
                     "archs": int(lora.arch.nunique()), "ranks": int(lora["rank"].nunique()),
                     "losses": int(lora.loss.nunique()), "seeds": int(lora.seed.nunique())}

    # 2/3 -------------------------------------------------------------- ablation over the rank sweep
    key = lora.set_index(["arch", "rank", "loss", "seed"])
    abl = {}
    for arch in ARCHS:
        a = {"full_lt_dice_seedpairs": 0, "full_lt_pmcbga_seedpairs": 0, "n_pairs": 0,
             "full_lt_dice_configmeans": 0, "full_lt_pmcbga_configmeans": 0,
             "ci_excl0_full_vs_dice": 0, "ci_excl0_full_vs_pmcbga": 0,
             "aji_full_gt_pmcbga_seedpairs": 0, "dice_full_gt_dice_seedpairs": 0,
             "lost_runs": {l: 0 for l in LOSSES}, "per_rank": {}}
        for r in RANKS:
            ipe = {l: {s: key.loc[(arch, r, l, s), "ipe_global_pct"] for s in SEEDS} for l in LOSSES}
            for s in SEEDS:
                a["n_pairs"] += 1
                a["full_lt_dice_seedpairs"] += ipe["full"][s] < ipe["dice_only"][s]
                a["full_lt_pmcbga_seedpairs"] += ipe["full"][s] < ipe["pmc_bga"][s]
                a["aji_full_gt_pmcbga_seedpairs"] += key.loc[(arch, r, "full", s), "aji"] > key.loc[(arch, r, "pmc_bga", s), "aji"]
                a["dice_full_gt_dice_seedpairs"] += key.loc[(arch, r, "full", s), "mean_dice"] > key.loc[(arch, r, "dice_only", s), "mean_dice"]
                for l in LOSSES:
                    a["lost_runs"][l] += int(key.loc[(arch, r, l, s), "n_lost"] > 0)
            d1 = paired_diff(ipe["dice_only"], ipe["full"])
            d2 = paired_diff(ipe["pmc_bga"], ipe["full"])
            a["full_lt_dice_configmeans"] += d1["mean"] < 0
            a["full_lt_pmcbga_configmeans"] += d2["mean"] < 0
            a["ci_excl0_full_vs_dice"] += bool(d1["excludes_zero"])
            a["ci_excl0_full_vs_pmcbga"] += bool(d2["excludes_zero"])
            a["per_rank"][r] = {"full_minus_dice": d1, "full_minus_pmcbga": d2}
        for k in list(a):
            if isinstance(a[k], (np.integer, np.bool_)):
                a[k] = int(a[k])
        abl[arch] = a
    res["ablation"] = abl

    # 4/5 -------------------------------------------------------------- storage (date clustered)
    gt = cells_gt()
    st = {"annotation": {c: date_clustered(per_image(gt, c), c, rng) for c in ("area_um2", "dry_mass_pg")}}
    gimg = per_image(gt, "dry_mass_pg").rename(columns={"dry_mass_pg": "ann"})
    st["annotation"]["per_image_median_dry_mass"] = {"mean": float(gimg.ann.mean()), "sd": float(gimg.ann.std(ddof=1)),
                                                     "min": float(gimg.ann.min()), "max": float(gimg.ann.max())}
    for arch, rank in (("edge_sam", 32), ("mobile_sam", 8), ("edge_sam", 8)):
        for s in SEEDS:
            p = cells_pred(arch, rank, "full", s)
            k = f"{arch}_r{rank}_s{s}"
            st[k] = {c: date_clustered(per_image(p, c), c, rng) for c in ("area_um2", "dry_mass_pg")}
            j = per_image(p, "dry_mass_pg").merge(gimg, on=["stem", "storage_day"])
            j["diff"] = j.dry_mass_pg - j.ann
            j["rel"] = 100 * j["diff"] / j.ann
            st[k]["bias_diff_pg"] = date_clustered(j, "diff", rng)
            st[k]["bias_rel_pct"] = date_clustered(j[["stem", "storage_day", "rel"]].rename(columns={"rel": "diff"}), "diff", rng)
            st[k]["bias_by_period_pg"] = {"early": float(j.loc[j.storage_day.isin(EARLY), "diff"].median()),
                                          "late": float(j.loc[j.storage_day.isin(LATE), "diff"].median()),
                                          "all_mean": float(j["diff"].mean())}
    res["storage"] = st

    # 6 ---------------------------------------------------------------- composition and component counts
    cg = composition(gt)
    ngt = gt.groupby("storage_day").size()
    comp = {}
    for arch, rank in (("edge_sam", 32), ("mobile_sam", 8), ("edge_sam", 8)):
        per = [composition(cells_pred(arch, rank, "full", s)) for s in SEEDS]
        mean = sum(per) / 3
        diff = (mean - cg).abs()
        idx = np.unravel_index(np.argmax(diff.to_numpy()), diff.shape)
        ratios = pd.concat([cells_pred(arch, rank, "full", s).groupby("storage_day").size() / ngt for s in SEEDS], axis=1)
        counts = pd.concat([cells_pred(arch, rank, "full", s).groupby("storage_day").size() for s in SEEDS], axis=1)
        comp[f"{arch}_r{rank}"] = {
            "seedmean_vs_ann_mean_abs_pp": float(diff.to_numpy().mean()),
            "seedmean_vs_ann_max_abs_pp": float(diff.to_numpy().max()),
            "max_at": [int(diff.index[idx[0]]), diff.columns[idx[1]]],
            "per_seed_max_abs_pp": [float((x - cg).abs().to_numpy().max()) for x in per],
            "per_seed_mean_abs_pp": [float((x - cg).abs().to_numpy().mean()) for x in per],
            "pred_share_seedmean": {int(d): {c: round(float(mean.loc[d, c]), 1) for c in CLASSES} for d in mean.index},
            "pred_share_seedrange": {int(d): {c: [round(float(min(x.loc[d, c] for x in per)), 1),
                                                  round(float(max(x.loc[d, c] for x in per)), 1)] for c in CLASSES}
                                     for d in mean.index},
            "kept_ratio_seedmean": {int(d): round(float(v), 3) for d, v in ratios.mean(axis=1).items()},
            "kept_counts": {int(d): [int(x) for x in row] for d, row in counts.iterrows()},
            "n_kept_total_nowrap": [int(len(cells_pred(arch, rank, "full", s))) for s in SEEDS],
            "n_kept_total_all": [int(len(cells_pred(arch, rank, "full", s, nowrap=False))) for s in SEEDS]}
    res["composition"] = {"annotation_share": {int(d): {c: round(float(cg.loc[d, c]), 1) for c in CLASSES} for d in cg.index},
                          "annotation_cells_per_day": {int(d): int(v) for d, v in ngt.items()},
                          "annotation_stomatocyte_cells_per_day": {int(d): int(v) for d, v in
                                                                   gt[gt.class_name == "stomatocyte"].groupby("storage_day").size().items()},
                          **comp}

    # 7 ---------------------------------------------------------------- agreement, per-cell IPE, strata, robustness
    agr = {}
    for arch, rank in (("edge_sam", 32), ("mobile_sam", 8), ("edge_sam", 8)):
        rows = {s: cell_unit(arch, rank, "full", s)["dry_mass"]["agreement_nowrap"] for s in SEEDS}
        agr[f"{arch}_r{rank}"] = {m: {"per_seed": [rows[s][m] for s in SEEDS], "mean": float(np.mean([rows[s][m] for s in SEEDS])),
                                      "sd": float(np.std([rows[s][m] for s in SEEDS], ddof=1))}
                                  for m in ("mdare_pct", "pearson_r", "lin_ccc", "bland_altman_mean_pg",
                                            "bland_altman_loa_low_pg", "bland_altman_loa_high_pg")}
        st_rows = {s: cell_unit(arch, rank, "full", s)["storage"]["pred_nowrap"] for s in SEEDS}
        agr[f"{arch}_r{rank}"]["early_late_pct"] = {
            q: {"per_seed": [st_rows[s][q]["late_vs_early_pct"] for s in SEEDS],
                "mean": float(np.mean([st_rows[s][q]["late_vs_early_pct"] for s in SEEDS])),
                "sd": float(np.std([st_rows[s][q]["late_vs_early_pct"] for s in SEEDS], ddof=1))} for q in ("area", "dry_mass")}
    res["agreement"] = agr

    pc = {}
    for arch, rank in (("edge_sam", 8), ("edge_sam", 32), ("mobile_sam", 8), ("mobilenet_unet", 8)):
        for loss in ("full", "pmc_bga"):
            sm = [cell_unit(arch, rank, loss, s)["phase_integral"]["summary"] for s in SEEDS]
            q1 = [next(x["error_rate_pct"] for x in cell_unit(arch, rank, loss, s)["phase_integral"]["strata"]
                       if x["stratum"].endswith("Q1")) for s in SEEDS]
            q24 = [max(x["error_rate_pct"] for x in cell_unit(arch, rank, loss, s)["phase_integral"]["strata"]
                       if x["stratum"][-2:] in ("Q2", "Q3", "Q4")) for s in SEEDS]
            f = lambda k: [float(np.mean([x[k] for x in sm])), float(np.std([x[k] for x in sm], ddof=1))]
            pc[f"{arch}_r{rank}_{loss}"] = {"matched_pct": f("matched_cells_pct"), "percell_med": f("percell_abs_median_pct"),
                                            "percell_p90": f("percell_abs_p90_pct"),
                                            "percell_med_nowrap": f("percell_abs_median_pct_nowrap"),
                                            "percell_p90_nowrap": f("percell_abs_p90_pct_nowrap"),
                                            "fnr_q1": [float(np.mean(q1)), float(np.std(q1, ddof=1)), q1],
                                            "fnr_q2q4_max": float(max(q24))}
    res["per_cell_and_strata"] = pc

    rob = compiled("robustness")["configurations"]
    out = {}
    for arch, rank in (("edge_sam", 8), ("edge_sam", 32), ("mobile_sam", 8), ("mobilenet_unet", 8)):
        k = config_key(arch, rank, "full")
        dist = rob[k]["distortions"]
        out[k] = {"max_abs_delta_dice": max(abs(v["delta_macro_dice_vs_none"][x]) for v in dist.values()
                                            for x in ("min", "max")),
                  "ipe_uncorr_none": dist["none"]["ipe_uncorrected_seen_pct"]["mean"],
                  "ipe_uncorr_by_dist": {d: round(v["ipe_uncorrected_seen_pct"]["mean"], 2) for d, v in dist.items()},
                  "dice_by_dist": {d: round(v["macro_dice"]["mean"], 4) for d, v in dist.items()}}
    res["robustness"] = out

    path = ART / "audit" / "claims_check.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("wrote", path)


if __name__ == "__main__":
    main()
