"""Every number written into the revised manuscript and SI, computed from outputs/ (read only), with its source.

facts() returns {key: Fact(text, source)}. The document builder (build_final_docs.py) fills the paragraph templates with these
keys and writes artifacts/manuscript_final/change_log.md listing every key used, its value and its source, so that each
number can be traced and re-checked.

Date-clustered statistics come from artifacts/audit/claims_check.json (artifacts/code/audit_claims.py; 20,000 date-block
permutations and 20,000 date-bootstrap resamples, RNG seed 20261029). Monte-Carlo p-values can differ in the third decimal
from another run with a different RNG seed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

import numpy as np

from common import (ART, CLASSES, COLLAPSE_DICE, LOSSES, OUT, RANKS, SEEDS, cell_unit, cells_gt, cfg_entry, compiled,
                    config_key, paired_diff, runs_frame)

CFG = {"m8": ("mobile_sam", 8), "m32": ("mobile_sam", 32), "e8": ("edge_sam", 8), "e32": ("edge_sam", 32),
       "u8": ("mobilenet_unet", 8)}
ARCH = {"m": "mobile_sam", "e": "edge_sam", "u": "mobilenet_unet"}
LOSS_KEY = {"dice_only": "D", "pmc": "P", "pmc_bga": "B", "full": "F"}
CC = "outputs/compiled/cell_analysis_multiseed.json"
BM = "outputs/compiled/benchmark_multiseed.json"
RB = "outputs/compiled/robustness_multiseed.json"
AUD = "artifacts/audit/claims_check.json"


@dataclass
class Fact:
    text: str
    source: str


def pm(m, s, nd):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def sgn(x, nd):
    return f"{x:+.{nd}f}".replace("-", "−")


def ci(d, nd):
    return f"{sgn(d['mean'], nd)} (95% interval {sgn(d['lo'], nd)} to {sgn(d['hi'], nd)})"


def pval(p):
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def num(x, nd=2):
    """Interval end points: three decimals when the value would otherwise print as 0.00."""
    return (f"{x:.3f}" if abs(x) < 0.01 else f"{x:.{nd}f}").replace("-", "−")


def facts() -> Dict[str, Fact]:
    F: Dict[str, Fact] = {}
    rf = runs_frame()
    lora = rf[rf.loss != "full_finetune"]
    aud = json.loads((ART / "audit" / "claims_check.json").read_text())

    def put(k, v, src):
        F[k] = Fact(str(v), src)

    def runs(a, r, loss="full"):
        return lora[(lora.arch == a) & (lora["rank"] == r) & (lora.loss == loss)].sort_values("seed")

    # ------------------------------------------------------------ design
    put("n_lora_runs", len(lora), f"{CC}: configurations x seeds_found, loss != full_finetune")
    put("n_ft_runs", int((rf.loss == "full_finetune").sum()), f"{CC}: *_full_finetune configurations")

    # ------------------------------------------------------------ segmentation, full objective
    for k, (a, r) in CFG.items():
        d = runs(a, r)
        src = f"{CC}: configurations['{config_key(a, r, 'full')}'].metrics.<metric> (mean, sd over seeds 42-44)"
        for m, nd in (("mean_dice", 3), ("aji", 3), ("bf1", 3), ("mean_iou", 3)):
            put(f"{k}_{m.replace('mean_', '')}", pm(d[m].mean(), d[m].std(ddof=1), nd), src)
        put(f"{k}_ipe", pm(d.ipe_global_pct.mean(), d.ipe_global_pct.std(ddof=1), 2), src.replace("<metric>", "ipe_global_pct"))
        put(f"{k}_ipe_sd", f"{d.ipe_global_pct.std(ddof=1):.2f}", src.replace("<metric>", "ipe_global_pct"))
        for cl in CLASSES:
            put(f"{k}_dice_{cl[:4]}", f"{d[f'dice_{cl}'].mean():.3f}", src.replace("<metric>", f"dice_{cl}"))
        p = int(d.trainable_params.iloc[0])
        put(f"{k}_params", f"{p / 1e6:.2f} M", f"outputs/*/runs/<run>/metrics.json: parameters.trainable ({p:,})")
        put(f"{k}_params_pct", f"{100 * p / d.total_params.iloc[0]:.1f}%", "metrics.json: parameters.trainable / total")
        lost = int((d.n_lost > 0).sum())
        put(f"{k}_lost", f"{lost}/3", f"{CC}: per-seed class Dice < {COLLAPSE_DICE}")
    e8 = runs("edge_sam", 8)
    put("e8_lost_seed", int(e8[e8.n_lost > 0].seed.iloc[0]), f"{CC}: edge_sam|lora_r8 per-seed dice_spherocyte")
    put("e8_lost_seed_dice", f"{e8[e8.n_lost > 0].mean_dice.iloc[0]:.3f}", f"{CC}: edge_sam|lora_r8 mean_dice per_seed")
    m32 = runs("mobile_sam", 32)
    put("m32_lost_seed", int(m32[m32.n_lost > 0].seed.iloc[0]), f"{CC}: mobile_sam|lora_r32 per-seed class Dice")
    for k, m in (("dice", "mean_dice"), ("aji", "aji")):
        a = dict(zip(runs("edge_sam", 8).seed, runs("edge_sam", 8)[m]))
        b = dict(zip(runs("edge_sam", 32).seed, runs("edge_sam", 32)[m]))
        put(f"e32_minus_e8_{k}", ci(paired_diff(a, b), 3), f"{CC}: seed-paired difference, t-interval n=3 (common.paired_diff)")

    # ------------------------------------------------------------ full fine-tuning (seed 42) and LoRA seed 42
    for a, ab in (("edge_sam", "e"), ("mobile_sam", "m"), ("mobilenet_unet", "u")):
        ft = rf[(rf.arch == a) & (rf.loss == "full_finetune")].iloc[0]
        l42 = runs(a, 8).set_index("seed").loc[42]
        for tag, row in (("ft", ft), ("l42", l42)):
            src = f"{CC}: configurations['{a}|full_finetune' if tag == 'ft' else '{a}|lora_r8'].metrics.*.per_seed['42']"
            put(f"{tag}_{ab}_dice", f"{row.mean_dice:.3f}", src)
            put(f"{tag}_{ab}_aji", f"{row.aji:.3f}", src)
            put(f"{tag}_{ab}_bf1", f"{row.bf1:.3f}", src)
            put(f"{tag}_{ab}_ipe", f"{row.ipe_global_pct:.2f}", src)
            put(f"{tag}_{ab}_params", f"{int(row.trainable_params) / 1e6:.2f} M", "metrics.json: parameters.trainable")

    # ------------------------------------------------------------ loss ablation over the rank sweep
    for ab, a in ARCH.items():
        A = aud["ablation"][a]
        put(f"abl_{ab}_seed_FltD", A["full_lt_dice_seedpairs"], f"{AUD}: ablation.{a}.full_lt_dice_seedpairs")
        put(f"abl_{ab}_seed_FltB", A["full_lt_pmcbga_seedpairs"], f"{AUD}: ablation.{a}.full_lt_pmcbga_seedpairs")
        put(f"abl_{ab}_ajiFgtB", A["aji_full_gt_pmcbga_seedpairs"], f"{AUD}: ablation.{a}.aji_full_gt_pmcbga_seedpairs")
        put(f"abl_{ab}_diceFgtD", A["dice_full_gt_dice_seedpairs"], f"{AUD}: ablation.{a}.dice_full_gt_dice_seedpairs")
        for loss in LOSSES:
            put(f"abl_{ab}_lost_{LOSS_KEY[loss]}", A["lost_runs"][loss], f"{AUD}: ablation.{a}.lost_runs.{loss} (of 15 runs)")
    tot = lambda key: sum(aud["ablation"][a][key] for a in ARCH.values())
    put("abl_seed_FltD", tot("full_lt_dice_seedpairs"), f"{AUD}: sum over architectures")
    put("abl_seed_FltB", tot("full_lt_pmcbga_seedpairs"), f"{AUD}: sum over architectures")
    put("abl_cfg_FltD", tot("full_lt_dice_configmeans"), f"{AUD}: ablation.*.full_lt_dice_configmeans (of 15)")
    put("abl_cfg_FltB", tot("full_lt_pmcbga_configmeans"), f"{AUD}: ablation.*.full_lt_pmcbga_configmeans (of 15)")
    put("abl_ci_FltD", tot("ci_excl0_full_vs_dice"), f"{AUD}: ablation.*.ci_excl0_full_vs_dice (of 15)")
    put("abl_ci_FltB", tot("ci_excl0_full_vs_pmcbga"), f"{AUD}: ablation.*.ci_excl0_full_vs_pmcbga (of 15)")
    for k in ("e32", "e8", "m8", "u8"):
        a, r = CFG[k]
        for loss in LOSSES:
            d = runs(a, r, loss)
            L = LOSS_KEY[loss]
            src = f"{CC}: configurations['{config_key(a, r, loss)}'].metrics"
            put(f"{k}{L}_dice", pm(d.mean_dice.mean(), d.mean_dice.std(ddof=1), 3), src)
            put(f"{k}{L}_bf1", pm(d.bf1.mean(), d.bf1.std(ddof=1), 3), src)
            put(f"{k}{L}_aji", f"{d.aji.mean():.3f}", src)
            put(f"{k}{L}_ipe", pm(d.ipe_global_pct.mean(), d.ipe_global_pct.std(ddof=1), 2), src)
            put(f"{k}{L}_sph", f"{d.dice_spherocyte.mean():.3f}", src + ".dice_spherocyte")
            lost = d[d.n_lost > 0].seed.tolist()
            put(f"{k}{L}_lostseeds", ", ".join(str(s) for s in lost) if lost else "none", src + " per-seed class Dice")
        ipe = {L: dict(zip(runs(a, r, l).seed, runs(a, r, l).ipe_global_pct)) for l, L in LOSS_KEY.items()}
        dice = {L: dict(zip(runs(a, r, l).seed, runs(a, r, l).mean_dice)) for l, L in LOSS_KEY.items()}
        src = f"{CC}: seed-paired differences, t-interval n=3 (common.paired_diff)"
        put(f"{k}_FmD_ipe", ci(paired_diff(ipe["D"], ipe["F"]), 2), src)
        put(f"{k}_FmB_ipe", ci(paired_diff(ipe["B"], ipe["F"]), 2), src)
        put(f"{k}_FmB_dice", ci(paired_diff(dice["B"], dice["F"]), 3), src)
        put(f"{k}_FmD_dice", ci(paired_diff(dice["D"], dice["F"]), 3), src)

    # ------------------------------------------------------------ per-cell IPE and low-phase strata
    pc = aud["per_cell_and_strata"]
    for k in ("e32", "m8", "e8", "u8"):
        a, r = CFG[k]
        for loss, L in (("full", "F"), ("pmc_bga", "B")):
            v = pc[f"{a}_r{r}_{loss}"]
            src = f"outputs/analysis/cell/<run>.json: phase_integral.summary / .strata (via {AUD} per_cell_and_strata)"
            put(f"{k}{L}_matched", pm(*v["matched_pct"], 1), src + " matched_cells_pct")
            put(f"{k}{L}_pc_med", pm(*v["percell_med_nowrap"], 2), src + " percell_abs_median_pct_nowrap")
            put(f"{k}{L}_pc_p90", pm(*v["percell_p90_nowrap"], 1), src + " percell_abs_p90_pct_nowrap")
            put(f"{k}{L}_fnrq1", pm(v["fnr_q1"][0], v["fnr_q1"][1], 1), src + " 'false negatives in phase quartile Q1'")
            put(f"{k}{L}_fnrq24max", f"{v['fnr_q2q4_max']:.1f}", src + " max over seeds of Q2-Q4 rates")
    put("fnr_e_q24max", f"{max(pc[f'edge_sam_r{r}_{l}']['fnr_q2q4_max'] for r in (8, 32) for l in ('full', 'pmc_bga')):.1f}",
        f"{AUD}: per_cell_and_strata, EdgeSAM r8/r32, full and pmc_bga, max over seeds of Q2-Q4")
    put("fnr_m_q24max", f"{max(pc[f'mobile_sam_r8_{l}']['fnr_q2q4_max'] for l in ('full', 'pmc_bga')):.1f}",
        f"{AUD}: per_cell_and_strata, MobileSAM r8")

    # ------------------------------------------------------------ robustness
    rob = aud["robustness"]
    put("rob_max_ddice", f"{max(v['max_abs_delta_dice'] for v in rob.values()):.4f}",
        f"{RB}: max over the four Table-1 configurations and all distortions of |delta_macro_dice_vs_none| (per seed)")
    put("rob_max_ddice_mean", f"{max(abs(d - v['dice_by_dist']['none']) for v in rob.values() for d in v['dice_by_dist'].values()):.4f}",
        f"{RB}: max over the four Table-1 configurations and all distortions of |mean macro Dice - mean macro Dice without distortion|")
    for k in ("e32", "e8"):
        a, r = CFG[k]
        v = rob[config_key(a, r, "full")]
        put(f"{k}_rob_ipe0", f"{v['ipe_uncorr_none']:.2f}", f"{RB}: {config_key(a, r, 'full')}.distortions.none.ipe_uncorrected_seen_pct")
        put(f"{k}_rob_ipe06", f"{v['ipe_uncorr_by_dist']['offset+0.60']:.2f}",
            f"{RB}: {config_key(a, r, 'full')}.distortions['offset+0.60'].ipe_uncorrected_seen_pct")

    # ------------------------------------------------------------ benchmark (rank 8, merged)
    rows = {(r["architecture"], r["backend"]): r for r in compiled("benchmark")["rows"]}
    for ab, a in ARCH.items():
        for b in ("pytorch", "onnx"):
            r = rows[(a, b)]
            src = f"{BM}: rows[{a}, {b}]"
            put(f"bm_{ab}_{b}_p50", pm(r["latency_p50_ms"]["mean"], r["latency_p50_ms"]["sd"], 2), src + ".latency_p50_ms")
            put(f"bm_{ab}_{b}_p99", pm(r["latency_p99_ms"]["mean"], r["latency_p99_ms"]["sd"], 2), src + ".latency_p99_ms")
            put(f"bm_{ab}_{b}_fps", f"{r['fps']['min']:.1f}–{r['fps']['max']:.1f}", src + ".fps min-max over passes")
            put(f"bm_{ab}_{b}_mem", f"{r['peak_vram_mb_max']:.0f}", src + ".peak_vram_mb_max")
    for e in compiled("benchmark")["equivalence"]["architectures"]:
        put(f"bm_epoch_{e['architecture']}", e["checkpoint_epoch"], f"{BM}: equivalence.architectures[{e['architecture']}].checkpoint_epoch")

    # ------------------------------------------------------------ storage, agreement, composition
    st = aud["storage"]
    ann = st["annotation"]
    put("ann_cells", len(cells_gt()), "outputs/cell_analysis/annotation/cells_gt.csv (wrapped == False)")
    put("ann_cells_all", len(cells_gt(nowrap=False)), "outputs/cell_analysis/annotation/cells_gt.csv")
    put("ann_wrapped", len(cells_gt(nowrap=False)) - len(cells_gt()), "cells_gt.csv: wrapped == True")
    for q, Q in (("area_um2", "area"), ("dry_mass_pg", "dm")):
        v = ann[q]
        put(f"ann_{Q}_chg", f"{v['early_late_pct']:.1f}".replace("-", "−"), f"{AUD}: storage.annotation.{q}.early_late_pct")
        put(f"ann_{Q}_p", pval(v["p_perm_dates"]), f"{AUD}: storage.annotation.{q}.p_perm_dates")
        put(f"ann_{Q}_boot", f"{num(v['boot95'][0])} to {num(v['boot95'][1])}", f"{AUD}: storage.annotation.{q}.boot95")
        put(f"ann_{Q}_rho", f"{v['rho']:.2f}".replace("-", "−"), f"{AUD}: storage.annotation.{q}.rho")
    im = ann["per_image_median_dry_mass"]
    put("ann_img_dm", f"{im['mean']:.2f} ± {im['sd']:.2f}", f"{AUD}: storage.annotation.per_image_median_dry_mass")
    put("ann_img_dm_range", f"{im['min']:.1f} to {im['max']:.1f}", f"{AUD}: storage.annotation.per_image_median_dry_mass")
    agr = aud["agreement"]
    for k in ("e32", "m8"):
        a, r = CFG[k]
        tag = f"{a}_r{r}"
        g = agr[tag]
        src = f"outputs/analysis/cell/<run>.json: dry_mass.agreement_nowrap (via {AUD} agreement.{tag})"
        put(f"{k}_mdare", pm(g["mdare_pct"]["mean"], g["mdare_pct"]["sd"], 1), src)
        put(f"{k}_r", pm(g["pearson_r"]["mean"], g["pearson_r"]["sd"], 2), src)
        put(f"{k}_ccc", pm(g["lin_ccc"]["mean"], g["lin_ccc"]["sd"], 2), src)
        put(f"{k}_bias", pm(g["bland_altman_mean_pg"]["mean"], g["bland_altman_mean_pg"]["sd"], 2), src)
        put(f"{k}_loa", f"{g['bland_altman_loa_low_pg']['mean']:.1f} to +{g['bland_altman_loa_high_pg']['mean']:.1f}".replace("-", "−"), src)
        for m, nd in (("mdare_pct", 1), ("pearson_r", 2), ("lin_ccc", 2), ("bland_altman_mean_pg", 2)):
            put(f"{k}_{m}_seeds", ", ".join(f"{x:.{nd}f}" for x in g[m]["per_seed"]).replace("-", "−"), src)
        for q, Q in (("area", "area"), ("dry_mass", "dm")):
            el = g["early_late_pct"][q]
            put(f"{k}_{Q}_chg", pm(el["mean"], el["sd"], 1).replace("-", "−"),
                f"outputs/analysis/cell/<run>.json: storage.pred_nowrap.{q}.late_vs_early_pct (mean, sd over seeds)")
        for q, Q in (("area_um2", "area"), ("dry_mass_pg", "dm"), ("bias_diff_pg", "bias")):
            ps = [st[f"{tag}_s{s}"][q]["p_perm_dates"] for s in SEEDS]
            bs = [st[f"{tag}_s{s}"][q]["boot95"] for s in SEEDS]
            rh = [st[f"{tag}_s{s}"][q]["rho"] for s in SEEDS]
            put(f"{k}_{Q}_p", ", ".join(pval(p) for p in ps), f"{AUD}: storage.{tag}_s<seed>.{q}.p_perm_dates (seeds 42, 43, 44)")
            put(f"{k}_{Q}_boot", "; ".join(f"{num(b[0])} to {num(b[1])}" for b in bs), f"{AUD}: storage.{tag}_s<seed>.{q}.boot95")
            put(f"{k}_{Q}_rho", ", ".join(f"{x:.2f}" for x in rh).replace("-", "−"), f"{AUD}: storage.{tag}_s<seed>.{q}.rho")
            put(f"{k}_{Q}_pnaive", ", ".join(pval(st[f"{tag}_s{s}"][q]["p_naive"]) for s in SEEDS),
                f"{AUD}: storage.{tag}_s<seed>.{q}.p_naive (image-level Spearman, images treated as independent)")
            put(f"{k}_{Q}_pmin", pval(min(ps)), f"{AUD}: min over seeds")
            put(f"{k}_{Q}_pmax", pval(max(ps)), f"{AUD}: max over seeds")
            put(f"{k}_{Q}_boot_excl0", sum(1 for b in bs if b[0] > 0 or b[1] < 0), f"{AUD}: count of seeds whose boot95 excludes 0")
        put(f"{k}_bias_early_late",
            ", ".join(f"{st[f'{tag}_s{s}']['bias_by_period_pg']['early']:+.2f}/{st[f'{tag}_s{s}']['bias_by_period_pg']['late']:+.2f}" for s in SEEDS).replace("-", "−"),
            f"{AUD}: storage.{tag}_s<seed>.bias_by_period_pg (early/late medians of per-image prediction minus annotation, pg)")
    comp = aud["composition"]
    a_sh = comp["annotation_share"]
    e_sh = comp["edge_sam_r32"]
    src = f"{AUD}: composition (cells_gt.csv / cells_pred.csv, wrapped cells excluded, six images per day pooled)"
    for d, cl in (("0", "discocyte"), ("47", "discocyte"), ("47", "spherocyte"), ("47", "echinocyte"), ("0", "stomatocyte"),
                  ("19", "stomatocyte"), ("47", "stomatocyte")):
        put(f"sh_ann_{cl[:4]}_{d}", f"{a_sh[d][cl]:.1f}%", src + f" annotation_share[{d}][{cl}]")
        put(f"sh_e32_{cl[:4]}_{d}", f"{e_sh['pred_share_seedmean'][d][cl]:.1f}%", src + f" edge_sam_r32.pred_share_seedmean[{d}][{cl}]")
    rng_ = e_sh["pred_share_seedrange"]["19"]["stomatocyte"]
    put("sh_e32_stom_19_range", f"{rng_[0]:.1f}–{rng_[1]:.1f}%", src + " edge_sam_r32.pred_share_seedrange[19][stomatocyte]")
    put("ann_stom_19_n", comp["annotation_stomatocyte_cells_per_day"]["19"], src + " annotation_stomatocyte_cells_per_day[19]")
    put("ann_cells_19", comp["annotation_cells_per_day"]["19"], src + " annotation_cells_per_day[19]")
    put("ann_cells_day_range", f"{min(comp['annotation_cells_per_day'].values())} to {max(comp['annotation_cells_per_day'].values())}",
        src + " annotation_cells_per_day")
    e8sh = comp["edge_sam_r8"]["pred_share_seedmean"]
    put("sh_e8_stom", f"{e8sh['0']['stomatocyte']:.1f}%, {e8sh['19']['stomatocyte']:.1f}% and {e8sh['47']['stomatocyte']:.1f}%",
        src + " edge_sam_r8.pred_share_seedmean[0|19|47][stomatocyte]")
    for k, tag in (("e32", "edge_sam_r32"), ("m8", "mobile_sam_r8")):
        c = comp[tag]
        put(f"{k}_comp_mean", f"{c['seedmean_vs_ann_mean_abs_pp']:.1f}", src + f" {tag}.seedmean_vs_ann_mean_abs_pp")
        put(f"{k}_comp_max", f"{c['seedmean_vs_ann_max_abs_pp']:.1f}", src + f" {tag}.seedmean_vs_ann_max_abs_pp")
        put(f"{k}_comp_max_at", f"{c['max_at'][1]}s on day {c['max_at'][0]}", src + f" {tag}.max_at")
        put(f"{k}_comp_seedmax", f"{max(c['per_seed_max_abs_pp']):.1f}", src + f" {tag}.per_seed_max_abs_pp")
        kr = c["kept_ratio_seedmean"]
        put(f"{k}_kept_8_12_15", f"{kr['8']:.2f}, {kr['12']:.2f} and {kr['15']:.2f}", src + f" {tag}.kept_ratio_seedmean[8|12|15]")
        other = [v for d, v in kr.items() if d not in ("8", "12", "15")]
        put(f"{k}_kept_other", f"{min(other):.2f} to {max(other):.2f}", src + f" {tag}.kept_ratio_seedmean, other days")
        put(f"{k}_kept_tot", ", ".join(f"{x:,}" for x in c["n_kept_total_nowrap"]), src + f" {tag}.n_kept_total_nowrap")
        put(f"{k}_kept_tot_all", ", ".join(f"{x:,}" for x in c["n_kept_total_all"]), src + f" {tag}.n_kept_total_all")
        put(f"{k}_kept_15", " / ".join(str(x) for x in c["kept_counts"]["15"]), src + f" {tag}.kept_counts[15]")
    put("ann_cells_15", comp["annotation_cells_per_day"]["15"], src + " annotation_cells_per_day[15]")
    put("ann_cells_15_all", int((cells_gt(nowrap=False).storage_day == 15).sum()), "cells_gt.csv (all cells, day 15)")
    return F


if __name__ == "__main__":
    for k, f in facts().items():
        print(f"{k:28s} {f.text:45s} {f.source}")
