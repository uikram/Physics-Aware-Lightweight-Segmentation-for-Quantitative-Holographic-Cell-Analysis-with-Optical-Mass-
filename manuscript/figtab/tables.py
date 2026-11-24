"""Every manuscript and Supplementary table, built from the result files.

Each builder takes the Inputs (which logs what it reads) and returns a Table. Column headings of the tables that
keep the definition of the previous build (Tables 1, 3, 4, S1, S2, S10) are kept word for word, so verify.py can
compare them cell by cell with the previous build.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Dict, List

import numpy as np

from .core import (MissingInput, NEW_FULL, OLD_FULL, PROPOSED, Inputs, Runs, arch_dir, arch_disp, arm_long, arm_short, ci,
                   cond_label, finite, fmt_p, mean_sd, num, p_range, representative, require_rescored, run_dir)
from .writers import Table

W = "additional_analysis/results"
SEL = "results/consolidated/selection"


def _lost(rows) -> str:
    n = sum(1 for r in rows if (r.get("n_lost_classes") or 0) > 0)
    return f"{n} of {len(rows)}"


def _params(rows):
    r = rows[0] if rows else {}
    tot, tr = r.get("params_total"), r.get("params_trainable")
    pct = 100.0 * tr / tot if tot and tr is not None else None
    return (f"{tr:,}" if tr is not None else "–"), num(pct, 2)


def _within_label(R: Runs) -> str:
    return f"{R.within_pct:g} %" if R.within_pct is not None else "the tolerance"


# ================================================================================================ main text

def table1(inp: Inputs, R: Runs, arm: str = PROPOSED) -> Table:
    rep = representative(inp)
    cols = ["Architecture", "Low-rank adaptation", "Trainable parameters", "Trainable parameters (%)",
            "Cells within 5 % of the reference integrated phase (%)",
            "Cell integrated-phase error, equivalently the dry-mass relative error (%)",
            "Dry mass, bias (pg)", "Dry mass, concordance correlation", "Mean Dice",
            "Aggregated Jaccard index, cells", "Aggregated Jaccard index, connected regions", "Boundary F1",
            "Global integrated-phase error (%)", "Runs that lost a class"]
    rows = []
    for a in R.archs:
        c = rep[a]["selected_condition"]
        xs = R.get(a, c, arm)
        p, pp = _params(xs)
        rows.append([a, cond_label(c), p, pp,
                     mean_sd(R.col(a, c, arm, "a2_scored_within_5pct_of_all_cells_pct"), 2),
                     mean_sd(R.col(a, c, arm, "a2_scored_cell_ipe_abs_median_pct"), 3),
                     mean_sd(R.col(a, c, arm, "a2_scored_drymass_bias_pg"), 3),
                     mean_sd(R.col(a, c, arm, "a2_scored_drymass_ccc"), 3),
                     mean_sd(R.col(a, c, arm, "dice_macro"), 3),
                     mean_sd(R.col(a, c, arm, "a2_aji_cells"), 3), mean_sd(R.col(a, c, arm, "aji_cc"), 3),
                     mean_sd(R.col(a, c, arm, "bf1"), 3),
                     mean_sd(R.col(a, c, arm, "ipe_global", 100.0), 3), _lost(xs)])
    not_applied = [a for a in R.archs if str(rep[a].get("every_rank_lost_a_class")).lower() == "true"]
    notes = [f"Mean ± standard deviation over the {len(R.seeds)} seeds ({', '.join(map(str, R.seeds))}); "
             f"objective {arm_short(arm)}; last checkpoint; {xs[0]['n_images'] if xs else '–'} test images.",
             ("Representative rank: chosen on the 41 held-out validation images, so no test image took part in "
              "choosing it; the values here are the test results at that rank."
              if all(str(rep[a].get("rank_chosen_on", "")) == "validation" for a in R.archs) else
              "Representative rank: chosen on the test images, so the values reported at it are exploratory."),
             "The rule: of the ranks at which the fewest seeds lost a class (training stability), the "
             "one with the best balanced score, the mean of the share of cells within 5 % and the mean Dice "
             "(cell-level phase preservation and segmentation accuracy); the smaller rank on a tie. The adapters "
             "are merged after training, so the deployed network and its inference cost are the same at every "
             "rank and computational cost enters only through that tie-break.",
             "Cells within 5 %: share of all scored reference cells whose integrated phase the prediction "
             "reproduces to within 5 % (a missed cell counts as a failure). The cell integrated-phase error is the "
             "median over matched cells; because dry mass is a fixed multiple of the integrated phase it is also "
             "the median relative dry-mass error.",
             "Aggregated Jaccard index, cells: after watershed separation; connected regions: before separation.",
             f"A class is lost in a run when its Dice is below {R.lost_class_dice}."]
    if not_applied:
        notes.append(f"{', '.join(not_applied)}: no rank kept every class in every seed, so the stability step "
                     "kept the ranks at which the fewest seeds lost one. Its morphology classification is "
                     "reported as a failure case; the cell-level measures do not depend on the predicted class.")
    return Table("1", "representative_configurations",
                 "Representative configuration of each architecture under the proposed objective", cols, rows, notes)


def table2(inp: Inputs, R: Runs) -> Table:
    """Per-architecture paired comparison of the proposed objective with every other loss (W-2)."""
    w2 = inp.json(f"{W}/w2_per_architecture/w2_per_architecture_{R.checkpoint}.json")
    metrics = [("a2_scored_within_5pct_of_all_cells_pct", "Cells within 5 %", 1.0, 2),
               ("a2_scored_cell_ipe_abs_median_pct", "Cell integrated-phase error, matched cells", 1.0, 3),
               ("ipe_global", "Global integrated-phase error", 1.0, 3)]   # W-2 stores it in per cent already
    cols = ["Architecture", "Compared objective"]
    for _, lab, _, _ in metrics:
        cols += [f"{lab}: proposed / compared (%)", f"{lab}: median difference (points)",
                 f"{lab}: pairs in which the proposed objective is better", f"{lab}: p (Holm)"]
    rows = []
    npairs = set()
    for a in R.archs:
        res = w2["results"].get(arch_dir(a)) or w2["results"].get(a)
        if not res:
            continue
        for cmp_ in res["lora_ranks_pooled"]["comparisons"]:
            if cmp_["loss_a"] != PROPOSED:
                continue
            row = [a, arm_short(cmp_["loss_b"])]
            for key, _, k, pl in metrics:
                m = cmp_["metrics"][key]
                npairs.add(m["n_pairs"])
                row += [f"{num(m['mean_a'], pl, k)} / {num(m['mean_b'], pl, k)}",
                        num(m["median_difference_a_minus_b"], pl, k, sign=True),
                        f"{m['pairs_a_better']} of {m['n_pairs']}",
                        fmt_p(m.get("wilcoxon_p_holm_within_architecture", m.get("wilcoxon_exact_p")))]
            rows.append(row)
    conds = ", ".join(cond_label(c) for c in w2["results"][arch_dir(R.archs[0])]["lora_ranks_pooled"]["conditions"])
    notes = [f"Paired by rank and seed over every low-rank adaptation rank ({conds}) and every seed: "
             f"{' or '.join(map(str, sorted(npairs)))} pairs per comparison. Full fine-tuning is not pooled.",
             "Proposed / compared: mean over the pairs. Median difference: proposed minus compared; positive is "
             "better for cells within 5 %, negative is better for the two errors.",
             "p: exact two-sided Wilcoxon signed-rank test, Holm-corrected within each architecture "
             f"({w2['metadata'].get('holm_family', 'over the comparisons and quantities of that architecture')}).",
             "The cell-level statistics with cells as the unit (hierarchical bootstrap and day-level permutation) "
             "are in Fig. 4 and Table S11."]
    return Table("2", "objective_comparison_per_architecture",
                 "The proposed objective against every other loss configuration, per architecture", cols, rows, notes,
                 group_cols=1)


def table3(inp: Inputs, R: Runs, arm: str = PROPOSED) -> Table:
    rep = representative(inp)
    cols = ["Architecture", "Low-rank adaptation",
            "Cells within 5 %, all scored cells (%)", "Cells within 5 %, touching cells (%)",
            "Cells within 5 %, isolated cells (%)", "Cell integrated-phase error, all scored cells (%)",
            "Cell integrated-phase error, touching cells (%)", "Cell integrated-phase error, isolated cells (%)",
            "Global integrated-phase error (%)", "Cells matched (%)", "Cells missed (%)", "Cells merged (%)",
            "Cells fragmented (%)"]
    rows = []
    for a in R.archs:
        c = rep[a]["selected_condition"]
        g = lambda k, p, s=1.0: mean_sd(R.col(a, c, arm, k, s), p)  # noqa: E731
        rows.append([a, cond_label(c),
                     g("a2_scored_within_5pct_of_all_cells_pct", 2), g("a2_touching_within_5pct_of_all_cells_pct", 2),
                     g("a2_isolated_within_5pct_of_all_cells_pct", 2), g("a2_scored_cell_ipe_abs_median_pct", 3),
                     g("a2_touching_cell_ipe_abs_median_pct", 3), g("a2_isolated_cell_ipe_abs_median_pct", 3),
                     g("ipe_global", 3, 100.0), g("a2_scored_match_pct", 2), g("a2_scored_missed_pct", 2),
                     g("a2_scored_merged_pct", 2), g("a2_scored_fragmented_pct", 2)])
    xs = R.get(R.archs[0], rep[R.archs[0]]["selected_condition"], arm)
    n = {k: xs[0].get(k) for k in ("a2_scored_n", "a2_touching_n", "a2_isolated_n")} if xs else {}
    notes = [f"Mean ± standard deviation over the {len(R.seeds)} seeds; objective {arm_short(arm)}.",
             f"Scored reference cells: {n.get('a2_scored_n', '–')} ({n.get('a2_touching_n', '–')} touching another "
             f"reference cell, {n.get('a2_isolated_n', '–')} isolated). A touching cell shares a four-neighbour "
             "boundary with another reference cell.",
             "The global error is the per-image error of the whole field's integrated phase; it is reported beside "
             "the cell-level quantities so the two are not read as the same thing."]
    return Table("3", "cell_level_phase_preservation",
                 "Cell-level integrated-phase preservation, separately for touching and isolated cells", cols, rows,
                 notes)


def table4(inp: Inputs, R: Runs) -> Table:
    rows_in = inp.csv("results/consolidated/benchmark/benchmark.csv")
    bj = inp.json("results/consolidated/benchmark/benchmark.json") if inp.exists(
        "results/consolidated/benchmark/benchmark.json") else {}
    backend = {"pytorch": "PyTorch", "onnx": "ONNX Runtime"}
    # No rank column: the adapters are merged before export, so the deployed network and its cost are the same
    # at every rank. Naming a rank here would imply the cost belongs to that rank, and it does not.
    cols = ["Architecture", "Runtime", "Input (pixels)", "Median latency (ms)",
            "99th percentile latency (ms)", "Images per second", "Memory (MiB)"]
    order = {a: i for i, a in enumerate(R.archs)}
    rows_in = sorted(rows_in, key=lambda r: (order.get(arch_disp(r["architecture_display"]), 9),
                                             list(backend).index(r["backend"]) if r["backend"] in backend else 9))
    rows = []
    for r in rows_in:
        ms = lambda k, p: f"{float(r[k]):,.{p}f} ± {float(r[k + '_sd']):,.{p}f}"  # noqa: E731
        rows.append([arch_disp(r["architecture_display"]), backend.get(r["backend"], r["backend"]),
                     f"{r['input_size']} x {r['input_size']}",
                     ms("latency_p50_ms", 2), ms("latency_p99_ms", 2), ms("images_per_second", 1),
                     ms("memory_mib", 0)])
    npass = sorted({r["n_passes"] for r in rows_in})
    agree = finite(r.get("class_map_agreement_pct") for r in rows_in)
    prot = bj.get("protocol", {}) if isinstance(bj, dict) else {}
    notes = [f"Mean ± sample standard deviation over {', '.join(npass)} timing passes, each in its own process. "
             "Percentiles are taken within a pass; images per second is 1000 divided by that pass's median.",
             f"Batch size 1, {', '.join(sorted({r['precision'] for r in rows_in}))}, forward pass only, inputs "
             "already on the device. Memory: device memory after the timed loop minus a baseline taken before the "
             "model was loaded (mean ± SD over the passes; a high-water footprint, not a sampled peak).",
             "Low-rank adapters are merged into the base weights before export, which deletes them, so the "
             "deployed network, its parameter count and its forward-pass cost are identical at every rank. The "
             "cost reported here therefore belongs to the architecture and not to any one rank, and no rank is "
             "named. Trainable parameters, which do differ by rank, are in Table 1 and Table S10.",
             "The architectures run at different input sizes, so a latency comparison between them must say so."]
    if agree:
        notes.append(f"Before timing, each exported graph was run on the same inputs as the PyTorch network: the "
                     f"predicted class agreed on {min(agree):.4f} % of pixels or more.")
    if prot.get("checkpoint"):
        notes.append(f"Checkpoint: {prot['checkpoint']}; objective: {arm_short(prot.get('loss', PROPOSED))}.")
    return Table("4", "inference_cost", "Inference cost of the deployed networks", cols, rows, notes)


def table5(inp: Inputs, R: Runs, arm: str = PROPOSED) -> Table:
    rep = representative(inp)
    st = inp.json(f"results/consolidated/storage/runs_{R.checkpoint}.json")
    ann = inp.json("results/consolidated/storage/annotation.json")
    sidx = {(arch_disp(r["arch"]), r["condition"], r["arm"], int(r["seed"])): r for r in st["runs"]}
    cfg = inp.config().get("storage", {})
    e, l = cfg.get("early_max_day"), cfg.get("late_min_day")
    window = (f"early (days ≤ {e}) to late (days ≥ {l})" if e is not None and l is not None else "early to late")
    cols = ["Segmentation", "Low-rank adaptation",
            "Dry mass, median relative error, matched cells (%)", "Dry mass, bias, matched cells (pg)",
            "Dry mass, Pearson r, matched cells", "Dry mass, concordance correlation, matched cells",
            "Dry mass, Pearson r, per-image median of all predicted cells",
            f"Area: change of the median, {window} (%)", "Area: Spearman rho with day",
            "Area: p (date-block permutation)",
            f"Dry mass: change of the median, {window} (%)", "Dry mass: Spearman rho with day",
            "Dry mass: p (date-block permutation)",
            "Spearman rho of the prediction-minus-annotation bias with day: area / dry mass"]
    ta, tm = ann["trends"]["area"], ann["trends"]["dry_mass"]
    rows = [["Annotation (reference)", "–", "–", "–", "–", "–", "–",
             num(ta["early_to_late_change_pct"], 2), num(ta["rho"], 3), fmt_p(ta["p_block_permutation"]),
             num(tm["early_to_late_change_pct"], 2), num(tm["rho"], 3), fmt_p(tm["p_block_permutation"]), "–"]]
    stale = []
    for a in R.archs:
        c = rep[a]["selected_condition"]
        S = [sidx[(a, c, arm, s)] for s in R.seeds if (a, c, arm, s) in sidx]
        img_r = []
        for s in R.seeds:
            rel = f"{run_dir(a, c, arm, s)}/metrics_{R.checkpoint}.json"
            crel = f"{run_dir(a, c, arm, s)}/cells_{R.checkpoint}.json"
            if inp.exists(rel) and inp.exists(crel):
                try:
                    require_rescored(inp, crel)
                except MissingInput:
                    stale.append(f"{a} seed {s}"); continue
                m = inp.json(rel)
                img_r.append(m["instances"]["A2"]["dry_mass"]["per_image_median"]["pearson_r"])
        g = lambda k, p: mean_sd(R.col(a, c, arm, k), p)  # noqa: E731
        f = lambda q, k, p: mean_sd([x[q][k] for x in S], p)  # noqa: E731
        rows.append([a, cond_label(c), g("a2_scored_drymass_mdare_pct", 3), g("a2_scored_drymass_bias_pg", 3),
                     g("a2_scored_drymass_pearson_r", 3), g("a2_scored_drymass_ccc", 3),
                     mean_sd(img_r, 3) if img_r else "–",
                     f("area", "early_to_late_change_pct", 2), f("area", "rho", 3),
                     p_range([x["area"]["p_block_permutation"] for x in S]),
                     f("dry_mass", "early_to_late_change_pct", 2), f("dry_mass", "rho", 3),
                     p_range([x["dry_mass"]["p_block_permutation"] for x in S]),
                     f"{mean_sd([x['area']['bias_vs_annotation']['rho'] for x in S], 3)} / "
                     f"{mean_sd([x['dry_mass']['bias_vs_annotation']['rho'] for x in S], 3)}"])
    notes = [f"Model rows: mean ± standard deviation over the {len(R.seeds)} seeds; the p columns give the range "
             "over the seeds (p values are not averaged; per-seed values in Table S7). The annotation is one series.",
             "Matched cells: reference cells matched one to one (IoU ≥ 0.5) by a predicted cell. Per-image median "
             "of all predicted cells: every usable predicted cell, without matching, summarised per image "
             f"({ann.get('per_image') and len(ann['per_image']) or '–'} images).",
             "Change of the median: per-image medians pooled over the early days against those of the late days. "
             f"Date-block permutation: exact test over all orderings of the {ta.get('n_days', '–')} storage days, "
             "keeping the images of one day together.",
             "Association with storage time in a single sample series; no biological mechanism is inferred."]
    if stale:
        notes.append("Per-image median r not shown for runs whose per-run files predate the rescore: " + ", ".join(stale))
    return Table("5", "dry_mass_and_storage", "Dry-mass agreement and the change over storage time", cols, rows, notes)


# ================================================================================================ supplementary

def tableS1(inp: Inputs, R: Runs, arm: str = PROPOSED) -> Table:
    cols = ["Architecture", "Low-rank adaptation", "Trainable parameters", "Trainable parameters (%)",
            "Cells within 5 % (%)", "Cell integrated-phase error (%)", "Mean Dice",
            "Aggregated Jaccard index, cells", "Aggregated Jaccard index, connected regions", "Boundary F1",
            "Global integrated-phase error (%)", "Dry mass, median absolute relative error (%)",
            "Dry mass, concordance correlation", "Runs that lost a class"]
    rows = []
    for a in R.archs:
        for c in R.conditions:
            xs = R.get(a, c, arm)
            if not xs:
                continue
            p, pp = _params(xs)
            g = lambda k, q, s=1.0: mean_sd(R.col(a, c, arm, k, s), q)  # noqa: E731
            rows.append([a, cond_label(c), p, pp, g("a2_scored_within_5pct_of_all_cells_pct", 2),
                         g("a2_scored_cell_ipe_abs_median_pct", 3), g("dice_macro", 3), g("a2_aji_cells", 3),
                         g("aji_cc", 3), g("bf1", 3), g("ipe_global", 3, 100.0), g("a2_scored_drymass_mdare_pct", 3),
                         g("a2_scored_drymass_ccc", 3), _lost(xs)])
    return Table("S1", "rank_sweep", f"Every low-rank adaptation rank and full fine-tuning under the proposed "
                 f"objective ({arm_short(arm, False)})", cols, rows,
                 [f"Mean ± standard deviation over the {len(R.seeds)} seeds.",
                  "Learning rates differ between low-rank adaptation and full fine-tuning (Table S5)."])


def tableS2(inp: Inputs, R: Runs) -> Table:
    cols = ["Architecture", "Low-rank adaptation", "Loss configuration", "Cells within 5 % (%)",
            "Cell integrated-phase error (%)", "Mean Dice", "Aggregated Jaccard index, cells",
            "Aggregated Jaccard index, connected regions", "Boundary F1", "Global integrated-phase error (%)",
            "Runs that lost a class"]
    rows = []
    for a in R.archs:
        for c in R.conditions:
            for arm in R.arms:
                xs = R.get(a, c, arm)
                if not xs:
                    continue
                g = lambda k, q, s=1.0: mean_sd(R.col(a, c, arm, k, s), q)  # noqa: E731
                rows.append([a, cond_label(c), arm_short(arm), g("a2_scored_within_5pct_of_all_cells_pct", 2),
                             g("a2_scored_cell_ipe_abs_median_pct", 3), g("dice_macro", 3), g("a2_aji_cells", 3),
                             g("aji_cc", 3), g("bf1", 3), g("ipe_global", 3, 100.0), _lost(xs)])
    return Table("S2", "loss_and_rank_sweep", "Every loss configuration at every rank and under full fine-tuning",
                 cols, rows, [f"Mean ± standard deviation over the {len(R.seeds)} seeds.", _glossary()], group_cols=2)


def _glossary() -> str:
    return ("PMC: phase-mask contrast; BGA: boundary-gradient alignment; IPP: global integrated-phase preservation; "
            "cIPP: cell-level integrated-phase preservation (the proposed term).")


def tableS3(inp: Inputs, R: Runs) -> Table:
    """Pooled paired comparison of the proposed objective with every other loss (selection/arm_confirmation.csv)."""
    head = ["a2_scored_within_5pct_of_all_cells_pct", "a2_touching_within_5pct_of_all_cells_pct",
            "a2_isolated_within_5pct_of_all_cells_pct", "a2_scored_cell_ipe_abs_median_pct",
            "a2_scored_drymass_ccc", "ipe_global", "dice_macro", "iou_macro", "bf1", "a2_aji_cells", "aji_cc",
            "a2_scored_match_pct", "a2_scored_missed_pct", "n_lost_classes"]
    keep = [r for r in inp.csv(f"{SEL}/arm_confirmation.csv") if r["loss_a"] == PROPOSED and r["metric"] in head]
    cols = ["Compared objective", "Quantity", "Proposed objective", "Compared objective ", "Median difference",
            "Runs in which the proposed objective is better", "95 % interval of the median difference",
            "p (Wilcoxon signed-rank)"]
    others = [a for a in R.arms if a != PROPOSED]
    rows = []
    for o in others:
        for m in head:
            r = next((x for x in keep if x["loss_b"] == o and x["metric"] == m), None)
            if r is None:
                continue
            s = 100.0 if m == "ipe_global" else 1.0
            lab = r["metric_label"] + (" (%)" if s != 1.0 else "")
            rows.append([arm_long(o), lab, num(r["mean_a"], 3, s), num(r["mean_b"], 3, s),
                         num(r["median_difference_a_minus_b"], 4, s), f"{r['pairs_a_better']} of {r['n_pairs']}",
                         f"[{num(r['ci_low'], 4, s)}, {num(r['ci_high'], 4, s)}]", fmt_p(r["wilcoxon_p"])])
    return Table("S3", "objective_comparison_pooled", "The proposed objective against every other loss "
                 "configuration, architectures and ranks pooled", cols, rows,
                 ["Pairs: same architecture, low-rank adaptation rank and seed (every architecture, rank and seed).",
                  "Interval: percentile bootstrap of the median paired difference. Wilcoxon p from the project's "
                  "selection script; the per-architecture exact tests with Holm correction are in Table 2."],
                 group_cols=1)


def tableS4(inp: Inputs, R: Runs, arm: str = PROPOSED) -> Table:
    rep = representative(inp)
    cols = ["Architecture", "Low-rank adaptation", "Class", "Dice", "Intersection over union",
            "Integrated-phase error of the class (%)", "Test images containing the class", "Runs in which the class was lost"]
    rows = []
    for a in R.archs:
        c = rep[a]["selected_condition"]
        xs = R.get(a, c, arm)
        for k in R.classes:
            n_img = sorted({int(x.get(f"n_images_with_{k}")) for x in xs if x.get(f"n_images_with_{k}") is not None})
            rows.append([a, cond_label(c), k, mean_sd(R.col(a, c, arm, f"dice_{k}"), 3),
                         mean_sd(R.col(a, c, arm, f"iou_{k}"), 3), mean_sd(R.col(a, c, arm, f"ipe_{k}", 100.0), 1),
                         ", ".join(map(str, n_img)) or "–",
                         f"{sum(bool(x.get(f'lost_{k}')) for x in xs)} of {len(xs)}"])
    return Table("S4", "per_class", "Per-class accuracy at the representative configurations", cols, rows,
                 [f"Mean ± standard deviation over the {len(R.seeds)} seeds.",
                  "The integrated-phase error of a class is measured over the test images that contain that class; a "
                  "class the network never predicts has an error of 100 %, because none of its phase is recovered."], group_cols=2)


def tableS5(inp: Inputs, R: Runs) -> Table:
    rows_in = inp.csv(f"{SEL}/lora_vs_full_finetuning.csv")
    qty = [("a2_scored_within_5pct_of_all_cells_pct", "Cells within 5 %, all scored cells (%)", 1.0, 3),
           ("a2_touching_within_5pct_of_all_cells_pct", "Cells within 5 %, touching cells (%)", 1.0, 3),
           ("a2_scored_cell_ipe_abs_median_pct", "Cell integrated-phase error (%)", 1.0, 3),
           ("a2_scored_drymass_ccc", "Dry mass, concordance correlation", 1.0, 3),
           ("ipe_global", "Global integrated-phase error (%)", 100.0, 3),
           ("dice_macro", "Mean Dice", 1.0, 3), ("a2_aji_cells", "Aggregated Jaccard index, cells", 1.0, 3),
           ("bf1", "Boundary F1", 1.0, 3)]
    cols = ["Architecture", "Representative rank", "Learning rate, low-rank adaptation / full fine-tuning",
            "Quantity", "At the representative rank", "Mean over every rank", "Full fine-tuning",
            "Representative rank minus full fine-tuning", "Difference per seed",
            "Seeds in which the representative rank is better", "Runs that lost a class: every rank / full fine-tuning"]
    rows = []
    for r in rows_in:
        if r.get("loss") not in (PROPOSED, None):
            continue
        for k, lab, s, p in qty:
            per = r.get(f"selected_minus_full_per_seed_{k}", "")
            per_s = ", ".join(f"{float(v) * s:+.{p}f}" for v in per.split(";") if v.strip()) if per else "–"
            rows.append([arch_disp(r["architecture_display"]), cond_label(r["selected_condition"]),
                         f"{r['learning_rate_lora']} / {r['learning_rate_full_finetune']}", lab,
                         num(r.get(f"selected_mean_{k}"), p, s), num(r.get(f"lora_mean_{k}"), p, s),
                         num(r.get(f"full_mean_{k}"), p, s), num(r.get(f"selected_minus_full_{k}"), p, s, sign=True),
                         per_s, f"{r.get(f'selected_better_in_{k}', '–')} of {len(per.split(';')) if per else '–'}",
                         f"{r['lora_runs_that_lost_a_class']} of {r['n_lora_runs']} / "
                         f"{r['full_runs_that_lost_a_class']} of {r['n_full_runs']}"])
    return Table("S5", "lora_vs_full_finetuning", "Low-rank adaptation against full fine-tuning", cols, rows,
                 ["Objective: proposed. The two regimes use different learning rates (shown), so the comparison is "
                  "of the two training recipes as run, not of the adaptation alone."], group_cols=3)


def tableS6(inp: Inputs, R: Runs) -> Table:
    cols = ["Architecture", "Loss configuration", "Runs", "Runs that lost a class",
            "Conditions in which a class was lost"] + [f"Runs that lost the {k}" for k in R.classes]
    rows = []
    for a in R.archs:
        for arm in R.arms:
            xs = [x for c in R.conditions for x in R.get(a, c, arm)]
            lost_conds = [c for c in R.conditions if any((x.get("n_lost_classes") or 0) > 0 for x in R.get(a, c, arm))]
            rows.append([a, arm_short(arm), str(len(xs)), str(sum((x.get("n_lost_classes") or 0) > 0 for x in xs)),
                         ", ".join(cond_label(c) for c in lost_conds) or "none"] +
                        [str(sum(bool(x.get(f"lost_{k}")) for x in xs)) for k in R.classes])
    return Table("S6", "class_retention", "Training stability: runs in which a morphology class was lost", cols, rows,
                 [f"Every rank and full fine-tuning, {len(R.seeds)} seeds each. A class is lost when its Dice is "
                  f"below {R.lost_class_dice}.", _glossary()], group_cols=1)


def tableS7(inp: Inputs, R: Runs, arm: str = PROPOSED) -> Table:
    st = inp.json(f"results/consolidated/storage/runs_{R.checkpoint}.json")
    rep = representative(inp)
    cols = ["Architecture", "Low-rank adaptation", "Seed", "Quantity", "Spearman rho", "p (asymptotic)",
            "p (date-block permutation)", "95 % interval of rho (date bootstrap)", "Range of rho, leaving one day out",
            "Change of the median, early to late (%)", "Spearman rho of the bias against day",
            "p of the bias against day (date-block permutation)"]
    rows = []
    for a in R.archs:
        for c in [rep[a]["selected_condition"]]:
            for s in R.seeds:
                r = next((x for x in st["runs"] if arch_disp(x["arch"]) == a and x["condition"] == c
                          and x["arm"] == arm and int(x["seed"]) == s), None)
                if not r:
                    continue
                for q, lab in (("area", "Area"), ("dry_mass", "Dry mass")):
                    t = r[q]
                    rows.append([a, cond_label(c), str(s), lab, num(t["rho"], 3), fmt_p(t["p_asymptotic"]),
                                 fmt_p(t["p_block_permutation"]), ci(t["rho_ci95_date_bootstrap"], 3),
                                 ci(t["rho_leave_one_day_out"], 3), num(t["early_to_late_change_pct"], 2),
                                 num(t["bias_vs_annotation"]["rho"], 3),
                                 fmt_p(t["bias_vs_annotation"]["p_block_permutation"])])
    return Table("S7", "storage_trends_per_seed", "Storage-time trends of the representative configurations, per seed",
                 cols, rows, ["Every configuration and objective is in results/consolidated/storage/runs_last.json."],
                 group_cols=3)


def tableS8(inp: Inputs, R: Runs, arm: str = PROPOSED) -> Table:
    import pandas as pd
    rep = representative(inp)
    cols = ["Source", "Architecture", "Low-rank adaptation", "Seed", "Storage day", "Cells",
            "Area (µm²), median", "Dry mass (pg), median", "Circularity, median"] + [f"{k} (%)" for k in R.classes]
    rows, no_circ, ref_done = [], [], False

    def block(D, src, a, c, s, area, mass, circ, cls):
        for day, g in D.groupby("storage_day"):
            rows.append([src, a, c, s, str(int(day)), str(len(g)), num(g[area].median(), 2), num(g[mass].median(), 2),
                         num(g[circ].median(), 4) if circ in g else "–"] +
                        [num(100.0 * float((g[cls] == k).mean()), 1) for k in R.classes])

    for a in R.archs:
        c = rep[a]["selected_condition"]
        for s in R.seeds:
            rel = f"{run_dir(a, c, arm, s)}/cells_{R.checkpoint}.json"
            if not inp.exists(rel):
                continue
            D = pd.DataFrame([x for x in require_rescored(inp, rel)["cells"] if x.get("instances") == "A2"])
            if "circularity" not in D:
                no_circ.append(f"{a} {cond_label(c)} seed {s}")
            sc = D[D["scored"].fillna(False).astype(bool)]
            if not ref_done:
                block(sc, "Annotation", "–", "–", "–", "area_um2", "dm_gt_pg", "circularity", "gt_class")
                ref_done = True
            u = sc[sc["matched"].fillna(False).astype(bool)]
            if "pred_wrapped" in u:
                u = u[~u["pred_wrapped"].fillna(False).astype(bool)]
            block(u, "Prediction", a, cond_label(c), str(s), "area_pred_um2", "dm_pred_pg", "circularity_pred", "pred_class")
    notes = ["Annotation: scored reference cells (area gate passed, not touching the unannotated area, not near phase "
             "wrapping); independent of the model, given once.",
             "Prediction: each run's matched cells whose predicted region is not near phase wrapping; area, dry mass, "
             "circularity and class are the predicted values.",
             "Circularity: 4πA/P² (Crofton perimeter), each cell measured alone."]
    if no_circ:
        notes.append("Circularity is written by the project's rescoring step; it is absent (–) in the cells files of: "
                     + "; ".join(no_circ) + ".")
    return Table("S8", "per_day_shape", "Area, dry mass, circularity and class composition per storage day", cols, rows,
                 notes, group_cols=4)


def tableS9(inp: Inputs, R: Runs) -> Table:
    rows_in = inp.csv(f"{SEL}/rank_selection_sensitivity.csv")
    tol = []
    for r in rows_in:
        t = r["tolerance_points"]
        if t not in tol:
            tol.append(t)
    archs = [a for a in R.archs if any(arch_disp(r["architecture_display"]) == a for r in rows_in)]
    cols = ["Tolerance (points of cells within 5 %)"] + archs
    rows = []
    for t in tol:
        row = [t]
        for a in archs:
            r = next(x for x in rows_in if x["tolerance_points"] == t and arch_disp(x["architecture_display"]) == a)
            lab = cond_label(r["selected_condition"])
            row.append(lab if str(r.get("every_rank_lost_a_class", "")).lower() != "true"
                       else f"{lab} (no rank kept every class)")
        rows.append(row)
    return Table("S9", "rank_selection_sensitivity", "The rank the selection rule chooses at each tolerance", cols, rows,
                 ["The manuscript uses a tolerance of zero; a non-zero tolerance would accept the smallest stable rank "
                  "within that many points of the best."])


def tableS10(inp: Inputs, R: Runs) -> Table:
    rows_in = inp.csv("results/consolidated/benchmark/benchmark_parameters.csv")
    cols = ["Architecture", "Condition", "Total parameters", "Trainable parameters",
            "Trainable (%) of the network while training", "Trainable (%) of the deployed network",
            "Adapter parameters", "Trainable outside the adapters", "Adapted layers",
            "Parameters after merging the adapters"]
    order = {a: i for i, a in enumerate(R.archs)}
    corder = {c: i for i, c in enumerate(R.conditions)}
    rows_in = sorted(rows_in, key=lambda r: (order.get(arch_disp(r["architecture_display"]), 9), corder.get(r["condition"], 9)))
    rows = []
    for r in rows_in:
        i = lambda k: f"{int(float(r[k])):,}"  # noqa: E731
        dep = 100.0 * float(r["parameters_trainable"]) / float(r["parameters_after_merge"])
        rows.append([arch_disp(r["architecture_display"]), cond_label(r["condition"]), i("parameters_total"),
                     i("parameters_trainable"), num(r["parameters_trainable_pct"], 2), num(min(dep, 100.0), 2),
                     i("adapter_parameters"), i("trainable_without_adapters"), str(int(float(r["adapted_layers"]))),
                     i("parameters_after_merge")])
    return Table("S10", "trainable_parameters", "Parameters trained at each rank, and the deployed network", cols, rows,
                 ["The deployed network is the same at every rank: the adapters are merged into the base weights.",
                  "Trainable (%) while training divides by the parameters carried during training (adapters "
                  "included); of the deployed network divides by the merged network."], group_cols=1)


def tableS11(inp: Inputs, R: Runs) -> Table:
    w1 = inp.json(f"{W}/w1_cell_paired/w1_cell_paired.json")
    st = [("d_within5_pp", "Cells within 5 %: difference (points)"),
          ("d_median_abs_e_assigned_pp", "Median |cell error|, matching-free: difference (points)"),
          ("d_median_abs_e_matched_both_pp", "Median |cell error|, cells matched under both: difference (points)"),
          ("d_median_abs_drymass_err_both_pg", "Median |dry-mass error|, matched under both: difference (pg)")]
    cols = ["Architecture", "Low-rank adaptation", "Comparison (A vs B)", "Cells", "Cells within 5 %: A / B (%)"]
    for _, lab in st:
        cols += [f"{lab}", "95 % interval (hierarchical bootstrap)", "p (day-level label swap)", "p (Holm)"]
    rows = []
    order = {a: i for i, a in enumerate(R.archs)}
    for r in sorted(w1["results"], key=lambda x: order.get(arch_disp(x["architecture"]), 9)):
        for grp in ("all", "touching", "isolated"):
            G = r["groups"].get(grp)
            if not G:
                continue
            L, S = G["levels"], G["statistics"]
            row = [arch_disp(r["architecture"]), cond_label(r["condition"]),
                   f"{arm_short(r['loss_a'])} vs {arm_short(r['loss_b'])}" + ("" if grp == "all" else f" — {grp} cells"),
                   str(G["n_cells"]), f"{num(L['within5_pct_A'], 2)} / {num(L['within5_pct_B'], 2)}"]
            for k, _ in st:
                s = S.get(k, {})
                row += [num(s.get("estimate"), 3, sign=True), ci(s.get("ci95_hierarchical_bootstrap"), 3),
                        fmt_p(s.get("p_day_label_swap")), fmt_p(s.get("p_holm")) if "p_holm" in s else "–"]
            rows.append(row)
    md = w1["metadata"]
    return Table("S11", "cell_paired_statistics", "Cell-level paired comparisons at the representative configurations",
                 cols, rows,
                 [f"Cells are the unit: the same scored reference cells under both objectives, pooled over the "
                  f"{len(r['seeds'])} seeds. Interval: hierarchical bootstrap over storage days, then images, then "
                  f"cells ({md.get('bootstrap', '–')} resamples). p: exact test swapping the two objectives' labels "
                  "within whole storage days (2^days arrangements); Holm over the primary family "
                  f"({md.get('primary_statistics', 'see metadata')}).",
                  "Matching-free error: each reference cell's integrated phase against the predicted phase assigned "
                  "to it, so a missed cell counts with an error near 100 %."], group_cols=2)


def tableS12(inp: Inputs, R: Runs) -> Table:
    w4 = inp.json(f"{W}/w4_cell_vs_global_ipp/w4_cell_vs_global_ipp.json")["results"]
    keys = ["a2_scored_within_5pct_of_all_cells_pct", "a2_touching_within_5pct_of_all_cells_pct",
            "a2_scored_cell_ipe_abs_median_pct", "ipe_global", "a2_scored_drymass_ccc", "a2_scored_match_pct",
            "a2_scored_merged_pct"]
    cols = ["Set of runs", "Quantity", f"{arm_short(NEW_FULL)}", f"{arm_short(OLD_FULL)}", "Median difference",
            "Pairs in which the new full objective is better", "95 % interval of the median difference", "p (Wilcoxon)"]
    rows = []
    blocks = [("All architectures, low-rank adaptation", w4["lora_pooled"])]
    blocks += [(f"{arch_disp(a)}, low-rank adaptation", v) for a, v in w4["per_architecture_lora"].items()]
    blocks += [("All architectures, full fine-tuning", w4["full_finetuning_pooled"])]
    for name, B in blocks:
        for k in keys:
            m = B["metrics"].get(k)
            if not m:
                continue
            s = 1.0   # W-4 stores every quantity on the printed scale (global error already in per cent)
            pa = m.get("pairs_new_full_better", m.get("pairs_a_better"))
            rows.append([name, m["label"], num(m.get("mean_new_full", m.get("mean_a")), 3, s),
                         num(m.get("mean_old_full", m.get("mean_b")), 3, s),
                         num(m["median_difference"] if "median_difference" in m else m.get("median_difference_a_minus_b"), 3, s, sign=True),
                         f"{pa} of {m['n_pairs']}", ci([v * s for v in (m.get("ci95_median_difference") or [])], 3)
                         if m.get("ci95_median_difference") else "–",
                         fmt_p(m.get("wilcoxon_p", m.get("wilcoxon_exact_p")))])
    return Table("S12", "cell_vs_global_ipp", "One-factor comparison: cell-level against global phase preservation",
                 cols, rows, [f"The two objectives differ only in the phase term ({w4['comparison']['only_difference']}).",
                              "Pairs: same architecture, rank and seed."], group_cols=1)


def tableS13(inp: Inputs, R: Runs) -> Table:
    bi = inp.json("results/consolidated/background_invariance/background_invariance.json")
    cols = ["Added background term", "Amplitude (rad)", "Images", "Cells", "Largest change of the corrected map (rad)",
            "Cell integrated phase: median |change| (%)", "Cell integrated phase: 95th percentile |change| (%)",
            "Cell integrated phase: largest |change| (%)", "Cell dry mass: median |change| (pg)",
            "Cell dry mass: largest |change| (pg)"]
    rows = []
    for r in bi["results"]:
        ph, dm = r["cell_integrated_phase_change_pct"], r["cell_dry_mass_change_pg"]
        rows.append([r["term"], num(r["amplitude_rad"], 2), str(r["images"]), str(r["cells"]),
                     f"{r['map_change_max_rad']:.2e}", f"{ph.get('median_abs', float('nan')):.2e}",
                     f"{ph.get('p95_abs', float('nan')):.2e}", f"{ph.get('max_abs', float('nan')):.2e}",
                     f"{dm.get('median_abs', float('nan')):.2e}", f"{dm.get('max_abs', float('nan')):.2e}"])
    b = bi["background_settings"]
    return Table("S13", "background_invariance", "Invariance of the annotation-free background correction",
                 cols, rows, [f"A known low-order term was added to each test image and the correction re-run "
                              f"(order {b.get('order')}, k_sigma {b.get('k_sigma')}, grow {b.get('grow_px')} px, "
                              f"at most {b.get('max_iter')} iterations). The changes are what the added term leaves "
                              "after correction."])


def tableS14(inp: Inputs, R: Runs) -> Table:
    q = inp.json("results/consolidated/phase_quartile_fnr/phase_quartile_fnr.json")
    edges = q["bin_edges_rad"]
    labels = q.get("bin_labels") or [f"bin {i + 1}" for i in range(q["bins"])]
    cols = ["Architecture", "Low-rank adaptation", "Loss configuration", "Seeds", "Missed annotated pixels, overall (%)"] + \
           [f"Missed, phase {lab} (%)" for lab in labels]
    groups = OrderedDict()
    for r in q["runs"]:
        groups.setdefault((arch_disp(r["architecture_display"]), r["condition"], r["loss"]), []).append(r)
    rows = []
    for (a, c, l), rs in groups.items():
        rows.append([a, cond_label(c), arm_short(l), str(len(rs)), mean_sd([r["false_negative_rate_pct"] for r in rs], 2)] +
                    [mean_sd([r[f"fnr_bin{i + 1}_pct"] for r in rs], 2) for i in range(q["bins"])])
    return Table("S14", "missed_pixels_by_phase", "Annotated cell pixels the network misses, by phase quartile", cols,
                 rows, [f"Quartiles of the annotated cell pixels' phase over the {q['n_test_images']} test images; "
                        f"edges {', '.join(f'{e:.3f}' for e in edges)} rad. Mean ± SD over seeds."], group_cols=2)


def tableS15(inp: Inputs, R: Runs) -> Table:
    w5 = inp.json(f"{W}/w5_reference_cells/w5_reference_cells.json")["results"]
    fov = inp.json(f"{W}/w6_superposition_and_fov/w6_fov.json")["results"]
    cols = ["Storage day", "Images", "Reference cells", "Scored cells", "Touching cells (%)",
            "Cells in multi-cell clusters", "Largest cluster (cells)", "Annotated foreground (% of annotated area)"]
    rows = [[str(d["storage_day"]), str(d["images"]), str(d["reference_cells"]), str(d["scored_cells"]),
             num(d["touching_pct"], 1), str(d["cells_in_multi_cell_clusters"]), str(d["largest_cluster_cells"]),
             num(100.0 * d["fg_mean"], 1) if d.get("fg_mean") is not None else "–"] for d in w5["crowding_per_day"]]
    names = {"train": "training images (training and validation sets)", "val": "test images"}
    notes = []
    for k, v in fov["per_split"].items():
        notes.append(f"Field of view without annotation, {names.get(k, k)} ({v['n_images']}): mean "
                     f"{100 * v['unannotated_mean']:.1f} %, median {100 * v['unannotated_median']:.1f} %, "
                     f"{v['images_fully_annotated']} images fully annotated.")
    a = fov["all_images"]
    notes.append(f"All {a['n_images']} images: mean {100 * a['unannotated_mean']:.1f} % of the field without annotation.")
    s = w5["superposition_screen"]
    for k, v in s.items():
        notes.append(f"Screen for possibly superposed cells (mean phase ≥ {v['phase_ratio_threshold']} × the median "
                     f"of the same class and day, area ≤ {v['area_ratio_threshold']} × its median): "
                     f"{v['candidates_all_reference_cells']} of {w5['reference_cells']['all']} reference cells "
                     f"({v['candidates_scored_cells']} scored).")
    return Table("S15", "crowding_and_field_of_view", "Crowding of the reference cells per storage day, and the "
                 "annotated share of the field", cols, rows, notes)


MAIN = [table1, table2, table3, table4, table5]
SUPP = [tableS1, tableS2, tableS3, tableS4, tableS5, tableS6, tableS7, tableS8, tableS9, tableS10, tableS11,
        tableS12, tableS13, tableS14, tableS15]
