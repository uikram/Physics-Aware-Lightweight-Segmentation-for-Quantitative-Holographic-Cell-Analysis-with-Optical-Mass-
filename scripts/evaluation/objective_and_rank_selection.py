#!/usr/bin/env python
"""Two decisions of the revision, taken from the consolidated results. No training, no retraining, CPU only.

    python scripts/evaluation/objective_and_rank_selection.py

1. **Objective.** Is Dice + cell-level integrated-phase preservation as good as the full objective? Every loss
   configuration is compared against every other one on the same architecture, rank and seed, so each comparison is a
   paired one: 3 architectures x 4 LoRA ranks x 3 seeds = 36 pairs. Full fine-tuning is kept apart in its own table,
   because it is a different adaptation mode and not a rank.

2. **Representative rank.** For each architecture, under the objective chosen in step 1, the rule is:

       a. keep the ranks at which the fewest seeds lost a class (a class Dice below the configured threshold)
          -- training stability;
       b. among those, take the one with the best balanced score -- cell-level phase preservation and
          segmentation accuracy, weighted as `selection.weights` in the configuration;
       c. on a tie, the smaller rank -- the only place computational cost enters.

   `selection.source` decides which split the rule READS. With `validation` the rank is the one the same
   rule chose on the 41 held-out validation images, taken from `additional_analysis/.../w9_validation_selection.json`,
   so no test image takes part in choosing it and the test metrics are only reported at that rank. With `test`
   the rule is applied to the test metrics, as it was before 18 November 2026. The provenance of every choice
   is written to `rank_selection.csv` as `rank_chosen_on`.

   Stability in (a) is graded, not binary. When some rank keeps every class this is exactly "discard any rank
   at which a seed lost a class"; when none does, the most stable ranks are kept rather than falling back to
   all of them. The rule therefore always produces a choice, with no exception for any architecture, and
   `every_rank_lost_a_class` in the output records which of the two cases applied.

   Every value comes from `selection` in the configuration; nothing here is hard-coded. The default score is
   the mean of the share of cells within 5 % (divided by 100) and the mean Dice, so phase preservation and
   segmentation accuracy carry equal weight. Setting `selection.rule: within5` reproduces the earlier rule,
   which read the share of cells within 5 % alone.

   Computational cost is deliberately not a term in the score. The adapters are merged into the base weights
   after training, so the deployed network and its inference cost are identical at every rank; the only cost
   that differs is the number of trainable parameters during training, and that enters through the tie-break.

   `--tolerance` relaxes (b) to "the smallest rank within this many score points of the best", which trades
   accuracy for a smaller adapter. The default comes from the configuration and is 0, so no such trade is made.
   The script prints which rank every tolerance from 0 to 0.05 would have selected, so the effect of relaxing
   it is visible rather than hidden.

3. **Low-rank adaptation against full fine-tuning.** Means over the LoRA ranks against the full fine-tuning runs of
   the same architecture, and the selected rank against full fine-tuning seed by seed. The two differ in learning
   rate as well as in adaptation mode, so this is a comparison of two configurations, not of adaptation modes alone;
   the rate of each is reported with it.

Outputs, in results/consolidated/selection/:

    arm_confirmation.csv            every paired comparison, LoRA ranks
    arm_confirmation_fullft.csv     the same for full fine-tuning
    class_retention.csv             runs that lost a class, and per-class Dice, by architecture and loss
    rank_selection.csv              the rule applied, per architecture
    lora_vs_full_finetuning.csv     the LoRA ranks against full fine-tuning, per architecture
    rank_selection_sensitivity.csv  the rank each tolerance would choose
    rank_table.csv                  every rank of every architecture with the metrics behind the rule
    selection_summary.md            a readable summary of both decisions
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from qpicell.cli import add_common, config  # noqa: E402

# The metrics the two decisions are read from. direction = +1 when larger is better, -1 when smaller is better.
METRICS = [
    ("a2_scored_within_5pct_of_all_cells_pct", +1, "cells within 5 % (all scored cells)"),
    ("a2_touching_within_5pct_of_all_cells_pct", +1, "cells within 5 % (touching cells)"),
    ("a2_isolated_within_5pct_of_all_cells_pct", +1, "cells within 5 % (isolated cells)"),
    ("a2_scored_cell_ipe_abs_median_pct", -1, "cell integrated-phase error, matched cells"),
    ("a2_touching_cell_ipe_abs_median_pct", -1, "cell integrated-phase error, touching cells"),
    ("a2_isolated_cell_ipe_abs_median_pct", -1, "cell integrated-phase error, isolated cells"),
    ("a2_scored_network_cell_ipe_abs_median_pct", -1, "cell integrated-phase error without matching"),
    # stored as a fraction; the tables print it as a percentage, like every other error in the manuscript
    ("ipe_global", -1, "global integrated-phase error"),
    ("a2_scored_drymass_mdare_pct", -1, "dry mass, median absolute relative error"),
    ("a2_scored_drymass_ccc", +1, "dry mass, concordance correlation"),
    ("a2_drymass_population_diff_pct", 0, "dry mass, population difference of medians"),
    ("a2_scored_match_pct", +1, "cells matched"),
    ("a2_scored_missed_pct", -1, "cells missed"),
    ("a2_scored_merged_pct", -1, "cells merged"),
    ("a2_scored_fragmented_pct", -1, "cells fragmented"),
    ("dice_macro", +1, "macro Dice"),
    ("iou_macro", +1, "macro intersection over union"),
    # two aggregated Jaccard indices: the first scores the separated cells against the reference cells, which
    # is what the cell-level results rest on; the second scores the connected regions of the foreground before
    # separation, so cells the network leaves joined count as one region. Reporting only the second understates
    # instance agreement badly, so both are compared. Neither takes part in the rank rule.
    ("a2_aji_cells", +1, "aggregated Jaccard index, cells"),
    ("aji_cc", +1, "aggregated Jaccard index, connected regions"),
    ("bf1", +1, "boundary F1"),
    ("n_lost_classes", -1, "classes lost"),
]
PRIMARY = "a2_scored_within_5pct_of_all_cells_pct"


def disp(name: str) -> str:
    """The architecture name as the manuscript writes it: the configuration spells MobileNet-UNet with an underscore."""
    return str(name).replace("_", "-")


def load(path: Path) -> list:
    return json.loads(path.read_text())["runs"]


def key(r) -> tuple:
    return (r["architecture"], r["condition"], r["seed"])


def wilcoxon(d: np.ndarray):
    """Two-sided Wilcoxon signed-rank p over the non-zero differences; None when scipy is absent or n < 6."""
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) < 6:
        return None
    try:
        from scipy.stats import wilcoxon as w
        return float(w(d).pvalue)
    except Exception:
        return None


def boot_ci(d: np.ndarray, n: int, rng) -> tuple:
    """Percentile interval of the median paired difference, resampling the pairs."""
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return (None, None)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    med = np.median(d[idx], axis=1)
    return (float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5)))


def compare(runs, arms, conditions, boots, rng) -> list:
    """Every ordered pair of loss configurations, on the runs they share."""
    by = {}
    for r in runs:
        if r["condition"] in conditions:
            by.setdefault(r["loss"], {})[key(r)] = r
    rows = []
    for a in arms:
        for b in arms:
            if a == b or a not in by or b not in by:
                continue
            shared = sorted(set(by[a]) & set(by[b]))
            if not shared:
                continue
            for metric, direction, label in METRICS:
                va = np.array([by[a][k].get(metric, np.nan) for k in shared], float)
                vb = np.array([by[b][k].get(metric, np.nan) for k in shared], float)
                ok = np.isfinite(va) & np.isfinite(vb)
                if ok.sum() == 0:
                    continue
                d = va[ok] - vb[ok]
                better = int((d * direction > 0).sum()) if direction else None
                worse = int((d * direction < 0).sum()) if direction else None
                lo, hi = boot_ci(d, boots, rng)
                rows.append({"loss_a": a, "loss_b": b, "metric": metric, "metric_label": label,
                             "larger_is_better": direction, "n_pairs": int(ok.sum()),
                             "mean_a": float(va[ok].mean()), "mean_b": float(vb[ok].mean()),
                             "median_difference_a_minus_b": float(np.median(d)),
                             "pairs_a_better": better, "pairs_b_better": worse,
                             "ci_low": lo, "ci_high": hi, "wilcoxon_p": wilcoxon(d)})
    return rows


def retention(runs, arms) -> list:
    """How often each loss configuration loses a class, and the per-class Dice behind it."""
    classes = [k for k in runs[0] if k.startswith("dice_") and k not in ("dice_macro",)]
    out = []
    for arch in sorted({r["architecture"] for r in runs}):
        for arm in arms:
            xs = [r for r in runs if r["architecture"] == arch and r["loss"] == arm]
            if not xs:
                continue
            row = {"architecture": arch, "loss": arm, "n_runs": len(xs),
                   "runs_that_lost_a_class": int(sum(1 for r in xs if (r.get("n_lost_classes") or 0) > 0)),
                   "conditions_that_lost_a_class": ",".join(sorted({r["condition"] for r in xs
                                                                    if (r.get("n_lost_classes") or 0) > 0}))}
            for c in classes:
                v = [r[c] for r in xs if isinstance(r.get(c), (int, float))]
                row[f"mean_{c}"] = float(np.mean(v)) if v else None
            out.append(row)
    return out


def rank_table(runs, arm, ranks) -> list:
    """One row per architecture and rank: the numbers the rule reads, averaged over the seeds."""
    out = []
    for arch in sorted({r["architecture"] for r in runs}):
        for cond in ranks:
            xs = [r for r in runs if r["architecture"] == arch and r["loss"] == arm and r["condition"] == cond]
            if not xs:
                continue
            row = {"architecture": arch, "architecture_display": disp(xs[0]["architecture_display"]), "condition": cond,
                   "rank": xs[0]["rank"], "n_seeds": len(xs),
                   "seeds_that_lost_a_class": int(sum(1 for r in xs if (r.get("n_lost_classes") or 0) > 0)),
                   "params_trainable": xs[0].get("params_trainable"),
                   "params_trainable_pct": (100.0 * xs[0]["params_trainable"] / xs[0]["params_total"]
                                            if xs[0].get("params_total") else None),
                   "adapted_layers": xs[0].get("adapted_layers"),
                   "seconds_per_epoch": float(np.mean([r["seconds_per_epoch"] for r in xs
                                                       if r.get("seconds_per_epoch")])) if any(
                       r.get("seconds_per_epoch") for r in xs) else None}
            for metric, _, _ in METRICS:
                v = [r[metric] for r in xs if isinstance(r.get(metric), (int, float))]
                row[f"mean_{metric}"] = float(np.mean(v)) if v else None
                row[f"sd_{metric}"] = float(np.std(v, ddof=1)) if len(v) > 1 else None
            out.append(row)
    return out


#: metrics that are percentages and must be divided by 100 before they are weighed against a 0..1 metric
_PERCENT_METRICS = {m for m, _, _ in METRICS if m.endswith("_pct")}


def score_of(row, weights) -> float:
    """The balanced score of one rank: sum(weight x metric), every metric scaled to 0..1.

    Returns None when any weighted metric is missing, so a rank with an incomplete row is never chosen on a
    partial score."""
    total = 0.0
    for metric, w in weights.items():
        v = row.get(f"mean_{metric}")
        if v is None:
            return None
        total += float(w) * (v / 100.0 if metric in _PERCENT_METRICS else v)
    return total


def selection_weights(cfg) -> dict:
    """The weights of the rule named in the configuration. No weight is hard-coded here."""
    sel = cfg["selection"]
    if sel["rule"] == "within5":
        return {PRIMARY: 1.0}
    if sel["rule"] != "balanced":
        raise SystemExit(f"selection.rule = {sel['rule']!r}: known rules are 'balanced' and 'within5'")
    return dict(sel["weights"])


#: how the architecture is spelled in W-9's output, keyed by the display name used here
_W9_ARCH = {"EdgeSAM": "EdgeSAM", "MobileSAM": "MobileSAM", "MobileNet-UNet": "MobileNet_UNet"}


def validation_choice(cfg) -> dict:
    """The rank each architecture's rule chose on the validation images, read from W-9's output.

    Returns {display name: rank}. Raises if the file is missing or does not cover an architecture, so a
    validation-based selection can never silently fall back to the test set."""
    rule = cfg["selection"]["rule"]
    path = ROOT / cfg["selection"]["validation_results"]
    if not path.exists():
        raise SystemExit(f"selection.source is 'validation' but {path} does not exist.\n"
                         f"Run: python additional_analysis/w9_validation_selection.py --select")
    data = json.loads(path.read_text(encoding="utf-8"))["rank_selection"]
    out = {}
    for disp, key in _W9_ARCH.items():
        if key not in data:
            raise SystemExit(f"{path} has no entry for {key}; W-9 must cover every architecture.")
        if rule not in data[key]:
            raise SystemExit(f"{path} has no '{rule}' rule for {key}; it holds {sorted(data[key])}.")
        out[disp] = int(str(data[key][rule]["chosen"]).lstrip("r"))
    return out


def apply_rule(table, tol: float, weights: dict, stability: str = "graded", forced: dict | None = None) -> list:
    """The rank rule: stability first, then the smallest rank within `tol` score points of the best.

    Stability is graded rather than binary: the pool is the ranks at which the fewest seeds lost a class. When
    some rank keeps every class this is exactly "discard any rank at which a seed lost a class"; when none
    does, the most stable ranks are kept instead of falling back to all of them. The rule therefore always
    produces a choice, and `every_rank_lost_a_class` records which of the two cases applied."""
    out = []
    for arch in sorted({t["architecture"] for t in table}):
        xs = sorted((t for t in table if t["architecture"] == arch), key=lambda t: t["rank"])
        stable = [t for t in xs if t["seeds_that_lost_a_class"] == 0]
        if stability == "graded":
            fewest = min(t["seeds_that_lost_a_class"] for t in xs)
            pool = [t for t in xs if t["seeds_that_lost_a_class"] == fewest]
        elif stability == "binary":
            fewest = 0 if stable else min(t["seeds_that_lost_a_class"] for t in xs)
            pool = stable if stable else list(xs)
        elif stability == "none":
            fewest, pool = 0, list(xs)
        else:
            raise SystemExit(f"selection.stability = {stability!r}: known values are graded, binary and none")
        applied = bool(stable)
        pool = [t for t in pool if score_of(t, weights) is not None]
        if not pool:
            continue
        best = max(score_of(t, weights) for t in pool)
        shown = disp(xs[0]["architecture_display"])
        if forced is not None:
            if shown not in forced:
                raise SystemExit(f"no validation-chosen rank for {shown}; W-9 must cover every architecture.")
            want = forced[shown]
            chosen = next((t for t in xs if t["rank"] == want and score_of(t, weights) is not None), None)
            if chosen is None:
                raise SystemExit(f"the validation rule chose r = {want} for {shown}, but the test results hold "
                                 f"no such rank with complete metrics.")
            where = "validation"
        else:
            chosen = min((t for t in pool if score_of(t, weights) >= best - tol), key=lambda t: t["rank"])
            where = "test"
        out.append({"architecture": arch, "architecture_display": disp(chosen["architecture_display"]),
                    "rule_applied": True, "rank_chosen_on": where,
                    "every_rank_lost_a_class": not applied,
                    "fewest_seeds_that_lost_a_class": fewest,
                    "ranks_scored": ",".join(str(t["rank"]) for t in pool),
                    "selection_weights": ";".join(f"{k}={v}" for k, v in weights.items()),
                    "best_score": best, "selected_score": score_of(chosen, weights),
                    "tolerance_points": tol,
                    "ranks_considered": ",".join(str(t["rank"]) for t in xs),
                    "ranks_passing_stability": ",".join(str(t["rank"]) for t in stable) or "none",
                    "best_mean_within_5pct": max(t[f"mean_{PRIMARY}"] for t in pool),
                    "selected_condition": chosen["condition"],
                    "selected_rank": chosen["rank"],
                    "selected_mean_within_5pct": chosen[f"mean_{PRIMARY}"],
                    "selected_sd_within_5pct": chosen[f"sd_{PRIMARY}"],
                    "selected_mean_dice_macro": chosen["mean_dice_macro"],
                    # named for which index it is, now that both are computed
                    "selected_mean_aji_cells": chosen["mean_a2_aji_cells"],
                    "selected_mean_aji_connected_regions": chosen["mean_aji_cc"],
                    "selected_mean_cell_ipe_pct": chosen["mean_a2_scored_cell_ipe_abs_median_pct"],
                    "selected_mean_global_ipe_pct": (100.0 * chosen["mean_ipe_global"]
                                                 if chosen["mean_ipe_global"] is not None else None),
                    "selected_params_trainable": chosen["params_trainable"],
                    "selected_params_trainable_pct": chosen["params_trainable_pct"],
                    "note": "" if applied else (f"no rank kept every class in all seeds; the most stable ranks "
                                                f"({fewest} of 3 seeds lost a class) were scored instead")})
    return out


def lora_vs_full(runs, arm, ranks, selected, cfg) -> list:
    """The LoRA ranks against full fine-tuning, per architecture.

    Two readings are given, because they answer different questions: the mean over every LoRA rank against the mean
    over the full fine-tuning runs, and the rank the rule selected against full fine-tuning seed by seed, which is
    the comparison the manuscript makes. The learning rates differ between the two modes, so both are reported with
    the row: adaptation mode and learning rate change together and no claim can separate them."""
    lr_lora = {a: cfg["models"][a]["learning_rate"] for a in cfg["models"]}
    lr_full = cfg["conditions"]["full_finetune"]["learning_rate"]
    chosen = {s["architecture"]: s["selected_condition"] for s in selected}
    out = []
    for arch in sorted({r["architecture"] for r in runs}):
        xs = [r for r in runs if r["architecture"] == arch and r["loss"] == arm]
        lo = [r for r in xs if r["condition"] in ranks]
        fu = [r for r in xs if r["condition"] == "fullft"]
        sel = [r for r in xs if r["condition"] == chosen.get(arch)]
        if not lo or not fu:
            continue
        row = {"architecture": arch, "architecture_display": disp(xs[0]["architecture_display"]), "loss": arm,
               "selected_condition": chosen.get(arch, ""), "n_lora_runs": len(lo), "n_full_runs": len(fu),
               "learning_rate_lora": lr_lora.get(arch), "learning_rate_full_finetune": lr_full,
               "lora_runs_that_lost_a_class": sum(1 for r in lo if (r.get("n_lost_classes") or 0) > 0),
               "full_runs_that_lost_a_class": sum(1 for r in fu if (r.get("n_lost_classes") or 0) > 0),
               "selected_runs_that_lost_a_class": sum(1 for r in sel if (r.get("n_lost_classes") or 0) > 0)}
        for metric, direction, _ in METRICS:
            a = [r[metric] for r in lo if isinstance(r.get(metric), (int, float))]
            b = [r[metric] for r in fu if isinstance(r.get(metric), (int, float))]
            c = [r[metric] for r in sel if isinstance(r.get(metric), (int, float))]
            row[f"lora_mean_{metric}"] = float(np.mean(a)) if a else None
            row[f"full_mean_{metric}"] = float(np.mean(b)) if b else None
            row[f"selected_mean_{metric}"] = float(np.mean(c)) if c else None
            if c and b:
                # paired by seed: three seeds, too few for a test, so the per-seed differences are given in full
                by_seed = {r["seed"]: r[metric] for r in sel if isinstance(r.get(metric), (int, float))}
                d = [by_seed[r["seed"]] - r[metric] for r in fu
                     if r["seed"] in by_seed and isinstance(r.get(metric), (int, float))]
                row[f"selected_minus_full_{metric}"] = float(np.mean(d)) if d else None
                row[f"selected_minus_full_per_seed_{metric}"] = ";".join(f"{x:+.4f}" for x in d)
                row[f"selected_better_in_{metric}"] = (sum(1 for x in d if x * direction > 0) if direction else None)
        out.append(row)
    return out


def write_csv(path: Path, rows: list) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[selection] {path.name}: {len(rows)} rows", flush=True)


def summary_md(cmp_rows, sel, sens, table, arm, reference_arms, tol, weights, lvf=()) -> str:
    def find(a, b, metric):
        for r in cmp_rows:
            if r["loss_a"] == a and r["loss_b"] == b and r["metric"] == metric:
                return r
        return None

    L = ["# Objective and rank selection", "",
         f"Primary metric: share of all scored cells whose integrated phase is within 5 % "
         f"(`{PRIMARY}`). Checkpoint: last. Pairs are matched on architecture, rank and seed.", "",
         "## 1. Is Dice + cell-level integrated-phase preservation enough?", ""]
    for ref in reference_arms:
        L.append(f"### {arm} against {ref}")
        L.append("")
        L.append("| metric | n pairs | mean, " + arm + " | mean, " + ref +
                 " | median difference | pairs better | 95 % interval | Wilcoxon p |")
        L.append("|---|---|---|---|---|---|---|---|")
        for metric, direction, label in METRICS:
            r = find(arm, ref, metric)
            if not r:
                continue
            ci = "-" if r["ci_low"] is None else f"{r['ci_low']:+.2f} to {r['ci_high']:+.2f}"
            p = "-" if r["wilcoxon_p"] is None else f"{r['wilcoxon_p']:.3g}"
            better = "-" if r["pairs_a_better"] is None else f"{r['pairs_a_better']} of {r['n_pairs']}"
            L.append(f"| {label} | {r['n_pairs']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | "
                     f"{r['median_difference_a_minus_b']:+.3f} | {better} | {ci} | {p} |")
        L.append("")
    terms = ", ".join(f"{w:g} x {next((lb for m, _, lb in METRICS if m == k), k)}" for k, w in weights.items())
    rule = (f"Rule: keep the ranks at which the fewest seeds lost a class (training stability); among those take "
            f"the best balanced score ({terms}); the smaller rank on a tie. Stability is graded, so when some "
            f"rank keeps every class this is exactly 'discard any rank at which a seed lost a class', and when "
            f"none does the most stable ranks are kept. The rule needs no exception for any architecture."
            if tol <= 0 else
            f"Rule: keep the ranks at which the fewest seeds lost a class (training stability); among those take "
            f"the best balanced score ({terms}); then the smallest rank within {tol:.4f} score points of that "
            f"best.")
    L += ["## 2. Representative rank", "", rule, "",
          "Computational cost is not a term in the score: the adapters are merged after training, so the deployed "
          "network and its inference cost are the same at every rank. Cost enters only through the tie-break.", "",
          "| architecture | ranks that kept every class | selected rank | balanced score | within 5 % | mean Dice | "
          "cell error % | global error % | trainable | note |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    # the consolidated global error is a fraction; every other error here is already a percentage
    for s in sel:
        pct = "" if s["selected_params_trainable_pct"] is None else f" ({s['selected_params_trainable_pct']:.1f} %)"
        L.append(f"| {s['architecture_display']} | {s['ranks_passing_stability']} | "
                 f"{'full fine-tuning' if s['selected_condition'] == 'fullft' else 'r = %d' % s['selected_rank']} | "
                 f"{s['selected_score']:.4f} | "
                 f"{s['selected_mean_within_5pct']:.2f} | {s['selected_mean_dice_macro']:.3f} | "
                 f"{s['selected_mean_cell_ipe_pct']:.2f} | {s['selected_mean_global_ipe_pct']:.2f} | "
                 f"{s['selected_params_trainable']:,}{pct} | {s['note']} |")
    if lvf:
        L += ["", "## 3. Low-rank adaptation against full fine-tuning", "",
              "The selected rank against full fine-tuning, seed by seed. The learning rates differ between the two "
              "modes, so adaptation mode and learning rate change together and neither can be credited alone.", "",
              "| architecture | rate, LoRA | rate, full | metric | selected rank | full fine-tuning | difference | "
              "selected better in |", "|---|---|---|---|---|---|---|---|"]
        show = ["a2_scored_within_5pct_of_all_cells_pct", "dice_macro", "a2_aji_cells", "aji_cc",
                "a2_scored_cell_ipe_abs_median_pct", "a2_scored_drymass_ccc"]
        for r in lvf:
            for metric in show:
                if r.get(f"selected_mean_{metric}") is None or r.get(f"full_mean_{metric}") is None:
                    continue
                label = next(lb for m, _, lb in METRICS if m == metric)
                better = r.get(f"selected_better_in_{metric}")
                L.append(f"| {r['architecture_display']} | {r['learning_rate_lora']} | "
                         f"{r['learning_rate_full_finetune']} | {label} | "
                         f"{r[f'selected_mean_{metric}']:.3f} | {r[f'full_mean_{metric}']:.3f} | "
                         f"{r[f'selected_minus_full_{metric}']:+.3f} | "
                         f"{'-' if better is None else f'{better} of 3 seeds'} |")
            L.append(f"| {r['architecture_display']} | | | runs that lost a class | "
                     f"{r['selected_runs_that_lost_a_class']} of 3 at the selected rank "
                     f"({r['lora_runs_that_lost_a_class']} of {r['n_lora_runs']} over all ranks) | "
                     f"{r['full_runs_that_lost_a_class']} of {r['n_full_runs']} | | |")
        L.append("")
    L += ["", "### Does the choice depend on the tolerance?", "",
          "| tolerance (points) | " + " | ".join(sorted({t['architecture_display'] for t in table})) + " |",
          "|---|" + "---|" * len({t["architecture_display"] for t in table})]
    archs = sorted({t["architecture_display"] for t in table})
    for t_val in sorted({s["tolerance_points"] for s in sens}):
        cells = []
        for a in archs:
            hit = [s for s in sens if s["tolerance_points"] == t_val and s["architecture_display"] == a]
            cells.append("-" if not hit else ("full fine-tuning" if hit[0]["selected_condition"] == "fullft"
                                              else f"r = {hit[0]['selected_rank']}"))
        L.append(f"| {t_val:.4f} | " + " | ".join(cells) + " |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(ap)
    ap.add_argument("--checkpoint", default="last", help="which evaluation checkpoint to read (default: last)")
    ap.add_argument("--objective", default="dice_cipp", help="the loss configuration proposed as the main objective")
    ap.add_argument("--reference", default="dice_pmc_bga_cipp,dice_pmc_cipp,dice_pmc_bga_ipp,dice",
                    help="the loss configurations it is compared against")
    ap.add_argument("--tolerance", type=float, default=None,
                    help="balanced-score points a smaller rank may give up. The default comes from "
                         "selection.tolerance in the configuration, and is 0, so the rule picks the best stable "
                         "rank and carries no chosen constant")
    ap.add_argument("--bootstrap", type=int, default=10000, help="resamples for the interval of the median difference")
    a = ap.parse_args()
    cfg = config(a)

    cons = Path(cfg["paths_abs"]["results"]) / "consolidated"
    src = cons / f"runs_{a.checkpoint}.json"
    if not src.exists():
        raise SystemExit(f"{src} not found - run scripts/evaluation/consolidate.py first")
    runs = load(src)
    out = cons / "selection"
    arms = list(cfg["arms"].keys())
    ranks = [f"r{r}" for r in cfg["conditions"]["lora_ranks"]]
    refs = [x for x in a.reference.split(",") if x]
    rng = np.random.default_rng(0)
    print(f"[selection] {len(runs)} runs at checkpoint '{a.checkpoint}'; LoRA conditions {ranks}", flush=True)

    lora = compare(runs, arms, ranks, a.bootstrap, rng)
    write_csv(out / "arm_confirmation.csv", lora)
    write_csv(out / "arm_confirmation_fullft.csv", compare(runs, arms, ["fullft"], a.bootstrap, rng))
    write_csv(out / "class_retention.csv", retention(runs, arms))

    weights = selection_weights(cfg)
    stability = str(cfg["selection"]["stability"])
    tol = float(cfg["selection"]["tolerance"]) if a.tolerance is None else a.tolerance
    print(f"[selection] rank rule '{cfg['selection']['rule']}', stability '{stability}', "
          f"weights {weights}, tolerance {tol:g}, read on the "
          f"{cfg['selection']['source']} split", flush=True)

    source = str(cfg["selection"]["source"])
    if source not in ("validation", "test"):
        raise SystemExit(f"selection.source = {source!r}: known values are 'validation' and 'test'")
    forced = validation_choice(cfg) if source == "validation" else None
    if forced:
        print(f"[selection] ranks taken from the validation split: "
              + ", ".join(f"{k} r = {v}" for k, v in sorted(forced.items())), flush=True)

    table = rank_table(runs, a.objective, ranks)
    write_csv(out / "rank_table.csv", table)
    sel = apply_rule(table, tol, weights, stability, forced)
    write_csv(out / "rank_selection.csv", sel)
    sens = []
    for t_val in [round(x, 4) for x in np.arange(0.0, 0.0501, 0.0025)]:
        sens.extend(apply_rule(table, t_val, weights, stability))
    write_csv(out / "rank_selection_sensitivity.csv", sens)
    lvf = lora_vs_full(runs, a.objective, ranks, sel, cfg)
    write_csv(out / "lora_vs_full_finetuning.csv", lvf)

    (out / "selection_summary.md").write_text(
        summary_md(lora, sel, sens, table, a.objective, refs, tol, weights, lvf), encoding="utf-8")
    print(f"[selection] selection_summary.md written to {out}", flush=True)
    for s in sel:
        what = "full fine-tuning" if s["selected_condition"] == "fullft" else f"r = {s['selected_rank']}"
        print(f"[selection] {s['architecture_display']}: {what}  "
              f"(within 5 % {s['selected_mean_within_5pct']:.2f}, Dice {s['selected_mean_dice_macro']:.3f})"
              f"{'  ** ' + s['note'] if s['note'] else ''}", flush=True)


if __name__ == "__main__":
    main()
