"""Multi-seed aggregation: per-configuration mean/SD over seeds, paired differences, run index with provenance.

Inputs
    * ``<main_output_root>/runs/<base run id>/metrics.json``   seed 42 (outputs/full, read only)
    * ``<output_root>/runs/<run id>/metrics.json``              seeds from configs/multiseed.yaml (outputs/multiseed)
Outputs (``<output_root>/summary/``)
    runs_index.csv          one row per (configuration, seed): which run directory the number comes from, its SHA-256
    config_summary.csv      one row per configuration: n, mean, SD, min, max of every metric
    paired_differences.csv  paired (same-seed) differences with 95% t-intervals
    tables.md               the tables MS-1 ... MS-6 (what each one is for is stated above it)
    multiseed_results.json  everything above in one file
Every number is read from a ``metrics.json``; nothing is typed in. Primary checkpoint = ``evaluation.last`` (last epoch).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

CLASSES = ["discocyte", "echinocyte", "spherocyte", "stomatocyte"]
ARCH_LABEL = {"edge_sam": "EdgeSAM", "mobile_sam": "MobileSAM", "mobilenet_unet": "MobileNet-UNet"}
COLLAPSE_DICE = 0.05          # a class with Dice below this is counted as "not detected" (collapse)
METRICS = ["mean_dice", "aji", "bf1", "ipe_global_pct"] + [f"dice_{c}" for c in CLASSES] + [f"ipe_{c}_pct" for c in CLASSES]
HIGHER_IS_BETTER = {"mean_dice": True, "aji": True, "bf1": True, "ipe_global_pct": False}


# ------------------------------------------------------------------------------------------------ identifiers
def parse_base_id(base_id: str) -> Tuple[str, str]:
    """'edge_sam_lora_r8' -> ('edge_sam', 'lora_r8'); '..._abl_pmc' -> ('..', 'abl_pmc'); '..._abl_pmc_r32'; '..._full_finetune'."""
    m = re.match(r"^(edge_sam|mobile_sam|mobilenet_unet)_(lora_r\d+|full_finetune|abl_[a-z_]+?(?:_r\d+)?)$", base_id)
    if not m:
        raise ValueError(f"cannot parse run id {base_id!r}")
    return m.group(1), m.group(2)


def config_key(base_id: str, sampler: Optional[str]) -> str:
    arch, label = parse_base_id(base_id)
    if sampler and sampler != "shuffle":
        label += "+wsamp"
    return f"{arch}|{label}"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ reading runs
def record_from_metrics(m: Dict[str, Any], ckpt: str = "last") -> Dict[str, Any]:
    ev = m["evaluation"][ckpt]
    rec: Dict[str, Any] = {"status": m["status"], "seed_in_file": m.get("seed"),
                           "epochs_completed": m["training"]["epochs_completed"],
                           "train_seconds": m["training"]["train_seconds"],
                           "trainable_params": m["parameters"]["trainable"], "total_params": m["parameters"]["total"],
                           "mean_dice": ev["mean_dice"], "aji": ev["aji"], "bf1": ev["bf1"],
                           "ipe_global_pct": 100.0 * ev["ipe_global"]}
    for c in CLASSES:
        rec[f"dice_{c}"] = ev[f"dice_{c}"]
        v = ev.get(f"ipe_{c}")
        rec[f"ipe_{c}_pct"] = None if v is None or (isinstance(v, float) and math.isnan(v)) else 100.0 * v
    rec["collapsed_classes"] = [c for c in CLASSES if not rec[f"dice_{c}"] >= COLLAPSE_DICE]
    rec["n_collapsed"] = len(rec["collapsed_classes"])
    return rec


def main_reference_ids(cfg: Dict[str, Any], specs: Sequence[Any]) -> List[str]:
    """Run ids of ``outputs/full`` (seed 42) that the multi-seed tables need.

    Every configuration of the main grid that a block adds seeds to, plus, for an ablation at another rank, the comparison
    runs of that architecture: the full objective at each rank of the block and at the selected rank, and the rank-8
    ablation runs (all of which exist in the main grid).
    """
    from qpi.experiments import build_grid

    main_ids = {s.run_id for s in build_grid({k: v for k, v in cfg.items() if k != "multiseed"}) if s.alias_of is None}
    sel = int(cfg["lora"]["selected_rank"])
    ids = {s.base_id for s in specs if not s.sampler and s.base_id in main_ids}
    for s in specs:
        if s.group == "ablation_rank":
            ids |= {f"{s.arch}_lora_r{s.lora_r}", f"{s.arch}_lora_r{sel}"}
            loss = re.match(rf"^{s.arch}_abl_(.+)_r\d+$", s.base_id).group(1)
            ids.add(f"{s.arch}_abl_{loss}")
    missing_in_main = sorted(i for i in ids if i not in main_ids)
    if missing_in_main:
        raise ValueError(f"not runs of the main grid: {missing_in_main}")
    return sorted(ids)


def collect(cfg: Dict[str, Any], root: Path, max_stage: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Index rows for every expected (configuration, seed); the second value lists missing/unfinished runs."""
    from qpi.experiments import build_grid, up_to_stage     # local import: keeps this module importable without the config stack

    ms = cfg["multiseed"]
    specs = up_to_stage(build_grid(cfg), max_stage)
    main_root = root / ms["main_output_root"]
    out_root = root / cfg["paths"]["output_root"]
    main_seed = int(ms["main_seed"])
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    def add(block: str, base_id: str, sampler: Optional[str], seed: int, run_id: str, run_root: Path, origin: str):
        mp = run_root / "runs" / run_id / "metrics.json"
        row = {"block": block, "config": config_key(base_id, sampler), "base_run_id": base_id, "seed": seed,
               "run_id": run_id, "origin": origin, "metrics_path": str(mp.relative_to(root)) if mp.is_relative_to(root) else str(mp)}
        if not mp.exists():
            missing.append(str(mp))
            rows.append({**row, "status": "missing"})
            return
        m = json.loads(mp.read_text())
        if m.get("seed") is not None and int(m["seed"]) != seed:
            raise ValueError(f"{mp}: seed in file {m['seed']} != expected {seed}")
        if m.get("status") != "complete" or "evaluation" not in m:
            missing.append(f"{mp} (status {m.get('status')})")
            rows.append({**row, "status": m.get("status", "unknown"), "metrics_sha256": sha256_file(mp)})
            return
        rows.append({**row, **record_from_metrics(m), "metrics_sha256": sha256_file(mp)})

    for b in main_reference_ids(cfg, specs):
        add("main_seed42", b, None, main_seed, b, main_root, "outputs/full (single-seed run)")
    for s in specs:
        add(s.block, s.base_id, s.sampler, int(s.seed), s.run_id, out_root, "outputs/multiseed")
    return rows, missing


# ------------------------------------------------------------------------------------------------ statistics
def _vals(rows: List[Dict[str, Any]], cfg_key: str, metric: str) -> Dict[int, float]:
    return {r["seed"]: r[metric] for r in rows
            if r["config"] == cfg_key and r.get("status") == "complete" and r.get(metric) is not None}


def summarise(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for key in sorted({r["config"] for r in rows}):
        rr = [r for r in rows if r["config"] == key and r.get("status") == "complete"]
        arch, label = key.split("|")
        s: Dict[str, Any] = {"config": key, "arch": arch, "label": label, "n": len(rr),
                             "seeds": sorted(r["seed"] for r in rr),
                             "trainable_params": rr[0]["trainable_params"] if rr else None,
                             "n_runs_with_a_collapsed_class": sum(1 for r in rr if r["n_collapsed"] > 0),
                             "collapsed_seeds": sorted(r["seed"] for r in rr if r["n_collapsed"] > 0)}
        for m in METRICS:
            v = np.array([r[m] for r in rr if r.get(m) is not None], float)
            s[f"{m}_mean"] = float(v.mean()) if len(v) else None
            s[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
            s[f"{m}_min"] = float(v.min()) if len(v) else None
            s[f"{m}_max"] = float(v.max()) if len(v) else None
        out.append(s)
    return out


def paired(rows: List[Dict[str, Any]], a: str, b: str, metric: str) -> Dict[str, Any]:
    """b - a over the seeds both configurations have; 95% CI from the t distribution of the differences."""
    va, vb = _vals(rows, a, metric), _vals(rows, b, metric)
    seeds = sorted(set(va) & set(vb))
    d = np.array([vb[s] - va[s] for s in seeds], float)
    res: Dict[str, Any] = {"a": a, "b": b, "metric": metric, "difference": "b - a", "n_pairs": len(d), "seeds": seeds}
    if len(d) == 0:
        return {**res, "mean_diff": None, "sd_diff": None, "ci95_lo": None, "ci95_hi": None, "n_b_greater": 0}
    res["mean_diff"] = float(d.mean())
    res["n_b_greater"] = int((d > 0).sum())
    if len(d) > 1:
        sd = float(d.std(ddof=1))
        h = float(stats.t.ppf(0.975, len(d) - 1) * sd / math.sqrt(len(d)))
        res.update(sd_diff=sd, ci95_lo=res["mean_diff"] - h, ci95_hi=res["mean_diff"] + h)
    else:
        res.update(sd_diff=None, ci95_lo=None, ci95_hi=None)
    return res


def _ci_contains_zero(p: Dict[str, Any]) -> Optional[bool]:
    if p["ci95_lo"] is None:
        return None
    return p["ci95_lo"] <= 0.0 <= p["ci95_hi"]


def rank_selection(rows: List[Dict[str, Any]], arch: str, ranks: Sequence[int], selected: int) -> Dict[str, Any]:
    """Descriptive, rule-based statement about the rank sweep (the rule is fixed here, before looking at results):

    for each headline metric: best rank by mean; ranks whose paired difference to the best has a 95% interval
    containing 0 are 'not distinguishable from the best'; the smallest such rank is reported.
    """
    keys = {r: f"{arch}|lora_r{r}" for r in ranks}
    out: Dict[str, Any] = {"arch": arch, "selected_rank": selected, "per_metric": {}}
    for m, hib in HIGHER_IS_BETTER.items():
        means = {}
        for r, k in keys.items():
            v = list(_vals(rows, k, m).values())
            if v:
                means[r] = float(np.mean(v))
        if not means:
            continue
        best = (max if hib else min)(means, key=means.get)
        indist = []
        for r in ranks:
            if r not in means:
                continue
            if r == best:
                indist.append(r)
                continue
            p = paired(rows, keys[best], keys[r], m)
            z = _ci_contains_zero(p)
            if z:
                indist.append(r)
        out["per_metric"][m] = {"best_rank_by_mean": best, "means": means, "not_distinguishable_from_best": sorted(indist),
                                "smallest_not_distinguishable": min(indist) if indist else None,
                                "selected_rank_not_distinguishable": selected in indist}
    return out


# ------------------------------------------------------------------------------------------------ formatting
def pm(mean: Optional[float], sd: Optional[float], nd: int = 3) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.{nd}f}" if sd is None else f"{mean:.{nd}f} ± {sd:.{nd}f}"


def _row(s: Dict[str, Any], cols: Sequence[Tuple[str, int]]) -> str:
    cells = [pm(s.get(f"{m}_mean"), s.get(f"{m}_sd"), nd) for m, nd in cols]
    coll = f"{s['n_runs_with_a_collapsed_class']}/{s['n']}"
    return f"| {s['label']} | {s['n']} | " + " | ".join(cells) + f" | {coll} |"


HEAD_COLS = [("mean_dice", 3), ("aji", 3), ("bf1", 3), ("ipe_global_pct", 2), ("dice_echinocyte", 3), ("dice_stomatocyte", 3)]
HEAD = "| Configuration | n | Mean Dice | AJI | BF1 | Global IPE (%) | Dice echi | Dice stom | Runs with a missing class |\n|---|---|---|---|---|---|---|---|---|"


def _fmt_paired(p: Dict[str, Any], nd: int = 3) -> str:
    if p["mean_diff"] is None:
        return "n/a"
    ci = "" if p["ci95_lo"] is None else f" [{p['ci95_lo']:+.{nd}f}, {p['ci95_hi']:+.{nd}f}]"
    return f"{p['mean_diff']:+.{nd}f}{ci} ({p['n_b_greater']}/{p['n_pairs']} seeds higher)"


def build_tables(cfg: Dict[str, Any], rows: List[Dict[str, Any]], summ: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    by = {s["config"]: s for s in summ}
    sel = int(cfg["lora"]["selected_rank"])
    ranks = list(cfg["lora"]["ranks"])
    md: List[str] = []
    js: Dict[str, Any] = {}
    paired_rows: List[Dict[str, Any]] = []

    def present(arch: str) -> bool:           # an architecture whose stage has not run yet gets no section
        return any(k.startswith(arch + "|") for k in by)

    def P(a, b, m):
        p = paired(rows, a, b, m)
        paired_rows.append(p)
        return p

    # ---- MS-1 rank sweep (every architecture that has one)
    js["rank_selection"] = {}
    for arch in ("edge_sam", "mobile_sam", "mobilenet_unet"):
        if sum(f"{arch}|lora_r{r}" in by and by[f"{arch}|lora_r{r}"]["n"] > 1 for r in ranks) < 2:
            continue      # a rank sweep needs at least two ranks with several seeds (each architecture has them once its stage has run)
        md += [f"## MS-1  LoRA rank sweep, {ARCH_LABEL[arch]} (full physics-aware loss)",
               "Purpose: is rank 8 a defensible choice? Mean ± SD over seeds (42 from outputs/full; others from outputs/multiseed). "
               "Last-epoch checkpoint. 'Runs with a missing class' = runs in which at least one class has Dice < "
               f"{COLLAPSE_DICE}.", "", HEAD]
        for r in ranks:
            k = f"{arch}|lora_r{r}"
            if k in by:
                md.append(_row(by[k], HEAD_COLS).replace(f"| lora_r{r} |", f"| r={r} |", 1))
        md += ["", f"Paired differences to r={sel} (rank r minus rank {sel}, same seed; mean [95% t-interval] (seeds in which r is higher)):", "",
               "| Rank | Mean Dice | AJI | BF1 | Global IPE (pp) |", "|---|---|---|---|---|"]
        for r in ranks:
            if r == sel:
                continue
            cells = [_fmt_paired(P(f"{arch}|lora_r{sel}", f"{arch}|lora_r{r}", m), 2 if m == "ipe_global_pct" else 3)
                     for m in ("mean_dice", "aji", "bf1", "ipe_global_pct")]
            md.append(f"| r={r} | " + " | ".join(cells) + " |")
        rs = rank_selection(rows, arch, ranks, sel)
        js["rank_selection"][arch] = rs
        md += ["", "Rule (fixed in qpi/analysis/multiseed.py before the runs): per metric, the best rank by mean is found; ranks whose "
               "paired difference to the best has a 95% interval containing 0 are 'not distinguishable from the best'.", ""]
        for m, v in rs["per_metric"].items():
            md.append(f"* {m}: best by mean r={v['best_rank_by_mean']}; not distinguishable from best: "
                      f"{v['not_distinguishable_from_best']}; selected r={sel} {'IS' if v['selected_rank_not_distinguishable'] else 'is NOT'} among them.")
        md.append("")

    # ---- MS-2 architectures
    md += ["## MS-2  Architectures at rank 8 (full physics-aware loss)",
           "Purpose: which architecture is best, and is the gap larger than the run-to-run spread?", "", HEAD]
    for a in ("mobilenet_unet", "mobile_sam", "edge_sam"):
        k = f"{a}|lora_r{sel}"
        if k in by:
            md.append(_row(by[k], HEAD_COLS).replace(f"| lora_r{sel} |", f"| {ARCH_LABEL[a]} |", 1))
    md += ["", "Paired differences (first minus second, same seed):", "", "| Comparison | Mean Dice | AJI | BF1 | Global IPE (pp) |", "|---|---|---|---|---|"]
    for a, b in (("edge_sam", "mobile_sam"), ("edge_sam", "mobilenet_unet")):
        if not (present(a) and present(b)):
            continue
        cells = [_fmt_paired(P(f"{a}|lora_r{sel}", f"{b}|lora_r{sel}", m), 2 if m == "ipe_global_pct" else 3)
                 for m in ("mean_dice", "aji", "bf1", "ipe_global_pct")]
        md.append(f"| {ARCH_LABEL[b]} minus {ARCH_LABEL[a]} | " + " | ".join(cells) + " |")

    # ---- MS-4 loss ablation
    md += ["", "## MS-4  Physics-aware loss ablation (rank 8)",
           "Purpose: which effects of PMC, BGA and IPP are reproducible across seeds? 'Full' = the rank-8 run of the sweep.", ""]
    for a in ("edge_sam", "mobile_sam", "mobilenet_unet"):
        if not present(a):
            continue
        md += [f"### {ARCH_LABEL[a]}", "", HEAD]
        for lab, nm in (("abl_dice_only", "Dice only"), ("abl_pmc", "+ PMC"), ("abl_pmc_bga", "+ PMC + BGA"), (f"lora_r{sel}", "+ PMC + BGA + IPP (full)")):
            k = f"{a}|{lab}"
            if k in by:
                md.append(_row(by[k], HEAD_COLS).replace(f"| {lab} |", f"| {nm} |", 1))
        md += ["", "Paired differences, full objective minus Dice only: " + "; ".join(
            f"{nm} {_fmt_paired(P(f'{a}|abl_dice_only', f'{a}|lora_r{sel}', m), 2 if m == 'ipe_global_pct' else 3)}"
            for m, nm in (("mean_dice", "Dice"), ("bf1", "BF1"), ("aji", "AJI"), ("ipe_global_pct", "IPE (pp)"))), ""]

    # ---- MS-6 loss ablation by rank
    blocks = [b for b in cfg["multiseed"]["blocks"] if "ablation_by_rank" in b]
    if blocks:
        md += ["## MS-6  Physics-aware loss ablation by LoRA rank",
               "Purpose: does the effect of PMC, BGA and IPP depend on the LoRA rank? Every rank has the seeds of the multi-seed plan "
               "(42, 43, 44); differences are paired by seed. 'Full' = the rank-sweep run of that rank.", ""]
        for blk in blocks:
            ab = blk["ablation_by_rank"]
            rks = sorted({int(r) for r in ab["ranks"]} | {sel})
            for a_ in ab["archs"]:
                if not present(a_):
                    continue
                md += [f"### {ARCH_LABEL[a_]}", "", HEAD.replace("| Configuration |", "| Rank, configuration |")]
                for r in rks:
                    for lab, nm in (("abl_dice_only", "Dice only"), ("abl_pmc", "+ PMC"), ("abl_pmc_bga", "+ PMC + BGA"), ("lora", "full")):
                        k = (f"{a_}|lora_r{r}" if lab == "lora" else f"{a_}|{lab}" + ("" if r == sel else f"_r{r}"))
                        if k in by:
                            row = {**by[k], "label": f"r={r}, {nm}"}
                            md.append(_row(row, HEAD_COLS))
                md += ["", "Paired differences within a rank (seed-matched; Dice, global IPE in percentage points):", "",
                       "| Rank | Full minus Dice only | Full minus (+ PMC + BGA) = effect of IPP |", "|---|---|---|"]
                for r in rks:
                    d0 = f"{a_}|abl_dice_only" + ("" if r == sel else f"_r{r}")
                    d2 = f"{a_}|abl_pmc_bga" + ("" if r == sel else f"_r{r}")
                    fl = f"{a_}|lora_r{r}"
                    cells = []
                    for ref in (d0, d2):
                        cells.append("; ".join(f"{nm} {_fmt_paired(P(ref, fl, m), 2 if m == 'ipe_global_pct' else 3)}"
                                               for m, nm in (("mean_dice", "Dice"), ("ipe_global_pct", "IPE"))))
                    md.append(f"| r={r} | " + " | ".join(cells) + " |")
                md.append("")

    # ---- MS-5 sampler
    k0, k1 = f"edge_sam|lora_r{sel}", f"edge_sam|lora_r{sel}+wsamp"
    if k1 in by:
        md += ["## MS-5  Sampler: plain shuffling vs weighted sampler (EdgeSAM, r=8)",
               "Purpose: the original manuscript describes a weighted sampler; this tests whether it matters. Weighted = inverse frequency of the "
               "image's dominant foreground class, sampling with replacement (DECISIONS.md D-18).", "", HEAD]
        md.append(_row(by[k0], HEAD_COLS).replace(f"| lora_r{sel} |", "| Plain shuffling |", 1))
        md.append(_row(by[k1], HEAD_COLS).replace(f"| lora_r{sel}+wsamp |", "| Weighted sampler |", 1))
        md += ["", "Paired differences (weighted minus shuffling): " + "; ".join(
            f"{nm} {_fmt_paired(P(k0, k1, m), 2 if m == 'ipe_global_pct' else 3)}"
            for m, nm in (("mean_dice", "Dice"), ("bf1", "BF1"), ("aji", "AJI"), ("ipe_global_pct", "IPE (pp)"),
                          ("dice_echinocyte", "Dice echi"), ("dice_stomatocyte", "Dice stom"))), ""]
    js["paired"] = paired_rows
    return "\n".join(md) + "\n", js


def check_prereqs(cfg: Dict[str, Any], root: Path, max_stage: Optional[int] = None) -> List[str]:
    """Problems that would make the new seeds incomparable with the seed-42 runs (empty list = fine).

    The seed-42 runs are only usable as the first seed of each configuration if they were trained on the same processed
    dataset and with the same base/experiment configuration, and if every seed-42 run needed here is complete.
    """
    from qpi.experiments import build_grid, up_to_stage

    ms = cfg["multiseed"]
    main_root = root / ms["main_output_root"]
    problems: List[str] = []
    meta_p = main_root / "run_metadata.json"
    if not meta_p.exists():
        return [f"{meta_p} missing: run the main experiment first"]
    meta = json.loads(meta_p.read_text())
    man = root / cfg["paths"]["processed_root"] / "manifest.csv"
    if not man.exists():
        problems.append(f"{man} missing: the processed dataset is needed")
    elif meta.get("data_manifest_sha256") != sha256_file(man):
        problems.append("data/processed/manifest.csv differs from the one the seed-42 runs were trained on "
                        f"({meta.get('data_manifest_sha256')} vs {sha256_file(man)})")
    for name in ("base.yaml", "experiments.yaml"):
        cur = sha256_file(root / "configs" / name)
        if meta.get("config_file_sha256", {}).get(name) != cur:
            problems.append(f"configs/{name} changed since the seed-42 runs (hash differs)")
    for b in main_reference_ids(cfg, up_to_stage(build_grid(cfg), max_stage)):
        mp = main_root / "runs" / b / "metrics.json"
        if not mp.exists() or json.loads(mp.read_text()).get("status") != "complete":
            problems.append(f"seed-42 run {b} is missing or not complete ({mp})")
    return problems


# ------------------------------------------------------------------------------------------------ driver
def write_outputs(cfg: Dict[str, Any], root: Path, allow_partial: bool = False, max_stage: Optional[int] = None) -> int:
    rows, missing = collect(cfg, root, max_stage)
    out = root / cfg["paths"]["output_root"] / "summary"
    out.mkdir(parents=True, exist_ok=True)
    summ = summarise(rows)
    md, js = build_tables(cfg, rows, summ)
    stage_note = "all stages" if max_stage is None else f"stages 1 to {max_stage}"
    cols = ["block", "config", "base_run_id", "seed", "run_id", "origin", "metrics_path", "metrics_sha256", "status",
            "epochs_completed", "train_seconds", "trainable_params"] + METRICS + ["n_collapsed", "collapsed_classes"]
    with open(out / "runs_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "collapsed_classes": " ".join(r.get("collapsed_classes", []))})
    scols = ["config", "arch", "label", "n", "seeds", "trainable_params", "n_runs_with_a_collapsed_class", "collapsed_seeds"] + \
            [f"{m}_{q}" for m in METRICS for q in ("mean", "sd", "min", "max")]
    with open(out / "config_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=scols, extrasaction="ignore")
        w.writeheader()
        for s in summ:
            w.writerow({**s, "seeds": " ".join(map(str, s["seeds"])), "collapsed_seeds": " ".join(map(str, s["collapsed_seeds"]))})
    pcols = ["a", "b", "metric", "difference", "n_pairs", "seeds", "mean_diff", "sd_diff", "ci95_lo", "ci95_hi", "n_b_greater"]
    with open(out / "paired_differences.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pcols, extrasaction="ignore")
        w.writeheader()
        for p in js["paired"]:
            w.writerow({**p, "seeds": " ".join(map(str, p["seeds"]))})
    header = ["# Multi-seed results", "", f"Scope: {stage_note}.", "", f"Missing or unfinished runs: {len(missing)}" + ("" if not missing else " (partial tables)"), ""]
    header += [f"* {m}" for m in missing[:60]] + [""] if missing else []
    (out / "tables.md").write_text("\n".join(header) + "\n" + md)
    result = {"collapse_dice_threshold": COLLAPSE_DICE, "checkpoint": "last", "n_runs_indexed": len(rows), "missing": missing,
              "config_summary": summ, **js}
    (out / "multiseed_results.json").write_text(json.dumps(result, indent=1, default=float))
    man = root / cfg["paths"]["manuscript_outputs"]
    man.mkdir(parents=True, exist_ok=True)
    (man / "multiseed_results.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"[multiseed] {len(rows)} runs indexed, {len(missing)} missing/unfinished -> {out}")
    return 0 if (allow_partial or not missing) else 1
