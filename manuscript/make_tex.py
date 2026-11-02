#!/usr/bin/env python
"""Write a short LaTeX report of the multi-seed results, with the complete tables (every architecture, rank, loss, seed and metric)
in an appendix.

Every number is read from results.json (written by manuscript/results.py from the run outputs); the report warns when results.json
is older than the run table it was built from. Page 1 to 4: how to read the numbers, key points (computed sentences), two plots, and compact tables. Appendix: the full tables.

Every number comes from manuscript/outputs/results.json (tables ms_runs, ms_summary, ms_paired and ms_meta). The ranks, seeds,
losses, architectures, class names, the checkpoint and the collapse threshold are read from ms_meta. Nothing is typed in here
except display names for the keys (for example "dice_only" -> "Dice only") and the number of decimals.

    python manuscript/make_tex.py                       # manuscript/outputs/results.json -> manuscript/outputs/multiseed_results.tex
    python manuscript/make_tex.py --pdf                 # also compiles it (pdflatex, xelatex, lualatex or tectonic)
    python manuscript/make_tex.py --no-appendix         # only the short report
    python manuscript/make_tex.py --results path.json --out path.tex
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Display names only. A key that is not listed is printed as it is (escaped).
ARCH_NAMES = {"edge_sam": "EdgeSAM", "mobile_sam": "MobileSAM", "mobilenet_unet": "MobileNet-UNet"}
LOSS_NAMES = {"dice_only": "Dice only", "pmc": "+PMC", "pmc_bga": "+PMC+BGA", "full": "Full (+IPP)"}
CLASS_SHORT = {"discocyte": "Disco", "spherocyte": "Sphero", "echinocyte": "Echi", "stomatocyte": "Stoma"}
DEC_DICE = 3   # Dice, AJI, BF1
DEC_IPE = 2    # IPE in percent


def esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def num(x, dec: int) -> str:
    if x is None:
        return "--"
    t = f"{float(x):.{dec}f}"
    return t.replace("-", "$-$")


def pm(mean, sd, dec: int) -> str:
    if mean is None:
        return "--"
    if sd is None:
        return num(mean, dec)
    return f"{num(mean, dec)} $\\pm$ {float(sd):.{dec}f}"


def params_m(x) -> str:
    return "--" if x is None else f"{float(x) / 1e6:.2f}"


METRIC_NAMES = {"mean_dice": "Dice", "aji": "AJI", "bf1": "BF1", "ipe_global_pct": "IPE (pp)"}


def cmp_label(x) -> str:
    """'full_r16' (a rank comparison inside the full objective) -> 'Full, r = 16'; a loss key -> its display name."""
    x = str(x)
    if x in LOSS_NAMES:
        return LOSS_NAMES[x]
    head, sep, tail = x.rpartition("_r")
    if sep and tail.isdigit() and head in LOSS_NAMES:
        return f"{LOSS_NAMES[head]}, r = {tail}"
    return esc(x)


def arch_name(a) -> str:
    return ARCH_NAMES.get(a, esc(a))


def loss_name(l) -> str:
    return LOSS_NAMES.get(l, esc(l))


def table(caption: str, label: str, cols: str, head: str, rows: list[str], size: str = "footnotesize") -> str:
    """A longtable that can break over pages."""
    out = [f"{{\\{size}", f"\\begin{{longtable}}{{{cols}}}", f"\\caption{{{caption}}}\\label{{{label}}}\\\\", "\\toprule", head + " \\\\",
           "\\midrule", "\\endfirsthead", "\\toprule", head + " \\\\", "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
    out += [r + " \\\\" for r in rows]
    out += ["\\end{longtable}", "}"]
    return "\n".join(out)


class Data:
    def __init__(self, path: Path):
        d = json.loads(path.read_text())
        t = d["tables"]
        for k in ("ms_runs", "ms_summary", "ms_paired", "ms_meta"):
            if k not in t:
                sys.exit(f"{path} has no tables.{k}; run manuscript/results.py first")
        self.meta = d.get("meta", {})
        self.ms = t["ms_meta"]
        self.runs = t["ms_runs"]
        self.summary = t["ms_summary"]
        self.paired = t["ms_paired"]
        self.archs = self.ms["archs"]
        self.ranks = self.ms["ranks"]
        self.seeds = self.ms["seeds"]
        self.losses = self.ms["losses"]
        self.classes = self.ms["class_names_by_mask_value"]
        self.thr = self.ms["collapse_dice_threshold"]
        self._s = {(r["arch"], r["loss"], r["rank"]): r for r in self.summary}
        self._r = {(r["arch"], r["loss"], r["rank"], r["seed"]): r for r in self.runs}
        self.full = "full" if "full" in self.losses else self.losses[-1]
        self.bench = t.get("table6_benchmark", [])
        self.bench_seeds = t.get("table6_benchmark_per_seed", [])
        self.bench_protocol = (d.get("benchmark") or {}).get("protocol") or {}
        self.host = (d.get("benchmark") or {}).get("host_details") or {}
        self.equiv = t.get("table_onnx_equivalence", [])
        self.equiv_acc = t.get("onnx_equivalence_acceptance") or {}

    def s(self, arch, loss, rank):
        return self._s.get((arch, loss, rank))

    def r(self, arch, loss, rank, seed):
        return self._r.get((arch, loss, rank, seed))


def k_of_n(s) -> str:
    if s is None:
        return "--"
    k = s["n_runs_with_a_collapsed_class"]
    seeds = ", ".join(str(x) for x in s.get("collapsed_seeds", []))
    return f"{k}/{s['n']}" + (f" (seed {seeds})" if k else "")


def header_section(D: Data, src: Path) -> str:
    m = D.meta
    lines = [
        r"\section*{Scope and conventions}",
        r"\begin{itemize}\itemsep0pt",
        f"\\item Source: \\texttt{{{esc(src.name)}}} (generated {esc(m.get('generated_utc', 'n/a'))}). "
        f"{D.ms['n_runs']} LoRA runs: {len(D.archs)} architectures $\\times$ {len(D.ranks)} ranks $\\times$ {len(D.losses)} losses "
        f"$\\times$ {len(D.seeds)} seeds ({', '.join(str(x) for x in D.seeds)}).",
        f"\\item Checkpoint: {esc(D.ms['checkpoint'])} epoch. Validation images: {esc(m.get('n_validation_images', 'n/a'))}. "
        "Input: 32-bit phase maps in radians (no 8-bit conversion).",
        f"\\item Losses: {', '.join(loss_name(l) for l in D.losses)}. Ranks: {', '.join(str(x) for x in D.ranks)}.",
        f"\\item A run \\emph{{lost a class}} when the Dice of any class is below {D.thr}. Such runs are counted in every mean, none is excluded.",
        "\\item Mean $\\pm$ SD over seeds (sample SD, $n$ as stated). IPE is the global integrated phase error in percent. "
        "Per-class columns follow the mask-value order: "
        + ", ".join(esc(c) for c in D.classes) + " (mask values "
        + ", ".join(f"{i + 1} = {esc(c)}" for i, c in enumerate(D.classes)) + ").",
        r"\item Paired differences are computed seed by seed (same seed in both runs), with a 95\% $t$-interval; with three seeds the interval is wide "
        r"and is given for description only, not as a significance test.",
        r"\end{itemize}",
    ]
    return "\n".join(lines)


def rank_sweep(D: Data) -> str:
    out = [r"\section{Rank sweep, full objective}"]
    for a in D.archs:
        seeds_h = " & ".join(f"Dice s{sd}" for sd in D.seeds) + " & " + " & ".join(f"IPE s{sd}" for sd in D.seeds)
        head = "Rank & Params (M) & Dice & AJI & BF1 & IPE (\\%) & Runs that lost a class & " + seeds_h
        rows = []
        for rk in D.ranks:
            s = D.s(a, D.full, rk)
            if s is None:
                continue
            dice_seed, ipe_seed = [], []
            for sd in D.seeds:
                r = D.r(a, D.full, rk, sd)
                if r is None:
                    dice_seed.append("--"); ipe_seed.append("--"); continue
                mark = "$^\\dagger$" if r["collapsed"] else ""
                dice_seed.append(num(r["mean_dice"], DEC_DICE) + mark)
                ipe_seed.append(num(r["ipe_global_pct"], DEC_IPE) + mark)
            rows.append(" & ".join([str(rk), params_m(s["trainable_params"]),
                                    pm(s["mean_dice_mean"], s["mean_dice_sd"], DEC_DICE), pm(s["aji_mean"], s["aji_sd"], DEC_DICE),
                                    pm(s["bf1_mean"], s["bf1_sd"], DEC_DICE), pm(s["ipe_global_pct_mean"], s["ipe_global_pct_sd"], DEC_IPE),
                                    k_of_n(s)] + dice_seed + ipe_seed))
        cols = "r r l l l l l " + "r " * (2 * len(D.seeds))
        out.append(table(f"{arch_name(a)}: LoRA rank sweep, {loss_name(D.full)} objective. $^\\dagger$ = run lost a class (Dice $<$ {D.thr}).",
                         f"tab:rank_{a}", cols.strip(), head, rows))
    return "\n".join(out)


def ablation(D: Data) -> str:
    out = [r"\section{Loss ablation at every rank}"]
    head = "Rank & Loss & Params (M) & Dice & AJI & BF1 & IPE (\\%) & Runs that lost a class"
    for a in D.archs:
        rows = []
        for rk in D.ranks:
            for l in D.losses:
                s = D.s(a, l, rk)
                if s is None:
                    continue
                rows.append(" & ".join([str(rk), loss_name(l), params_m(s["trainable_params"]),
                                        pm(s["mean_dice_mean"], s["mean_dice_sd"], DEC_DICE), pm(s["aji_mean"], s["aji_sd"], DEC_DICE),
                                        pm(s["bf1_mean"], s["bf1_sd"], DEC_DICE), pm(s["ipe_global_pct_mean"], s["ipe_global_pct_sd"], DEC_IPE),
                                        k_of_n(s)]))
        out.append(table(f"{arch_name(a)}: all losses at all ranks (mean $\\pm$ SD over seeds).", f"tab:abl_{a}", "r l r l l l l l", head, rows))
    return "\n".join(out)


RUNTIME = {"pytorch": "PyTorch", "onnx": "ONNX Runtime"}


def span(vals, dec: int) -> str:
    """Smallest and largest of the per-seed values ('42' when they agree at this precision)."""
    v = [float(x) for x in vals if x is not None]
    if not v:
        return "--"
    lo, hi = f"{min(v):.{dec}f}", f"{max(v):.{dec}f}"
    return lo if lo == hi else f"{lo}--{hi}"


def benchmark_section(D: Data) -> str:
    """Inference benchmark (protocol v5): FP32, PyTorch and ONNX Runtime; P50, P99, FPS and Peak VRAM, per seed and as mean over seeds."""
    if not D.bench:
        return ""
    P = D.bench_protocol
    seeds = ", ".join(str(x) for x in P.get("seeds", []))
    summary, last = [], None
    for r in D.bench:
        name = esc(r["architecture"]) if r["architecture"] != last else ""
        last = r["architecture"]
        f = lambda k, dec: pm(r.get(k), r.get(k + "_sd"), dec)
        summary.append(" & ".join([name, RUNTIME.get(r["backend"], r["backend"]), str(r["input_size"]), f("latency_p50_ms", 2),
                                   f("latency_p99_ms", 2), span(r.get("fps_per_seed", []), 1), span(r.get("peak_vram_mb_per_seed", []), 0)]))
    cap = ("Inference benchmark on %s: FP32, batch 1, model forward only, trained reference checkpoint with LoRA merged (rank independent), "
           "each model at its deployment input size; %s warm-up and %s timed forward passes per seed; accuracy is never computed inside a timed "
           "loop. P50 and P99: mean $\\pm$ SD over the seeds %s of the per-seed values; FPS and Peak VRAM: smallest--largest per-seed value. P50 and P99 are percentiles of the per-inference latencies of one seed "
           "(latency samples of different seeds are not pooled; the timed runs of the %s fresh ONNX Runtime sessions of a seed are pooled). "
           "FPS = 1000 / P50. Peak VRAM: GPU memory used at the device after the timed loop minus before the model was placed (CUDA context "
           "excluded; ONNX Runtime: largest value over the sessions)." % (
               esc(D.host.get("gpu_name") or "the benchmark GPU"), P.get("warmup_iterations", "?"), P.get("timed_iterations", "?"), seeds,
               P.get("ort_sessions_per_seed", "?")))
    out = [r"\section{Inference benchmark}",
           table(cap, "tab:bench", "lllllll", "Model & Runtime & Input (px) & P50 (ms) & P99 (ms) & FPS (range) & Peak VRAM (MB, range)", summary,
                 size="scriptsize")]
    per, last = [], None
    for r in D.bench_seeds:
        name = esc(r["architecture"]) if r["architecture"] != last else ""
        last = r["architecture"]
        per.append(" & ".join([name, RUNTIME.get(r["backend"], r["backend"]), str(r["seed"]), num(r["latency_p50_ms"], 2),
                               num(r["latency_p99_ms"], 2), num(r["fps"], 1), num(r["peak_vram_mb"], 0)]))
    if per:
        out.append(table("Inference benchmark, per seed (same protocol and definitions as the summary table).", "tab:bench_seeds", "lllllll",
                         "Model & Runtime & Seed & P50 (ms) & P99 (ms) & FPS & Peak VRAM (MB)", per, size="scriptsize"))
    return "\n\n".join(out)


def equivalence_section(D: Data) -> str:
    """ONNX Runtime FP32 against PyTorch FP32 with the trained checkpoints: logits, Dice and per-image dry mass (tables onnx_equivalence)."""
    if not D.equiv:
        return ""
    a = D.equiv_acc
    rows = []
    for r in D.equiv:
        ok = r["quality_verdict"]
        rows.append(" & ".join([esc(r["architecture"]), num(r["argmax_agreement_pct"], 4), "%.3g" % r["max_abs_logit_diff"],
                                num(r["quality_argmax_agreement_pct"], 4), num(r["mean_dice_pytorch"], 4), num(r["quality_mean_dice"], 4),
                                "%+.1e" % r["quality_mean_dice_change"],
                                "%.2g" % r["quality_dry_mass_mean_abs_rel_error"], "%.2g" % r["quality_dry_mass_p99_rel_error"],
                                str(r["quality_n_images"]), esc(ok)]))
    cap = ("ONNX Runtime FP32 against PyTorch FP32 with the trained rank-%s checkpoints. Logits: arg-max agreement and largest absolute "
           "logit difference on %s real validation maps. Validation images (rotation 0): arg-max agreement, mean Dice against the annotation "
           "of both models and its change, and the per-image total dry-mass error relative to the PyTorch prediction (mean and 99th "
           "percentile over images). Accepted: agreement $\\geq$ %s\\%%, $|\\Delta$Dice$|$ $\\leq$ %s, mean dry-mass error $\\leq$ %s, "
           "99th percentile $\\leq$ %s (fixed before the first trial)." % (
               D.equiv[0]["rank"], D.equiv[0]["n_inputs"], a.get("argmax_agreement_pct_min"), a.get("mean_dice_abs_change_max"),
               a.get("dry_mass_mean_abs_rel_error_max"), a.get("dry_mass_p99_rel_error_max")))
    return table(cap, "tab:onnxequiv", "lllllllllll",
                 "Model & Logit agr. (\\%) & Max $|\\Delta$logit$|$ & Image agr. (\\%) & Dice PyTorch & Dice ONNX & $\\Delta$Dice & "
                 "Dry mass mean err. & Dry mass p99 err. & Images & Verdict", rows, size="scriptsize")


def per_class(D: Data) -> str:
    out = [r"\section{Per-class Dice and per-class IPE}"]
    cdice = " & ".join(f"Dice {esc(CLASS_SHORT.get(c, c))}" for c in D.classes)
    cipe = " & ".join(f"IPE {esc(CLASS_SHORT.get(c, c))} (\\%)" for c in D.classes)
    head = "Rank & Loss & " + cdice + " & " + cipe
    for a in D.archs:
        rows = []
        for rk in D.ranks:
            for l in D.losses:
                s = D.s(a, l, rk)
                if s is None:
                    continue
                cells = [pm(s.get(f"dice_{c}_mean"), s.get(f"dice_{c}_sd"), DEC_DICE) for c in D.classes]
                cells += [pm(s.get(f"ipe_{c}_pct_mean"), s.get(f"ipe_{c}_pct_sd"), DEC_IPE) for c in D.classes]
                rows.append(" & ".join([str(rk), loss_name(l)] + cells))
        cols = "r l " + "l " * (2 * len(D.classes))
        out.append(table(f"{arch_name(a)}: per-class Dice and per-class IPE (mean $\\pm$ SD over seeds). "
                         "A class IPE far above 100\\% means over-estimation of that class.", f"tab:cls_{a}", cols.strip(), head, rows, "scriptsize"))
    return "\n".join(out)


def paired(D: Data) -> str:
    out = [r"\section{Paired differences (same seed in both runs)}"]
    head = "Rank & Comparison (b $-$ a) & Metric & Mean difference & 95\\% $t$-interval & b higher in"
    for a in D.archs:
        rows = []
        for p in D.paired:
            if p["arch"] != a:
                continue
            dec = DEC_IPE if "ipe" in p["metric"] else DEC_DICE
            rk = "--" if p.get("rank") is None else str(p["rank"])
            rows.append(" & ".join([rk, f"{cmp_label(p['b'])} $-$ {cmp_label(p['a'])}",
                                    METRIC_NAMES.get(p["metric"], esc(p["metric"])), num(p["mean_diff"], dec),
                                    f"[{num(p['ci95_lo'], dec)}, {num(p['ci95_hi'], dec)}]", f"{p['n_b_greater']} of {p['n_pairs']} seeds"]))
        out.append(table(f"{arch_name(a)}: paired differences. IPE in percentage points, Dice and AJI as fractions.", f"tab:pair_{a}",
                         "r l l r l l", head, rows))
    return "\n".join(out)


def all_runs(D: Data) -> str:
    out = [r"\section{Every run}"]
    cdice = " & ".join(f"D {esc(CLASS_SHORT.get(c, c))}" for c in D.classes)
    cipe = " & ".join(f"I {esc(CLASS_SHORT.get(c, c))}" for c in D.classes)
    head = "Arch & Loss & Rank & Seed & Dice & AJI & BF1 & IPE & " + cdice + " & " + cipe + " & Min class Dice & Lost classes"
    rows = []
    for a in D.archs:
        for l in D.losses:
            for rk in D.ranks:
                for sd in D.seeds:
                    r = D.r(a, l, rk, sd)
                    if r is None:
                        continue
                    cells = [num(r.get(f"dice_{c}"), DEC_DICE) for c in D.classes] + [num(r.get(f"ipe_{c}_pct"), DEC_IPE) for c in D.classes]
                    lost = esc(r["collapsed_classes"]) if r.get("collapsed_classes") else ""
                    rows.append(" & ".join([arch_name(a), loss_name(l), str(rk), str(sd), num(r["mean_dice"], DEC_DICE), num(r["aji"], DEC_DICE),
                                            num(r["bf1"], DEC_DICE), num(r["ipe_global_pct"], DEC_IPE)] + cells
                                           + [num(r["min_class_dice"], DEC_DICE), lost]))
    cols = "l l r r " + "r " * 4 + "r " * (2 * len(D.classes)) + "r l"
    out.append(table("All runs, last epoch. D = Dice of the class, I = IPE of the class in percent, IPE = global IPE in percent.",
                     "tab:runs", cols, head, rows, "tiny"))
    return "\n".join(out)



# ----------------------------------------------------------------------------------------------------------------------------
# short report: computed sentences, plots, compact tables
# ----------------------------------------------------------------------------------------------------------------------------
def lost_marks(s) -> str:
    return "" if not s or not s["n_runs_with_a_collapsed_class"] else f"$^{{{s['n_runs_with_a_collapsed_class']}}}$"


def key_points(D: Data) -> str:
    items = []
    for a in D.archs:
        sums = [D.s(a, D.full, rk) for rk in D.ranks if D.s(a, D.full, rk)]
        if not sums:
            continue
        best = max(sums, key=lambda x: x["mean_dice_mean"])
        lo = min(sums, key=lambda x: x["mean_dice_mean"])
        n_full = sum(x["n"] for x in sums)
        k_full = sum(x["n_runs_with_a_collapsed_class"] for x in sums)
        n_all = n_loss = 0
        k_all = 0
        cls_count: dict = {}
        for r in D.runs:
            if r["arch"] != a:
                continue
            n_all += 1
            if r["collapsed"]:
                k_all += 1
                for c in str(r["collapsed_classes"]).split():
                    cls_count[c] = cls_count.get(c, 0) + 1
        lower = tot = 0
        for rk in D.ranks:
            for sd in D.seeds:
                x, y = D.r(a, "dice_only", rk, sd), D.r(a, D.full, rk, sd)
                if x is None or y is None:
                    continue
                tot += 1
                lower += y["ipe_global_pct"] < x["ipe_global_pct"]
        txt = (f"\\textbf{{{arch_name(a)}.}} With the full objective, the mean Dice over seeds is highest at rank {best['rank']} "
               f"({num(best['mean_dice_mean'], DEC_DICE)}) and lowest at rank {lo['rank']} ({num(lo['mean_dice_mean'], DEC_DICE)}). "
               f"{k_full} of {n_full} full-objective runs lost a class")
        if k_full:
            ranks_lost = ", ".join(str(x["rank"]) for x in sums if x["n_runs_with_a_collapsed_class"])
            txt += f" (at rank {ranks_lost})"
        txt += f"; over all four losses {k_all} of {n_all} runs did."
        if cls_count:
            top = sorted(cls_count.items(), key=lambda kv: -kv[1])
            txt += " Classes lost most often: " + ", ".join(f"{esc(c)} ({n})" for c, n in top[:3]) + "."
        txt += (f" The full objective has a lower global IPE than Dice only in {lower} of {tot} seed-by-seed comparisons "
                f"(same rank and seed).")
        items.append("\\item " + txt)
    return "\n".join([r"\begin{itemize}\itemsep3pt", *items, r"\end{itemize}"])


def how_to_read(D: Data) -> str:
    return "\n".join([
        r"\section*{How to read the numbers}",
        r"\begin{description}\itemsep1pt",
        r"\item[Dice] overlap between predicted and annotated cells, averaged over the four cell classes. 1 is perfect, higher is better.",
        r"\item[AJI, BF1] AJI: how well single cells are separated. BF1: how well the cell outlines match. Both 0 to 1, higher is better.",
        r"\item[IPE] error of the integrated phase (the quantity that gives the dry mass), in percent. Lower is better.",
        f"\\item[Lost class] a run in which at least one cell class has Dice below {D.thr}, so that class is practically not found. "
        "Such runs are kept in every mean. A small superscript number next to a value is the number of runs, out of "
        f"{len(D.seeds)} seeds, that lost a class.",
        r"\item[Seeds] each setting was trained " + f"{len(D.seeds)}" + r" times with different random seeds ("
        + ", ".join(str(x) for x in D.seeds) + r"). $\pm$ is the standard deviation over these seeds. "
        r"With three seeds the numbers describe what happened; they are not significance tests.",
        r"\item[Setting] 32-bit phase maps in radians, last-epoch model, " + f"{esc(D.meta.get('n_validation_images', 'n/a'))}"
        + r" validation images. The rank is the LoRA rank (2 to 32). Losses: "
        + ", ".join(loss_name(l) for l in D.losses) + ".",
        r"\end{description}",
    ])


def glance_table(D: Data) -> str:
    cols = "r " + "l l l " * len(D.archs)
    top = "".join(f" & \\multicolumn{{3}}{{c}}{{{arch_name(a)}}}" for a in D.archs)
    cmid = "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(D.archs)))
    sub = "Rank" + " & Dice & IPE (\\%) & Lost" * len(D.archs)
    rows = []
    for rk in D.ranks:
        cells = [str(rk)]
        for a in D.archs:
            s = D.s(a, D.full, rk)
            if s is None:
                cells += ["--"] * 3
                continue
            cells += [pm(s["mean_dice_mean"], s["mean_dice_sd"], DEC_DICE), pm(s["ipe_global_pct_mean"], s["ipe_global_pct_sd"], DEC_IPE),
                      f"{s['n_runs_with_a_collapsed_class']}/{s['n']}"]
        rows.append(" & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    return (f"\\begin{{table}}[H]\\centering\\footnotesize\n\\begin{{tabular}}{{{cols.strip()}}}\n\\toprule\n{top} \\\\\n{cmid}\n{sub} \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
            f"\\caption{{Full objective at each LoRA rank (mean $\\pm$ SD over {len(D.seeds)} seeds). ``Lost'' = runs that lost a class.}}"
            f"\\label{{tab:glance}}\n\\end{{table}}")


def matrix_table(D: Data, key: str, dec: int, caption: str, label: str) -> str:
    cols = "l l " + "r " * len(D.ranks)
    head = "Model & Loss & " + " & ".join(f"r = {rk}" for rk in D.ranks)
    rows = []
    for ai, a in enumerate(D.archs):
        for li, l in enumerate(D.losses):
            cells = [arch_name(a) if li == 0 else "", loss_name(l)]
            for rk in D.ranks:
                s = D.s(a, l, rk)
                cells.append("--" if s is None else num(s[key], dec) + lost_marks(s))
            rows.append(" & ".join(cells) + " \\\\")
        if ai < len(D.archs) - 1:
            rows.append("\\midrule")
    body = "\n".join(rows)
    return (f"\\begin{{table}}[H]\\centering\\footnotesize\n\\begin{{tabular}}{{{cols.strip()}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}\n\\caption{{{caption}}}\\label{{{label}}}\n\\end{{table}}")


def class_table(D: Data) -> str:
    cols = "l r " + "l " * len(D.classes)
    head = "Model & Rank & " + " & ".join(esc(CLASS_SHORT.get(c, c)) for c in D.classes)
    rows = []
    for ai, a in enumerate(D.archs):
        for ri, rk in enumerate(D.ranks):
            s = D.s(a, D.full, rk)
            if s is None:
                continue
            cells = [arch_name(a) if ri == 0 else "", str(rk)] + [pm(s.get(f"dice_{c}_mean"), s.get(f"dice_{c}_sd"), DEC_DICE) for c in D.classes]
            rows.append(" & ".join(cells) + " \\\\")
        if ai < len(D.archs) - 1:
            rows.append("\\midrule")
    body = "\n".join(rows)
    return (f"\\begin{{table}}[H]\\centering\\footnotesize\n\\begin{{tabular}}{{{cols.strip()}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}\n\\caption{{Dice of each cell class, full objective (mean $\\pm$ SD over seeds). A class with "
            f"Dice near 0 is not found by that model. Class order: {', '.join(esc(c) for c in D.classes)}.}}\\label{{tab:classes}}\n\\end{{table}}")


def make_figures(D: Data, figdir: Path) -> list[str]:
    """Two small plots from results.json. Returns the file names written (empty when matplotlib is missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception:
        print("matplotlib not available: plots skipped")
        return []
    figdir.mkdir(parents=True, exist_ok=True)
    cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    xs = list(range(len(D.ranks)))
    n = len(D.archs)
    written = []

    def frame(ax, title):
        ax.set_title(title, fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(r) for r in D.ranks], fontsize=8)
        ax.set_xlabel("LoRA rank", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Figure 1: Dice by rank, full objective, seed points
    fig, axes = plt.subplots(1, n, figsize=(7.0, 2.5), sharey=True)
    axes = [axes] if n == 1 else list(axes)
    allv = []
    for ax, a, c in zip(axes, D.archs, cols):
        means = []
        for i, rk in enumerate(D.ranks):
            s = D.s(a, D.full, rk)
            means.append(s["mean_dice_mean"] if s else float("nan"))
            for j, sd in enumerate(D.seeds):
                r = D.r(a, D.full, rk, sd)
                if r is None:
                    continue
                allv.append(r["mean_dice"])
                off = (j - (len(D.seeds) - 1) / 2) * 0.12
                ax.scatter(i + off, r["mean_dice"], s=18, color=("#C00000" if r["collapsed"] else c), marker=("x" if r["collapsed"] else "o"),
                           alpha=0.9, zorder=3)
        ax.plot(xs, means, color=c, lw=1.4, zorder=2)
        frame(ax, arch_name(a))
    axes[0].set_ylabel("Dice (mean of 4 classes)", fontsize=8)
    if allv:
        pad = 0.05 * (max(allv) - min(allv) or 1)
        axes[0].set_ylim(min(allv) - pad, max(allv) + pad)
    fig.text(0.5, -0.02, "line = mean over seeds, dots = single seeds, red x = run that lost a class", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig_dice_by_rank.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    written.append("fig_dice_by_rank.png")

    # Figure 2: IPE by rank, Dice only versus full objective (log axis)
    fig, axes = plt.subplots(1, n, figsize=(7.0, 2.5), sharey=True)
    axes = [axes] if n == 1 else list(axes)
    base = D.losses[0]
    styles = [(base, "#7F7F7F", "o"), (D.full, "#0072B2", "s")]
    for ax, a in zip(axes, D.archs):
        for l, c, mk in styles:
            means = []
            for i, rk in enumerate(D.ranks):
                s = D.s(a, l, rk)
                means.append(s["ipe_global_pct_mean"] if s else float("nan"))
                for j, sd in enumerate(D.seeds):
                    r = D.r(a, l, rk, sd)
                    if r is None:
                        continue
                    off = (j - (len(D.seeds) - 1) / 2) * 0.1 + (0.12 if l == D.full else -0.12)
                    ax.scatter(i + off, r["ipe_global_pct"], s=12, color=c, marker=mk, alpha=0.55, zorder=3)
            ax.plot(xs, means, color=c, lw=1.4, label=loss_name(l), zorder=2)
        frame(ax, arch_name(a))
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].set_ylabel("global IPE (%), log scale", fontsize=8)
    axes[-1].legend(fontsize=7, frameon=False)
    fig.text(0.5, -0.02, "lower is better; line = mean over seeds, dots = single seeds", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig_ipe_by_rank.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    written.append("fig_ipe_by_rank.png")
    return written


def figure_block(files: list[str], figdir_name: str, D: Data) -> str:
    caps = {
        "fig_dice_by_rank.png": f"Dice by LoRA rank, {loss_name(D.full)} objective. Higher is better.",
        "fig_ipe_by_rank.png": f"Global IPE by LoRA rank, {loss_name(D.losses[0])} versus {loss_name(D.full)} objective. Lower is better.",
    }
    out = []
    for f in files:
        out.append(f"\\begin{{figure}}[H]\\centering\\includegraphics[width=\\linewidth]{{{figdir_name}/{f}}}\n"
                   f"\\caption{{{caps.get(f, esc(f))}}}\\end{{figure}}")
    return "\n".join(out)


def short_report(D: Data, src: Path, figs: list[str], figdir_name: str) -> str:
    intro = (f"Results of {D.ms['n_runs']} LoRA training runs: {len(D.archs)} models ({', '.join(arch_name(a) for a in D.archs)}), "
             f"{len(D.ranks)} ranks ({', '.join(str(r) for r in D.ranks)}), {len(D.losses)} loss settings and {len(D.seeds)} seeds. "
             f"Source: \\texttt{{{esc(src.name)}}}, generated {esc(D.meta.get('generated_utc', 'n/a'))}. "
             "Class names follow the manuscript order: "
             + ", ".join(f"mask value {i + 1} = {esc(c)}" for i, c in enumerate(D.classes)) + ".")
    parts = [
        f"{{\\Large\\bfseries Multi-seed results}}\\\\[2pt]{{\\small 32-bit phase data, LoRA rank, loss and seed}}\\par\\vspace{{4pt}}",
        intro,
        how_to_read(D),
        r"\section*{Key points}",
        key_points(D),
        r"\section*{Full objective at each rank}",
        glance_table(D),
        figure_block(figs, figdir_name, D),
        r"\section*{Effect of the loss, at every rank}",
        matrix_table(D, "mean_dice_mean", DEC_DICE,
                     f"Dice by loss and rank (mean over {len(D.seeds)} seeds). Superscript = runs that lost a class. Higher is better.",
                     "tab:dice_matrix"),
        matrix_table(D, "ipe_global_pct_mean", DEC_IPE,
                     f"Global IPE in percent by loss and rank (mean over {len(D.seeds)} seeds). Superscript = runs that lost a class. Lower is better.",
                     "tab:ipe_matrix"),
        r"\section*{Dice of each cell class}",
        class_table(D),
        r"The complete tables (every seed, every metric, every run, paired differences) follow in the appendix.",
    ]
    return "\n\n".join(parts)


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{booktabs,longtable,array,graphicx,pdflscape,float}
\usepackage[font=small,labelfont=bf]{caption}
\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}\setlength{\LTcapwidth}{\textwidth}
\setlength{\tabcolsep}{4pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
\begin{document}
"""


def build(D: Data, src: Path, figs: list[str], figdir_name: str, appendix: bool = True) -> str:
    parts = [PREAMBLE, short_report(D, src, figs, figdir_name)]
    if appendix:
        parts += [r"\clearpage\appendix\begin{landscape}", header_section(D, src), benchmark_section(D), equivalence_section(D), rank_sweep(D), ablation(D), per_class(D), paired(D),
                  all_runs(D), r"\end{landscape}"]
    parts.append("\\end{document}\n")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "manuscript" / "outputs" / "results.json"))
    ap.add_argument("--out", default=str(ROOT / "manuscript" / "outputs" / "multiseed_results.tex"))
    ap.add_argument("--no-appendix", action="store_true", help="only the short report, without the full tables")
    ap.add_argument("--pdf", action="store_true", help="compile with pdflatex (two passes) if it is installed")
    a = ap.parse_args()
    src, out = Path(a.results), Path(a.out)
    D = Data(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    figdir_name = out.stem + "_figures"
    figs = make_figures(D, out.parent / figdir_name)
    out.write_text(build(D, src, figs, figdir_name, appendix=not a.no_appendix), encoding="utf-8")
    print(f"wrote {out} ({len(D.runs)} runs, {len(D.summary)} configurations, {len(D.paired)} paired rows)")
    print(f"  results.json: generated {D.meta.get('generated_utc', 'n/a')}, file time "
          f"{datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec='seconds')}")
    print(f"  class names by mask value: {D.classes}")
    # staleness check: results.json must be newer than the run table it was built from
    for key in ("source_runs_index", "source_aggregator"):
        f = Path(str(D.ms.get(key, "")))
        if f.is_file() and f.stat().st_mtime > src.stat().st_mtime + 1:
            print(f"  WARNING: {f.name} is newer than {src.name}; run  python manuscript/results.py  first, this report is out of date")
    if a.pdf:
        # tectonic first (self-contained), then pdflatex, xelatex, lualatex; a compiler that exists but fails is skipped
        attempts = [("tectonic", [["tectonic", out.name]])]
        attempts += [(t, [[t, "-interaction=nonstopmode", "-halt-on-error", out.name]] * 2) for t in ("pdflatex", "xelatex", "lualatex")]
        done = False
        for tool, cmds in attempts:
            if not shutil.which(tool):
                continue
            try:
                for cmd in cmds:
                    subprocess.run(cmd, cwd=out.parent, check=True, stdout=subprocess.DEVNULL)
                done = True
                print(f"compiled with {tool}")
                break
            except (subprocess.CalledProcessError, OSError) as e:
                print(f"{tool} failed ({e}); trying the next compiler")
        if done:
            print(f"wrote {out.with_suffix('.pdf')}")
        else:
            print("no working TeX compiler (tectonic, pdflatex, xelatex, lualatex); compile the .tex file yourself (for example on Overleaf)")


if __name__ == "__main__":
    main()
