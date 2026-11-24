"""Data figures of the manuscript and the Supplementary Information.

Each function returns (figure, section, file stem, figure number, caption). The caption is generated from the
same values the figure plots, so the two cannot disagree. Image figures (qualitative segmentation, the
superposition screen) need the prepared images and are in image_figures.py.
"""
from __future__ import annotations

import shutil
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from . import style as S
from .core import (NEW_FULL, OLD_FULL, PROPOSED, Inputs, MissingInput, Runs, arch_dir, arch_disp, arm_math, LOSS_GLOSSARY,
                   cond_label, cond_label_m, cond_short, cond_short_m, cls_name, finite, fmt_p, representative, require_rescored, run_dir)

W = "additional_analysis/results"


def _rep(inp):
    return representative(inp)


def _cells(inp: Inputs, R: Runs, a: str, c: str, arm: str, seed: int):
    import pandas as pd
    rel = f"{run_dir(a, c, arm, seed)}/cells_{R.checkpoint}.json"
    D = pd.DataFrame([x for x in require_rescored(inp, rel)["cells"] if x.get("instances") == "A2"])
    return D


def _m(t: str) -> str:
    """Typographic minus sign in figure text."""
    return t.replace("-", "\u2212")


def _ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx, sy = x.var(), y.var()
    return 2 * np.cov(x, y, bias=True)[0, 1] / (sx + sy + (x.mean() - y.mean()) ** 2)


# ================================================================================================ Fig. 1

def fig1_framework(inp: Inputs, R: Runs):
    """Framework schematic, drawn in code; every constant shown is read from configs/config.yaml or the results."""
    cfg = inp.config()
    if not cfg:
        raise MissingInput("configs/config.yaml (and PyYAML)")
    rep = _rep(inp)
    ph, bg, cells, loss, models = cfg["physics"], cfg["background"], cfg["cells"], cfg["loss"], cfg["models"]
    cp = loss["cipp"]
    fig = plt.figure(figsize=(S.DOUBLE, 72 * S.MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 174); ax.set_ylim(0, 72); ax.axis("off")
    heads = ["1  Phase map", "2  Encoder + LoRA", "3  Decoder", "4  Training objective", "5  Cell readout"]
    xs = [1, 35.5, 70, 97.5, 139]
    ws = [32, 32, 25.5, 39.5, 34]
    top, h = 65, 54
    shades = ["#f3f2ee", "#eef4fb", "#eef4fb", "#fdf1ec", "#f3f2ee"]
    for x, w, t, sh in zip(xs, ws, heads, shades):
        ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=sh,
                                    ec=S.INK2, lw=0.6))
        ax.text(x + 1.5, top - 3.2, t, fontsize=7.5, fontweight="bold", va="center")
    for i in range(4):
        x0 = xs[i] + ws[i]; x1 = xs[i + 1]
        ax.add_patch(FancyArrowPatch((x0 + 0.3, top - h / 2), (x1 - 0.3, top - h / 2), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.8, color=S.INK2))

    def lines(x, y, items, size=6.2, gap=3.4):
        for it in items:
            kw = dict(fontsize=size, va="top", ha="left")
            g = gap
            if isinstance(it, tuple):
                txt, extra = it; extra = dict(extra); g = extra.pop("_gap", gap); kw.update(extra)
            else:
                txt = it
            ax.text(x, y, txt, **kw)
            y -= g * (txt.count("\n") + 1)
        return y

    px_um = ph["pixel_size_um"]
    lines(xs[0] + 1.5, top - 7, [
        ("Reconstructed phase φ (rad)", {"fontweight": "bold"}),
        f"{cfg['data']['native_size']} × {cfg['data']['native_size']} px,\nΔx = {px_um} µm",
        ("Annotation-free background", {"fontweight": "bold"}),
        f"order-{bg['order']} polynomial fitted to\npixels within {bg['k_sigma']:g} σ, cells grown\n"
        f"by {bg['grow_px']} px, ≤ {bg['max_iter']} iterations;\nsubtracted once per image",
        ("Network input", {"fontweight": "bold"}),
        "phase standardised per image\n(zero mean, unit variance);\nthe loss sees φ in radians"])
    arch_lines = []
    for k, m in models.items():
        a = arch_disp(m.get("display", k))
        rr = rep.get(a, {})
        pct = rr.get("selected_params_trainable_pct")
        arch_lines.append(f"{a}: {m['image_size']} px input,\nselected {cond_label(rr.get('selected_condition', '?'))}"
                          + (f", {float(pct):.1f} % trained" if pct not in (None, "") else ""))
    lines(xs[1] + 1.5, top - 7, [("Frozen pretrained encoder", {"fontweight": "bold"})] + arch_lines +
          [("Low-rank adapters", {"fontweight": "bold"}),
           "W′ = W + (α/r) B A,  α = r\n"
           f"r = {', '.join(str(r) for r in cfg['conditions']['lora_ranks'])}; full fine-tuning\nas a control"],
          gap=3.3)
    y = lines(xs[2] + 1.5, top - 7, [("Trained in full", {"fontweight": "bold"}),
                                    "stem, normalisation,\ndecoder and head",
                                    ("Output", {"fontweight": "bold"}),
                                    f"{cfg['data']['num_classes']}-class map:\nbackground and"])
    for i, k in enumerate(R.classes):
        yy = y - 0.6 - i * 3.3
        ax.add_patch(plt.Rectangle((xs[2] + 1.6, yy - 2.0), 2.2, 2.2, fc=S.CLASS_COLOR.get(k, S.MUTED), ec="none"))
        ax.text(xs[2] + 4.6, yy - 0.9, k, fontsize=6.2, va="center")
    lines(xs[3] + 1.5, top - 7, [
        (r"Proposed:  $\mathcal{L}$ = $\mathcal{L}_{\mathrm{Dice}}$ + λ $\mathcal{L}_{\mathrm{cIPP}}$", {"fontweight": "bold"}),
        f"λ = {loss['lambda_cipp']:g}; per annotated cell c:",
        ("e$_c$ = (S$_c^{pred}$ − S$_c^{ref}$) / max(|S$_c^{ref}$|, κ·med)", {"_gap": 4.4}),
        r"$\mathcal{L}_{\mathrm{cIPP}}$ = mean ρ(e$_c$) + β mean O$_c$ + γ F",
        f"ρ: smooth |·| (δ = {cp['delta']:g}); κ = {cp['kappa']:g}",
        f"O$_c$: per-cell overlap (β = {cp['beta']:g})",
        f"F: phase far from every cell (γ = {cp['gamma']:g})",
        f"S$_c$ summed over the cell's territory\n(≤ {cells['territory_radius_native_px']} native px around it)",
        ("Compared terms", {"fontweight": "bold"}),
        r"$\mathcal{L}_{\mathrm{PMC}}$" f" (λ = {loss['lambda_pmc']:g}), " r"$\mathcal{L}_{\mathrm{BGA}}$"
        f" (λ = {loss['lambda_bga']:g}),\n" r"$\mathcal{L}_{\mathrm{IPP}}$" f" (λ = {loss['lambda_ipp']:g}): "
        f"{len(R.arms)} objectives"], gap=3.25)
    alpha = ph["refractive_increment_ml_per_g"]
    lines(xs[4] + 1.5, top - 7, [
        ("Cells", {"fontweight": "bold"}),
        f"watershed of the predicted\nforeground (h-maxima, h = {cells['watershed_h']})",
        ("Per cell", {"fontweight": "bold"}),
        "S = Σ φ$_i$ Δx²   (integrated phase)",
        f"m = λ$_0$ S / (2π α)   (dry mass)\nλ$_0$ = {ph['wavelength_um'] * 1000:g} nm, α = {alpha:g} ml g$^{{-1}}$",
        ("Evaluation", {"fontweight": "bold"}),
        f"cell integrated-phase error;\nshare of cells within {cfg['evaluation']['within_pct']:g} %;\n"
        f"cells matched at IoU ≥ {cfg['evaluation']['match_iou']:g}"], gap=3.3)
    # feedback arrow: gradients reach adapters and decoder only
    ax.add_patch(FancyArrowPatch((xs[3] + ws[3] / 2, top - h - 0.3), (xs[1] + ws[1] / 2, top - h - 0.3),
                                 connectionstyle="arc3,rad=-0.12", arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                 color=S.INK2, linestyle=(0, (3, 2))))
    ax.text((xs[1] + xs[3] + ws[3]) / 2 + 4, 1.2, "gradients update the adapters, stem, normalisation and decoder; "
            "the pretrained encoder weights stay fixed", fontsize=6, ha="center", style="italic", color=S.INK2)
    ax.add_patch(FancyArrowPatch((xs[0] + ws[0] / 2, top + 0.3), (xs[3] + ws[3] / 2, top + 0.3),
                                 connectionstyle="arc3,rad=-0.06", arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                 color=S.INK2, linestyle=(0, (3, 2))))
    ax.text((xs[0] + xs[3] + ws[3]) / 2 - 8, top + 4.6, "background-referenced phase φ (rad) enters the loss directly",
            fontsize=6, ha="center", style="italic", color=S.INK2)
    cap = ("Framework. (1) The reconstructed phase map is referenced to an annotation-free polynomial background and "
           "standardised for the network. (2) A pretrained lightweight encoder is held fixed and adapted with low-rank "
           "adapters (or fully fine-tuned as a control). (3) The decoder predicts background and four morphology "
           "classes. (4) The proposed objective adds a cell-level integrated-phase term ℒ<sub>cIPP</sub> to the Dice "
           "loss ℒ<sub>Dice</sub>; ℒ<sub>PMC</sub>, phase-mask contrast; ℒ<sub>BGA</sub>, boundary-gradient "
           "alignment; ℒ<sub>IPP</sub>, global integrated-phase preservation. "
           "S<sub>c</sub>: integrated phase of cell c (prediction: foreground probability × phase summed over the "
           "cell's territory); med: median |S<sub>ref</sub>| over the cells of the image; ρ: smooth absolute value. "
           
           "(5) Touching cells are separated by a watershed, and each cell's integrated phase and dry mass are read "
           "out. All constants are those of configs/config.yaml; the selected ranks are those of the selection rule.")
    return fig, "main", "fig1_framework", "1", cap


# ================================================================================================ Fig. 3

def fig3_objectives(inp: Inputs, R: Runs):
    metrics = [("a2_scored_within_5pct_of_all_cells_pct", "Cells within 5 % (%)", 1.0, "linear"),
               ("a2_scored_cell_ipe_abs_median_pct", "Cell integrated-phase error,\nmatched cells (%)", 1.0, "linear"),
               ("ipe_global", "Global integrated-phase\nerror (%)", 100.0, "log")]
    arms = R.arms
    conds = R.lora_conditions
    fig, axes = plt.subplots(len(R.archs), len(metrics), figsize=(S.DOUBLE, 150 * S.MM), sharey=True)
    ypos = {a: len(arms) - 1 - i for i, a in enumerate(arms)}
    rng = np.random.default_rng(0)
    medians: Dict = {}
    for i, a in enumerate(R.archs):
        for j, (k, lab, s, scale) in enumerate(metrics):
            ax = axes[i, j]
            for arm in arms:
                vals = [v for c in conds for v in R.col(a, c, arm, k, s)]
                y = ypos[arm]
                if arm == PROPOSED:
                    ax.axhspan(y - 0.45, y + 0.45, color="#efeeea", lw=0, zorder=0)
                jit = rng.uniform(-0.18, 0.18, len(vals))
                ax.scatter(vals, y + jit, s=7, color=S.ARCH_COLOR[a], alpha=0.55, lw=0, zorder=2)
                if vals:
                    m = float(np.median(vals)); medians[(a, arm, k)] = (m, len(vals))
                    ax.plot([m, m], [y - 0.32, y + 0.32], color=S.INK, lw=1.4, zorder=3, solid_capstyle="butt")
            ax.set_xscale(scale)
            if scale == "log":
                from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
            S.vgrid(ax)
            ax.set_yticks([ypos[a_] for a_ in arms])
            ax.set_yticklabels([arm_math(a_) for a_ in arms])
            ax.tick_params(axis="y", length=0)
            if i == len(R.archs) - 1:
                ax.set_xlabel(lab)
            if j == len(metrics) // 2:
                ax.set_title(f"({'abc'[i]})  {a}", loc="center", fontsize=9, pad=6)
            for sp in ("left",):
                ax.spines[sp].set_visible(False)
    for j in range(len(metrics)):
        lo = min(ax_.get_xlim()[0] for ax_ in axes[:, j]); hi = max(ax_.get_xlim()[1] for ax_ in axes[:, j])
        for ax_ in axes[:, j]:
            ax_.set_xlim(lo, hi)
    handles = [Line2D([], [], marker="o", ls="", color=S.MUTED, markersize=3, label="one run (rank × seed)"),
               Line2D([], [], color=S.INK, lw=1.4, label="median")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=1.2, w_pad=1.0)
    k0 = metrics[0][0]
    txt = "; ".join(f"{a} {medians[(a, PROPOSED, k0)][0]:.1f} % (Dice alone {medians[(a, 'dice', k0)][0]:.1f} %)"
                    for a in R.archs if (a, PROPOSED, k0) in medians and (a, "dice", k0) in medians)
    n = sorted({v[1] for v in medians.values()})
    cap = ("Cell-level against global phase fidelity for the seven training objectives. Each dot is one run "
           f"({len(conds)} low-rank adaptation ranks × {len(R.seeds)} seeds = {n[-1] if n else '–'} runs per "
           "objective and architecture; full fine-tuning not shown, see Table S2); the bar is the median. "
           "Left: share of all scored reference cells whose integrated phase is reproduced to within 5 %. Middle: "
           "median cell integrated-phase error of matched cells. Right: global (whole-field) integrated-phase error, "
           f"log scale. Median share within 5 % under the proposed objective: {txt}. The shaded row is the proposed "
           "objective. " + LOSS_GLOSSARY)
    return fig, "main", "fig3_objectives", "3", cap


# ================================================================================================ Fig. 4 and S6

def _forest(inp: Inputs, R: Runs, groups, number, stem, section, title_note, keys=None, row_mm=5.0):
    w1 = inp.json(f"{W}/w1_cell_paired/w1_cell_paired.json")
    order_cmp = [(PROPOSED, "dice"), (PROPOSED, "dice_pmc_cipp"), (PROPOSED, NEW_FULL), (PROPOSED, OLD_FULL),
                 (NEW_FULL, OLD_FULL)]
    res = {(arch_disp(r["architecture"]), r["loss_a"], r["loss_b"]): r for r in w1["results"]}
    stats = [("d_within5_pp", "Δ cells within 5% (pp)\n> 0 favours A"),
             ("d_median_abs_e_assigned_pp", "Δ median |cell error|, matching-free\n(pp); < 0 favours A")]
    if keys:
        stats = [x for x in stats if x[0] in keys]
    rows = []
    for a in R.archs:
        for la, lb in order_cmp:
            if (a, la, lb) in res:
                rows.append((a, la, lb, res[(a, la, lb)]))
    ng = len(groups)
    fig, axes = plt.subplots(1, len(stats) * ng, figsize=(S.DOUBLE, (22 + row_mm * len(rows)) * S.MM), sharey=True)
    axes = np.atleast_1d(axes)
    y, ylab, yv, seps = 0.0, [], [], []
    ys = []
    prev = None
    for a, la, lb, r in rows:
        if prev is not None and a != prev:
            y -= 0.7 * row_mm / 5.0; seps.append(y + 0.35 * row_mm / 5.0)
        ys.append(y); ylab.append(f"{arm_math(la)}  vs  {arm_math(lb)}"); prev = a; y -= 1
    for gi, grp in enumerate(groups):
        for si, (k, lab) in enumerate(stats):
            ax = axes[gi * len(stats) + si]
            for (a, la, lb, r), yy in zip(rows, ys):
                st = r["groups"].get(grp, {}).get("statistics", {}).get(k)
                if not st:
                    continue
                lo, hi = st["ci95_hierarchical_bootstrap"]
                ax.plot([lo, hi], [yy, yy], color=S.ARCH_COLOR[a], lw=1.3, solid_capstyle="butt")
                ax.plot(st["estimate"], yy, marker=S.ARCH_MARKER[a], color=S.ARCH_COLOR[a], ms=4, mec="white", mew=0.5)
                p = st.get("p_holm", st.get("p_day_label_swap"))
                if p is not None:
                    ax.annotate(f"{fmt_p(p)}" + ("" if "p_holm" in st else "*"), (1.0, yy),
                                xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                                va="center", fontsize=7, color="black")
            ax.axvline(0, color=S.INK2, lw=0.6)
            for s_ in seps:
                ax.axhline(s_, color=S.GRID, lw=0.6)
            S.vgrid(ax)
            ax.set_xlabel(lab)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            if any(r["groups"].get(grp, {}).get("statistics", {}).get(k, {}).get("p_day_label_swap") is not None
                   for _, _, _, r in rows):
                ax.annotate("p", (1.0, 1.0), xycoords="axes fraction", xytext=(4, 2), textcoords="offset points",
                            fontsize=7.5, color="black", fontstyle="italic", va="bottom")
    axes[0].set_yticks(ys); axes[0].set_yticklabels(ylab)
    axes[0].tick_params(axis="y", pad=8)  # loss labels moved a little to the left of the plot
    for ax in axes:
        ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    for i, ax in enumerate(axes):
        grp = groups[i // len(stats)]
        S.panel_title(ax, "abcdef"[i], f"{grp.capitalize()} cells" if ng > 1 else "", pad=8)
    fig.tight_layout(w_pad=4.5)
    fig.subplots_adjust(left=fig.subplotpars.left + 0.03)
    # model names once, vertical, in the left margin beside their block of rows
    for a in R.archs:
        yy = [y_ for (a_, *_), y_ in zip(rows, ys) if a_ == a]
        if yy:
            axes[0].annotate(a, (0.004, np.mean(yy)), xycoords=("figure fraction", "data"), rotation=90,
                             va="center", ha="left", fontsize=9, color=S.INK)
    reps = ", ".join(f"{a} {cond_label(r['condition'])}" for a, r in OrderedDict(
        (a, r) for a, _, _, r in rows).items())
    md = w1["metadata"]
    est = {(a, la, lb): r["groups"]["all"]["statistics"]["d_within5_pp"] for a, la, lb, r in rows}
    ex = "; ".join(f"{a} {e['estimate']:+.1f} points [{e['ci95_hierarchical_bootstrap'][0]:.1f}, "
                   f"{e['ci95_hierarchical_bootstrap'][1]:.1f}]" for (a, la, lb), e in est.items()
                   if la == PROPOSED and lb == "dice")
    cap = (f"Cell-level paired comparisons at the representative configurations ({reps}){title_note}. Each "
           "row compares objective A with objective B on the same reference cells, pooled over the "
           f"{len(R.seeds)} seeds. Points: estimate; lines: 95 % interval from a hierarchical bootstrap over storage "
           f"days, images and cells ({md.get('bootstrap', '–')} resamples). p: exact day-level label-swap test, "
           "Holm-corrected over the primary family (* not part of that family: unadjusted). The matching-free error compares each reference cell's integrated "
           "phase with the predicted phase in its territory, so a missed cell counts as an error near 100 %. "
           f"Proposed objective against ℒ<sub>Dice</sub> alone, cells within 5 %: {ex}. " + LOSS_GLOSSARY)
    return fig, section, stem, number, cap


def fig4_cell_paired(inp: Inputs, R: Runs):
    return _forest(inp, R, ["all"], "4", "fig4_cell_paired_comparisons", "main", "")


def figS6_cell_paired_touching(inp: Inputs, R: Runs):
    return _forest(inp, R, ["touching", "isolated"], "S5", "figS5_cell_paired_touching_isolated", "supplementary",
                   ", separately for touching and isolated reference cells (intervals only; the tests are run on all "
                   "cells, Fig. 4)", keys=["d_within5_pp"], row_mm=6.6)


# ================================================================================================ Fig. 5

def fig5_rank_selection(inp: Inputs, R: Runs, arm: str = PROPOSED):
    rep = _rep(inp)
    metrics = [("a2_scored_within_5pct_of_all_cells_pct", "Cells within 5 % (%)", 1.0),
               # "Mean Dice" is the name the manuscript and every table use for this quantity: the Dice
               # averaged over the five classes. The axis read "Macro Dice", which no table did.
               ("dice_macro", "Mean Dice", 1.0)]
    fig, axes = plt.subplots(len(metrics), len(R.archs), figsize=(S.DOUBLE, 112 * S.MM), sharex=True)
    xpos = {c: i + (0.6 if c == "fullft" else 0) for i, c in enumerate(R.conditions)}
    offs = {s: o for s, o in zip(R.seeds, np.linspace(-0.16, 0.16, len(R.seeds)))}
    lost_any = False
    for j, a in enumerate(R.archs):
        sel = rep[a]["selected_condition"]
        for i, (k, lab, s) in enumerate(metrics):
            ax = axes[i, j]
            ax.axvspan(xpos[sel] - 0.4, xpos[sel] + 0.4, color="#efeeea", lw=0, zorder=0)
            means = []
            for c in R.conditions:
                rs = R.get(a, c, arm)
                for r in rs:
                    v = r.get(k)
                    if v is None:
                        continue
                    lost = (r.get("n_lost_classes") or 0) > 0
                    lost_any |= lost
                    ax.plot(xpos[c] + offs[int(r["seed"])], v * s, marker=S.ARCH_MARKER[a], ms=4,
                            mfc="white" if lost else S.ARCH_COLOR[a], mec=S.ARCH_COLOR[a], mew=0.9, ls="", zorder=3)
                vals = finite(r.get(k) for r in rs)
                means.append(np.mean(vals) * s if vals else np.nan)
                ax.plot([xpos[c] - 0.28, xpos[c] + 0.28], [means[-1]] * 2, color=S.INK, lw=1.2, zorder=4)
            lx = [xpos[c] for c in R.lora_conditions]
            ax.plot(lx, means[:len(lx)], color=S.INK2, lw=0.6, zorder=2)
            S.hgrid(ax)
            if j == 0:
                ax.set_ylabel(lab)
            ax.set_xticks([xpos[c] for c in R.conditions])
            ax.set_xticklabels([cond_short_m(c) for c in R.conditions], rotation=45, ha="right",
                               rotation_mode="anchor")
            if i == 0:
                S.panel_key_title(ax, "abc"[j], a, S.ARCH_COLOR[a])
            else:
                S.panel_title(ax, "def"[j])
    for i in range(len(metrics)):
        lo = min(ax.get_ylim()[0] for ax in axes[i]); hi = max(ax.get_ylim()[1] for ax in axes[i])
        pad = 0.03 * (hi - lo)
        for ax in axes[i]:
            ax.set_ylim(lo - pad, hi + pad)
    h = [Line2D([], [], marker="o", ls="", mfc=S.MUTED, mec=S.MUTED, ms=4, label="run, every class kept"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=S.MUTED, ms=4, mew=0.9, label="run that lost a class"),
         Line2D([], [], color=S.INK, lw=1.2, label="mean of seeds"),
         Patch(fc="#efeeea", ec="none", label="selected rank")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.0)
    parts = []
    for a in R.archs:
        r = rep[a]
        parts.append(f"{a}: {cond_label(r['selected_condition'])}, {float(r['selected_mean_within_5pct']):.2f} ± "
                     f"{float(r['selected_sd_within_5pct']):.2f} % within 5 %"
                     + ("" if str(r.get("every_rank_lost_a_class", "")).lower() != "true"
                        else " (no rank kept every class in every seed, so the most stable ranks were scored)"))
    cap = ("Rank selection under the proposed objective. Top: share of scored cells within 5 %; bottom: mean Dice, "
           "the Dice averaged over the five classes. "
           f"Each marker is one seed ({', '.join(map(str, R.seeds))}); open markers are runs in which a morphology class "
           f"was lost (Dice < {R.lost_class_dice}); bars are seed means; FFT, full fine-tuning (different learning "
           "rate). The rule keeps the ranks at which the fewest seeds lost a class (training stability), then takes "
           "the best balanced score, the mean of the share within 5 % and the mean Dice (shaded), and the smaller "
           "rank on a tie. " + "; ".join(parts) + ".")
    return fig, "main", "fig5_rank_selection", "5", cap


# ================================================================================================ Fig. 7

def fig7_dry_mass(inp: Inputs, R: Runs, arm: str = PROPOSED):
    rep = _rep(inp)
    seed = R.seeds[0]
    fig, axes = plt.subplots(2, len(R.archs), figsize=(S.DOUBLE, 116 * S.MM))
    stats = {}
    allv = []
    data = {}
    for a in R.archs:
        c = rep[a]["selected_condition"]
        D = _cells(inp, R, a, c, arm, seed)
        # every scored reference cell with a one-to-one match: the cell set of runs_last.json (Tables 1, 3, 5), so
        # the figure and the tables agree and MdARE equals the median cell integrated-phase error.
        # (metrics_last.json additionally drops the 0-3 matched cells per run whose prediction is near wrapping.)
        D = D[D["scored"].fillna(False).astype(bool) & D["matched"].fillna(False).astype(bool)]
        x, y = D["dm_gt_pg"].to_numpy(float), D["dm_pred_pg"].to_numpy(float)
        data[a] = (x, y, c)
        allv += [x, y]
    allv = np.concatenate(allv)
    lo, hi = np.percentile(allv, 0.2), np.percentile(allv, 99.8)
    pad = 0.04 * (hi - lo); lo, hi = lo - pad, hi + pad
    dmax = max(np.percentile(np.abs(y - x), 99.8) for x, y, _ in data.values()) * 1.1
    for j, a in enumerate(R.archs):
        x, y, c = data[a]
        d, m = y - x, (x + y) / 2
        rr = float(np.corrcoef(x, y)[0, 1]); cc = float(_ccc(x, y)); bias = float(d.mean()); sd = float(d.std(ddof=1))
        mdare = float(np.median(np.abs(d) / np.abs(x)) * 100)
        stats[a] = dict(n=len(x), r=rr, ccc=cc, bias=bias, loa=(bias - 1.96 * sd, bias + 1.96 * sd), mdare=mdare, cond=c)
        ax = axes[0, j]
        ax.plot([lo, hi], [lo, hi], color=S.INK2, lw=0.7, ls=(0, (4, 2)), zorder=1)
        ax.scatter(x, y, s=2.5, color=S.ARCH_COLOR[a], alpha=0.35, lw=0, rasterized=True, zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        S.panel_key_title(ax, "abc"[j], f"{a} ({cond_label_m(c)})", S.ARCH_COLOR[a])
        ax.text(0.97, 0.03, f"$n$ = {len(x):,}\n$r$ = {rr:.3f}\nCCC = {cc:.3f}\nMdARE = {mdare:.2f} %",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=7, zorder=6,
                bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.85))
        if j == len(R.archs) // 2:
            ax.set_xlabel("Reference dry mass (pg)")
        if j == 0:
            ax.set_ylabel("Predicted dry mass (pg)")
        ax = axes[1, j]
        ax.scatter(m, d, s=2.5, color=S.ARCH_COLOR[a], alpha=0.35, lw=0, rasterized=True)
        ax.axhline(0, color=S.INK2, lw=0.6)
        ax.axhline(bias, color=S.INK, lw=1.0)
        for v in stats[a]["loa"]:
            ax.axhline(v, color=S.INK, lw=0.7, ls=(0, (4, 2)))
        ax.text(1.0, bias, _m(f" bias {bias:+.2f}"), transform=ax.get_yaxis_transform(), va="center", fontsize=6)
        ax.text(1.0, stats[a]["loa"][1], _m(f" {stats[a]['loa'][1]:+.2f}"), transform=ax.get_yaxis_transform(),
                va="center", fontsize=6)
        ax.text(1.0, stats[a]["loa"][0], _m(f" {stats[a]['loa'][0]:+.2f}"), transform=ax.get_yaxis_transform(),
                va="center", fontsize=6)
        ax.set_xlim(lo, hi); ax.set_ylim(-dmax, dmax)
        ax.set_box_aspect(1)  # same square size as the panels above
        if j == len(R.archs) // 2:
            ax.set_xlabel("Mean of reference and prediction (pg)")
        if j == 0:
            ax.set_ylabel("Prediction − reference (pg)")
        S.hgrid(ax)
        S.panel_title(ax, "def"[j])
    fig.tight_layout(w_pad=2.5, h_pad=1.0)
    fig._figtab_stats = stats  # for verify.py
    s = "; ".join(f"{a}: r = {v['r']:.3f}, CCC = {v['ccc']:.3f}, bias {v['bias']:+.3f} pg, limits of agreement "
                  f"{v['loa'][0]:+.2f} to {v['loa'][1]:+.2f} pg" for a, v in stats.items())
    cap = (f"Dry-mass agreement of single cells, seed {seed}, representative configurations, proposed objective. "
           "(a–c) Predicted against reference dry mass of every scored reference cell matched one to one by a "
           "predicted cell (dashed: identity). (d–f) Bland–Altman: difference against mean, with the bias (solid) and "
           f"the 95 % limits of agreement (dashed). {s}. MdARE: median absolute relative error, which equals the "
           "median cell integrated-phase error; the cells and statistics are those of Tables 1 and 5. Axes cover the "
           "central 99.6 % of values; no point is excluded from the statistics. Seed means are in Table 5.")
    return fig, "main", "fig7_dry_mass_agreement", "7", cap


# ================================================================================================ Fig. 8

def fig8_storage(inp: Inputs, R: Runs, arm: str = PROPOSED):
    import pandas as pd
    rep = _rep(inp)
    ann = inp.json("results/consolidated/storage/annotation.json")
    st = inp.json(f"results/consolidated/storage/runs_{R.checkpoint}.json")
    pi = pd.DataFrame(ann["per_image"])
    ref = {q: pi.groupby("storage_day")[q].median() for q in ("area", "dry_mass")}
    days = sorted(int(d) for d in ref["area"].index)
    fig = plt.figure(figsize=(S.DOUBLE, 118 * S.MM))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.3)
    axA, axM, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    off = {a: o for a, o in zip(R.archs, np.linspace(-0.9, 0.9, len(R.archs)))}
    for q, ax, lab in (("area", axA, "Projected area\n(per-day median, µm²)"),
                       ("dry_mass", axM, "Dry mass\n(per-day median, pg)")):
        ax.plot(days, [ref[q][d] for d in days], color=S.REFERENCE_COLOR, lw=1.4, marker="o", ms=3, label="annotation",
                zorder=5)
        for a in R.archs:
            c = rep[a]["selected_condition"]
            rs = [r for r in st["runs"] if arch_disp(r["arch"]) == a and r["condition"] == c and r["arm"] == arm]
            M = np.array([[r["per_day_median"][q].get(str(d), np.nan) for d in days] for r in rs], float)
            mu = np.nanmean(M, 0)
            ax.errorbar(np.array(days) + off[a], mu, yerr=[mu - np.nanmin(M, 0), np.nanmax(M, 0) - mu], fmt=S.ARCH_MARKER[a],
                        ms=3, color=S.ARCH_COLOR[a], mec="white", mew=0.4, elinewidth=0.8, capsize=0, lw=0,
                        label=f"{a} ({cond_label_m(c)})", zorder=4)
            ax.plot(np.array(days) + off[a], mu, color=S.ARCH_COLOR[a], lw=0.6, alpha=0.7, zorder=3)
            if q == "dry_mass":
                B = M - np.array([ref[q][d] for d in days])[None, :]
                mb = np.nanmean(B, 0)
                axB.errorbar(np.array(days) + off[a], mb, yerr=[mb - np.nanmin(B, 0), np.nanmax(B, 0) - mb],
                             fmt=S.ARCH_MARKER[a], ms=3, color=S.ARCH_COLOR[a], mec="white", mew=0.4, elinewidth=0.8,
                             lw=0, label=a)
                axB.plot(np.array(days) + off[a], mb, color=S.ARCH_COLOR[a], lw=0.6, alpha=0.7)
        ax.set_ylabel(lab); ax.set_xlabel("Storage day"); S.hgrid(ax)
    axB.axhline(0, color=S.INK2, lw=0.6)
    axB.set_ylabel("Dry mass, prediction − annotation\n(per-day median, pg)"); axB.set_xlabel("Storage day"); S.hgrid(axB)
    # composition of the annotation
    pdy = {int(d["storage_day"]): d for d in ann["per_day"]}
    bottom = np.zeros(len(days))
    xb = np.arange(len(days))
    for k in R.classes:
        v = np.array([pdy[d].get(f"pct_{k}", np.nan) for d in days], float)
        axC.bar(xb, v, bottom=bottom, color=S.CLASS_COLOR.get(k, S.MUTED), width=0.82, ec="white", lw=0.5, label=cls_name(k))
        bottom += v
    axC.set_xticks(xb); axC.set_xticklabels([str(d) for d in days])
    axC.set_ylim(0, 100); axC.set_ylabel("Annotated cells (%)"); axC.set_xlabel("Storage day")
    axC.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.27), handlelength=1.0, columnspacing=0.8,
               frameon=False)
    axA.legend(loc="lower left", fontsize=7)
    for ax, l in zip((axA, axM, axB, axC), "abcd"):
        S.panel_title(ax, l, pad=9)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.17)
    ta, tm = ann["trends"]["area"], ann["trends"]["dry_mass"]
    cfg = inp.config().get("storage", {})
    parts = []
    for a in R.archs:
        c = rep[a]["selected_condition"]
        rs = [r for r in st["runs"] if arch_disp(r["arch"]) == a and r["condition"] == c and r["arm"] == arm]
        b = [r["dry_mass"]["bias_vs_annotation"] for r in rs]
        parts.append(f"{a} rho {min(x['rho'] for x in b):.2f} to {max(x['rho'] for x in b):.2f}, p "
                     f"{fmt_p(min(x['p_block_permutation'] for x in b))} to {fmt_p(max(x['p_block_permutation'] for x in b))}")
    cap = ("Storage time. (a) Projected area and (b) dry mass of the cells: per-day median of the per-image medians, "
           f"for the annotation (black) and each representative configuration (markers: mean of the {len(R.seeds)} "
           "seeds; whiskers: range over seeds; markers offset horizontally for legibility). (c) Prediction minus "
           "annotation of the per-day median dry mass. (d) Morphology composition of the annotated cells. In the "
           f"annotation, the median area changes by {ta['early_to_late_change_pct']:.1f} % from early (days ≤ "
           f"{cfg.get('early_max_day', '?')}) to late (days ≥ {cfg.get('late_min_day', '?')}) storage (Spearman rho "
           f"{ta['rho']:.2f}, date-block permutation p = {fmt_p(ta['p_block_permutation'])}) and the median dry mass by "
           f"{tm['early_to_late_change_pct']:.2f} % (rho {tm['rho']:.3f}, p = {fmt_p(tm['p_block_permutation'])}). "
           "Association of the dry-mass bias with storage day: " + "; ".join(parts) + ".")
    return fig, "main", "fig8_storage", "8", cap


# ================================================================================================ supplementary

def figS1_training_curves(inp: Inputs, R: Runs, arm: str = PROPOSED):
    import pandas as pd
    rep = _rep(inp)
    qs = [("val_dice", "Validation Dice"), ("val_ipe_global", "Validation global integrated-\nphase error (%)")]
    fig, axes = plt.subplots(len(qs), len(R.archs), figsize=(S.DOUBLE, 100 * S.MM), sharex=True)
    found = 0
    ls = [(0, ()), (0, (4, 1.5)), (0, (1, 1.2))]
    for j, a in enumerate(R.archs):
        c = rep[a]["selected_condition"]
        for si, s in enumerate(R.seeds):
            rel = f"{run_dir(a, c, arm, s, 'training_results')}/history.csv"
            if not inp.exists(rel):
                continue
            H = pd.read_csv(inp.need(rel)); found += 1
            for i, (k, lab) in enumerate(qs):
                sc = 100.0 if k == "val_ipe_global" else 1.0
                axes[i, j].plot(H["epoch"], H[k] * sc, color=S.ARCH_COLOR[a], lw=0.9, ls=ls[si % 3], label=f"seed {s}")
        for i, (k, lab) in enumerate(qs):
            ax = axes[i, j]; S.hgrid(ax)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                S.panel_key_title(ax, "abc"[j], f"{a} ({cond_label_m(c)})", S.ARCH_COLOR[a])
            else:
                S.panel_title(ax, "def"[j])
            if i == len(qs) - 1:
                if j == len(R.archs) // 2:
                    ax.set_xlabel("Epoch")
                ax.set_yscale("log")
                from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
        axes[0, j].legend(loc="lower right", fontsize=7)
    if not found:
        raise MissingInput("training_results/<architecture>/<rank>/dice_cipp/seed*/history.csv")
    fig.tight_layout(h_pad=1.2)
    cap = ("Training of the representative configurations under the proposed objective: Dice (top) and global "
           "integrated-phase error (bottom, log scale) on the validation fold after every epoch, one line per seed. "
           "The validation fold is part of the training split; the test images are never seen in training. The last "
           "epoch is the reported checkpoint.")
    return fig, "supplementary", "figS3_training_curves", "S3", cap


def figS2_class_retention(inp: Inputs, R: Runs):
    fig, axes = plt.subplots(1, len(R.archs), figsize=(S.DOUBLE, 72 * S.MM), sharey=True)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ffffff", "#d8e6f6", "#8db5e6", "#2a5fa8"][:len(R.seeds) + 1])
    totals = {}
    for j, a in enumerate(R.archs):
        ax = axes[j]
        M = np.array([[sum((r.get("n_lost_classes") or 0) > 0 for r in R.get(a, c, arm)) for c in R.conditions]
                      for arm in R.arms], float)
        totals[a] = int(M.sum())
        ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=len(R.seeds) + 0.5, aspect="auto")
        for (yy, xx), v in np.ndenumerate(M):
            ax.text(xx, yy, f"{int(v)}", ha="center", va="center", fontsize=7.5, color="white" if v >= len(R.seeds) else "black")
        ax.set_xticks(range(len(R.conditions)))
        ax.set_xticklabels(["FFT" if c == "fullft" else str(c)[1:] for c in R.conditions])
        ax.set_yticks(range(len(R.arms))); ax.set_yticklabels([arm_math(x) for x in R.arms])
        S.panel_title(ax, "abc"[j], a)
        ax.tick_params(length=0, labelcolor="black")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(R.conditions)), minor=True); ax.set_yticks(np.arange(-0.5, len(R.arms)), minor=True)
        ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    cap = (f"Training stability: number of seeds (of {len(R.seeds)}) in which a morphology class was lost (Dice < "
           f"{R.lost_class_dice}), per objective and adaptation. Runs that lost a class, out of "
           f"{len(R.arms) * len(R.conditions) * len(R.seeds)} per architecture: "
           + ", ".join(f"{a} {v}" for a, v in totals.items()) + ". FFT, full fine-tuning. " + LOSS_GLOSSARY)
    return fig, "supplementary", "figS4_class_retention", "S4", cap


def figS3_per_class(inp: Inputs, R: Runs, arm: str = PROPOSED):
    rep = _rep(inp)
    fig, ax = plt.subplots(figsize=(S.ONEHALF, 58 * S.MM))
    n = len(R.archs)
    for j, a in enumerate(R.archs):
        c = rep[a]["selected_condition"]
        for i, k in enumerate(R.classes):
            v = R.col(a, c, arm, f"dice_{k}")
            x = i + (j - (n - 1) / 2) * 0.24
            ax.bar(x, np.mean(v) if v else 0, width=0.21, color=S.ARCH_COLOR[a], alpha=0.85,
                   label=f"{a} ({cond_label_m(c)})" if i == 0 else None)
            ax.plot([x] * len(v), v, ls="", marker="o", ms=2.2, color=S.INK, zorder=3)
    if R.lost_class_dice is not None:
        ax.axhline(R.lost_class_dice, color=S.INK2, lw=0.6, ls=(0, (3, 2)))
        ax.text(len(R.classes) - 0.5, R.lost_class_dice, " lost-class\n threshold", va="bottom", fontsize=7, color=S.INK2)
    ax.set_xticks(range(len(R.classes))); ax.set_xticklabels([cls_name(k) for k in R.classes])
    ax.set_ylabel("Dice"); ax.set_ylim(0, 1); S.hgrid(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(R.archs), fontsize=7.5, frameon=False,
              handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    cap = ("Per-class Dice of the representative configurations under the proposed objective. Bars: mean of the "
           f"{len(R.seeds)} seeds; dots: single seeds. Values in Table S4.")
    return fig, "supplementary", "figS6_per_class_dice", "S6", cap


def figS4_missed_by_phase(inp: Inputs, R: Runs):
    q = inp.json("results/consolidated/phase_quartile_fnr/phase_quartile_fnr.json")
    nb = q["bins"]
    edges = q["bin_edges_rad"]
    labs = [f"< {edges[0]:.2f}"] + [f"{edges[i]:.2f}–{edges[i + 1]:.2f}" for i in range(len(edges) - 1)] + [f"≥ {edges[-1]:.2f}"]
    fig, axes = plt.subplots(1, len(R.archs), figsize=(S.DOUBLE, 66 * S.MM), sharey=True)
    arms = q["arms"]
    for j, a in enumerate(R.archs):
        ax = axes[j]
        rs = [r for r in q["runs"] if arch_disp(r["architecture_display"]) == a]
        for ai, arm in enumerate(arms):
            M = np.array([[r[f"fnr_bin{b + 1}_pct"] for b in range(nb)] for r in rs if r["loss"] == arm], float)
            if not len(M):
                continue
            x = np.arange(nb) + (ai - (len(arms) - 1) / 2) * 0.12
            ax.plot(x, M.mean(0), color=S.ARCH_COLOR[a], lw=1.2, ls="-" if arm == PROPOSED else (0, (3, 2)),
                    marker="o" if arm == PROPOSED else "s", ms=3, mfc=S.ARCH_COLOR[a] if arm == PROPOSED else "white",
                    label=arm_math(arm))
            for row in M:
                ax.plot(x, row, ls="", marker=".", ms=2, color=S.INK2, alpha=0.6)
        cond = rs[0]["condition"] if rs else ""
        S.panel_title(ax, "abc"[j], f"{a} ({cond_label_m(cond)})")
        ax.set_xticks(range(nb)); ax.set_xticklabels(labs, fontsize=7.5, rotation=45, ha="right",
                                                     rotation_mode="anchor")
        if j == len(R.archs) // 2:
            ax.set_xlabel("Phase of the annotated pixel (rad, quartiles)")
        S.hgrid(ax); ax.legend(fontsize=7, loc="upper center")
    axes[0].set_ylabel("Annotated cell pixels missed (%)")
    fig.tight_layout()
    cap = ("Annotated cell pixels predicted as background, by quartile of the pixel's phase over the "
           f"{q['n_test_images']} test images. Lines: mean over seeds; dots: single seeds. The low-phase quartile "
           "holds the cell rims and the thinnest cells. " + LOSS_GLOSSARY)
    return fig, "supplementary", "figS7_missed_pixels_by_phase", "S7", cap


def figS5_crowding_fov(inp: Inputs, R: Runs):
    import pandas as pd
    w5 = inp.json(f"{W}/w5_reference_cells/w5_reference_cells.json")["results"]
    fov = inp.json(f"{W}/w6_superposition_and_fov/w6_fov.json")
    C = pd.DataFrame(w5["crowding_per_day"])
    P = pd.DataFrame(fov["per_image"])
    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, 62 * S.MM), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax = axes[0]
    xb = np.arange(len(C))
    ax.bar(xb, C["scored_cells"], color="#2a78d6", width=0.85, label="scored")
    ax.bar(xb, C["reference_cells"] - C["scored_cells"], bottom=C["scored_cells"], color="#a9c9ef", width=0.85,
           label="not scored")
    ax.set_xticks(xb); ax.set_xticklabels(C["storage_day"].astype(int), fontsize=6.5)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.set_xlim(-0.6, len(xb) - 0.4)
    ax.set_ylabel("Reference cells (test images)"); ax.legend(loc="upper right", fontsize=7); S.hgrid(ax)
    ax = axes[1]
    ax.plot(C["storage_day"], C["touching_pct"], color="#eb6834", marker="o", ms=3.5, lw=1.2, mec="white", mew=0.4)
    ax.set_xlabel("Storage day"); ax.set_ylabel("Reference cells touching\nanother (%)"); S.hgrid(ax)
    ax = axes[2]
    names = {"train": "training split", "val": "test split"}
    for k, g in P.groupby("split"):
        # the two splits: colour and marker, named in the legend
        ax.scatter(g["storage_day"] + (0.6 if k == "val" else -0.6), 100 * g["unannotated_fraction"], s=9,
                   marker="o" if k == "val" else "^", color="#1baf7a" if k == "val" else "#6250d6", lw=0,
                   alpha=0.8, label=f"{'Test' if k == 'val' else 'Training'} images ({len(g)})")
    ax.set_ylabel("Field without annotation (%)")
    ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.9, edgecolor="none", handletextpad=0.3,
              borderpad=0.3)
    S.hgrid(ax)
    for ax, l in zip(axes, "abc"):
        S.panel_title(ax, l)
    fig.tight_layout(w_pad=1.5)
    ps = fov["results"]["per_split"]
    cap = ("Reference cells and annotated field. (a) Reference cells of the test images per storage day, scored "
           "(area gate passed, clear of the unannotated area and of phase wrapping) and not scored. (b) Share of "
           "reference cells that touch another reference cell. (c) Share of each image's field of view without "
           "annotation; mean "
           + ", ".join(f"{100 * v['unannotated_mean']:.1f} % ({names.get(k, k)})" for k, v in ps.items())
           + f", {100 * fov['results']['all_images']['unannotated_mean']:.1f} % over all images. Unannotated pixels "
           "are excluded from training and evaluation.")
    return fig, "supplementary", "figS1_crowding_and_field_of_view", "S1", cap


MAIN = [fig1_framework, fig3_objectives, fig4_cell_paired, fig5_rank_selection, fig7_dry_mass, fig8_storage]
SUPP = [figS1_training_curves, figS2_class_retention, figS3_per_class, figS4_missed_by_phase, figS5_crowding_fov,
        figS6_cell_paired_touching]
