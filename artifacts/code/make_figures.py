#!/usr/bin/env python
"""Figures of the revised manuscript and SI from outputs/ (read only). PNG (300 dpi) and PDF.

    python artifacts/code/make_figures.py [--only F3 F4 ...]

Conventions: bars/lines are the mean over seeds 42, 43, 44; every seed is drawn as a dot; a run that lost a class (Dice < 0.05)
is drawn as a black cross. Colour follows the architecture (MobileSAM blue, EdgeSAM orange, MobileNet-UNet green) in every
figure; the loss ablation uses a light-to-dark ramp of one hue. No dual axes. Class names in the manuscript order (D-25).

The qualitative figures that need the phase maps (data/processed, not in the repository) are in make_qualitative_figures.py.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import ARCH_LABEL, CLASSES, RANKS, SEEDS  # noqa: E402

ARCH_COLOR = {"mobile_sam": "#2a78d6", "edge_sam": "#eb6834", "mobilenet_unet": "#1baf7a"}
CLASS_COLOR = {"discocyte": "#2a78d6", "spherocyte": "#eda100", "echinocyte": "#6250d6", "stomatocyte": "#e87ba4"}
LOSS_COLOR = {"dice_only": "#c9d9ef", "pmc": "#8fb3e3", "pmc_bga": "#4c84cf", "full": "#1d4b8f"}
LOSS_SHORT = {"dice_only": "Dice only", "pmc": "+PMC", "pmc_bga": "+PMC+BGA", "full": "+PMC+BGA+IPP"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 300, "pdf.fonttype": 42, "font.family": "DejaVu Sans", "legend.frameon": False})

FIGS: Dict[str, Dict] = {}


def figure(fid: str, part: str):
    def deco(fn):
        FIGS[fid] = {"fn": fn, "part": part}
        return fn
    return deco


def save(fig, part: str, name: str) -> None:
    d = C.FIGURES / part
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", bbox_inches="tight")
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def seed_vals(arch, rank, loss, name) -> Dict[int, float]:
    return {int(s): v for s, v in C.metric(arch, rank, loss, name)["per_seed"].items() if v is not None}


def lost_seeds(arch, rank, loss) -> set:
    c = C.cfg_entry(arch, rank, loss)
    return {int(s) for s in c["seeds_found"]
            if any(not c["metrics"][f"dice_{cl}"]["per_seed"][str(s)] >= C.COLLAPSE_DICE for cl in CLASSES)}


def dots(ax, x, vals: Dict[int, float], lost: set, width=0.08, color=INK, size=10, lost_color=INK):
    for i, s in enumerate(sorted(vals)):
        xx = x + (i - 1) * width
        if s in lost:
            ax.scatter(xx, vals[s], marker="x", s=size + 8, color=lost_color, linewidths=1.1, zorder=6)
        else:
            ax.scatter(xx, vals[s], s=size, facecolor="white", edgecolor=color, linewidths=0.9, zorder=6)


def lost_handle():
    return Line2D([], [], marker="x", ls="", color=INK, label="seed with a lost class (Dice < 0.05)")


def seed_handle():
    return Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="individual seed")


# ================================================================================================ MAIN
@figure("F3", "main")
def f3():
    """Per-class Dice: preferred configurations and the same-rank comparison."""
    cfgs = [("mobile_sam", 8, None), ("edge_sam", 32, None), ("edge_sam", 8, "////"), ("mobilenet_unet", 8, None)]
    cats = CLASSES + ["mean"]
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    w = 0.2
    for j, (a, rk, hatch) in enumerate(cfgs):
        for i, cat in enumerate(cats):
            name = "mean_dice" if cat == "mean" else f"dice_{cat}"
            v = seed_vals(a, rk, "full", name)
            x = i + (j - 1.5) * w
            ax.bar(x, np.mean(list(v.values())), w * 0.92, color=ARCH_COLOR[a] if not hatch else "white",
                   edgecolor=ARCH_COLOR[a], hatch=hatch, linewidth=0.8)
            dots(ax, x, v, {s for s in lost_seeds(a, rk, "full") if v[s] < C.COLLAPSE_DICE or cat == "mean"}, width=0.05)
    ax.set_xticks(range(len(cats)), [C.CLASS_DISPLAY[c].capitalize() if c != "mean" else "Mean (4 classes)" for c in cats])
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.0)
    handles = [Patch(facecolor=ARCH_COLOR[a] if not h else "white", edgecolor=ARCH_COLOR[a], hatch=h,
                     label=f"{ARCH_LABEL[a]} r = {rk}") for a, rk, h in cfgs] + [seed_handle(), lost_handle()]
    ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    save(fig, "main", "F3_per_class_dice")


@figure("F4", "main")
def f4():
    """Rank sweep: mean Dice, AJI, boundary F1 and global IPE vs LoRA rank, all seeds."""
    mets = [("mean_dice", "Mean Dice"), ("aji", "AJI"), ("bf1", "Boundary F1"), ("ipe_global_pct", "Global IPE (%)")]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.4), gridspec_kw={"wspace": 0.42})
    xs = np.log2(RANKS)
    for ax, (m, lab) in zip(axes, mets):
        for k, a in enumerate(C.ARCHS):
            means = []
            for x, rk in zip(xs, RANKS):
                v = seed_vals(a, rk, "full", m)
                means.append(np.mean(list(v.values())))
                dots(ax, x + (k - 1) * 0.12, v, lost_seeds(a, rk, "full"), width=0.0, color=ARCH_COLOR[a], size=7,
                     lost_color=ARCH_COLOR[a])
            ax.plot(xs + (k - 1) * 0.12, means, color=ARCH_COLOR[a], lw=1.6, label=ARCH_LABEL[a])
        ax.set_xticks(xs, [str(r) for r in RANKS])
        ax.set_xlabel("LoRA rank r")
        ax.set_title(lab)
    axes[0].legend(handles=[Line2D([], [], color=ARCH_COLOR[a], lw=1.6, label=ARCH_LABEL[a]) for a in C.ARCHS] +
                   [Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, label="seed"),
                    Line2D([], [], marker="x", ls="", color=MUTED, label="seed with a lost class")],
                   loc="upper center", bbox_to_anchor=(2.6, 1.42), ncol=5)
    save(fig, "main", "F4_rank_sweep")


@figure("F5", "main")
def f5():
    """Training curves (validation Dice per epoch) of the three seeds; runs that lost a class dashed."""
    cfgs = [("mobile_sam", 8), ("mobile_sam", 32), ("edge_sam", 8), ("edge_sam", 32)]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), sharey=True)
    for ax, (a, rk) in zip(axes, cfgs):
        lost = lost_seeds(a, rk, "full")
        for s, shade in zip(SEEDS, (1.0, 0.75, 0.5)):
            h = C.history(a, rk, "full", s)
            ax.plot(h.epoch, h.val_mean_dice, color=ARCH_COLOR[a], alpha=shade, lw=1.3, ls="--" if s in lost else "-",
                    label=f"seed {s}" + (" (lost class)" if s in lost else ""))
        ax.set_title(f"{ARCH_LABEL[a]} r = {rk}")
        ax.set_xlabel("Epoch")
        ax.legend(loc="lower right", fontsize=6)
    axes[0].set_ylabel("Validation mean Dice")
    axes[0].set_ylim(0, 0.9)
    save(fig, "main", "F5_training_curves")


@figure("F6", "main")
def f6():
    """Loss ablation: global IPE and mean Dice for each objective (preferred configurations and MobileNet-UNet r8)."""
    cfgs = [("mobile_sam", 8), ("edge_sam", 32), ("edge_sam", 8), ("mobilenet_unet", 8)]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True)
    for j, (a, rk) in enumerate(cfgs):
        for i, (m, lab) in enumerate((("ipe_global_pct", "Global IPE (%)"), ("mean_dice", "Mean Dice"))):
            ax = axes[i, j]
            for k, loss in enumerate(C.LOSSES):
                v = seed_vals(a, rk, loss, m)
                ax.bar(k, np.mean(list(v.values())), 0.75, color=LOSS_COLOR[loss], edgecolor="white", linewidth=1)
                dots(ax, k, v, lost_seeds(a, rk, loss), width=0.16)
            if i == 0:
                ax.set_title(f"{ARCH_LABEL[a]} r = {rk}")
            if j == 0:
                ax.set_ylabel(lab)
            ax.set_xticks(range(4), [LOSS_SHORT[l] for l in C.LOSSES], rotation=35, ha="right")
    axes[0, 0].legend(handles=[seed_handle(), lost_handle()], loc="upper center", bbox_to_anchor=(2.4, 1.45), ncol=2)
    save(fig, "main", "F6_loss_ablation")


@figure("F7", "main")
def f7():
    """LoRA (preferred rank, 3 seeds) vs full fine-tuning (single seed-42 run): per-class Dice."""
    pref = dict(C.PREFERRED + [C.BASELINE])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    for ax, a in zip(axes, C.ARCHS):
        for i, cl in enumerate(CLASSES):
            v = seed_vals(a, pref[a], "full", f"dice_{cl}")
            ax.bar(i - 0.2, np.mean(list(v.values())), 0.38, color=ARCH_COLOR[a])
            dots(ax, i - 0.2, v, {s for s in v if v[s] < C.COLLAPSE_DICE}, width=0.07)
            ft = C.metric(a, None, "full_finetune", f"dice_{cl}")["per_seed"]["42"]
            ax.bar(i + 0.2, ft, 0.38, color="white", edgecolor=ARCH_COLOR[a], hatch="////", linewidth=0.8)
            if ft < C.COLLAPSE_DICE:
                ax.scatter(i + 0.2, 0.02, marker="x", color=INK, s=18, zorder=6)
        ax.set_xticks(range(4), [C.CLASS_SHORT[c] for c in CLASSES])
        ax.set_title(f"{ARCH_LABEL[a]} (LoRA r = {pref[a]})")
    axes[0].set_ylabel("Dice")
    axes[0].set_ylim(0, 1)
    axes[1].legend(handles=[Patch(color=INK, label="LoRA (mean of 3 seeds)"), Patch(facecolor="white", edgecolor=INK, hatch="////",
                                                                                   label="full fine-tuning (seed 42)"), lost_handle()],
                   loc="upper center", bbox_to_anchor=(0.5, 1.4), ncol=3)
    save(fig, "main", "F7_lora_vs_full_finetuning")


@figure("F8", "main")
def f8():
    """Accuracy against latency. Latency is rank independent after the LoRA merge; points for both ranks share the x value."""
    b = C.compiled("benchmark")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, be, title in zip(axes, ("pytorch", "onnx"), ("PyTorch FP32", "ONNX Runtime FP32")):
        for a in C.ARCHS:
            r = next(x for x in b["rows"] if x["architecture"] == a and x["backend"] == be)
            lat, lsd = r["latency_p50_ms"]["mean"], r["latency_p50_ms"]["sd"]
            ranks = sorted({8, dict(C.PREFERRED + [C.BASELINE])[a]})
            for rk in ranks:
                m = C.metric(a, rk, "full", "mean_dice")
                pref = (a, rk) in C.PREFERRED or (a, rk) == C.BASELINE
                ax.errorbar(lat, m["mean"], xerr=lsd, yerr=m["sd"], fmt="o" if pref else "s", ms=6, color=ARCH_COLOR[a],
                            mfc=ARCH_COLOR[a] if pref else "white", capsize=2, lw=1)
                ax.annotate(f"{ARCH_LABEL[a]} r{rk}", (lat, m["mean"]), textcoords="offset points", xytext=(6, 3), fontsize=6.5, color=INK)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 30], ["2", "5", "10", "20", "30"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(1.8, 45)
        ax.set_xlabel("Median latency P50 (ms, batch 1, log scale)")
        ax.set_title(title)
    axes[0].set_ylabel("Mean Dice (± SD over seeds)")
    axes[0].set_ylim(0.3, 0.9)
    save(fig, "main", "F8_accuracy_vs_latency")


def per_image_medians(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("stem")[col].median()


@figure("F9", "main")
def f9():
    """Dry-mass agreement (per-image median dry mass, wrapped cells excluded): scatter and Bland-Altman, all seeds."""
    gt = per_image_medians(C.cells_gt(), "dry_mass_pg")
    cfgs = C.PREFERRED
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.2))
    markers = {42: "o", 43: "s", 44: "^"}
    for j, (a, rk) in enumerate(cfgs):
        diffs_all, means_all = [], []
        for s in SEEDS:
            pr = per_image_medians(C.cells_pred(a, rk, "full", s), "dry_mass_pg")
            jn = pd.concat([pr, gt], axis=1, keys=["pred", "ann"]).dropna()   # not "gt": DataFrame.gt is a method
            mdare = 100 * ((jn.pred - jn.ann) / jn.ann).abs().median()
            ref = C.metric(a, rk, "full", "mdare_pct")["per_seed"][str(s)]
            if abs(mdare - ref) > 1e-6:
                raise RuntimeError(f"recomputed MdARE {mdare} differs from the pipeline value {ref} ({a} r{rk} s{s})")
            axes[0, j].scatter(jn.ann, jn.pred, s=12, marker=markers[s], facecolor="white", edgecolor=ARCH_COLOR[a], lw=0.8, label=f"seed {s}")
            d, mn = jn.pred - jn.ann, (jn.pred + jn.ann) / 2
            axes[1, j].scatter(mn, d, s=12, marker=markers[s], facecolor="white", edgecolor=ARCH_COLOR[a], lw=0.8)
            diffs_all.append(d)
            means_all.append(mn)
        lim = [20, 45]
        axes[0, j].plot(lim, lim, color=MUTED, lw=1, ls="--")
        axes[0, j].set_xlim(lim)
        axes[0, j].set_ylim(lim)
        axes[0, j].set_xlabel("Annotation: per-image median dry mass (pg)")
        axes[0, j].set_ylabel("Prediction (pg)")
        m = C.cfg_entry(a, rk, "full")["metrics"]
        axes[0, j].set_title(f"{ARCH_LABEL[a]} r = {rk} (MdARE {m['mdare_pct']['mean']:.1f} ± {m['mdare_pct']['sd']:.1f}%)")
        axes[0, j].legend(loc="upper left", fontsize=6)
        d = pd.concat(diffs_all)
        bias, sd = d.mean(), d.std(ddof=1)
        for y, ls in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
            axes[1, j].axhline(y, color=ARCH_COLOR[a], lw=1, ls=ls)
        axes[1, j].axhline(0, color=MUTED, lw=0.8)
        axes[1, j].set_xlabel("Mean of prediction and annotation (pg)")
        axes[1, j].set_ylabel("Prediction − annotation (pg)")
        axes[1, j].set_title("Bland-Altman (3 seeds pooled)")
        axes[1, j].text(0.98, 0.97, f"bias {bias:+.2f} pg\nLoA {bias - 1.96 * sd:.1f} to {bias + 1.96 * sd:.1f} pg", transform=axes[1, j].transAxes,
                        ha="right", va="top", fontsize=6.5, color=INK)
    fig.tight_layout()
    save(fig, "main", "F9_dry_mass_agreement")


def per_day_image_median(df: pd.DataFrame, col: str) -> pd.Series:
    img = df.groupby(["storage_day", "stem"])[col].median().reset_index()
    return img.groupby("storage_day")[col].mean()


@figure("F10", "main")
def f10():
    """Storage trends of per-image median area and dry mass: annotation vs predicted masks (mean over seeds, band = min-max)."""
    gt = C.cells_gt()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, col, lab in zip(axes, ("area_um2", "dry_mass_pg"), ("Projected area (µm²)", "Dry mass (pg)")):
        g = per_day_image_median(gt, col)
        ax.plot(g.index, g.values, color=INK, lw=1.8, marker="o", ms=3.5, label="Annotation")
        for a, rk in C.PREFERRED + [C.BASELINE]:
            ser = pd.concat([per_day_image_median(C.cells_pred(a, rk, "full", s), col) for s in SEEDS], axis=1)
            ax.plot(ser.index, ser.mean(axis=1), color=ARCH_COLOR[a], lw=1.4, marker="o", ms=3, label=f"{ARCH_LABEL[a]} r = {rk}")
            ax.fill_between(ser.index, ser.min(axis=1), ser.max(axis=1), color=ARCH_COLOR[a], alpha=0.18, lw=0)
        ax.set_xlabel("Storage day")
        ax.set_ylabel(lab)
    axes[1].set_ylim(20, 40)
    axes[0].set_title("Per-image median area (mean of the 6 images per day)")
    axes[1].set_title("Per-image median dry mass (mean of the 6 images per day)")
    axes[0].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "main", "F10_storage_trends")


@figure("F11", "main")
def f11():
    """Morphology composition per storage day: annotation vs predicted (mean over seeds)."""
    ann = C.compiled("cell_analysis")["annotation"]["per_day_nowrap"]
    panels = [("Annotation", {d["storage_day"]: d["composition_pct"] for d in ann})]
    for a, rk in C.PREFERRED:
        comp = {}
        for d in C.cfg_entry(a, rk, "full")["per_day_pred_nowrap"]:
            comp[d["storage_day"]] = {cl: d["metrics"][f"composition_pct.{cl}"]["mean"] for cl in CLASSES}
        panels.append((f"{ARCH_LABEL[a]} r = {rk}", comp))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, (title, comp) in zip(axes, panels):
        days = sorted(comp)
        bottom = np.zeros(len(days))
        for cl in CLASSES:
            v = np.array([comp[d][cl] for d in days])
            ax.bar(range(len(days)), v, 0.82, bottom=bottom, color=CLASS_COLOR[cl], edgecolor="white", linewidth=0.8, label=C.CLASS_DISPLAY[cl])
            bottom += v
        ax.set_xticks(range(len(days)), [str(d) for d in days], fontsize=6)
        ax.set_xlabel("Storage day")
        ax.set_title(title)
        ax.grid(False)
    axes[0].set_ylabel("Cells (%)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.33), ncol=4)
    save(fig, "main", "F11_composition")


# ================================================================================================ SI
@figure("FS1", "si")
def fs1():
    """Rank sweep per class (full objective)."""
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.2), sharey=True, gridspec_kw={"wspace": 0.15})
    xs = np.log2(RANKS)
    for ax, cl in zip(axes, CLASSES):
        for k, a in enumerate(C.ARCHS):
            means = []
            for x, rk in zip(xs, RANKS):
                v = seed_vals(a, rk, "full", f"dice_{cl}")
                means.append(np.mean(list(v.values())))
                dots(ax, x + (k - 1) * 0.12, v, {s for s in v if v[s] < C.COLLAPSE_DICE}, width=0.0, color=ARCH_COLOR[a], size=7,
                     lost_color=ARCH_COLOR[a])
            ax.plot(xs + (k - 1) * 0.12, means, color=ARCH_COLOR[a], lw=1.4, label=ARCH_LABEL[a])
        ax.set_xticks(xs, [str(r) for r in RANKS])
        ax.set_title(C.CLASS_DISPLAY[cl].capitalize())
        ax.set_xlabel("LoRA rank r")
    axes[0].set_ylabel("Dice")
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.3, 1.38), ncol=3)
    save(fig, "si", "FS1_rank_sweep_per_class")


@figure("FS2", "si")
def fs2():
    """Global IPE (mean over seeds) of every architecture, loss and rank; number of runs with a lost class in brackets."""
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4), sharey=True)
    for ax, a in zip(axes, C.ARCHS):
        mat = np.array([[C.metric(a, rk, l, "ipe_global_pct")["mean"] for rk in RANKS] for l in C.LOSSES])
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=21, aspect="auto")
        for i, l in enumerate(C.LOSSES):
            for j, rk in enumerate(RANKS):
                n = C.n_lost_runs(a, rk, l)
                ax.text(j, i, f"{mat[i, j]:.1f}" + (f"\n({n}/3)" if n else ""), ha="center", va="center", fontsize=5.8,
                        color="white" if mat[i, j] > 11 else INK)
        ax.set_xticks(range(5), [str(r) for r in RANKS])
        ax.set_yticks(range(4), [LOSS_SHORT[l] for l in C.LOSSES])
        ax.set_xlabel("LoRA rank r")
        ax.set_title(ARCH_LABEL[a])
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Global IPE (%), mean of 3 seeds")
    save(fig, "si", "FS2_ablation_ipe_heatmap")


@figure("FS3", "si")
def fs3():
    """Cumulative distribution of the per-cell |IPE| (matched, not wrapped cells; three seeds pooled)."""
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for a, rk in C.PREFERRED + [C.BASELINE]:
        v = []
        for s in SEEDS:
            p = pd.read_csv(C.OUT / "cell_analysis" / "phase_integral" / f"{C.run_id(a, rk, 'full', s)}__per_cell.csv")
            p = p[p.matched.astype(bool) & ~p.wrapped.astype(bool)]
            v.append(100 * p.err_abs)
        v = np.sort(pd.concat(v).values)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=ARCH_COLOR[a], lw=1.5, label=f"{ARCH_LABEL[a]} r = {rk}")
    ax.set_xscale("log")
    ax.set_xlabel("Per-cell |IPE| (%)")
    ax.set_ylabel("Fraction of matched cells")
    ax.axvline(10, color=MUTED, lw=0.8, ls="--")
    ax.legend(loc="lower right", fontsize=6.5)
    save(fig, "si", "FS3_percell_ipe_cdf")


@figure("FS4", "si")
def fs4():
    """Robustness: Dice and IPE under synthetic background distortions (tilt, bowl, astigmatism)."""
    r = C.compiled("robustness")["configurations"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    for ax, kind in zip(axes, ("tilt", "bowl", "astig")):
        for a, rk in C.PREFERRED + [C.BASELINE]:
            d = r[C.config_key(a, rk, "full")]["distortions"]
            amps = [0.0, 0.1, 0.25, 0.5]
            ys = [d["none"]["macro_dice"]["mean"]] + [d[f"{kind}{x:.2f}"]["macro_dice"]["mean"] for x in amps[1:]]
            ax.plot(amps, ys, marker="o", ms=3, color=ARCH_COLOR[a], lw=1.4, label=f"{ARCH_LABEL[a]} r = {rk}")
        ax.set_title({"tilt": "Tilt", "bowl": "Defocus-like (bowl)", "astig": "Astigmatism-like"}[kind])
        ax.set_xlabel("Peak amplitude (rad)")
    axes[0].set_ylabel("Mean Dice")
    axes[0].legend(fontsize=6, loc="lower left")
    save(fig, "si", "FS4_robustness")


@figure("FS5", "si")
def fs5():
    """Trainable parameters versus LoRA rank."""
    b = C.compiled("benchmark")["parameters_by_rank"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for a in C.ARCHS:
        p = [x for x in b if x["architecture"] == a]
        ax.plot([x["lora_rank"] for x in p], [x["params_trainable"] / 1e6 for x in p], marker="o", ms=3, color=ARCH_COLOR[a], lw=1.5,
                label=ARCH_LABEL[a])
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS, [str(r) for r in RANKS])
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("Trainable parameters (M)")
    ax.legend(fontsize=6.5)
    save(fig, "si", "FS5_parameters_vs_rank")


@figure("FS6", "si")
def fs6():
    """Error strata: pixel error near the boundary and false negatives by phase quartile."""
    cfgs = C.PREFERRED + [C.BASELINE]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, prefix, lab in ((axes[0], "boundary band", "Pixel error rate (%)"), (axes[1], "false negatives in phase quartile", "False-negative rate (%)")):
        keys = [k for k in C.cfg_entry("edge_sam", 32, "full")["error_strata"] if k.startswith(prefix)]
        for j, (a, rk) in enumerate(cfgs):
            es = C.cfg_entry(a, rk, "full")["error_strata"]
            for i, k in enumerate(keys):
                v = {int(s): x for s, x in es[k]["per_seed"].items()}
                ax.bar(i + (j - 1) * 0.27, np.mean(list(v.values())), 0.25, color=ARCH_COLOR[a], label=f"{ARCH_LABEL[a]} r = {rk}" if i == 0 else None)
                dots(ax, i + (j - 1) * 0.27, v, set(), width=0.05, size=6)
        ax.set_xticks(range(len(keys)), [k.replace("boundary band ", "").replace("false negatives in phase quartile ", "") for k in keys])
        ax.set_xlabel("Distance from annotated boundary" if prefix.startswith("boundary") else "Phase quartile inside annotated cells (Q1 lowest)")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=6.5)
    save(fig, "si", "FS6_error_strata")


@figure("FS7", "si")
def fs7():
    """Training curves of every rank (full objective), three seeds each; runs with a lost class dashed."""
    fig, axes = plt.subplots(3, 5, figsize=(7.4, 4.6), sharex=True, sharey=True)
    for i, a in enumerate(C.ARCHS):
        for j, rk in enumerate(RANKS):
            ax = axes[i, j]
            lost = lost_seeds(a, rk, "full")
            for s, shade in zip(SEEDS, (1.0, 0.75, 0.5)):
                h = C.history(a, rk, "full", s)
                ax.plot(h.epoch, h.val_mean_dice, color=ARCH_COLOR[a], alpha=shade, lw=1.1, ls="--" if s in lost else "-")
            if i == 0:
                ax.set_title(f"r = {rk}")
            if j == 0:
                ax.set_ylabel(f"{ARCH_LABEL[a]}\nval. Dice")
            if i == 2:
                ax.set_xlabel("Epoch")
    axes[0, 0].set_ylim(0, 0.9)
    save(fig, "si", "FS7_training_curves_all_ranks")


@figure("FS8", "si")
def fs8():
    """Annotation-derived per-image median dry mass and area against storage day (the reference trends)."""
    gt = C.cells_gt()
    img = gt.groupby("stem").agg(day=("storage_day", "first"), area=("area_um2", "median"), dm=("dry_mass_pg", "median"))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3))
    for ax, col, lab in zip(axes, ("area", "dm"), ("Per-image median area (µm²)", "Per-image median dry mass (pg)")):
        ax.scatter(img.day, img[col], s=12, facecolor="white", edgecolor=INK, lw=0.8)
        ax.set_xlabel("Storage day")
        ax.set_ylabel(lab)
    axes[1].set_ylim(20, 40)
    save(fig, "si", "FS8_annotation_trends")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    failed: List[str] = []
    for fid, f in FIGS.items():
        if a.only and fid not in a.only:
            continue
        try:
            f["fn"]()
            print(f"[figures] {f['part']}/{fid} ok")
        except Exception:
            failed.append(fid)
            traceback.print_exc()
    if failed:
        print(f"[figures] FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
