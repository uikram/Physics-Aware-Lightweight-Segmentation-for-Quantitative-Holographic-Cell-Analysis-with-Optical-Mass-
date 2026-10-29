"""Figures of the revised manuscript and SI, one consistent style (read only on outputs/).

Main: Fig. 3 per-class Dice, Fig. 4 LoRA vs full fine-tuning, Fig. 5 loss ablation (Dice and IPE as separate panels, no
second y-axis), Fig. 7 class composition, Fig. 8 dry-mass agreement and storage trends.
SI: Fig. S-1 rank sweep (every seed), Fig. S-2 validation curves of the unstable configurations, Fig. S-3 accuracy versus
forward-pass throughput (rank 8).
Written to artifacts/figures/manuscript/ as PNG (300 dpi) and PDF.

Notation: configurations are written "EdgeSAM (r=32)". Colours: one hue per architecture, validated categorical slots
(MobileNet-UNet aqua, MobileSAM orange, EdgeSAM blue); EdgeSAM (r=8) uses a light tint of the EdgeSAM blue with hatching.
Class colours follow the qualitative figure (Fig. 6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (ART, CLASSES, COLLAPSE_DICE, LOSSES, OUT, RANKS, SEEDS, cells_gt, cells_pred, compiled, history,
                    runs_frame)

FIG = ART / "figures" / "manuscript"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "savefig.dpi": 300, "figure.dpi": 150, "hatch.linewidth": 0.6, "hatch.color": "white"})
INK, MUTED = "#2b2b2b", "#6b6b66"

ARCH_COL = {"mobilenet_unet": "#1baf7a", "mobile_sam": "#eb6834", "edge_sam": "#2a78d6"}
EDGE_R8 = "#9cc3ee"
CONFIGS = [("mobilenet_unet", 8), ("mobile_sam", 8), ("edge_sam", 8), ("edge_sam", 32)]
NAME = {"mobilenet_unet": "MobileNet-UNet", "mobile_sam": "MobileSAM", "edge_sam": "EdgeSAM"}
CLS_COL = {"discocyte": "#0072B2", "spherocyte": "#009E73", "echinocyte": "#D55E00", "stomatocyte": "#CC79A7"}
CLS_LAB = {"discocyte": "Discocyte (1)", "spherocyte": "Spherocyte (2)", "echinocyte": "Echinocyte (3)",
           "stomatocyte": "Stomatocyte (4)"}
LOSS_TICK = {"dice_only": "Dice", "pmc": "+PMC", "pmc_bga": "+BGA", "full": "+IPP"}
LOST = dict(marker="x", color="#c0392b", s=22, linewidths=1.1, zorder=6)
SEED_MARK = {42: "o", 43: "^", 44: "s"}


def label(arch, rank):
    return f"{NAME[arch]} (r={rank})"


def colour(arch, rank):
    return EDGE_R8 if (arch, rank) == ("edge_sam", 8) else ARCH_COL[arch]


def hatch(arch, rank):
    return "////" if (arch, rank) == ("edge_sam", 8) else None


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def jitter(n, w):
    return np.linspace(-w, w, n) if n > 1 else np.zeros(1)


RF = runs_frame()
LORA = RF[RF.loss != "full_finetune"]


def runs(arch, rank, loss="full"):
    return LORA[(LORA.arch == arch) & (LORA["rank"] == rank) & (LORA.loss == loss)].sort_values("seed")


# ---------------------------------------------------------------- Fig. 3 per-class Dice
def fig3():
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    w = 0.19
    for i, (a, r) in enumerate(CONFIGS):
        d = runs(a, r)
        for j, cl in enumerate(CLASSES):
            x = j + (i - 1.5) * w
            v = d[f"dice_{cl}"].to_numpy()
            ax.bar(x, v.mean(), w * 0.92, color=colour(a, r), hatch=hatch(a, r), edgecolor="white", linewidth=0.4,
                   label=label(a, r) if j == 0 else None)
            ok = v >= COLLAPSE_DICE
            ax.scatter(x + jitter(3, w * 0.25)[ok], v[ok], s=7, color=INK, zorder=5, linewidths=0)
            if (~ok).any():
                ax.scatter(x + jitter(3, w * 0.25)[~ok], np.full((~ok).sum(), 0.015), **LOST)
    ax.set_xticks(range(4), [CLS_LAB[c] for c in CLASSES])
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], marker="o", ls="", color=INK, ms=3), Line2D([], [], marker="x", ls="", color="#c0392b", ms=5)]
    l += ["individual seed", "class lost in a seed (Dice < 0.05)"]
    ax.legend(h, l, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "Fig3_per_class_dice")


# ---------------------------------------------------------------- Fig. 4 LoRA vs full fine-tuning
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4), sharey=True)
    for ax, a in zip(axes, ["mobilenet_unet", "mobile_sam", "edge_sam"]):
        d = runs(a, 8)
        ft = RF[(RF.arch == a) & (RF.loss == "full_finetune")].iloc[0]
        for j, cl in enumerate(CLASSES):
            v = d[f"dice_{cl}"].to_numpy()
            ax.bar(j - 0.2, v.mean(), 0.38, color=ARCH_COL[a], edgecolor="white", linewidth=0.4)
            ok = v >= COLLAPSE_DICE
            ax.scatter(j - 0.2 + jitter(3, 0.08)[ok], v[ok], s=7, color=INK, zorder=5, linewidths=0)
            if (~ok).any():
                ax.scatter(j - 0.2 + jitter(3, 0.08)[~ok], np.full((~ok).sum(), 0.015), **LOST)
            f = ft[f"dice_{cl}"]
            ax.bar(j + 0.2, f, 0.38, color="#bdbdb8", hatch="////", edgecolor="white", linewidth=0.4)
            if f < COLLAPSE_DICE:
                ax.scatter(j + 0.2, 0.015, **LOST)
        ax.set_title(NAME[a])
        ax.set_xticks(range(4), ["Disco. (1)", "Sphero. (2)", "Echino. (3)", "Stomato. (4)"], rotation=30, ha="right")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Dice")
    axes[0].set_ylim(0, 1)
    h = [Patch(color=MUTED), Patch(facecolor="#bdbdb8", hatch="////", edgecolor="white"),
         Line2D([], [], marker="o", ls="", color=INK, ms=3), Line2D([], [], marker="x", ls="", color="#c0392b", ms=5)]
    fig.legend(h, ["LoRA (r=8), mean of 3 seeds", "full fine-tuning (seed 42)", "individual seed",
                   "class lost (Dice < 0.05)"], ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "Fig4_lora_vs_full_finetuning")


# ---------------------------------------------------------------- Fig. 5 loss ablation
def fig5():
    cfgs = [("edge_sam", 32), ("mobile_sam", 8), ("mobilenet_unet", 8)]
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.0), sharex=True)
    for c, (a, r) in enumerate(cfgs):
        for row, (m, yl) in enumerate([("mean_dice", "Mean Dice"), ("ipe_global_pct", "Global IPE (%)")]):
            ax = axes[row, c]
            for k, loss in enumerate(LOSSES):
                d = runs(a, r, loss)
                v = d[m].to_numpy()
                lost = d.n_lost.to_numpy() > 0
                ax.bar(k, v.mean(), 0.62, color=ARCH_COL[a], alpha=1.0 if row == 0 else 0.85, edgecolor="white")
                xs = k + jitter(3, 0.14)
                ax.scatter(xs[~lost], v[~lost], s=8, color=INK, zorder=5, linewidths=0)
                ax.scatter(xs[lost], v[lost], **LOST)
            ax.set_ylabel(yl if c == 0 else "")
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(label(a, r))
                ax.set_ylim(0, 0.9)
            else:
                ax.set_ylim(0, None)
                ax.set_xticks(range(4), [LOSS_TICK[l] for l in LOSSES])
    axes[1, 0].set_ylabel("Global IPE (%)")
    axes[0, 0].set_ylabel("Mean Dice")
    h = [Patch(color=MUTED), Line2D([], [], marker="o", ls="", color=INK, ms=3),
         Line2D([], [], marker="x", ls="", color="#c0392b", ms=5)]
    fig.legend(h, ["mean of 3 seeds", "individual seed", "seed in which a class was lost (Dice < 0.05)"], ncol=3,
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "Fig5_loss_ablation")


# ---------------------------------------------------------------- Fig. 7 composition
def share(df):
    t = df.groupby(["storage_day", "class_name"]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
    return 100 * t.div(t.sum(axis=1), axis=0)


def fig7():
    ann = share(cells_gt())
    pred = sum(share(cells_pred("edge_sam", 32, "full", s)) for s in SEEDS) / 3
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    for ax, t, ttl in zip(axes, [ann, pred], ["(a) Annotation", "(b) EdgeSAM (r=32), mean of 3 seeds"]):
        x = np.arange(len(t))
        bottom = np.zeros(len(t))
        for cl in CLASSES:
            ax.bar(x, t[cl].to_numpy(), 0.78, bottom=bottom, color=CLS_COL[cl], edgecolor="white", linewidth=0.5,
                   label=CLS_LAB[cl])
            bottom += t[cl].to_numpy()
        ax.set_xticks(x, [str(d) for d in t.index])
        ax.set_xlabel("Storage day")
        ax.set_title(ttl, loc="left")
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of cells (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "Fig7_composition")


# ---------------------------------------------------------------- Fig. 8 dry mass and area
def per_image(df, col):
    return df.groupby(["stem", "storage_day"])[col].median().reset_index()


def day_index(df, col):
    d = per_image(df, col).groupby("storage_day")[col].median()
    return d / d.loc[0]


def fig8():
    gt = cells_gt()
    gimg = per_image(gt, "dry_mass_pg").rename(columns={"dry_mass_pg": "ann"})
    fig = plt.figure(figsize=(7.0, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    cmap = plt.get_cmap("Blues")
    stats_ = {"mdare": [], "r": []}
    for s in SEEDS:
        p = per_image(cells_pred("edge_sam", 32, "full", s), "dry_mass_pg").merge(gimg, on=["stem", "storage_day"])
        ax.scatter(p.ann, p.dry_mass_pg, c=p.storage_day, cmap=cmap, vmin=-12, vmax=47, s=11, marker=SEED_MARK[s],
                   edgecolors=INK, linewidths=0.25, zorder=4)
        stats_["mdare"].append(100 * np.median(np.abs(p.dry_mass_pg - p.ann) / p.ann))
        stats_["r"].append(np.corrcoef(p.ann, p.dry_mass_pg)[0, 1])
    lim = (22, 40)
    ax.plot(lim, lim, ls="--", color=MUTED, lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Annotation, per-image median (pg)")
    ax.set_ylabel("EdgeSAM (r=32), per-image median (pg)")
    ax.set_title("(a) Dry mass per image", loc="left", pad=14)
    m, r = np.array(stats_["mdare"]), np.array(stats_["r"])
    ax.text(0.0, 1.01, f"MdARE {m.mean():.1f} ± {m.std(ddof=1):.1f}%,  r = {r.mean():.2f} ± {r.std(ddof=1):.2f}",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=6.5, color=MUTED)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-12, 47))
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 15, 30, 47])
    cb.set_label("Storage day", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.ax.set_ylim(0, 47)
    ax.legend([Line2D([], [], marker=SEED_MARK[s], ls="", mfc="white", mec=INK, ms=4) for s in SEEDS],
              [f"seed {s}" for s in SEEDS], loc="lower right", frameon=False, fontsize=6.5, handletextpad=0.2)
    ngt = gt.groupby("storage_day").size()
    for k, (col, ttl) in enumerate([("area_um2", "(b) Projected area"), ("dry_mass_pg", "(c) Dry mass")]):
        ax = fig.add_subplot(gs[k + 1])
        a_idx = day_index(gt, col)
        ax.plot(a_idx.index, a_idx.values, ls="--", color=INK, lw=1.1, label="Annotation", zorder=5)
        for arch, rank in [("edge_sam", 32), ("mobile_sam", 8)]:
            per = pd.concat([day_index(cells_pred(arch, rank, "full", s), col) for s in SEEDS], axis=1)
            ratio = pd.concat([cells_pred(arch, rank, "full", s).groupby("storage_day").size() / ngt for s in SEEDS],
                              axis=1).mean(axis=1)
            c = ARCH_COL[arch]
            ax.fill_between(per.index, per.min(axis=1), per.max(axis=1), color=c, alpha=0.18, lw=0)
            ax.plot(per.index, per.mean(axis=1), color=c, lw=1.4, label=label(arch, rank))
            low = ratio < 0.6
            ax.scatter(per.index[~low], per.mean(axis=1)[~low], color=c, s=12, zorder=6)
            ax.scatter(per.index[low], per.mean(axis=1)[low], facecolor="white", edgecolor=c, s=12, lw=1, zorder=6)
        ax.axhline(1, color=MUTED, lw=0.5)
        ax.set_xlabel("Storage day")
        ax.set_ylabel("Median / day-0 median" if k == 0 else "")
        ax.set_title(ttl, loc="left")
        ax.set_ylim(0.4, 1.25)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=4))
    l.append("kept components < 0.6 × annotated cells")
    fig.legend(h, l, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "Fig8_dry_mass_and_area")


# ---------------------------------------------------------------- Fig. S-1 rank sweep, every seed
def figS1():
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.9), sharex=True)
    for c, a in enumerate(["edge_sam", "mobile_sam", "mobilenet_unet"]):
        for row, (m, yl) in enumerate([("mean_dice", "Mean Dice"), ("ipe_global_pct", "Global IPE (%)")]):
            ax = axes[row, c]
            means = []
            for k, r in enumerate(RANKS):
                d = runs(a, r)
                v = d[m].to_numpy()
                lost = d.n_lost.to_numpy() > 0
                means.append(v.mean())
                xs = k + jitter(3, 0.12)
                ax.scatter(xs[~lost], v[~lost], s=10, color=ARCH_COL[a], edgecolor=INK, linewidths=0.3, zorder=5)
                ax.scatter(xs[lost], v[lost], **LOST)
            ax.plot(range(5), means, color=ARCH_COL[a], lw=1.2, zorder=4)
            ax.set_xticks(range(5), [str(r) for r in RANKS])
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(NAME[a])
                ax.set_ylim(0.3, 0.9)
            else:
                ax.set_xlabel("LoRA rank r")
                ax.set_ylim(0, None)
            if c == 0:
                ax.set_ylabel(yl)
    h = [Line2D([], [], color=MUTED, lw=1.2), Line2D([], [], marker="o", ls="", mfc=MUTED, mec=INK, ms=4),
         Line2D([], [], marker="x", ls="", color="#c0392b", ms=5)]
    fig.legend(h, ["mean of 3 seeds", "individual seed (42, 43, 44)", "seed in which a class was lost (Dice < 0.05)"],
               ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "FigS1_rank_sweep_seeds")


# ---------------------------------------------------------------- Fig. S-2 validation curves
def figS2():
    cfgs = [("edge_sam", 8), ("edge_sam", 32), ("mobile_sam", 8), ("mobile_sam", 32)]
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.0), sharey=True)
    for ax, (a, r) in zip(axes, cfgs):
        for s, ls, mk in zip(SEEDS, ["-", "--", ":"], ["", "", "o"]):
            h = history(a, r, "full", s)
            lost = runs(a, r).set_index("seed").loc[s, "n_lost"] > 0
            # seed 44 gets markers and is drawn on top, so it stays visible where it overlaps seed 42
            ax.plot(h.epoch, h.val_mean_dice, ls=ls, lw=1.1, color=ARCH_COL[a], marker=mk or None, markevery=4, ms=2.5,
                    zorder=3 if mk else 2, label=f"seed {s}" + (" (class lost)" if lost else ""))
        ax.set_title(label(a, r))
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 0.9)
        ax.legend(frameon=False, fontsize=6, loc="lower right", handlelength=1.6)
    axes[0].set_ylabel("Validation mean Dice")
    fig.tight_layout()
    save(fig, "FigS2_validation_curves")


# ---------------------------------------------------------------- Fig. S-3 accuracy vs throughput (rank 8)
def figS3():
    rows = {(r["architecture"], r["backend"]): r for r in compiled("benchmark")["rows"]}
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    notes = {"mobilenet_unet": "classes 2 and 4 lost in 3/3 seeds", "mobile_sam": "no class lost",
             "edge_sam": "class 2 lost in 1/3 seeds"}
    for a in ["mobilenet_unet", "mobile_sam", "edge_sam"]:
        d = runs(a, 8).mean_dice
        fps = [rows[(a, b)]["fps"]["mean"] for b in ("pytorch", "onnx")]
        for f, mk in zip(fps, ["o", "s"]):
            ax.errorbar(f, d.mean(), yerr=d.std(ddof=1), fmt=mk, color=colour(a, 8), mec=INK, mew=0.4, ms=6,
                        capsize=2, elinewidth=0.8, zorder=5)
        ax.annotate("", xy=(fps[1], d.mean()), xytext=(fps[0], d.mean()),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
        if a == "mobilenet_unet":
            ax.text(max(fps), d.mean() - 0.04, f"{label(a, 8)}\n{notes[a]}", va="top", ha="center", fontsize=6.5)
        else:
            ax.text(max(fps) + 12, d.mean(), f"{label(a, 8)}\n{notes[a]}", va="center", fontsize=6.5)
    ax.set_xlabel("FP32 forward-pass throughput (frames/s)")
    ax.set_ylabel("Mean Dice (3 seeds)")
    ax.set_xlim(0, 560)
    ax.set_ylim(0.25, 0.95)
    ax.legend([Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=5),
               Line2D([], [], marker="s", ls="", mfc="white", mec=INK, ms=5)],
              ["PyTorch", "ONNX Runtime"], frameon=False, loc="upper right")
    save(fig, "FigS3_accuracy_vs_throughput")


if __name__ == "__main__":
    for f in (fig3, fig4, fig5, fig7, fig8, figS1, figS2, figS3):
        f()
