#!/usr/bin/env python
"""Manuscript figures generated from the actual experiment outputs (no training/evaluation code here).

    python manuscript/figures.py [--config-overlay smoke.yaml] [--outdir DIR] [--only fig3 fig7 ...]

Inputs: ``<manuscript_outputs>/results.json`` (run ``manuscript/results.py`` first), prediction files of
the runs, ``analysis/drymass/*.csv`` and the processed validation data. Every figure is written as PNG
(300 dpi) and PDF. Figures 1 and 2 of the manuscript are schematics and are not produced here.

Multi-seed figures (3, 4, 5, S1 to S3, S6) show the mean over seeds 42, 43 and 44 as the bar or line and every seed as a dot;
runs in which a class was lost (Dice < c.thr) are drawn as red crosses. The rank of the single-rank figures is
``results.json: meta.selected_lora_rank`` (set with ``results.py --rank``); S1 and S6 show all ranks. Qualitative, dry-mass and
composition figures are from the single seed-42 run named in ``meta``. Benchmark figures use FP32 only (D-H).
Class names follow the manuscript (mask value 2 = spherocyte, 3 = echinocyte, D-25).

Selection rules for qualitative panels are deterministic and stated in the function docstrings.
A failing figure makes the script exit with status 1 after all figures were attempted.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from skimage.measure import find_contours  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qpi.analysis.common import load_predictions, load_validation  # noqa: E402
from qpi.config import load_config  # noqa: E402
from qpi.utils.classnames import CLASSES  # noqa: E402   # mask-value order 1..4: discocyte, spherocyte, echinocyte, stomatocyte
from qpi.utils.io import read_json  # noqa: E402
CLASS_LABEL = {"discocyte": "Discocyte", "echinocyte": "Echinocyte", "spherocyte": "Spherocyte",
               "stomatocyte": "Stomatocyte"}
CLASS_COLOR = {"discocyte": "#0072B2", "echinocyte": "#D55E00", "spherocyte": "#009E73", "stomatocyte": "#CC79A7"}
ARCH = ["mobilenet_unet", "mobile_sam", "edge_sam"]
ARCH_LABEL = {"edge_sam": "EdgeSAM", "mobile_sam": "MobileSAM", "mobilenet_unet": "MobileNet-UNet"}
ARCH_COLOR = {"edge_sam": "#0072B2", "mobile_sam": "#E69F00", "mobilenet_unet": "#009E73"}
LABEL_CMAP = ListedColormap(["#00000000"] + [CLASS_COLOR[k] for k in CLASSES])      # index = mask value
LOSS_ORDER = ["dice_only", "pmc", "pmc_bga", "full"]      # display order of loss names; the ones present come from results.json
LOSS_SHORT = {"dice_only": "Dice", "pmc": "+PMC", "pmc_bga": "+PMC+BGA", "full": "+IPP"}
LOSS_COLOR = {"dice_only": "#999999", "pmc": "#56B4E9", "pmc_bga": "#E69F00", "full": "#000000"}
LOST = "#C00000"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 300,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


class Ctx:
    def __init__(self, cfg: Dict, outdir: Path):
        self.cfg, self.outdir = cfg, outdir
        self.root = Path(cfg["paths_abs"]["output_root"])
        self.res = read_json(Path(cfg["paths_abs"]["manuscript_outputs"]) / "results.json")
        meta = self.res["meta"]
        self.sel_rank = int(meta["selected_lora_rank"])
        self.sel_arch = meta["selected_architecture"]
        self.drymass_run = meta.get("drymass_run")
        self._ms = None
        self.meta_ms = self.res["tables"]["ms_meta"]              # ranks, seeds, losses, lost-class threshold: all from results.json
        self.ranks = list(self.meta_ms["ranks"])
        self.seeds = list(self.meta_ms["seeds"])
        self.losses = [l for l in LOSS_ORDER if l in self.meta_ms["losses"]]
        self.thr = float(self.meta_ms["collapse_dice_threshold"])
        self.ckpt = cfg["evaluation"]["checkpoint"]
        self._val = None

    @property
    def val(self):
        if self._val is None:
            self._val = load_validation(self.cfg)
        return self._val

    def table(self, name):
        return self.res["tables"][name]

    @property
    def ms(self) -> pd.DataFrame:
        """One row per multi-seed run (arch, loss, rank, seed, metrics, collapsed)."""
        if self._ms is None:
            if "ms_runs" not in self.res["tables"]:
                raise RuntimeError("results.json has no multi-seed tables: run manuscript/results.py (needs outputs/multiseed/summary)")
            self._ms = pd.DataFrame(self.res["tables"]["ms_runs"])
        return self._ms

    def ms_sel(self, arch=None, loss=None, rank=None) -> pd.DataFrame:
        d = self.ms
        if arch is not None:
            d = d[d.arch == arch]
        if loss is not None:
            d = d[d.loss == loss]
        if rank is not None:
            d = d[d["rank"] == rank]
        return d

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def preds(self, run_id: str):
        return load_predictions(self.run_dir(run_id), self.ckpt)

    def save(self, fig, name: str):
        self.outdir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(self.outdir / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)
        print(f"[figures] wrote {self.outdir / name}.png")


def nz(v):
    """JSON null (undefined metric, e.g. class absent) -> NaN for plotting."""
    return np.nan if v is None else float(v)


def seed_dots(ax, x, values, collapsed, color="black", width=0.07, size=9, zorder=5):
    """Every seed as a dot beside the mean; runs that lost a class are red crosses."""
    values = np.asarray(values, float); collapsed = np.asarray(collapsed, bool)
    xs = x + np.linspace(-width, width, len(values)) if len(values) > 1 else np.array([x])
    ok = ~collapsed
    ax.scatter(xs[ok], values[ok], s=size, color=color, edgecolor="white", linewidth=0.4, zorder=zorder)
    if collapsed.any():
        ax.scatter(xs[collapsed], values[collapsed], s=size * 1.6, marker="x", color=LOST, linewidth=0.9, zorder=zorder + 1)


def _plain_log_axis(ax):
    """Log axis with plain numbers on the ticks (ticks are chosen by matplotlib from the data)."""
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 3.0, 5.0)))


def _lost_legend_handle(c: "Ctx"):
    return Line2D([0], [0], marker="x", ls="", color=LOST, label=f"a class was lost (Dice < {c.thr:g})")


def _arch_key(label: str) -> str:
    return {v: k for k, v in ARCH_LABEL.items()}[label]


def _best_window(mask: np.ndarray, valid: np.ndarray, size: int = 256, stride: int = 32):
    """Top-left corner of the size x size window with most mask pixels (ties -> first in raster order)."""
    best, pos = -1, (0, 0)
    H, W = mask.shape
    for y in range(0, H - size + 1, stride):
        for x in range(0, W - size + 1, stride):
            n = int(mask[y:y + size, x:x + size].sum())
            if n > best:
                best, pos = n, (y, x)
    return pos


def _gray(ax, img, vmin=None, vmax=None):
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def _overlay(ax, phase, labels, valid, vmin, vmax):
    _gray(ax, phase, vmin, vmax)
    ax.imshow(np.ma.masked_where(labels == 0, labels), cmap=LABEL_CMAP, vmin=0, vmax=4, alpha=0.55, interpolation="nearest")
    ax.imshow(np.ma.masked_where(valid, np.ones_like(phase)), cmap=ListedColormap(["#888888"]), alpha=0.45,
              interpolation="nearest")


def _class_legend(ax_or_fig, **kw):
    handles = [Line2D([0], [0], marker="s", ls="", color=CLASS_COLOR[c], label=CLASS_LABEL[c]) for c in CLASSES]
    return ax_or_fig.legend(handles=handles, frameon=False, **kw)


# --------------------------------------------------------------------------------------------------
def _rank_note(c: Ctx) -> str:
    return f"LoRA r={c.sel_rank}"


def fig3(c: Ctx):
    """Per-class Dice at the selected rank, three architectures: bar = mean of seeds 42, 43, 44, dots = seeds."""
    d = c.ms_sel(loss="full", rank=c.sel_rank)
    fig, ax = plt.subplots(figsize=(3.9, 2.8))
    w = 0.26
    for i, a in enumerate(ARCH):
        g = d[d.arch == a]
        for k, cls in enumerate(CLASSES):
            x = k + (i - 1) * w
            ax.bar(x, g[f"dice_{cls}"].mean(), w, color=ARCH_COLOR[a], label=ARCH_LABEL[a] if k == 0 else None)
            seed_dots(ax, x, g[f"dice_{cls}"], g[f"dice_{cls}"] < c.thr, width=w * 0.3, size=6)
    ax.set_xticks(range(4)); ax.set_xticklabels([CLASS_LABEL[k] for k in CLASSES], rotation=20)
    ax.set_ylabel("Dice"); ax.set_ylim(bottom=0)
    ax.set_title(f"{_rank_note(c)}, mean of {len(c.seeds)} seeds (dots = seeds)", fontsize=7)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [_lost_legend_handle(c)], l + ["class lost in a seed"], frameon=False, ncol=1, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), fontsize=6.5)
    c.save(fig, "fig3_per_class_dice")


def fig4(c: Ctx):
    """Per-class Dice, LoRA (mean of 3 seeds, dots = seeds) against full fine-tuning (one seed-42 run), per architecture."""
    ft = c.table("table2_lora_vs_full_finetune")
    d = c.ms_sel(loss="full", rank=c.sel_rank)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
    for ax, a in zip(axes, ARCH):
        g = d[d.arch == a]
        r = next(x for x in ft if _arch_key(x["architecture"]) == a and x["method"] == "Full fine-tuning")
        for k, cls in enumerate(CLASSES):
            ax.bar(k - 0.19, g[f"dice_{cls}"].mean(), 0.38, color=ARCH_COLOR[a], edgecolor="white")
            seed_dots(ax, k - 0.19, g[f"dice_{cls}"], g[f"dice_{cls}"] < c.thr, width=0.1, size=6)
            ax.bar(k + 0.19, nz(r[f"dice_{cls}"]), 0.38, color="#999999", hatch="//", edgecolor="white")
        ax.set_title(ARCH_LABEL[a]); ax.set_xticks(range(4)); ax.set_xticklabels([k[:5].capitalize() + "." for k in CLASSES])
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Dice")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#555555", label=f"{_rank_note(c)}: mean of {len(c.seeds)} seeds, dots = seeds"),
                        Patch(facecolor="#999999", hatch="//", edgecolor="white", label=f"Full fine-tuning: one run (seed {c.res['meta']['single_run_seed']})"),
                        _lost_legend_handle(c)],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=6.5)
    fig.subplots_adjust(bottom=0.27)
    c.save(fig, "fig4_lora_vs_full_per_class")


def fig5(c: Ctx):
    """Loss ablation at the selected rank: mean Dice (bars) and global IPE (line, right axis); dots = seeds."""
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
    fig.subplots_adjust(wspace=0.85, bottom=0.3)
    for ax, a in zip(axes, ARCH):
        ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
        for x, loss in enumerate(c.losses):
            g = c.ms_sel(arch=a, loss=loss, rank=c.sel_rank)
            ax.bar(x, g["mean_dice"].mean(), 0.6, color=ARCH_COLOR[a], alpha=0.85)
            seed_dots(ax, x, g["mean_dice"], g["collapsed"], width=0.15, size=7)
            ax2.scatter(np.full(len(g), x + 0.34), g["ipe_global_pct"], s=5, color="#444444", zorder=6)
        m = [c.ms_sel(arch=a, loss=l, rank=c.sel_rank)["ipe_global_pct"].mean() for l in c.losses]
        ax2.plot(np.arange(4) + 0.34, m, "k--", lw=0.9, marker="_", ms=6)
        ax.set_ylim(bottom=0); ax.set_xticks(range(4)); ax.set_xticklabels([LOSS_SHORT[k] for k in c.losses], rotation=40, ha="right")
        ax.set_title(ARCH_LABEL[a])
        if a == ARCH[0]:
            ax.set_ylabel("Mean Dice")
        ax2.set_ylabel("Global IPE (%)"); ax2.set_ylim(bottom=0)
    fig.suptitle(f"{_rank_note(c)}; bars and black dots: mean Dice of {len(c.seeds)} seeds and each seed; "
                 f"dashed line and grey dots: mean IPE and each seed (right axis)", fontsize=7, y=0.99)
    fig.legend(handles=[_lost_legend_handle(c)], loc="lower center", frameon=False, fontsize=6.5)
    c.save(fig, "fig5_loss_ablation")


def fig6(c: Ctx):
    """Qualitative results. For each class the 256 x 256 window with most annotated pixels of that class is shown
    (over all validation images; raster-order tie-break), together with the annotation and the three seed-42 predictions at the
    selected rank. Classes are in mask-value order (spherocyte = 2, echinocyte = 3)."""
    runs = {a: c.preds(f"{a}_lora_r{c.sel_rank}") for a in ARCH}
    val = c.val
    fig, axes = plt.subplots(4, 5, figsize=(7.2, 6.0))
    for r, k in enumerate(CLASSES):
        cls = r + 1
        best = max(val, key=lambda s: (int(((val[s]["gt"] == cls) & val[s]["valid"]).sum()), -sorted(val).index(s)))
        v = val[best]
        y, x = _best_window(v["gt"] == cls, v["valid"])
        sl = (slice(y, y + 256), slice(x, x + 256))
        lo, hi = np.percentile(v["phase"][sl], [1, 99])
        _gray(axes[r, 0], v["phase"][sl], lo, hi)
        panels = [("Annotation", v["gt"])] + [(ARCH_LABEL[a], runs[a][best]) for a in ARCH]
        for j, (t, lab) in enumerate(panels, start=1):
            _overlay(axes[r, j], v["phase"][sl], lab[sl], v["valid"][sl], lo, hi)
            if r == 0:
                axes[r, j].set_title(t)
        axes[r, 0].set_ylabel(f"{CLASS_LABEL[k]}\n{best}", fontsize=7)
        if r == 0:
            axes[r, 0].set_title("Phase")
    _class_legend(fig, loc="lower center", ncol=4)
    c.save(fig, "fig6_qualitative")


def _predicted_or_smoke_fallback(c: Ctx):
    """Predicted-cell tables. An under-trained smoke model may detect no cells; then (smoke mode only) the annotation-derived
    tables are used so that the plotting code is still exercised. Such figures are titled accordingly."""
    pred = c.table("table7_storage_per_day_predicted")
    if pred:
        return "pred", pred
    if c.cfg.get("smoke"):
        return "gt", c.table("table7_storage_per_day_annotated")
    raise RuntimeError("no predicted cells: cannot draw a figure from an empty prediction table")


def _compositions(c: Ctx):
    src, rows = _predicted_or_smoke_fallback(c)
    return src, pd.DataFrame(rows)


def fig7(c: Ctx):
    """Predicted morphology composition per storage day (seed-42 run named in meta.drymass_run)."""
    src, d = _compositions(c)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    if src == "gt":
        ax.set_title("SMOKE TEST: annotation-derived data", fontsize=7, color="red")
    x = np.arange(len(d)); bottom = np.zeros(len(d))
    for k in CLASSES:
        v = np.array([row["composition_pct"].get(k, 0.0) for row in d.to_dict("records")])
        ax.bar(x, v, 0.7, bottom=bottom, color=CLASS_COLOR[k], label=CLASS_LABEL[k]); bottom += v
    ax.set_xticks(x); ax.set_xticklabels(d["storage_day"].astype(int))
    ax.set_xlabel("Storage day"); ax.set_ylabel("Share of detected cells (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    c.save(fig, "fig7_composition_over_storage")


def _per_image(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    if "wrapped" in df.columns:               # phase-wrapped cells are left out of the dry-mass figure (D-21)
        df = df[~df["wrapped"].astype(bool)]
    if df.empty:
        raise RuntimeError(f"{csv} is empty")
    return df.groupby("stem").agg(storage_day=("storage_day", "first"), dry_mass_pg=("dry_mass_pg", "median"),
                                  area_um2=("area_um2", "median")).reset_index()


def fig8(c: Ctx):
    """(a) per-image median dry mass: prediction vs annotation; (b) day-0 normalised median area and dry mass."""
    gt = _per_image(c.root / "analysis" / "drymass" / "cells_gt.csv")
    src, _ = _predicted_or_smoke_fallback(c)
    pred = _per_image(c.root / "analysis" / "drymass" / "cells_pred.csv") if src == "pred" else gt.copy()
    m = pred.merge(gt, on="stem", suffixes=("_pred", "_gt"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.8), layout="constrained")
    sc = a.scatter(m["dry_mass_pg_gt"], m["dry_mass_pg_pred"], c=m["storage_day_gt"], cmap="viridis", s=14)
    lim = [0, float(max(m["dry_mass_pg_gt"].max(), m["dry_mass_pg_pred"].max())) * 1.05]
    a.plot(lim, lim, "k--", lw=0.8); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("Annotation, median dry mass (pg)"); a.set_ylabel(f"{c.drymass_run}, median dry mass (pg)")
    plt.colorbar(sc, ax=a, label="Storage day", fraction=0.046)
    by = pred.groupby("storage_day")[["area_um2", "dry_mass_pg"]].median().sort_index()
    b.plot(by.index, by["area_um2"] / by["area_um2"].iloc[0], "o-", label="Area", color="#E69F00")
    b.plot(by.index, by["dry_mass_pg"] / by["dry_mass_pg"].iloc[0], "s-", label="Dry mass", color="#0072B2")
    b.axhline(1, color="grey", lw=0.6); b.set_xlabel("Storage day"); b.set_ylabel("Median / day-0 median")
    b.legend(frameon=False)
    a.set_title("(a)" + ("  SMOKE TEST: annotation vs itself" if src == "gt" else ""), loc="left"); b.set_title("(b)", loc="left")
    c.save(fig, "fig8_dry_mass")


def figS1(c: Ctx):
    """LoRA rank sweep, full objective, all architectures: (a) mean Dice, (b) global IPE, (c) trainable parameters.
    Lines = mean of seeds 42, 43, 44; dots = seeds; red crosses = a class was lost in that run."""
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    x = np.arange(len(c.ranks))
    for a in ARCH:
        for ax, col in zip(axes[:2], ("mean_dice", "ipe_global_pct")):
            means = []
            for k, r in enumerate(c.ranks):
                g = c.ms_sel(arch=a, loss="full", rank=r)
                means.append(g[col].mean())
                seed_dots(ax, k, g[col], g["collapsed"], color=ARCH_COLOR[a], width=0.12, size=8)
            ax.plot(x, means, "-", color=ARCH_COLOR[a], lw=1.1, label=ARCH_LABEL[a])
        axes[2].plot(x, [c.ms_sel(arch=a, loss="full", rank=r)["trainable_params"].iloc[0] / 1e6 for r in c.ranks], "o-",
                     color=ARCH_COLOR[a], ms=3, lw=1.1)
    axes[0].set_ylabel("Mean Dice"); axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("Global IPE (%)"); axes[1].set_yscale("log")
    _plain_log_axis(axes[1])
    axes[2].set_ylabel("Trainable parameters (M)"); axes[2].set_yscale("log"); _plain_log_axis(axes[2])
    for ax, t in zip(axes, ("(a)", "(b)", "(c)")):
        ax.set_xticks(x); ax.set_xticklabels(c.ranks); ax.set_xlabel("LoRA rank r"); ax.set_title(t, loc="left")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.legend(handles=[_lost_legend_handle(c)], loc="lower center", frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    c.save(fig, "figS1_rank_vs_dice")


def figS6(c: Ctx):
    """Loss ablation at every rank (the rank-8 ablation is one column of this figure): rows = mean Dice and global IPE,
    columns = architectures, one line per loss; dots = seeds; red crosses = a class was lost."""
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), sharex=True)
    x = np.arange(len(c.ranks))
    for j, a in enumerate(ARCH):
        for loss in c.losses:
            for i, col in enumerate(("mean_dice", "ipe_global_pct")):
                means = []
                for k, r in enumerate(c.ranks):
                    g = c.ms_sel(arch=a, loss=loss, rank=r)
                    means.append(g[col].mean())
                    seed_dots(axes[i, j], k + (c.losses.index(loss) - 1.5) * 0.06, g[col], g["collapsed"],
                              color=LOSS_COLOR[loss], width=0.0, size=6)
                axes[i, j].plot(x + (c.losses.index(loss) - 1.5) * 0.06, means, "-", color=LOSS_COLOR[loss], lw=1.0,
                                label=LOSS_SHORT[loss] if (i, j) == (0, 0) else None)
        axes[0, j].set_title(ARCH_LABEL[a]); axes[0, j].set_ylim(bottom=0)
        axes[1, j].set_yscale("log"); _plain_log_axis(axes[1, j])
        axes[1, j].set_xticks(x); axes[1, j].set_xticklabels(c.ranks); axes[1, j].set_xlabel("LoRA rank r")
    axes[0, 0].set_ylabel("Mean Dice"); axes[1, 0].set_ylabel("Global IPE (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h + [_lost_legend_handle(c)], l + ["class lost in that run"], loc="lower center", ncol=5, frameon=False,
               fontsize=6.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    c.save(fig, "figS6_ablation_by_rank")


def _bench(c: Ctx, arch: str, backend: str, precision: str):
    for r in c.table("table6_benchmark"):
        if _arch_key(r["architecture"]) == arch and r["backend"] == backend and r["precision"] == precision:
            return r
    return None


def _fp32_rows(c: Ctx, arch: str):
    return [r for r in c.table("table6_benchmark") if _arch_key(r["architecture"]) == arch and r["precision"] == "fp32"]


def figS2(c: Ctx):
    """Radar of mean Dice, AJI, Boundary F1 (3-seed means at the selected rank), best FP32 throughput and memory efficiency
    (1 / PyTorch FP32 peak VRAM); each axis divided by the best architecture."""
    axes_names = ["Mean Dice", "AJI", "Boundary F1", "Throughput\n(FP32)", "Memory efficiency"]
    vals = {}
    for a in ARCH:
        g = c.ms_sel(arch=a, loss="full", rank=c.sel_rank)
        rows = _fp32_rows(c, a)
        pt = [r["peak_vram_mb"] for r in rows if r["backend"] == "pytorch" and r.get("peak_vram_mb")]
        vals[a] = [g["mean_dice"].mean(), g["aji"].mean(), g["bf1"].mean(), max(r["fps"] for r in rows),
                   1.0 / min(pt) if pt else np.nan]
    best = np.nanmax(np.array(list(vals.values())), axis=0)
    ang = np.linspace(0, 2 * np.pi, len(axes_names), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(3.8, 3.6)); ax = fig.add_subplot(111, projection="polar")
    for a in ARCH:
        v = (np.array(vals[a]) / best).tolist(); v += v[:1]
        ax.plot(ang, v, color=ARCH_COLOR[a], label=ARCH_LABEL[a]); ax.fill(ang, v, color=ARCH_COLOR[a], alpha=0.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(axes_names); ax.set_ylim(0, 1)      # axes are divided by the best architecture, so 1 is the maximum by construction
    ax.set_title(f"{_rank_note(c)}", fontsize=7, pad=12)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    c.save(fig, "figS2_aggregate_radar")


def figS3(c: Ctx):
    """Accuracy-throughput at FP32 (D-H: no FP16 results are reported): 3-seed mean Dice (bar = SD) against FPS; arrows join
    the PyTorch and ONNX Runtime points of an architecture (an architecture without a working ONNX export has only the PyTorch point)."""
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    for a in ARCH:
        g = c.ms_sel(arch=a, loss="full", rank=c.sel_rank)["mean_dice"]
        p, o = _bench(c, a, "pytorch", "fp32"), _bench(c, a, "onnx", "fp32")
        pts = [(r["fps"], g.mean()) for r in (p, o) if r]
        for (x, y), mk in zip(pts, ("o", "s")):
            ax.errorbar(x, y, yerr=g.std(ddof=1), fmt=mk, color=ARCH_COLOR[a], ms=4, capsize=2, lw=0.8)
        if len(pts) == 2:
            ax.annotate("", xy=pts[1], xytext=pts[0], arrowprops=dict(arrowstyle="->", color=ARCH_COLOR[a]))
        if pts:
            ax.annotate(ARCH_LABEL[a], pts[-1], textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Throughput, FP32 (FPS)"); ax.set_ylabel("Mean Dice"); ax.set_ylim(bottom=0)
    ax.set_title(f"{_rank_note(c)}, mean of {len(c.seeds)} seeds ± SD", fontsize=7)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color="grey", label="PyTorch"),
                       Line2D([0], [0], marker="s", ls="", color="grey", label="ONNX Runtime")], frameon=False)
    c.save(fig, "figS3_accuracy_throughput")


def figS4(c: Ctx):
    """Phase, phase-gradient magnitude and selected-architecture contours on the gradient map. Image: validation
    image with most annotated foreground pixels; window: 256 x 256 with most foreground."""
    val = c.val
    stem = max(val, key=lambda s: (int(((val[s]["gt"] > 0) & val[s]["valid"]).sum()), -sorted(val).index(s)))
    v = val[stem]; pred = c.preds(f"{c.sel_arch}_lora_r{c.sel_rank}")[stem]
    y, x = _best_window(v["gt"] > 0, v["valid"]); sl = (slice(y, y + 256), slice(x, x + 256))
    gy, gx = np.gradient(v["phase"]); g = np.hypot(gx, gy)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5))
    lo, hi = np.percentile(v["phase"][sl], [1, 99])
    ax[0].imshow(v["phase"][sl], cmap="gray", vmin=lo, vmax=hi); ax[0].set_title("Phase (rad)")
    ax[1].imshow(g[sl], cmap="magma"); ax[1].set_title("|grad phase|")
    ax[2].imshow(g[sl], cmap="magma")
    for cont in find_contours((pred[sl] > 0).astype(float), 0.5):
        ax[2].plot(cont[:, 1], cont[:, 0], color="cyan", lw=0.7)
    ax[2].set_title(f"{ARCH_LABEL[c.sel_arch]} contours ({stem})")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    c.save(fig, "figS4_gradient_alignment")


def figS5(c: Ctx):
    """Representative phase images over storage: first image (alphabetical stem) of five evenly spaced storage days."""
    val = c.val
    days = sorted({v["day"] for v in val.values()})
    pick = [days[int(round(i))] for i in np.linspace(0, len(days) - 1, min(5, len(days)))]
    fig, axes = plt.subplots(1, len(pick), figsize=(1.9 * len(pick), 2.1), squeeze=False)
    for ax, d in zip(axes[0], pick):
        stem = sorted(s for s in val if val[s]["day"] == d)[0]
        ph = val[stem]["phase"]; lo, hi = np.percentile(ph[val[stem]["valid"]], [1, 99.5])
        _gray(ax, ph, lo, hi); ax.set_title(f"Day {d}")
    c.save(fig, "figS5_storage_examples")


FIGURES: Dict[str, Callable[[Ctx], None]] = {
    "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7, "fig8": fig8,
    "figS1": figS1, "figS2": figS2, "figS3": figS3, "figS4": figS4, "figS5": figS5, "figS6": figS6,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-overlay", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config_overlay)
    outdir = Path(a.outdir) if a.outdir else Path(cfg["paths_abs"]["manuscript_outputs"]) / "figures"
    res_path = Path(cfg["paths_abs"]["manuscript_outputs"]) / "results.json"
    if not res_path.exists():
        print(f"figures.py: {res_path} not found; run manuscript/results.py first", file=sys.stderr)
        return 1
    ctx = Ctx(cfg, outdir)
    failed: List[str] = []
    for name, fn in FIGURES.items():
        if a.only and name not in a.only:
            continue
        try:
            fn(ctx)
        except Exception:
            failed.append(name)
            print(f"[figures] {name} FAILED", file=sys.stderr)
            traceback.print_exc()
    if failed:
        print(f"figures.py: failed figures: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
