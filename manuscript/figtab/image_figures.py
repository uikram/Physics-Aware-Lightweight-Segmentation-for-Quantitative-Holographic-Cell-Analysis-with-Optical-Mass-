"""Figures drawn from the phase images themselves. They need data/prepared/ and the runs' predictions_last.npz,
which exist only on the training server:

    python manuscript/make_all.py --images

Fig. 6   qualitative segmentation: one test image per morphology class (a different image for every row), a window
         centred on cells of that class; phase, reference and each architecture's prediction at its representative
         rank. (The previous build picked the window without regard to the row's class, so two rows showed the same
         day-47 window; here each row must contain the most separately visible cells of its own class.)
Fig. S2  every reference cell flagged by the superposition screen (W-5), with its phase and reference outline.

The cell separation of the predictions uses the project's own watershed (qpicell.evaluation.instances.a2) when it
can be imported, so the outlines are exactly the cells that were scored; without it the outlines are omitted and
the caption says so.
"""
from __future__ import annotations

import csv
import sys
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle

from . import style as S
from .core import PROPOSED, Inputs, MissingInput, Runs, cls_name, cond_label, cond_label_m, representative, run_dir

W = "additional_analysis/results"


class Images:
    def __init__(self, inp: Inputs, R: Runs):
        self.inp, self.R = inp, R
        cfg = inp.config()
        if not cfg:
            raise MissingInput("configs/config.yaml (and PyYAML)")
        self.cfg = cfg
        self.base = f"{cfg['paths']['prepared']}/val"
        if not inp.path(f"{self.base}/phase").is_dir():
            raise MissingInput(f"{self.base}/phase (run on the server, where data/prepared exists)")
        try:
            import tifffile  # noqa: F401
        except ImportError as e:
            raise MissingInput("python package tifffile") from e
        self.px = float(cfg["physics"]["pixel_size_um"])
        self.ignore = int(cfg["data"]["ignore_index"])
        self.names = cfg["data"]["class_names"]
        self.lo, self.hi = cfg["cells"]["area_range_um2"]
        self.h = int(cfg["cells"]["watershed_h"])
        self.stems = sorted(p.stem for p in inp.path(f"{self.base}/phase").glob("*.tif"))
        self.day = {}
        man = f"{cfg['paths']['prepared']}/manifest.csv"
        if inp.exists(man):
            for r in inp.csv(man):
                if r.get("split") == "val":
                    self.day[r["stem"]] = int(r["storage_day"])
        self._c: Dict[str, Dict] = {}
        self._p: Dict[str, object] = {}
        try:
            sys.path.insert(0, str(inp.root))
            from qpicell.evaluation.instances import a2  # type: ignore
            self.a2 = a2
        except Exception:  # noqa: BLE001
            self.a2 = None

    def image(self, stem: str) -> Dict:
        import tifffile
        if stem not in self._c:
            r = lambda sub: tifffile.imread(self.inp.need(f"{self.base}/{sub}/{stem}.tif"))  # noqa: E731
            mask = r("mask")
            self._c[stem] = dict(stem=stem, phase=r("phase").astype(np.float64), mask=mask,
                                 cells=r("cells").astype(np.int64), valid=mask != self.ignore, day=self.day.get(stem))
        return self._c[stem]

    def prediction(self, arch: str, cond: str, arm: str, seed: int, stem: str) -> Optional[np.ndarray]:
        rel = f"{run_dir(arch, cond, arm, seed)}/predictions_{self.R.checkpoint}.npz"
        if not self.inp.exists(rel):
            return None
        if rel not in self._p:
            self._p[rel] = np.load(self.inp.need(rel))
        z = self._p[rel]
        return z[stem] if stem in z.files else None

    def instances(self, d: Dict):
        """Reference instances: labels, area (µm²), class value, touching flag."""
        lab = np.where(d["valid"], d["cells"], 0)
        n = int(lab.max())
        area = np.bincount(lab.ravel(), minlength=n + 1) * self.px ** 2
        cls = np.zeros(n + 1, np.int32)
        cls[lab.ravel()] = d["mask"].ravel()
        t = np.zeros(n + 1, bool)
        for a, b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
            A, B = lab[a], lab[b]; m = (A > 0) & (B > 0) & (A != B); t[A[m]] = True; t[B[m]] = True
        return lab, area, cls, t


def _boxes(lab):
    from scipy import ndimage as ndi
    return ndi.find_objects(lab)


def _window_for_class(I: Images, d: Dict, v: int, size: int):
    """The window that holds the most whole, separately visible reference cells of class v (ties: more cells of
    any class; then lying fully inside the annotated field)."""
    lab, area, cls, t = I.instances(d)
    sl = _boxes(lab)
    good = [(s[0].start, s[0].stop, s[1].start, s[1].stop, cls[i + 1] == v)
            for i, s in enumerate(sl) if s is not None and I.lo <= area[i + 1] <= I.hi and not t[i + 1]]
    n = d["phase"].shape[0]
    step = max(size // 16, 8)
    best, score = (0, 0), (-1, -1, -1)
    for y0 in range(0, n - size + 1, step):
        for x0 in range(0, n - size + 1, step):
            inside = [g for g in good if g[0] >= y0 and g[1] <= y0 + size and g[2] >= x0 and g[3] <= x0 + size]
            sc = (sum(g[4] for g in inside), len(inside), int(d["valid"][y0:y0 + size, x0:x0 + size].all()))
            if sc > score:
                score, best = sc, (y0, x0)
    return np.s_[best[0]:best[0] + size, best[1]:best[1] + size], score[0]


def _class_rgba(classmap, valid, names, alpha=0.55):
    rgba = np.zeros(classmap.shape + (4,))
    for v, name in enumerate(names):
        if v == 0 or name not in S.CLASS_COLOR:
            continue
        rgba[classmap == v] = (*to_rgb(S.CLASS_COLOR[name]), alpha)
    rgba[~valid] = (*to_rgb("#bdbcb8"), 0.85)
    return rgba


def _scale(ax, n_px, px_um, um=10.0, color="white"):
    frac = um / (n_px * px_um)
    ax.add_patch(Rectangle((0.06, 0.06), frac, 0.025, transform=ax.transAxes, fc=color, ec="none", zorder=5))
    import matplotlib.patheffects as pe
    ax.text(0.06 + frac / 2, 0.1, f"{um:g} µm", transform=ax.transAxes, color=color, ha="center", va="bottom",
            fontsize=7, zorder=5, path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])


def fig6_qualitative(inp: Inputs, R: Runs, arm: str = PROPOSED, size: int = 192):
    I = Images(inp, R)
    rep = representative(inp)
    seed = R.seeds[0]
    rows, used = [], set()
    for v, name in enumerate(I.names):
        if v == 0:
            continue
        best = None
        for s in I.stems:
            if s in used:
                continue
            d = I.image(s)
            win, k = _window_for_class(I, d, v, size)
            if best is None or k > best[2]:
                best = (s, win, k)
        if best and best[2] > 0:
            used.add(best[0]); rows.append((v, name) + best)
    if not rows:
        raise MissingInput("no test image with a separately visible reference cell")
    cols = ["Phase", "Reference"] + [f"{a}\n({cond_label_m(rep[a]['selected_condition'])})" for a in R.archs]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(S.DOUBLE, S.DOUBLE / len(cols) * len(rows) + 10 * S.MM),
                             squeeze=False)
    for i, (v, name, stem, win, k) in enumerate(rows):
        d = I.image(stem)
        ph, valid = d["phase"][win], d["valid"][win]
        lo, hi = np.percentile(d["phase"][d["valid"]], [0.5, 99.5]) if d["valid"].any() else (None, None)
        ref_lab = np.where(d["valid"], d["cells"], 0)[win]
        panels = [("phase", None, None), ("ref", np.where(d["valid"], d["mask"], 0)[win], ref_lab)]
        for a in R.archs:
            p = I.prediction(a, rep[a]["selected_condition"], arm, seed, stem)
            if p is None:
                panels.append(("missing", None, None)); continue
            lab = I.a2(p, I.h)[win] if I.a2 is not None else None
            panels.append(("pred", np.asarray(p)[win], lab))
        for j, (kind, cmap_, lab) in enumerate(panels):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if kind == "missing":
                ax.text(0.5, 0.5, "prediction\nnot found", ha="center", va="center", fontsize=7, transform=ax.transAxes)
                continue
            ax.imshow(ph, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            if cmap_ is not None:
                ax.imshow(_class_rgba(cmap_, valid, I.names), interpolation="nearest")
            if lab is not None:
                e = np.zeros(lab.shape, bool)
                e[:-1] |= (lab[:-1] != lab[1:]); e[:, :-1] |= (lab[:, :-1] != lab[:, 1:])
                e &= (lab > 0) | np.roll(lab > 0, 1, 0) | np.roll(lab > 0, 1, 1)
                ov = np.zeros(lab.shape + (4,)); ov[e] = (1, 1, 1, 0.9)
                ax.imshow(ov, interpolation="nearest")
            if j == 0:
                _scale(ax, size, I.px)
                ax.set_ylabel(cls_name(name), fontsize=8.5)  # class name only (no image name)
            if i == 0:
                # every column title centred on the same line, so one-line and two-line titles align
                ax.text(0.5, 1.0, cols[j], transform=ax.transAxes, ha="center", va="center", fontsize=8.5,
                        linespacing=1.3, color=S.INK)
    leg = [Patch(fc=S.CLASS_COLOR[n], label=cls_name(n)) for n in I.names[1:] if n in S.CLASS_COLOR]
    leg.append(Patch(fc="#bdbcb8", label="Outside the annotated field"))
    fig.legend(handles=leg, loc="lower center", ncol=len(leg), bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.035, 1, 0.955), h_pad=0.3, w_pad=0.3)
    # move the column titles just above the first row (centred at the same height)
    top = axes[0, 0].get_position().y1
    for ax in axes[0]:
        for t in ax.texts:
            if t.get_position() == (0.5, 1.0):
                t.set_transform(fig.transFigure); t.set_position((ax.get_position().x0 + ax.get_position().width / 2,
                                                                  top + 0.022))
    outl = ("White lines: cell boundaries after the watershed separation (h = %d)." % I.h) if I.a2 else \
        "Cell boundaries of the predictions are not drawn (qpicell could not be imported)."
    cap = (f"Qualitative segmentation of {len(rows)} test windows ({size} × {size} px, {size * I.px:.1f} µm), one per "
           "morphology class, each from a different image and centred on separately visible cells of that class. "
           f"Phase (grey scale, 0.5–99.5th percentile of the image); reference; predictions of each architecture at its "
           f"representative rank, proposed objective, seed {seed}. Colour: class; grey: outside the annotated field. "
           f"{outl}")
    return fig, "main", "fig6_qualitative", "6", cap


def figS7_superposition(inp: Inputs, R: Runs, half: int = 40):
    I = Images(inp, R)
    rel = f"{W}/w5_reference_cells/w5_superposition_candidates.csv"
    cand = inp.csv(rel)
    if not cand:
        raise MissingInput(rel)
    n = len(cand)
    ncol = min(6, n); nrow = int(np.ceil(n / ncol))
    # each panel spans two columns of a 2*ncol grid, so a shorter last row can be centred under the full rows
    fig = plt.figure(figsize=(S.DOUBLE, S.DOUBLE / ncol * nrow * 1.15 + 16 * S.MM))
    gs = fig.add_gridspec(nrow, 2 * ncol, left=0.01, right=0.99, top=0.93, bottom=0.2, wspace=0.22, hspace=0.32)
    axes = []
    for k in range(n):
        row, col = divmod(k, ncol)
        in_row = min(ncol, n - row * ncol)
        c0 = (ncol - in_row) + 2 * col
        axes.append(fig.add_subplot(gs[row, c0:c0 + 2]))
    allph = []
    for r in cand:
        allph.append(I.image(r["stem"])["phase"])
    lo, hi = np.percentile(np.concatenate([p[np.isfinite(p)].ravel()[::7] for p in allph]), [0.5, 99.8])
    im = None
    for ax, r in zip(axes, cand):
        d = I.image(r["stem"]); c = int(r["cell"])
        ys, xs = np.nonzero(d["cells"] == c)
        ax.set_xticks([]); ax.set_yticks([])
        if not len(ys):
            ax.text(0.5, 0.5, "cell not found", ha="center", transform=ax.transAxes, fontsize=7); continue
        cy, cx = int(ys.mean()), int(xs.mean())
        n_ = d["phase"].shape[0]
        y0, x0 = np.clip(cy - half, 0, n_ - 2 * half), np.clip(cx - half, 0, n_ - 2 * half)
        win = np.s_[y0:y0 + 2 * half, x0:x0 + 2 * half]
        # phase on one colour scale for all panels; the flagged cell outlined in the colour of its class
        im = ax.imshow(d["phase"][win], cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
        m = (d["cells"][win] == c).astype(float)
        col_ = S.CLASS_COLOR.get(r["gt_class"], "#ffffff")
        import matplotlib.patheffects as pe
        cs = ax.contour(m, levels=[0.5], colors=[col_], linewidths=1.3)
        for coll in getattr(cs, "collections", [cs]):
            coll.set_path_effects([pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])
        touch = "touching" if str(r["touching"]).lower() == "true" else "isolated"
        ax.set_title(f"Day {r['storage_day']}, {cls_name(r['gt_class'])}\nphase × {float(r['phase_ratio']):.2f}, {touch}",
                     fontsize=6.5, linespacing=1.25, pad=3)
    _scale(axes[0], 2 * half, I.px, um=5.0)
    leg = [Patch(fc="none", ec=S.CLASS_COLOR[k], lw=1.3, label=cls_name(k))
           for k in sorted({r["gt_class"] for r in cand}, key=lambda k: I.names.index(k) if k in I.names else 99)
           if k in S.CLASS_COLOR]
    if im is not None:
        cax = fig.add_axes([0.30, 0.09, 0.40, 0.022])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Phase (rad)", fontsize=8); cb.ax.tick_params(labelsize=7.5, length=2)
        cb.outline.set_linewidth(0.5)
    if leg:
        fig.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=1, frameon=False,
                   title="Outline: reference class", title_fontsize=7.5, fontsize=7.5, handlelength=1.2)
    scr = inp.json(f"{W}/w5_reference_cells/w5_reference_cells.json")["results"]["superposition_screen"]
    th = next((v for v in scr.values() if v["candidates_all_reference_cells"] == n), None)
    rule = (f"mean phase at least {th['phase_ratio_threshold']:g} × the median of the same class and storage day, "
            f"area at most {th['area_ratio_threshold']:g} × its median" if th else "see W-5")
    cap = (f"The {n} reference cells flagged by the superposition screen ({rule}). Phase on one colour "
           f"scale for all panels ({lo:.2f} to {hi:.2f} rad); outline: the reference cell, coloured by its class; "
           "'phase ×': ratio to the class-and-day median. The flagged cells are inspected visually; the automatic "
           "centre/rim labels of W-6 are not used.")
    return fig, "supplementary", "figS2_superposition_screen", "S2", cap


PRODUCTS = [fig6_qualitative, figS7_superposition]
