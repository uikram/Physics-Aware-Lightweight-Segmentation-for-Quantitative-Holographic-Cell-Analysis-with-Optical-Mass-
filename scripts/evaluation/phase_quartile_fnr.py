#!/usr/bin/env python
"""Does the cell-level term cost anything where the phase is low? False negatives by phase quartile. No GPU.

    python scripts/evaluation/phase_quartile_fnr.py
    python scripts/evaluation/phase_quartile_fnr.py --arms dice,dice_cipp --bins 4

Why this exists. A reviewer asked whether a loss term that rewards preserving integrated phase biases the
network against the parts of a cell where the phase is low - thin rims, the edge of a discocyte - because
missing them costs little phase. The way to answer it is to split the annotated foreground by how much phase a
pixel carries and ask how often each part is missed. If the term did bias the network that way, the lowest
quartile would get worse while the highest got better.

What is measured. A pixel is a false negative when the annotation calls it cell and the prediction does not.
The false-negative rate of a quartile is its false negatives divided by the annotated pixels in it. Nothing is
recomputed from a model: the class maps saved by the evaluation are read as they are, so these numbers come from
the very predictions every other result was scored on.

The quartile edges are computed once, over the annotated foreground pixels of every test image, and are the same
for every run and every loss configuration - otherwise "the lowest quartile" would mean a different set of
pixels for each run and the comparison would be meaningless. The edges are written into the output.

Cost: processor only, about a second per run plus one pass over the test images to find the edges.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from qpicell.cli import add_common, config, parse_list  # noqa: E402
from qpicell.io import write_json  # noqa: E402
from qpicell.training.runs import Run, results_dir  # noqa: E402


TAG = "quartile"

def usable_rows(prepared: Path) -> List[Dict]:
    with open(prepared / "manifest.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("registration_ok") == "True"]
    if not rows:
        raise SystemExit(f"[{TAG}] no usable rows in {prepared / 'manifest.csv'}")
    return rows


def evaluation_split(cfg: Dict, rows: List[Dict]) -> str:
    """Which prepared folder holds the held-out evaluation images.

    The configuration names the splits by their role (train, validation, test) while the prepared folders are
    named after the directories the dataset was delivered in, so the two do not line up and the folder name is
    not written into this file. The evaluation folder is the one holding exactly as many usable images as
    split.expected.test. If that does not identify one folder, this stops and says what it found rather than
    guessing which folder to measure."""
    want = int(cfg["split"]["expected"]["test"])
    counts = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    hits = [s for s, n in counts.items() if n == want]
    if len(hits) != 1:
        raise SystemExit(f"[{TAG}] cannot tell which prepared folder holds the {want} evaluation images: "
                         + ", ".join(f"{s} has {n}" for s, n in sorted(counts.items()))
                         + ". Name it explicitly instead.")
    return hits[0]


def test_images(cfg: Dict) -> tuple:
    """(folder, stems) of the held-out evaluation images, from the preparation manifest.

    Read from the manifest rather than through the dataset module so this script needs neither torch nor a data
    loader: it only ever opens saved arrays."""
    prepared = Path(cfg["paths_abs"]["prepared"])
    rows = usable_rows(prepared)
    split = evaluation_split(cfg, rows)
    return split, sorted(r["stem"] for r in rows if r["split"] == split)


def load_arrays(cfg: Dict, split: str, stems: Sequence[str]) -> Dict[str, Dict]:
    prepared = Path(cfg["paths_abs"]["prepared"])
    ign = int(cfg["data"]["ignore_index"])
    out = {}
    for s in stems:
        phase = tifffile.imread(prepared / split / "phase" / f"{s}.tif").astype(np.float64)
        mask = tifffile.imread(prepared / split / "mask" / f"{s}.tif").astype(np.int64)
        valid = mask != ign
        out[s] = {"phase": phase, "fg": (mask > 0) & valid, "valid": valid}
    return out


def quartile_edges(arrays: Dict[str, Dict], bins: int) -> np.ndarray:
    """Edges of the phase bins, from the annotated foreground pixels of every test image pooled together."""
    v = np.concatenate([a["phase"][a["fg"]] for a in arrays.values()])
    qs = np.linspace(0, 100, bins + 1)[1:-1]
    return np.percentile(v, qs)


def representative(cfg: Dict) -> List[Dict]:
    p = Path(cfg["paths_abs"]["results"]) / "consolidated" / "selection" / "rank_selection.csv"
    if not p.exists():
        raise SystemExit(f"[quartile] {p} not found: run objective_and_rank_selection.py first")
    with open(p) as fh:
        return list(csv.DictReader(fh))


def run_rates(npz_path: Path, arrays: Dict[str, Dict], edges: np.ndarray, bins: int):
    """-> (annotated pixels per bin, false negatives per bin) for one saved set of class maps."""
    ann = np.zeros(bins, np.int64)
    fn = np.zeros(bins, np.int64)
    with np.load(npz_path) as Z:
        keys = [k for k in Z.files if k in arrays]
        if len(keys) != len(arrays):
            return None, None, f"{len(keys)} of {len(arrays)} test images in the saved predictions"
        for k in keys:
            a = arrays[k]
            fg = a["fg"]
            if not fg.any():
                continue
            pred_fg = (Z[k] > 0) & a["valid"]
            b = np.digitize(a["phase"][fg], edges)          # 0..bins-1
            ann += np.bincount(b, minlength=bins)
            fn += np.bincount(b[~pred_fg[fg]], minlength=bins)
    return ann, fn, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="dice,dice_cipp",
                    help="comma-separated loss configurations to compare (default: dice,dice_cipp)")
    ap.add_argument("--bins", type=int, default=4, help="how many phase bins (default: 4, quartiles)")
    ap.add_argument("--checkpoint", default="last", help="which checkpoint (default: last)")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds (default: every configured seed)")
    add_common(ap)
    a = ap.parse_args()
    cfg = config(a)
    if a.bins < 2:
        raise SystemExit("[quartile] --bins must be at least 2")

    arms = parse_list(a.arms) or []
    known = list(cfg["arms"])
    bad = [x for x in arms if x not in known]
    if bad:
        raise SystemExit(f"[quartile] unknown loss configuration(s) {', '.join(bad)}; "
                         f"the configuration has {', '.join(known)}")
    seeds = [int(s) for s in (parse_list(a.seeds) or cfg["training"]["seeds"])]
    sel = representative(cfg)

    split, stems = test_images(cfg)
    print(f"[quartile] {len(stems)} evaluation images from the '{split}' folder; "
          f"loading phase and annotation", flush=True)
    arrays = load_arrays(cfg, split, stems)
    edges = quartile_edges(arrays, a.bins)
    total_fg = int(sum(int(x["fg"].sum()) for x in arrays.values()))
    print(f"[quartile] {total_fg:,} annotated cell pixels; bin edges (radians): "
          f"{', '.join(f'{e:.4f}' for e in edges)}", flush=True)

    label = ["lowest phase"] + [f"bin {i + 1}" for i in range(1, a.bins - 1)] + ["highest phase"]
    rows: List[Dict] = []
    skipped: List[str] = []
    for s in sel:
        arch, cond = s["architecture"], s["selected_condition"]
        for arm in arms:
            for seed in seeds:
                rdir = results_dir(cfg, Run(arch, cond, arm, seed))
                npz = rdir / f"predictions_{a.checkpoint}.npz"
                if not npz.exists():
                    skipped.append(f"{arch} {cond} {arm} seed {seed}: no saved predictions")
                    continue
                ann, fn, why = run_rates(npz, arrays, edges, a.bins)
                if why:
                    skipped.append(f"{arch} {cond} {arm} seed {seed}: {why}")
                    continue
                r = {"architecture": arch, "architecture_display": s.get("architecture_display", arch),
                     "condition": cond, "loss": arm, "seed": seed,
                     "annotated_pixels": int(ann.sum()), "false_negatives": int(fn.sum()),
                     "false_negative_rate_pct": 100.0 * float(fn.sum()) / max(int(ann.sum()), 1)}
                for i in range(a.bins):
                    r[f"fnr_bin{i + 1}_pct"] = 100.0 * float(fn[i]) / max(int(ann[i]), 1)
                    r[f"annotated_bin{i + 1}"] = int(ann[i])
                rows.append(r)
                print(f"[quartile] {s.get('architecture_display', arch):<16}{cond:<8}{arm:<20}seed {seed}  "
                      + "  ".join(f"{r[f'fnr_bin{i + 1}_pct']:6.2f}" for i in range(a.bins)), flush=True)

    if not rows:
        raise SystemExit("[quartile] no run had saved predictions; nothing was written")

    # the comparison the reviewer asked for: the same architecture, rank and seed, one loss against another
    deltas: List[Dict] = []
    if len(arms) >= 2:
        base = arms[0]
        for other in arms[1:]:
            for s in sel:
                for seed in seeds:
                    g = {r["loss"]: r for r in rows if r["architecture"] == s["architecture"]
                         and r["condition"] == s["selected_condition"] and r["seed"] == seed}
                    if base not in g or other not in g:
                        continue
                    d = {"architecture_display": g[base]["architecture_display"],
                         "condition": s["selected_condition"], "seed": seed,
                         "from": base, "to": other}
                    for i in range(a.bins):
                        k = f"fnr_bin{i + 1}_pct"
                        d[f"change_bin{i + 1}_pp"] = g[other][k] - g[base][k]
                    d["change_overall_pp"] = (g[other]["false_negative_rate_pct"]
                                              - g[base]["false_negative_rate_pct"])
                    deltas.append(d)

    out = Path(cfg["paths_abs"]["results"]) / "consolidated" / "phase_quartile_fnr"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "phase_quartile_fnr.json",
               {"checkpoint": a.checkpoint, "bins": a.bins, "bin_edges_rad": [float(e) for e in edges],
                "bin_labels": label, "arms": arms, "seeds": seeds, "n_test_images": len(stems),
                "annotated_cell_pixels": total_fg, "runs": rows, "changes": deltas, "skipped": skipped})
    with open(out / "phase_quartile_fnr.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    L = ["# False negatives by phase, and what the cell-level term does to them", "",
         f"{len(stems)} held-out test images, {total_fg:,} annotated cell pixels, checkpoint "
         f"'{a.checkpoint}'. A pixel is a false negative when the annotation calls it cell and the prediction "
         "does not. The bins split the annotated cell pixels by how much phase they carry; the edges are the "
         "same for every run, so a bin means the same pixels throughout.", "",
         "Bin edges (radians): " + ", ".join(f"{e:.4f}" for e in edges), "",
         "| Architecture | Low-rank adaptation | Loss | Seed | " +
         " | ".join(f"{n} (%)" for n in label) + " | All pixels (%) |",
         "|---" * (5 + a.bins) + "|"]
    for r in rows:
        L.append(f"| {str(r['architecture_display']).replace('_', '-')} | "
                 f"{'full fine-tuning' if r['condition'] == 'fullft' else 'r = ' + r['condition'][1:]} | "
                 f"{r['loss']} | {r['seed']} | "
                 + " | ".join(f"{r[f'fnr_bin{i + 1}_pct']:.2f}" for i in range(a.bins))
                 + f" | {r['false_negative_rate_pct']:.2f} |")
    if deltas:
        L += ["", "## Change from one loss to another, same architecture, rank and seed", "",
              "A negative number means fewer false negatives under the second loss.", "",
              "| Architecture | Seed | From | To | " + " | ".join(f"{n} (pp)" for n in label) +
              " | All pixels (pp) |", "|---" * (5 + a.bins) + "|"]
        for d in deltas:
            L.append(f"| {str(d['architecture_display']).replace('_', '-')} | {d['seed']} | {d['from']} | "
                     f"{d['to']} | "
                     + " | ".join(f"{d[f'change_bin{i + 1}_pp']:+.2f}" for i in range(a.bins))
                     + f" | {d['change_overall_pp']:+.2f} |")
        lo = [d["change_bin1_pp"] for d in deltas]
        hi = [d[f"change_bin{a.bins}_pp"] for d in deltas]
        L += ["", f"Lowest-phase bin: {sum(1 for x in lo if x < 0)} of {len(lo)} comparisons improve, median "
                  f"change {np.median(lo):+.2f} percentage points. Highest-phase bin: "
                  f"{sum(1 for x in hi if x < 0)} of {len(hi)} improve, median {np.median(hi):+.2f}.",
              "",
              "A term that bought phase accuracy by giving up the faint edges of cells would show the lowest "
              "bin getting worse while the highest improved. Read the two lines above against each other.", ""]
    if skipped:
        L += ["## Not included", ""] + [f"* {s}" for s in skipped] + [""]
    (out / "phase_quartile_fnr_summary.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n[quartile] {len(rows)} runs, {len(deltas)} paired comparisons", flush=True)
    print(f"[quartile] written to {out}", flush=True)


if __name__ == "__main__":
    main()
