#!/usr/bin/env python
"""Does a change in the background phase change the measured dry mass? Measured, not argued. No model, no GPU.

    python scripts/evaluation/background_invariance.py
    python scripts/evaluation/background_invariance.py --splits val --amplitudes 0.25,0.5,1.0

Why this exists. A reviewer asked whether the result survives a background offset or a low-order distortion of
the phase. The pipeline removes the background with a polynomial of order 2, whose terms are a constant, two
tilts and three quadratics, so any perturbation of that form lies inside what the polynomial can represent. That
is often stated as if it settled the question. It does not, for one reason: the fit is not made on every pixel.
The estimator first decides which pixels are background, by an Otsu threshold and a local-texture test, and only
then fits. That decision is not linear in the phase, so a tilt can move the boundary of the background set, and
a fit on a different set of pixels is a different fit. Invariance is exact only for a fixed support. Whether it
holds in practice is a measurement, and this script makes it.

What it does, per image and per perturbation: it takes the prepared phase the pipeline actually uses, adds the
perturbation, re-estimates the background from scratch with the configured settings, subtracts it, and compares
the result with the prepared phase. It reports the largest and the typical change in the map, and - the number
that matters - the change in each reference cell's integrated phase and dry mass.

A perturbation is specified by which polynomial term it uses and how large it is at the edge of the field, in
radians. The default set is a constant, both tilts, and all three quadratic terms, at several amplitudes.

Cost: processor only, a few minutes per amplitude on the test split. It writes only into
results/consolidated/background_invariance/ and touches nothing else.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from qpicell.cli import add_common, config, parse_list  # noqa: E402
from qpicell.data.background import design_matrix, estimate_background  # noqa: E402
from qpicell.io import write_json  # noqa: E402

# Which polynomial term each perturbation uses, written out. The index is the column of the order-2 design
# matrix: 1 | x, y | x^2, x y, y^2. The amplitude is the value the term reaches at the edge of the field, where
# the normalised coordinate is 1, so every perturbation is quoted in radians and they are comparable.
TERMS: Sequence = (("constant offset", 0), ("tilt along x", 1), ("tilt along y", 2),
                   ("curvature along x", 3), ("saddle, x times y", 4), ("curvature along y", 5))


def perturbation(shape, column: int, amplitude: float) -> np.ndarray:
    a = np.zeros(6)
    a[column] = float(amplitude)
    return (design_matrix(shape, 2) @ a).reshape(shape)


def cell_sums(phase: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    """Integrated phase of every reference cell, label 1..n, by one pass over the image."""
    if n == 0:
        return np.zeros(0)
    return np.bincount(labels.ravel(), weights=phase.ravel(), minlength=n + 1)[1:]


TAG = "background"

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


def manifest_rows(prepared: Path, cfg: Dict, splits) -> List[Dict]:
    rows = usable_rows(prepared)
    if splits is None:                                   # default: the held-out evaluation images
        splits = [evaluation_split(cfg, rows)]
    known = sorted({r["split"] for r in rows})
    bad = [s for s in splits if s not in known]
    if bad:
        raise SystemExit(f"[background] unknown folder(s) {', '.join(bad)}; the manifest has "
                         f"{', '.join(known)}")
    rows = [r for r in rows if r["split"] in set(splits)]
    if not rows:
        raise SystemExit(f"[background] nothing to measure in {', '.join(splits)}")
    return sorted(rows, key=lambda r: (r["split"], r["stem"]))


def measure_one(job):
    """One image, every perturbation. Returns small arrays, so this is cheap to send back from a worker.

    The expensive part is re-estimating the background, which is an iterative fit with an Otsu threshold and a
    local-texture test on a 768 x 768 map, and it has to be done once per perturbation per image. On the real
    data that is far too slow to do one image after another, so the images are spread over processes."""
    prepared, row, k_pg, bg, terms, amps = job
    split, stem = row["split"], row["stem"]
    phi = tifffile.imread(Path(prepared) / split / "phase" / f"{stem}.tif").astype(np.float64)
    labels = tifffile.imread(Path(prepared) / split / "cells" / f"{stem}.tif").astype(np.int64)
    n = int(labels.max())
    s0 = cell_sums(phi, labels, n)

    # control: re-estimating the background of the prepared map with no perturbation is not a no-op, because
    # the background has already been removed once. It is measured so the perturbations can be read against
    # it: a change no larger than this one is a change the estimator makes anyway.
    b0, _ = estimate_background(phi, bg["order"], bg["k_sigma"], bg["grow_px"], bg["max_iter"])
    out = {"stem": stem, "control": float(np.abs(b0).max()), "n_cells": n, "per": {}}

    for name, colno in terms:
        for amp in amps:
            psi = phi + perturbation(phi.shape, colno, amp)
            b, _ = estimate_background(psi, bg["order"], bg["k_sigma"], bg["grow_px"], bg["max_iter"])
            back = psi - b                                   # what the pipeline would use for this image
            s1 = cell_sums(back, labels, n)
            rel = ((s1 - s0) / np.where(np.abs(s0) > 1e-9, s0, np.nan)) if n else np.zeros(0)
            out["per"][f"{name}|{amp}"] = {
                "map": float(np.abs(back - phi).max()),
                "rel": rel[np.isfinite(rel)].tolist(),
                "dm": ((s1 - s0) * k_pg).tolist()}
    return out


def summarise(v: np.ndarray) -> Dict:
    if v.size == 0:
        return {"n": 0}
    a = np.abs(v)
    return {"n": int(v.size), "median_abs": float(np.median(a)), "p95_abs": float(np.percentile(a, 95)),
            "max_abs": float(a.max()), "mean_signed": float(v.mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default=None,
                    help="comma-separated prepared folders, or 'all' (default: the held-out evaluation images)")
    ap.add_argument("--amplitudes", default="0.5",
                    help="comma-separated perturbation amplitudes in radians at the edge of the field. The "
                         "default is the single amplitude the previous revision reported. Each extra amplitude "
                         "costs another full re-estimate of the background for every image and every term.")
    ap.add_argument("--workers", type=int, default=8,
                    help="processes; each handles one image at a time (0 or 1 = no pool)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images (0 = every image)")
    add_common(ap)
    a = ap.parse_args()
    cfg = config(a)

    prepared = Path(cfg["paths_abs"]["prepared"])
    bg = cfg["background"]
    ph = cfg["physics"]
    px_um = float(ph["pixel_size_um"])
    px2 = px_um ** 2
    k_pg = float(ph["wavelength_um"]) / (2.0 * np.pi * float(ph["refractive_increment_ml_per_g"])) * px2

    splits = parse_list(a.splits)
    amps = [float(x) for x in str(a.amplitudes).split(",") if x.strip()]
    rows = manifest_rows(prepared, cfg, splits)
    if a.limit:
        rows = rows[:a.limit]

    print(f"[background] {len(rows)} images; order {bg['order']}, k {bg['k_sigma']}, grow {bg['grow_px']} px, "
          f"at most {bg['max_iter']} iterations", flush=True)
    print(f"[background] amplitudes (radians at the edge of the field): "
          f"{', '.join(f'{x:g}' for x in amps)}", flush=True)
    fits = len(rows) * (len(TERMS) * len(amps) + 1)
    print(f"[background] {len(TERMS)} polynomial terms x {len(amps)} amplitude(s) = "
          f"{len(TERMS) * len(amps)} perturbations per image, {fits:,} background fits in all, "
          f"over {max(1, int(a.workers))} process(es)", flush=True)

    jobs = [(str(prepared), r, k_pg, dict(bg), list(TERMS), amps) for r in rows]
    results: List[Dict] = []
    workers = max(1, int(a.workers))
    if workers > 1 and len(jobs) > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx) as ex:
            for i, got in enumerate(ex.map(measure_one, jobs, chunksize=1), 1):
                results.append(got)
                if i <= 2 or i % 5 == 0 or i == len(jobs):
                    print(f"[background] {i} of {len(jobs)} images", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            results.append(measure_one(job))
            if i <= 2 or i % 5 == 0 or i == len(jobs):
                print(f"[background] {i} of {len(jobs)} images", flush=True)

    control = [g["control"] for g in results]
    per: Dict = {}
    for g in results:
        for key, v in g["per"].items():
            name, amp = key.rsplit("|", 1)
            acc = per.setdefault((name, float(amp)),
                                 {"map": [], "cell_rel": [], "cell_dm": [], "images": 0, "cells": 0})
            acc["map"].append(v["map"])
            acc["cell_rel"].append(np.asarray(v["rel"], float))
            acc["cell_dm"].append(np.asarray(v["dm"], float))
            acc["images"] += 1
            acc["cells"] += g["n_cells"]

    out_rows = []
    for (name, amp), acc in sorted(per.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rel = np.concatenate(acc["cell_rel"]) if acc["cell_rel"] else np.zeros(0)
        dm = np.concatenate(acc["cell_dm"]) if acc["cell_dm"] else np.zeros(0)
        out_rows.append({"term": name, "amplitude_rad": amp, "images": acc["images"], "cells": int(rel.size),
                         "map_change_max_rad": float(max(acc["map"])) if acc["map"] else None,
                         "map_change_median_of_image_maxima_rad": float(np.median(acc["map"])) if acc["map"] else None,
                         "cell_integrated_phase_change_pct": {k: (v * 100 if isinstance(v, float) else v)
                                                              for k, v in summarise(rel).items()},
                         "cell_dry_mass_change_pg": summarise(dm)})

    out = Path(cfg["paths_abs"]["results"]) / "consolidated" / "background_invariance"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "background_invariance.json",
               {"splits": sorted({r["split"] for r in rows}), "n_images": len(rows),
                "background_settings": dict(bg), "amplitudes_rad": amps,
                "terms": [t[0] for t in TERMS],
                "control_reestimate_max_change_rad": summarise(np.array(control)),
                "results": out_rows})

    flat = []
    for r in out_rows:
        flat.append({"term": r["term"], "amplitude_rad": r["amplitude_rad"], "images": r["images"],
                     "cells": r["cells"], "map_change_max_rad": r["map_change_max_rad"],
                     "cell_phase_change_median_abs_pct": r["cell_integrated_phase_change_pct"].get("median_abs"),
                     "cell_phase_change_p95_abs_pct": r["cell_integrated_phase_change_pct"].get("p95_abs"),
                     "cell_phase_change_max_abs_pct": r["cell_integrated_phase_change_pct"].get("max_abs"),
                     "cell_dry_mass_change_median_abs_pg": r["cell_dry_mass_change_pg"].get("median_abs"),
                     "cell_dry_mass_change_max_abs_pg": r["cell_dry_mass_change_pg"].get("max_abs")})
    with open(out / "background_invariance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]))
        w.writeheader(); w.writerows(flat)

    ctl = summarise(np.array(control))
    L = ["# Does a background perturbation change the measurement?", "",
         f"Measured on {len(rows)} prepared images ({', '.join(sorted({r['split'] for r in rows}))}), "
         f"{out_rows[0]['cells'] if out_rows else 0:,} reference cells. For each perturbation the background is "
         "re-estimated from scratch with the settings the pipeline uses, subtracted, and the result compared "
         "with the phase map the pipeline actually uses.", "",
         "The amplitude is the value the perturbing term reaches at the edge of the field, in radians.", "",
         f"Re-estimating the background of an unperturbed map already changes it by at most "
         f"{ctl.get('max_abs', float('nan')):.2e} radians (median over images "
         f"{ctl.get('median_abs', float('nan')):.2e}). A perturbation that changes nothing more than this has "
         "changed nothing the estimator does not change anyway.", "",
         "| Perturbation | Amplitude (rad) | Largest change in the map (rad) | Cell integrated phase, median "
         "(%) | Cell integrated phase, largest (%) | Cell dry mass, largest (pg) |", "|---|---|---|---|---|---|"]
    for r in flat:
        L.append(f"| {r['term']} | {r['amplitude_rad']:g} | {r['map_change_max_rad']:.3e} | "
                 f"{(r['cell_phase_change_median_abs_pct'] or 0):.4f} | "
                 f"{(r['cell_phase_change_max_abs_pct'] or 0):.4f} | "
                 f"{(r['cell_dry_mass_change_max_abs_pg'] or 0):.4e} |")
    L += ["", "Read it this way: the background estimator removes a perturbation it can represent, but only as "
              "far as its choice of background pixels is unchanged by that perturbation. The numbers above are "
              "what is left over after it has done so.", ""]
    (out / "background_invariance_summary.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n[background] largest cell dry-mass change over every perturbation: "
          f"{max((r['cell_dry_mass_change_max_abs_pg'] or 0) for r in flat):.3e} pg", flush=True)
    print(f"[background] written to {out}", flush=True)


if __name__ == "__main__":
    main()
