#!/usr/bin/env python
"""How large the phase actually gets, per image and per reference cell. No model, no graphics card, no retraining.

Why this exists: the manuscript applies no phase unwrapping, and that has to be justified rather than asserted. A
phase map can only be ambiguous where the phase approaches the 2 pi ceiling of the wrapped measurement, so the
justification is a statement about the largest phase the data actually reaches. The preparation already records the
extreme phase of each image and the pixels its wrapping detector flags, but nothing so far reports the phase of an
individual cell, which is what the cell-level readouts are computed from.

    python scripts/evaluation/phase_range.py                     every prepared image, every split
    python scripts/evaluation/phase_range.py --splits val        one split
    python scripts/evaluation/phase_range.py --no-cells          image-level only, much faster

What is measured. For every prepared image: the smallest and largest phase of the background-subtracted map that the
training and the evaluation both read, and the number of pixels the preparation's wrapping detector flagged on the
raw map before any background was removed. For every reference cell, that is every label of the saved watershed
split of the annotated foreground: its largest phase, its 99th percentile, its mean, its integrated phase, its area,
and whether any flagged pixel falls inside it. Every value is read off the arrays on disk; nothing is modelled.

The ceiling is read from the configuration (cells.wrapping.ceiling_rad), not written into this file, and the cells
are the same reference cells the rest of the evaluation scores, read from the saved cell maps rather than split
again here, so these numbers describe the very images the results were produced from.

Cost: processor only, one image at a time, a few minutes for the whole dataset. Nothing already on disk is
overwritten: the outputs go to results/consolidated/phase_range/.
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

CELL_KEYS = ("split", "stem", "storage_day", "cell", "area_px", "area_um2", "phase_max_rad",
             "phase_p99_rad", "phase_mean_rad", "integrated_phase_rad", "dry_mass_pg",
             "wrapped_pixels_in_cell", "headroom_to_ceiling_rad")
IMAGE_KEYS = ("split", "stem", "storage_day", "phase_min_rad", "phase_max_rad", "n_cells",
              "wrapped_pixels", "headroom_to_ceiling_rad")


def manifest_rows(prepared: Path) -> List[Dict]:
    with open(prepared / "manifest.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("registration_ok") == "True"]
    if not rows:
        raise SystemExit(f"[phase range] no usable rows in {prepared / 'manifest.csv'}")
    return sorted(rows, key=lambda r: (r["split"], r["stem"]))


def quantiles(v: np.ndarray) -> Dict[str, float]:
    """The shape of a distribution, reported the same way everywhere in this script."""
    if v.size == 0:
        return {k: None for k in ("n", "min", "median", "p95", "p99", "max", "mean", "sd")}
    return {"n": int(v.size), "min": float(v.min()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max()), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0}


def measure_image(prepared: Path, row: Dict, px_um: float, to_pg: float, ceiling: float,
                  with_cells: bool) -> Dict:
    split, stem = row["split"], row["stem"]
    phase = tifffile.imread(prepared / split / "phase" / f"{stem}.tif").astype(np.float64)
    wrapped = tifffile.imread(prepared / split / "wrapped" / f"{stem}.tif").astype(bool)
    day = int(row.get("storage_day", -1))
    image = {"split": split, "stem": stem, "storage_day": day,
             "phase_min_rad": float(phase.min()), "phase_max_rad": float(phase.max()),
             "wrapped_pixels": int(wrapped.sum()),
             "n_cells": int(row["n_cells"]) if row.get("n_cells") not in (None, "") else 0,
             "headroom_to_ceiling_rad": float(ceiling - phase.max())}
    # the preparation recorded the same two extremes; disagreeing with them would mean the arrays on disk are not
    # the arrays the manifest describes, so it is checked rather than trusted
    for key in ("phase_min_rad", "phase_max_rad"):
        recorded = row.get(key.replace("_rad", ""))
        if recorded not in (None, "") and abs(float(recorded) - image[key]) > 1e-5:
            raise SystemExit(f"[phase range] {stem}: manifest {key} is {recorded}, the array gives {image[key]}")
    if row.get("wrapped_pixels") not in (None, "") and int(row["wrapped_pixels"]) != image["wrapped_pixels"]:
        raise SystemExit(f"[phase range] {stem}: manifest wrapped_pixels is {row['wrapped_pixels']}, "
                         f"the array gives {image['wrapped_pixels']}")

    cells: List[Dict] = []
    if with_cells:
        labels = tifffile.imread(prepared / split / "cells" / f"{stem}.tif").astype(np.int32)
        if image["n_cells"] and int(labels.max()) != image["n_cells"]:
            raise SystemExit(f"[phase range] {stem}: manifest n_cells is {image['n_cells']}, the cell map has "
                             f"{int(labels.max())}")
        image["n_cells"] = int(labels.max())
        # one sort instead of one pass per cell: the pixels of every cell end up as a contiguous block
        flat = labels.ravel()
        inside = flat > 0
        lab, val, wr = flat[inside], phase.ravel()[inside], wrapped.ravel()[inside]
        order = np.argsort(lab, kind="stable")
        lab, val, wr = lab[order], val[order], wr[order]
        edges = np.searchsorted(lab, np.arange(1, image["n_cells"] + 2))
        for label in range(1, image["n_cells"] + 1):
            lo, hi = int(edges[label - 1]), int(edges[label])
            if hi <= lo:                     # a label the watershed did not place; counted, never measured
                continue
            v = val[lo:hi]
            integrated = float(v.sum())
            cells.append({"split": split, "stem": stem, "storage_day": day, "cell": label,
                          "area_px": hi - lo, "area_um2": (hi - lo) * px_um * px_um,
                          "phase_max_rad": float(v.max()), "phase_p99_rad": float(np.percentile(v, 99)),
                          "phase_mean_rad": float(v.mean()), "integrated_phase_rad": integrated,
                          "dry_mass_pg": integrated * px_um * px_um * to_pg,
                          "wrapped_pixels_in_cell": int(wr[lo:hi].sum()),
                          "headroom_to_ceiling_rad": float(ceiling - v.max())})
    return {"image": image, "cells": cells}


def summarise(images: List[Dict], cells: List[Dict], ceiling: float, jump: float) -> Dict:
    im_max = np.array([r["phase_max_rad"] for r in images], dtype=np.float64)
    cl_max = np.array([r["phase_max_rad"] for r in cells], dtype=np.float64)
    cl_mean = np.array([r["phase_mean_rad"] for r in cells], dtype=np.float64)
    flagged_cells = sum(1 for r in cells if r["wrapped_pixels_in_cell"] > 0)
    flagged_images = sum(1 for r in images if r["wrapped_pixels"] > 0)
    return {
        "ceiling_rad": ceiling, "half_ceiling_rad": jump,
        "n_images": len(images), "n_cells": len(cells),
        "image_phase_max": quantiles(im_max),
        "cell_phase_max": quantiles(cl_max),
        "cell_phase_mean": quantiles(cl_mean),
        "images_reaching_ceiling": int((im_max >= ceiling).sum()),
        "images_above_half_ceiling": int((im_max >= jump).sum()),
        "cells_reaching_ceiling": int((cl_max >= ceiling).sum()) if cl_max.size else 0,
        "cells_above_half_ceiling": int((cl_max >= jump).sum()) if cl_max.size else 0,
        "images_with_a_flagged_pixel": flagged_images,
        "cells_containing_a_flagged_pixel": flagged_cells,
        "flagged_pixels_total": int(sum(r["wrapped_pixels"] for r in images)),
        "smallest_headroom_to_ceiling_rad": float(ceiling - im_max.max()) if im_max.size else None,
    }


def summary_md(overall: Dict, per_split: Dict[str, Dict], px_um: float) -> str:
    def q(d: Dict, k: str, p: int = 3) -> str:
        v = d.get(k)
        return "-" if v is None else f"{v:,.{p}f}"

    c, h = overall["ceiling_rad"], overall["half_ceiling_rad"]
    L = ["# How large the phase gets, and why no unwrapping is applied", "",
         f"Measured on the prepared images themselves: {overall['n_images']} images and "
         f"{overall['n_cells']:,} reference cells, at {px_um} micrometres per pixel. The phase is the "
         "background-subtracted map the training and the evaluation both read; the flagged pixels are those the "
         "preparation's wrapping detector marked on the raw map, before any background was removed.", "",
         f"The wrapped measurement has a ceiling of {c:.6f} radians; half of it is {h:.6f} radians.", "",
         "| Quantity | Smallest | Median | 95th percentile | 99th percentile | Largest |", "|---|---|---|---|---|---|"]
    for name, key in (("Largest phase in an image (radians)", "image_phase_max"),
                      ("Largest phase in a cell (radians)", "cell_phase_max"),
                      ("Mean phase of a cell (radians)", "cell_phase_mean")):
        d = overall[key]
        L.append(f"| {name} | {q(d,'min')} | {q(d,'median')} | {q(d,'p95')} | {q(d,'p99')} | {q(d,'max')} |")
    L += ["", "## Against the ceiling", "",
          f"* Images whose largest phase reaches the ceiling: {overall['images_reaching_ceiling']} of "
          f"{overall['n_images']}.",
          f"* Images whose largest phase passes half the ceiling: {overall['images_above_half_ceiling']} of "
          f"{overall['n_images']}.",
          f"* Reference cells whose largest phase reaches the ceiling: {overall['cells_reaching_ceiling']} of "
          f"{overall['n_cells']:,}.",
          f"* Reference cells whose largest phase passes half the ceiling: "
          f"{overall['cells_above_half_ceiling']} of {overall['n_cells']:,}.",
          f"* Smallest distance from any image's largest phase to the ceiling: "
          f"{q(overall,'smallest_headroom_to_ceiling_rad')} radians.", "",
          "## What the wrapping detector flagged", "",
          f"* Pixels flagged over the whole dataset: {overall['flagged_pixels_total']:,}.",
          f"* Images with at least one flagged pixel: {overall['images_with_a_flagged_pixel']} of "
          f"{overall['n_images']}.",
          f"* Reference cells containing at least one flagged pixel: "
          f"{overall['cells_containing_a_flagged_pixel']:,} of {overall['n_cells']:,}.", "",
          "## Per split", "",
          "| Split | Images | Cells | Largest phase in any image | Largest phase in any cell | "
          "Cells containing a flagged pixel |", "|---|---|---|---|---|---|"]
    for split in sorted(per_split):
        s = per_split[split]
        L.append(f"| {split} | {s['n_images']} | {s['n_cells']:,} | "
                 f"{q(s['image_phase_max'],'max')} | {q(s['cell_phase_max'],'max')} | "
                 f"{s['cells_containing_a_flagged_pixel']:,} |")
    return "\n".join(L) + "\n"


def write_csv(path: Path, keys: Sequence[str], rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(keys), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default=None, help="comma-separated splits (default: every split in the manifest)")
    ap.add_argument("--no-cells", action="store_true", help="image-level only; skip the per-cell measurements")
    add_common(ap)
    a = ap.parse_args()
    cfg = config(a)

    prepared = Path(cfg["paths_abs"]["prepared"])
    ph = cfg["physics"]
    px_um = float(ph["pixel_size_um"])
    # dry mass per unit of integrated phase and area, the same constant the evaluation uses
    to_pg = float(ph["wavelength_um"]) / (2.0 * np.pi * float(ph["refractive_increment_ml_per_g"]))
    ceiling = float(cfg["cells"]["wrapping"]["ceiling_rad"])
    jump = float(cfg["cells"]["wrapping"]["jump_rad"])

    rows = manifest_rows(prepared)
    wanted = parse_list(a.splits) if a.splits else None
    if wanted:
        known = sorted({r["split"] for r in rows})
        unknown = [s for s in wanted if s not in known]
        if unknown:
            raise SystemExit(f"[phase range] unknown split(s) {', '.join(unknown)}; the manifest has "
                             f"{', '.join(known)}")
        rows = [r for r in rows if r["split"] in set(wanted)]

    print(f"[phase range] {len(rows)} images from {prepared / 'manifest.csv'}; "
          f"ceiling {ceiling:.6f} rad, pixel {px_um} um", flush=True)
    if a.no_cells:
        print("[phase range] --no-cells: the per-cell measurements are skipped", flush=True)

    images: List[Dict] = []
    cells: List[Dict] = []
    for i, row in enumerate(rows, 1):
        got = measure_image(prepared, row, px_um, to_pg, ceiling, not a.no_cells)
        images.append(got["image"])
        cells.extend(got["cells"])
        if i % 25 == 0 or i == len(rows):
            print(f"[phase range] {i} of {len(rows)} images, {len(cells):,} cells so far", flush=True)

    overall = summarise(images, cells, ceiling, jump)
    per_split = {s: summarise([r for r in images if r["split"] == s],
                              [r for r in cells if r["split"] == s], ceiling, jump)
                 for s in sorted({r["split"] for r in images})}

    out = Path(cfg["paths_abs"]["results"]) / "consolidated" / "phase_range"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "phase_range.json",
               {"physics": {"wavelength_um": ph["wavelength_um"],
                            "refractive_increment_ml_per_g": ph["refractive_increment_ml_per_g"],
                            "pixel_size_um": px_um},
                "wrapping": dict(cfg["cells"]["wrapping"]),
                "splits_measured": sorted({r["split"] for r in images}),
                "per_cell_measured": not a.no_cells,
                "overall": overall, "per_split": per_split})
    write_csv(out / "phase_range_images.csv", IMAGE_KEYS, images)
    if not a.no_cells:
        write_csv(out / "phase_range_cells.csv", CELL_KEYS, cells)
    (out / "phase_range_summary.md").write_text(summary_md(overall, per_split, px_um), encoding="utf-8")

    print(f"\n[phase range] largest phase in any image: {overall['image_phase_max']['max']:.4f} rad "
          f"({overall['smallest_headroom_to_ceiling_rad']:.4f} rad below the ceiling)", flush=True)
    if not a.no_cells:
        d = overall["cell_phase_max"]
        print(f"[phase range] largest phase in a cell: median {d['median']:.4f}, 99th {d['p99']:.4f}, "
              f"max {d['max']:.4f} rad over {overall['n_cells']:,} cells", flush=True)
        print(f"[phase range] cells reaching the ceiling: {overall['cells_reaching_ceiling']}; "
              f"cells containing a flagged pixel: {overall['cells_containing_a_flagged_pixel']:,}", flush=True)
    print(f"[phase range] written to {out}", flush=True)


if __name__ == "__main__":
    main()
