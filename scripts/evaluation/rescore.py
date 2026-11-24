#!/usr/bin/env python
"""Recompute the results of finished runs from their saved predictions. No model, no graphics card, no retraining.

Why this exists: the shape measurements, the perimeter and the circularity of every cell, were added to the scoring
code after the runs had finished. Nothing about the trained networks changed, and every array the scoring needs was
kept on disk, so the numbers are recomputed instead of retrained.

    python scripts/evaluation/rescore.py                                   every run, every checkpoint
    python scripts/evaluation/rescore.py --dry-run                         compare only, write nothing
    python scripts/evaluation/rescore.py --archs edge_sam --seeds 42       part of the grid
    python scripts/evaluation/rescore.py --workers 6                       more processes, more memory

How it works, per run and per checkpoint. The saved class maps are read from predictions_<tag>.npz and each one is
paired with the prepared phase map, annotation, cell map and wrapping mask that produced it, exactly as the
evaluation pairs them. Then the same score_predictions() the evaluation calls is called on that pairing. The saved
class maps are the arg-max the network produced, so every result that existed before has to come back identical,
and that is checked rather than assumed: the recomputed semantic, instance and reference-cell results are compared
field by field against the stored metrics_<tag>.json, and a checkpoint whose numbers disagree by more than
--tolerance is reported and left on disk untouched. The new measurements are per cell, so they appear in
cells_<tag>.json; metrics_<tag>.json carries no shape field and is expected to be reproduced exactly.

It rewrites metrics_<tag>.json, cells_<tag>.json and per_image_<tag>.json in place, keeping the metadata the
evaluation wrote and adding a record of the rescoring to it. predictions_<tag>.npz is never written. Pass --dry-run
first if you want the agreement check without touching anything.

Cost: no network is loaded and no image is pushed through one, so this is processor work only and no graphics card
is used. Each process holds the prepared test images in memory, about 0.6 GB, so --workers trades memory for time.
The script prints its own rate and a finishing estimate once the first checkpoints are done, rather than predicting
one in advance.

A checkpoint with no predictions_<tag>.npz, or no stored metrics_<tag>.json to check against, is skipped and
counted. The run ends with a line per skipped and per disagreeing checkpoint, and a log in
results/consolidated/rescore_log.json. The exit status is non-zero if anything disagreed.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402  (reads the prepared arrays; no model and no device)

from qpicell.cli import add_common, config, parse_list  # noqa: E402
from qpicell.data.dataset import PhaseDataset, read_manifest  # noqa: E402
from qpicell.evaluation.evaluate import score_predictions  # noqa: E402
from qpicell.io import _clean, read_json, write_json  # noqa: E402
from qpicell.physics import assert_physics, split_stems, verify_prepared  # noqa: E402
from qpicell.training.runs import Run, conditions, grid, results_dir  # noqa: E402

# The parts of a results file that the scoring produces. The rest of the file is metadata, which carries timestamps
# and timings and so is not comparable between two runs of the scoring.
SCORED = ("semantic", "instances", "reference_cells")

RESCORE_REASON = "shape measurements (perimeter, circularity) added after the run finished"


def prepared_arrays(cfg: Dict, image_size: int, stems: Sequence[str]) -> Dict[str, Dict]:
    """The native arrays of every test image, in the form the evaluation hands them to the scoring.

    The conversions below are the ones in qpicell.evaluation.evaluate.predict, so the scoring receives exactly what
    it received when the run was evaluated. Only the predicted class map is missing; that comes from the saved file.

    These arrays are the same for all three architectures: the dataset reads each one straight from its prepared
    image and resizes only the copies the network would see, which the scoring never touches. `image_size`
    therefore only has to be a size the dataset accepts, and the check below confirms that what came back is on the
    native grid rather than a resized one.
    """
    ds = PhaseDataset(cfg, "val", image_size, cfg["data"]["eval_rotations"], return_native=True, stems=list(stems))
    out: Dict[str, Dict] = {}
    for b in DataLoader(ds, batch_size=1, shuffle=False, num_workers=0):
        for i, sid in enumerate(b["sample_id"]):
            out[sid] = dict(phase=b["phase_native"][i, 0].numpy(),
                            mask=b["mask_native"][i].numpy().astype(np.uint8),
                            cells=b["cells_native"][i].numpy(),
                            wrapped=b["wrapped_native"][i].numpy(),
                            day=int(b["storage_day"][i]))
    if len(out) != len(stems):
        raise RuntimeError(f"{len(out)} prepared test images for {len(stems)} test stems")
    native = int(cfg["data"]["native_size"])
    for sid, d in out.items():
        shapes = {n: tuple(d[n].shape) for n in ("phase", "mask", "cells", "wrapped")}
        if len(set(shapes.values())) != 1:
            raise RuntimeError(f"{sid}: the prepared arrays have different shapes {shapes}")
        if shapes["phase"] != (native, native):
            raise RuntimeError(f"{sid}: prepared arrays are {shapes['phase']}, not the native {native} x {native} grid")
    return out


def leaves(x, path: str = "") -> Iterator[Tuple[str, object]]:
    """Every leaf of a nested results tree, as (dotted name, value)."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def disagreements(stored: Dict, fresh: Dict, tol: float) -> List[Tuple[str, object, object]]:
    """Fields where the stored and the recomputed results differ, as (name, stored, recomputed).

    A field counts as equal when the two values differ by no more than `tol` relative to the stored magnitude. Both
    trees are passed through the normalisation the results writer applies, so a value written as null because it
    was not a finite number is compared as null on both sides rather than as a number that equals nothing, itself
    included.
    """
    S = dict(leaves(json.loads(json.dumps(_clean(stored)))))
    F = dict(leaves(json.loads(json.dumps(_clean(fresh)))))
    bad: List[Tuple[str, object, object]] = []
    for name in sorted(set(S) | set(F)):
        a = S.get(name, "<not in the stored results>")
        b = F.get(name, "<not in the recomputed results>")
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b))
        if numeric:
            if abs(float(a) - float(b)) > tol * max(1.0, abs(float(a))):
                bad.append((name, a, b))
        # a plain inequality would let True pass for 1 and False for 0, so the kind of value has to match as well
        elif a != b or isinstance(a, bool) != isinstance(b, bool):
            bad.append((name, a, b))
    return bad


def rescore_checkpoint(cfg: Dict, run: Run, tag: str, arrays: Dict[str, Dict], sha: Dict[str, str],
                       stems: Sequence[str], tol: float, dry_run: bool) -> Dict:
    """One run, one checkpoint. Returns a record whose "outcome" is rescored, skipped or disagreed."""
    rdir = results_dir(cfg, run)
    npz, mj = rdir / f"predictions_{tag}.npz", rdir / f"metrics_{tag}.json"
    rec = {"run": run.run_id, "checkpoint": tag}
    if not npz.exists():
        return {**rec, "outcome": "skipped", "reason": "no saved predictions"}
    if not mj.exists():
        return {**rec, "outcome": "skipped", "reason": "no stored results to check against"}
    stored = read_json(mj)
    with np.load(npz) as Z:
        keys = list(Z.files)
        missing = [k for k in keys if k not in arrays]
        if missing:
            return {**rec, "outcome": "disagreed",
                    "reason": f"the saved predictions name {len(missing)} images that are not prepared test images, "
                              f"the first being {missing[0]}"}
        if len(keys) != len(stems):
            return {**rec, "outcome": "disagreed", "reason": f"{len(keys)} saved predictions for {len(stems)} test images"}
        preds = {k: dict(arrays[k], pred=Z[k]) for k in keys}

    t0 = time.time()
    res = score_predictions(cfg, preds, sha)
    seconds = time.time() - t0
    tables = res.pop("cells_tables"); per_image = res.pop("per_image")

    bad = disagreements({k: stored.get(k) for k in SCORED}, {k: res[k] for k in SCORED}, tol)
    if bad:
        return {**rec, "outcome": "disagreed", "seconds": seconds,
                "reason": f"{len(bad)} field{'' if len(bad) == 1 else 's'} differ"
                          f"{'s' if len(bad) == 1 else ''} from the stored results",
                "fields": [{"field": f, "stored": s, "recomputed": v} for f, s, v in bad[:20]]}

    records = tables["A2"].to_dict("records")
    shaped = sum(1 for row in records if row.get("circularity") is not None)
    if not dry_run:
        meta = dict(stored["metadata"])
        meta["rescored"] = {"reason": RESCORE_REASON, "source": f"predictions_{tag}.npz",
                            "stored_results_reproduced": True, "tolerance": tol,
                            "scoring_seconds": seconds, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        cells = []
        for mode, C in tables.items():
            cells += [{"instances": mode, **row} for row in C.to_dict("records")]
        write_json(mj, {"metadata": meta, **res})
        write_json(rdir / f"cells_{tag}.json", {"metadata": meta, "cells": cells})
        write_json(rdir / f"per_image_{tag}.json", {"metadata": meta, "images": per_image})
    return {**rec, "outcome": "rescored", "seconds": seconds,
            "reference_cells": len(records), "cells_with_shape": shaped}


def worker(job: Tuple) -> List[Dict]:
    """One process: read the prepared images once, then rescore its share of the runs."""
    cfg, run_dicts, tags, stems, sha, tol, dry_run, size, label = job
    runs = [Run(d["arch"], d["condition"], d["arm"], d["seed"]) for d in run_dicts]
    arrays = prepared_arrays(cfg, size, stems)
    out: List[Dict] = []
    for n, r in enumerate(runs, 1):
        for tag in tags:
            rec = rescore_checkpoint(cfg, r, tag, arrays, sha, stems, tol, dry_run)
            out.append(rec)
            if rec["outcome"] == "rescored":
                print(f"[rescore] {label} {n}/{len(runs)} {r.run_id} {tag}: stored results reproduced, "
                      f"{rec['cells_with_shape']}/{rec['reference_cells']} cells carry a perimeter and a "
                      f"circularity ({rec['seconds']:.1f} s)", flush=True)
            elif rec["outcome"] == "disagreed":
                first = rec.get("fields", [{}])[0]
                detail = (f" (first: {first['field']} stored {first['stored']} recomputed {first['recomputed']})"
                          if first else "")
                print(f"[rescore] {label} {n}/{len(runs)} {r.run_id} {tag}: {rec['reason']}, "
                      f"LEFT UNTOUCHED{detail}", flush=True)
            else:
                print(f"[rescore] {label} {n}/{len(runs)} {r.run_id} {tag}: skipped, {rec['reason']}", flush=True)
    return out


def chunks(items: List, n: int) -> List[List]:
    """Split a list into at most n near-equal consecutive pieces."""
    n = max(1, min(n, len(items)))
    size, extra = divmod(len(items), n)
    out, i = [], 0
    for k in range(n):
        j = i + size + (1 if k < extra else 0)
        out.append(items[i:j]); i = j
    return [c for c in out if c]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs"); ap.add_argument("--conditions"); ap.add_argument("--arms"); ap.add_argument("--seeds")
    ap.add_argument("--tolerance", type=float, default=1e-9,
                    help="relative difference allowed between a stored and a recomputed number (default 1e-9)")
    ap.add_argument("--dry-run", action="store_true", help="compare only; write nothing")
    ap.add_argument("--workers", type=int, default=3,
                    help="processes; each one holds the prepared test images, about 0.6 GB (default 3)")
    ap.add_argument("--report", default=None,
                    help="where to write the log (default results/consolidated/rescore_log.json)")
    add_common(ap)
    a = ap.parse_args(); cfg = config(a)
    assert_physics(cfg); verify_prepared(cfg)

    # a mistyped selection would otherwise produce a run for every name given and skip them all, one line each
    allowed = {"--archs": (parse_list(a.archs), list(cfg["models"])),
               "--conditions": (parse_list(a.conditions), conditions(cfg)),
               "--arms": (parse_list(a.arms), list(cfg["arms"])),
               "--seeds": (parse_list(a.seeds, int), list(cfg["training"]["seeds"]))}
    for flag, (asked, known) in allowed.items():
        unknown = [str(x) for x in (asked or []) if x not in known]
        if unknown:
            sys.exit(f"[rescore] {flag}: {', '.join(unknown)} is not in the configuration; "
                     f"it knows {', '.join(str(k) for k in known)}")

    stems = split_stems(cfg)["test"]
    sha = {r["stem"]: r.get("phase_sha256") for r in read_manifest(cfg["paths_abs"]["prepared"], "val")}
    runs = grid(cfg, parse_list(a.archs), parse_list(a.conditions), parse_list(a.arms), parse_list(a.seeds, int))
    tags = list(cfg["evaluation"]["checkpoints"])
    if not runs:
        sys.exit("[rescore] the selection --archs/--conditions/--arms/--seeds matches no run")

    size = int(cfg["models"][runs[0].arch]["image_size"])
    parts = chunks([r.to_dict() for r in runs], a.workers)
    print("==============================================================================", flush=True)
    print(f"[rescore] {len(runs)} runs x {len(tags)} checkpoints = {len(runs) * len(tags)} checkpoints", flush=True)
    print(f"[rescore] {len(stems)} test images, checkpoints {', '.join(tags)}", flush=True)
    print(f"[rescore] {'comparing only, nothing will be written' if a.dry_run else 'results rewritten in place'}; "
          f"agreement tolerance {a.tolerance:g}", flush=True)
    print(f"[rescore] {len(parts)} process(es), {', '.join(str(len(p)) for p in parts)} runs each", flush=True)
    print("==============================================================================\n", flush=True)

    t_start = time.time()
    jobs = [(cfg, p, tags, stems, sha, a.tolerance, a.dry_run, size, f"[{i + 1}/{len(parts)}]")
            for i, p in enumerate(parts)]
    if len(jobs) == 1:
        results = worker(jobs[0])
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = [rec for part in pool.map(worker, jobs) for rec in part]

    done = [r for r in results if r["outcome"] == "rescored"]
    skipped = [r for r in results if r["outcome"] == "skipped"]
    failed = [r for r in results if r["outcome"] == "disagreed"]
    minutes = (time.time() - t_start) / 60

    print(f"\n[rescore] finished in {minutes:.1f} min", flush=True)
    print(f"[rescore] {len(done)} checkpoints reproduced the stored results"
          f"{'' if a.dry_run else ' and were rewritten with the shape measurements'}", flush=True)
    print(f"[rescore] {len(skipped)} skipped, {len(failed)} disagreed", flush=True)
    if done:
        per = sum(r["seconds"] for r in done) / len(done)
        shaped = sum(r["cells_with_shape"] for r in done); cells = sum(r["reference_cells"] for r in done)
        print(f"[rescore] {per:.1f} s of scoring per checkpoint; "
              f"{shaped} of {cells} reference-cell records carry a perimeter and a circularity", flush=True)
    for s in skipped:
        print(f"[rescore]   skipped   {s['run']} {s['checkpoint']}: {s['reason']}", flush=True)
    for f in failed:
        print(f"[rescore]   DISAGREED {f['run']} {f['checkpoint']}: {f['reason']}", flush=True)

    log = Path(a.report) if a.report else Path(cfg["paths_abs"]["results"]) / "consolidated" / "rescore_log.json"
    write_json(log, {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "dry_run": bool(a.dry_run),
                     "tolerance": a.tolerance, "n_test_images": len(stems), "checkpoints": tags,
                     "minutes": minutes, "reason": RESCORE_REASON,
                     "rescored": done, "skipped": skipped, "disagreed": failed})
    print(f"[rescore] log: {log}", flush=True)
    if failed:
        print("[rescore] some stored results could not be reproduced; nothing was written for those checkpoints",
              flush=True)
        sys.exit(1)
