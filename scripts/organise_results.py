#!/usr/bin/env python
"""ONE command that puts the existing results into the layout of DECISIONS.md D-29. Safe to run again and again; no GPU; seconds.

    python scripts/organise_results.py [--dry-run] [--force] [--root PATH]

What it does (nothing is deleted, overwritten, retrained or recomputed; the older files stay and the older scripts keep working):
  1. split the older aggregate files into individual files, one per run (or timing pass):
       outputs/cell_analysis/cell_analysis.json  -> outputs/analysis/cell/<run id>.json
       outputs/full/analysis/robustness.json     -> outputs/analysis/robustness/<run id>.json
       outputs/full/benchmark/benchmark.json     -> outputs/analysis/benchmark/<arch>__<backend>__s<seed>.json
  2. compile the multi-seed JSONs into outputs/compiled/ (segmentation, cell analysis, robustness, benchmark);
  3. check every compiled number against the older file it replaces (mean, n seeds, max Peak VRAM, ...).
Every step prints a line; COMPLETE is the last line only if nothing failed and every check passed. A step whose source file is not on
this machine is reported as skipped (not as success of that step).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from qpi.analysis import compile as cp  # noqa: E402
from qpi.analysis import migrate as mg  # noqa: E402
from qpi.utils.io import read_json  # noqa: E402
from qpi.utils.layout import KINDS, compiled_path, unit_files  # noqa: E402
from qpi.utils.progress import Progress  # noqa: E402


def close(a, b, rel=1e-9, ab=1e-12) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float) and (math.isnan(a) or math.isnan(b)):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= max(ab, rel * max(abs(a), abs(b)))


def check_cell(root: Path):
    src = root / mg.CELL_JSON
    if not src.exists():
        return None
    legacy, comp = read_json(src)["by_configuration"], read_json(compiled_path("cell_analysis", root))["configurations"]
    bad, n = [], 0
    for cfg, v in legacy.items():
        c = comp.get(cfg)
        if c is None:
            bad.append(f"{cfg}: missing in the compiled file")
            continue
        if c["n_seeds"] != v["n_seeds"] or sorted(c["seeds_found"]) != sorted(v["seeds"]):
            bad.append(f"{cfg}: seeds differ")
        for m, d in v["metrics"].items():
            n += 1
            for q in ("mean", "sd", "min", "max"):
                if not close(d[q], c["metrics"][m][q]):
                    bad.append(f"{cfg} {m} {q}: {d[q]} vs {c['metrics'][m][q]}")
    return n, bad


def check_segmentation(root: Path):
    src = root / "outputs" / "multiseed" / "summary" / "config_summary.csv"
    if not src.exists():
        return None
    legacy = pd.read_csv(src)
    comp = read_json(compiled_path("segmentation", root))["configurations"]
    bad, n = [], 0
    for _, r in legacy.iterrows():
        c = comp.get(r["config"])
        if c is None:
            bad.append(f"{r['config']}: missing in the compiled file")
            continue
        if int(r["n"]) != c["n_seeds"] or int(r["n_runs_with_a_collapsed_class"]) != c["n_runs_with_a_collapsed_class"]:
            bad.append(f"{r['config']}: n seeds / collapsed runs differ")
        for m, d in c["metrics"].items():
            for q in ("mean", "sd", "min", "max"):
                n += 1
                v = r.get(f"{m}_{q}")
                v = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
                if not close(v, d[q], rel=1e-9):
                    bad.append(f"{r['config']} {m} {q}: {v} vs {d[q]}")
    return n, bad


def check_robustness(root: Path):
    src = root / mg.ROBUSTNESS_JSON
    if not src.exists():
        return None
    legacy = read_json(src)["rows"]
    comp = read_json(compiled_path("robustness", root))["configurations"]
    bad = []
    for r in legacy:
        info = cp.parse_run_id(r["run_id"])
        c = comp.get(info["config"])
        if c is None or r["distortion"] not in c["distortions"]:
            bad.append(f"{r['run_id']} {r['distortion']}: missing in the compiled file")
            continue
        d = c["distortions"][r["distortion"]]
        for f in ("macro_dice", "fg_iou", "ipe_bgcorr_true_pct", "ipe_uncorrected_seen_pct"):
            if not close(r[f], d[f]["per_seed"].get(str(info["seed"])), rel=1e-9):
                bad.append(f"{r['run_id']} {r['distortion']} {f}: {r[f]} vs {d[f]['per_seed'].get(str(info['seed']))}")
    return len(legacy) * 4, bad


def check_benchmark(root: Path):
    src = root / mg.BENCHMARK_JSON
    if not src.exists():
        return None
    legacy = read_json(src)["rows"]
    comp = {(r["architecture"], r["backend"]): r for r in read_json(compiled_path("benchmark", root))["rows"]}
    bad, n = [], 0
    for r in legacy:
        c = comp.get((r["architecture"], r["backend"]))
        if c is None:
            bad.append(f"{r['architecture']} {r['backend']}: missing in the compiled file")
            continue
        for f in ("latency_p50_ms", "latency_p99_ms", "fps", "peak_vram_mb"):
            n += 2
            if not close(r[f], c[f]["mean"]):
                bad.append(f"{r['architecture']} {r['backend']} {f} mean: {r[f]} vs {c[f]['mean']}")
            if not close(r[f + "_sd"], c[f]["sd"]):
                bad.append(f"{r['architecture']} {r['backend']} {f} sd: {r[f + '_sd']} vs {c[f]['sd']}")
        n += 1
        if not close(max(r["peak_vram_mb_per_seed"]), c["peak_vram_mb_max"]):
            bad.append(f"{r['architecture']} {r['backend']} peak VRAM maximum differs")
    return n, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="only report what exists and what would be written")
    ap.add_argument("--force", action="store_true", help="rewrite individual files that already exist")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    root = Path(a.root)

    legacy = {"cell analysis": mg.CELL_JSON, "robustness": mg.ROBUSTNESS_JSON, "benchmark": mg.BENCHMARK_JSON,
              "segmentation index": Path("outputs/multiseed/summary/runs_index.csv")}
    print(f"[organise_results] project root {root}")
    for name, p in legacy.items():
        print(f"[organise_results]   {name:<20} {p}: {'found' if (root / p).exists() else 'NOT FOUND'}")
    if a.dry_run:
        print("[organise_results] dry run: nothing written")
        print("COMPLETE")
        return 0

    steps = [("split cell analysis", lambda: mg.split_cell(root, a.force)),
             ("split robustness", lambda: mg.split_robustness(root, a.force)),
             ("split benchmark", lambda: mg.split_benchmark(root, a.force))]
    prog = Progress("organise_results", len(steps) + len(cp.COMPILERS) + 4)
    prog.stage("1/3 split the older files into individual files")
    for name, fn in steps:
        try:
            r = fn()
        except Exception as exc:                                  # noqa: BLE001
            prog.fail(name, exc)
            continue
        if r.get("note"):
            prog.skipped(name, r["note"])
        else:
            prog.done(name, note=f"{r['written']} written, {r['kept']} already there, {r['skipped']} not complete (not converted)")

    prog.stage("2/3 compile the multi-seed JSONs")
    compiled_ok = {}
    incomplete = []
    for kind in cp.COMPILERS:
        if kind in KINDS and not unit_files(kind, root):
            prog.skipped(f"compile {kind}", "no individual results yet")
            continue
        try:
            res = cp.write_compiled(kind, root)
        except FileNotFoundError as exc:
            prog.skipped(f"compile {kind}", f"no source: {Path(str(exc)).name}")
            continue
        except Exception as exc:                                  # noqa: BLE001
            prog.fail(f"compile {kind}", exc)
            continue
        doc = res["doc"]
        n = doc.get("n_runs", len(doc.get("rows", [])))
        compiled_ok[kind] = True
        if res["errors"]:
            prog.fail(f"compile {kind}", "; ".join(res["errors"]))
        elif doc["complete"]:
            prog.done(f"compile {kind}", note=f"{n} runs/rows -> {res['path'].relative_to(root)}")
        else:                                                    # the files were organised; the data set itself is not complete
            incomplete.append((kind, doc["problems"]))
            prog.done(f"compile {kind}", note=f"{n} runs/rows -> {res['path'].relative_to(root)}; INCOMPLETE DATA, "
                                              f"{len(doc['problems'])} problem(s), see 'problems' in the file")

    prog.stage("3/3 check the compiled numbers against the older files")
    for kind, fn in (("cell", check_cell), ("segmentation", check_segmentation), ("robustness", check_robustness),
                     ("benchmark", check_benchmark)):
        if kind not in compiled_ok:
            prog.skipped(f"check {kind}", "not compiled")
            continue
        try:
            r = fn(root)
        except Exception as exc:                                  # noqa: BLE001
            prog.fail(f"check {kind}", exc)
            continue
        if r is None:
            prog.skipped(f"check {kind}", "no older file to compare with")
        elif r[1]:
            prog.fail(f"check {kind}", f"{len(r[1])} of {r[0]} values differ, e.g. {r[1][0]}")
        else:
            prog.done(f"check {kind}", note=f"{r[0]} values identical to the older file")
    for kind, probs in incomplete:
        print(f"[organise_results] NOTE {kind}: the results themselves are incomplete ({len(probs)} entries), e.g. {probs[0]}", flush=True)
    if any(k == "robustness" for k, _ in incomplete):
        print("[organise_results] NOTE robustness exists for seed 42 only; for 3 seeds run: python scripts/run_robustness.py --scope all", flush=True)
    if any(k == "cell" for k, _ in incomplete):
        print("[organise_results] NOTE cell analysis is incomplete; run: python scripts/run_cell_analysis.py --all --jobs 4", flush=True)
    return prog.finish()


if __name__ == "__main__":
    sys.exit(main())
