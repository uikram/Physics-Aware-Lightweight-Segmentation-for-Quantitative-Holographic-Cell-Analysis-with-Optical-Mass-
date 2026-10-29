#!/usr/bin/env python
"""Is every result file newer than the files it was computed from? (read-only; no training, no GPU)

    python scripts/check_freshness.py [--warn-only]

Compares the modification times of each analysis, benchmark, summary and manuscript JSON/TeX file with its inputs and says, for
every file that is missing or older than an input, which command regenerates it. The refresh commands are printed in the order in
which they have to be run. Exit status 1 if anything is missing or stale (0 with --warn-only).

Limits: modification times are only comparable on the machine that produced the files (copying files, for example with scp,
changes them); run this where the results were made. A file that was regenerated from unchanged inputs is simply newer, which is
fine. ``outputs/cell_analysis/cell_analysis.json`` is checked more precisely (size and time of every predictions file it used,
and new finished runs it does not contain).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpi.analysis import cell_analysis as ca  # noqa: E402
from qpi.config import load_config  # noqa: E402
from qpi.utils.io import read_json  # noqa: E402
from qpi.utils.outputs import mode_of  # noqa: E402

TOL = 1.0                       # seconds


def _newest(files: Sequence[Path]):
    files = [f for f in files if f.exists()]
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warn-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config(None)
    if mode_of(cfg) != "full":
        print("check_freshness.py: full configuration only", file=sys.stderr)
        return 1
    main_root = Path(cfg["paths_abs"]["output_root"])
    man = Path(cfg["paths_abs"]["manuscript_outputs"])
    ms_root = ROOT / "outputs" / "multiseed"
    proc = Path(cfg["paths_abs"]["processed_root"])
    tag = cfg["evaluation"]["checkpoint"]
    sel_arch, sel_rank = cfg["analysis"]["selected_arch"], int(cfg["analysis"]["selected_rank"])
    bench_rank = int(cfg["benchmark"].get("reference_rank") or sel_rank)
    runs = main_root / "runs"
    manifest = proc / "manifest.csv"
    pred = sorted(runs.glob(f"*/predictions_{tag}.npz"))
    metr = sorted(runs.glob("*/metrics.json"))
    ckpt = sorted(runs.glob("*/checkpoints/last.pt"))
    ms_metr = sorted((ms_root / "runs").glob("*/metrics.json"))
    archs = list(cfg["architectures"])

    # (file, inputs, command that regenerates it); listed in the order the commands must be run
    items = [
        (main_root / "analysis" / "phase_integral.json", pred + metr + [manifest], "python scripts/run_analysis.py --skip-robustness"),
        (main_root / "analysis" / "drymass.json", [runs / f"{sel_arch}_lora_r{sel_rank}" / f"predictions_{tag}.npz", manifest],
         "python scripts/run_analysis.py --skip-robustness"),
        (main_root / "analysis" / "robustness.json", ckpt, "python scripts/run_analysis.py   (needs a GPU)"),
        (main_root / "analysis" / "architecture_comparison.json",
         [runs / f"{x}_lora_r{sel_rank}" / f"predictions_{tag}.npz" for x in archs], "python scripts/compare_architectures.py"),
        (ms_root / "summary" / "multiseed_results.json", metr + ms_metr, "python scripts/aggregate_multiseed.py"),
        (ms_root / "summary_manuscript_names" / "multiseed_results.json", [ms_root / "summary" / "multiseed_results.json"],
         "python scripts/relabel_summary.py"),
        (main_root / "benchmark" / "benchmark.json", [runs / f"{x}_lora_r{bench_rank}" / "checkpoints" / "last.pt" for x in archs],
         "./ops/run_benchmark.sh   (re-times the models on a GPU)"),
        (main_root / "benchmark" / "equivalence.json", [main_root / "benchmark" / "benchmark.json"], "./ops/run_benchmark.sh"),
    ]
    cj = ROOT / "outputs" / "cell_analysis" / "cell_analysis.json"
    items.append((cj, pred + metr + ms_metr, "./ops/run_cell_analysis.sh"))
    deps = [i[0] for i in items]
    items.append((man / "results.json", deps, "./ops/make_manuscript_outputs.sh"))
    items.append((man / "multiseed_results.tex", [man / "results.json"], "./ops/make_manuscript_outputs.sh"))

    rows: List[tuple] = []
    refresh: List[str] = []
    for path, inputs, cmd in items:
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        if not path.exists():
            rows.append((str(rel), "MISSING", cmd))
            refresh.append(cmd)
            continue
        problems: List[str] = []
        newest = _newest(inputs)
        if newest is not None and newest.stat().st_mtime > path.stat().st_mtime + TOL:
            problems.append(f"older than {newest.relative_to(ROOT) if newest.is_relative_to(ROOT) else newest}")
        if path == cj:
            problems += ca.freshness_problems(read_json(path), main_root, ms_root, cfg)[:5]
        if problems:
            rows.append((str(rel), "STALE", "; ".join(problems) + f"  ->  {cmd}"))
            refresh.append(cmd)
        else:
            rows.append((str(rel), "ok", ""))
    # added layout (D-29): compiled files are compared with their inputs by sha256 (only where they exist)
    from qpi.analysis import compile as cp  # noqa: E402
    from qpi.utils.layout import compiled_path  # noqa: E402
    for kind in cp.COMPILERS:
        cpath = compiled_path(cp.NAMES[kind], ROOT)
        if not cpath.exists():
            continue
        probs = cp.stale_problems(kind, ROOT)
        cmd = "python scripts/organise_results.py"
        if probs:
            rows.append((str(cpath.relative_to(ROOT)), "STALE", "; ".join(probs[:4]) + f"  ->  {cmd}"))
            refresh.append(cmd)
        else:
            rows.append((str(cpath.relative_to(ROOT)), "ok", ""))
    w = max(len(r[0]) for r in rows)
    for name, status, detail in rows:
        print(f"{status:<8}{name:<{w}}  {detail}")
    order = list(dict.fromkeys(refresh))
    if order:
        print("\nregenerate, in this order:")
        for c in order:
            print(f"  {c}")
    else:
        print("\nevery file is newer than its inputs")
    return 0 if (not order or a.warn_only) else 1


if __name__ == "__main__":
    sys.exit(main())
