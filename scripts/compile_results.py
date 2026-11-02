#!/usr/bin/env python
"""Build the multi-seed JSONs from the individual result files (DECISIONS.md D-29). No GPU, seconds.

    python scripts/compile_results.py [segmentation|cell|robustness|benchmark|all] [--expect-all-cells]

Writes outputs/compiled/<name>_multiseed.json (segmentation, cell_analysis, robustness, benchmark): per configuration the per-seed
values, mean, sample SD, min, max, seeds expected / found / missing, the list of input files with sha256, and ``problems``.
A missing or failed seed is listed, never counted as zero. Exit status 1 (and no COMPLETE) if any file has problems.
Individual files are produced by scripts/run_cell_analysis.py, scripts/run_robustness.py, scripts/run_benchmark.py, and
scripts/organise_results.py (conversion of the older files). Models, ranks and seeds are never selected here: see configs/views.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpi.analysis import compile as cp  # noqa: E402
from qpi.utils.progress import Progress  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["all", *cp.COMPILERS])
    ap.add_argument("--expect-all-cells", action="store_true", help="cell analysis: require the whole grid (4 losses, 5 ranks, 3 seeds)")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    kinds = list(cp.COMPILERS) if a.what == "all" else [a.what]
    prog = Progress("compile_results", len(kinds))
    prog.stage("compile")
    ok_all = True
    for k in kinds:
        try:
            kw = {"expect": "all"} if (k == "cell" and a.expect_all_cells) else {}
            res = cp.write_compiled(k, Path(a.root), **kw)
        except FileNotFoundError as exc:
            prog.fail(k, exc)
            continue
        except Exception as exc:                                  # noqa: BLE001
            prog.fail(k, exc)
            continue
        doc = res["doc"]
        n = doc.get("n_runs", len(doc.get("rows", [])))
        note = f"{n} runs/rows, {len(doc['problems'])} problem(s) -> {res['path']}"
        if res["ok"] and doc["complete"]:
            prog.done(k, note=note)
        else:
            prog.fail(k, note + "; " + "; ".join(res["errors"] + doc["problems"][:5]))
            ok_all = False
    return prog.finish(extra_ok=ok_all)


if __name__ == "__main__":
    sys.exit(main())
