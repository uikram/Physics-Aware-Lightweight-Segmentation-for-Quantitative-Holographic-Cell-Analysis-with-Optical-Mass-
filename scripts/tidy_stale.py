#!/usr/bin/env python
"""Move files that are stale since the new result layout (D-29) into _to_delete/, keeping their folder structure.

    python scripts/tidy_stale.py --dry-run      # list what would be moved
    python scripts/tidy_stale.py                # move it

Nothing is deleted: delete _to_delete/ yourself when you do not need it. Missing paths are skipped, so the same command is
safe on the PC and on the server. Used by ops/tidy.sh. Works on Windows, Linux and macOS.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STALE = [
    "scripts/run_benchmark",                       # old protocol-v2 copy (the current script is run_benchmark.py)
    "tests/DECISIONS.md",                          # old duplicate of DECISIONS.md
    "scripts/compare_with_old_run.py",             # half-scale first run (D-20) and its tools
    "tests/test_old_run_tools.py",
    "outputs/full/analysis/comparison_with_old_run.csv",
    "outputs/full/analysis/comparison_with_old_run.md",
    "ops/legacy",                                  # finished one-time launchers
    "old_run",                                     # the half-scale first delivery
    "bench_run.out", "study_run.out",              # logs of the old benchmark / precision study
    "run_benchmark.py",                            # stray copy in the folder root
    "multiseed_results.pdf",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="only list what would be moved")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    root = Path(a.root)
    present = [p for p in STALE if (root / p).exists()]
    if not present:
        print("[tidy_stale] nothing stale found")
        print("COMPLETE")
        return 0
    failed = 0
    for p in present:
        src, dst = root / p, root / "_to_delete" / p
        if dst.exists():
            dst = dst.with_name(dst.name + time.strftime(".%H%M%S"))
        print(f"[tidy_stale] {'would move' if a.dry_run else 'move'} {p} -> {dst.relative_to(root).as_posix()}")
        if a.dry_run:
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except OSError as exc:
            failed += 1
            print(f"[tidy_stale] FAILED {p}: {exc}", file=sys.stderr)
    if failed:
        print(f"FAILED: {failed} of {len(present)} could not be moved", file=sys.stderr)
        return 1
    print(f"[tidy_stale] {len(present)} item(s) {'listed (dry run, nothing moved)' if a.dry_run else 'moved to _to_delete/'}")
    print("COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
