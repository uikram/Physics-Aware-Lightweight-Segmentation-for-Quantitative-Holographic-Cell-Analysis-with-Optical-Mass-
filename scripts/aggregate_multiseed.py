#!/usr/bin/env python
"""Aggregate the multi-seed runs (and the seed-42 runs of outputs/full) into tables with a provenance index.

    python scripts/aggregate_multiseed.py [--max-stage N] [--allow-partial | --check-prereqs]

Reads configs/multiseed.yaml. Writes outputs/multiseed/summary/{runs_index.csv, config_summary.csv,
paired_differences.csv, tables.md, multiseed_results.json} and manuscript/outputs_multiseed/multiseed_results.json.
Exit status 1 if runs are missing or unfinished (unless --allow-partial; tables are written either way).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qpi.analysis.multiseed import check_prereqs, write_outputs  # noqa: E402
from qpi.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-overlay", default="multiseed.yaml")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--max-stage", type=int, default=None, help="only the blocks of stages 1..N (configs/multiseed.yaml)")
    ap.add_argument("--check-prereqs", action="store_true",
                    help="only check that the seed-42 runs, dataset and base configuration match; write nothing")
    a = ap.parse_args()
    cfg = load_config(a.config_overlay)
    if not cfg.get("multiseed"):
        ap.error("the overlay has no 'multiseed' section")
    if a.check_prereqs:
        problems = check_prereqs(cfg, ROOT, a.max_stage)
        for p in problems:
            print(f"[prereq] PROBLEM: {p}", file=sys.stderr)
        print("[prereq] seed-42 runs, dataset and base configuration are consistent" if not problems else "[prereq] FAILED")
        return 1 if problems else 0
    return write_outputs(cfg, ROOT, allow_partial=a.allow_partial, max_stage=a.max_stage)


if __name__ == "__main__":
    sys.exit(main())
