#!/usr/bin/env python
"""Write copies of the multi-seed summary files with the manuscript class names (DECISIONS.md D-25).

    python scripts/relabel_summary.py [--summary outputs/multiseed/summary] [--outdir outputs/multiseed/summary_manuscript_names]

Reads the aggregator output (legacy names: mask value 2 = 'echinocyte', 3 = 'spherocyte') and swaps the two names in the
column headers, metric names and text. The originals are not modified. Numbers are copied unchanged.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qpi.utils.classnames import swap_obj, swap_text, swap_text_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="outputs/multiseed/summary")
    ap.add_argument("--outdir", default="outputs/multiseed/summary_manuscript_names")
    a = ap.parse_args()
    src, dst = ROOT / a.summary, ROOT / a.outdir
    if not src.is_dir():
        print(f"relabel_summary.py: {src} not found", file=sys.stderr)
        return 1
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("runs_index.csv", "config_summary.csv", "paired_differences.csv"):
        p = src / name
        if not p.exists():
            continue
        with open(p, newline="") as fh:
            rows = list(csv.reader(fh))
        out = [[swap_text(c) for c in r] for r in rows]
        # runs_index.csv: keep the hash and path columns untouched (they point at files that carry legacy names only in content)
        with open(dst / name, "w", newline="") as fh:
            csv.writer(fh).writerows(out)
    md = src / "tables.md"
    if md.exists():
        (dst / "tables.md").write_text(swap_text_report(md.read_text()))
    js = src / "multiseed_results.json"
    if js.exists():
        import json
        (dst / "multiseed_results.json").write_text(json.dumps(swap_obj(json.loads(js.read_text())), indent=1))
    (dst / "README.txt").write_text(
        "Copies of ../summary with the manuscript class names: mask value 2 = spherocyte, mask value 3 = echinocyte (D-25).\n"
        "The numbers are identical to the originals; only the names of the two classes are exchanged.\n")
    print(f"[relabel] wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
