#!/usr/bin/env python
"""Refuse to run on the old, half-scale date-named phase maps (DECISIONS.md D-20).

The first delivery of the date-named maps (e.g. 20110617_4.tif) was confined to [0, pi]; the corrected delivery is exactly
2x that, confined to [0, 2*pi]. This script samples date-named maps in <raw_root>/raw_data and fails if they still look
like the old ones (maximum <= pi + 0.1 in every sampled file).

    python scripts/check_raw_scale.py [--raw-root data] [--n 25]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()
    if a.raw_root:
        raw = Path(a.raw_root)
    else:
        from qpi.config import load_config
        raw = Path(load_config()["paths_abs"]["raw_root"])
    files = sorted(p for p in (raw / "raw_data").glob("*.tif") if re.match(r"^\d{8}_\d+$", p.stem))
    if not files:
        print(f"[check_raw_scale] no date-named maps in {raw / 'raw_data'}", file=sys.stderr)
        return 2
    step = max(1, len(files) // a.n)
    sample = files[::step][: a.n]
    maxima = np.array([float(tifffile.imread(p).max()) for p in sample])
    print(f"[check_raw_scale] {len(files)} date-named maps in {raw / 'raw_data'}; sampled {len(sample)}: "
          f"max of maxima {maxima.max():.3f}, median of maxima {np.median(maxima):.3f}, ceiling 2*pi = {2 * math.pi:.3f}")
    if maxima.max() <= math.pi + 0.1:
        print("[check_raw_scale] FATAL: every sampled map is confined to [0, pi]. These are the OLD half-scale maps. "
              "Put the corrected maps (former data/raw_data_2) at data/raw_data (see ops/legacy/move_old_run.sh).", file=sys.stderr)
        return 1
    print("[check_raw_scale] OK: date-named maps reach beyond pi (corrected scale)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
