#!/usr/bin/env python
"""Build data/prepared from data/raw ONCE (CPU): registration of the 8-bit masks onto the 32-bit phase grid, the
annotation-free order-2 background subtracted once per image, reference cells (watershed h = 3), phase-wrapping flags,
manifest with a SHA-256 checksum of every prepared phase map.
    python scripts/utilities/prepare_data.py --workers 16
Refuses to overwrite an existing data/prepared."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qpicell.cli import add_common, config  # noqa: E402
from qpicell.data.prepare import prepare  # noqa: E402
from qpicell.physics import assert_physics  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None, help="only the first N images per split (testing)")
    add_common(ap)
    a = ap.parse_args(); cfg = config(a); assert_physics(cfg)
    prepare(cfg, a.workers, a.limit)
