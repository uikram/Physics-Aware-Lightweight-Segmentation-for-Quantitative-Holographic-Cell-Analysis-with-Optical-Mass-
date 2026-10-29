#!/usr/bin/env python
"""Build data/processed from the raw float32 phase maps, 8-bit exports and masks.

    python scripts/prepare_dataset.py [--workers 8] [--raw-root data] [--out data/processed]

Every base image that has a float32 map is registered to its mask frame (ECC); images failing the quality
gate abort the build unless --allow-failures is given. Base images without a float map are listed in
missing_float_data.csv and are NOT used.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qpi.config import load_config
from qpi.data.prepare import prepare_dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-overlay", default=None)
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="at most N evenly spaced images per split (smoke tests)")
    ap.add_argument("--allow-failures", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config_overlay)
    raw = Path(a.raw_root) if a.raw_root else cfg["paths_abs"]["raw_root"]
    out = Path(a.out) if a.out else cfg["paths_abs"]["processed_root"]
    rep = prepare_dataset(cfg, raw, out, a.workers, a.limit, a.allow_failures)
    print(f"[prepare] ok={rep['n_processed_ok']} missing_float={rep['n_missing_float_data']} "
          f"failed={len(rep['n_registration_failed'])} valid_fraction={rep['valid_fraction']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
