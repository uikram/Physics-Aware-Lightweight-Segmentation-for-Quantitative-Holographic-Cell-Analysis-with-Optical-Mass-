#!/usr/bin/env python
"""Verify the raw dataset and data/processed. Exit code 1 if any check fails.

    python scripts/verify_dataset.py [--raw-root data] [--processed data/processed]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qpi.config import load_config
from qpi.data.verify import verify_dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-overlay", default=None)
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--processed", default=None)
    ap.add_argument("--allow-partial", action="store_true", help="smoke tests on a data subset: report but do not fail")
    a = ap.parse_args()
    cfg = load_config(a.config_overlay)
    raw = Path(a.raw_root) if a.raw_root else cfg["paths_abs"]["raw_root"]
    proc = Path(a.processed) if a.processed else cfg["paths_abs"]["processed_root"]
    rep = verify_dataset(cfg, raw, proc)
    for name, c in rep["checks"].items():
        tag = "FAIL" if not c["ok"] else ("WARN" if c.get("warning") else "PASS")
        print(f"[{tag}] {name}: " + ", ".join(f"{k}={v}" for k, v in c.items() if k != "ok"))
    if not rep["ok"]:
        print("DATASET VERIFICATION FAILED" + (" (ignored: --allow-partial)" if a.allow_partial else ""))
        return 0 if a.allow_partial else 1
    print("dataset verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
