#!/usr/bin/env python
"""Train and evaluate ONE run.
    python scripts/training/train.py --arch edge_sam --condition r32 --arm dice_pmc_bga_cipp --seed 42 --gpu 0
    python scripts/training/train.py --arch mobile_sam --condition fullft --arm dice --seed 43 --gpu 1
condition: r8, r16, r32, r64 (LoRA rank) or fullft (full fine-tuning). GPU index = visible index. A finished run is
skipped unless --force. The queue (run_queue.py) calls this script for every run."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qpicell.cli import add_common, config  # noqa: E402
from qpicell.training.run import execute  # noqa: E402
from qpicell.training.runs import Run, conditions  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=["edge_sam", "mobile_sam", "mobilenet_unet"])
    ap.add_argument("--condition", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="batches per epoch (smoke tests only)")
    add_common(ap)
    a = ap.parse_args(); cfg = config(a)
    if a.arm not in cfg["arms"]:
        ap.error(f"unknown loss {a.arm}; known: {list(cfg['arms'])}")
    if a.condition not in conditions(cfg):
        ap.error(f"unknown condition {a.condition}; known: {conditions(cfg)}")
    execute(cfg, Run(a.arch, a.condition, a.arm, a.seed), a.gpu, a.force, a.max_steps if a.max_steps or not a.smoke else 2)
