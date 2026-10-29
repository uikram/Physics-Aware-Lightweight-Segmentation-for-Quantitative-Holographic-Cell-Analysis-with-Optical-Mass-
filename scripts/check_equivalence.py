#!/usr/bin/env python
"""Correctness and quality gate of the benchmark (no training, nothing timed): ONNX Runtime FP32 against PyTorch FP32 with the trained
checkpoints. See qpi/equivalence.py.

    python scripts/check_equivalence.py [--gpu 0]

Writes <output_root>/benchmark/equivalence.json (read by manuscript/results.py). Exit status 1 when a model fails the gate.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path

for m in (".*Converting a tensor to a Python boolean.*", ".*Using len to get tensor shape.*", ".*Constant folding.*",
          ".*Memory Efficient attention.*", ".*character detection.*"):
    warnings.filterwarnings("ignore", message=m)
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")
if os.environ.get("BENCH_TF32", "0") != "1":
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"       # strict FP32 in cuBLAS/cuDNN for PyTorch and ONNX Runtime; must precede any CUDA library load
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qpi.config import load_config  # noqa: E402
from qpi.equivalence import run_equivalence  # noqa: E402
from qpi.utils.env import enforce_gpu_policy, resolve_device  # noqa: E402
from qpi.utils.outputs import claim_output_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-overlay", default=None)
    ap.add_argument("--gpu", type=int, default=0, help="visible GPU index")
    a = ap.parse_args()
    cfg = load_config(a.config_overlay)
    enforce_gpu_policy(cfg["hardware"]["allowed_physical_gpus"])
    dev = resolve_device(a.gpu)
    out = claim_output_root(cfg) / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "_equivalence_tmp"
    res = run_equivalence(cfg, dev, out / "equivalence.json", scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    for r in res["architectures"]:
        lc = r["logit_comparison"]
        print(f"[equivalence] {r['architecture']:>15} logits ({r['n_inputs']} maps): agree {lc['argmax_agreement_pct']:.4f}% "
              f"max|dlogit| {lc['max_abs_logit_diff']:.3g}")
        for c in r["quality"]["candidates"]:
            if c["candidate"] == "pytorch_fp32":
                continue
            print(f"[equivalence] {r['architecture']:>15} {c['candidate']} on {c['n_images']} images: agree {c['argmax_agreement_pct']:.4f}% "
                  f"dDice {c['mean_dice_change']:+.5f} dry-mass err mean {c['dry_mass_mean_abs_rel_error']:.3g} "
                  f"p99 {c['dry_mass_p99_rel_error']:.3g} -> {c['verdict']}")
    print(f"[equivalence] gate (agreement >= {res['acceptance']['argmax_agreement_pct_min']} %, |dDice| <= "
          f"{res['acceptance']['mean_dice_abs_change_max']}, dry mass mean <= {res['acceptance']['dry_mass_mean_abs_rel_error_max']}, "
          f"p99 <= {res['acceptance']['dry_mass_p99_rel_error_max']}): {'all accepted' if res['all_accepted'] else 'FAILED'}")
    return 0 if res["all_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
