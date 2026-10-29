#!/usr/bin/env python
"""Inference benchmark, protocol v5 (qpi/benchmark.py, D-26). No training.

    python scripts/run_benchmark.py [--config-overlay smoke.yaml] [--seeds 42 43 44] [--gpu 0]

FP32 only: EdgeSAM, MobileSAM and MobileNet-UNet, each in PyTorch and in ONNX Runtime (six combinations), trained reference
checkpoints, batch 1. One pass per seed, each in its own process, one after the other on the visible GPU ``--gpu``. Writes
<output_root>/benchmark/benchmark.json, benchmark.csv (mean and SD over seeds), benchmark_per_seed.csv,
benchmark_parameters_by_rank.csv and passes/pass_seed<seed>.json (raw per-run latencies); the previous benchmark outputs are replaced.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

# Known harmless library warnings (ONNX tracing of data-dependent shape checks, deprecated imports, a missing optional dependency)
# would otherwise fill the log; every real error still stops the run.
for m in (".*Converting a tensor to a Python boolean.*", ".*Using len to get tensor shape.*", ".*Constant folding.*",
          ".*Memory Efficient attention is a non-deterministic.*", ".*Unable to find acceptable character detection.*"):
    warnings.filterwarnings("ignore", message=m)
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")
if os.environ.get("BENCH_TF32", "0") != "1":
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"       # strict FP32 in cuBLAS/cuDNN for PyTorch and ONNX Runtime; must precede any CUDA library load
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qpi.benchmark import DEFAULT_SEEDS, benchmark_architectures, parameters_by_rank, run_pass, write_outputs  # noqa: E402
from qpi.config import load_config  # noqa: E402
from qpi.utils.env import enforce_gpu_policy, resolve_device  # noqa: E402
from qpi.utils.io import read_json, write_json  # noqa: E402
from qpi.utils.outputs import claim_output_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-overlay", default=None)
    ap.add_argument("--gpu", type=int, default=None, help="visible GPU index (default: benchmark.gpu)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--worker", type=int, default=None, help="internal: run one pass for this seed")
    ap.add_argument("--pass-index", type=int, default=0)
    ap.add_argument("--pass-out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config_overlay)
    out = claim_output_root(cfg) / "benchmark"
    gpu = a.gpu if a.gpu is not None else cfg["benchmark"].get("gpu", 0)
    enforce_gpu_policy(cfg["hardware"]["allowed_physical_gpus"])

    if a.worker is not None:
        dev = resolve_device(gpu)
        scratch = out / f"_onnx_tmp_{a.worker}"
        write_json(a.pass_out, run_pass(cfg, dev, a.worker, a.pass_index, scratch))
        shutil.rmtree(scratch, ignore_errors=True)
        return 0

    # ---- orchestrator (touches no CUDA context, so it cannot disturb the memory measurements) ----------------------------
    if out.exists():                                                   # previous benchmark outputs are replaced
        shutil.rmtree(out)
    passes_dir = out / "passes"
    passes_dir.mkdir(parents=True)
    for i, seed in enumerate(a.seeds):
        pj = passes_dir / f"pass_seed{seed}.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(seed), "--pass-index", str(i),
               "--pass-out", str(pj), "--gpu", str(gpu)]
        if a.config_overlay:
            cmd += ["--config-overlay", a.config_overlay]
        print(f"[benchmark] pass {i + 1}/{len(a.seeds)} (seed {seed}) on visible GPU {gpu}", flush=True)
        if subprocess.Popen(cmd, env=dict(os.environ)).wait() != 0:
            sys.exit(f"pass for seed {seed} failed")
    passes = [read_json(passes_dir / f"pass_seed{s}.json") for s in a.seeds]
    params = parameters_by_rank(cfg, benchmark_architectures(cfg))
    res = write_outputs(cfg, out, passes, params)
    print(f"[benchmark] wrote {out / 'benchmark.json'}: {len(res['rows'])} model/runtime rows, {len(res['per_seed'])} per-seed rows")
    from qpi.analysis.migrate import split_benchmark              # individual timing files (D-29); benchmark.json is unchanged
    from qpi.utils.layout import ROOT as _ROOT
    n_units = split_benchmark(_ROOT, force=True, src=out / "benchmark.json")
    print(f"[benchmark] wrote {n_units['written']} individual timing files to outputs/analysis/benchmark/ "
          "(compile with scripts/compile_results.py benchmark)")
    for r in res["rows"]:
        sd = lambda k: f"{r[k + '_sd']:.2f}" if r[k + "_sd"] is not None else "n/a"      # noqa: E731
        print(f"[benchmark] {r['architecture']:>15} {r['backend']:>8}  P50 {r['latency_p50_ms']:.2f} +- {sd('latency_p50_ms')} ms  "
              f"P99 {r['latency_p99_ms']:.2f} +- {sd('latency_p99_ms')} ms  FPS {r['fps']:.1f} +- {sd('fps')}  "
              f"Peak VRAM {r['peak_vram_mb']:.0f} +- {sd('peak_vram_mb')} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
