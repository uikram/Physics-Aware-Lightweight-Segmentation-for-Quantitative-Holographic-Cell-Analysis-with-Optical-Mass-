#!/usr/bin/env bash
# FP32 inference benchmark (no training): PyTorch and ONNX Runtime for EdgeSAM, MobileSAM and MobileNet-UNet (six combinations),
# trained checkpoints, batch 1. Then every file with benchmark numbers is regenerated.
#
#   GPU=1 SEEDS="42 43 44" ./ops/run_benchmark.sh      # physical GPU 1 (default), seeds one after the other
#   TF32=1 ./ops/run_benchmark.sh                      # leave TF32 on in PyTorch and ONNX Runtime (default: strict FP32 in both)
#
# 1. scripts/run_benchmark.py replaces <output_root>/benchmark/ (benchmark.json, benchmark.csv, benchmark_per_seed.csv,
#    benchmark_parameters_by_rank.csv, passes/). Metrics per model and runtime: P50, P99, FPS (= 1000 / P50), Peak VRAM.
# 2. scripts/check_equivalence.py: ONNX FP32 against PyTorch FP32 (logits, Dice, per-image dry mass on the validation images,
#    fixed acceptance gate) -> benchmark/equivalence.json. Fails the run when a model does not pass.
# 3. ops/make_manuscript_outputs.sh regenerates results.json, the figures and the tex report and runs the verification, which
#    fails when results.json and benchmark.json disagree or when a combination or a metric is missing.
# Using a GPU outside the default 2,3 lifts the GPU policy for this launch (D-23); the GPU is recorded in the outputs.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME:-segmentation}"
fi
# shellcheck source=scripts/_pipeline_lib.sh
source scripts/_pipeline_lib.sh
mkdir -p manuscript/outputs/logs
exec > >(tee -a manuscript/outputs/logs/run_benchmark.log) 2>&1
GPU="${GPU:-1}"
SEEDS="${SEEDS:-42 43 44}"
export CUDA_VISIBLE_DEVICES="$GPU"
export QPI_ALLOW_ALL_GPUS=1
export ORT_LOGGING_LEVEL=3
# FP32 means FP32 in both runtimes (TF32 off) unless TF32=1 is given; see apply_tf32_policy in qpi/benchmark.py
if [ "${TF32:-0}" = 1 ]; then unset NVIDIA_TF32_OVERRIDE; export BENCH_TF32=1; else export BENCH_TF32=0 NVIDIA_TF32_OVERRIDE=0; fi
require_gpu_policy
# shellcheck disable=SC2086
step "benchmark (protocol v5, FP32, seeds $SEEDS one after the other on GPU $GPU)" python scripts/run_benchmark.py --seeds $SEEDS --gpu 0
step "ONNX FP32 against PyTorch FP32 (logits, Dice, dry mass, gate)" python scripts/check_equivalence.py --gpu 0
step "regenerate results.json, figures, tex and verify" ./ops/make_manuscript_outputs.sh
pipeline_footer
