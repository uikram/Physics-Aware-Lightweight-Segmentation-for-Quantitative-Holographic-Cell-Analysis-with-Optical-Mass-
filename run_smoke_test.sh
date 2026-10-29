#!/usr/bin/env bash
# Server smoke test: exercises every stage of the full pipeline on a tiny budget with SEPARATE settings and outputs.
#
#   ./run_smoke_test.sh
#
# * data:   at most SMOKE_IMAGES evenly spaced images per split are registered into data/processed_smoke
# * train:  1 epoch, 2 steps per epoch, all three architectures, LoRA r=8, full fine-tuning and loss ablations
# * output: outputs/smoke and manuscript/outputs_smoke ONLY (configs/smoke.yaml); the code refuses to mix modes
# Uses the same GPU policy as the full run (CUDA_VISIBLE_DEVICES=2,3). It does not read or write outputs/full.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/_pipeline_lib.sh
source scripts/_pipeline_lib.sh

SMOKE_IMAGES="${SMOKE_IMAGES:-6}"
SLOTS="${SLOTS:-2}"      # GPU slots (visible cuda:0 and cuda:1); both GPUs are exercised by default
# Developer dry run without GPUs (CI/laptop only): ALLOW_CPU_DRYRUN=1 skips the GPU check. RAW_ROOT overrides the raw data folder.
ALLOW_CPU_DRYRUN="${ALLOW_CPU_DRYRUN:-0}"
RAW_ARG=(); if [ -n "${RAW_ROOT:-}" ]; then RAW_ARG=(--raw-root "$RAW_ROOT"); fi
OV=(--config-overlay smoke.yaml)
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_HOME="$PWD/cache"
export PYTHONUNBUFFERED=1

OUT=outputs/smoke
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/pipeline.log") 2>&1
log "SMOKE TEST started; project $(pwd)"

step "python environment"  check_python_env
if [ "$ALLOW_CPU_DRYRUN" = "1" ]; then
  log "ALLOW_CPU_DRYRUN=1: GPU policy check skipped (CPU dry run, not a server smoke test)"
else
  step "GPU policy (GPUs 2,3 only)" require_gpu_policy
fi
step "pretrained weights"  check_weights
if python -c "import pytest" 2>/dev/null; then
  step "unit tests" python -m pytest -q tests
else
  log "pytest not installed: unit tests skipped (pip install pytest)"
fi

step "record run metadata (start)" python scripts/record_run.py "${OV[@]}" --stage start --note "run_smoke_test.sh"
step "build smoke dataset subset" python scripts/prepare_dataset.py "${OV[@]}" ${RAW_ARG[@]+"${RAW_ARG[@]}"} --workers 4 --limit "$SMOKE_IMAGES"
step "verify smoke dataset (partial)" python scripts/verify_dataset.py "${OV[@]}" ${RAW_ARG[@]+"${RAW_ARG[@]}"} --allow-partial
step "write experiment grid" python main.py write-grid "${OV[@]}"
step "train (smoke budget, ${SLOTS} GPU slots)" python scripts/run_queue.py "${OV[@]}" --slots "$SLOTS"
step "analysis" python scripts/run_analysis.py "${OV[@]}" --gpu 0
step "benchmark" python scripts/run_benchmark.py "${OV[@]}"
step "manuscript results" python manuscript/results.py "${OV[@]}"
step "manuscript figures" python manuscript/figures.py "${OV[@]}"
step "record run metadata (end)" python scripts/record_run.py "${OV[@]}" --stage end
step "verify outputs" python scripts/verify_outputs.py "${OV[@]}"

log "SMOKE TEST PASSED. Numbers produced here are NOT results; delete outputs/smoke, manuscript/outputs_smoke and data/processed_smoke when done."
pipeline_footer
