#!/usr/bin/env bash
# Multi-seed extension of the full experiment (see EXPERIMENTS.md, configs/multiseed.yaml).
#
#   CUDA_VISIBLE_DEVICES=2,3 STAGE=1 ./run_multiseed.sh      # stage 1: EdgeSAM, 52 runs
#   CUDA_VISIBLE_DEVICES=2,3 STAGE=2 ./run_multiseed.sh      # stages 1+2: + MobileSAM, 52 runs (finished runs are skipped)
#   CUDA_VISIBLE_DEVICES=2,3 STAGE=3 ./run_multiseed.sh      # stages 1+2+3: + MobileNet-UNet, 52 runs
# (./ops/start_multiseed.sh N does the same in the background.)
#
# STAGE N runs the blocks of stages 1..N and aggregates them; it is required, so that nothing large starts by accident.
# Needs a finished ./run_full_experiment.sh (seed 42: outputs/full, data/processed). Writes ONLY to outputs/multiseed and
# manuscript/outputs_multiseed. Resumable: finished runs (metrics.json complete/diverged) are skipped on a re-run.
# Environment knobs: SLOTS (parallel runs; default 2, or one per GPU when QPI_ALLOW_ALL_GPUS=1). GPU time: roughly 35 GPU-hours for all 156 runs.
# More than GPUs 2,3: use ./ops/start_multiseed.sh N (GPUS=free|all|0,1,...), which sets QPI_ALLOW_ALL_GPUS=1 (D-23).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/_pipeline_lib.sh
source scripts/_pipeline_lib.sh

STAGE="${STAGE:-}"
case "$STAGE" in 1|2|3) ;; *) die "set STAGE=1, 2 or 3 (see the header of this script)";; esac
if [ "${QPI_ALLOW_ALL_GPUS:-0}" = 1 ]; then
  SLOTS="${SLOTS:-$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:-}")}"      # one run per listed GPU
else
  SLOTS="${SLOTS:-2}"
fi
OVERLAY=multiseed.yaml
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_HOME="$PWD/cache"
export PYTHONUNBUFFERED=1

OUT=outputs/multiseed
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/pipeline.log") 2>&1
log "multi-seed experiment started (stages 1 to ${STAGE}); project $(pwd)"

step "python environment"  check_python_env
step "GPU policy" require_gpu_policy
step "pretrained weights"  check_weights
step "seed-42 runs, dataset and configuration match" python scripts/aggregate_multiseed.py --check-prereqs --max-stage "$STAGE"
step "mirror the seed-42 runs next to the new seeds (hard links)" python scripts/link_seed42.py --config-overlay "$OVERLAY"
step "record run metadata (start)" python scripts/record_run.py --stage start --config-overlay "$OVERLAY" --note "run_multiseed.sh STAGE=$STAGE GPUS=${CUDA_VISIBLE_DEVICES:-} SLOTS=$SLOTS"
step "write experiment grid" python main.py write-grid --config-overlay "$OVERLAY"
step "train the multi-seed runs of stages 1-${STAGE} (${SLOTS} GPU slots)" python scripts/run_queue.py --slots "$SLOTS" --config-overlay "$OVERLAY" --max-stage "$STAGE"
step "aggregate (tables, run index)" python scripts/aggregate_multiseed.py --max-stage "$STAGE"
step "record run metadata (end)" python scripts/record_run.py --stage end --config-overlay "$OVERLAY"

pipeline_footer
