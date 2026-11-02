#!/usr/bin/env bash
# Full experiment: dataset build + verification -> training of all runs -> analysis -> benchmark ->
# manuscript results -> manuscript figures -> final output verification.
#
#   CUDA_VISIBLE_DEVICES=2,3 ./run_full_experiment.sh        (the variable is set by the script; any other value is refused)
#
# Resumable: finished runs (metrics.json with status complete/diverged) are skipped on a re-run.
# Everything is written to outputs/full and manuscript/outputs; smoke-test outputs are never touched.
# Environment knobs: PREP_WORKERS (default 8), SLOTS (GPU slots, default 2 = visible cuda:0 and cuda:1).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/_pipeline_lib.sh
source scripts/_pipeline_lib.sh

PREP_WORKERS="${PREP_WORKERS:-8}"
SLOTS="${SLOTS:-2}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_HOME="$PWD/cache"
export PYTHONUNBUFFERED=1

OUT=outputs/full
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/pipeline.log") 2>&1
log "full experiment started; project $(pwd)"

step "python environment"  check_python_env
step "GPU policy" require_gpu_policy
step "pretrained weights"  check_weights

if [ ! -s data/processed/manifest.csv ]; then
  step "build processed dataset" python scripts/prepare_dataset.py --workers "$PREP_WORKERS"
else
  log "data/processed/manifest.csv exists: not rebuilding (delete data/processed to force)"
fi
step "verify dataset" python scripts/verify_dataset.py

step "record run metadata (start)" python scripts/record_run.py --stage start --note "run_full_experiment.sh"

step "write experiment grid" python main.py write-grid
step "train all runs (${SLOTS} GPU slots)" python scripts/run_queue.py --slots "$SLOTS"
step "analysis (phase integral, dry mass, robustness)" python scripts/run_analysis.py --gpu 0
step "inference benchmark (visible cuda:0, nothing else running)" python scripts/run_benchmark.py
step "manuscript results (results.json)" python manuscript/results.py
step "manuscript figures" python manuscript/figures.py
step "record run metadata (end)" python scripts/record_run.py --stage end
step "verify outputs" python scripts/verify_outputs.py

pipeline_footer
