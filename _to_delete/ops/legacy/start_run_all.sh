#!/usr/bin/env bash
# Start run_all.sh in the background on GPUs 2,3 (activates the conda env 'segmentation' itself). Refuses to start twice.
#
#   ./ops/legacy/start_run_all.sh            # then:  tail -f outputs/run_all.log     or     ./ops/status.sh
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ENV_NAME="${ENV_NAME:-segmentation}"

if pgrep -f "(^|[ /])(run_all|run_full_experiment|run_multiseed)\.sh|run_queue\.py|main\.py run" > /dev/null; then
  echo "[FATAL] a pipeline is already running (see ./ops/status.sh)." >&2
  exit 1
fi
if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
fi
python - <<'PY'
import sys
import torch
print("python", sys.version.split()[0], "| torch", torch.__version__)
PY
chmod +x run_all.sh run_full_experiment.sh run_multiseed.sh
mkdir -p outputs
CUDA_VISIBLE_DEVICES=2,3 nohup ./run_all.sh > outputs/run_all_console.log 2>&1 < /dev/null &
echo "started run_all.sh (pid $!). Log: outputs/run_all.log   Console: outputs/run_all_console.log   Status: ./ops/status.sh"
