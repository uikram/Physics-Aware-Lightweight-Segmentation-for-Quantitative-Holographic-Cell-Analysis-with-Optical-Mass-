#!/usr/bin/env bash
# Start the multi-seed extension in the background (activates the conda env 'segmentation' itself).
# Refuses to start while another pipeline is running. Stage N runs the blocks of stages 1..N; finished runs are skipped.
#
#   ./ops/start_multiseed.sh 3                 # everything (156 runs) on GPUs 2,3 (default policy)
#   GPUS=free ./ops/start_multiseed.sh 3       # everything on every currently idle GPU (< 1.5 GB used); policy lifted (D-23)
#   GPUS=all  ./ops/start_multiseed.sh 3       # every GPU of the machine, idle or not (only if you know they are yours)
#   GPUS=0,1,4,5 ./ops/start_multiseed.sh 1    # an explicit list
#   SLOTS=4 GPUS=free ./ops/start_multiseed.sh 3   # fewer parallel runs than GPUs (e.g. if the CPU or the RAM is the limit)
# Stages = architectures: 1 = EdgeSAM (52 runs), 2 = + MobileSAM (52), 3 = + MobileNet-UNet (52). 3 seeds for every configuration.
# Progress: ./ops/progress.sh --watch   or   ./ops/status.sh      Log: outputs/multiseed/logs/pipeline.log
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_NAME="${ENV_NAME:-segmentation}"
STAGE="${1:-}"
case "$STAGE" in 1|2|3) ;; *) echo "usage: [GPUS=free|all|2,3|<list>] [SLOTS=n] ./ops/start_multiseed.sh 1|2|3" >&2; exit 2;; esac
GPUS="${GPUS:-2,3}"

if pgrep -f "(^|[ /])(run_all|run_full_experiment|run_multiseed)\.sh|run_queue\.py|main\.py run" > /dev/null; then
  echo "[FATAL] a pipeline is already running (see ./ops/status.sh)." >&2
  exit 1
fi

case "$GPUS" in
  2,3) LIST="2,3"; LIFT=0;;
  free) LIST=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2 < 1500 {printf "%s%s", sep, $1; sep=","}'); LIFT=1;;
  all)  LIST=$(nvidia-smi --query-gpu=index --format=csv,noheader | awk '{printf "%s%s", sep, $1; sep=","}'); LIFT=1;;
  *)    LIST="$GPUS"; LIFT=1;;
esac
[ -n "$LIST" ] || { echo "[FATAL] no GPU selected (GPUS=$GPUS)." >&2; exit 1; }
echo "GPUs for this launch: $LIST   (policy $([ "$LIFT" = 1 ] && echo 'LIFTED for this launch, D-23' || echo 'default: 2,3 only'))"

if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
fi
chmod +x run_multiseed.sh
mkdir -p outputs/multiseed/logs
CUDA_VISIBLE_DEVICES="$LIST" QPI_ALLOW_ALL_GPUS="$LIFT" STAGE="$STAGE" nohup ./run_multiseed.sh > "outputs/multiseed/logs/console_stage${STAGE}.log" 2>&1 < /dev/null &
echo "started run_multiseed.sh, stages 1 to ${STAGE} (pid $!). Progress: ./ops/progress.sh --watch   Log: outputs/multiseed/logs/pipeline.log"
