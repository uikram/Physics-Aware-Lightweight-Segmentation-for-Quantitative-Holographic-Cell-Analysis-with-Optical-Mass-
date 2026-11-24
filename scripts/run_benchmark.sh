#!/usr/bin/env bash
# Inference cost of the three architectures: latency, throughput, memory, trainable parameters.
# Separate from the analysis because it needs a quiet graphics card: nothing else should be running on it.
#
#     bash scripts/run_benchmark.sh            # graphics card 0
#     GPU=3 bash scripts/run_benchmark.sh      # another card
#     CPU=1 bash scripts/run_benchmark.sh      # processor only, exploratory
#
# Options, as environment variables:
#     GPU=0          which graphics card
#     CPU=1          time on the processor instead (labelled exploratory in the output)
#     TIMED=500      timed passes per runtime per pass
#     WARMUP=50      passes discarded first
#     PASSES=        comma-separated pass seeds (default: the seeds of the configuration). A pass seed chooses
#                    which validation images are used; it is not a training seed
#     LOSS=dice_cipp which trained run's weights to load
#     CHECKPOINT=last which checkpoint to time
#     RANK=          time this rank for every architecture instead of the representative one
#     ONNX=1         also export each merged network and time it through ONNX Runtime
#     OPSET=17       ONNX operator set to export
#
# Each architecture is timed at the rank the selection rule chose, read from
# results/consolidated/selection/rank_selection.csv. Merging folds the adapters into the base weights, so the
# deployed network is the same at every rank; the script checks that the merged parameter counts agree across
# ranks rather than assuming it.
#
# The benchmark is repeated once per pass seed, each pass in its own process, with the architecture order
# rotated. Latencies of different passes are never pooled: each percentile is taken inside one pass and the
# summary is the mean and the sample standard deviation over the passes.
#
# ONNX needs two extra packages, and only when ONNX=1 is asked for:
#     pip install onnx onnxruntime-gpu        on a machine with a graphics card
#     pip install onnx onnxruntime            processor only
# Inputs and outputs are bound to the graphics card, so ONNX Runtime is not charged for host transfers the
# PyTorch path does not pay. Before timing, the exported graph is compared with PyTorch on the same inputs and
# the agreement is reported next to the timings.

set -euo pipefail

cd "$(dirname "$0")/.."
GPU="${GPU:-0}"

# TensorFloat-32 changes what float32 means, and the libraries read this when they load, so it is set here.
export NVIDIA_TF32_OVERRIDE=0

ARGS=(--timed "${TIMED:-500}" --warmup "${WARMUP:-50}" --loss "${LOSS:-dice_cipp}" \
      --checkpoint "${CHECKPOINT:-last}")
if [ "${CPU:-0}" = "1" ]; then ARGS+=(--cpu); else ARGS+=(--gpu "${GPU}"); fi
[ -n "${PASSES:-}" ] && ARGS+=(--passes "${PASSES}")
[ -n "${RANK:-}" ] && ARGS+=(--rank "${RANK}")
[ "${ONNX:-0}" = "1" ] && ARGS+=(--onnx --onnx-opset "${OPSET:-17}")

echo "=============================================================================="
echo " Inference benchmark"
echo " device  : $([ "${CPU:-0}" = "1" ] && echo "processor (exploratory)" || echo "graphics card ${GPU}")"
echo " timing  : ${TIMED:-500} timed, ${WARMUP:-50} discarded, per runtime per pass"
echo " onnx    : $([ "${ONNX:-0}" = "1" ] && echo "yes, operator set ${OPSET:-17}" || echo "no")"
echo " started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
echo
echo "Make sure nothing else is using this card. Current state:"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader || echo "  nvidia-smi not available"
echo

python scripts/evaluation/benchmark.py "${ARGS[@]}"

echo
echo "=============================================================================="
echo " Finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo
echo "   results/consolidated/benchmark/benchmark_summary.md      readable table"
echo "   results/consolidated/benchmark/benchmark.csv              mean and SD over the passes"
echo "   results/consolidated/benchmark/benchmark_per_pass.csv     one row per pass"
echo "   results/consolidated/benchmark/benchmark_parameters.csv  per architecture and rank"
echo "=============================================================================="
