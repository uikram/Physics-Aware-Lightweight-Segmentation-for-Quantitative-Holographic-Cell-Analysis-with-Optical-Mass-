#!/usr/bin/env bash
# Everything the manuscript needs, in one run, in the right order.
#
#     bash scripts/run_everything.sh
#
# Options, as environment variables:
#     WORKERS=16     processes for the rescoring; each holds the prepared test images, about 0.6 GB
#     GPU=0          which graphics card the benchmark should use
#     SKIP_RESCORE=1 skip the rescoring, if the shape measurements are already in the cell tables
#     SKIP_FIGURES=1 skip the figures
#     NO_ONNX=1      do not export and time the ONNX graphs
#
# It checks the machine first and refuses to start if anything is missing, so a gap cannot surface hours later.
# It stops at the first failure. Each step prints what it is doing.
#
# Order matters and is not negotiable: the rescoring writes the per-cell shape measurements into the cell tables,
# and the tables of the Supplementary Information are then built from those files. Running the analysis first would
# produce a shape table full of dashes.

set -euo pipefail

cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-16}"
GPU="${GPU:-0}"
START=$(date +%s)

echo "=============================================================================="
echo " Everything for the manuscript"
echo " project : $(pwd)"
echo " started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
echo

echo "###### Step 0 of 3   Checking the machine"
echo
python scripts/preflight.py
echo

if [ "${SKIP_RESCORE:-0}" != "1" ]; then
  echo "###### Step 1 of 3   Per-cell shape measurements, from the saved predictions"
  echo "       the long one: no graphics card, ${WORKERS} processes"
  echo
  WRITE=1 WORKERS="${WORKERS}" bash scripts/run_rescore.sh
else
  echo "###### Step 1 of 3   Rescoring skipped (SKIP_RESCORE=1)"
fi
echo

echo "###### Step 2 of 3   Analysis and the tables"
echo
bash scripts/run_analysis.sh
echo

echo "###### Step 3 of 3   Inference cost"
echo
if [ "${NO_ONNX:-0}" = "1" ]; then
  GPU="${GPU}" bash scripts/run_benchmark.sh
else
  ONNX=1 GPU="${GPU}" bash scripts/run_benchmark.sh
fi

MIN=$(( ($(date +%s) - START) / 60 ))
echo
echo "=============================================================================="
echo " Finished at $(date '+%Y-%m-%d %H:%M:%S'), ${MIN} min in total"
echo
echo " Outputs:"
echo "   results/consolidated/        manuscript/captions.md"
echo "   manuscript/figures/          manuscript/SOURCES.json"
echo "   manuscript/tables/           manuscript/BUILD_REPORT.md"
echo
echo " Start reading at:"
echo "   manuscript/BUILD_REPORT.md"
echo "=============================================================================="
