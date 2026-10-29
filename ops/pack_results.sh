#!/usr/bin/env bash
# Pack the results of the current run (without checkpoints) into /tmp/qpi_results.tgz for download to the PC.
# Used by tools_pc/download_results.ps1; WITH_CHECKPOINTS=1 includes the model checkpoints (large).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LIST=""
for p in outputs manuscript/outputs manuscript/outputs_multiseed data/processed/dataset_report.json \
         data/processed/manifest.csv data/processed/verification_report.json data/processed/missing_float_data.csv; do
  [ -e "$p" ] && LIST="$LIST $p"
done
EXC="--exclude=checkpoints"
[ "${WITH_CHECKPOINTS:-0}" = 1 ] && EXC=""
# shellcheck disable=SC2086
tar czf /tmp/qpi_results.tgz $EXC $LIST
ls -l /tmp/qpi_results.tgz
