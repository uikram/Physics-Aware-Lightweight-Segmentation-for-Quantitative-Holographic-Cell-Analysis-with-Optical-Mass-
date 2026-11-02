#!/usr/bin/env bash
# Progress bars of the multi-seed runs (total, per stage, running runs with epoch, estimated time left).
#
#   ./ops/progress.sh            # one snapshot
#   ./ops/progress.sh --watch    # live view, refreshes every 5 s; Ctrl+C closes the view only (the training keeps running)
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)" 2> /dev/null
  conda activate "${ENV_NAME:-segmentation}" 2> /dev/null
fi
exec python scripts/progress.py "$@"
