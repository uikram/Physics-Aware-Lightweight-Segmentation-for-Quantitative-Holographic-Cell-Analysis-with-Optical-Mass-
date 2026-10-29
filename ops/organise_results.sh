#!/usr/bin/env bash
# The one command that puts the results into the individual / multi-seed layout and checks it (no GPU, a few seconds).
#
#   ./ops/organise_results.sh              # split the older files, compile outputs/compiled/*_multiseed.json, check against the older files
#   ./ops/organise_results.sh --dry-run    # say what would be done
#   ./ops/organise_results.sh --force      # rewrite the individual files that already exist
#
# Safe to repeat. The older result files and the older scripts are not changed. Prints COMPLETE only after every check passed.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME:-segmentation}"
fi
python scripts/organise_results.py "$@"
