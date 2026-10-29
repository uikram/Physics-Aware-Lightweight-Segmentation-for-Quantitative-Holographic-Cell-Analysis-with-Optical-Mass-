#!/usr/bin/env bash
# Delete (part of) old_run/, the superseded results of the half-scale maps (DECISIONS.md D-20). Asks before deleting.
#
#   ./ops/legacy/delete_old_run.sh            # "heavy": deletes only checkpoints and prediction files inside old_run, keeps metrics, logs, docs
#   ./ops/legacy/delete_old_run.sh all        # deletes the whole old_run/ folder (also the old maps and the old processed dataset)
#
# Refuses to run until the comparison with the new runs exists (outputs/full/analysis/comparison_with_old_run.md),
# because scripts/compare_with_old_run.py reads old_run/. Set FORCE=1 to skip that check.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODE="${1:-heavy}"
die() { printf '[FATAL] %s\n' "$*" >&2; exit 1; }

[ -d old_run ] || die "old_run/ does not exist."
if [ "${FORCE:-0}" != 1 ] && [ ! -s outputs/full/analysis/comparison_with_old_run.md ]; then
  die "outputs/full/analysis/comparison_with_old_run.md not found: the new run has not finished. (FORCE=1 to override)"
fi
if pgrep -f "(^|[ /])(run_all|run_full_experiment|run_multiseed)\.sh|run_queue\.py|main\.py run" > /dev/null; then
  die "the pipeline is still running."
fi

case "$MODE" in
  heavy)
    echo "Will delete checkpoints/ folders and predictions_*.npz inside old_run/outputs (metrics, logs, docs stay):"
    find old_run/outputs -type d -name checkpoints -prune -exec du -sc {} + 2>/dev/null | tail -1 || true
    find old_run/outputs -name 'predictions_*.npz' -exec du -sc {} + 2>/dev/null | tail -1 || true
    read -r -p 'Type DELETE heavy to continue: ' ans
    [ "$ans" = "DELETE heavy" ] || die "not confirmed. Nothing deleted."
    find old_run/outputs -type d -name checkpoints -prune -exec rm -rf {} +
    find old_run/outputs -name 'predictions_*.npz' -delete
    ;;
  all)
    du -sh old_run
    read -r -p 'Type DELETE old_run to delete the WHOLE folder: ' ans
    [ "$ans" = "DELETE old_run" ] || die "not confirmed. Nothing deleted."
    rm -rf old_run
    ;;
  *) die "unknown mode '$MODE' (heavy | all)";;
esac
echo "Done."
