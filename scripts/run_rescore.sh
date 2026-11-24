#!/usr/bin/env bash
# Recompute the results of the finished runs from their saved predictions, to add the per-cell shape measurements
# (perimeter and circularity) that were added to the scoring code after the runs had finished.
#
# No training, no model and no graphics card: the saved class maps are re-scored on the processor.
#
#     bash scripts/run_rescore.sh                  check first, write nothing
#     WRITE=1 bash scripts/run_rescore.sh          check and rewrite the results
#     WORKERS=6 WRITE=1 bash scripts/run_rescore.sh
#
# Options, as environment variables:
#     WRITE=1        actually rewrite the results (without it the script only checks)
#     WORKERS=3      processes; each one holds the prepared test images, about 0.6 GB
#     TOLERANCE      relative difference allowed between a stored and a recomputed number (default 1e-9)
#     ARCHS          limit to these architectures, comma separated
#     CONDITIONS     limit to these conditions, comma separated
#     ARMS           limit to these loss configurations, comma separated
#     SEEDS          limit to these seeds, comma separated
#
# Every metric that existed before has to come back identical, because the saved class maps are the arg-max the
# networks produced. The script checks that for every checkpoint rather than assuming it, and leaves on disk,
# untouched, any checkpoint whose numbers do not come back the same. Run it without WRITE first: that reports the
# agreement for all of them and changes nothing.
#
# Run this once before scripts/run_analysis.sh.

set -euo pipefail

cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-3}"

ARGS=(--workers "${WORKERS}")
[ "${WRITE:-0}" = "1" ] || ARGS+=(--dry-run)
[ -n "${TOLERANCE:-}" ] && ARGS+=(--tolerance "${TOLERANCE}")
[ -n "${ARCHS:-}" ] && ARGS+=(--archs "${ARCHS}")
[ -n "${CONDITIONS:-}" ] && ARGS+=(--conditions "${CONDITIONS}")
[ -n "${ARMS:-}" ] && ARGS+=(--arms "${ARMS}")
[ -n "${SEEDS:-}" ] && ARGS+=(--seeds "${SEEDS}")

echo "=============================================================================="
echo " Rescoring the finished runs from their saved predictions"
echo " mode     : $([ "${WRITE:-0}" = "1" ] && echo "check and rewrite the results" || echo "check only, nothing is written")"
echo " processes: ${WORKERS}"
echo " started  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
echo
echo "No training and no graphics card: this is processor work only."
echo

python scripts/evaluation/rescore.py "${ARGS[@]}"

echo
echo "=============================================================================="
echo " Finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo
if [ "${WRITE:-0}" = "1" ]; then
  echo "   results/<Architecture>/<condition>/<loss>/seed<seed>/cells_<checkpoint>.json"
  echo "       now carries perimeter_um and circularity per cell"
  echo "   results/consolidated/rescore_log.json      what was reproduced, skipped or disagreed"
  echo
  echo " Next: bash scripts/run_analysis.sh"
else
  echo "   results/consolidated/rescore_log.json      the agreement check; nothing was written"
  echo
  echo " If every checkpoint reproduced its stored results, run it again with WRITE=1:"
  echo "     WRITE=1 bash scripts/run_rescore.sh"
fi
echo "=============================================================================="
