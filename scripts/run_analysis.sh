#!/usr/bin/env bash
# Everything the manuscript revision needs from the finished runs. No training, nothing is retrained.
#
#     bash scripts/run_analysis.sh
#
# Options, as environment variables:
#     TOLERANCE=...     balanced-score points a smaller LoRA rank may give up. Unset (the default) means the
#                       value in configs/config.yaml (selection.tolerance) is used, so the rule carries no
#                       constant chosen here.
#     OBJECTIVE=dice_cipp   the loss configuration proposed as the main objective
#     WORKERS=16        processes for the storage-time trends and the background perturbations
#     SKIP_COLLECT=1    skip steps 1 and 2 only: gathering the runs, and the storage-time trends. These two
#                       read every run and take about ten minutes, and nothing in them changes unless a run
#                       is re-scored, so they can be skipped when only the analysis or the output scripts
#                       changed. Steps 3 to 6 still run; they are cheap and they do depend on those scripts.
#     ONLY_OUTPUTS=1    skip steps 1 to 6 and redraw only the tables and the figures. Correct ONLY when no
#                       analysis script changed - a change to the metric list or the selection rule needs
#                       step 6, so use SKIP_COLLECT=1 for that instead.
#     SKIP_FIGURES=1    build the tables but not the figures
#     SKIP_IMAGES=1      do not build the image figures (Fig. 6 and Fig. S2); they need data/prepared/ and the
#                       saved predictions, so they only work on the training server
#
# It stops at the first failure, and every step prints what it is doing.
#
# Run scripts/run_rescore.sh once before this if the per-cell shape measurements (perimeter, circularity) are not
# yet in the results: it recomputes them from the saved predictions and leaves every other number untouched.

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT="$(pwd)"
TOLERANCE="${TOLERANCE:-}"
OBJECTIVE="${OBJECTIVE:-dice_cipp}"
WORKERS="${WORKERS:-16}"

echo "=============================================================================="
echo " Manuscript analysis"
echo " project    : ${PROJECT}"
echo " objective  : ${OBJECTIVE}"
echo " tolerance  : ${TOLERANCE:-from configs/config.yaml}"
echo " started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

step() { echo; echo "------ $1"; echo; }

need() {
  for f in "$@"; do
    [ -f "$f" ] || { echo "  $f is missing: run without the skip flags first"; exit 1; }
  done
}

if [ "${ONLY_OUTPUTS:-0}" = "1" ]; then
  step "1-6/8  Every analysis skipped (ONLY_OUTPUTS=1); only the tables and figures are redrawn"
  mkdir -p manuscript && date '+%Y-%m-%d %H:%M:%S' > manuscript/.build_started
  need "results/consolidated/runs_last.json" "results/consolidated/selection/rank_selection.csv"
else
  if [ "${SKIP_COLLECT:-0}" != "1" ]; then
    step "1/8  Collecting every run into results/consolidated/"
    python scripts/evaluation/consolidate.py

    step "2/8  Storage-time trends (day 0 to day 47)"
    python scripts/evaluation/storage_trends.py --workers "${WORKERS}"
  else
    step "1-2/8  Gathering and storage trends skipped (SKIP_COLLECT=1); what is on disk is reused"
    need "results/consolidated/runs_last.json" "results/consolidated/storage/runs_last.json"
  fi

  # record when the outputs start being produced, so archive_superseded.py can tell later which files this
  # build wrote and which are left over from an earlier one. Written before anything is produced.
  mkdir -p manuscript && date '+%Y-%m-%d %H:%M:%S' > manuscript/.build_started

  step "3/8  How large the phase gets, per image and per reference cell"
  python scripts/evaluation/phase_range.py

  step "4/8  Objective and representative rank"
# This writes results/consolidated/selection/rank_selection.csv. It runs BEFORE the false-negative analysis
# below, which reads that file to know each architecture's representative rank; with the old ordering that
# analysis used the PREVIOUS selection and silently reported the wrong rank whenever the selection changed.
  if [ -n "${TOLERANCE}" ]; then
    python scripts/evaluation/objective_and_rank_selection.py --objective "${OBJECTIVE}" --tolerance "${TOLERANCE}"
  else
    python scripts/evaluation/objective_and_rank_selection.py --objective "${OBJECTIVE}"
  fi
fi
  step "5/8  False negatives by phase, with and without the cell-level term"
  python scripts/evaluation/phase_quartile_fnr.py

  step "6/8  Does a background perturbation change the measurement"
  python scripts/evaluation/background_invariance.py --workers "${WORKERS}"


step "7/8  Manuscript figures and tables"
# One generator for both, manuscript/make_all.py, which writes manuscript/figures/, manuscript/tables/,
# captions.md, SOURCES.json and BUILD_REPORT.md. It replaced the separate table and figure scripts on
# 15 November 2026. It reads the selection written by step 4,
# so it must run after it.
MAKE_ALL_ARGS=()
[ "${SKIP_FIGURES:-0}" = "1" ] && MAKE_ALL_ARGS+=(--only tables)
if [ "${SKIP_FIGURES:-0}" != "1" ] && [ "${SKIP_IMAGES:-0}" != "1" ]; then
  if [ -d data/prepared ]; then
    MAKE_ALL_ARGS+=(--images)
  else
    echo "  data/prepared/ is absent, so Fig. 6 and Fig. S2 are skipped (run this on the training server for them)"
  fi
fi
python manuscript/make_all.py --checkpoint last "${MAKE_ALL_ARGS[@]+"${MAKE_ALL_ARGS[@]}"}"

step "8/8  Done"

echo
echo "=============================================================================="
echo " Finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo
echo " Everything needed to write the paper is under manuscript/ :"
echo
echo "   manuscript/tables/main/              the main-text tables, .csv and .md"
echo "   manuscript/tables/supplementary/     the Supplementary tables"
echo "   manuscript/tables/all_tables_*.docx  every table in one document, main and supplementary"
echo "   manuscript/figures/main/             Figures 1 to 8, as pdf, svg and png"
echo "   manuscript/figures/supplementary/    Figures S1 to S7"
echo "   manuscript/captions.md               every caption, generated with the figure"
echo "   manuscript/SOURCES.json              for every output, the files (with SHA-256) it was built from"
echo "   manuscript/BUILD_REPORT.md           what was built and what was skipped"
echo
echo "   results/consolidated/phase_range/                   how large the phase gets (no unwrapping needed)"
echo "   results/consolidated/phase_quartile_fnr/            false negatives by phase"
echo "   results/consolidated/background_invariance/         robustness to a background perturbation"
echo
echo "=============================================================================="
