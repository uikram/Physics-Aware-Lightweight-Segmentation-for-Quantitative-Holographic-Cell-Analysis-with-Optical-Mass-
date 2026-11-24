#!/usr/bin/env bash
# Regenerate every manuscript and Supplementary table from the current results, in one go:
#   1. rerun the additional analyses whose outputs the tables read (W-2, W-4, W-5, W-8, W-7, W-1),
#      each at the representative ranks in results/consolidated/selection/rank_selection.csv;
#   2. run tables_for_manuscript.py, which writes results/tables_for_manuscript/ (JSON, CSV, XLSX).
#
# CPU only, no training or inference; reads results/ and writes only additional_analysis/results/ and
# results/tables_for_manuscript/. Needs the rescored per-run files (cells_last.json with metadata.rescored).
#
#     bash additional_analysis/make_tables.sh              # everything (W-1 bootstrap dominates, about 10 min)
#     BOOT=2000 bash additional_analysis/make_tables.sh    # quicker W-1 bootstrap (not for the final tables)
#     SKIP_ANALYSES=1 bash additional_analysis/make_tables.sh   # tables only, from the existing outputs
#     PYTHON=python3 bash additional_analysis/make_tables.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
BOOT="${BOOT:-10000}"
mkdir -p additional_analysis/results results/tables_for_manuscript
LOG="results/tables_for_manuscript/make_tables.log"
{
echo "=== make_tables, started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)"
"$PY" -c "import numpy, pandas, scipy, yaml, openpyxl; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'scipy', scipy.__version__, 'openpyxl', openpyxl.__version__)"
for f in results/consolidated/runs_last.json results/consolidated/selection/rank_selection.csv; do
  [ -f "$f" ] || { echo "missing $f: run the pipeline's consolidation and selection first"; exit 1; }
done
"$PY" -c "import csv; print('representative ranks:', {r['architecture_display']: r['selected_condition'] for r in csv.DictReader(open('results/consolidated/selection/rank_selection.csv', newline=''))})"

if [ "${SKIP_ANALYSES:-0}" != "1" ]; then
  echo; echo "=== W-2 per-architecture tests (Table 2, Table S6)"
  # --out-name: the script's default folder is w2_per_architecture_rerun, which the tables do not read.
  # --compare-with none: the rerun check compares against the folder being written, i.e. against this run's own
  # output, and then reports the one-factor comparisons as "missing" (it looks for them only in the old
  # lora_ranks_pooled block). It was a one-off check of the Holm 7 -> 6 migration and does not apply here.
  "$PY" additional_analysis/w2_per_architecture_tests.py --checkpoint last --out-name w2_per_architecture --compare-with none
  "$PY" additional_analysis/w2_per_architecture_tests.py --checkpoint best --out-name w2_per_architecture --compare-with none

  echo; echo "=== W-4 full cIPP vs full global IPP (Table S7)"
  "$PY" additional_analysis/w4_cell_vs_global_ipp.py

  echo; echo "=== W-5 reference cells per day (Table S1)"
  "$PY" additional_analysis/w5_reference_cells_checks.py

  echo; echo "=== W-8 signed errors (Table S16)"
  "$PY" additional_analysis/w8_signed_errors.py

  echo; echo "=== W-7 dry-mass outliers and class confusion (Tables S14, S15)"
  "$PY" additional_analysis/w7_followup_checks.py --threshold 10

  echo; echo "=== W-1 cell-paired statistics (Table S8; bootstrap ${BOOT})"
  "$PY" additional_analysis/w1_cell_paired_stats.py --bootstrap "$BOOT"

  # W-7 must have run at the current ranks: its outlier CSV feeds the area ratio of Table S14
  "$PY" - <<'EOF'
import csv, sys
rep = {r["architecture"]: r["selected_condition"]
       for r in csv.DictReader(open("results/consolidated/selection/rank_selection.csv", newline=""))}
key = {"edge_sam": "EdgeSAM", "mobile_sam": "MobileSAM", "mobilenet_unet": "MobileNet_UNet"}
want = {key[a]: c for a, c in rep.items()}
got = {(r["architecture"], r["rank"]) for r in
       csv.DictReader(open("additional_analysis/results/w7_followup/w7a_dry_mass_outliers.csv", newline=""))}
bad = [g for g in got if want.get(g[0]) != g[1]]
if bad:
    sys.exit(f"W-7 outlier CSV holds ranks {sorted(bad)}, not {want}")
print("W-7 ranks match rank_selection.csv:", want)
EOF
fi

echo; echo "=== tables"
"$PY" additional_analysis/tables_for_manuscript.py

echo; echo "=== flags (values that could not be traced)"
cat results/tables_for_manuscript/flags.md
echo "=== finished $(date '+%Y-%m-%d %H:%M:%S')"
} 2>&1 | tee "$LOG"
