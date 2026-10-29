#!/usr/bin/env bash
# Per-cell analyses for every architecture, LoRA rank and seed, then every JSON/TeX file that depends on them is regenerated.
# No training, no GPU. Run on the machine that holds the runs (the server).
#
#   ./ops/run_cell_analysis.sh                                   # 3 architectures x ranks 2,4,8,16,32 x seeds 42,43,44, full objective
#   JOBS=4 ./ops/run_cell_analysis.sh                            # analyse four runs at a time (CPU processes)
#   ARGS="--losses full dice_only pmc pmc_bga --include-full-finetune" ./ops/run_cell_analysis.sh     # also the ablations / full fine-tuning
#
#   ARGS="--all" ./ops/run_cell_analysis.sh                       # all four losses and full fine-tuning (every run that exists)
#
# 1. scripts/run_cell_analysis.py -> one file per run in outputs/analysis/cell/ (resumable: finished runs are skipped, so the
#    command can simply be repeated), plus the older outputs/cell_analysis/cell_analysis.json, CSV summaries and tables.
#    It also checks the numbers of outputs/full/analysis/{drymass,phase_integral,architecture_comparison}.json.
# 2. scripts/organise_results.py -> outputs/compiled/*_multiseed.json (mean, SD, per seed, missing seeds) and the check.
# 3. scripts/check_freshness.py -> says which result file is older than its inputs, and the command that regenerates it.
# results.json, figures and tex are NOT regenerated here (run ./ops/make_manuscript_outputs.sh when the numbers are final).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME:-segmentation}"
fi
# shellcheck source=scripts/_pipeline_lib.sh
source scripts/_pipeline_lib.sh
mkdir -p manuscript/outputs/logs
exec > >(tee -a manuscript/outputs/logs/run_cell_analysis.log) 2>&1
# one thread per worker process when runs are analysed in parallel (avoids oversubscribing the CPUs)
if [ "${JOBS:-1}" -gt 1 ]; then export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; fi
# shellcheck disable=SC2086
step "cell analysis (all requested runs)" python scripts/run_cell_analysis.py --jobs "${JOBS:-1}" ${ARGS:-}
step "individual -> compiled multi-seed files and checks" python scripts/organise_results.py
step "freshness of every result file" python scripts/check_freshness.py --warn-only
pipeline_footer
