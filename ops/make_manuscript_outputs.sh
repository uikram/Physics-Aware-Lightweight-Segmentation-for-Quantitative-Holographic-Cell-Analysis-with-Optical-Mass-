#!/usr/bin/env bash
# Tables and figures for the manuscript from the finished runs. No training, no GPU.
#
#   ./ops/make_manuscript_outputs.sh                      # rank and architecture of the single-rank figures: configs/base.yaml
#   RANK=16 ARCH=mobile_sam ./ops/make_manuscript_outputs.sh
#
# Steps: (1) copies of the multi-seed summary with the manuscript class names (D-25); (2) manuscript/results.py ->
# manuscript/outputs/results.json (every number of the tables and figures); (3) manuscript/figures.py (reads only results.json,
# the prediction files and the analysis outputs); (4) manuscript/make_tex.py -> manuscript/outputs/multiseed_results.tex (every rank,
# loss, seed and metric, read from results.json). The figures and the tex file never contain numbers that are typed in.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME:-segmentation}"
fi
# shellcheck source=scripts/_pipeline_lib.sh
source scripts/_pipeline_lib.sh
mkdir -p manuscript/outputs/logs
exec > >(tee -a manuscript/outputs/logs/make_manuscript_outputs.log) 2>&1
args=()
[ -n "${RANK:-}" ] && args+=(--rank "$RANK")
[ -n "${ARCH:-}" ] && args+=(--arch "$ARCH")
step "multi-seed summaries with the manuscript class names" python scripts/relabel_summary.py
step "results.json" python manuscript/results.py "${args[@]+"${args[@]}"}"
step "figures" python manuscript/figures.py
step "tex file with every multi-seed result" python manuscript/make_tex.py
step "verify outputs" python scripts/verify_outputs.py
pipeline_footer
