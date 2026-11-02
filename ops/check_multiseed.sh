#!/usr/bin/env bash
# One-time preparation and check before a multi-seed stage. Does not train anything and needs no GPU.
#
#   EXPECT_SHA=ab31b31b677ac673 ./ops/check_multiseed.sh 1
#
#  1. fills the dataset hash in outputs/full/run_metadata.json if it is missing (the main run recorded it as null because the
#     dataset was built after the start record). EXPECT_SHA, if given, must be the beginning of the hash of the current
#     data/processed/manifest.csv; otherwise nothing is written.
#  2. checks that the seed-42 runs, the dataset and the base configuration match what the stage needs.
# Exit status 0 = the stage can be started.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STAGE="${1:-1}"
case "$STAGE" in 1|2|3) ;; *) echo "usage: [EXPECT_SHA=<hex prefix>] ./ops/check_multiseed.sh 1|2|3" >&2; exit 2;; esac
if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME:-segmentation}"
fi
sha_arg=()
[ -n "${EXPECT_SHA:-}" ] && sha_arg=(--expect-sha-prefix "$EXPECT_SHA")
python scripts/record_run.py --stage fill-hash "${sha_arg[@]}"
python scripts/aggregate_multiseed.py --check-prereqs --max-stage "$STAGE"
echo "OK: stage $STAGE can be started with ./ops/start_multiseed.sh $STAGE"
