#!/usr/bin/env bash
# Put everything produced from the OLD (half-scale) date-named phase maps into old_run/ and install the corrected maps.
# Nothing is deleted. Safe to run once; it refuses to run if old_run/ exists or if jobs are still running.
#
#   ./ops/legacy/move_old_run.sh            # does it
#   DRY_RUN=1 ./ops/legacy/move_old_run.sh  # only prints what it would move
#
# Result:  old_run/{data,outputs,manuscript,logs,docs}   superseded material (see old_run/README_OLD_RUN.md)
#          data/raw_data                                  = the corrected maps (former data/raw_data_2)
# Untouched: data/dataset, data/X_train_raw, weights, cache, configs, code.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OLD=old_run
DRY="${DRY_RUN:-0}"

die() { printf '[FATAL] %s\n' "$*" >&2; exit 1; }
run() { if [ "$DRY" = 1 ]; then printf '[dry] %s\n' "$*"; else "$@"; fi; }
move() {                       # move <src> <dst>   (skips silently when src does not exist)
  if [ -e "$1" ]; then
    run mkdir -p "$(dirname "$2")"
    run mv "$1" "$2"
    if [ "$DRY" != 1 ]; then printf 'moved %s -> %s\n' "$1" "$2"; fi
  fi
}

[ -e "$OLD" ] && die "$OLD already exists. Nothing done."
[ -d data/raw_data_2 ] || die "data/raw_data_2 (the corrected maps) not found."
[ -d data/raw_data ]   || die "data/raw_data (the old maps) not found."
if pgrep -f "(^|[ /])(run_all|run_full_experiment|run_multiseed)\.sh|run_queue\.py|main\.py run" > /dev/null; then
  die "training jobs are still running. Wait for them or stop them first."
fi
n_new=$(find data/raw_data_2 -maxdepth 1 -name '*.tif' | wc -l)
n_old=$(find data/raw_data   -maxdepth 1 -name '*.tif' | wc -l)
[ "$n_new" = "$n_old" ] || die "raw_data has $n_old maps but raw_data_2 has $n_new. Refusing."

run mkdir -p "$OLD/data" "$OLD/outputs" "$OLD/manuscript" "$OLD/logs" "$OLD/docs"

# data: processed dataset and old maps
move data/processed          "$OLD/data/processed"
move data/processed_smoke    "$OLD/data/processed_smoke"
move data/raw_data           "$OLD/data/raw_data_half_scale"
run mv data/raw_data_2 data/raw_data
printf 'installed corrected maps: data/raw_data_2 -> data/raw_data (%s files)\n' "$n_new"

# results of the old data
move outputs/full            "$OLD/outputs/full"
move outputs/multiseed       "$OLD/outputs/multiseed"
move outputs/smoke           "$OLD/outputs/smoke"
move manuscript/outputs            "$OLD/manuscript/outputs"
move manuscript/outputs_multiseed  "$OLD/manuscript/outputs_multiseed"
move manuscript/outputs_smoke      "$OLD/manuscript/outputs_smoke"

# console logs and notes that describe the old numbers
for f in full_console.log full_console2.log multiseed_console.log smoke_console.log nohup.out; do move "$f" "$OLD/logs/$f"; done
for f in results_verification.md MULTISEED_INTERPRETATION.md; do move "$f" "$OLD/docs/$f"; done

if [ "$DRY" != 1 ]; then
cat > "$OLD/README_OLD_RUN.md" <<EOF
# old_run: superseded results (moved $(date -u +%Y-%m-%d))

Everything here was produced from the FIRST delivery of the date-named float phase maps, which were confined to [0, pi].
The data provider found that their range had been halved before delivery; the corrected maps are exactly 2x (range
[0, 2*pi]) and replaced them (DECISIONS.md D-20). Nothing in this folder is valid for the manuscript.

| path | content |
|---|---|
| data/raw_data_half_scale | the old date-named maps (103 files) |
| data/processed           | the dataset built from them |
| outputs/full             | main experiment, 27 runs, seed 42 (EXPERIMENTS.md: series F0) |
| outputs/multiseed        | multi-seed extension, 51 runs, seeds 42-46 (series M0) |
| manuscript/outputs*      | results.json and figures built from these runs |
| logs, docs               | console logs, results_verification.md, MULTISEED_INTERPRETATION.md (they describe the old numbers) |

scripts/compare_with_old_run.py compares the current runs with these ones, run by run.
EOF
fi
printf '\nDone. Next: ./run_all.sh   (./ops/legacy/start_run_all.sh)\n'
