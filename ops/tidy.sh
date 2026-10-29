#!/usr/bin/env bash
# Tidy the project folder: moves leftovers of earlier versions out of the way. Nothing that matters is deleted:
# stale files go to _to_delete/ (delete that folder yourself when you are happy), caches (__pycache__, .pytest_cache) are removed.
#
#   DRY_RUN=1 ./ops/tidy.sh      # only prints what it would do (do this first)
#   ./ops/tidy.sh
#
# What it does
#   * root copies of scripts that now live in ops/ (move_old_run.sh, delete_old_run.sh, pack_results.sh, start_run_all.sh,
#     status_run_all.sh)                                   -> _to_delete/   (only when the ops/ version exists)
#   * ops/move_old_run.sh, delete_old_run.sh, start_run_all.sh (finished one-time helpers) -> ops/legacy/
#   * outputs_small.tgz, move_to_delete.cmd, nohup.out     -> _to_delete/
#   * run_all_console.log in the root                      -> outputs/
#   * MULTISEED_INTERPRETATION.md, results_verification.md -> old_run/docs/ (they describe the old, half-scale numbers)
#   * __pycache__ and .pytest_cache folders                -> removed (regenerated automatically)
#   * stale since the new result layout (D-29), moved to _to_delete/ with their folder structure kept: scripts/run_benchmark (old v2 copy),
#     tests/DECISIONS.md (old duplicate), the half-scale old-run tools and comparison (scripts/compare_with_old_run.py,
#     tests/test_old_run_tools.py, outputs/full/analysis/comparison_with_old_run.*), ops/legacy/, and, when present, the old_run/ folder,
#     bench_run.out, study_run.out, a root run_benchmark.py and multiseed_results.pdf
# Left alone: data/, outputs/ (results), weights/, cache/, manuscript/, configs/, all other code, Seongwahn_tracker_copy.md,
#   manuscript/outputs_multiseed/ and outputs/multiseed/summary_manuscript_names/ (still read until the final numbers are in manuscript/outputs).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DRY="${DRY_RUN:-0}"
run() { if [ "$DRY" = 1 ]; then printf '[dry] %s\n' "$*"; else "$@"; fi; }
say() { if [ "$DRY" = 1 ]; then :; else printf '%s\n' "$*"; fi; }
if pgrep -f "(^|[ /])(run_all|run_full_experiment|run_multiseed)\.sh|run_queue\.py|main\.py run" > /dev/null; then
  if [ "$DRY" = 1 ]; then echo "note: a pipeline is running; a real tidy would refuse (./ops/clean.sh stops it first)."
  else echo "[FATAL] a pipeline is running; tidy when it has finished (or use ./ops/clean.sh, which stops it first)." >&2; exit 1; fi
fi

trash() {              # trash <file>   -> _to_delete/, never overwriting
  [ -e "$1" ] || return 0
  local dst
  dst="_to_delete/$(basename "$1")"
  [ -e "$dst" ] && dst="${dst}.$(date +%H%M%S)"
  run mkdir -p _to_delete
  run mv "$1" "$dst"
  say "moved $1 -> $dst"
}
relocate() {           # relocate <src> <dst-dir>
  [ -e "$1" ] || return 0
  run mkdir -p "$2"
  if [ -e "$2/$(basename "$1")" ]; then trash "$1"; return 0; fi
  run mv "$1" "$2/"
  say "moved $1 -> $2/"
}

for f in move_old_run.sh delete_old_run.sh pack_results.sh start_run_all.sh; do
  trash "$f"                                                         # stale root copies
done
for f in move_old_run.sh delete_old_run.sh start_run_all.sh; do    # finished one-time helpers -> ops/legacy/
  relocate "ops/$f" ops/legacy
done
[ -e ops/status.sh ] && trash status_run_all.sh
for f in outputs_small.tgz move_to_delete.cmd nohup.out; do trash "$f"; done
relocate run_all_console.log outputs
if [ -d old_run ]; then
  for f in MULTISEED_INTERPRETATION.md results_verification.md; do relocate "$f" old_run/docs; done
fi

# stale since the new result layout (D-29): the list lives in scripts/tidy_stale.py (also usable on the PC without bash)
if [ "$DRY" = 1 ]; then python scripts/tidy_stale.py --dry-run; else python scripts/tidy_stale.py; fi
if [ "$DRY" = 1 ]; then
  find . -path ./data -prune -o -path ./cache -prune -o -path ./outputs -prune -o \
       \( -name __pycache__ -o -name .pytest_cache \) -print | sed 's/^/[dry] remove /'
else
  find . -path ./data -prune -o -path ./cache -prune -o -path ./outputs -prune -o \
       \( -name __pycache__ -o -name .pytest_cache \) -type d -prune -exec rm -rf {} +
fi
chmod +x run_*.sh ops/*.sh 2>/dev/null || true
[ "$DRY" = 1 ] && echo "(dry run: nothing changed)" || echo "tidy done."
[ -d _to_delete ] && echo "_to_delete/ holds the moved files; delete the folder when you do not need them."
[ -e Seongwahn_tracker_copy.md ] && echo "note: Seongwahn_tracker_copy.md was left in place (a copy of the tracker, not code)."
exit 0
