#!/usr/bin/env bash
# Stop old processes and clean the project folder on the server. Nothing is deleted unless you say so: leftovers go to _to_delete/.
#
#   DRY_RUN=1 ./ops/clean.sh      # only prints what it would do
#   ./ops/clean.sh                # stop your pipeline processes, tidy the folder, set aside unfinished/obsolete multi-seed runs
#   PURGE=1 ./ops/clean.sh        # same, then delete _to_delete/ for good
#
# Set aside (moved to _to_delete/): leftovers of earlier versions (see ops/tidy.sh), multi-seed run folders without a finished
# metrics.json (killed runs), runs of seeds that are not in configs/multiseed.yaml any more (45, 46), old logs, the stale summary.
# Kept: data/, weights/, cache/, outputs/full (the main experiment), every finished run. old_run/ is set aside by ops/tidy.sh.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DRY="${DRY_RUN:-0}"
run() { if [ "$DRY" = 1 ]; then printf '[dry] %s\n' "$*"; else "$@"; fi; }

echo "== 1. stop old processes"
if [ "$DRY" = 1 ]; then pgrep -u "$(id -u)" -af '[r]un_multiseed\.sh|[r]un_all\.sh|[r]un_full_experiment\.sh|[r]un_queue\.py|[m]ain\.py run' || echo "nothing running (a dry run does not stop anything)"; else ./ops/stop.sh; fi

echo "== 2. tidy the folder"
DRY_RUN="$DRY" ./ops/tidy.sh

echo "== 3. multi-seed leftovers"
ms=outputs/multiseed
aside() {                      # aside <path> <subfolder>
  [ -e "$1" ] || return 0
  run mkdir -p "_to_delete/$2"
  run mv "$1" "_to_delete/$2/"
  [ "$DRY" = 1 ] || echo "moved $1 -> _to_delete/$2/"
}
if [ -d "$ms/runs" ]; then
  for d in "$ms"/runs/*; do
    [ -d "$d" ] || continue
    case "$(basename "$d")" in
      *__s45|*__s46) aside "$d" seeds45_46; continue;;
    esac
    if ! grep -qE '"status": *"(complete|diverged)"' "$d/metrics.json" 2> /dev/null; then aside "$d" unfinished_runs; fi
  done
  for l in "$ms"/logs/train/*__s45.log "$ms"/logs/train/*__s46.log; do aside "$l" seeds45_46; done
fi
aside "$ms/summary" old_summary
aside "$ms/grid.json" old_summary
aside outputs/run_all.log old_logs
aside outputs/run_all_console.log old_logs
for l in "$ms"/logs/console_stage*.log; do aside "$l" old_logs; done

echo "== 4. GPUs"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2> /dev/null | sed 's/^/  GPU /' || true
echo "   (a GPU showing a few MiB and 0 % is free; anything else is in use by someone)"
echo "   CPU cores: $(nproc 2> /dev/null || echo '?')    free memory: $(free -g 2> /dev/null | awk '/^Mem:/{print $7" GB available"}')"

if [ "${PURGE:-0}" = 1 ] && [ -d _to_delete ]; then
  if [ "$DRY" = 1 ]; then echo "[dry] rm -r _to_delete"; else rm -rf _to_delete; echo "_to_delete/ deleted."; fi
fi
[ "$DRY" = 1 ] && echo "(dry run: nothing changed)" || echo "clean done."
