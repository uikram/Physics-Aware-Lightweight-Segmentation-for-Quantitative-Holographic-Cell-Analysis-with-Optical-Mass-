#!/usr/bin/env bash
# Stop every training pipeline process of YOUR user (run_all / run_full_experiment / run_multiseed, the queue and the runs).
# Does nothing to other users' processes. Finished runs are kept; runs in progress are lost and start over next time.
#
#   ./ops/stop.sh            # stop them
#   ./ops/stop.sh --check    # only look: exit status 3 and a message if something is running, 0 if not
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
me=$(id -u)
LAUNCHERS='run_multiseed\.sh|run_all\.sh|run_full_experiment\.sh|run_queue\.py'
RUNS='main\.py run'

# this script and the shells above it are never stopped, even if their command line mentions one of the names
protected=" $$ "
p=$$
while [ "$p" -gt 1 ] 2> /dev/null; do
  p=$(ps -o ppid= -p "$p" 2> /dev/null | tr -d ' ')
  [ -n "$p" ] && protected="$protected$p "
done
pids() {                       # pids <regex>
  local x
  for x in $(pgrep -u "$me" -f "$1" || true); do
    case "$protected" in *" $x "*) ;; *) echo "$x";; esac
  done
}
show() { local x; for x in $(pids "$LAUNCHERS|$RUNS"); do ps -o pid=,etime=,args= -p "$x" 2> /dev/null | cut -c1-150; done; }

if [ -z "$(pids "$LAUNCHERS|$RUNS")" ]; then echo "nothing of yours is running."; exit 0; fi
if [ "${1:-}" = "--check" ]; then
  echo "A TRAINING JOB IS RUNNING; nothing was changed. Run 2_clean_server.ps1 first (or 1_upload.ps1 -Stop):"; show; exit 3
fi
echo "stopping:"; show
for x in $(pids "$LAUNCHERS"); do kill "$x" 2> /dev/null || true; done      # launchers first, so that no new run is started
sleep 2
for x in $(pids "$RUNS"); do kill "$x" 2> /dev/null || true; done
for _ in 1 2 3 4 5 6 7 8 9 10; do [ -z "$(pids "$LAUNCHERS|$RUNS")" ] && break; sleep 2; done
if [ -n "$(pids "$LAUNCHERS|$RUNS")" ]; then
  echo "still alive after 20 s, forcing:"; show
  for x in $(pids "$LAUNCHERS|$RUNS"); do kill -9 "$x" 2> /dev/null || true; done
  sleep 2
fi
if [ -z "$(pids "$LAUNCHERS|$RUNS")" ]; then echo "stopped."; else echo "[FATAL] could not stop:"; show; exit 1; fi
