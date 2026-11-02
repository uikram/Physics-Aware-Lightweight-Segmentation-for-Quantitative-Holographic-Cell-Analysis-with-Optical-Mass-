#!/usr/bin/env bash
# Progress: running or not, finished runs of the main experiment and of every multi-seed stage, last log lines, GPUs.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
if pgrep -f "(^|[ /])(run_all|run_full_experiment|run_multiseed)\.sh|run_queue\.py|main\.py run" > /dev/null; then echo "STATE: running"; else echo "STATE: not running"; fi
count() { [ -d "$1" ] && find "$1" -maxdepth 2 -name metrics.json 2>/dev/null | wc -l || echo 0; }
echo "finished runs: main experiment $(count outputs/full/runs) / 27"
if [ -f outputs/multiseed/grid.json ] && command -v python > /dev/null 2>&1; then
  python - <<'PY'
import json, os
grid = json.load(open("outputs/multiseed/grid.json"))["runs"]
def done(rid):
    return os.path.exists(os.path.join("outputs", "multiseed", "runs", rid, "metrics.json"))
for st in sorted(set(g["stage"] for g in grid)):
    ids = [g["run_id"] for g in grid if g["stage"] == st]
    print("multi-seed stage %s: %d / %d" % (st, sum(1 for i in ids if done(i)), len(ids)))
PY
else
  echo "multi-seed: grid not written yet (starts with ./ops/start_multiseed.sh 1)"
fi
# the log that was written last (an old run_all.log from an earlier attempt is not shown unless it is the newest)
newest=$(ls -t outputs/run_all.log outputs/full/logs/pipeline.log outputs/multiseed/logs/pipeline.log 2>/dev/null | head -n 1)
if [ -n "$newest" ]; then echo "--- last lines of $newest"; tail -n 6 "$newest"; fi
latest_train=$(ls -t outputs/multiseed/logs/train/*.log 2>/dev/null | head -n 2)
for f in $latest_train; do echo "--- $f"; tail -n 2 "$f"; done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed -n '3,4p' | sed 's/^/GPU /'
exit 0
