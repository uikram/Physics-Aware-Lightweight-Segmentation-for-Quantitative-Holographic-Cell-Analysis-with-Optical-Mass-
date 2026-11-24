#!/usr/bin/env bash
# Start the full experiment queue in the background (GPUs 0-6, seed 42 -> 43 -> 44) and follow it.
#   bash scripts/training/start_queue.sh            (from the project folder)
# Extra arguments are passed to run_queue.py, e.g.  bash scripts/training/start_queue.sh --seeds 42
cd "$(dirname "$0")/../.." || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
mkdir -p logs; LOG="logs/queue_$(date +%Y%m%d_%H%M%S).log"
nohup python scripts/training/run_queue.py "$@" > "$LOG" 2>&1 &
echo "queue started: PID $! | log $LOG"
echo "follow it with:   tail -f $LOG        (Ctrl+C stops only the viewer, not the queue)"
echo "progress table:   python scripts/utilities/status.py"
echo "stop the queue:   kill $!   (running runs finish their current process; restart later - finished runs are skipped)"
sleep 3; tail -n 20 "$LOG"
