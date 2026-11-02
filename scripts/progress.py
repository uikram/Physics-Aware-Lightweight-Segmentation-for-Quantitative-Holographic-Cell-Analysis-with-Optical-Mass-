#!/usr/bin/env python
"""Live progress of the multi-seed runs: overall and per-stage bars, one line per running run, estimated finish.

    python scripts/progress.py [--watch] [--interval 5] [--root DIR]

Reads outputs/multiseed/grid.json (written when a stage starts), the metrics.json of finished runs and the training logs of
running ones. Read only. Written to run on Python 2 and 3 (the server's default python is old). --watch refreshes the screen
until Ctrl+C; stopping the viewer does not stop the training.
"""
from __future__ import print_function

import argparse
import json
import os
import re
import sys
import time

EPOCH_RE = re.compile(r"\] epoch (\d+)/(\d+) loss=([0-9.]+) val_dice=([0-9.]+)")
DEFAULT_SECONDS = {"mobile_sam": 1800.0, "edge_sam": 600.0, "mobilenet_unet": 400.0}     # used until a run of that kind has finished
COST = {"mobile_sam": 3, "edge_sam": 2, "mobilenet_unet": 1}
ACTIVE_WITHIN = 180                                                                       # a log written in the last 3 minutes = running


def bar(done, total, width=30):
    frac = float(done) / total if total else 1.0
    n = int(round(width * frac))
    return "[" + "#" * n + "." * (width - n) + "] %3d%% %d/%d" % (int(round(100 * frac)), done, total)


def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def last_epoch(log_path, tail_bytes=16384):
    """(epoch, epochs, val_dice) from the end of a training log, or None."""
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - tail_bytes))
            text = fh.read().decode("utf-8", "replace")
    except Exception:
        return None
    hits = EPOCH_RE.findall(text)
    if not hits:
        return None
    e, n, _, dice = hits[-1]
    return int(e), int(n), float(dice)


def fmt_duration(sec):
    sec = int(max(0, sec))
    return "%dh %02dmin" % (sec // 3600, (sec % 3600) // 60) if sec >= 3600 else "%dmin" % max(1, sec // 60)


def eta_seconds(remaining_running, pending_costs):
    """Greedy simulation: running runs free their slot after `remaining_running` seconds; pending runs (in queue order) take the
    slot that frees first. Returns the time until the last run finishes."""
    slots = sorted(remaining_running) if remaining_running else [0.0]
    for c in pending_costs:
        slots[0] += c
        slots.sort()
    return max(slots)


def collect(root):
    out_dir = os.path.join(root, "outputs", "multiseed")
    grid = read_json(os.path.join(out_dir, "grid.json"))
    if not grid:
        return None
    runs = grid["runs"]
    state = {"finished": [], "running": [], "pending": [], "failed": []}
    durations = {}
    now = time.time()
    for r in runs:
        rid = r["run_id"]
        mp = os.path.join(out_dir, "runs", rid, "metrics.json")
        m = read_json(mp)
        if m and m.get("status") in ("complete", "diverged"):
            state["finished"].append(r)
            if m.get("wall_seconds"):
                durations.setdefault(r["arch"], []).append(float(m["wall_seconds"]))
            continue
        log = os.path.join(out_dir, "logs", "train", rid + ".log")
        if os.path.exists(log) and now - os.path.getmtime(log) < ACTIVE_WITHIN:
            ep = last_epoch(log)
            state["running"].append(dict(r, epoch=ep))
        else:
            state["pending"].append(r)
    mean = dict((a, sum(v) / len(v)) for a, v in durations.items())
    return runs, state, mean


def render(root):
    got = collect(root)
    lines = []
    if got is None:
        return "No multi-seed grid yet (outputs/multiseed/grid.json is written when a stage starts)."
    runs, st, mean = got
    total = len(runs)
    done = len(st["finished"])
    secs = lambda arch: mean.get(arch, DEFAULT_SECONDS.get(arch, 900.0))
    rem_running, pend_costs = [], []
    for r in st["running"]:
        ep = r["epoch"]
        frac = (float(ep[0]) / ep[1]) if ep else 0.0
        rem_running.append(max(30.0, secs(r["arch"]) * (1 - frac)))
    pending = sorted(st["pending"], key=lambda r: (r["stage"], r["priority"], -COST.get(r["arch"], 0), r["run_id"]))
    pend_costs = [secs(r["arch"]) for r in pending]
    eta = eta_seconds(rem_running, pend_costs) if (st["running"] or pending) else 0.0
    lines.append("Multi-seed runs        %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("")
    lines.append("TOTAL    %s" % bar(done, total, 40))
    for stg in sorted(set(r["stage"] for r in runs)):
        n = len([r for r in runs if r["stage"] == stg])
        d = len([r for r in st["finished"] if r["stage"] == stg])
        lines.append("stage %d  %s" % (stg, bar(d, n, 40)))
    lines.append("")
    lines.append("running now (%d):" % len(st["running"]))
    for r in sorted(st["running"], key=lambda r: r["run_id"]):
        ep = r["epoch"]
        if ep:
            e, n, dice = ep
            lines.append("  %-44s epoch %2d/%d %s  val_dice %.3f" % (r["run_id"], e, n, bar(e, n, 16)[:18], dice))
        else:
            lines.append("  %-44s starting" % r["run_id"])
    lines.append("waiting: %d    finished: %d" % (len(st["pending"]), done))
    if done < total:
        fin = time.strftime("%H:%M", time.localtime(time.time() + eta))
        basis = "measured" if mean else "rough guess until the first runs finish"
        lines.append("estimated time left: %s (about %s server time; %s)" % (fmt_duration(eta), fin, basis))
    else:
        lines.append("ALL RUNS FINISHED")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    if not a.watch:
        print(render(root))
        return 0
    try:
        while True:
            sys.stdout.write("\033[H\033[J" + render(root) + "\n\n(Ctrl+C closes this view; the training keeps running)\n")
            sys.stdout.flush()
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
