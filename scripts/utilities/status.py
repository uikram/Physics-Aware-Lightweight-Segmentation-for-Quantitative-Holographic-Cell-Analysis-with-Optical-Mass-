#!/usr/bin/env python
"""Progress of the training runs at any time (starts nothing).
    python scripts/utilities/status.py [--archs edge_sam] [--conditions r32,fullft] [--seeds 42] [--failed-logs]"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qpicell.cli import add_common, config, parse_list  # noqa: E402
from qpicell.progress import bar, hms, run_state, tail  # noqa: E402
from qpicell.training.runs import grid, training_dir  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs"); ap.add_argument("--conditions"); ap.add_argument("--arms"); ap.add_argument("--seeds")
    ap.add_argument("--failed-logs", action="store_true")
    add_common(ap)
    a = ap.parse_args(); cfg = config(a); E = int(cfg["training"]["epochs"])
    runs = grid(cfg, parse_list(a.archs), parse_list(a.conditions), parse_list(a.arms), parse_list(a.seeds, int))
    states = [(r, run_state(training_dir(cfg, r), E)) for r in runs]
    n = {k: sum(1 for _, s in states if s["state"] == k) for k in ("complete", "diverged", "running", "failed", "waiting")}
    print(f"{bar(n['complete'] + n['diverged'], len(runs))} | running {n['running']} | failed {n['failed']} | waiting {n['waiting']} | diverged {n['diverged']}")
    for key, f in (("seed", lambda r: r.seed), ("architecture", lambda r: r.arch), ("condition", lambda r: r.condition)):
        print(f"by {key}:")
        for v in sorted({f(r) for r in runs}, key=str):
            sub = [s for r, s in states if f(r) == v]
            print(f"  {str(v):15s} {bar(sum(s['state'] in ('complete', 'diverged') for s in sub), len(sub), 20)}")
    for r, s in states:
        if s["state"] == "running":
            vd = "-" if s["val_dice"] is None else f"{s['val_dice']:.4f}"
            print(f"  RUNNING {bar(s['epoch'], E, 20)} val Dice {vd} elapsed {hms(s['elapsed_s'])} left ~{hms(s['eta_s'])}  {r.run_id}")
    for r, s in states:
        if s["state"] in ("failed", "diverged"):
            print(f"  {s['state'].upper():8s} {r.run_id}")
            if a.failed_logs and s["state"] == "failed":
                print("    | " + tail(training_dir(cfg, r) / "train.log", 20).replace("\n", "\n    | "))
