#!/usr/bin/env python
"""Run all trainable experiment runs with a fixed number of GPU slots.

Each slot is bound to one *visible* CUDA index (0, 1, ...; physical GPUs 2 and 3 under CUDA_VISIBLE_DEVICES=2,3).
Runs are started by stage, then priority (RunSpec.stage / .priority, lower first; 1 and 5 in the main grid), then longest-first. Finished runs are skipped by ``main.py``. If a run fails, no further runs are
started, running ones are allowed to finish and the exit status is 1.

    python scripts/run_queue.py --slots 2 [--config-overlay smoke.yaml] [--log-dir DIR] [--max-stage N]

``--max-stage N`` (multi-seed overlay only) queues the runs of stages 1..N; finished runs are skipped either way.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qpi.config import load_config  # noqa: E402
from qpi.experiments import build_grid, trainable_specs, up_to_stage  # noqa: E402
from qpi.utils.io import read_json  # noqa: E402
from qpi.utils.outputs import claim_output_root  # noqa: E402

COST = {"mobile_sam": 3, "edge_sam": 2, "mobilenet_unet": 1}   # relative run length, for ordering only


def bar(done: int, total: int, width: int = 30) -> str:
    """'[#########.....................] 30% 17/58'"""
    frac = done / total if total else 1.0
    n = int(round(width * frac))
    return "[" + "#" * n + "." * (width - n) + "] %3d%% %d/%d" % (round(100 * frac), done, total)


def is_finished(root: Path, run_id: str) -> bool:
    m = root / "runs" / run_id / "metrics.json"
    try:
        return m.exists() and read_json(m).get("status") in ("complete", "diverged")
    except Exception:  # noqa: BLE001  (a half-written file counts as not finished)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-overlay", default=None)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-stage", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config_overlay)
    root = claim_output_root(cfg)
    log_dir = Path(a.log_dir) if a.log_dir else root / "logs" / "train"
    log_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted(trainable_specs(up_to_stage(build_grid(cfg), a.max_stage)), key=lambda s: (s.stage, s.priority, -COST.get(s.arch, 0), s.run_id))
    total = len(runs)
    pending = [s.run_id for s in runs if a.force or not is_finished(root, s.run_id)]
    finished = total - len(pending)
    print(f"[queue] {total} runs: {finished} already finished, {len(pending)} to train, {a.slots} slot(s)", flush=True)
    active: dict = {}
    failed = []
    while pending or active:
        for slot in range(a.slots):
            if slot not in active and pending and not failed:
                rid = pending.pop(0)
                cmd = [sys.executable, str(ROOT / "main.py"), "run", "--run-id", rid, "--gpu", str(slot)]
                if a.config_overlay:
                    cmd += ["--config-overlay", a.config_overlay]
                if a.force:
                    cmd.append("--force")
                fh = open(log_dir / f"{rid}.log", "w")
                print(f"[queue] start {rid} on visible GPU {slot}", flush=True)
                active[slot] = (rid, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=os.environ.copy()), fh)
        time.sleep(2)
        for slot, (rid, proc, fh) in list(active.items()):
            rc = proc.poll()
            if rc is not None:
                fh.close()
                del active[slot]
                print(f"[queue] {'done' if rc == 0 else 'FAILED'} {rid} (exit {rc})", flush=True)
                if rc != 0:
                    failed.append(rid)
                else:
                    finished += 1
                print(f"[queue] progress {bar(finished, total)}  running {len(active)}  waiting {len(pending)}", flush=True)
        if failed and not active:
            break
    if failed:
        print(f"[queue] failed runs: {failed}; logs in {log_dir}", file=sys.stderr)
        return 1
    print("[queue] all runs finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
