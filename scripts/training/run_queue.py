#!/usr/bin/env python
"""Train (and evaluate) the experiment matrix, one run per GPU at a time, seed by seed.

    export CUDA_DEVICE_ORDER=PCI_BUS_ID            # CUDA index = nvidia-smi index
    nohup python scripts/training/run_queue.py > logs/queue.log 2>&1 &
    tail -f logs/queue.log

GPUs: configs/config.yaml -> queue.gpus (0-6) unless --gpus. A run starts on a GPU only if no training process of
this or the previous pipeline (scripts/training/train.py, scripts/train.py) is on it AND it has at least
models.<arch>.min_free_gpu_mib free; otherwise the queue waits (other users' jobs are never disturbed).
Order: seed 42 for every run first, then seed 43, then seed 44. A seed's runs start only when every run of the earlier
seed has started (with --seed-barrier: finished). Inside a seed: MobileSAM, EdgeSAM, MobileNet-UNet; full fine-tuning,
r64, r32, r16, r8; the seven losses in config order.
Console: every start / finish / failure, each running run's epoch lines (prefixed with its GPU; --quiet to hide), and
every queue.status_every_s seconds a progress summary. Finished runs are skipped; a run that fails within 3 min with a
GPU error is retried once and that GPU is not used again by this queue.
Subsets: --archs, --conditions (r8,r16,r32,r64,fullft), --arms, --seeds (comma-separated)."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qpicell.cli import add_common, config, parse_list  # noqa: E402
from qpicell.progress import bar, hms, run_state, tail  # noqa: E402
from qpicell.training.runs import grid, training_dir  # noqa: E402

GPU_ERRORS = ("CUDA", "cuda", "out of range", "device-side", "out of memory", "NCCL")
TRAIN_SCRIPTS = ("scripts/training/train.py", "scripts/train.py")
SHOW = ("START", "epoch", "eval[", "COMPLETE", "DIVERGED", "parameters", "checks ok", "Error", "error", "Traceback")


def smi(*q):
    return subprocess.run(["nvidia-smi", *q, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout


def gpu_state(phys: int):
    """(free MiB, number of training processes of this user's pipelines on the GPU); (None, 0) without nvidia-smi."""
    try:
        rows = {r.split(",")[0].strip(): [x.strip() for x in r.split(",")] for r in smi("--query-gpu=index,uuid,memory.used,memory.total").strip().splitlines()}
        _, uuid, used, total = rows[str(phys)]
        n = 0
        for l in smi("--query-compute-apps=pid,gpu_uuid").strip().splitlines():
            pid, u = [x.strip() for x in l.split(",")]
            if u == uuid:
                try:
                    cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore").replace("\0", " ")
                    n += any(s in cmd for s in TRAIN_SCRIPTS)
                except OSError:
                    pass
        return int(float(total)) - int(float(used)), n
    except Exception:  # noqa: BLE001
        return None, 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default=None, help="GPU indices (nvidia-smi numbering), default configs/config.yaml -> queue.gpus")
    ap.add_argument("--archs"); ap.add_argument("--conditions"); ap.add_argument("--arms"); ap.add_argument("--seeds")
    ap.add_argument("--seed-barrier", action="store_true", help="start a seed only when every run of the earlier seed has FINISHED")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    add_common(ap)
    a = ap.parse_args(); cfg = config(a); E = int(cfg["training"]["epochs"])
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        sys.exit("set CUDA_DEVICE_ORDER=PCI_BUS_ID first (so CUDA indices equal nvidia-smi indices): export CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit("unset CUDA_VISIBLE_DEVICES: the queue gives each run exactly one GPU itself")
    gpus = [int(g) for g in (a.gpus.split(",") if a.gpus else cfg["queue"]["gpus"])]
    runs = grid(cfg, parse_list(a.archs), parse_list(a.conditions), parse_list(a.arms), parse_list(a.seeds, int))
    todo, elsewhere = [], []
    for r in runs:
        st = run_state(training_dir(cfg, r), E)["state"]
        if a.force or st in ("waiting", "failed"):
            todo.append(r)
        elif st == "running":
            elsewhere.append(r.run_id)
    total = len(runs)
    print(f"[queue] {time.strftime('%Y-%m-%d %H:%M:%S')} | {total} runs | finished {total - len(todo) - len(elsewhere)} | running elsewhere "
          f"{len(elsewhere)} | to train {len(todo)} | GPUs {gpus} | {E} epochs | seeds in order {sorted({r.seed for r in runs})}", flush=True)
    active, failed, retried, bad, waiting_msg = {}, [], set(), set(), {}
    offsets, last = {}, 0.0
    while todo or active:
        for g in gpus:
            if g in active or g in bad or not todo:
                continue
            seeds_left = sorted({r.seed for r in todo})
            cur = seeds_left[0]
            if a.seed_barrier and any(r.seed < cur for r, *_ in active.values()):
                continue
            free, n_train = gpu_state(g)
            pick = None
            for r in todo:
                if r.seed != cur:
                    break
                need = int(cfg["models"][r.arch]["min_free_gpu_mib"])
                if n_train == 0 and (free is None or free >= need):
                    pick = r; break
            if pick is None:
                why = f"{n_train} training process(es) on it" if n_train else f"{free} MiB free"
                if waiting_msg.get(g) != why:
                    waiting_msg[g] = why; print(f"[queue] GPU {g} not used now ({why}); waiting", flush=True)
                continue
            waiting_msg.pop(g, None)
            if run_state(training_dir(cfg, pick), E)["state"] not in ("waiting", "failed") and not a.force:
                print(f"[queue] skip {pick.run_id}: started or finished elsewhere", flush=True); todo.remove(pick); continue
            todo.remove(pick)
            cmd = [sys.executable, str(ROOT / "scripts" / "training" / "train.py"), "--arch", pick.arch, "--condition", pick.condition,
                   "--arm", pick.arm, "--seed", str(pick.seed), "--gpu", "0"]
            if a.config: cmd += ["--config", a.config]
            if a.smoke: cmd.append("--smoke")
            if a.max_steps: cmd += ["--max-steps", str(a.max_steps)]
            if a.force or pick.run_id in retried: cmd.append("--force")
            d = training_dir(cfg, pick); d.mkdir(parents=True, exist_ok=True)
            fh = open(d / "stdout.log", "w"); offsets[str(d / "stdout.log")] = 0
            env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(g)}
            print(f"[queue] {time.strftime('%H:%M:%S')} START  GPU {g}  {pick.run_id}  ({total - len(todo)}/{total} started or done)", flush=True)
            active[g] = (pick, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env), fh, time.time())
        time.sleep(5)
        if not a.quiet:                                      # echo the epoch lines of every running run
            for g, (r, p, fh, t0) in sorted(active.items()):
                f = training_dir(cfg, r) / "stdout.log"
                try:
                    with open(f, errors="replace") as fr:
                        fr.seek(offsets.get(str(f), 0)); new = fr.read(); offsets[str(f)] = fr.tell()
                except OSError:
                    new = ""
                for line in new.splitlines():
                    if any(k in line for k in SHOW):
                        print(f"  GPU {g} | {line}", flush=True)
        for g, (r, p, fh, t0) in list(active.items()):
            rc = p.poll()
            if rc is None:
                continue
            fh.close(); del active[g]
            if rc == 0:
                print(f"[queue] {time.strftime('%H:%M:%S')} DONE   GPU {g}  {r.run_id} ({hms(time.time() - t0)})", flush=True); continue
            txt = tail(training_dir(cfg, r) / "stdout.log", 25); quick = time.time() - t0 < 180
            why = {-9: "killed by signal 9 - usually out of host RAM", -15: "terminated (signal 15)"}.get(rc, f"exit {rc}")
            print(f"[queue] {time.strftime('%H:%M:%S')} FAILED GPU {g}  {r.run_id} ({why}); last lines:\n    | " + txt.replace("\n", "\n    | "), flush=True)
            if quick and any(k in txt for k in GPU_ERRORS):
                bad.add(g); print(f"[queue] GPU {g} not used again by this queue (check nvidia-smi)", flush=True)
            if quick and r.run_id not in retried:
                retried.add(r.run_id); todo.insert(0, r); print(f"[queue] {r.run_id} will be retried once", flush=True)
            else:
                failed.append(r.run_id)
            if len(bad) == len(gpus):
                print("[queue] no usable GPU left; stopping", flush=True); todo.clear()
        if time.time() - last >= int(cfg["queue"]["status_every_s"]) or (not todo and not active):
            last = time.time()
            done = sum(1 for r in runs if run_state(training_dir(cfg, r), E)["state"] in ("complete", "diverged"))
            by_seed = " ".join(f"s{s}:{sum(1 for r in runs if r.seed == s and run_state(training_dir(cfg, r), E)['state'] in ('complete', 'diverged'))}"
                               f"/{sum(1 for r in runs if r.seed == s)}" for s in sorted({r.seed for r in runs}))
            print(f"[status {time.strftime('%H:%M:%S')}] {bar(done, total)} | {by_seed} | running {len(active)} | waiting {len(todo)} | failed {len(failed)}"
                  + (f" | GPUs not used {sorted(bad)}" if bad else ""), flush=True)
            for g, (r, p, fh, t0) in sorted(active.items()):
                st = run_state(training_dir(cfg, r), E); el = time.time() - t0
                eta = el / st["epoch"] * (E - st["epoch"]) if st["epoch"] else None
                vd = "-" if st["val_dice"] is None else f"{st['val_dice']:.4f}"
                print(f"    GPU {g}  {bar(st['epoch'], E, 20)}  val Dice {vd}  elapsed {hms(el)}  left ~{hms(eta)}  {r.run_id}", flush=True)
    if failed:
        print(f"[queue] finished with {len(failed)} failed run(s): {failed}", flush=True); sys.exit(1)
    print("[queue] all runs finished", flush=True)
