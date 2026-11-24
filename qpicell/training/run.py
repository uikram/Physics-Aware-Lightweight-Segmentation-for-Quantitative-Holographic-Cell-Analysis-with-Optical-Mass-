"""One complete run: checks, training (validation on fold 0), then evaluation of last.pt (primary) and best.pt on the
66 test images.

training_results/<Arch>/<condition>/<loss>/seed<seed>/
  config.json     run identity, configuration and hash, split stems, environment
  history.csv     per epoch: learning rate, every loss component, validation Dice / AJI / IPE, times
  training.json   status, epochs, best epoch, parameter report, timings, short evaluation summary
  train.log       console output of the run;  checkpoints/last.pt, best.pt
  running.json    exists only while the run is alive (pid, host, GPU)
results/<Arch>/<condition>/<loss>/seed<seed>/   see evaluation/evaluate.py
"""
from __future__ import annotations

import os
import platform
import socket
import sys
import time
from typing import Dict, Optional

from qpicell.config import config_hash
from qpicell.io import read_json, write_json
from qpicell.training.runs import Run, learning_rate, results_dir, training_dir


def environment(device) -> Dict:
    import torch
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(), "host": socket.gethostname(),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": str(device)}
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
    return env


def is_finished(cfg: Dict, run: Run) -> bool:
    p = training_dir(cfg, run) / "training.json"
    try:
        return p.exists() and read_json(p).get("status") in ("complete", "diverged")
    except Exception:
        return False


def execute(cfg: Dict, run: Run, gpu: Optional[int] = None, force: bool = False, max_steps: Optional[int] = None) -> Dict:
    import torch
    from qpicell.data.dataset import build_loaders
    from qpicell.evaluation.evaluate import evaluate_checkpoint
    from qpicell.models import build
    from qpicell.models.lora import parameter_report
    from qpicell.physics import assert_physics, split_stems, verify_prepared
    from qpicell.seeding import seed_everything
    from qpicell.training.trainer import Trainer
    tdir, rdir = training_dir(cfg, run), results_dir(cfg, run)
    if is_finished(cfg, run) and not force:
        print(f"[skip] {run.run_id}: already finished"); return read_json(tdir / "training.json")
    tdir.mkdir(parents=True, exist_ok=True)
    logf = open(tdir / "train.log", "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] [{run.run_id}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    t_start = time.time()
    lock = tdir / "running.json"
    write_json(lock, {"pid": os.getpid(), "host": socket.gethostname(), "gpu": gpu, "started_unix": time.time()})
    try:
        assert_physics(cfg)
        chk = verify_prepared(cfg)
        sp = split_stems(cfg)
        log(f"checks ok: physics constants, class mapping, {chk['images']} prepared float32 phase maps unchanged since preparation | "
            f"split train {len(sp['train'])} / validation {len(sp['validation'])} / test {len(sp['test'])} (disjoint)")
        device = torch.device(f"cuda:{gpu}") if (gpu is not None and torch.cuda.is_available()) else torch.device("cpu")
        seed_everything(run.seed, bool(cfg["training"]["deterministic"]))
        lr = learning_rate(cfg, run)
        write_json(tdir / "config.json", {"run": run.to_dict(), "learning_rate": lr, "config_hash": config_hash(cfg), "split": sp,
                                          "config": {k: v for k, v in cfg.items() if k not in ("paths_abs",)}, "environment": environment(device)})
        train_loader, val_loader = build_loaders(cfg, run.arch, run.seed)
        model = build(cfg, run.arch, run.rank, pretrained=True)
        rep = parameter_report(model, run.rank)
        log(f"START {run.arch} | {'full fine-tuning' if run.full_finetune else f'LoRA rank {run.rank}'} | loss {run.arm} | seed {run.seed} | "
            f"lr {lr:g} | {cfg['training']['epochs']} epochs | device {device}")
        log(f"parameters total {rep['total']:,} | trainable {rep['trainable']:,} ({rep['trainable_pct']:.2f} %) | LoRA layers "
            f"{rep['lora_layers']} with {rep['lora_parameters']:,} adapter parameters | other trainable {rep['trainable_non_lora']:,}")
        log(f"data: train {len(train_loader.dataset)} views ({len(sp['train'])} images x {len(cfg['data']['train_rotations'])} rotations) | "
            f"validation {len(val_loader.dataset)} images (fold 0)")
        trainer = Trainer(model, cfg, run.arch, run.arm, device, tdir, train_loader, val_loader, max_steps, log, learning_rate=lr)
        fit = trainer.fit()
        summary = {"run": run.to_dict(), "learning_rate": lr, "epochs_configured": int(cfg["training"]["epochs"]), **fit,
                   "training_status": fit["status"], "status": "diverged" if fit["status"] == "diverged" else "complete",
                   "parameters": rep, "n_train_views": len(train_loader.dataset), "n_validation_images": len(val_loader.dataset),
                   "config_hash": config_hash(cfg), "environment": environment(device), "history": trainer.history}
        summary["evaluation"] = {}
        model_info = {"parameters": rep, "learning_rate": lr, "epochs_completed": fit["epochs_completed"], "best_epoch": fit["best_epoch"],
                      "train_seconds": fit["train_seconds"],
                      "seconds_per_epoch_median": float(sorted(h["time_train_s"] for h in trainer.history)[len(trainer.history) // 2]) if trainer.history else None}
        if fit["epochs_completed"] > 0:
            for tag in cfg["evaluation"]["checkpoints"]:
                ck = tdir / "checkpoints" / f"{tag}.pt"
                if not ck.exists():
                    continue
                log(f"evaluating {tag}.pt on the {len(sp['test'])} test images ...")
                res = evaluate_checkpoint(cfg, model, ck, run.arch, device, rdir, tag, run.to_dict(), model_info)
                s, a2 = res["semantic"], res["instances"]["A2"]
                t = a2["cells"]["touching"]; i = a2["cells"]["isolated"]
                summary["evaluation"][tag] = {"epoch": res["metadata"]["checkpoint_epoch"], "mean_dice": s["mean_dice"], "bf1": s["bf1"],
                                              "ipe_global": s["ipe_global"], "a2_touching": t, "a2_isolated": i}
                fmt = lambda v: "-" if v is None else f"{v:+.2f}"  # noqa: E731
                log(f"eval[{tag}] epoch {res['metadata']['checkpoint_epoch']}: Dice {s['mean_dice']:.4f} BF1 {s['bf1']:.4f} global IPE {s['ipe_global']:.4f} | "
                    f"A2 touching match {t.get('match_pct', 0):.1f}% signed IPE {fmt(t.get('cell_ipe_signed_median_pct'))}% | "
                    f"isolated match {i.get('match_pct', 0):.1f}% signed IPE {fmt(i.get('cell_ipe_signed_median_pct'))}% | "
                    f"unassigned px {a2['assignment']['unassigned_fg_px']} | lost classes {s['lost_classes'] or 'none'}")
        summary["wall_seconds"] = time.time() - t_start
        write_json(tdir / "training.json", summary)
        log(f"{summary['status'].upper()} in {summary['wall_seconds'] / 60:.1f} min -> {tdir}")
        return summary
    finally:
        lock.unlink(missing_ok=True); logf.close()
