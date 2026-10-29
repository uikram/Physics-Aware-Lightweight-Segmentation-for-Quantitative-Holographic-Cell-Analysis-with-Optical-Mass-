#!/usr/bin/env python
"""Entry point for training + evaluation of a single experiment run.

    python main.py list-runs   [--config-overlay smoke.yaml]
    python main.py write-grid  [--config-overlay smoke.yaml]
    python main.py run --run-id edge_sam_lora_r8 --gpu 0 [--config-overlay smoke.yaml] [--force]

GPU indices are *visible* indices (after CUDA_VISIBLE_DEVICES remapping). Physical GPUs outside
hardware.allowed_physical_gpus are refused.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from qpi.config import load_config  # noqa: E402
from qpi.experiments import build_grid, resolve, run_config, run_dir, trainable_specs  # noqa: E402
from qpi.utils.io import read_json, write_json  # noqa: E402
from qpi.utils.outputs import claim_output_root, mode_of  # noqa: E402


def cmd_list(cfg) -> int:
    for s in trainable_specs(build_grid(cfg)):
        print(s.run_id)
    return 0


def cmd_write_grid(cfg) -> int:
    root = claim_output_root(cfg)
    grid = [s.to_dict() for s in build_grid(cfg)]
    write_json(root / "grid.json", {"mode": mode_of(cfg), "runs": grid})
    print(f"wrote {root / 'grid.json'} ({len(grid)} entries)")
    return 0


def cmd_run(cfg, args) -> int:
    os.environ.setdefault("TORCH_HOME", str(cfg["paths_abs"]["torch_cache"]))
    warnings.filterwarnings("ignore", category=FutureWarning)
    from qpi.data.dataset import build_loaders
    from qpi.evaluation.evaluate import evaluate_checkpoint
    from qpi.models import build_model
    from qpi.training.trainer import Trainer
    from qpi.utils.env import collect_environment, enforce_gpu_policy, resolve_device
    from qpi.utils.seed import seed_everything

    specs = build_grid(cfg)
    spec = resolve(specs, args.run_id)
    if spec.run_id != args.run_id:
        print(f"{args.run_id} is an alias of {spec.run_id}; nothing to train")
        return 0
    claim_output_root(cfg)
    rdir = run_dir(cfg, spec.run_id)
    if (rdir / "metrics.json").exists() and read_json(rdir / "metrics.json").get("status") in ("complete", "diverged") \
            and not args.force:
        print(f"[skip] {spec.run_id}: already finished ({rdir / 'metrics.json'})")
        return 0
    rdir.mkdir(parents=True, exist_ok=True)

    enforce_gpu_policy(cfg["hardware"]["allowed_physical_gpus"])
    device = resolve_device(args.gpu)
    rc = run_config(cfg, spec)
    seed = int(rc["seed"])                       # cfg seed, or the seed of a multi-seed run (RunSpec.seed)
    seed_everything(seed)
    smoke = cfg.get("smoke")
    if smoke:
        rc["training"]["epochs"] = cfg["training"]["epochs"]
    write_json(rdir / "config.json", {"run": rc, "run_spec": spec.to_dict(), "mode": mode_of(cfg),
                                      "data": cfg["data"], "evaluation": cfg["evaluation"], "overlay": cfg["overlay"]})

    train_loader, val_loader = build_loaders(cfg, rc, seed)
    write_json(rdir / "sampler_info.json", train_loader.sampler_info)
    model = build_model(rc, cfg["paths_abs"]["weights_dir"], pretrained=True)
    counts = model.count_parameters()
    print(f"[{spec.run_id}] params total={counts['total']:,} trainable={counts['trainable']:,} "
          f"train_views={len(train_loader.dataset)} val_images={len(val_loader.dataset)} device={device}", flush=True)

    t0 = time.time()
    trainer = Trainer(model, rc, cfg, device, rdir, train_loader, val_loader,
                      smoke["max_steps_per_epoch"] if smoke else None)
    fit = trainer.fit()
    result = {
        "run_id": spec.run_id, "group": spec.group, "arch": spec.arch, "lora_r": spec.lora_r,
        "status": "diverged" if fit["status"] == "diverged" else "complete",
        "mode": mode_of(cfg), "seed": seed,
        "base_run_id": spec.base_id, "block": spec.block, "sampler": rc["training"].get("sampler", "shuffle"),
        "parameters": {"total": counts["total"], "trainable": counts["trainable"],
                       "trainable_percent": 100.0 * counts["trainable"] / counts["total"]},
        "training": {"epochs_completed": fit["epochs_completed"], "epochs_configured": rc["training"]["epochs"],
                     "train_seconds": fit["train_seconds"], "best_epoch_by_val_dice": fit["best_by_val"]},
        "primary_checkpoint": cfg["evaluation"]["checkpoint"],
        "n_train_views": len(train_loader.dataset), "n_val_images": len(val_loader.dataset),
        "environment": collect_environment(device),
    }
    if fit["epochs_completed"] > 0:
        primary = cfg["evaluation"]["checkpoint"]
        tags = [primary] + (["best"] if cfg["evaluation"]["also_evaluate_best"] and primary != "best" else [])
        result["evaluation"] = {}
        for tag in tags:
            ck = rdir / "checkpoints" / f"{tag}.pt"
            res = evaluate_checkpoint(model, ck, rc, cfg, device, rdir, tag)
            res["selected_on_evaluation_split"] = tag == "best"
            result["evaluation"][tag] = res
            print(f"[{spec.run_id}] eval[{tag}] dice={res['mean_dice']:.4f} aji={res['aji']:.4f} "
                  f"bf1={res['bf1']:.4f} ipe={res['ipe_global']:.4f}", flush=True)
    result["wall_seconds"] = time.time() - t0
    write_json(rdir / "metrics.json", result)
    print(f"[{spec.run_id}] {result['status']} -> {rdir / 'metrics.json'}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list-runs", "write-grid", "run"])
    ap.add_argument("--config-overlay", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--gpu", type=int, default=0, help="visible GPU index")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config_overlay)
    if args.command == "list-runs":
        return cmd_list(cfg)
    if args.command == "write-grid":
        return cmd_write_grid(cfg)
    if not args.run_id:
        ap.error("--run-id is required for 'run'")
    return cmd_run(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
