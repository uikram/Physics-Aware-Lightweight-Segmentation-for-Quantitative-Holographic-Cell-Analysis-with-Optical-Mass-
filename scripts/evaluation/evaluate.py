#!/usr/bin/env python
"""(Re-)evaluate finished runs on the 66 test images without training (e.g. after an interrupted evaluation).
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/evaluate.py --archs edge_sam --conditions r32
Only runs with training.json status complete and a checkpoint are evaluated; existing metrics are kept unless --force.
Training already evaluates every run at its end, so this script is normally not needed."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from qpicell.cli import add_common, config, parse_list  # noqa: E402
from qpicell.evaluation.evaluate import evaluate_checkpoint  # noqa: E402
from qpicell.io import read_json  # noqa: E402
from qpicell.models import build  # noqa: E402
from qpicell.models.lora import parameter_report  # noqa: E402
from qpicell.physics import assert_physics, verify_prepared  # noqa: E402
from qpicell.training.runs import grid, results_dir, training_dir  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs"); ap.add_argument("--conditions"); ap.add_argument("--arms"); ap.add_argument("--seeds")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--cpu", action="store_true")
    add_common(ap)
    a = ap.parse_args(); cfg = config(a); assert_physics(cfg); verify_prepared(cfg)
    dev = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda:0")
    runs = grid(cfg, parse_list(a.archs), parse_list(a.conditions), parse_list(a.arms), parse_list(a.seeds, int))
    for r in runs:
        tdir, rdir = training_dir(cfg, r), results_dir(cfg, r)
        tj = tdir / "training.json"
        if not tj.exists() or read_json(tj).get("status") != "complete":
            continue
        T = read_json(tj)
        for tag in cfg["evaluation"]["checkpoints"]:
            ck = tdir / "checkpoints" / f"{tag}.pt"
            if not ck.exists() or ((rdir / f"metrics_{tag}.json").exists() and not a.force):
                continue
            model = build(cfg, r.arch, r.rank, pretrained=False)
            info = {"parameters": parameter_report(model, r.rank), "learning_rate": T.get("learning_rate"),
                    "epochs_completed": T.get("epochs_completed"), "best_epoch": T.get("best_epoch"), "train_seconds": T.get("train_seconds")}
            print(f"[evaluate] {r.run_id} {tag} ...", flush=True)
            res = evaluate_checkpoint(cfg, model, ck, r.arch, dev, rdir, tag, r.to_dict(), info)
            print(f"[evaluate] {r.run_id} {tag}: Dice {res['semantic']['mean_dice']:.4f} -> {rdir}", flush=True)
