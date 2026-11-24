"""Controlled end-to-end checks without data or GPU: one training epoch on synthetic batches (checkpoints, history,
learning rate of the condition), JSON results of several runs -> consolidation -> every figure script."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]


def _batch_items(cfg, arch, n=2):
    from qpicell.data.cells import contact_lines, split_cells, territories
    S = cfg["models"][arch]["image_size"]; C = min(S, cfg["cells"]["cell_map_size"])
    yy, xx = np.mgrid[0:C, 0:C]
    fg = ((yy - C // 2) ** 2 + (xx - C // 3) ** 2 < (C // 6) ** 2) | ((yy - C // 2) ** 2 + (xx - C // 2) ** 2 < (C // 6) ** 2)
    cells = split_cells(fg, 3).astype(np.int64)
    m = torch.from_numpy(fg.astype(np.int64))
    up = lambda t: torch.nn.functional.interpolate(t[None, None].float(), size=(S, S), mode="nearest")[0, 0]
    items = []
    for i in range(n):
        ph = (up(m) * 1.0 + 0.01 * torch.randn(S, S)).float()[None]
        items.append({"input": (ph - ph.mean()) / ph.std(), "mask": up(m).long(), "phase": ph, "cells": torch.from_numpy(cells),
                      "territories": torch.from_numpy(territories(cells, 3).astype(np.int64)),
                      "contacts": torch.from_numpy(contact_lines(cells)).float()})
    return items


def test_one_epoch_training_and_checkpoints(cfg, tmp_path):
    from qpicell.config import load_config
    from qpicell.models import build
    from qpicell.training.runs import Run, learning_rate
    from qpicell.training.trainer import Trainer
    c = load_config(overrides={"training": {"epochs": 1}})
    torch.manual_seed(0)
    for run in (Run("edge_sam", "r8", "dice_pmc_bga_cipp", 42), Run("edge_sam", "fullft", "dice_pmc_bga_ipp", 42)):
        items = _batch_items(c, run.arch)
        dl = DataLoader(items, batch_size=2)
        out = tmp_path / run.run_id
        tr = Trainer(build(c, run.arch, run.rank, pretrained=False), c, run.arch, run.arm, torch.device("cpu"), out, dl, dl, 1,
                     lambda m: None, learning_rate=learning_rate(c, run))
        fit = tr.fit()
        assert fit["status"] == "trained" and fit["epochs_completed"] == 1
        assert (out / "checkpoints" / "last.pt").exists() and (out / "checkpoints" / "best.pt").exists() and (out / "history.csv").exists()
        assert tr.opt.param_groups[0]["initial_lr"] == pytest.approx(1e-4 if run.full_finetune else 5e-4)


def _fake_metrics(cfg, run, rng):
    from qpicell.data.cells import split_cells
    from qpicell.evaluation.evaluate import score_predictions
    N = 128; yy, xx = np.mgrid[0:N, 0:N]
    d1 = (yy - 40) ** 2 + (xx - 35) ** 2 < 196; d2 = (yy - 40) ** 2 + (xx - 60) ** 2 < 196; d3 = (yy - 95) ** 2 + (xx - 90) ** 2 < 196
    mask = np.zeros((N, N), np.uint8); mask[d1 | d2] = 1; mask[d3] = 2
    phase = (rng.normal(0, 0.01, (N, N)) + (mask > 0)).astype(np.float32)
    pred = mask.copy(); pred[rng.random((N, N)) < 0.02] = 0
    c = dict(cfg); c["cells"] = dict(cfg["cells"], area_range_um2=[1.0, 1000.0])
    res = score_predictions(c, {"img": dict(pred=pred, phase=phase, mask=mask, cells=split_cells(mask > 0, 3), wrapped=np.zeros_like(mask), day=0)}, {})
    T = res.pop("cells_tables"); res.pop("per_image")
    meta = {"run": run.to_dict(), "architecture": run.arch, "architecture_display": cfg["models"][run.arch]["display"], "checkpoint": "last",
            "checkpoint_epoch": 100, "n_images": 1, "class_names": cfg["data"]["class_names"],
            "model": {"parameters": {"total": 1000, "trainable": 10 * (run.rank or 100), "lora_parameters": 5 * (run.rank or 0), "lora_layers": 3},
                      "learning_rate": 5e-4, "best_epoch": {"epoch": 90}}}
    cells = [{"instances": m, **r} for m, C in T.items() for r in C.to_dict("records")]
    return {"metadata": meta, **res}, {"metadata": meta, "cells": cells}


def test_json_consolidation(cfg, tmp_path):
    from qpicell.config import load_config
    from qpicell.io import write_json
    from qpicell.training.runs import Run, results_dir
    raw = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    raw["paths"].update(results=str(tmp_path / "results"), manuscript=str(tmp_path / "manuscript"), logs=str(tmp_path / "logs"))
    cf = tmp_path / "config.yaml"; cf.write_text(yaml.safe_dump(raw))
    c = load_config(cf); rng = np.random.default_rng(0)
    for arch in ("edge_sam", "mobile_sam"):
        for cond in ("r8", "r32", "fullft"):
            for arm in ("dice", "dice_pmc_bga_cipp"):
                for seed in (42, 43, 44):
                    run = Run(arch, cond, arm, seed); m, cl = _fake_metrics(c, run, rng)
                    write_json(results_dir(c, run) / "metrics_last.json", m); write_json(results_dir(c, run) / "cells_last.json", cl)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/evaluation/consolidate.py"), "--config", str(cf)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    runs = json.loads((tmp_path / "results/consolidated/runs_last.json").read_text())["runs"]
    assert len(runs) == 36 and {"a2_touching_cell_ipe_signed_median_pct", "a2_drymass_population_diff_pct", "dice_macro"} <= set(runs[0])
    summ = json.loads((tmp_path / "results/consolidated/seed_summary_last.json").read_text())["groups"]
    assert len(summ) == 12 and all(g["n_seeds"] == 3 for g in summ)
