#!/usr/bin/env python
"""Check that the revision artifacts are traceable to the experiment outputs and up to date.

    python artifacts/code/verify_artifacts.py

1. configs/base.yaml and configs/experiments.yaml are the ones the runs were trained with (run_metadata.json hashes).
2. Every compiled file is complete (no missing seed, no problem listed).
3. Every LoRA run's metrics.json has the SHA-256 recorded in outputs/multiseed/summary/runs_index.csv (180 runs),
   and the three full fine-tuning runs exist.
4. The class-name exchange (D-25) is consistent between the compiled segmentation file (legacy names) and the compiled
   cell-analysis file (manuscript names) for every run.
5. Every table under artifacts/tables is exactly what make_tables.py produces now (no hand edits, no stale file).
Exit status 1 on any failure; the last line is OK or FAILED.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main() -> int:
    errors = []
    root = C.ROOT
    for mode in ("full", "multiseed"):
        meta = C.read_json(C.OUT / mode / "run_metadata.json")
        for name in ("base.yaml", "experiments.yaml"):
            if meta["config_file_sha256"].get(name) != C.sha256(root / "configs" / name):
                errors.append(f"configs/{name} differs from the version outputs/{mode} was trained with")
    print(f"[1] configuration hashes checked ({len(errors)} problems)")

    for name in ("segmentation", "cell_analysis", "benchmark", "robustness"):
        try:
            C.compiled(name)
        except RuntimeError as e:
            errors.append(str(e))
    print("[2] compiled files complete" if not errors else f"[2] {errors[-1]}")

    ri = pd.read_csv(C.OUT / "multiseed" / "summary" / "runs_index.csv")
    bad = [r.run_id for r in ri.itertuples() if C.sha256(root / r.metrics_path) != r.metrics_sha256]
    errors += [f"metrics.json changed since indexing: {b}" for b in bad]
    for a in C.ARCHS:
        if not (C.run_dir(a, None, "full_finetune", 42) / "metrics.json").exists():
            errors.append(f"missing full fine-tuning run {a}")
    print(f"[3] {len(ri)} indexed runs, {len(bad)} with a changed metrics.json")

    seg = C.compiled("segmentation")["configurations"]
    cell = C.cell_configs()
    swap = {"dice_spherocyte": "dice_echinocyte", "dice_echinocyte": "dice_spherocyte", "mean_dice": "mean_dice"}
    n = 0
    for k, v in seg.items():
        for man, leg in swap.items():
            a, b = cell[k]["metrics"][man]["per_seed"], v["metrics"][leg]["per_seed"]
            n += 1
            if any(abs(a[s] - b[s]) > 1e-12 for s in a):
                errors.append(f"class-name exchange inconsistent for {k} {man}")
    print(f"[4] class-name exchange checked on {n} metric series")

    import make_tables as T
    stale = 0
    for t in T.REGISTRY:
        res = t["fn"]()
        f = C.TABLES / t["part"] / f"{t['id']}.csv"
        if not f.exists():
            errors.append(f"table missing: {f}")
            stale += 1
            continue
        disk = pd.read_csv(f, dtype=str, keep_default_na=False)
        now = res["df"].fillna("").astype(str)
        now.columns = [str(c) for c in now.columns]
        if disk.shape != now.shape or not (disk.values == now.values).all():
            errors.append(f"table out of date or edited by hand: {f.relative_to(root)} (rerun make_tables.py)")
            stale += 1
    print(f"[5] {len(T.REGISTRY)} tables compared with a fresh build, {stale} differ")

    # 6. Class labels tied to mask values (author: 2 = spherocyte, 3 = echinocyte). Uses only the raw predictions: a class whose
    #    Dice is 0 must have no predicted pixel of its mask value, and a class with Dice >= 0.05 must have some.
    import numpy as np
    g = pd.read_csv(C.OUT / "cell_analysis" / "annotation" / "cells_gt.csv")
    got = {int(k): set(v) for k, v in g.groupby("class_idx").class_name.unique().items()}
    if got != {v: {k} for k, v in C.MASK_VALUE.items()}:
        errors.append(f"cells_gt.csv class_idx -> name is {got}, expected {C.MASK_VALUE}")
    checked, notes = 0, []
    for _, r in C.runs_frame().iterrows():
        rk = None if pd.isna(r["rank"]) else int(r["rank"])
        pf = C.run_dir(r.arch, rk, r.loss, int(r.seed)) / "predictions_last.npz"
        if not pf.exists():
            notes.append(f"{r.run_id}: predictions_last.npz not in the repository (its cell analysis exists; copy the file from the server)")
            continue
        with np.load(pf) as z:
            counts = np.zeros(5, np.int64)
            for k in z.files:
                counts += np.bincount(z[k].ravel(), minlength=5)[:5]
        for cl, mv in C.MASK_VALUE.items():
            d = r[f"dice_{cl}"]
            if (d == 0) != (counts[mv] == 0) and not (0 < d < C.COLLAPSE_DICE):
                errors.append(f"{r.run_id}: {cl} Dice {d:.3f} but {counts[mv]} predicted pixels of mask value {mv}")
        checked += 1
    print(f"[6] class names tied to mask values in {checked} runs (2 = spherocyte, 3 = echinocyte)")
    for n in notes:
        print("  NOTE:", n)

    for e in errors:
        print("  ERROR:", e)
    print("OK" if not errors else "FAILED")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
