#!/usr/bin/env python
"""Make the finished seed-42 runs of the main experiment browsable next to the new seeds.

    python scripts/link_seed42.py [--config-overlay multiseed.yaml] [--dry-run]

For every trained run of the main grid, ``outputs/full/runs/<id>`` is mirrored as ``outputs/multiseed/runs/<id>__s42`` (the
naming of the other seeds), so that all seeds of a configuration sit side by side:

    outputs/multiseed/runs/edge_sam_lora_r8__s42   (the main-experiment run, nothing retrained)
    outputs/multiseed/runs/edge_sam_lora_r8__s43   (new)
    outputs/multiseed/runs/edge_sam_lora_r8__s44   (new)

Files are hard links (no extra disk space; copies where hard links are impossible), the SHA-256 of every metrics.json is
checked against the original, and a note file marks the folder. outputs/full stays the source of truth (the aggregator reads
it and records its hash); an existing mirror is verified, never overwritten. Aliases (``*_abl_full``) are not runs and are skipped.
"""
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qpi.config import load_config  # noqa: E402
from qpi.experiments import build_grid, trainable_specs  # noqa: E402

NOTE = "SEED42_MIRROR_OF_outputs_full.txt"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mirror(main_runs: Path, out_runs: Path, run_ids, dry_run: bool = False):
    """Returns (created, verified, skipped_unfinished, problems)."""
    created, verified, skipped, problems = [], [], [], []
    for rid in run_ids:
        src, dst = main_runs / rid, out_runs / f"{rid}__s42"
        if not (src / "metrics.json").exists():
            skipped.append(rid)
            continue
        if dst.exists():
            if (dst / "metrics.json").exists() and sha(dst / "metrics.json") == sha(src / "metrics.json"):
                verified.append(rid)
            else:
                problems.append(f"{dst} exists but its metrics.json differs from {src}/metrics.json; not touched")
            continue
        if dry_run:
            created.append(rid)
            continue
        shutil.copytree(src, dst, copy_function=link_or_copy)
        (dst / NOTE).write_text(f"Mirror of outputs/full/runs/{rid} (seed 42, main experiment; nothing was retrained).\n"
                                "Hard-linked files: do not edit. outputs/full is the source of truth.\n")
        if sha(dst / "metrics.json") != sha(src / "metrics.json"):
            problems.append(f"{dst}: metrics.json differs after copying")
        created.append(rid)
    return created, verified, skipped, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-overlay", default="multiseed.yaml")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config_overlay)
    ms = cfg["multiseed"]
    base_cfg = {k: v for k, v in cfg.items() if k != "multiseed"}
    ids = [s.run_id for s in trainable_specs(build_grid(base_cfg))]
    main_runs = ROOT / ms["main_output_root"] / "runs"
    out_runs = ROOT / cfg["paths"]["output_root"] / "runs"
    out_runs.mkdir(parents=True, exist_ok=True)
    created, verified, skipped, problems = mirror(main_runs, out_runs, ids, a.dry_run)
    print(f"[link_seed42] {len(created)} {'would be ' if a.dry_run else ''}mirrored, {len(verified)} already there and identical, "
          f"{len(skipped)} main runs not finished (skipped)")
    for p in problems:
        print(f"[link_seed42] PROBLEM: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
