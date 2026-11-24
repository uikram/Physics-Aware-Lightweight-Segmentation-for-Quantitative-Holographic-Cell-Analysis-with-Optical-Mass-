"""Shared command-line helpers for scripts/*/*.py: configuration with an optional smoke mode, list parsing."""
from __future__ import annotations

from qpicell.config import load_config

SMOKE_PATHS = {"training_results": "smoke/training_results", "results": "smoke/results", "manuscript": "smoke/manuscript",
               "logs": "smoke/logs"}


def add_common(ap) -> None:
    ap.add_argument("--config", default=None, help="configuration file (default: configs/config.yaml)")
    ap.add_argument("--smoke", action="store_true",
                    help="code-path test only: 1 epoch, --max-steps batches, every output under smoke/ (never mixed with real results)")


def config(args):
    ov = {"training": {"epochs": 1}, "paths": SMOKE_PATHS} if getattr(args, "smoke", False) else None
    return load_config(args.config, ov)


def parse_list(s, cast=str):
    return None if s in (None, "", "all") else [cast(x) for x in str(s).split(",")]
