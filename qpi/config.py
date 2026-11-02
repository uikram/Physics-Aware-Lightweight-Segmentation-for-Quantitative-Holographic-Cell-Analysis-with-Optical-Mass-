"""Configuration loading: base.yaml + experiments.yaml (+ optional overlay) -> nested dict."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(overlay: Optional[str | Path] = None, root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Load the project configuration.

    Order: configs/base.yaml, configs/experiments.yaml, then the optional overlay (e.g. smoke.yaml).
    Paths in ``cfg['paths']`` are resolved against the project root and stored in ``cfg['paths_abs']``.
    """
    cfg: Dict[str, Any] = {}
    for name in ("base.yaml", "experiments.yaml"):
        with open(root / "configs" / name) as fh:
            cfg = deep_update(cfg, yaml.safe_load(fh) or {})
    if overlay is not None:
        ov = Path(overlay)
        if not ov.is_absolute() and not ov.exists():
            ov = root / "configs" / ov
        with open(ov) as fh:
            cfg = deep_update(cfg, yaml.safe_load(fh) or {})
        cfg["overlay"] = str(ov)
    else:
        cfg["overlay"] = None
    cfg["paths_abs"] = {k: (Path(v) if Path(v).is_absolute() else root / v) for k, v in cfg["paths"].items()}
    cfg["project_root"] = str(root)
    return cfg
