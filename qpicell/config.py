"""Configuration: configs/config.yaml -> dict with absolute paths in cfg["paths_abs"]."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def _merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None,
                root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Load the configuration. `overrides` (nested dict) is merged on top (used by tests)."""
    with open(Path(path) if path else DEFAULT_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = _merge(cfg, overrides)
    root = Path(root)
    cfg["project_root"] = str(root)
    cfg["paths_abs"] = {k: ([root / x for x in v] if isinstance(v, list) else root / v) for k, v in cfg["paths"].items()}
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    """Short hash of the configuration (paths excluded), written next to every result."""
    c = {k: v for k, v in cfg.items() if k not in ("paths_abs", "project_root")}
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:16]


def dry_mass_factor_pg(cfg: Dict[str, Any]) -> float:
    """Dry mass per integrated phase: m = lambda / (2 pi alpha) * sum(phi) * dx^2  (pg per rad px)."""
    import math
    p = cfg["physics"]
    return p["wavelength_um"] / (2 * math.pi * p["refractive_increment_ml_per_g"]) * p["pixel_size_um"] ** 2


def pixel_area_um2(cfg: Dict[str, Any]) -> float:
    return cfg["physics"]["pixel_size_um"] ** 2
