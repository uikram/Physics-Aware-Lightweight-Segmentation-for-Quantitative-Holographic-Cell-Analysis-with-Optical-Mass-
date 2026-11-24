"""Small file helpers: JSON with NumPy types, atomic writes, image lookup."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, Path):
        return str(o)
    return o


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")     # per-process name: no clash between parallel writers
    with open(tmp, "w") as fh:
        json.dump(_clean(obj), fh, indent=2)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


IMAGE_SUFFIXES = (".tif", ".tiff", ".TIF", ".TIFF")


def find_image(folder: Path, stem: str) -> Optional[Path]:
    for s in IMAGE_SUFFIXES:
        p = Path(folder) / f"{stem}{s}"
        if p.exists():
            return p
    return None


def list_stems(folder: Path, exclude_prefix: Iterable[str] = ("rotation",)) -> list:
    out = set()
    for p in Path(folder).iterdir():
        if p.suffix in IMAGE_SUFFIXES and not any(p.stem.startswith(x) for x in exclude_prefix):
            out.add(p.stem)
    return sorted(out)
