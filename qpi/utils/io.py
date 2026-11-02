"""Small I/O helpers: atomic JSON writes, hashing, stem/extension resolution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

IMAGE_EXTENSIONS = (".tif", ".TIF", ".tiff", ".TIFF")


def write_json(path: Path | str, obj: Any) -> None:
    """Atomically write ``obj`` as JSON (a crash never leaves a half-written file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(sanitize(obj), fh, indent=2, default=_json_default, allow_nan=False)
    os.replace(tmp, path)


def sanitize(o: Any) -> Any:
    """Recursively replace non-finite floats by None so every written JSON is strictly valid."""
    import math
    try:
        import numpy as np
        if isinstance(o, np.floating):
            o = float(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.ndarray):
            o = o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    return o


def read_json(path: Path | str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _json_default(o: Any):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_image(directory: Path, stem: str, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> Optional[Path]:
    """Return ``directory/stem<ext>`` for the first extension that exists, else None."""
    for ext in extensions:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None
