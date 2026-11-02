"""Individual result files ("units"): one JSON per run (or per timing pass) with provenance, written atomically.

A unit file is the analysis result itself plus a ``provenance`` block::

    {... result fields ..., "unit_schema": 1,
     "provenance": {"generated_utc": ..., "sources": [{"path": "outputs/full/runs/x/predictions_last.npz", "bytes": ..., "sha256": ...}],
                    "config_sha256": {...}, ...}}

Paths are always relative to the project root, so a file written on the server is valid on the PC.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from qpi.utils.io import sha256_file, write_json
from qpi.utils.layout import ROOT, rel

UNIT_SCHEMA = 1
CONFIG_FILES = ("configs/base.yaml", "configs/experiments.yaml", "configs/multiseed.yaml")


def source_record(p: Path | str, root: Path = ROOT) -> Dict[str, Any]:
    p = Path(p)
    return {"path": rel(p, root), "bytes": int(p.stat().st_size), "sha256": sha256_file(p)}


def config_hashes(root: Path = ROOT) -> Dict[str, Optional[str]]:
    return {f: (sha256_file(Path(root) / f) if (Path(root) / f).exists() else None) for f in CONFIG_FILES}


def provenance(sources: Iterable[Path | str] = (), root: Path = ROOT, **extra: Any) -> Dict[str, Any]:
    return {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": [source_record(s, root) for s in sources if Path(s).exists()],
            "config_sha256": config_hashes(root), **extra}


def write_unit(path: Path | str, payload: Dict[str, Any], sources: Iterable[Path | str] = (), root: Path = ROOT,
               **prov_extra: Any) -> None:
    doc = dict(payload)
    doc["unit_schema"] = UNIT_SCHEMA
    doc["provenance"] = provenance(sources, root, **prov_extra)
    write_json(path, doc)


def read_unit(path: Path | str) -> Optional[Dict[str, Any]]:
    """The stored unit, or None if the file is missing, unreadable or from another unit schema."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("unit_schema") == UNIT_SCHEMA else None


def absolute_paths_in(obj: Any) -> list:
    """Strings that look like absolute paths (POSIX '/...' with a directory part, or 'C:\\...'); used by the tests."""
    bad = []
    if isinstance(obj, dict):
        for v in obj.values():
            bad += absolute_paths_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            bad += absolute_paths_in(v)
    elif isinstance(obj, str):
        if (obj.startswith("/") and "/" in obj[1:] and len(obj) > 3) or (len(obj) > 2 and obj[1:3] == ":\\"):
            bad.append(obj)
    return bad
