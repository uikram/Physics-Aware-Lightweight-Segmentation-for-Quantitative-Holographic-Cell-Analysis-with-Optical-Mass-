"""Guards that keep smoke-test and full-run outputs apart."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from qpi.utils.io import read_json, write_json


def mode_of(cfg: Dict) -> str:
    return "smoke" if cfg.get("smoke") else "full"


def claim_output_root(cfg: Dict) -> Path:
    """Create the output root and bind it to this mode. Refuse to mix smoke and full outputs."""
    mode = mode_of(cfg)
    root = Path(cfg["paths_abs"]["output_root"])
    man = Path(cfg["paths_abs"]["manuscript_outputs"])
    proc = Path(cfg["paths_abs"]["processed_root"])
    if mode == "smoke" and any("smoke" not in p.name for p in (root, man, proc)):
        raise RuntimeError(f"smoke configuration must use '*smoke*' directories, got {root}, {man} and {proc}")
    if mode == "full" and any("smoke" in p.name for p in (root, man, proc)):
        raise RuntimeError(f"full configuration must not use smoke directories ({root}, {man}, {proc})")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "OUTPUT_MODE.json"
    if marker.exists():
        existing = read_json(marker)
        if existing.get("mode") != mode:
            raise RuntimeError(f"{root} belongs to a '{existing.get('mode')}' run; refusing to write '{mode}' outputs")
    else:
        write_json(marker, {"mode": mode, "seed": cfg["seed"]})
    return root
