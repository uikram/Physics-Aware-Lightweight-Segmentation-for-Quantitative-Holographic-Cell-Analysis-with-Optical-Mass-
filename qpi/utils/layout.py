"""Where the results live. One place, so no script hard-codes a path.

Legacy locations (unchanged, still written and read by the older scripts)::

    outputs/full/runs/<id>/metrics.json              seed-42 runs            outputs/multiseed/runs/<id>/metrics.json   seeds 43, 44 (+ ablations)
    outputs/multiseed/summary/*                       aggregator output       manuscript/outputs_multiseed/multiseed_results.json (copy, written by the aggregator)
    outputs/full/analysis/{phase_integral,drymass,robustness,architecture_comparison}.json
    outputs/full/benchmark/{benchmark,equivalence}.json
    outputs/cell_analysis/cell_analysis.json (+ csv)  manuscript/outputs/results.json

Added layout (DECISIONS.md D-29): one JSON per run for every analysis, and compiled multi-seed JSONs built from them::

    outputs/analysis/cell/<run_id>.json              individual cell / dry-mass / phase-integral result of one run
    outputs/analysis/robustness/<run_id>.json        individual robustness result of one run
    outputs/analysis/benchmark/<arch>__<backend>__s<seed>.json   individual timing result
    outputs/compiled/{segmentation,cell_analysis,robustness,benchmark}_multiseed.json   mean / SD / per-seed / traceability
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_REL = Path("outputs") / "analysis"
COMPILED_REL = Path("outputs") / "compiled"
KINDS = ("cell", "robustness", "benchmark")


def analysis_dir(kind: str, root: Path = ROOT) -> Path:
    if kind not in KINDS:
        raise ValueError(f"unknown analysis kind {kind!r}; choose from {KINDS}")
    return Path(root) / ANALYSIS_REL / kind


def compiled_dir(root: Path = ROOT) -> Path:
    return Path(root) / COMPILED_REL


def compiled_path(name: str, root: Path = ROOT) -> Path:
    return compiled_dir(root) / f"{name}_multiseed.json"


def rel(p: Path | str, root: Path = ROOT) -> str:
    """Path relative to the project root with forward slashes; paths outside the root are reduced to the part from 'outputs/' on."""
    s = str(p).replace("\\", "/")
    r = str(Path(root)).replace("\\", "/").rstrip("/") + "/"
    if s.startswith(r):
        return s[len(r):]
    i = s.find("outputs/")
    return s[i:] if i >= 0 else s


def unit_files(kind: str, root: Path = ROOT):
    """Individual result files of one kind (the helper files starting with '_' are not units)."""
    d = analysis_dir(kind, root)
    return sorted(p for p in d.glob("*.json") if not p.name.startswith("_")) if d.exists() else []
