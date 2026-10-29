"""One-off conversion of the older result files into individual result files (DECISIONS.md D-29). Nothing is recomputed and
nothing is deleted or overwritten: the older files stay where they are and keep being read by the older scripts.

    outputs/cell_analysis/cell_analysis.json     -> outputs/analysis/cell/<run id>.json (+ _annotation.json)
    outputs/full/analysis/robustness.json        -> outputs/analysis/robustness/<run id>.json
    outputs/full/benchmark/benchmark.json        -> outputs/analysis/benchmark/<arch>__<backend>__s<seed>.json (+ _protocol.json)

An individual file that already exists is kept unless ``force`` is set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from qpi.analysis.compile import parse_run_id
from qpi.utils.io import read_json, write_json
from qpi.utils.layout import ROOT, analysis_dir, rel
from qpi.utils.results_io import read_unit, write_unit

CELL_JSON = Path("outputs/cell_analysis/cell_analysis.json")
ROBUSTNESS_JSON = Path("outputs/full/analysis/robustness.json")
BENCHMARK_JSON = Path("outputs/full/benchmark/benchmark.json")


def split_cell(root: Path = ROOT, force: bool = False) -> Dict[str, Any]:
    src = Path(root) / CELL_JSON
    if not src.exists():
        return {"written": 0, "kept": 0, "skipped": 0, "note": f"{CELL_JSON} not found"}
    doc = read_json(src)
    meta = doc["meta"]
    store = analysis_dir("cell", root)
    store.mkdir(parents=True, exist_ok=True)
    w = k = sk = 0
    for rid, e in doc["runs"].items():
        if e.get("status") != "complete":
            sk += 1
            continue
        path = store / f"{rid}.json"
        if path.exists() and not force and read_unit(path) is not None:
            k += 1
            continue
        unit = dict(e)
        unit["run_dir"] = rel(e["run_dir"], root)
        unit["schema_version"] = meta["schema_version"]
        write_unit(path, unit, root=root, migrated_from=str(CELL_JSON), source_generated_utc=meta["generated_utc"],
                   manifest_sha256=meta.get("manifest_sha256"))
        w += 1
    write_json(store / "_annotation.json", {"schema_version": meta["schema_version"], "manifest_sha256": meta.get("manifest_sha256"),
                                            "n_validation_images": meta["n_validation_images"], "annotation": doc["annotation"],
                                            "migrated_from": str(CELL_JSON)})
    return {"written": w, "kept": k, "skipped": sk}


def split_robustness(root: Path = ROOT, force: bool = False) -> Dict[str, Any]:
    from qpi.analysis.robustness import unit_rows

    src = Path(root) / ROBUSTNESS_JSON
    if not src.exists():
        return {"written": 0, "kept": 0, "skipped": 0, "note": f"{ROBUSTNESS_JSON} not found"}
    doc = read_json(src)
    tag = doc["checkpoint"]
    store = analysis_dir("robustness", root)
    store.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(doc["rows"])
    w = k = 0
    for rid, g in df.groupby("run_id", sort=False):
        path = store / f"{rid}.json"
        if path.exists() and not force and read_unit(path) is not None:
            k += 1
            continue
        info = parse_run_id(rid)
        if not info:
            raise ValueError(f"cannot parse run id {rid!r}")
        unit = {"run_id": rid, **info, "checkpoint": tag, "status": "complete",
                "checkpoint_file": {"path": None, "sha256": None}, "rows": unit_rows(g.reset_index(drop=True))}
        write_unit(path, unit, root=root, migrated_from=str(ROBUSTNESS_JSON))
        w += 1
    return {"written": w, "kept": k, "skipped": 0}


def split_benchmark(root: Path = ROOT, force: bool = False, src: Path | None = None) -> Dict[str, Any]:
    src = Path(src) if src else Path(root) / BENCHMARK_JSON
    if not src.exists():
        return {"written": 0, "kept": 0, "skipped": 0, "note": f"{rel(src, root)} not found"}
    b = read_json(src)
    store = analysis_dir("benchmark", root)
    store.mkdir(parents=True, exist_ok=True)
    write_json(store / "_protocol.json", {k: b[k] for k in ("protocol", "gpu_name", "host_details", "parameters_by_rank") if k in b})
    w = k = 0
    for r in b["per_seed"]:
        path = store / f"{r['architecture']}__{r['backend']}__s{r['seed']}.json"
        if path.exists() and not force and read_unit(path) is not None:
            k += 1
            continue
        unit = {**r, "status": "complete", "gpu_at_start": b.get("gpu_at_start_of_seeds", {}).get(str(r["seed"]))}
        write_unit(path, unit, sources=[src], root=root, migrated_from=rel(src, root))
        w += 1
    return {"written": w, "kept": k, "skipped": 0}
