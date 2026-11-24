"""Shared helpers for the additional analyses (W-1 ... W-5).

Read-only with respect to the project: every script here reads results/, manuscript/tables_for_manuscript/ and
configs/config.yaml, and writes only below additional_analysis/results/<analysis>/.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # lightweight_qpi_segmentation_final_pipeline/
OUT_ROOT = HERE / "results"

ARCH_DISPLAY = {"edge_sam": "EdgeSAM", "mobile_sam": "MobileSAM", "mobilenet_unet": "MobileNet_UNet"}
DISPLAY_ARCH = {v: k for k, v in ARCH_DISPLAY.items()}
ARMS = ["dice", "dice_pmc", "dice_pmc_bga", "dice_pmc_bga_ipp", "dice_cipp", "dice_pmc_cipp", "dice_pmc_bga_cipp"]
ARM_LABEL = {
    "dice": "Dice",
    "dice_pmc": "Dice + PMC",
    "dice_pmc_bga": "Dice + PMC + BGA",
    "dice_pmc_bga_ipp": "Dice + PMC + BGA + global IPP (old full)",
    "dice_cipp": "Dice + cIPP (proposed)",
    "dice_pmc_cipp": "Dice + PMC + cIPP",
    "dice_pmc_bga_cipp": "Dice + PMC + BGA + cIPP (new full)",
}
SEEDS = [42, 43, 44]
PIXEL_SIZE_UM = 0.1441


def _representative() -> Dict[str, str]:
    """The representative condition of each architecture, read from the selection the pipeline wrote.

    This used to be a literal {"EdgeSAM": "r8", "MobileSAM": "r64", "MobileNet_UNet": "r32"}. That silently
    went stale when the rule changed on 18 November 2026: w1 then recomputed at r64 and r32 while Table 1
    reported r32 and r16, so Table S11 and Table 1 disagreed in print about the same configuration. Reading
    the selection makes that impossible. It is also the project's rule that no analysis carries a value that
    the pipeline already computes.
    """
    path = ROOT / "results" / "consolidated" / "selection" / "rank_selection.csv"
    if not path.exists():
        raise SystemExit(f"[common] {path} is missing: run\n"
                         f"  python scripts/evaluation/objective_and_rank_selection.py\n"
                         f"first, so the representative rank comes from the selection and not from a literal.")
    import csv
    out: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            # rank_selection.csv spells the architecture for the manuscript ("MobileNet-UNet"); the result
            # folders use the underscored form ("MobileNet_UNet"), which is what ARCH_DISPLAY holds.
            disp = ARCH_DISPLAY.get(row["architecture"], row["architecture_display"])
            out[disp] = row["selected_condition"]
    if not out:
        raise SystemExit(f"[common] {path} holds no rows.")
    return out


REPRESENTATIVE = _representative()


def load_config() -> Dict:
    import yaml
    with open(ROOT / "configs" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def results_dir(arch_display: str, condition: str, arm: str, seed: int) -> Path:
    return ROOT / "results" / arch_display / condition / arm / f"seed{seed}"


def out_dir(name: str) -> Path:
    d = OUT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean(x: Any) -> Any:
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python, tuples -> lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    return x


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    print(f"[write] {path.relative_to(ROOT)}", flush=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[write] {path.relative_to(ROOT)}", flush=True)


def provenance(script: str, inputs: Iterable[str], extra: Optional[Dict] = None) -> Dict:
    import platform
    meta = {"script": f"additional_analysis/{script}", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version.split()[0], "numpy": np.__version__, "host": platform.node(),
            "inputs": sorted(set(inputs)),
            "note": "Read-only analysis of stored results; no training, no inference, no project file modified."}
    try:
        import pandas, scipy  # noqa: E401
        meta["pandas"] = pandas.__version__; meta["scipy"] = scipy.__version__
    except Exception:
        pass
    if extra:
        meta.update(extra)
    return meta


def holm(pvals: List[Optional[float]]) -> List[Optional[float]]:
    """Holm step-down adjusted p-values; None stays None."""
    idx = [i for i, p in enumerate(pvals) if p is not None and np.isfinite(p)]
    m = len(idx)
    out: List[Optional[float]] = [None] * len(pvals)
    if not m:
        return out
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for k, i in enumerate(order):
        adj = min(1.0, (m - k) * float(pvals[i]))
        running = max(running, adj)
        out[i] = running
    return out


def fmt(v: Any, nd: int = 3, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return f"{int(v):,}"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def fmt_p(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "–"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def md_table(header: List[str], rows: List[List[str]]) -> str:
    L = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(L)


def wilcoxon_signed_rank_exact(d) -> "Optional[float]":
    """Exact two-sided Wilcoxon signed-rank p (zero differences dropped; ties get mid-ranks and the exact conditional
    permutation distribution of the signs is enumerated by dynamic programming). Independent of the scipy version.
    Returns None with fewer than 6 non-zero differences."""
    from scipy.stats import rankdata
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n < 6:
        return None
    r2 = np.rint(2 * rankdata(np.abs(d))).astype(np.int64)        # doubled mid-ranks are integers
    total = int(r2.sum())
    dist = np.zeros(total + 1); dist[0] = 1.0
    for r in r2:
        new = dist.copy(); new[r:] += dist[:-r] if r else 0; dist = new
    dist /= dist.sum()
    t_obs = int(r2[d > 0].sum()); mean = total / 2.0
    sums = np.arange(total + 1)
    p = float(dist[np.abs(sums - mean) >= abs(t_obs - mean) - 1e-9].sum())
    return min(1.0, p)
