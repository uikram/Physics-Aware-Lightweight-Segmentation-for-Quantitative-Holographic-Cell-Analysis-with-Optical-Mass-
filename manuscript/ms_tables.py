"""Multi-seed tables for the manuscript, built from ``outputs/multiseed/summary/runs_index.csv`` (aggregator output).

Pure functions (no GPU, no training code). The class names are exchanged for the manuscript order on reading
(``qpi/utils/classnames.py``, DECISIONS.md D-25). Nothing is typed in: every number is copied from the aggregator output or derived
from it; the ranks, seeds and losses present, and the threshold that defines a lost class, are read from those files.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qpi.utils.classnames import CLASSES, swap_text  # noqa: E402

LOSS_ORDER = ["dice_only", "pmc", "pmc_bga", "full"]      # display order of the loss names; which ones exist comes from the data
METRICS = (["mean_dice", "aji", "bf1", "ipe_global_pct"] + [f"dice_{c}" for c in CLASSES] + [f"ipe_{c}_pct" for c in CLASSES])


def parse_config(cfg_key: str, selected_rank: int = 8) -> Optional[Dict[str, Any]]:
    """'edge_sam|lora_r16' -> arch, loss full, rank 16;  'edge_sam|abl_pmc' -> rank 8 (the rank-8 ablation);  '..._r4' -> rank 4."""
    arch, label = cfg_key.split("|")
    m = re.fullmatch(r"lora_r(\d+)", label)
    if m:
        return {"arch": arch, "loss": "full", "rank": int(m.group(1))}
    m = re.fullmatch(r"abl_(dice_only|pmc_bga|pmc)(?:_r(\d+))?", label)
    if m:
        return {"arch": arch, "loss": m.group(1), "rank": int(m.group(2)) if m.group(2) else 8}
    return None            # full fine-tuning and anything else is not part of this grid


def load_runs(runs_index: Path, collapse_dice: float) -> pd.DataFrame:
    df = pd.read_csv(runs_index)
    df.columns = [swap_text(c) for c in df.columns]
    df["collapsed_classes"] = df["collapsed_classes"].fillna("").astype(str).map(swap_text)
    if (df["status"] != "complete").any():
        bad = df.loc[df["status"] != "complete", "config"].tolist()
        raise RuntimeError(f"{runs_index}: runs not complete: {bad[:5]} ...")
    meta = df["config"].map(parse_config)
    keep = meta.notna()
    df = df[keep].copy()
    meta = meta[keep]
    for k in ("arch", "loss", "rank"):
        df[k] = [m[k] for m in meta]
    # collapse flag recomputed from the per-class Dice (the aggregator's n_collapsed agrees; checked below)
    df["min_class_dice"] = np.nanmin(df[[f"dice_{c}" for c in CLASSES]].to_numpy(dtype=float), axis=1)
    df["collapsed"] = df["min_class_dice"] < collapse_dice
    if not (df["collapsed"] == (df["n_collapsed"] > 0)).all():
        raise RuntimeError("collapse flag differs from the aggregator's n_collapsed")
    dup = df.duplicated(["arch", "loss", "rank", "seed"])
    if dup.any():
        raise RuntimeError(f"duplicate (arch, loss, rank, seed): {df.loc[dup, ['arch', 'loss', 'rank', 'seed']].values.tolist()}")
    return df.sort_values(["arch", "loss", "rank", "seed"]).reset_index(drop=True)


def per_run_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = ["arch", "loss", "rank", "seed", "base_run_id", "trainable_params", "n_collapsed", "collapsed_classes",
            "collapsed", "min_class_dice", "metrics_sha256"] + METRICS
    out = []
    for r in df[cols].to_dict("records"):
        out.append({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})
    return out


def summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for (arch, loss, rank), g in df.groupby(["arch", "loss", "rank"]):
        r: Dict[str, Any] = {"arch": arch, "loss": loss, "rank": int(rank), "n": int(len(g)),
                             "seeds": sorted(int(s) for s in g["seed"]),
                             "trainable_params": int(g["trainable_params"].iloc[0]),
                             "n_runs_with_a_collapsed_class": int(g["collapsed"].sum()),
                             "collapsed_seeds": sorted(int(s) for s in g.loc[g["collapsed"], "seed"])}
        for m in METRICS:
            v = g[m].dropna().to_numpy(float)
            r[f"{m}_mean"] = float(v.mean()) if len(v) else None
            r[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
        rows.append(r)
    return rows


def _paired(df: pd.DataFrame, arch: str, rank: int, a: str, b: str, metric: str) -> Optional[Dict[str, Any]]:
    """b minus a, same seed; 95% t-interval of the differences."""
    A = df[(df.arch == arch) & (df["rank"] == rank) & (df.loss == a)].set_index("seed")[metric]
    B = df[(df.arch == arch) & (df["rank"] == rank) & (df.loss == b)].set_index("seed")[metric]
    seeds = sorted(set(A.index) & set(B.index))
    if not seeds:
        return None
    d = np.array([B[s] - A[s] for s in seeds], float)
    res = {"arch": arch, "rank": rank, "a": a, "b": b, "metric": metric, "difference": "b - a", "seeds": seeds,
           "n_pairs": len(d), "mean_diff": float(d.mean()), "n_b_greater": int((d > 0).sum()),
           "ci95_lo": None, "ci95_hi": None}
    if len(d) > 1:
        h = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))
        res["ci95_lo"], res["ci95_hi"] = float(d.mean() - h), float(d.mean() + h)
    return res


def paired_tables(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    ranks = sorted(int(r) for r in df["rank"].unique())
    pairs = [("dice_only", "pmc"), ("dice_only", "pmc_bga"), ("dice_only", "full"), ("pmc_bga", "full")]
    for arch in sorted(df.arch.unique()):
        for rank in ranks:
            for a, b in pairs:
                for m in ("ipe_global_pct", "mean_dice", "aji"):
                    p = _paired(df, arch, rank, a, b, m)
                    if p:
                        out.append(p)
        # rank effect at the full objective, relative to every other rank (for the written rank rule)
        for r0 in ranks:
            for r1 in ranks:
                if r0 < r1:
                    A = df[(df.arch == arch) & (df.loss == "full") & (df["rank"] == r0)].set_index("seed")
                    B = df[(df.arch == arch) & (df.loss == "full") & (df["rank"] == r1)].set_index("seed")
                    seeds = sorted(set(A.index) & set(B.index))
                    for m in ("mean_dice", "aji", "ipe_global_pct"):
                        d = np.array([B.loc[s, m] - A.loc[s, m] for s in seeds], float)
                        if len(d) > 1:
                            h = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))
                            out.append({"arch": arch, "rank": None, "a": f"full_r{r0}", "b": f"full_r{r1}", "metric": m,
                                        "difference": "b - a", "seeds": seeds, "n_pairs": len(d), "mean_diff": float(d.mean()),
                                        "n_b_greater": int((d > 0).sum()), "ci95_lo": float(d.mean() - h),
                                        "ci95_hi": float(d.mean() + h)})
    return out


def build(runs_index: Path, aggregator_json: Path) -> Dict[str, Any]:
    """Tables from ``runs_index.csv``; the lost-class threshold is the one the aggregator used (``multiseed_results.json``)."""
    agg = json.loads(aggregator_json.read_text())
    thr = float(agg["collapse_dice_threshold"])
    df = load_runs(runs_index, thr)
    return {"runs": per_run_records(df), "summary": summary(df), "paired": paired_tables(df),
            "meta": {"source_runs_index": str(runs_index), "source_aggregator": str(aggregator_json), "n_runs": int(len(df)),
                     "collapse_dice_threshold": thr, "checkpoint": agg.get("checkpoint"),
                     "archs": sorted(df.arch.unique()), "ranks": sorted(int(r) for r in df["rank"].unique()),
                     "seeds": sorted(int(x) for x in df["seed"].unique()),
                     "losses": [l for l in LOSS_ORDER if l in set(df.loss)],
                     "class_names_by_mask_value": CLASSES,
                     "note": "class names in manuscript order (mask value 2 = spherocyte, 3 = echinocyte), D-25"}}
