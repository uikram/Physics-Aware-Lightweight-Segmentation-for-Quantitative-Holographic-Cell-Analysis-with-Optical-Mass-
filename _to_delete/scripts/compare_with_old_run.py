#!/usr/bin/env python
"""Run-by-run comparison of the current results with the superseded half-scale runs kept in old_run/ (D-20).

Writes <output_root>/analysis/comparison_with_old_run.{csv,md}. Same run id = same configuration and seed; the only
difference is the date-named phase maps (2x scale). Expectation (D-20): Dice-only runs and the segmentation inputs are
unchanged (z-scored input), so differences there are floating-point/GPU nondeterminism; runs with PMC / BGA can differ
because the margin and the gradient term act on raw radians.

    python scripts/compare_with_old_run.py [--old old_run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qpi.config import load_config  # noqa: E402

FIELDS = (("mean_dice", "mean_dice", 1.0), ("aji", "aji", 1.0), ("bf1", "bf1", 1.0), ("ipe_pct", "ipe_global", 100.0))


def rows_for(root: Path, label: str) -> dict:
    out = {}
    for m in sorted((root / "runs").glob("*/metrics.json")):
        j = json.loads(m.read_text())
        if j.get("status") != "complete" or "evaluation" not in j:
            continue
        ev = j["evaluation"]["last"]
        out[m.parent.name] = {k: ev[src] * mult for k, src, mult in FIELDS}
        out[m.parent.name]["classes_lost"] = sum(1 for c in ("discocyte", "echinocyte", "spherocyte", "stomatocyte")
                                                 if ev.get(f"dice_{c}", 1.0) < 0.05)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="old_run")
    a = ap.parse_args()
    cfg = load_config()
    new_root = Path(cfg["paths_abs"]["output_root"])
    rootdir = Path(cfg["paths_abs"]["output_root"]).parents[1]
    recs = []
    for name in ("full", "multiseed"):
        old_root, nr = rootdir / a.old / "outputs" / name, rootdir / "outputs" / name
        if not old_root.exists() or not nr.exists():
            print(f"[compare] skip {name}: {'old' if not old_root.exists() else 'new'} output missing")
            continue
        old, new = rows_for(old_root, "old"), rows_for(nr, "new")
        for rid in sorted(set(old) & set(new)):
            r = {"series": name, "run_id": rid}
            for k, _, _ in FIELDS:
                r[f"{k}_old"], r[f"{k}_new"], r[f"{k}_diff"] = old[rid][k], new[rid][k], new[rid][k] - old[rid][k]
            r["classes_lost_old"], r["classes_lost_new"] = old[rid]["classes_lost"], new[rid]["classes_lost"]
            recs.append(r)
    if not recs:
        print("[compare] nothing to compare")
        return 0
    df = pd.DataFrame(recs)
    out = new_root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / "comparison_with_old_run.csv", index=False)
    lines = ["# Current runs against the superseded half-scale runs (old_run/)", "",
             "Positive difference = current minus old. Dice-only runs are the control (segmentation input unchanged).", ""]
    df["dice_only"] = df.run_id.str.contains("abl_dice_only")
    for title, g in (("Dice-only runs (control)", df[df.dice_only]), ("Runs with PMC / BGA / IPP", df[~df.dice_only])):
        if len(g):
            lines += [f"## {title}", "", f"{len(g)} runs. Median absolute difference: "
                      f"Dice {g.mean_dice_diff.abs().median():.4f}, AJI {g.aji_diff.abs().median():.4f}, "
                      f"BF1 {g.bf1_diff.abs().median():.4f}, IPE {g.ipe_pct_diff.abs().median():.3f} pp; "
                      f"largest Dice difference {g.mean_dice_diff.abs().max():.4f}; "
                      f"runs whose number of lost classes changed: {int((g.classes_lost_old != g.classes_lost_new).sum())}.", ""]
    lines += ["## Per run", "", "| series | run | Dice old | Dice new | AJI old | AJI new | BF1 old | BF1 new | IPE old (%) | IPE new (%) | classes lost old/new |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.series} | {r.run_id} | {r.mean_dice_old:.3f} | {r.mean_dice_new:.3f} | {r.aji_old:.3f} | {r.aji_new:.3f} | "
                     f"{r.bf1_old:.3f} | {r.bf1_new:.3f} | {r.ipe_pct_old:.2f} | {r.ipe_pct_new:.2f} | {r.classes_lost_old}/{r.classes_lost_new} |")
    (out / "comparison_with_old_run.md").write_text("\n".join(lines) + "\n")
    print(f"[compare] {len(df)} runs compared -> {out / 'comparison_with_old_run.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
