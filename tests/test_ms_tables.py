"""Multi-seed tables for the manuscript are built from the aggregator output with the manuscript class names."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manuscript"))
import ms_tables  # noqa: E402


def _write(tmp_path, lost_seed=44):
    rows = []
    for seed in (42, 43, 44):
        for cfg in ("edge_sam|lora_r8", "edge_sam|abl_dice_only", "edge_sam|abl_pmc_r4", "edge_sam|full_finetune"):
            lost = seed == lost_seed and cfg == "edge_sam|lora_r8"
            rows.append({"block": "b", "config": cfg, "base_run_id": "x", "seed": seed, "run_id": "r", "origin": "o",
                         "metrics_path": "p", "metrics_sha256": "h", "status": "complete", "epochs_completed": 50,
                         "train_seconds": 1.0, "trainable_params": 5, "mean_dice": 0.7, "aji": 0.3, "bf1": 0.6,
                         "ipe_global_pct": 3.0, "dice_discocyte": 0.9,
                         "dice_echinocyte": 0.0 if lost else 0.8,      # legacy name = mask value 2
                         "dice_spherocyte": 0.6, "dice_stomatocyte": 0.5, "ipe_discocyte_pct": 1.0, "ipe_echinocyte_pct": 1.0,
                         "ipe_spherocyte_pct": 1.0, "ipe_stomatocyte_pct": 1.0, "n_collapsed": int(lost),
                         "collapsed_classes": "echinocyte" if lost else ""})
    idx = tmp_path / "runs_index.csv"
    pd.DataFrame(rows).to_csv(idx, index=False)
    agg = tmp_path / "multiseed_results.json"
    agg.write_text(json.dumps({"collapse_dice_threshold": 0.05, "checkpoint": "last"}))
    return idx, agg


def test_build_swaps_names_and_reads_threshold(tmp_path):
    idx, agg = _write(tmp_path)
    out = ms_tables.build(idx, agg)
    assert out["meta"]["collapse_dice_threshold"] == 0.05 and out["meta"]["seeds"] == [42, 43, 44]
    assert out["meta"]["ranks"] == [4, 8] and out["meta"]["losses"] == ["dice_only", "pmc", "full"]
    lost = [r for r in out["runs"] if r["collapsed"]]
    assert len(lost) == 1 and lost[0]["seed"] == 44 and lost[0]["collapsed_classes"] == "spherocyte"
    assert lost[0]["dice_spherocyte"] == 0.0 and lost[0]["dice_echinocyte"] == 0.6      # the legacy values, new names
    assert all(r["arch"] == "edge_sam" for r in out["runs"])                              # full fine-tuning is not in this grid
    s = next(x for x in out["summary"] if x["loss"] == "full" and x["rank"] == 8)
    assert s["n"] == 3 and s["n_runs_with_a_collapsed_class"] == 1 and s["collapsed_seeds"] == [44]
