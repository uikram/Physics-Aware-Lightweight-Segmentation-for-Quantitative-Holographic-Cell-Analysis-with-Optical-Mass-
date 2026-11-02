#!/usr/bin/env python
"""Per-cell analyses for every architecture, LoRA rank and seed (read-only on the runs; no training, no GPU).

    python scripts/run_cell_analysis.py [--archs ...] [--ranks ...] [--seeds ...] [--losses ...] [--jobs N]

Default scope: the three architectures x ranks 2, 4, 8, 16, 32 x seeds 42, 43, 44 with the full objective (45 runs). Seed 42 is
read from ``outputs/full``, the other seeds from ``outputs/multiseed``. Add ``--losses full dice_only pmc pmc_bga`` for the loss
ablations and ``--include-full-finetune`` for the single full fine-tuning run per architecture. Nothing is chosen here: every
requested run is analysed and stored, the choice of the manuscript configuration is the author's.

Resumable (D-29): every finished run is written at once to ``outputs/analysis/cell/<run id>.json`` (an individual result with
provenance); a rerun skips the runs whose stored result is still valid (same predictions file hash, checkpoint, dataset manifest)
and only computes the rest. ``--force`` recomputes everything, ``--all`` means every loss and the full fine-tuning runs.
Nothing is deleted. Progress is printed per run, ``COMPLETE`` is the last line only if every requested run succeeded.

Writes ``outputs/cell_analysis/`` (the legacy files the older scripts read; rebuilt from the individual results on every run):
  cell_analysis.json                    everything: meta, annotation, one entry per run, mean +- SD over seeds per configuration
  cell_analysis_runs.csv                one row of scalars per run
  cell_analysis_by_configuration.csv    mean, SD, min, max over seeds per configuration and metric
  annotation/cells_gt.csv, runs/<run id>/cells_pred.csv, phase_integral/<run id>__per_{image,cell}.csv
Class names follow the manuscript order (D-25). Exit status 1 if a requested run is missing or unfinished (unless
--allow-missing), if a run failed, or if a stored earlier result disagrees with the number computed here (the reference check).
The multi-seed file over ALL individual results is built by scripts/compile_results.py cell.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from qpi.analysis import cell_analysis as ca  # noqa: E402
from qpi.analysis import wrapping as wrp  # noqa: E402
from qpi.analysis.common import load_validation  # noqa: E402
from qpi.config import load_config  # noqa: E402
from qpi.utils.classnames import MANUSCRIPT_NAMES  # noqa: E402
from qpi.utils.io import read_json, sha256_file, write_json  # noqa: E402
from qpi.utils.layout import analysis_dir  # noqa: E402
from qpi.utils.progress import Progress  # noqa: E402
from qpi.utils.outputs import mode_of  # noqa: E402

README = """cell_analysis: per-cell analyses of every requested run, computed from the saved predictions (no training).
Class names are in the MANUSCRIPT order (D-25): mask value 2 = spherocyte, mask value 3 = echinocyte. Nothing here is to be
swapped again (manuscript/results.py reads cell_analysis.json without the swap).
Statistics over seeds: mean and sample SD (n - 1) over the seeds present (3). Cells flagged as phase-wrapped (D-21) are
excluded from the *_nowrap quantities; the *_all quantities keep them.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=None, help="default: all architectures of the configuration")
    ap.add_argument("--ranks", type=int, nargs="+", default=None, help="default: lora.ranks of the configuration")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--losses", nargs="+", default=["full"], choices=list(ca.LOSSES))
    ap.add_argument("--include-full-finetune", action="store_true")
    ap.add_argument("--multiseed-root", default="outputs/multiseed")
    ap.add_argument("--out", default="outputs/cell_analysis")
    ap.add_argument("--jobs", type=int, default=1, help="runs analysed in parallel (CPU processes)")
    ap.add_argument("--no-csv", action="store_true", help="do not write the per-run cell tables")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--all", action="store_true", help="every loss (full, dice_only, pmc, pmc_bga) and the full fine-tuning runs")
    ap.add_argument("--force", action="store_true", help="recompute runs whose stored result is still valid")
    ap.add_argument("--store", default=None, help="folder of the individual results (default outputs/analysis/cell)")
    ap.add_argument("--trust-missing-sources", action="store_true",
                    help="accept a stored result when the run folder is not on this machine (cannot verify the hash)")
    a = ap.parse_args()
    if a.all:
        a.losses, a.include_full_finetune = list(ca.LOSSES), True

    cfg = load_config(None)
    if mode_of(cfg) != "full":
        print("run_cell_analysis.py: use the full configuration (no overlay)", file=sys.stderr)
        return 1
    main_root = Path(cfg["paths_abs"]["output_root"])
    ms_root = Path(a.multiseed_root) if Path(a.multiseed_root).is_absolute() else ROOT / a.multiseed_root
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    tag = cfg["evaluation"]["checkpoint"]
    archs = a.archs or list(cfg["architectures"])
    for x in archs:
        if x not in cfg["architectures"]:
            ap.error(f"unknown architecture {x!r}; choose from {list(cfg['architectures'])}")
    ranks = a.ranks or [int(r) for r in cfg["lora"]["ranks"]]
    refs = ca.expected_runs(cfg, main_root, ms_root, archs, ranks, a.seeds, a.losses, a.include_full_finetune)
    print(f"[cell_analysis] {len(refs)} runs requested: archs {archs}, ranks {ranks}, seeds {a.seeds}, losses {a.losses}"
          f"{', full fine-tuning' if a.include_full_finetune else ''}; checkpoint '{tag}'", flush=True)

    cfg_m = ca.manuscript_cfg(cfg)
    manifest = Path(cfg["paths_abs"]["processed_root"]) / "manifest.csv"
    manifest_sha = sha256_file(manifest) if manifest.exists() else None
    store = Path(a.store) if a.store else analysis_dir("cell", ROOT)
    out.mkdir(parents=True, exist_ok=True)
    prog = Progress("cell_analysis", len(refs))
    holder = {}

    def load_val():
        holder["val"] = load_validation(cfg_m)
        print(f"[cell_analysis] {len(holder['val'])} validation images", flush=True)
        return holder["val"]

    entries, cells_gt, val = ca.run_units(cfg_m, refs, tag, out, store, manifest_sha, load_val, prog, jobs=a.jobs,
                                          write_csv=not a.no_csv, force=a.force, allow_missing=a.allow_missing,
                                          trust_missing_sources=a.trust_missing_sources)

    # the annotation block is the same for every run: cached next to the individual results
    prog.stage("annotation and legacy summary files")
    ann_path = store / "_annotation.json"
    cache = read_json(ann_path) if ann_path.exists() else None
    if cache and cache.get("manifest_sha256") == manifest_sha and cache.get("schema_version") == ca.SCHEMA_VERSION and val is None:
        annotation, n_val = cache["annotation"], cache["n_validation_images"]
        if not (out / "annotation" / "cells_gt.csv").exists() and not a.no_csv:
            print("[cell_analysis] note: annotation/cells_gt.csv is not on this machine; the annotation block is read from the cache")
    else:
        val = val if val is not None else load_val()
        cells_gt = cells_gt if cells_gt is not None else ca.dm.cells_table(cfg_m, val, None)
        annotation, n_val = ca.annotation_block(cfg_m, cells_gt, val), len(val)
        write_json(ann_path, {"schema_version": ca.SCHEMA_VERSION, "manifest_sha256": manifest_sha,
                              "n_validation_images": n_val, "annotation": annotation})
    (out / "annotation").mkdir(exist_ok=True)
    if cells_gt is not None and not a.no_csv:
        cells_gt.to_csv(out / "annotation" / "cells_gt.csv", index=False)

    done = [e for e in entries if e["status"] == "complete"]
    problems = [{"run_id": e["run_id"], "status": e["status"], "reason": e.get("reason")} for e in entries if e["status"] != "complete"]
    by_cfg = ca.aggregate(done)
    checks = ca.reference_checks(main_root, done, int(cfg["seed"]), cfg["analysis"]["selected_arch"],
                                 int(cfg["analysis"]["selected_rank"]))
    bad = [c for c in checks if not c["ok"]]

    doc = {
        "meta": {
            "schema_version": ca.SCHEMA_VERSION, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checkpoint": tag, "n_validation_images": n_val,
            "class_names_by_mask_value": MANUSCRIPT_NAMES,
            "class_name_note": "manuscript order (D-25): mask value 2 = spherocyte, 3 = echinocyte; written directly, do not swap again",
            "scope": {"archs": archs, "ranks": ranks, "seeds": a.seeds, "losses": a.losses,
                      "include_full_finetune": bool(a.include_full_finetune)},
            "n_runs_requested": len(refs), "n_runs_analysed": len(done), "complete": not problems, "problems": problems,
            "wrapping_rule": wrp.rule(cfg), "physics": cfg["physics"],
            "phase_reference": cfg["data"]["phase_reference"],
            "manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
            "statistics": "mean and sample SD (n - 1) over seeds; *_nowrap excludes phase-wrapped cells (D-21)",
            "reference_check": {"n_compared": len(checks), "n_differences": len(bad), "differences": bad,
                                "note": "values recomputed here against analysis/drymass.json, phase_integral.json and "
                                        "architecture_comparison.json of outputs/full (seed-42 runs); a difference means that "
                                        "earlier file is out of date"},
        },
        "annotation": annotation,
        "runs": {e["run_id"]: e for e in entries},
        "by_configuration": by_cfg,
    }
    write_json(out / "cell_analysis.json", doc)
    pd.DataFrame([ca.flatten(e) for e in done]).to_csv(out / "cell_analysis_runs.csv", index=False)
    rows = [{"config": k, "arch": v["arch"], "lora_rank": v["lora_rank"], "loss": v["loss"], "n_seeds": v["n_seeds"], "metric": m,
             **{s: d[s] for s in ("mean", "sd", "min", "max")}} for k, v in by_cfg.items() for m, d in v["metrics"].items()]
    pd.DataFrame(rows).to_csv(out / "cell_analysis_by_configuration.csv", index=False)
    (out / "README.txt").write_text(README)

    show = ["mean_dice", "gt_cells_matched_pct", "percell_abs_median_pct", "mdare_pct", "cells_predicted_nowrap"]
    print("\nmean over seeds (SD in brackets) per configuration:")
    for k, v in by_cfg.items():
        cells = "  ".join(f"{m}={v['metrics'][m]['mean']:.3g}" + (f" ({v['metrics'][m]['sd']:.2g})" if v['metrics'][m]['sd'] is not None else "")
                          for m in show if v["metrics"][m]["mean"] is not None)
        print(f"  {k:<28} n={v['n_seeds']}  {cells}")
    print(f"\n[cell_analysis] reference check: {len(checks)} values compared against earlier analysis files, {len(bad)} differ")
    for c in bad[:10]:
        print(f"  DIFFERS {c['source']} {c['run_id']} {c['key']}: earlier {c['expected']} now {c['got']}")
    print(f"[cell_analysis] wrote {out / 'cell_analysis.json'}: {len(done)} of {len(refs)} runs; individual results in {store}")
    for p in problems:
        print(f"[cell_analysis] NOT ANALYSED {p['run_id']}: {p['reason']}", file=sys.stderr)
    return prog.finish(extra_ok=not bad, extra_message=f"{len(bad)} values differ from the earlier analysis files (reference check)")


if __name__ == "__main__":
    sys.exit(main())
