#!/usr/bin/env python
"""Robustness to constant phase offsets and low-order phase aberrations for any set of runs (inference only; a GPU is used).

    python scripts/run_robustness.py [--scope selected|all] [--archs ...] [--ranks ...] [--seeds ...] [--losses ...] [--gpu 0]

``--scope selected`` (default) is the set of the manuscript analysis: the rank-8 full objective and the rank-8 Dice-only run of
every architecture, seed 42 (6 runs). ``--scope all`` is every LoRA run (3 architectures x 5 ranks x 4 losses x 3 seeds) plus the
single full fine-tuning run per architecture. Explicit --archs/--ranks/--seeds/--losses override the scope.

Every finished run is written at once to ``outputs/analysis/robustness/<run id>.json`` (13 distortions, change of macro Dice against
the undistorted map, flag ``invariant_by_construction`` on the constant-offset rows). A rerun skips runs whose result is current
(same checkpoint file hash). When the six runs of the manuscript analysis are present, ``outputs/full/analysis/robustness.{json,csv}``
(the files the older scripts read) are rebuilt from them. The multi-seed file is built by scripts/compile_results.py robustness.
Progress is printed per run; ``COMPLETE`` is the last line only if every run succeeded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpi.analysis import cell_analysis as ca  # noqa: E402
from qpi.analysis import robustness as rb  # noqa: E402
from qpi.config import load_config  # noqa: E402
from qpi.utils.layout import analysis_dir  # noqa: E402
from qpi.utils.outputs import mode_of  # noqa: E402
from qpi.utils.progress import Progress  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["selected", "all"], default="selected")
    ap.add_argument("--archs", nargs="+", default=None)
    ap.add_argument("--ranks", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--losses", nargs="+", default=None, choices=list(ca.LOSSES))
    ap.add_argument("--include-full-finetune", action="store_true")
    ap.add_argument("--multiseed-root", default="outputs/multiseed")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--trust-missing-sources", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N runs (to time one run before launching all)")
    a = ap.parse_args()

    cfg = load_config(None)
    if mode_of(cfg) != "full":
        print("run_robustness.py: use the full configuration (no overlay)", file=sys.stderr)
        return 1
    main_root = Path(cfg["paths_abs"]["output_root"])
    ms_root = ROOT / a.multiseed_root
    sel = int(cfg["lora"]["selected_rank"])
    if a.scope == "all":
        archs, ranks = a.archs or list(cfg["architectures"]), a.ranks or [int(r) for r in cfg["lora"]["ranks"]]
        seeds, losses = a.seeds or [42, 43, 44], a.losses or list(ca.LOSSES)
        ft = True
    else:
        archs, ranks = a.archs or list(cfg["architectures"]), a.ranks or [sel]
        seeds, losses = a.seeds or [int(cfg["seed"])], a.losses or ["full", "dice_only"]
        ft = a.include_full_finetune
    ft = ft or a.include_full_finetune
    refs = ca.expected_runs(cfg, main_root, ms_root, archs, ranks, seeds, losses, ft)
    if a.limit:
        refs = refs[: a.limit]
    tag = cfg["evaluation"]["checkpoint"]
    print(f"[robustness] {len(refs)} runs: archs {archs}, ranks {ranks}, seeds {seeds}, losses {losses}"
          f"{', full fine-tuning' if ft else ''}; checkpoint '{tag}'", flush=True)

    from qpi.utils.env import enforce_gpu_policy, resolve_device
    enforce_gpu_policy(cfg["hardware"]["allowed_physical_gpus"])
    device = resolve_device(a.gpu)
    store = analysis_dir("robustness", ROOT)
    prog = Progress("robustness", len(refs))
    prog.stage("distortions")
    rb.run_robustness_units(cfg, refs, tag, device, cfg["paths_abs"]["weights_dir"], store, prog, ca.unit_label,
                            force=a.force, trust_missing=a.trust_missing_sources)
    ids = [f"{x}_lora_r{sel}" for x in cfg["architectures"]] + [f"{x}_abl_dice_only" for x in cfg["architectures"]]
    if rb.write_legacy_robustness(store, ids, tag, main_root / "analysis"):
        print(f"[robustness] rebuilt {main_root / 'analysis' / 'robustness.json'} from the individual results", flush=True)
    return prog.finish()


if __name__ == "__main__":
    sys.exit(main())
