#!/usr/bin/env python
"""Check that this machine can produce every result the manuscript needs, before anything long is started.

    python scripts/preflight.py

It runs in about a minute and writes nothing. Every check that fails prints the exact fix, so a missing file is
found now rather than after a run that takes hours. The exit status is non-zero if anything that would spoil a run
is wrong.

What it checks, in order:

    code        the analysis scripts are present, and are the versions that produce the current tables
    library     the scoring code computes the per-cell shape measurements, and the measurement is correct
    packages    everything the scripts import, including the optional ONNX packages
    data        the prepared images and the manifest, and that the test split is the expected size
    results     every run has its saved predictions and results, for both checkpoints
    config      the experiment grid, the checkpoints and the thresholds the tables quote
    machine     a writable results tree and a graphics card for the benchmark

A warning is something that limits one table but does not spoil the run. A failure means a run would be wasted.
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, WARN, FAIL = "pass", "warn", "FAIL"
rows: list = []


def check(group: str, what: str, state: str, detail: str = "", fix: str = "") -> None:
    rows.append({"group": group, "what": what, "state": state, "detail": detail, "fix": fix})
    mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[state]
    print(f"[{mark}] {group:9s} {what:46s} {detail}", flush=True)


def contains(path: Path, needles, what: str, group: str, fix: str) -> None:
    """A file must exist and contain every marker, which is how a stale copy is told from a current one."""
    if not path.exists():
        return check(group, what, FAIL, f"{path.relative_to(ROOT)} is missing", fix)
    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [n for n in needles if n not in text]
    if missing:
        return check(group, what, FAIL, f"{path.relative_to(ROOT)} is an older copy "
                                        f"(no {missing[0]!r})", fix)
    check(group, what, OK, str(path.relative_to(ROOT)))


def gpu_state():
    """Each card as (index, memory used in MiB, utilization %), read before torch is imported. None if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            return None
        cards = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                cards.append((int(parts[0]), float(parts[1]), float(parts[2])))
        return cards or None
    except Exception:                                             # noqa: BLE001
        return None


def main() -> int:
    print("=" * 78)
    print(f" Preflight for {ROOT}")
    print("=" * 78)

    # The cards are read first, before anything imports torch. Importing torch and asking for a device name
    # creates a CUDA context, which makes every visible card look occupied; reading first avoids blaming the
    # machine for memory this check itself allocated.
    cards = gpu_state()

    # ------------------------------------------------------------------ code
    scp = "restore it from the repository: scripts/ and qpicell/ must match"
    contains(ROOT / "scripts" / "evaluation" / "rescore.py",
             ["score_predictions", "disagreements"], "rescore.py present", "code", scp)
    contains(ROOT / "scripts" / "evaluation" / "objective_and_rank_selection.py",
             ["lora_vs_full", '"bf1"', "iou_macro"], "selection compares boundary F1 and macro IoU", "code", scp)
    contains(ROOT / "scripts" / "evaluation" / "consolidate.py",
             ["pearson_r", "lost_classes", "ipe_"], "consolidate exposes Pearson r and per-class fields",
             "code", scp)
    contains(ROOT / "scripts" / "evaluation" / "benchmark.py",
             ["onnx_export_patches", "run_with_iobinding", "ORT_SESSIONS_PER_PASS"],
             "benchmark follows the three-pass protocol with device binding", "code", scp)
    contains(ROOT / "manuscript" / "make_all.py",
             ["figtab", "BUILD_REPORT"], "the manuscript generator is present", "code", scp)
    contains(ROOT / "manuscript" / "figtab" / "figures.py",
             ['"Mean Dice"', "the fewest seeds lost a class"],
             "Fig. 5 names the quantity as the tables do and states the rank rule", "code", scp)
    contains(ROOT / "scripts" / "run_analysis.sh",
             ["manuscript/make_all.py", "7/8", "8/8"],
             "run_analysis builds the manuscript with manuscript/make_all.py", "code", scp)
    contains(ROOT / "configs" / "config.yaml",
             ["selection:", "rule: balanced", "stability: graded"],
             "the rank rule and its weights live in the configuration", "code", scp)
    contains(ROOT / "scripts" / "run_rescore.sh", ["WRITE"], "run_rescore.sh present", "code", scp)
    contains(ROOT / "scripts" / "run_benchmark.sh", ["ONNX"], "run_benchmark.sh supports ONNX", "code", scp)

    # --------------------------------------------------------------- library
    contains(ROOT / "qpicell" / "evaluation" / "cells.py", ["circularity_per_label", "perimeter_um"],
             "the scoring code records perimeter and circularity", "library",
             "restore qpicell/evaluation/cells.py from the repository")
    shape = ROOT / "qpicell" / "evaluation" / "shape.py"
    if not shape.exists():
        check("library", "shape.py present", FAIL, "qpicell/evaluation/shape.py is missing",
              "restore qpicell/evaluation/shape.py from the repository")
    else:
        check("library", "shape.py present", OK, "qpicell/evaluation/shape.py")

    # the measurement itself, on a shape whose answer is known
    try:
        import numpy as np
        from qpicell.evaluation.shape import circularity_per_label
        n = 200
        y, x = np.ogrid[:n, :n]
        lab = np.zeros((n, n), np.int32)
        lab[(y - 100) ** 2 + (x - 100) ** 2 <= 1600] = 1
        P, C = circularity_per_label(lab, 1, 0.1441)
        if 0.95 < float(C[1]) < 1.06:
            check("library", "circularity of a disc is 1", OK, f"measured {float(C[1]):.4f}")
        else:
            check("library", "circularity of a disc is 1", FAIL, f"measured {float(C[1]):.4f}, expected about 1",
                  "the shape code is wrong or a dependency changed; do not run")
    except Exception as e:                                        # noqa: BLE001
        check("library", "circularity of a disc is 1", FAIL, f"{type(e).__name__}: {e}",
              "restore qpicell/evaluation/{shape,cells}.py from the repository")

    # --------------------------------------------------------------- packages
    for mod, why, needed in (("numpy", "everything", True), ("scipy", "shape and statistics", True),
                             ("skimage", "the Crofton perimeter", True), ("pandas", "the cell tables", True),
                             ("yaml", "the configuration", True), ("tifffile", "the prepared images", True),
                             ("torch", "reading the prepared arrays and the benchmark", True),
                             ("matplotlib", "the figures", True),
                             ("onnx", "the ONNX export", False), ("onnxruntime", "timing the ONNX graph", False)):
        try:
            m = importlib.import_module(mod)
            check("packages", f"{mod} ({why})", OK, getattr(m, "__version__", ""))
        except Exception as e:                                    # noqa: BLE001
            check("packages", f"{mod} ({why})", FAIL if needed else WARN, f"{type(e).__name__}",
                  f"pip install {mod}" if needed else
                  "pip install onnx onnxruntime-gpu  (only needed for ONNX=1; the rest of the run is unaffected)")
            if mod == "torch":
                print("\n[preflight] torch is required; stopping here.\n", flush=True)
                return 2

    # ----------------------------------------------------------------- config
    from qpicell.cli import config as load_config
    from qpicell.training.runs import conditions, grid, results_dir

    class A:
        config = str(ROOT / "configs" / "config.yaml")
        smoke = False

    try:
        cfg = load_config(A())
    except Exception as e:                                        # noqa: BLE001
        check("config", "configs/config.yaml loads", FAIL, f"{type(e).__name__}: {e}", "fix the configuration")
        return 2
    arms = list(cfg["arms"]); conds = conditions(cfg); seeds = list(cfg["training"]["seeds"])
    tags = list(cfg["evaluation"]["checkpoints"])
    runs = grid(cfg)
    check("config", "experiment grid", OK,
          f"{len(runs)} runs = {len({r.arch for r in runs})} architectures x {len(conds)} conditions "
          f"x {len(arms)} losses x {len(seeds)} seeds")
    check("config", "checkpoints", OK, ", ".join(tags))
    check("config", "thresholds the tables quote", OK,
          f"within {cfg['evaluation']['within_pct']:g} %, lost class below Dice "
          f"{cfg['evaluation']['lost_class_dice']}, primary instances {cfg['evaluation']['instances_primary']}")
    check("config", "pixel pitch and wavelength", OK,
          f"{cfg['physics']['pixel_size_um']} um/px, {cfg['physics']['wavelength_um']} um")

    # ------------------------------------------------------------------- data
    prepared = Path(cfg["paths_abs"]["prepared"])
    man = prepared / "manifest.csv"
    if not man.exists():
        check("data", "prepared manifest", FAIL, f"{man} is missing", "prepare the data on this machine")
    else:
        check("data", "prepared manifest", OK, str(man.relative_to(ROOT)))
        from qpicell.physics import split_stems
        try:
            stems = split_stems(cfg)["test"]
            want = int(cfg["split"]["expected"]["test"]) if "expected" in cfg.get("split", {}) else len(stems)
            check("data", "test images", OK if len(stems) == want else FAIL,
                  f"{len(stems)} stems, expected {want}",
                  "" if len(stems) == want else "the prepared data does not match the configuration")
            for sub in ("phase", "mask", "cells", "wrapped"):
                d = prepared / "val" / sub
                have = len(list(d.glob("*.tif"))) if d.exists() else 0
                check("data", f"prepared {sub} images", OK if have >= len(stems) else FAIL,
                      f"{have} files", "" if have >= len(stems) else f"{d} is incomplete")
        except Exception as e:                                    # noqa: BLE001
            check("data", "test split", FAIL, f"{type(e).__name__}: {e}", "check the manifest and the split")

    # ---------------------------------------------------------------- results
    missing_pred, missing_metrics = [], []
    for r in runs:
        d = results_dir(cfg, r)
        for tag in tags:
            if not (d / f"predictions_{tag}.npz").exists():
                missing_pred.append(f"{r.run_id} {tag}")
            if not (d / f"metrics_{tag}.json").exists():
                missing_metrics.append(f"{r.run_id} {tag}")
    n = len(runs) * len(tags)
    check("results", "saved predictions", OK if not missing_pred else FAIL,
          f"{n - len(missing_pred)} of {n}",
          "" if not missing_pred else f"missing, first {missing_pred[0]}: the rescore cannot cover these")
    check("results", "stored results to check against", OK if not missing_metrics else FAIL,
          f"{n - len(missing_metrics)} of {n}",
          "" if not missing_metrics else f"missing, first {missing_metrics[0]}")

    # ---------------------------------------------------------------- machine
    res = Path(cfg["paths_abs"]["results"])
    try:
        probe = res / ".preflight_write_test"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("x", encoding="utf-8"); probe.unlink()
        check("machine", "results tree is writable", OK, str(res))
    except Exception as e:                                        # noqa: BLE001
        check("machine", "results tree is writable", FAIL, f"{type(e).__name__}", f"fix permissions on {res}")
    free = shutil.disk_usage(res if res.exists() else ROOT).free / 1024 ** 3
    check("machine", "free disk space", OK if free > 5 else WARN, f"{free:.1f} GB",
          "" if free > 5 else "the rescore rewrites the cell tables in place; a few GB of headroom is wise")
    try:
        import torch
        if torch.cuda.is_available():
            check("machine", "graphics card for the benchmark", OK,
                  f"{torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
        else:
            check("machine", "graphics card for the benchmark", WARN, "none visible to torch",
                  "the rescore and the tables do not need one; the benchmark would fall back to the processor")
    except Exception:                                             # noqa: BLE001
        check("machine", "graphics card for the benchmark", WARN, "could not be determined")
    if cards is None:
        check("machine", "graphics cards idle for the benchmark", WARN, "nvidia-smi gave no reading")
    else:
        # a card doing nothing still reports a megabyte or two; real work shows as hundreds, or as utilisation
        busy = [c for c in cards if c[1] > 64 or c[2] > 5]
        idle = [c[0] for c in cards if c not in busy]
        if not busy:
            check("machine", "graphics cards idle for the benchmark", OK, f"{len(cards)} idle")
        elif idle:
            check("machine", "graphics cards idle for the benchmark", WARN,
                  f"{len(busy)} of {len(cards)} in use, idle: {', '.join(str(i) for i in idle)}",
                  f"time the benchmark on an idle card: GPU={idle[0]} ONNX=1 bash scripts/run_benchmark.sh")
        else:
            check("machine", "graphics cards idle for the benchmark", WARN,
                  f"all {len(cards)} in use ({max(c[1] for c in cards):.0f} MiB, "
                  f"{max(c[2] for c in cards):.0f} % at the busiest)",
                  "latencies timed now would be inflated; re-run the benchmark when a card is free")

    # ----------------------------------------------------------------- verdict
    fails = [r for r in rows if r["state"] == FAIL]
    warns = [r for r in rows if r["state"] == WARN]
    print("\n" + "=" * 78)
    if fails:
        print(f" NOT READY: {len(fails)} check(s) failed. Fix these first, or the run is wasted.\n")
        for r in fails:
            print(f"   {r['group']}: {r['what']}")
            print(f"       {r['detail']}")
            if r["fix"]:
                print(f"       -> {r['fix']}")
        print("\n" + "=" * 78)
        return 1
    print(f" READY. {len(rows) - len(warns)} checks passed"
          f"{f', {len(warns)} warning(s)' if warns else ''}.\n")
    for r in warns:
        print(f"   warning - {r['what']}: {r['detail']}")
        if r["fix"]:
            print(f"       {r['fix']}")
    print("\n Next:  bash scripts/run_everything.sh")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
