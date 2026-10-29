"""Scale guard for the corrected maps and the run-by-run comparison with old_run/ (D-20)."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]


def _maps(root: Path, scale: float):
    d = root / "raw_data"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(6):
        a = np.clip(rng.random((32, 32)) * math.pi * scale, 0, math.pi * scale).astype(np.float32)
        a[0, 0] = math.pi * scale * 0.999
        tifffile.imwrite(d / f"2011051{i}_1.tif", a)
    tifffile.imwrite(d / "X_not_a_date.tif", np.zeros((4, 4), np.float32))


def _run(args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / args[0]), *args[1:]], capture_output=True, text=True)


def test_scale_guard_rejects_half_scale_and_accepts_corrected(tmp_path):
    _maps(tmp_path / "old", 1.0)
    _maps(tmp_path / "new", 2.0)
    assert _run(["check_raw_scale.py", "--raw-root", str(tmp_path / "old")]).returncode == 1
    r = _run(["check_raw_scale.py", "--raw-root", str(tmp_path / "new")])
    assert r.returncode == 0 and "OK" in r.stdout


def _metrics(path: Path, dice, echi=0.8):
    path.mkdir(parents=True)
    ev = {"mean_dice": dice, "aji": 0.3, "bf1": 0.6, "ipe_global": 0.03, "dice_discocyte": 0.9,
          "dice_echinocyte": echi, "dice_spherocyte": 0.6, "dice_stomatocyte": 0.6}
    (path / "metrics.json").write_text(json.dumps({"status": "complete", "evaluation": {"last": ev}}))


def test_comparison_with_old_run(tmp_path):
    # the script resolves the project root from the configured output root, so run it on a copy of the config tree
    import shutil
    proj = tmp_path / "p"
    shutil.copytree(ROOT / "configs", proj / "configs")
    shutil.copytree(ROOT / "qpi", proj / "qpi", ignore=shutil.ignore_patterns("__pycache__"))
    (proj / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "compare_with_old_run.py", proj / "scripts")
    for rid, od, nd, ne in (("edge_sam_abl_dice_only", 0.737, 0.7371, 0.77), ("edge_sam_lora_r8", 0.715, 0.70, 0.0)):
        _metrics(proj / "old_run/outputs/full/runs" / rid, od)
        _metrics(proj / "outputs/full/runs" / rid, nd, ne)
    r = subprocess.run([sys.executable, str(proj / "scripts/compare_with_old_run.py")], capture_output=True, text=True, cwd=proj)
    assert r.returncode == 0, r.stderr
    md = (proj / "outputs/full/analysis/comparison_with_old_run.md").read_text()
    assert "Dice-only runs (control)" in md and "edge_sam_lora_r8" in md and "| 0/1 |" in md
