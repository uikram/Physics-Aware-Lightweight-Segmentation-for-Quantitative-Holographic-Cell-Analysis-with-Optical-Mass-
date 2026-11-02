"""Revision artifacts (artifacts/code): class labels, statistics helpers, and the full verification on the real outputs.

The last test runs artifacts/code/verify_artifacts.py and is skipped when outputs/compiled is absent (e.g. on a copy without results).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts" / "code"))

import common as C  # noqa: E402


def test_mask_values_are_the_confirmed_labels():
    assert C.MASK_VALUE == {"discocyte": 1, "spherocyte": 2, "echinocyte": 3, "stomatocyte": 4}
    assert C.CLASS_DISPLAY["spherocyte"] == "spherocyte (2)" and C.CLASS_SHORT["echinocyte"] == "echino(3)"


def test_run_ids_and_directories():
    assert C.run_id("edge_sam", 8, "full", 42) == "edge_sam_lora_r8"
    assert C.run_id("edge_sam", 8, "pmc", 43) == "edge_sam_abl_pmc__s43"
    # ablations at ranks other than 8 exist only in outputs/multiseed, seed 42 included
    assert C.run_id("mobile_sam", 16, "dice_only", 42) == "mobile_sam_abl_dice_only_r16__s42"
    assert C.run_dir("mobile_sam", 16, "dice_only", 42).parent.parent.name == "multiseed"
    assert C.run_dir("edge_sam", None, "full_finetune", 42).name == "edge_sam_full_finetune"
    assert C.config_key("edge_sam", 32, "pmc_bga") == "edge_sam|abl_pmc_bga_r32"


def test_paired_difference_interval():
    p = C.paired_diff({"42": 1.0, "43": 2.0, "44": 3.0}, {"42": 2.0, "43": 3.0, "44": 4.0})
    assert p["mean"] == pytest.approx(1.0) and p["lo"] == pytest.approx(1.0) and p["hi"] == pytest.approx(1.0)
    q = C.paired_diff({"42": 0.0, "43": 0.0, "44": 0.0}, {"42": 1.0, "43": -1.0, "44": 0.5})
    assert q["excludes_zero"] is False and q["n_b_greater"] == 2


@pytest.mark.skipif(not (ROOT / "outputs" / "compiled" / "cell_analysis_multiseed.json").exists(), reason="outputs not present")
def test_verify_artifacts_on_real_outputs(capsys):
    import verify_artifacts
    assert verify_artifacts.main() == 0
    assert capsys.readouterr().out.rstrip().endswith("OK")
