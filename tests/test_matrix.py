"""Experiment matrix: every architecture x condition x loss x seed, seed-major order, folders, learning rates, eta = 0."""
import pytest

from qpicell.losses.objective import Objective
from qpicell.training.runs import Run, conditions, grid, learning_rate, parse, results_dir, training_dir


def test_matrix_complete_and_ordered(cfg):
    runs = grid(cfg)
    assert len(runs) == 315 and len({r.run_id for r in runs}) == 315
    assert conditions(cfg) == ["r8", "r16", "r32", "r64", "fullft"]
    assert sum(r.full_finetune for r in runs) == 63
    assert {(r.arch, r.condition, r.arm, r.seed) for r in runs} == {
        (a, c, m, s) for a in cfg["models"] for c in conditions(cfg) for m in cfg["arms"] for s in (42, 43, 44)}
    seeds = [r.seed for r in runs]
    assert seeds == sorted(seeds)                                  # seed 42 first, then 43, then 44
    assert runs[0].arch == "mobile_sam" and runs[0].condition == "fullft"


def test_run_identity_and_folders(cfg):
    r = Run("edge_sam", "r32", "dice_pmc_bga_cipp", 43)
    assert parse(r.run_id, cfg) == r and r.rank == 32 and not r.full_finetune
    f = Run("mobilenet_unet", "fullft", "dice", 44)
    assert parse(f.run_id, cfg) == f and f.rank is None and f.full_finetune
    assert training_dir(cfg, r).parts[-4:] == ("EdgeSAM", "r32", "dice_pmc_bga_cipp", "seed43")
    assert results_dir(cfg, f).parts[-4:] == ("MobileNet_UNet", "fullft", "dice", "seed44")


def test_learning_rates(cfg):
    assert learning_rate(cfg, Run("edge_sam", "r8", "dice", 42)) == 5e-4
    assert learning_rate(cfg, Run("mobilenet_unet", "r8", "dice", 42)) == 1e-3
    for a in cfg["models"]:
        assert learning_rate(cfg, Run(a, "fullft", "dice", 42)) == 1e-4


def test_eta_zero_enforced(cfg):
    assert cfg["loss"]["cipp"]["eta"] == 0.0
    import copy
    c = copy.deepcopy(cfg); c["loss"]["cipp"]["eta"] = 1.0
    with pytest.raises(ValueError):
        Objective(c, "dice_cipp")
