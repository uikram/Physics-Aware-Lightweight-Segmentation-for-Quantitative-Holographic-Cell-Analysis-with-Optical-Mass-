"""Storage-time statistics on synthetic per-image data (no real data needed)."""
import numpy as np
import pandas as pd

from qpicell.evaluation.storage import (exact_block_permutation_p, monte_carlo_block_permutation_p, per_image, trend)


DAYS = np.array([0, 2, 5, 8, 13, 16, 20, 27, 37, 41, 48])   # 11 storage days, as in the test images


def _images(days, slope, noise, seed=0):
    rng = np.random.default_rng(seed)
    d = np.repeat(days, 6)
    return pd.DataFrame({"storage_day": d, "dry_mass": 30 + slope * d + noise * rng.standard_normal(len(d))})


def test_monotone_trend_is_detected(cfg):
    img = _images(DAYS, slope=-0.1, noise=0.0)
    t = trend(img, "dry_mass", cfg, exact=True)
    assert t["rho"] < -0.9 and t["n_days"] == 11 and t["n_images"] == 66
    assert t["permutation_method"] == "exact_11!"
    assert abs(t["p_block_permutation"] - 2 / 39916800) < 1e-9   # only the two monotone orderings are as extreme
    assert t["early_to_late_change_pct"] < 0


def test_exact_and_monte_carlo_agree():
    img = _images(DAYS, slope=-0.03, noise=1.0, seed=3)
    pe = exact_block_permutation_p(img.storage_day.to_numpy(), img.dry_mass.to_numpy())
    pm = monte_carlo_block_permutation_p(img.storage_day.to_numpy(), img.dry_mass.to_numpy(), n=50000)
    assert 0 < pe <= 1 and abs(pe - pm) < 0.02
    assert np.isnan(exact_block_permutation_p(DAYS[:7].repeat(6), np.arange(42.0)))   # exact only for 11 equal blocks


def test_constant_values_not_computed(cfg):
    img = pd.DataFrame({"storage_day": np.repeat([0, 5, 10], 6), "dry_mass": 1.0})
    assert np.isnan(trend(img, "dry_mass", cfg)["rho"])


def test_per_image_medians():
    cells = pd.DataFrame({"stem": ["a", "a", "a", "b"], "storage_day": [0, 0, 0, 5],
                          "area_um2": [10.0, 20.0, 30.0, 40.0], "dm": [1.0, 2.0, 9.0, 4.0]})
    p = per_image(cells, "area_um2", "dm").set_index("stem")
    assert p.loc["a", "area"] == 20.0 and p.loc["a", "dry_mass"] == 2.0 and p.loc["b", "storage_day"] == 5
