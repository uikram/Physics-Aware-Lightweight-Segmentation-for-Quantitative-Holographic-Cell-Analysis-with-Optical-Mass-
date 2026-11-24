"""Background reference, cell instances, registration helpers (synthetic images)."""
import numpy as np

from qpicell.data.background import design_matrix, estimate_background
from qpicell.data.cells import contact_lines, split_cells, territories


def _image(seed=0, N=256):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N, 0:N] / N
    bg = 0.3 + 0.8 * xx - 0.4 * yy + 0.6 * xx * yy - 0.5 * yy ** 2
    cells = np.zeros((N, N))
    for cy, cx in rng.integers(20, N - 20, (25, 2)):
        r2 = (np.arange(N)[:, None] - cy) ** 2 + (np.arange(N)[None] - cx) ** 2
        cells = np.maximum(cells, np.clip(1 - r2 / 100.0, 0, None) * 2.0)
    return bg + cells + rng.normal(0, 0.02, (N, N)), bg


def test_background_recovers_surface():
    img, bg = _image()
    surf, info = estimate_background(img, 2)
    assert np.median(np.abs(surf - bg)) < 0.02


def test_background_equivariant_to_constant():
    img, _ = _image(1)
    a, _ = estimate_background(img, 2); b, _ = estimate_background(img + 1.7, 2)
    assert np.allclose(b - a, 1.7, atol=1e-6)


def test_design_matrix_columns():
    assert design_matrix((8, 8), 2).shape == (64, 6)


def test_split_touching_cells():
    N = 64; yy, xx = np.mgrid[0:N, 0:N]
    fg = ((yy - 32) ** 2 + (xx - 22) ** 2 < 100) | ((yy - 32) ** 2 + (xx - 40) ** 2 < 100)
    lab = split_cells(fg, 3)
    assert lab.max() == 2 and ((lab > 0) == fg).all()
    t = territories(lab, 9)
    assert ((t == lab) | (lab == 0)).all() and (t > 0).sum() > fg.sum()
    assert contact_lines(lab).sum() > 0
