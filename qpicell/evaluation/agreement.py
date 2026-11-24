"""Agreement between predicted and reference values (dry mass in pg, or integrated phase)."""
from __future__ import annotations

from typing import Dict

import numpy as np


def ccc(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    return float(2 * np.cov(a, b, ddof=0)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def agreement(pred, ref) -> Dict:
    """n, MdARE / mean |relative error| (%), bias and 95 % limits of agreement, Pearson r, Lin's CCC, SD of the
    reference and of the errors, and r_expected = SD_ref / sqrt(SD_ref^2 + SD_err^2): the correlation that a
    prediction = reference + independent error would give - it shows how much of a low r is caused by a narrow range
    of the reference values (range restriction)."""
    p, g = np.asarray(pred, float), np.asarray(ref, float)
    ok = np.isfinite(p) & np.isfinite(g) & (np.abs(g) > 0)
    p, g = p[ok], g[ok]
    n = int(len(p))
    if n == 0:
        return {"n": 0}
    d = p - g; rel = np.abs(d) / np.abs(g)
    sd_ref = float(g.std(ddof=1)) if n > 1 else float("nan"); sd_err = float(d.std(ddof=1)) if n > 1 else float("nan")
    return {"n": n, "mdare_pct": 100 * float(np.median(rel)), "mean_abs_rel_pct": 100 * float(rel.mean()),
            "bias": float(d.mean()), "loa_low": float(d.mean() - 1.96 * d.std(ddof=1)) if n > 1 else float("nan"),
            "loa_high": float(d.mean() + 1.96 * d.std(ddof=1)) if n > 1 else float("nan"),
            "pearson_r": pearson(p, g), "ccc": ccc(p, g), "sd_reference": sd_ref, "sd_error": sd_err,
            "r_expected_from_sd": float(sd_ref / np.sqrt(sd_ref ** 2 + sd_err ** 2)) if n > 1 and (sd_ref or sd_err) else float("nan"),
            "mean_reference": float(g.mean()), "mean_prediction": float(p.mean())}
