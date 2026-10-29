"""Class names: pipeline (legacy) order versus manuscript order (DECISIONS.md D-25).

The pipeline, the configuration and every ``metrics.json`` written so far name the mask values 2 and 3 as
``echinocyte`` and ``spherocyte`` (``configs/base.yaml``). The source paper of the data set (Kim et al. 2022, IEEE JBHI 26(3)) and
the shape of the annotated cells give the opposite order: mask value 2 = spherocyte, mask value 3 = echinocyte.
The author adopted that order for the manuscript on 25 Oct 2026.

Nothing that was trained or measured is changed: the mask VALUES are the same, only their names are swapped, and only
where the results are read for the manuscript (``manuscript/results.py``, ``manuscript/figures.py``,
``scripts/relabel_summary.py``). The configuration, the run directories and ``outputs/`` keep the legacy names on purpose,
so that the evidence stays exactly as it was produced. Apply ``swap_obj`` once, to data read from a legacy file; applying it
twice restores the legacy names.
"""
from __future__ import annotations

import re
from typing import Any

LEGACY_NAMES = ["background", "discocyte", "echinocyte", "spherocyte", "stomatocyte"]       # index = mask value
MANUSCRIPT_NAMES = ["background", "discocyte", "spherocyte", "echinocyte", "stomatocyte"]   # index = mask value
CLASSES = MANUSCRIPT_NAMES[1:]                                                                # mask values 1..4

_PAIR = {"echinocyte": "spherocyte", "spherocyte": "echinocyte"}
_RX = re.compile(r"(echinocyte|spherocyte)", re.IGNORECASE)
_ABBREV = re.compile(r"\bechi\b")


def _keep_case(src: str, new: str) -> str:
    if src.isupper():
        return new.upper()
    if src[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def swap_text(s: str) -> str:
    """Exchange the words echinocyte and spherocyte in a string (case kept); other text is left as it is."""
    return _RX.sub(lambda m: _keep_case(m.group(0), _PAIR[m.group(0).lower()]), s)


def swap_text_report(s: str) -> str:
    """``swap_text`` plus the table abbreviation 'echi' (mask value 2 in the legacy summaries) -> 'sphero'."""
    return _ABBREV.sub("sphero", swap_text(s))


def swap_obj(o: Any) -> Any:
    """Recursive ``swap_text`` over the keys and string values of JSON-like data (dict, list, tuple, str)."""
    if isinstance(o, dict):
        return {swap_text(k) if isinstance(k, str) else k: swap_obj(v) for k, v in o.items()}
    if isinstance(o, list):
        return [swap_obj(v) for v in o]
    if isinstance(o, tuple):
        return tuple(swap_obj(v) for v in o)
    if isinstance(o, str):
        return swap_text(o)
    return o


def legacy_name(manuscript_name: str) -> str:
    """Name under which a manuscript class is stored in a legacy file."""
    return _PAIR.get(manuscript_name, manuscript_name)
