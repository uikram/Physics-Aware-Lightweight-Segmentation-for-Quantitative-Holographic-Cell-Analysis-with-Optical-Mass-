"""Shared plumbing: where the inputs are, a record of every file each product reads, number formatting, labels.

No experimental number is defined in this package. Every value that reaches a figure or a table is read from a
file under results/, additional_analysis/results/, training_results/ or configs/, and every file read is logged
against the product that read it (see Inputs.log), so SOURCES.json can say exactly where each number came from.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# ------------------------------------------------------------------------------------------------- names
# Names, not numbers. They turn the codes of the results files into words a reader sees.

TERM_SHORT = {"dice": "Dice", "pmc": "PMC", "bga": "BGA", "ipp": "IPP", "cipp": "cIPP"}
TERM_LONG = {"dice": "Dice", "pmc": "phase-mask contrast", "bga": "boundary-gradient alignment",
             "ipp": "global phase preservation", "cipp": "cell-level phase preservation"}
# Roles of three loss configurations in the manuscript's argument (codes, not results).
PROPOSED = "dice_cipp"
OLD_FULL = "dice_pmc_bga_ipp"
NEW_FULL = "dice_pmc_bga_cipp"
ROLE = {PROPOSED: "proposed", OLD_FULL: "old full", NEW_FULL: "new full"}
CLASSES_FALLBACK = ["discocyte", "spherocyte", "echinocyte", "stomatocyte"]


def arm_short(arm: str, role: bool = True) -> str:
    s = " + ".join(TERM_SHORT.get(t, t) for t in str(arm).split("_"))
    return f"{s} ({ROLE[arm]})" if role and arm in ROLE else s


def arm_math(arm: str) -> str:
    """Loss notation for figures (matplotlib mathtext), in the manuscript's style:
    dice -> 'ℒ_Dice only', dice_pmc -> '+ ℒ_PMC', dice_pmc_bga_ipp -> 'Full ℒ (+ ℒ_IPP)',
    dice_cipp -> '+ ℒ_cIPP', dice_pmc_bga_cipp -> 'Full ℒ (+ ℒ_cIPP)'. Every objective contains ℒ_Dice, so the
    '+' terms are what is added to it; 'Full' is ℒ_Dice + ℒ_PMC + ℒ_BGA plus the named phase term."""
    L = lambda t: r"$\mathcal{L}_{\mathrm{%s}}$" % t  # noqa: E731
    terms = str(arm).split("_")[1:]
    if not terms:
        return f"{L('Dice')} only"
    if terms[:2] == ["pmc", "bga"] and len(terms) == 3 and terms[2] in ("ipp", "cipp"):
        return r"Full $\mathcal{L}$ (+ %s)" % L(TERM_SHORT[terms[2]])
    return " + ".join([""] + [L(TERM_SHORT.get(t, t)) for t in terms]).strip()


LOSS_GLOSSARY = ("Every objective is ℒ<sub>Dice</sub> plus the terms shown: ℒ<sub>PMC</sub>, phase-mask contrast; "
                 "ℒ<sub>BGA</sub>, boundary-gradient alignment; ℒ<sub>IPP</sub>, global integrated-phase "
                 "preservation; ℒ<sub>cIPP</sub>, cell-level integrated-phase preservation (proposed: ℒ<sub>Dice</sub> "
                 "+ ℒ<sub>cIPP</sub>). Full ℒ = ℒ<sub>Dice</sub> + ℒ<sub>PMC</sub> + ℒ<sub>BGA</sub> plus the "
                 "phase term in brackets.")


def arm_long(arm: str) -> str:
    return " + ".join(TERM_LONG.get(t, t) for t in str(arm).split("_"))


def arch_disp(name: str) -> str:
    """edge_sam / EdgeSAM / MobileNet_UNet -> EdgeSAM / MobileNet-UNet."""
    k = str(name).strip().replace("-", "_").lower()
    return {"edge_sam": "EdgeSAM", "edgesam": "EdgeSAM", "mobile_sam": "MobileSAM", "mobilesam": "MobileSAM",
            "mobilenet_unet": "MobileNet-UNet"}.get(k, str(name).replace("_", "-"))


def arch_dir(name: str) -> str:
    """Display name -> results folder name (MobileNet-UNet -> MobileNet_UNet)."""
    return arch_disp(name).replace("-", "_")


def cond_label(c: str) -> str:
    return "full fine-tuning" if c == "fullft" else f"r = {str(c)[1:]}"


def cond_short(c: str) -> str:
    return "FFT" if c == "fullft" else f"r = {str(c)[1:]}"


def cond_label_m(c: str) -> str:
    """Figure text (mathtext): the rank variable r in italic."""
    return "full fine-tuning" if c == "fullft" else f"$r$ = {str(c)[1:]}"


def cond_short_m(c: str) -> str:
    return "FFT" if c == "fullft" else f"$r$ = {str(c)[1:]}"


def cls_name(k: str) -> str:
    """Class name with a capital first letter (discocyte -> Discocyte)."""
    return str(k)[:1].upper() + str(k)[1:]


# ------------------------------------------------------------------------------------------------- numbers

def finite(values) -> List[float]:
    out = []
    for x in values:
        if isinstance(x, bool) or x is None:
            continue
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def num(x, places: int = 3, scale: float = 1.0, sign: bool = False) -> str:
    v = finite([x])
    if not v:
        return "–"
    return f"{v[0] * scale:{'+' if sign else ''},.{places}f}"


def mean_sd(values, places: int = 3, scale: float = 1.0) -> str:
    v = np.array(finite(values)) * scale
    if v.size == 0:
        return "–"
    if v.size == 1:
        return f"{v[0]:,.{places}f}"
    return f"{v.mean():,.{places}f} ± {v.std(ddof=1):,.{places}f}"


def fmt_p(p) -> str:
    v = finite([p])
    if not v:
        return "–"
    p = v[0]
    if p == 0:
        return "0"
    if p < 1e-3:
        m, e = f"{p:.2e}".split("e")
        sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        return f"{m} × 10{str(int(e)).translate(sup)}"
    return f"{p:.3g}"


def p_range(values) -> str:
    v = finite(values)
    if not v:
        return "–"
    lo, hi = min(v), max(v)
    return fmt_p(lo) if lo == hi else f"{fmt_p(lo)} to {fmt_p(hi)}"


def ci(lohi, places: int = 2) -> str:
    v = finite(lohi or [])
    return f"[{v[0]:.{places}f}, {v[1]:.{places}f}]" if len(v) == 2 else "–"


# ------------------------------------------------------------------------------------------------- inputs

class MissingInput(Exception):
    pass


class Inputs:
    """Reads the project's result files and remembers, per product, which files were read (with their SHA-256)."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.current = "-"
        self.log: Dict[str, "OrderedDict[str, str]"] = {}
        self._hash: Dict[str, str] = {}
        self._cache: Dict[str, object] = {}
        self.allow_stale = False

    # -- bookkeeping
    def product(self, name: str):
        self.current = name
        self.log.setdefault(name, OrderedDict())

    def _note(self, p: Path) -> None:
        rel = p.relative_to(self.root).as_posix() if p.is_relative_to(self.root) else str(p)
        if rel not in self._hash:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            self._hash[rel] = h.hexdigest()
        self.log.setdefault(self.current, OrderedDict())[rel] = self._hash[rel]

    def path(self, rel: str) -> Path:
        return self.root / rel

    def exists(self, rel: str) -> bool:
        return (self.root / rel).is_file()

    def need(self, rel: str) -> Path:
        p = self.root / rel
        if not p.is_file():
            raise MissingInput(rel)
        self._note(p)
        return p

    # -- readers
    def json(self, rel: str):
        p = self.need(rel)
        if rel not in self._cache:
            with open(p, encoding="utf-8") as fh:
                self._cache[rel] = json.load(fh)
        return self._cache[rel]

    def csv(self, rel: str) -> List[Dict[str, str]]:
        p = self.need(rel)
        if rel not in self._cache:
            with open(p, newline="", encoding="utf-8") as fh:
                self._cache[rel] = list(csv.DictReader(fh))
        return self._cache[rel]

    def frame(self, rel: str):
        import pandas as pd
        p = self.need(rel)
        key = rel + "#frame"
        if key not in self._cache:
            self._cache[key] = pd.read_csv(p)
        return self._cache[key]

    def config(self) -> Dict:
        """configs/config.yaml, or {} if it or PyYAML is missing (every use has a results-derived fallback)."""
        rel = "configs/config.yaml"
        if not self.exists(rel):
            return {}
        try:
            import yaml
        except ImportError:
            return {}
        p = self.need(rel)
        if rel not in self._cache:
            with open(p, encoding="utf-8") as fh:
                self._cache[rel] = yaml.safe_load(fh)
        return self._cache[rel]


# ------------------------------------------------------------------------------------------------- run index

class Runs:
    """results/consolidated/runs_last.json, indexed by (architecture display, condition, loss, seed)."""

    def __init__(self, inp: Inputs, checkpoint: str = "last"):
        self.inp = inp
        self.checkpoint = checkpoint
        self.rel = f"results/consolidated/runs_{checkpoint}.json"
        data = inp.json(self.rel)
        if data.get("checkpoint") not in (None, checkpoint):
            raise ValueError(f"{self.rel} says checkpoint {data.get('checkpoint')!r}, expected {checkpoint!r}")
        self.rows: List[Dict] = data["runs"]
        self.index = {(arch_disp(r["architecture_display"]), r["condition"], r["loss"], int(r["seed"])): r
                      for r in self.rows}
        cfg = inp.config()
        # Orders come from the configuration when it is there, otherwise from the runs themselves.
        ranks = (cfg.get("conditions") or {}).get("lora_ranks")
        conds = sorted({r["condition"] for r in self.rows}, key=lambda c: (c == "fullft", int(c[1:]) if c != "fullft" else 0))
        self.conditions = ([f"r{k}" for k in ranks] + ["fullft"]) if ranks else conds
        self.conditions = [c for c in self.conditions if c in conds]
        self.lora_conditions = [c for c in self.conditions if c != "fullft"]
        arms_cfg = list((cfg.get("arms") or {}).keys())
        arms = sorted({r["loss"] for r in self.rows})
        self.arms = [a for a in arms_cfg if a in arms] or arms
        self.seeds = sorted({int(r["seed"]) for r in self.rows})
        arch_order = [arch_disp((cfg.get("models") or {}).get(k, {}).get("display", k)) for k in (cfg.get("models") or {})]
        archs = sorted({arch_disp(r["architecture_display"]) for r in self.rows})
        self.archs = [a for a in arch_order if a in archs] or archs
        names = (cfg.get("data") or {}).get("class_names")
        self.classes = [c for c in names if c != "background"] if names else CLASSES_FALLBACK
        ev = cfg.get("evaluation") or {}
        self.lost_class_dice = ev.get("lost_class_dice")
        self.within_pct = ev.get("within_pct")

    def get(self, arch, cond, loss, seed=None) -> List[Dict]:
        seeds = self.seeds if seed is None else [seed]
        return [self.index[(arch, cond, loss, s)] for s in seeds if (arch, cond, loss, s) in self.index]

    def col(self, arch, cond, loss, key, scale: float = 1.0) -> List[float]:
        return [v * scale for v in finite(r.get(key) for r in self.get(arch, cond, loss))]


def representative(inp: Inputs) -> "OrderedDict[str, Dict]":
    """Architecture -> its row of selection/rank_selection.csv (selected_condition, rule_applied, ...)."""
    out = OrderedDict()
    for r in inp.csv("results/consolidated/selection/rank_selection.csv"):
        out[arch_disp(r["architecture_display"] or r["architecture"])] = r
    return out


def require_rescored(inp: Inputs, rel: str) -> Dict:
    """A run's cells file, refused if it predates the project's rescoring pass (13 Nov 2026).

    runs_last.json was consolidated from the rescored per-run files; a copy written before the rescore carries no
    metadata.rescored and gives slightly different dry-mass statistics (and no circularity), so a figure drawn from
    it would disagree with the tables. --allow-stale overrides this for testing only."""
    d = inp.json(rel)
    if not (d.get("metadata") or {}).get("rescored") and not inp.allow_stale:
        raise MissingInput(f"{rel} predates the rescore (no metadata.rescored): use the server's rescored copy")
    return d


def run_dir(arch: str, cond: str, loss: str, seed: int, base: str = "results") -> str:
    return f"{base}/{arch_dir(arch)}/{cond}/{loss}/seed{seed}"
