#!/usr/bin/env python3
"""Every table of the main manuscript (Tables 1-5) and of the Supplementary Information (Tables S1-S16),
generated from the project's result files, in JSON, CSV and XLSX.

    bash additional_analysis/make_tables.sh            # reruns the analyses the tables need, then this script
    python additional_analysis/tables_for_manuscript.py # this script alone, from the current outputs

    Optional check of the generated tables against a printed version (nothing is written to the documents):
    python additional_analysis/tables_for_manuscript.py --manuscript MS.docx --supplement SI.docx

Output: results/tables_for_manuscript/
    <id>_<name>.json / .csv / .xlsx    one file of each kind per table
    all_tables.json, all_tables.xlsx   every table together (one sheet per table)
    provenance.json                    every file read, with its SHA-256, and the fields each table uses
    flags.md                           values that cannot be traced to a result file, and why
    verification_report.md / .csv      (with --manuscript/--supplement) cell-by-cell comparison with the printed tables

Rules this script keeps
  * No number is typed into this file. Every value is read from results/ (consolidated outputs, per-run files)
    or additional_analysis/results/; the representative ranks come from selection/rank_selection.csv.
    What the script does contain is layout: column headings, row order, display labels and rounding,
    chosen to reproduce the printed tables of FV-1.4 and SI 1.4.
  * A value the result files cannot give is written as "UNTRACEABLE" (or with a flag) and listed in flags.md;
    it is never estimated.
  * Per-run files (cells_last.json, metrics_last.json) are refused as stale unless they carry
    metadata.rescored, the same rule manuscript/figtab applies (--allow-stale overrides it, for testing only).

Read-only for the project: writes only below results/tables_for_manuscript/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from xml.etree import ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "tables_for_manuscript"
CONS = "results/consolidated"
SEL = f"{CONS}/selection"
W = "additional_analysis/results"

# ------------------------------------------------------------------------------------------- layout constants
# Names and labels only; no result value.
ARCH_ORDER = ["edge_sam", "mobile_sam", "mobilenet_unet"]
ARCH_FOLDER = {"edge_sam": "EdgeSAM", "mobile_sam": "MobileSAM", "mobilenet_unet": "MobileNet_UNet"}
ARCH_PRINT = {"edge_sam": "EdgeSAM", "mobile_sam": "MobileSAM", "mobilenet_unet": "MobileNet-UNet"}
FOLDER_ARCH = {v: k for k, v in ARCH_FOLDER.items()}
PROPOSED, OLD_FULL, NEW_FULL, DICE = "dice_cipp", "dice_pmc_bga_ipp", "dice_pmc_bga_cipp", "dice"
ARMS = ["dice", "dice_pmc", "dice_pmc_bga", "dice_pmc_bga_ipp", "dice_cipp", "dice_pmc_cipp", "dice_pmc_bga_cipp"]
ARM_SHORT = {"dice": "Dice alone", "dice_pmc": "Dice + PMC", "dice_pmc_bga": "Dice + PMC + BGA",
             "dice_pmc_bga_ipp": "Full, global IPP", "dice_cipp": "Proposed", "dice_pmc_cipp": "Dice + PMC + cIPP",
             "dice_pmc_bga_cipp": "Full, cIPP"}
ARM_LONG = {"dice": "Dice", "dice_pmc": "Dice + PMC", "dice_pmc_bga": "Dice + PMC + BGA",
            "dice_pmc_bga_ipp": "Dice + PMC + BGA + IPP (full, global IPP)", "dice_cipp": "Dice + cIPP (proposed)",
            "dice_pmc_cipp": "Dice + PMC + cIPP", "dice_pmc_bga_cipp": "Dice + PMC + BGA + cIPP (full, cIPP)"}
CLASSES = ["discocyte", "spherocyte", "echinocyte", "stomatocyte"]
LORA_CONDS = ["r8", "r16", "r32", "r64"]
CONDS = LORA_CONDS + ["fullft"]
SEEDS = [42, 43, 44]
MINUS = "−"


# ------------------------------------------------------------------------------------------------ formatting
def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _m(s: str) -> str:
    return s.replace("-", MINUS)


def num(x, dp: int, sign: bool = False, comma: bool = True) -> str:
    v = _f(x)
    if v is None:
        return "–"
    s = f"{v:{'+' if sign else ''}{',' if comma else ''}.{dp}f}"
    if sign and re.fullmatch(r"[+-]0\.?0*", s):     # a rounded zero carries no sign in print
        s = s[1:]
    return _m(s)


def msd(vals, dp: int, scale: float = 1.0, sign: bool = False) -> str:
    v = np.array([_f(x) for x in vals if _f(x) is not None], dtype=float) * scale
    if v.size == 0:
        return "–"
    if v.size == 1:
        return num(v[0], dp, sign)
    return f"{num(v.mean(), dp, sign)} ± {num(v.std(ddof=1), dp)}"


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x, sig: int = 3) -> str:
    v = _f(x)
    if v is None:
        return "–"
    m, e = f"{v:.{sig - 1}e}".split("e")
    return f"{_m(m)} × 10{str(int(e)).translate(_SUP)}"


def pval(p, sig: int = 3) -> str:
    v = _f(p)
    if v is None:
        return "–"
    if v == 0:
        return "0"
    if v < 1e-3:
        return sci(v, sig)
    s = f"{v:.{sig}g}"
    return s


def prange(vals, sig: int = 2) -> str:
    v = [x for x in (_f(y) for y in vals) if x is not None]
    if not v:
        return "–"
    lo, hi = min(v), max(v)
    return pval(lo, sig) if pval(lo, sig) == pval(hi, sig) else f"{pval(lo, sig)}–{pval(hi, sig)}"


def irange(vals) -> str:
    v = sorted(int(x) for x in vals)
    return f"{v[0]:,}" if v[0] == v[-1] else f"{v[0]:,}–{v[-1]:,}"


def ci(lohi, dp: int) -> str:
    v = [x for x in (_f(y) for y in (lohi or [])) if x is not None]
    return f"[{num(v[0], dp)}, {num(v[1], dp)}]" if len(v) == 2 else "–"


def cond_rank(c: str) -> str:
    return "Full fine-tuning" if c == "fullft" else c.lstrip("r")


def cond_eq(c: str) -> str:
    return "Full fine-tuning" if c == "fullft" else f"r = {c.lstrip('r')}"


def arch_any(name: str) -> str:
    """Any spelling of an architecture -> its key (edge_sam, ...)."""
    n = str(name).strip()
    if n in ARCH_FOLDER:
        return n
    for k in ARCH_ORDER:
        if n in (ARCH_FOLDER[k], ARCH_PRINT[k]) or n.replace("-", "_") == ARCH_FOLDER[k]:
            return k
    raise KeyError(name)


# ------------------------------------------------------------------------------------------------ inputs
class Untraceable(Exception):
    pass


class Inputs:
    def __init__(self, root: Path, allow_stale: bool = False):
        self.root, self.allow_stale = root, allow_stale
        self.read: Dict[str, str] = OrderedDict()
        self.stale: List[str] = []
        self._cache: Dict[str, Any] = {}

    def path(self, rel: str) -> Path:
        return self.root / rel

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def _log(self, rel: str):
        if rel not in self.read:
            self.read[rel] = hashlib.sha256(self.path(rel).read_bytes()).hexdigest()

    def json(self, rel: str):
        if rel not in self._cache:
            if rel in self.stale:
                raise Untraceable(f"{rel} is older than runs_last.json / rank_selection.csv: rerun "
                                  "additional_analysis/make_tables.sh")
            if not self.exists(rel):
                raise Untraceable(f"missing input {rel}")
            self._log(rel)
            self._cache[rel] = json.loads(self.path(rel).read_text(encoding="utf-8"))
        return self._cache[rel]

    def csv(self, rel: str) -> List[Dict[str, str]]:
        if rel not in self._cache:
            if rel in self.stale:
                raise Untraceable(f"{rel} is older than runs_last.json / rank_selection.csv: rerun "
                                  "additional_analysis/make_tables.sh")
            if not self.exists(rel):
                raise Untraceable(f"missing input {rel}")
            self._log(rel)
            with open(self.path(rel), newline="", encoding="utf-8-sig") as fh:
                self._cache[rel] = list(csv.DictReader(fh))
        return self._cache[rel]

    def cells(self, arch: str, cond: str, arm: str, seed: int):
        rel = f"results/{ARCH_FOLDER[arch]}/{cond}/{arm}/seed{seed}/cells_last.json"
        d = self.json(rel)
        if not (d.get("metadata") or {}).get("rescored") and not self.allow_stale:
            raise Untraceable(f"{rel} predates the rescore (no metadata.rescored); use the server's rescored copy")
        return d

    def metrics(self, arch: str, cond: str, arm: str, seed: int):
        self.cells(arch, cond, arm, seed)       # same staleness rule as the cells file of the run
        return self.json(f"results/{ARCH_FOLDER[arch]}/{cond}/{arm}/seed{seed}/metrics_last.json")


class Runs:
    """results/consolidated/runs_last.json, indexed by (arch, condition, loss)."""

    def __init__(self, inp: Inputs):
        d = inp.json(f"{CONS}/runs_last.json")
        self.idx: Dict[tuple, List[Dict]] = {}
        for r in d["runs"]:
            self.idx.setdefault((r["architecture"], r["condition"], r["loss"]), []).append(r)
        for v in self.idx.values():
            v.sort(key=lambda r: int(r["seed"]))
        self.archs = [a for a in ARCH_ORDER if any(k[0] == a for k in self.idx)]

    def get(self, a, c, arm) -> List[Dict]:
        return self.idx.get((a, c, arm), [])

    def col(self, a, c, arm, key):
        return [r.get(key) for r in self.get(a, c, arm)]

    def lora(self, a, arm) -> List[Dict]:
        return [r for c in LORA_CONDS for r in self.get(a, c, arm)]


def lost(rows) -> str:
    return f"{sum(1 for r in rows if (r.get('n_lost_classes') or 0) > 0)} of {len(rows)}"


def nlost(rows) -> int:
    return sum(1 for r in rows if (r.get("n_lost_classes") or 0) > 0)


# Additional-analysis outputs the tables read, and the consolidated files they are computed from. An output older
# than any of these was computed from earlier results (or at earlier ranks) and is refused.
DERIVED = [f"{W}/w1_cell_paired/w1_cell_paired.json", f"{W}/w2_per_architecture/w2_per_architecture_last.json",
           f"{W}/w2_per_architecture/w2_per_architecture_best.json", f"{W}/w4_cell_vs_global_ipp/w4_cell_vs_global_ipp.json",
           f"{W}/w5_reference_cells/w5_reference_cells.json", f"{W}/w7_followup/w7_followup.json",
           f"{W}/w7_followup/w7a_dry_mass_outliers.csv", f"{W}/w8_signed_errors/w8_signed_errors.json"]
UPSTREAM = [f"{CONS}/runs_last.json", f"{SEL}/rank_selection.csv"]


def stale_outputs(root: Path) -> List[str]:
    newest = max((root / u).stat().st_mtime for u in UPSTREAM if (root / u).exists())
    return [d for d in DERIVED if (root / d).exists() and (root / d).stat().st_mtime < newest]


def representative(inp: Inputs) -> Dict[str, str]:
    out = OrderedDict()
    for r in inp.csv(f"{SEL}/rank_selection.csv"):
        out[arch_any(r["architecture"] or r["architecture_display"])] = r["selected_condition"]
    return OrderedDict((a, out[a]) for a in ARCH_ORDER if a in out)


# ------------------------------------------------------------------------------------------------ table object
class Table:
    def __init__(self, tid: str, name: str, title: str, columns: List[str], part: str):
        self.id, self.name, self.title, self.columns, self.part = tid, name, title, columns, part
        self.rows: List[List[str]] = []
        self.group_rows: List[int] = []
        self.sources: List[str] = []
        self.fields: List[str] = []
        self.flags: List[str] = []
        self.notes: List[str] = []

    def group(self, label: str):
        self.group_rows.append(len(self.rows))
        self.rows.append([label] + [""] * (len(self.columns) - 1))

    def add(self, row: Sequence[str]):
        assert len(row) == len(self.columns), (self.id, len(row), len(self.columns), row)
        self.rows.append([str(x) for x in row])

    def to_dict(self):
        return {"id": self.id, "name": self.name, "part": self.part, "title": self.title, "columns": self.columns,
                "rows": self.rows, "group_rows": self.group_rows, "sources": self.sources,
                "fields": self.fields, "flags": self.flags, "notes": self.notes}

    @property
    def stem(self):
        return f"{'Table_' + self.id}_{self.name}"


def uses(t: Table, inp: Inputs, *fields: str):
    t.fields += [f for f in fields if f not in t.fields]


# =============================================================================================== main text
def table1(inp, R, rep):
    t = Table("1", "representative_configurations", "Selected configurations under the proposed objective",
              ["Architecture", "LoRA rank", "Cells within 5% (%)", "Matched-cell IPE (%)", "Global IPE (%)",
               "Mean Dice", "AJI, cells", "AJI, connected regions", "Boundary F1", "Runs that lost a class"], "main")
    for a in R.archs:
        c = rep[a]
        g = lambda k, dp, s=1.0: msd(R.col(a, c, PROPOSED, k), dp, s)  # noqa: E731
        t.add([ARCH_PRINT[a], cond_rank(c), g("a2_scored_within_5pct_of_all_cells_pct", 1),
               g("a2_scored_cell_ipe_abs_median_pct", 2), g("ipe_global", 2, 100.0), g("dice_macro", 3),
               g("a2_aji_cells", 3), g("aji_cc", 3), g("bf1", 3), lost(R.get(a, c, PROPOSED))])
    t.sources = [f"{CONS}/runs_last.json", f"{SEL}/rank_selection.csv"]
    uses(t, inp, "a2_scored_within_5pct_of_all_cells_pct", "a2_scored_cell_ipe_abs_median_pct", "ipe_global (x100)",
         "dice_macro", "a2_aji_cells", "aji_cc", "bf1", "n_lost_classes")
    t.notes.append("Mean ± SD over seeds, last checkpoint, objective dice_cipp, rank = selected_condition.")
    return t


def _w2(inp, ckpt="last"):
    return inp.json(f"{W}/w2_per_architecture/w2_per_architecture_{ckpt}.json")


def _w2cmp(w2, a, loss_b, block="lora_ranks_pooled"):
    res = w2["results"].get(ARCH_FOLDER[a]) or w2["results"].get(ARCH_PRINT[a])
    for c in (res or {}).get(block, {}).get("comparisons", []):
        if c["loss_a"] == PROPOSED and c["loss_b"] == loss_b:
            return c
    return None


def _holm(m):
    return m.get("wilcoxon_p_holm_within_architecture")


def table2(inp, R, rep):
    w2 = _w2(inp)
    t = Table("2", "objective_comparison_per_architecture",
              "Proposed objective (Dice + cIPP) against each other loss configuration, per architecture",
              ["Compared configuration", "Within 5% (%)", "Δ (pairs)", "p", "Matched-cell IPE (%)", "Δ (pairs)", "p",
               "Global IPE (%)", "Δ (pairs)", "p"], "main")
    met = [("a2_scored_within_5pct_of_all_cells_pct", 1, 2), ("a2_scored_cell_ipe_abs_median_pct", 2, 3),
           ("ipe_global", 2, 3)]
    for a in R.archs:
        cmps = [_w2cmp(w2, a, b) for b in ARMS if b != PROPOSED]
        first = next(c for c in cmps if c)
        mm = first["metrics"]
        t.group(f"{ARCH_PRINT[a]} (proposed objective: {num(mm[met[0][0]]['mean_a'], 1)}% within 5%; "
                f"matched-cell IPE {num(mm[met[1][0]]['mean_a'], 2)}%; global IPE {num(mm[met[2][0]]['mean_a'], 2)}%)")
        for b, c in zip([x for x in ARMS if x != PROPOSED], cmps):
            if c is None:
                t.add([ARM_SHORT[b]] + ["UNTRACEABLE"] * 9)
                t.flags.append(f"{ARCH_PRINT[a]} vs {b}: comparison absent from W-2")
                continue
            row = [ARM_SHORT[b]]
            for k, dpm, dpd in met:
                m = c["metrics"][k]
                row += [num(m["mean_b"], dpm),
                        f"{num(m['median_difference_a_minus_b'], dpd, sign=True)} ({m['pairs_a_better']}/{m['n_pairs']})",
                        pval(_holm(m))]
            t.add(row)
    t.sources = [f"{W}/w2_per_architecture/w2_per_architecture_last.json"]
    uses(t, inp, "results[arch].lora_ranks_pooled.comparisons[loss_a=dice_cipp].metrics[k].{mean_a, mean_b, "
                 "median_difference_a_minus_b, pairs_a_better, n_pairs, wilcoxon_p_holm_within_architecture}")
    t.notes.append(f"Holm family size per architecture: {w2.get('metadata', {}).get('holm_family', '–')}")
    return t


def table3(inp, R, rep):
    t = Table("3", "touching_isolated_and_outcomes",
              "Touching and isolated cells, and cell outcomes, at the selected configurations",
              ["Configuration", "Cells within 5%, touching (%)", "Cells within 5%, isolated (%)",
               "Matched-cell IPE, touching (%)", "Matched-cell IPE, isolated (%)", "Matched (%)", "Missed (%)",
               "Merged (%)", "Fragmented (%)"], "main")
    for a in R.archs:
        c = rep[a]
        g = lambda k, dp: msd(R.col(a, c, PROPOSED, k), dp)  # noqa: E731
        t.add([f"{ARCH_PRINT[a]} ({cond_eq(c)})", g("a2_touching_within_5pct_of_all_cells_pct", 1),
               g("a2_isolated_within_5pct_of_all_cells_pct", 1), g("a2_touching_cell_ipe_abs_median_pct", 2),
               g("a2_isolated_cell_ipe_abs_median_pct", 2), g("a2_scored_match_pct", 1), g("a2_scored_missed_pct", 2),
               g("a2_scored_merged_pct", 1), g("a2_scored_fragmented_pct", 1)])
    t.sources = [f"{CONS}/runs_last.json", f"{SEL}/rank_selection.csv"]
    uses(t, inp, "a2_{touching,isolated}_within_5pct_of_all_cells_pct", "a2_{touching,isolated}_cell_ipe_abs_median_pct",
         "a2_scored_{match,missed,merged,fragmented}_pct")
    return t


BACKEND = {"pytorch": "PyTorch", "onnx": "ONNX Runtime"}


def _bench_order(rows):
    return sorted(rows, key=lambda r: (int(float(r["input_size"])), ARCH_ORDER.index(arch_any(r["architecture_display"])),
                                       list(BACKEND).index(r["backend"])))


def table4(inp, R, rep):
    rows = _bench_order(inp.csv(f"{CONS}/benchmark/benchmark.csv"))
    t = Table("4", "inference_cost", "Forward-pass latency, throughput and device memory of the deployed networks",
              ["Model", "Runtime", "Median latency (ms)", "99th-percentile latency (ms)", "Throughput (images/s)",
               "Memory (MiB)"], "main")
    seen = set()
    for r in rows:
        a = arch_any(r["architecture_display"])
        lab = "" if a in seen else f"{ARCH_PRINT[a]} ({r['input_size']} × {r['input_size']} px)"
        seen.add(a)
        ms = lambda k, dp: f"{num(r[k], dp)} ± {num(r[k + '_sd'], dp)}"  # noqa: E731
        t.add([lab, BACKEND.get(r["backend"], r["backend"]), ms("latency_p50_ms", 2), ms("latency_p99_ms", 2),
               ms("images_per_second", 1), ms("memory_mib", 0)])
    t.sources = [f"{CONS}/benchmark/benchmark.csv"]
    uses(t, inp, "latency_p50_ms(_sd)", "latency_p99_ms(_sd)", "images_per_second(_sd)", "memory_mib(_sd)", "input_size")
    t.notes.append("Mean ± SD over " + ", ".join(sorted({r['n_passes'] for r in rows})) + " passes.")
    return t


def table5(inp, R, rep):
    st = inp.json(f"{CONS}/storage/runs_last.json")
    ann = inp.json(f"{CONS}/storage/annotation.json")
    sidx = {(r["arch"], r["condition"], r["arm"], int(r["seed"])): r for r in st["runs"]}
    t = Table("5", "dry_mass_and_storage",
              "Single-cell dry-mass agreement and storage-time trends at the selected configurations",
              ["Configuration", "Bias (pg)", "Pearson r", "CCC", "Per-image r", "Area change (%)", "Area ρ (p)",
               "Dry-mass change (%)", "Dry-mass ρ (p)", "Bias ρ"], "main")
    ta, tm = ann["trends"]["area"], ann["trends"]["dry_mass"]
    t.add(["Annotation", "–", "–", "–", "–", num(ta["early_to_late_change_pct"], 1),
           f"{num(ta['rho'], 2)} ({pval(ta['p_block_permutation'], 2)})", num(tm["early_to_late_change_pct"], 1),
           f"{num(tm['rho'], 2)} ({pval(tm['p_block_permutation'], 2)})", "–"])
    for a in R.archs:
        c = rep[a]
        S = [sidx[(a, c, PROPOSED, s)] for s in SEEDS if (a, c, PROPOSED, s) in sidx]
        try:
            img_r = [inp.metrics(a, c, PROPOSED, s)["instances"]["A2"]["dry_mass"]["per_image_median"]["pearson_r"]
                     for s in SEEDS]
            img = msd(img_r, 3)
        except Untraceable as e:
            img = "UNTRACEABLE"
            t.flags.append(f"Per-image r, {ARCH_PRINT[a]} {c}: {e}")
        g = lambda k, dp, sign=False: msd(R.col(a, c, PROPOSED, k), dp, sign=sign)  # noqa: E731
        f = lambda q, k, dp: msd([x[q][k] for x in S], dp)  # noqa: E731
        t.add([f"{ARCH_PRINT[a]} ({cond_eq(c)})", g("a2_scored_drymass_bias_pg", 2, True),
               g("a2_scored_drymass_pearson_r", 3), g("a2_scored_drymass_ccc", 3), img,
               f("area", "early_to_late_change_pct", 1),
               f"{f('area', 'rho', 2)} ({prange([x['area']['p_block_permutation'] for x in S])})",
               f("dry_mass", "early_to_late_change_pct", 1),
               f"{f('dry_mass', 'rho', 2)} ({prange([x['dry_mass']['p_block_permutation'] for x in S])})",
               msd([x["dry_mass"]["bias_vs_annotation"]["rho"] for x in S], 2)])
    t.sources = [f"{CONS}/runs_last.json", f"{CONS}/storage/runs_last.json", f"{CONS}/storage/annotation.json",
                 "results/<arch>/<rank>/dice_cipp/seed<s>/metrics_last.json (per-image r)"]
    uses(t, inp, "a2_scored_drymass_{bias_pg,pearson_r,ccc}", "metrics_last: instances.A2.dry_mass.per_image_median.pearson_r",
         "storage: {area,dry_mass}.{early_to_late_change_pct, rho, p_block_permutation}",
         "storage: dry_mass.bias_vs_annotation.rho")
    t.notes.append("Bias ρ: dry-mass bias against day (the printed column has a single value per row; "
                   "the area-bias ρ is in Table S12).")
    return t


# ============================================================================================ supplementary
def tableS1(inp, R, rep):
    w5 = inp.json(f"{W}/w5_reference_cells/w5_reference_cells.json")["results"]
    t = Table("S1", "test_set_per_day", "Test set per storage day",
              ["Storage day", "Reference cells", "Scored cells", "Touching cells (%)",
               "Cell pixels (% of annotated field)", "Median area of scored cells (µm²)"], "SI")
    med = {}
    try:
        a0 = R.archs[0]
        D = [x for x in inp.cells(a0, rep[a0], PROPOSED, SEEDS[0])["cells"] if x.get("instances") == "A2" and x.get("scored")]
        for d in sorted({x["storage_day"] for x in D}):
            med[int(d)] = float(np.median([x["area_um2"] for x in D if x["storage_day"] == d]))
    except Untraceable as e:
        t.flags.append(f"Median area of scored cells: {e}")
    for d in w5["crowding_per_day"]:
        day = int(d["storage_day"])
        t.add([str(day), f"{d['reference_cells']:,}", f"{d['scored_cells']:,}", num(d["touching_pct"], 1),
               num(100 * d["fg_mean"], 1) if d.get("fg_mean") is not None else "–",
               num(med[day], 2) if day in med else "UNTRACEABLE"])
    t.sources = [f"{W}/w5_reference_cells/w5_reference_cells.json",
                 "results/<arch>/<rank>/dice_cipp/seed42/cells_last.json (reference columns; W-5 checks they are "
                 "identical in every run)"]
    uses(t, inp, "crowding_per_day[].{reference_cells, scored_cells, touching_pct, fg_mean}",
         "cells: median area_um2 of scored A2 reference cells per storage_day")
    return t


def tableS2(inp, R, rep):
    pr = inp.json(f"{CONS}/phase_range/phase_range.json")
    man = inp.csv("data/prepared/manifest.csv")
    ceil, tol = pr["wrapping"]["ceiling_rad"], pr["wrapping"]["ceiling_tolerance_rad"]
    t = Table("S2", "phase_range", "Phase range and possible phase wrapping",
              ["Set", "Reference cells", "Image maximum (rad)", "Cell maximum, median (rad)",
               "Cell maximum, 99th percentile (rad)", "Cell maximum, largest (rad)", f"Maps within {tol:g} rad of 2π",
               "Images with flagged pixels", "Reference cells with flagged pixels"], "SI")
    names = [("train", "Training split"), ("val", "Test set"), (None, "All")]
    for key, lab in names:
        P = pr["per_split"][key] if key else pr["overall"]
        M = [r for r in man if (key is None or r["split"] == key)]
        near = sum(float(r["raw_max"]) >= ceil - tol for r in M)
        t.add([f"{lab} ({P['n_images']} images)", f"{P['n_cells']:,}", num(P["image_phase_max"]["max"], 2),
               num(P["cell_phase_max"]["median"], 2), num(P["cell_phase_max"]["p99"], 2),
               num(P["cell_phase_max"]["max"], 2), f"{near} of {len(M)}",
               str(P["images_with_a_flagged_pixel"]), str(P["cells_containing_a_flagged_pixel"])])
    # The column "Dated maps reaching 2π" of SI 1.4 is dropped: the raw maxima approach 2π continuously, so a count
    # of maps "reaching" it depends on an arbitrary cut-off that no result file records.
    t.sources = [f"{CONS}/phase_range/phase_range.json", "data/prepared/manifest.csv"]
    uses(t, inp, "per_split/overall: n_images, n_cells, image_phase_max.max, cell_phase_max.{median,p99,max}, "
                 "images_with_a_flagged_pixel, cells_containing_a_flagged_pixel",
         "wrapping.{ceiling_rad, ceiling_tolerance_rad}", "manifest: raw_max")
    return t


BI_ORDER = ["constant offset", "tilt along x", "tilt along y", "curvature along x", "curvature along y",
            "saddle, x times y"]


def tableS3(inp, R, rep):
    bi = inp.json(f"{CONS}/background_invariance/background_invariance.json")
    amps = sorted({r["amplitude_rad"] for r in bi["results"]})
    t = Table("S3", "background_invariance", "Invariance of the background correction to added low-order terms",
              [f"Added term ({' / '.join(num(a, 1) for a in amps)} rad)", "Largest change of the corrected map (rad)",
               "Cell integrated phase, median |change| (%)", "Cell integrated phase, 95th percentile |change| (%)",
               "Cell integrated phase, largest |change| (%)", "Cell dry mass, largest |change| (pg)"], "SI")
    res = sorted(bi["results"], key=lambda r: BI_ORDER.index(r["term"]) if r["term"] in BI_ORDER else 99)
    for r in res:
        ph, dm = r["cell_integrated_phase_change_pct"], r["cell_dry_mass_change_pg"]
        t.add([r["term"][0].upper() + r["term"][1:], sci(r["map_change_max_rad"], 3), num(ph["median_abs"], 4),
               num(ph["p95_abs"], 3), num(ph["max_abs"], 3), num(dm["max_abs"], 3)])
    t.sources = [f"{CONS}/background_invariance/background_invariance.json"]
    uses(t, inp, "results[].{term, amplitude_rad, map_change_max_rad, cell_integrated_phase_change_pct."
                 "{median_abs,p95_abs,max_abs}, cell_dry_mass_change_pg.max_abs}")
    return t


def tableS4(inp, R, rep):
    rows = inp.csv(f"{CONS}/benchmark/benchmark_per_pass.csv")
    t = Table("S4", "timing_per_pass", "Forward-pass timing per pass (passes 1 / 2 / 3)",
              ["Runtime", "Input (px)", "Median latency per pass (ms)", "99th-percentile latency per pass (ms)",
               "Memory per pass (MiB)", "Agreement with PyTorch: pixels (%) / largest logit difference"], "SI")
    for a in R.archs:
        A = [r for r in rows if arch_any(r["architecture_display"]) == a]
        if not A:
            continue
        t.group(ARCH_PRINT[a])
        for b in BACKEND:
            P = sorted([r for r in A if r["backend"] == b], key=lambda r: int(r["pass_seed"]))
            if not P:
                continue
            j = lambda k, dp: " / ".join(num(r[k], dp, comma=False) for r in P)  # noqa: E731
            agr = [_f(r["class_map_agreement_pct"]) for r in P if _f(r["class_map_agreement_pct"]) is not None]
            lg = [_f(r["max_abs_logit_difference"]) for r in P if _f(r["max_abs_logit_difference"]) is not None]
            t.add([BACKEND[b], f"{P[0]['input_size']} × {P[0]['input_size']}", j("latency_p50_ms", 2),
                   j("latency_p99_ms", 2), j("memory_mib", 0),
                   f"{num(min(agr), 1)} / {sci(max(lg), 3)}" if agr and lg else "–"])
    t.sources = [f"{CONS}/benchmark/benchmark_per_pass.csv"]
    uses(t, inp, "per pass (ordered by pass_seed): latency_p50_ms, latency_p99_ms, memory_mib",
         "class_map_agreement_pct (smallest over passes), max_abs_logit_difference (largest over passes)")
    return t


def tableS5(inp, R, rep):
    t = Table("S5", "loss_ablation", "Loss-configuration ablation over all ranks and full fine-tuning",
              ["Loss configuration"] + [cond_eq(c) for c in CONDS] + ["LoRA mean", "Matched-cell IPE (%)",
               "Global IPE (%)", "Mean Dice", "Merged cells (%)", "Runs that lost a class (LoRA / FFT)"], "SI")
    for a in R.archs:
        t.group(ARCH_PRINT[a])
        for arm in ARMS:
            L = R.lora(a, arm)
            mean = lambda rows, k, s=1.0: (np.mean([r[k] for r in rows]) * s if rows else None)  # noqa: E731
            t.add([ARM_LONG[arm]] + [num(mean(R.get(a, c, arm), "a2_scored_within_5pct_of_all_cells_pct"), 1) for c in CONDS]
                  + [num(mean(L, "a2_scored_within_5pct_of_all_cells_pct"), 1),
                     num(mean(L, "a2_scored_cell_ipe_abs_median_pct"), 2), num(mean(L, "ipe_global", 100.0), 2),
                     num(mean(L, "dice_macro"), 3), num(mean(L, "a2_scored_merged_pct"), 1),
                     f"{nlost(L)} / {nlost(R.get(a, 'fullft', arm))}"])
    t.sources = [f"{CONS}/runs_last.json"]
    uses(t, inp, "a2_scored_within_5pct_of_all_cells_pct (seed mean per condition; LoRA mean over the 12 LoRA runs)",
         "a2_scored_cell_ipe_abs_median_pct, ipe_global, dice_macro, a2_scored_merged_pct (LoRA means)",
         "n_lost_classes (runs counted)")
    return t


S6_LAST = [("a2_scored_drymass_ccc", "Dry-mass CCC", 4, True), ("dice_macro", "Mean Dice", 4, True),
           ("a2_aji_cells", "AJI, watershed cells", 4, True), ("bf1", "Boundary F1", 4, True),
           ("a2_scored_match_pct", "Matched cells (%)", 2, True), ("a2_scored_missed_pct", "Missed cells (%)", 2, True),
           ("a2_scored_merged_pct", "Merged cells (%)", 2, False),
           ("a2_scored_fragmented_pct", "Fragmented cells (%)", 2, False),
           ("a2_scored_drymass_bias_pg", "Dry-mass bias (pg)", 3, False)]       # last: pooled value printed?
S6_BEST = [("a2_scored_within_5pct_of_all_cells_pct", "Cells within 5% (points)", 2),
           ("a2_scored_cell_ipe_abs_median_pct", "Matched-cell IPE (points)", 3),
           ("ipe_global", "Global IPE (points)", 2)]


def tableS6(inp, R, rep):
    last, best = _w2(inp, "last"), _w2(inp, "best")
    ac = [r for r in inp.csv(f"{SEL}/arm_confirmation.csv") if r["loss_a"] == PROPOSED]
    npool = sorted({r["n_pairs"] for r in ac if r["loss_b"] in (DICE, OLD_FULL)})
    t = Table("S6", "per_architecture_further_metrics",
              "Per-architecture tests of further metrics and of the checkpoint choice",
              ["Metric", "Compared objective"] + [ARCH_PRINT[a] for a in R.archs] + [f"All ({'/'.join(npool)} pairs)"], "SI")

    def cell(w2, a, b, k, dp):
        c = _w2cmp(w2, a, b)
        if not c or k not in c["metrics"]:
            return "UNTRACEABLE"
        m = c["metrics"][k]
        return f"{num(m['median_difference_a_minus_b'], dp, sign=True)} ({pval(_holm(m))})"

    t.group("Last-epoch checkpoint")
    for k, lab, dp, pooled in S6_LAST:
        for i, b in enumerate((DICE, OLD_FULL)):
            pc = "–"
            if pooled:
                r = next((x for x in ac if x["loss_b"] == b and x["metric"] == k), None)
                pc = f"{num(r['median_difference_a_minus_b'], dp, sign=True)} ({pval(r['wilcoxon_p'])})" if r else "UNTRACEABLE"
            t.add([lab if i == 0 else "", ARM_SHORT[b]] + [cell(last, a, b, k, dp) for a in R.archs] + [pc])
    t.group("Checkpoint of highest validation Dice")
    for k, lab, dp in S6_BEST:
        for i, b in enumerate((DICE, OLD_FULL)):
            t.add([lab if i == 0 else "", ARM_SHORT[b]] + [cell(best, a, b, k, dp) for a in R.archs] + ["–"])
    t.sources = [f"{W}/w2_per_architecture/w2_per_architecture_last.json",
                 f"{W}/w2_per_architecture/w2_per_architecture_best.json", f"{SEL}/arm_confirmation.csv"]
    uses(t, inp, "W-2: median_difference_a_minus_b (wilcoxon_p_holm_within_architecture)",
         "arm_confirmation.csv (loss_a = dice_cipp): median_difference_a_minus_b (wilcoxon_p), pooled column")
    t.notes.append("Pooled column printed only for the metrics FV-1.4 prints it for (layout choice).")
    return t


S7_KEYS = [("a2_scored_within_5pct_of_all_cells_pct", "Cells within 5% (%)", 2),
           ("a2_scored_cell_ipe_abs_median_pct", "Matched-cell IPE (%)", 3), ("ipe_global", "Global IPE (%)", 3),
           ("a2_scored_drymass_ccc", "Dry-mass CCC", 3), ("a2_scored_match_pct", "Matched cells (%)", 2),
           ("a2_scored_merged_pct", "Merged cells (%)", 2)]


def tableS7(inp, R, rep):
    w4 = inp.json(f"{W}/w4_cell_vs_global_ipp/w4_cell_vs_global_ipp.json")["results"]
    t = Table("S7", "full_cipp_vs_full_global_ipp", "Full objective with cIPP against full objective with global IPP",
              ["Quantity", "Full objective with cIPP", "Full objective with global IPP",
               "Median difference [95% interval]", "Pairs favouring cIPP", "p"], "SI")
    blocks = [("All architectures, LoRA", w4["lora_pooled"])]
    pa = {arch_any(k): v for k, v in w4["per_architecture_lora"].items()}
    blocks += [(f"{ARCH_PRINT[a]}, LoRA", pa[a]) for a in R.archs if a in pa]
    blocks += [("All architectures, full fine-tuning", w4["full_finetuning_pooled"])]
    for name, B in blocks:
        n = B.get("n_pairs") or next(iter(B["metrics"].values()))["n_pairs"]
        t.group(f"{name} ({n} pairs)")
        for k, lab, dp in S7_KEYS:
            m = B["metrics"].get(k)
            if not m:
                t.add([lab] + ["UNTRACEABLE"] * 5)
                continue
            t.add([lab, num(m["mean_new_full"], dp), num(m["mean_old_full"], dp),
                   f"{num(m['median_difference'], dp, sign=True)} {ci(m['ci95_median_difference'], dp)}",
                   f"{m['pairs_new_full_better']} of {m['n_pairs']}", pval(m["wilcoxon_p"])])
    t.sources = [f"{W}/w4_cell_vs_global_ipp/w4_cell_vs_global_ipp.json"]
    uses(t, inp, "metrics[k].{mean_new_full, mean_old_full, median_difference, ci95_median_difference, "
                 "pairs_new_full_better, n_pairs, wilcoxon_p}")
    return t


def _armlab(x):
    return ARM_SHORT[x]


def tableS8(inp, R, rep):
    w1 = inp.json(f"{W}/w1_cell_paired/w1_cell_paired.json")
    t = Table("S8", "cell_level_paired", "Cell-level paired comparisons at the selected configurations",
              ["Comparison (A vs B)", "Cells within 5%, A / B (%)", "Difference (points) [95% interval]",
               "p / Holm p", "Matching-free median error, difference (points) [95% interval]", "p / Holm p"], "SI")
    for a in R.archs:
        rs = [r for r in w1["results"] if arch_any(r["architecture"]) == a]
        if not rs:
            continue
        if any(r["condition"] != rep[a] for r in rs):
            t.flags.append(f"W-1 for {ARCH_PRINT[a]} is at {rs[0]['condition']}, not the representative {rep[a]}")
        t.group(f"{ARCH_PRINT[a]} ({cond_eq(rs[0]['condition'])})")
        for r in rs:
            G = r["groups"]["all"]
            L, S = G["levels"], G["statistics"]
            la, lb = _armlab(r["loss_a"]), _armlab(r["loss_b"])
            d1, d2 = S["d_within5_pp"], S["d_median_abs_e_assigned_pp"]
            t.add([f"{la} vs {lb[0].lower() + lb[1:] if lb.startswith('Full') else lb}", f"{num(L['within5_pct_A'], 2)} / {num(L['within5_pct_B'], 2)}",
                   f"{num(d1['estimate'], 1, sign=True)} {ci(d1['ci95_hierarchical_bootstrap'], 1)}",
                   f"{pval(d1['p_day_label_swap'])} / {pval(d1.get('p_holm'))}",
                   f"{num(d2['estimate'], 2, sign=True)} {ci(d2['ci95_hierarchical_bootstrap'], 2)}",
                   f"{pval(d2['p_day_label_swap'])} / {pval(d2.get('p_holm'))}"])
    t.sources = [f"{W}/w1_cell_paired/w1_cell_paired.json"]
    uses(t, inp, "results[].groups.all.levels.within5_pct_{A,B}",
         "statistics.{d_within5_pp, d_median_abs_e_assigned_pp}.{estimate, ci95_hierarchical_bootstrap, "
         "p_day_label_swap, p_holm}")
    return t


def tableS9(inp, R, rep):
    t = Table("S9", "rank_sweep", "Every LoRA rank and full fine-tuning under the proposed objective",
              ["Condition", "Trainable parameters (% of network during training)", "Cells within 5% (%)",
               "Matched-cell IPE (%)", "Global IPE (%)", "Mean Dice", "AJI, cells", "Boundary F1", "Dry-mass CCC",
               "Runs that lost a class"], "SI")
    for a in R.archs:
        t.group(ARCH_PRINT[a])
        for c in CONDS:
            xs = R.get(a, c, PROPOSED)
            if not xs:
                continue
            tot, tr = xs[0].get("params_total"), xs[0].get("params_trainable")
            g = lambda k, dp, s=1.0: msd(R.col(a, c, PROPOSED, k), dp, s)  # noqa: E731
            t.add([cond_eq(c), f"{tr:,} ({num(100.0 * tr / tot, 2)}%)", g("a2_scored_within_5pct_of_all_cells_pct", 2),
                   g("a2_scored_cell_ipe_abs_median_pct", 3), g("ipe_global", 3, 100.0), g("dice_macro", 3),
                   g("a2_aji_cells", 3), g("bf1", 3), g("a2_scored_drymass_ccc", 3), lost(xs)])
    t.sources = [f"{CONS}/runs_last.json"]
    uses(t, inp, "params_trainable, params_total", "a2_scored_within_5pct_of_all_cells_pct",
         "a2_scored_cell_ipe_abs_median_pct", "ipe_global (x100)", "dice_macro", "a2_aji_cells", "bf1",
         "a2_scored_drymass_ccc", "n_lost_classes")
    return t


def tableS10(inp, R, rep):
    t = Table("S10", "per_class", "Per-class accuracy at the selected configurations",
              ["Class", "Dice", "IoU", "Class-specific IPE (%)", "Test images with the class", "Runs that lost the class"], "SI")
    for a in R.archs:
        c = rep[a]
        xs = R.get(a, c, PROPOSED)
        t.group(f"{ARCH_PRINT[a]} ({cond_eq(c)})")
        for k in CLASSES:
            n_img = sorted({int(x[f"n_images_with_{k}"]) for x in xs if x.get(f"n_images_with_{k}") is not None})
            t.add([k.capitalize(), msd(R.col(a, c, PROPOSED, f"dice_{k}"), 3), msd(R.col(a, c, PROPOSED, f"iou_{k}"), 3),
                   msd(R.col(a, c, PROPOSED, f"ipe_{k}"), 1, 100.0), ", ".join(map(str, n_img)) or "–",
                   f"{sum(bool(x.get(f'lost_{k}')) for x in xs)} of {len(xs)}"])
    t.sources = [f"{CONS}/runs_last.json"]
    uses(t, inp, "dice_<class>, iou_<class>, ipe_<class> (x100), n_images_with_<class>, lost_<class>")
    return t


def tableS11(inp, R, rep):
    q = inp.json(f"{CONS}/phase_quartile_fnr/phase_quartile_fnr.json")
    nb = q.get("bins", 4)
    t = Table("S11", "missed_pixels_by_phase_quartile",
              "Annotated cell pixels predicted as background (%), by phase quartile",
              ["Architecture", "Objective", "All pixels", "Lowest-phase quartile", "Second quartile", "Third quartile",
               "Highest-phase quartile"][: 3 + nb], "SI")
    for a in R.archs:
        c = rep[a]
        for i, arm in enumerate((DICE, PROPOSED)):
            rs = [r for r in q["runs"] if arch_any(r["architecture"]) == a and r["condition"] == c and r["loss"] == arm]
            t.add([f"{ARCH_PRINT[a]} ({cond_eq(c)})" if i == 0 else "",
                   "Dice alone" if arm == DICE else "Proposed (Dice + cIPP)",
                   msd([r["false_negative_rate_pct"] for r in rs], 2)] +
                  [msd([r[f"fnr_bin{j + 1}_pct"] for r in rs], 2) for j in range(nb)])
    t.sources = [f"{CONS}/phase_quartile_fnr/phase_quartile_fnr.json"]
    uses(t, inp, "runs[].{false_negative_rate_pct, fnr_bin1..4_pct}")
    return t


def tableS12(inp, R, rep):
    st = inp.json(f"{CONS}/storage/runs_last.json")
    ann = inp.json(f"{CONS}/storage/annotation.json")["trends"]
    t = Table("S12", "storage_trends_per_seed", "Storage-time trends per seed",
              ["Seed", "Quantity", "Spearman ρ", "p (images independent)", "p (date-block permutation)",
               "95% date-bootstrap interval of ρ", "ρ, leave-one-day-out range", "Change of median, early to late (%)",
               "ρ of bias with day", "p of bias (date-block permutation)"], "SI")

    def row(seed, lab, x, bias=True):
        b = x.get("bias_vs_annotation") if bias else None
        return [seed, lab, num(x["rho"], 3), pval(x["p_asymptotic"]), pval(x["p_block_permutation"]),
                ci(x["rho_ci95_date_bootstrap"], 3), ci(x["rho_leave_one_day_out"], 3),
                num(x["early_to_late_change_pct"], 2), num(b["rho"], 3) if b else "–",
                pval(b["p_block_permutation"]) if b else "–"]

    t.group("Annotation (scored reference cells)")
    t.add(row("–", "Area", ann["area"], False))
    t.add(row("–", "Dry mass", ann["dry_mass"], False))
    for a in R.archs:
        c = rep[a]
        t.group(f"{ARCH_PRINT[a]} ({cond_eq(c)})")
        for s in SEEDS:
            r = next((x for x in st["runs"] if x["arch"] == a and x["condition"] == c and x["arm"] == PROPOSED
                      and int(x["seed"]) == s), None)
            if r is None:
                t.flags.append(f"storage run {a} {c} seed {s} absent")
                continue
            t.add(row(str(s), "Area", r["area"]))
            t.add(row("", "Dry mass", r["dry_mass"]))
    t.sources = [f"{CONS}/storage/runs_last.json", f"{CONS}/storage/annotation.json"]
    uses(t, inp, "{area,dry_mass}.{rho, p_asymptotic, p_block_permutation, rho_ci95_date_bootstrap, "
                 "rho_leave_one_day_out, early_to_late_change_pct, bias_vs_annotation.{rho,p_block_permutation}}")
    return t


def _day_block(cells, pred: bool):
    """Per storage day: n, median area, median dry mass, class shares; reference (pred=False) or prediction."""
    out = {}
    for d in sorted({x["storage_day"] for x in cells}):
        g = [x for x in cells if x["storage_day"] == d]
        cls = [x["pred_class" if pred else "gt_class"] for x in g]
        out[int(d)] = {"n": len(g), "area": float(np.median([x["area_pred_um2" if pred else "area_um2"] for x in g])),
                       "dm": float(np.median([x["dm_pred_pg" if pred else "dm_gt_pg"] for x in g])),
                       "cls": {k: 100.0 * cls.count(k) / len(g) for k in CLASSES}}
    return out


def tableS13(inp, R, rep):
    t = Table("S13", "per_day_composition", "Per-day area, dry mass and class composition",
              ["Storage day", "Cells (matched fraction)", "Median area (µm²)", "Median dry mass (pg)",
               "Dry mass − annotation (pg)"] + [f"{k.capitalize()} (%)" for k in CLASSES], "SI")
    ref = None
    for a in R.archs:
        c = rep[a]
        per = []
        try:
            for s in SEEDS:
                C = [x for x in inp.cells(a, c, PROPOSED, s)["cells"] if x.get("instances") == "A2"]
                sc = [x for x in C if x.get("scored")]
                u = [x for x in sc if x.get("matched") and not x.get("pred_wrapped")]
                if ref is None:
                    ref = _day_block(sc, False)
                    t.group("Annotation (scored reference cells)")
                    for d, v in ref.items():
                        t.add([str(d), str(v["n"]), num(v["area"], 2), num(v["dm"], 2), "–"] +
                              [num(v["cls"][k], 1) for k in CLASSES])
                per.append(_day_block(u, True))
        except Untraceable as e:
            t.flags.append(f"{ARCH_PRINT[a]} {c}: {e}")
            t.group(f"{ARCH_PRINT[a]} ({cond_eq(c)}): UNTRACEABLE (per-run cells files needed)")
            continue
        t.group(f"{ARCH_PRINT[a]} ({cond_eq(c)}), mean of seeds {SEEDS[0]}–{SEEDS[-1]}")
        for d in ref:
            P = [p[d] for p in per if d in p]
            n = np.mean([p["n"] for p in P])
            t.add([str(d), f"{n:.0f} ({num(n / ref[d]['n'], 2)})", num(np.mean([p['area'] for p in P]), 2),
                   num(np.mean([p["dm"] for p in P]), 2), num(np.mean([p["dm"] for p in P]) - ref[d]["dm"], 2, sign=True)] +
                  [num(np.mean([p["cls"][k] for p in P]), 1) for k in CLASSES])
    if ref is None:
        t.group("Annotation (scored reference cells): UNTRACEABLE (per-run cells files needed)")
    t.sources = ["results/<arch>/<rank>/dice_cipp/seed<s>/cells_last.json (rescored)"]
    uses(t, inp, "cells (instances A2): scored; prediction = matched and not pred_wrapped",
         "area_um2 / area_pred_um2, dm_gt_pg / dm_pred_pg, gt_class / pred_class; per-day medians; mean over seeds; "
         "matched fraction = mean matched cells / scored reference cells of the day")
    return t


def tableS14(inp, R, rep):
    w7 = inp.json(f"{W}/w7_followup/w7_followup.json")
    A = w7["A_dry_mass_outliers"]
    thr = w7["metadata"].get("threshold_pg")
    crel = f"{W}/w7_followup/w7a_dry_mass_outliers.csv"
    try:
        cs = inp.csv(crel)
    except Untraceable as e:
        cs = []
        print(f"[tables] S14 area ratio: {e}", file=sys.stderr)
    blocks = {}
    for a in R.archs:
        blocks[a] = [A[k] for k in A if k.startswith(ARCH_FOLDER[a] + " ") and k.split()[1] == rep[a]]
    # the storage day with the largest share of outliers, over every architecture
    tot = {}
    for a, B in blocks.items():
        for b in B:
            for d, n in b["outliers_by_day"].items():
                tot[d] = tot.get(d, 0) + n
    dmax = max(tot, key=tot.get) if tot else None
    t = Table("S14", "dry_mass_outliers",
              f"Matched cells whose predicted dry mass differs from the reference by more than {num(thr, 0)} pg",
              ["Configuration", "Matched cells", f"> +{num(thr, 0)} pg", f"< −{num(thr, 0)} pg",
               f"Share > {num(thr, 0)} pg (%)", "Merged (%)", "Touching (%)", "Area ratio", f"Day {dmax} (%)",
               "Limits of agreement, all (pg)", f"Limits of agreement, without > {num(thr, 0)} pg (pg)"], "SI")
    for a in R.archs:
        B = blocks[a]
        if len(B) != len(SEEDS):
            t.add([f"{ARCH_PRINT[a]} ({cond_eq(rep[a])})"] + ["UNTRACEABLE"] * 10)
            t.flags.append(f"W-7 holds {len(B)} runs of {ARCH_PRINT[a]} at {rep[a]}")
            continue
        n = sum(b["matched_scored_cells"] for b in B)
        hi = sum(b["outliers_above_threshold"] for b in B)
        lo = sum(b["outliers_below_minus_threshold"] for b in B)
        o = [b["outliers_above_threshold"] + b["outliers_below_minus_threshold"] for b in B]
        share = [100.0 * x / b["matched_scored_cells"] for x, b in zip(o, B)]
        wsh = lambda key: 100.0 * sum(b[key]["outliers"] * x for b, x in zip(B, o)) / sum(o)  # noqa: E731
        day = 100.0 * sum(b["outliers_by_day"].get(dmax, 0) for b in B) / sum(o)
        ratio = "UNTRACEABLE"
        rows = [r for r in cs if arch_any(r["architecture"]) == a]
        if rows and all(r["rank"] == rep[a] for r in rows):
            meds = []
            for s in SEEDS:
                v = [float(r["area_pred_um2"]) / float(r["area_um2"]) for r in rows if int(r["seed"]) == s]
                if v:
                    meds.append(float(np.median(v)))
            ratio = num(np.mean(meds), 2)
        else:
            t.flags.append(f"Area ratio, {ARCH_PRINT[a]}: w7a_dry_mass_outliers.csv holds rank "
                           f"{sorted({r['rank'] for r in rows}) or 'none'}, not {rep[a]}; rerun W-7 (the CSV is stale)")
        mlo = lambda key: np.mean([b[key]["loa_low_pg"] for b in B])  # noqa: E731
        mhi = lambda key: np.mean([b[key]["loa_high_pg"] for b in B])  # noqa: E731
        t.add([f"{ARCH_PRINT[a]} ({cond_eq(rep[a])})", f"{n:,}", str(hi), str(lo),
               f"{num(100.0 * (hi + lo) / n, 1)} ({num(min(share), 1)}–{num(max(share), 1)})",
               num(wsh("share_merged"), 0), num(wsh("share_touching"), 0), ratio, num(day, 0),
               f"{num(mlo('limits_of_agreement_all'), 1)} to {num(mhi('limits_of_agreement_all'), 1, sign=True)}",
               f"{num(mlo('limits_of_agreement_without_outliers'), 1)} to "
               f"{num(mhi('limits_of_agreement_without_outliers'), 1, sign=True)}"])
    t.sources = [f"{W}/w7_followup/w7_followup.json", crel]
    uses(t, inp, "A_dry_mass_outliers['<arch> <rank> seed<s>']: matched_scored_cells, outliers_above/below, "
                 "share_merged/share_touching.outliers (weighted by outliers), outliers_by_day, "
                 "limits_of_agreement_{all,without_outliers} (mean over seeds)",
         "w7a csv: area ratio = mean over seeds of the median area_pred_um2 / area_um2 of the outliers")
    return t


def tableS15(inp, R, rep):
    B = inp.json(f"{W}/w7_followup/w7_followup.json")["B_class_confusion"]
    cols = CLASSES + ["(unmatched)", "(missed)"]
    t = Table("S15", "class_confusion", "Reference class against predicted class of the scored reference cells",
              ["Reference class"] + [k.capitalize() for k in CLASSES] + ["Unmatched", "Missed", "Total"], "SI")
    for a in R.archs:
        ks = [k for k in B if k.startswith(ARCH_FOLDER[a] + " ") and k.split()[1] == rep[a]]
        t.group(f"{ARCH_PRINT[a]} ({cond_eq(rep[a])})")
        if len(ks) != len(SEEDS):
            t.flags.append(f"W-7 confusion holds {len(ks)} runs of {ARCH_PRINT[a]} at {rep[a]}")
        for g in CLASSES:
            v = [sum(B[k].get(g, {}).get(p, 0) for k in ks) for p in cols]
            t.add([g.capitalize()] + [str(x) for x in v] + [str(sum(v))])
    t.sources = [f"{W}/w7_followup/w7_followup.json"]
    uses(t, inp, "B_class_confusion['<arch> <rank> seed<s>'][reference][predicted], summed over seeds")
    return t


def tableS16(inp, R, rep):
    w8 = inp.json(f"{W}/w8_signed_errors/w8_signed_errors.json")["mean_sd_over_seeds"]
    t = Table("S16", "signed_errors", "Signed and absolute integrated-phase errors of touching and isolated matched cells",
              ["Architecture", "Objective", "Cells", "Matched cells per seed", "Signed median error (%)",
               "Absolute median error (%)", "Cells within 5% (%)"], "SI")
    ms = lambda v, dp, sign=False: f"{num(v[0], dp, sign)} ± {num(v[1], dp)}"  # noqa: E731
    for a in R.archs:
        first = True
        for arm in (DICE, PROPOSED):
            for j, sub in enumerate(("touching", "isolated")):
                r = next((x for x in w8 if arch_any(x["arch"]) == a and x["rank"] == rep[a] and x["arm"] == arm
                          and x["subset"] == sub), None)
                lab_a = f"{ARCH_PRINT[a]} ({cond_eq(rep[a])})" if first else ""
                lab_o = ("Dice alone" if arm == DICE else "Proposed (Dice + cIPP)") if j == 0 else ""
                first = False
                if r is None:
                    t.add([lab_a, lab_o, sub.capitalize()] + ["UNTRACEABLE"] * 4)
                    t.flags.append(f"W-8 has no row {a} {rep[a]} {arm} {sub}")
                    continue
                t.add([lab_a, lab_o, sub.capitalize(), irange(r["n_matched_per_seed"]),
                       ms(r["signed_median_pct"], 2, True), ms(r["abs_median_pct"], 2), ms(r["within5_pct"], 1)])
    t.sources = [f"{W}/w8_signed_errors/w8_signed_errors.json"]
    uses(t, inp, "mean_sd_over_seeds[].{n_matched_per_seed, signed_median_pct, abs_median_pct, within5_pct}")
    return t


MAIN = [table1, table2, table3, table4, table5]
SUPP = [tableS1, tableS2, tableS3, tableS4, tableS5, tableS6, tableS7, tableS8, tableS9, tableS10, tableS11,
        tableS12, tableS13, tableS14, tableS15, tableS16]


# ================================================================================================ writing
def write(tables: List[Table], inp: Inputs, rep):
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import openpyxl
        from openpyxl.styles import Font
    except ImportError:
        openpyxl = None
        print("[tables] openpyxl not installed: XLSX skipped (pip install --user openpyxl)", file=sys.stderr)
    wb_all = openpyxl.Workbook() if openpyxl else None
    if wb_all:
        wb_all.remove(wb_all.active)
    for t in tables:
        d = t.to_dict()
        (OUT / f"{t.stem}.json").write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        with open(OUT / f"{t.stem}.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow([f"Table {t.id}. {t.title}"])
            w.writerow(t.columns)
            w.writerows(t.rows)
        if openpyxl:
            for wb in (openpyxl.Workbook(), wb_all):
                ws = wb.active if wb is not wb_all else wb.create_sheet(f"Table {t.id}")
                if wb is not wb_all:
                    ws.title = f"Table {t.id}"
                ws.append([f"Table {t.id}. {t.title}"])
                ws.append(t.columns)
                for i, r in enumerate(t.rows):
                    ws.append(r)
                    if i in t.group_rows:
                        ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
                ws["A1"].font = Font(bold=True)
                for c in ws[2]:
                    c.font = Font(bold=True)
                for col in ws.columns:
                    ws.column_dimensions[col[0].column_letter].width = min(48, max(10, *(len(str(c.value or "")) for c in col[1:])))
                if wb is not wb_all:
                    wb.save(OUT / f"{t.stem}.xlsx")
    if wb_all:
        wb_all.save(OUT / "all_tables.xlsx")
    (OUT / "all_tables.json").write_text(json.dumps({"representative": rep, "tables": [t.to_dict() for t in tables]},
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps(
        {"script": "additional_analysis/tables_for_manuscript.py", "allow_stale": inp.allow_stale,
         "representative_from": f"{SEL}/rank_selection.csv", "representative": rep, "files_read_sha256": inp.read,
         "tables": {t.id: {"sources": t.sources, "fields": t.fields} for t in tables}}, indent=2), encoding="utf-8")
    lines = ["# Values that could not be traced to a result file", ""]
    for t in tables:
        for f in t.flags:
            lines.append(f"- Table {t.id}: {f}")
    if len(lines) == 2:
        lines.append("None.")
    (OUT / "flags.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================================ docx compare
WNS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MNS = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def _text(el) -> str:
    out = []
    for n in el.iter():
        if n.tag in (WNS + "t", MNS + "t") and n.text:
            out.append(n.text)
        elif n.tag == WNS + "tab":
            out.append(" ")
        elif n.tag == WNS + "p" and out:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def docx_tables(path: Path) -> Dict[str, List[List[str]]]:
    """'1', 'S1', ... -> grid of cell text, keyed by the 'TABLE n' caption in the paragraphs before each table."""
    x = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    out, recent = {}, []
    for el in x.find(WNS + "body"):
        if el.tag == WNS + "p":
            t = _text(el)
            if t:
                recent = (recent + [t])[-4:]
        elif el.tag == WNS + "tbl":
            key = None
            for t in reversed(recent):
                m = re.match(r"^TABLE\s+(S?\d+)\b", t, re.I)
                if m:
                    key = m.group(1).upper()
                    break
            grid = []
            for tr in el.findall(WNS + "tr"):
                row = []
                for tc in tr.findall(WNS + "tc"):
                    sp = tc.find(WNS + "tcPr/" + WNS + "gridSpan")
                    row.append(_text(tc))
                    row += [""] * ((int(sp.get(WNS + "val")) if sp is not None else 1) - 1)
                grid.append(row)
            if key and key not in out:
                out[key] = grid
            recent = []
    return out


_SUPR = {v: k for k, v in zip("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")}
NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*×\s*10\^?-?\d+)?")


def _numbers(s: str):
    s = s.replace(MINUS, "-").replace("–", " – ").replace(" ", " ").replace(" ", " ")
    s = re.sub(r"10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "10^" + "".join(_SUPR[c] for c in m.group(1)), s)
    s = re.sub(r"(\d)\s*-\s*(\d)", r"\1 – \2", s)          # ranges written with a hyphen
    out = []
    for tok in NUM.findall(s):
        tok = tok.replace(",", "")
        m = re.match(r"([-+]?\d+(?:\.\d+)?)(?:\s*×\s*10\^?(-?\d+))?$", tok.strip())
        if not m:
            continue
        mant, e = m.group(1), int(m.group(2) or 0)
        dec = len(mant.split(".")[1]) if "." in mant else 0
        out.append((float(mant) * 10 ** e, 0.5 * 10 ** (e - dec)))
    return out


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def compare_cell(gen: str, pr: str) -> str:
    if gen.replace(" ", "") == pr.replace(" ", ""):
        return "exact"
    if "UNTRACEABLE" in gen:
        return "untraceable"
    g, p = _numbers(gen), _numbers(pr)
    if len(g) != len(p):
        return "MISMATCH" if (g or p) else ("text" if _norm_text(gen) != _norm_text(pr) else "exact")
    edge = False
    for (a, ta), (b, tb) in zip(g, p):
        d, h = abs(a - b), max(ta, tb)
        if d > 2 * h * 1.0001 + 1e-12:
            return "MISMATCH"
        edge |= d > h * 1.0001 + 1e-12
    if edge:
        # one unit of the last printed digit: the printed value was rounded twice or truncated; the generated
        # value is the stored value rounded once
        return "rounding-edge"
    return "rounding" if _norm_text(gen) == _norm_text(pr) else "text"


def compare(tables: List[Table], ms: Optional[Path], si: Optional[Path]):
    printed = {}
    for p in (ms, si):
        if p:
            printed.update(docx_tables(p))
    rows, summary = [], []
    for t in tables:
        P = printed.get(t.id)
        if P is None:
            summary.append((t.id, "not found in the documents", 0, 0, 0))
            continue
        G = [t.columns] + t.rows
        cnt = {"exact": 0, "rounding": 0, "rounding-edge": 0, "text": 0, "MISMATCH": 0, "untraceable": 0, "shape": 0}
        for i in range(max(len(G), len(P))):
            for j in range(max(len(G[i]) if i < len(G) else 0, len(P[i]) if i < len(P) else 0)):
                g = G[i][j] if i < len(G) and j < len(G[i]) else "<absent>"
                p = P[i][j] if i < len(P) and j < len(P[i]) else "<absent>"
                st = "shape" if "<absent>" in (g, p) else compare_cell(g, p)
                cnt[st] += 1
                if st != "exact":
                    rows.append({"table": t.id, "row": i, "col": j,
                                 "column": (t.columns[j] if j < len(t.columns) else ""),
                                 "row_label": (G[i][0] if i < len(G) and G[i] else ""),
                                 "status": st, "generated": g, "printed": p})
        summary.append((t.id, f"{len(P)}x{max(len(r) for r in P)} printed, {len(G)}x{len(t.columns)} generated",
                        cnt["exact"] + cnt["rounding"] + cnt["text"], cnt["MISMATCH"] + cnt["shape"], cnt["untraceable"], cnt))
    with open(OUT / "verification_report.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["table", "row", "col", "column", "row_label", "status", "generated", "printed"])
        w.writeheader()
        w.writerows(rows)
    L = ["# Generated tables against the printed tables", "",
         f"Manuscript: {ms}", f"Supplement: {si}", "",
         "exact: identical text; rounding: same numbers within the printed precision; text: numbers agree, wording "
         "differs; rounding-edge: a number differs by one unit of its last printed digit (the printed value was "
         "rounded twice or truncated; the generated one is the stored value rounded once); MISMATCH: a number "
         "differs by more than that; shape: the cell exists in one table only.", "",
         "| Table | Shape | agree (exact, rounding, text) | MISMATCH + shape | untraceable | detail |", "|---|---|---|---|---|---|"]
    for s in summary:
        L.append(f"| {s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]} | "
                 f"{', '.join(f'{k} {v}' for k, v in s[5].items() if v) if len(s) > 5 else ''} |")
    L += ["", "## Cells that differ", "", "| Table | row | column | status | generated | printed |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] in ("MISMATCH", "shape", "untraceable", "rounding-edge"):
            L.append(f"| {r['table']} | {r['row_label'][:40]} | {r['column'][:40]} | {r['status']} | {r['generated']} | {r['printed']} |")
    (OUT / "verification_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return summary


# ================================================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="pipeline root (default: the parent of this folder)")
    ap.add_argument("--manuscript", type=Path, help="manuscript .docx (FV-1.4) to compare with, cell by cell")
    ap.add_argument("--supplement", type=Path, help="Supplementary Information .docx (1.4) to compare with")
    ap.add_argument("--only", nargs="*", help="table ids to build, e.g. 1 2 S6")
    ap.add_argument("--allow-stale", action="store_true", help="accept per-run files without metadata.rescored (testing)")
    a = ap.parse_args(argv)
    global OUT
    OUT = a.root / "results" / "tables_for_manuscript"
    inp = Inputs(a.root, a.allow_stale)
    inp.stale = [] if a.allow_stale else stale_outputs(a.root)
    for d in inp.stale:
        print(f"[tables] STALE (refused): {d}", file=sys.stderr)
    R = Runs(inp)
    rep = representative(inp)
    tables = []
    for fn in MAIN + SUPP:
        tid = fn.__name__.replace("table", "")
        if a.only and tid not in a.only:
            continue
        try:
            tables.append(fn(inp, R, rep))
        except Untraceable as e:
            t = Table(tid, "untraceable", f"could not be built: {e}", ["status"], "main" if tid[0] != "S" else "SI")
            t.add([f"UNTRACEABLE: {e}"])
            t.flags.append(str(e))
            tables.append(t)
        print(f"[tables] Table {tid}: {len(tables[-1].rows)} rows" + (f", {len(tables[-1].flags)} flag(s)" if tables[-1].flags else ""))
    write(tables, inp, rep)
    if a.manuscript or a.supplement:
        for s in compare(tables, a.manuscript, a.supplement):
            print(f"[compare] Table {s[0]}: agree {s[2]}, differ {s[3]}, untraceable {s[4]}"
                  + (f", rounding-edge {s[5]['rounding-edge']}" if len(s) > 5 and s[5]['rounding-edge'] else ""))
    print(f"[tables] written to {OUT}")


if __name__ == "__main__":
    main()
