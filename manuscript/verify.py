#!/usr/bin/env python
"""Check the generated figures and tables against the experimental results, independently of how they were built.

    python manuscript/verify.py                    # after make_all.py
    python manuscript/verify.py --previous DIR     # also compare with a previous build's tables_for_manuscript/

Checks
  A  every numeric cell of the generated tables that keeps a previous table's definition equals that table's cell
     (previous build: manuscript/to_delete/**/tables_for_manuscript, or --previous);
  B  the representative runs' headline numbers recomputed from each run's own cells_last.json (cells as the unit)
     equal runs_last.json, including the matched dry-mass statistics;
  C  Table 2 means recomputed from runs_last.json equal the additional-analysis W-2 values it prints;
  D  the dry-mass statistics drawn in Fig. 7 equal runs_last.json (the tables' source) for the same runs;
  E  the generated tables and captions mention no rank, arm or architecture outside the results, every expected
     output exists, and Fig. 2's wavelength equals the configuration.
Writes manuscript/VERIFICATION.md and exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figtab.core import PROPOSED, Inputs, Runs, arch_dir, finite, representative, run_dir  # noqa: E402

TOL_REL = 1e-9


class Report:
    def __init__(self):
        self.lines, self.fail, self.n = [], 0, 0

    def ok(self, cond: bool, what: str, detail: str = ""):
        self.n += 1
        if not cond:
            self.fail += 1
            self.lines.append(f"* **FAIL** {what} {detail}")
        return cond

    def section(self, title: str, summary: str):
        self.lines.append(f"\n## {title}\n\n{summary}\n")


def read_csv(p: Path):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.reader(fh))


def nums(cell: str):
    """All numbers in a table cell, as printed (thousands separators removed, unicode minus/superscripts handled)."""
    s = cell.replace(",", "").replace("−", "-")
    sup = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
    s = re.sub(r"× 10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "e" + m.group(1).translate(sup), s)
    s = re.sub(r"(\d)\s*e", r"\1e", s)
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", s)]


# previous build file -> (new file, number of leading key columns), for the tables whose definitions did not change.
# Rows are matched by their key columns, not by position (Table S2 now also lists full fine-tuning).
SAME = {"main/table1_representative_configurations.csv": ("main/table1_representative_configurations.csv", 2),
        "main/table3_cell_level_phase_preservation.csv": ("main/table3_cell_level_phase_preservation.csv", 2),
        "main/table4_inference_cost.csv": ("main/table4_inference_cost.csv", 2),
        "main/table2_objective_comparison.csv": ("supplementary/tableS3_objective_comparison_pooled.csv", 2),
        "supplementary/tableS1_rank_sweep.csv": ("supplementary/tableS1_rank_sweep.csv", 2),
        "supplementary/tableS2_loss_and_rank_sweep.csv": ("supplementary/tableS2_loss_and_rank_sweep.csv", 3),
        "supplementary/tableS9_trainable_parameters.csv": ("supplementary/tableS10_trainable_parameters.csv", 2),
        "supplementary/tableS7_per_day_shape.csv": ("supplementary/tableS8_per_day_shape.csv", 5)}

ARM_OLD_TEXT = None


def _key(row, k):
    from figtab.core import TERM_LONG, TERM_SHORT
    out = []
    for c in row[:k]:
        c = c.strip().replace("–", "-")
        for code, longname in TERM_LONG.items():      # previous build wrote the terms out in full
            c = c.replace(longname, TERM_SHORT[code])
        c = c.replace(" (proposed)", "").replace(" (old full)", "").replace(" (new full)", "")
        out.append(c)
    return tuple(out)


def check_previous(rep: Report, prev: Path, out: Path):
    done = 0
    for old_rel, (new_rel, k) in SAME.items():
        po, pn = prev / old_rel, out / "tables" / new_rel
        if not po.exists() or not pn.exists():
            rep.lines.append(f"* skipped {old_rel}: {'previous' if not po.exists() else 'new'} file missing")
            continue
        O, N = read_csv(po), read_csv(pn)
        ho, hn = O[0], N[0]
        newrows = {}
        for r in N[1:]:
            newrows.setdefault(_key(r, k), []).append(r)
        cells = mism = unmatched = 0
        for orow in O[1:]:
            cand = newrows.get(_key(orow, k))
            if not cand:
                unmatched += 1
                continue
            nrow = cand.pop(0)
            for j, h in enumerate(ho):
                if j < k or h not in hn:
                    continue
                a, b = nums(orow[j]), nums(nrow[hn.index(h)])
                if not a:
                    continue
                cells += 1
                if len(a) != len(b) or any(abs(x - y) > 1e-9 + 1e-9 * abs(x) for x, y in zip(a, b)):
                    mism += 1
                    if mism <= 5:
                        rep.ok(False, f"{new_rel} row {_key(orow, k)} column '{h}'",
                               f"previous {orow[j]!r}, new {nrow[hn.index(h)]!r}")
        rep.ok(unmatched == 0, f"{new_rel}: every previous row found", f"({unmatched} not found)")
        rep.ok(mism == 0, f"{new_rel}: numeric cells equal the previous build", f"({mism} of {cells} differ)")
        rep.lines.append(f"* {new_rel} ← previous {old_rel}: {cells} numeric cells compared, {mism} differ, "
                         f"{unmatched} previous rows not found; new rows: {len(N) - 1}, previous: {len(O) - 1}")
        done += cells
    return done


def check_cells(rep: Report, inp: Inputs, R: Runs):
    import pandas as pd
    reps = representative(inp)
    k = 0
    stale, defs = [], []
    for a in R.archs:
        c = reps[a]["selected_condition"]
        for s in R.seeds:
            rel = f"{run_dir(a, c, PROPOSED, s)}/cells_{R.checkpoint}.json"
            if not inp.exists(rel):
                rep.lines.append(f"* skipped {rel}: not found")
                continue
            r = R.index[(a, c, PROPOSED, s)]
            cj = inp.json(rel)
            if not (cj.get("metadata") or {}).get("rescored"):
                stale.append(f"{a} {c} seed {s}")
            D = pd.DataFrame([x for x in cj["cells"] if x.get("instances") == "A2"])
            sc = D[D["scored"].astype(bool)]
            mt = sc[sc["matched"].astype(bool)]
            calc = {"a2_scored_n": len(sc),
                    "a2_scored_within_5pct_of_all_cells_pct": 100.0 * float((mt["e_matched"].abs() <= 0.05).sum()) / len(sc),
                    "a2_scored_match_pct": 100.0 * len(mt) / len(sc),
                    "a2_scored_cell_ipe_abs_median_pct": 100.0 * float(mt["e_matched"].abs().median())}
            for key, v in calc.items():
                rep.ok(np.isclose(v, r[key], rtol=1e-9, atol=1e-9), f"{a} {c} seed {s} {key}",
                       f"cells file {v!r}, runs_last {r[key]!r}")
                k += 1
            x, y = mt["dm_gt_pg"].to_numpy(float), mt["dm_pred_pg"].to_numpy(float)
            d = y - x
            ccc = 2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
            for name, v, rk in (("dry mass r", np.corrcoef(x, y)[0, 1], "a2_scored_drymass_pearson_r"),
                                ("dry mass CCC", ccc, "a2_scored_drymass_ccc"),
                                ("dry mass bias", d.mean(), "a2_scored_drymass_bias_pg"),
                                ("dry mass MdARE", 100 * np.median(np.abs(d) / np.abs(x)), "a2_scored_drymass_mdare_pct")):
                rep.ok(np.isclose(v, r[rk], rtol=1e-9, atol=1e-12), f"{a} {c} seed {s} {name} (cells file) = runs_last {rk}",
                       f"{v!r} vs {r[rk]!r}")
                k += 1
            if "pred_wrapped" in mt:
                nw = int(mt["pred_wrapped"].fillna(False).astype(bool).sum())
                if nw:
                    defs.append(f"{a} {c} seed {s}: {nw}")
    if defs:
        rep.lines.append("* NOTE (not an error): metrics_last.json computes the matched dry-mass statistics without the "
                         "matched cells whose prediction is near phase wrapping; runs_last.json, the tables and Fig. 7 keep "
                         "them. Such cells per run: " + "; ".join(defs))
    if stale:
        rep.lines.append("* NOTE: these per-run files predate the rescore (no metadata.rescored; no circularity), so "
                         "make_all.py builds Fig. 7 and Table S8 only from the rescored copies on the server: "
                         + "; ".join(stale))
    return k


def check_w2(rep: Report, inp: Inputs, R: Runs):
    rel = f"additional_analysis/results/w2_per_architecture/w2_per_architecture_{R.checkpoint}.json"
    if not inp.exists(rel):
        rep.lines.append(f"* skipped: {rel} not found"); return 0
    w2 = inp.json(rel)["results"]
    k = 0
    for a in R.archs:
        res = w2.get(arch_dir(a))
        if not res:
            continue
        conds = res["lora_ranks_pooled"]["conditions"]
        for cmp_ in res["lora_ranks_pooled"]["comparisons"]:
            for key, m in cmp_["metrics"].items():
                if key not in ("a2_scored_within_5pct_of_all_cells_pct", "a2_scored_cell_ipe_abs_median_pct", "ipe_global"):
                    continue
                for arm, mean in ((cmp_["loss_a"], m["mean_a"]), (cmp_["loss_b"], m["mean_b"])):
                    v = [x for c in conds for x in R.col(a, c, arm, key, 100.0 if key == "ipe_global" else 1.0)]
                    rep.ok(np.isclose(np.mean(v), mean, rtol=1e-9), f"W-2 {a} {arm} {key} mean", f"{np.mean(v)} vs {mean}")
                    k += 1
    return k


def check_fig7(rep: Report, inp: Inputs, R: Runs):
    import matplotlib
    matplotlib.use("Agg")
    from figtab import figures as F, style
    style.apply()
    from figtab.core import MissingInput
    try:
        fig, *_ = F.fig7_dry_mass(inp, R)
    except MissingInput as e:
        rep.lines.append(f"* skipped Fig. 7: {e}")
        return 0
    st = fig._figtab_stats
    reps = representative(inp)
    k = 0
    for a, v in st.items():
        r = R.index.get((a, v["cond"], PROPOSED, R.seeds[0]))
        if r is None:
            continue
        n_matched = round(r["a2_scored_n"] * r["a2_scored_match_pct"] / 100.0)
        for name, x, y in (("n", v["n"], n_matched), ("r", v["r"], r["a2_scored_drymass_pearson_r"]),
                           ("CCC", v["ccc"], r["a2_scored_drymass_ccc"]), ("bias", v["bias"], r["a2_scored_drymass_bias_pg"]),
                           ("lower limit", v["loa"][0], r["a2_scored_drymass_loa_low"]),
                           ("upper limit", v["loa"][1], r["a2_scored_drymass_loa_high"]),
                           ("MdARE", v["mdare"], r["a2_scored_drymass_mdare_pct"]),
                           ("MdARE = cell integrated-phase error", v["mdare"], r["a2_scored_cell_ipe_abs_median_pct"])):
            rep.ok(np.isclose(x, y, rtol=1e-6, atol=1e-9), f"Fig. 7 {a} {name}", f"figure {x!r}, runs_last {y!r}")
            k += 1
    return k


def check_outputs(rep: Report, inp: Inputs, R: Runs, out: Path):
    src = json.loads((out / "SOURCES.json").read_text(encoding="utf-8"))
    for m in src["outputs"]:
        for f in m["files"]:
            rep.ok((out / f).exists(), f"output exists: {f}")
    rep.ok(not src.get("skipped"), "no product was skipped", str(src.get("skipped")))
    allowed_r = {c[1:] for c in R.lora_conditions}
    bad = []
    for p in list((out / "tables").rglob("*.csv")) + [out / "captions.md"]:
        if not p.exists():
            continue
        for r in re.findall(r"r = (\d+)(?![.\d])", p.read_text(encoding="utf-8-sig")):
            if r not in allowed_r:
                bad.append(f"{p.name}: r = {r}")
    rep.ok(not bad, "every rank printed exists in the results", "; ".join(bad[:5]))
    cfg = inp.config()
    svg = HERE / "assets" / "fig2_setup" / "fig2_setup.png"
    if cfg and svg.exists():
        rep.ok(abs(cfg["physics"]["wavelength_um"] * 1000 - 666) < 1e-9, "Fig. 2 wavelength (666 nm) = config wavelength")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parent))
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--previous", default=None, help="folder holding main/ and supplementary/ of a previous build")
    ap.add_argument("--checkpoint", default="last")
    ap.add_argument("--allow-stale", action="store_true", help="TESTING ONLY (see make_all.py)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    inp = Inputs(Path(a.root)); inp.allow_stale = a.allow_stale; R = Runs(inp, a.checkpoint)
    rep = Report()
    prev = Path(a.previous) if a.previous else None
    if prev is None:
        cands = sorted((HERE / "to_delete").rglob("tables_for_manuscript")) + [HERE / "tables_for_manuscript"]
        cands = [c for c in cands if (c / "main" / "table1_representative_configurations.csv").exists()
                 and "old_data" not in c.as_posix()]
        prev = cands[0] if cands else None
    rep.section("A. Tables against the previous build", f"Previous build: `{prev}`" if prev else "No previous build found.")
    nA = check_previous(rep, prev, out) if prev else 0
    rep.section("B. Representative runs recomputed from cells_last.json", "")
    nB = check_cells(rep, inp, R)
    rep.section("C. Table 2 against runs_last.json", "")
    nC = check_w2(rep, inp, R)
    rep.section("D. Fig. 7 against runs_last.json", "")
    nD = check_fig7(rep, inp, R)
    rep.section("E. Outputs, labels, Fig. 2", "")
    check_outputs(rep, inp, R, out)
    head = [f"# Verification of the generated figures and tables", "",
            f"{rep.n} checks, **{rep.fail} failed**. Numeric cells compared with the previous build: {nA}; "
            f"values recomputed from per-cell files: {nB}; Table 2 means: {nC}; Fig. 7 statistics: {nD}."]
    (out / "VERIFICATION.md").write_text("\n".join(head + rep.lines) + "\n", encoding="utf-8")
    print("\n".join(head))
    for l in rep.lines:
        if "FAIL" in l:
            print(l)
    return 1 if rep.fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
