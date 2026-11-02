#!/usr/bin/env python
"""Build the line-by-line edit guides and the revised manuscript / SI from the revision specs.

    python artifacts/code/build_revision_docs.py            (needs pandoc; run make_tables.py and make_figures.py first)

Inputs   artifacts/manuscript_in_prog/{revised_manuscript_fv - ui.docx, supplementry_info_fv - ui.docx}   (the CURRENT author files)
         artifacts/manuscript_revision/{manuscript_spec.py, si_spec.py}                                  (what to do with every block)
         artifacts/tables/**, artifacts/figures/**                                                       (generated results)
Outputs  artifacts/manuscript_revision/
             EDIT_GUIDE_MANUSCRIPT.md, EDIT_GUIDE_SI.md   for every paragraph of the current file: original text, action, complete new text, reason
             REVISED_MANUSCRIPT.md/.docx, REVISED_SI.md/.docx   the revised documents (changed text highlighted in yellow)
             NUMBER_CHECK.md                              numbers in the new text that were not found in the generated tables

Checks: every block of the current file is covered exactly once, in order (the build stops otherwise, e.g. after the author edited the
Word file: then update the "find" text of the affected entry); every number in the new text is looked up in the generated tables.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
REV = ART / "manuscript_revision"
SRC = ART / "manuscript_in_prog"
TAB = ART / "tables"
FIG = ART / "figures"

MANUSCRIPT_DOCX = SRC / "revised_manuscript_fv - ui.docx"
SI_DOCX = SRC / "supplementry_info_fv - ui.docx"

FIG_FILES = {"F3": "main/F3_per_class_dice", "F4": "main/F4_rank_sweep", "F5": "main/F5_training_curves", "F6": "main/F6_loss_ablation",
             "F7": "main/F7_lora_vs_full_finetuning", "F8": "main/F8_accuracy_vs_latency", "F9": "main/F9_dry_mass_agreement",
             "F10": "main/F10_storage_trends", "F11": "main/F11_composition", "FQ1": "main/FQ1_qualitative_segmentation",
             "FS1": "si/FS1_rank_sweep_per_class", "FS2": "si/FS2_ablation_ipe_heatmap", "FS3": "si/FS3_percell_ipe_cdf",
             "FS4": "si/FS4_robustness", "FS5": "si/FS5_parameters_vs_rank", "FS6": "si/FS6_error_strata",
             "FS7": "si/FS7_training_curves_all_ranks", "FS8": "si/FS8_annotation_trends", "FQS1": "si/FQS1_gradient_alignment",
             "FQS2": "si/FQS2_storage_examples"}
SI_TABLE_FILES = {"T1": "main/T1_dataset", "S1": "si/S1_rank_sweep", "S2": "si/S2_rank_sweep_per_seed", "S3": "si/S3_rank_paired",
                  "S4": "si/S4_ablation_all_ranks", "S5": "si/S5_ablation_paired", "S7": "si/S7_per_class_ipe", "S8": "si/S8_parameters_by_rank",
                  "S9": "si/S9_benchmark_per_seed", "S10": "si/S10_robustness", "S11": "si/S11_wrapping", "S12": "si/S12_error_strata",
                  "S13": "si/S13_cell_metrics_all_configs", "S14": "si/S14_reference_sensitivity"}


def load_spec(name: str):
    spec = importlib.util.spec_from_file_location(name, REV / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MS = load_spec("manuscript_spec")


# ------------------------------------------------------------------------------------------------ reading the author's file
def unmark(s: str) -> str:
    """Remove the yellow-highlight spans of the author's file ([text]{.mark})."""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\[((?:\\.|[^\[\]\\])*)\]\{\.mark\}", r"\1", s)     # escaped brackets (citations \\[1\\]) allowed inside
    return s


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", unmark(s))       # e.g. the ohm sign and the Greek capital omega compare equal
    s = re.sub(r"\\(.)", r"\1", s)
    s = re.sub(r"^\d+\.\s+", "", s.strip())
    s = s.replace("*", "").replace("^", "")
    return re.sub(r"\s+", " ", s).strip()


def read_blocks(docx: Path, media_dir: Path) -> List[str]:
    md = subprocess.run(["pandoc", str(docx), "-t", "markdown", "--wrap=none", f"--extract-media={media_dir}"],
                        check=True, capture_output=True, text=True).stdout
    md = md.replace(f"{media_dir}/media/", "media/")       # pandoc writes absolute paths for extracted images
    return [b.strip() for b in re.split(r"\n\s*\n", md) if b.strip()]


def assign(blocks: List[str], entries: List[Dict]) -> List[Dict]:
    """Attach the original block(s) to every entry; every block must be used exactly once, in order."""
    i, out = 0, []
    for k, e in enumerate(entries):
        e = dict(e, id=k + 1)
        f = e["find"]
        if f is None:
            e["orig"] = []
            out.append(e)
            continue
        start, end = (f, f) if isinstance(f, str) else f
        if i >= len(blocks) or not norm(blocks[i]).startswith(norm(start)):
            raise SystemExit(f"entry {k + 1} ({start[:50]!r}) does not match the next block of the file:\n  {norm(blocks[i])[:160]!r}\n"
                             "Update the 'find' text of this entry (the Word file was probably edited).")
        j = i
        while not norm(blocks[j]).startswith(norm(end)) or (j == i and end != start and not norm(blocks[j]).startswith(norm(end))):
            j += 1
            if j >= len(blocks):
                raise SystemExit(f"entry {k + 1}: end text {end!r} not found after {start!r}")
        e["orig"] = blocks[i:j + 1]
        i = j + 1
        out.append(e)
    if i != len(blocks):
        raise SystemExit(f"{len(blocks) - i} block(s) at the end of the file are not covered, first: {norm(blocks[i])[:120]!r}")
    return out


# ------------------------------------------------------------------------------------------------ tables for the manuscript
def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / f"{name}.csv", dtype=str, keep_default_na=False)


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    esc = lambda v: str(v).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(esc(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] +
                     ["| " + " | ".join(esc(v) for v in r) + " |" for r in df.values.tolist()])


def table_seg() -> str:
    t2, t3, t3b = csv("main/T2_preferred"), csv("main/T3_same_rank"), csv("main/T3_same_rank_b")
    rows = [t2.iloc[0], t2.iloc[1], t3.iloc[1], t2.iloc[2]]
    role = ["preferred; same rank", "preferred", "same rank", "baseline; same rank"]
    cols = ["Architecture", "LoRA rank", "Trainable params", "Mean Dice", "Mean IoU", "AJI", "Boundary F1", "Global IPE (%)",
            "Runs with a lost class"]
    df = pd.DataFrame([r[cols] for r in rows]).reset_index(drop=True)
    df.insert(2, "Role", role)
    return md_table(df) + "\n\nSeed-paired differences at *r* = 8 (mean [95% t-interval]):\n\n" + md_table(t3b)


def table_stab() -> str:
    return md_table(csv("main/T4_training_stability"))


def table_ft() -> str:
    t = csv("main/T6_lora_vs_full_ft")
    pref = {"MobileSAM": "8", "EdgeSAM": "32", "MobileNet-UNet": "8"}
    keep = t[[r["LoRA rank"] in (pref[r["Architecture"]], f"{pref[r['Architecture']]} (seed 42 only)", "full fine-tuning")
              for _, r in t.iterrows()]]
    return md_table(keep[["Architecture", "LoRA rank", "Seeds", "Trainable params", "Mean Dice", "AJI", "Boundary F1", "Global IPE (%)",
                          "Runs with a lost class"]])


def table_abl() -> str:
    t, tb = csv("main/T5_loss_ablation"), csv("main/T5_loss_ablation_b")
    sel = ["MobileSAM r = 8", "EdgeSAM r = 32", "MobileNet-UNet r = 8"]
    a = t[t["Configuration"].isin(sel)][["Configuration", "Objective", "Mean Dice", "Boundary F1", "Global IPE (%)", "Runs with a lost class"]]
    b = tb[tb["Configuration"].isin(sel)]
    b = b[["Configuration"] + [c for c in b.columns if c.startswith("Δ IPE")]]
    return md_table(a) + "\n\nSeed-paired differences of the global IPE (percentage points, mean [95% t-interval]):\n\n" + md_table(b)


def table_mass() -> str:
    t = csv("main/T7_phase_and_mass")
    return md_table(t[["Configuration", "Global IPE (%)", "Per-cell |IPE| median (%)", "Per-cell |IPE| P90 (%)",
                       "Annotated cells matched (IoU ≥ 0.5, %)", "Predicted / annotated cells", "Dry-mass MdARE (%)", "Dry-mass bias (pg)",
                       "LoA low (pg)", "LoA high (pg)", "Day-0 median dry mass (pg)"]])


def table_bench() -> str:
    t = csv("main/T8_efficiency")
    return md_table(t.drop(columns=["Trainable params, r = 8"]))


def table_day() -> str:
    t = csv("main/T9_storage_per_day")
    cols = ["Day", "Annotated cells", "Area (µm²), annotation", "Dry mass (pg), annotation", "Composition (%), annotation",
            "Area (µm²), MobileSAM r8", "Dry mass (pg), MobileSAM r8", "Area (µm²), EdgeSAM r32", "Dry mass (pg), EdgeSAM r32"]
    return md_table(t[cols])


def table_stat() -> str:
    return md_table(csv("main/T10_storage_statistics"))


TABLES = {"seg": table_seg, "stab": table_stab, "ft": table_ft, "abl": table_abl, "mass": table_mass, "bench": table_bench,
          "day": table_day, "stat": table_stat}


def si_table(key: str) -> str:
    df = csv(SI_TABLE_FILES[key])
    if key == "S2":
        return md_table(df)
    return md_table(df)


# ------------------------------------------------------------------------------------------------ rendering
def labels(s: str) -> str:
    s = re.sub(r"\[\[tab:(\w+)\]\]", lambda m: f"Table {MS.MAIN_TABLES[m.group(1)]}", s)
    s = re.sub(r"\[\[fig:(\w+)\]\]", lambda m: f"Fig. {MS.MAIN_FIGS[m.group(1)]}", s)
    s = re.sub(r"\[\[stab:(\w+)\]\]", lambda m: f"Table S{MS.SI_TABLES[m.group(1)]}", s)
    s = re.sub(r"\[\[sfig:(\w+)\]\]", lambda m: f"Fig. S{MS.SI_FIGS[m.group(1)]}", s)
    s = re.sub(r"\[\[STAB:(\w+)\]\]", lambda m: f"TABLE S{MS.SI_TABLES[m.group(1)]}", s)
    s = re.sub(r"\[\[SFIG:(\w+)\]\]", lambda m: f"Fig. S{MS.SI_FIGS[m.group(1)]}", s)
    return s


def fig_md(key: str, rel_to: Path) -> str:
    f = FIG / f"{FIG_FILES[key]}.png"
    if not f.exists():
        return (f"[FIGURE NOT YET GENERATED: {FIG_FILES[key]}.png. Run `python artifacts/code/make_qualitative_figures.py` on the server "
                "(needs data/processed), then rebuild.]{.mark}")
    return f"![]({Path('..') / 'figures' / (FIG_FILES[key] + '.png')}){{width=6.5in}}"


def expand(text: str, for_doc: bool) -> str:
    """Placeholders -> numbers, tables, figures. for_doc: the revised document (otherwise the guide, which names the files)."""
    text = labels(text)
    def tab(m):
        return TABLES[m.group(1)]()
    def stab(m):
        return si_table(m.group(1))
    def fig(m):
        return fig_md(m.group(1), REV) if for_doc else f"*[insert figure file artifacts/figures/{FIG_FILES[m.group(1)]}.png]*"
    text = re.sub(r"\{\{TABLE:(\w+)\}\}", tab, text)
    text = re.sub(r"\{\{STABLE:(\w+)\}\}", stab, text)
    return re.sub(r"\{\{FIG:(\w+)\}\}", fig, text)


def paragraphs(text: str) -> List[str]:
    """Spec text -> paragraphs (single newlines inside a paragraph are joined; tables and headings stay as they are)."""
    out = []
    for chunk in re.split(r"\n\s*\n", text.strip()):
        lines = chunk.split("\n")
        if any(l.lstrip().startswith("|") for l in lines) or chunk.lstrip().startswith("!["):
            out.append(chunk)
        else:
            out.append(" ".join(l.strip() for l in lines))
    return out


def highlight(p: str) -> str:
    if p.startswith("|") or p.startswith("![") or p.startswith("[FIGURE NOT YET") or not p.strip():
        return p
    m = re.match(r"^(#+ )(.*)$", p)
    if m:
        return f"{m.group(1)}[{m.group(2)}]{{.mark}}"
    return f"[{p}]{{.mark}}"


def keep_text(e: Dict, media_rel: str) -> str:
    if e["new"]:
        return e["new"]
    text = "\n\n".join(unmark(b) for b in e["orig"])
    return text.replace("](media/", f"]({media_rel}/media/")


def revised_markdown(entries: List[Dict], media_rel: str) -> str:
    parts = []
    for e in entries:
        if e["action"] == "DELETE":
            continue
        if e["action"] == "KEEP":
            parts.append(expand(keep_text(e, media_rel), True))
            continue
        for p in paragraphs(expand(e["new"], True)):
            parts.append(highlight(p))
    return "\n\n".join(parts) + "\n"


def short(s: str, n: int = 30) -> str:
    w = norm(s).split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def quote(s: str) -> str:
    return "\n".join("> " + l if l.strip() else ">" for l in s.split("\n"))


def guide(entries: List[Dict], title: str, docname: str, prefix: str) -> str:
    n = {a: sum(1 for e in entries if e["action"] == a) for a in ("KEEP", "REPLACE", "DELETE", "NEW")}
    L = [f"# {title}", "",
         f"Source file: `artifacts/manuscript_in_prog/{docname}` (your current version). Every paragraph of that file appears below, in "
         "order, with one of four actions:", "",
         "* **KEEP**: leave the paragraph exactly as it is.",
         "* **REPLACE**: select the whole paragraph in Word (find it with the quoted first words) and paste the complete new text. "
         "Never delete only part of a sentence: every new paragraph is complete and reads correctly.",
         "* **DELETE**: remove the whole paragraph (or table / figure).",
         "* **NEW**: insert the new text at this position (after the previous paragraph of this list).", "",
         f"Counts: {n['KEEP']} keep, {n['REPLACE']} replace, {n['DELETE']} delete, {n['NEW']} new. The finished result of all edits is "
         f"`REVISED_{'MANUSCRIPT' if prefix == 'M' else 'SI'}.docx` in this folder (changed text highlighted in yellow), so you can also work "
         "from that file instead. 'R1-4' = reviewer 1, comment 4 (for the response letter).", ""]
    L += ["## Overview of all changes (tick when done)", "", "| Done | Id | Action | Where | First words |", "|---|---|---|---|---|"]
    sec = ""
    for e in entries:
        if e["new"].startswith("#"):
            sec = e["new"].lstrip("# ").strip()
        elif e["action"] == "KEEP" and e["orig"] and len(norm(e["orig"][0])) < 40 and not norm(e["orig"][0]).endswith("."):
            sec = norm(e["orig"][0])
        if e["action"] == "KEEP":
            continue
        first = short(e["orig"][0], 8) if e["orig"] else short(labels(e["new"]), 8)
        L.append(f"| [ ] | {prefix}{e['id']:03d} | {e['action']} | {sec} | {first.replace('|', '/')} |")
    L += ["", "## Paragraph by paragraph", ""]
    section = ""
    for e in entries:
        tag = f"{prefix}{e['id']:03d}"
        head = e["new"].startswith("#") or (e["orig"] and len(norm(e["orig"][0])) < 90 and not norm(e["orig"][0]).endswith(".")
                                            and e["action"] == "KEEP" and e["new"].startswith("#"))
        if e["new"].startswith("#"):
            section = e["new"].lstrip("# ").strip()
            L += ["", f"## {section}", ""]
        elif e["action"] == "KEEP" and e["orig"] and len(norm(e["orig"][0])) < 40 and not norm(e["orig"][0]).endswith("."):
            section = norm(e["orig"][0])
        orig = "\n\n".join(unmark(b) for b in e["orig"])
        is_table = len(e["orig"]) > 1 and any(b.lstrip().startswith("---") or re.match(r"^\S.*\s{3,}\S", b) for b in e["orig"][1:])
        if e["action"] == "KEEP":
            if e["orig"] and not head:
                L.append(f"**{tag} KEEP** \"{short(orig, 14)}\"" + (f"  \n*Note:* {labels(e['why'])}" if e["why"] else ""))
                L.append("")
            elif head and e["why"]:
                L += [f"*{tag}: {labels(e['why'])}*", ""]
            continue
        L.append(f"### {tag} {e['action']}" + (f": {section}" if section else ""))
        L.append("")
        if e["orig"]:
            if is_table:
                L += [f"**Find:** \"{short(e['orig'][0], 14)}\" (the old table: caption and all {len(e['orig']) - 1} table block(s))", ""]
                L += ["**Original caption:**", "", quote(unmark(e["orig"][0])), ""]
            else:
                L += [f"**Find:** \"{short(orig, 10)}\"", "", "**Original:**", "", quote(orig), ""]
        if e["action"] == "DELETE":
            L += ["**Delete the whole block.**", ""]
        else:
            L += ["**New text (complete):**" if e["action"] == "REPLACE" else "**Insert:**", "",
                  "\n\n".join(paragraphs(expand(e["new"], False))), ""]
        if e["why"]:
            L += [f"**Why:** {labels(e['why'])}", ""]
        L += ["---", ""]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ number check
NUM = re.compile(r"(?<![\w.])[-−+]?\d+(?:[.,]\d+)*(?:\.\d+)?")


def _rounded_set(text: str) -> set:
    """Every number of ``text`` (thousands separators removed) rounded to 0..4 decimals, as strings, signs dropped."""
    out = set()
    variants = text.replace(",", " ") + " " + text.replace(",", "")     # comma as a field separator and as a thousands separator
    for m in re.finditer(r"\d+(?:\.\d+)?(?:e[-+]?\d+)?", variants):
        try:
            v = abs(float(m.group(0)))
        except ValueError:
            continue
        for d in range(5):
            out.add(f"{v:.{d}f}")
            if 0 < v < 1:
                out.add(f"{100 * v:.{d}f}")           # fractions written as percentages
    return out


def number_check(entries_by_doc: Dict[str, List[Dict]], orig_text: str) -> str:
    gen = _rounded_set(" ".join(p.read_text() for p in TAB.rglob("*.csv")) + (ART / "numbers.json").read_text())
    orig = _rounded_set(orig_text)
    allowed = {"0", "1", "2", "3", "4", "5", "6", "8", "9", "10", "11", "15", "16", "20", "25", "32", "42", "43", "44", "47", "50", "60",
               "90", "95", "99", "100", "180", "183", "200", "256", "500", "768", "1024", "0.05", "0.1", "0.2", "0.3", "0.5", "1.0",
               "1.5", "2.0", "0.001", "99.9", "0.75", "40", "666", "0.1441", "30", "37", "12", "19", "23", "27", "13", "14", "7"}
    rows, n_checked = [], 0
    for doc, entries in entries_by_doc.items():
        for e in entries:
            if e["action"] in ("KEEP", "DELETE"):
                continue
            txt = re.sub(r"\{\{[^}]*\}\}|\[\[[^\]]*\]\]|\$[^$]*\$", " ", e["new"])
            txt = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", txt)         # citations
            for m in re.finditer(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?", txt):
                a = m.group(0).replace(",", "")
                n_checked += 1
                if a in allowed or re.fullmatch(r"(19|20)\d\d", a) or a in gen or a in orig:
                    continue
                rows.append(f"| {doc} | {e['id']} | {m.group(0)} | {short(e['new'], 12)} |")
    head = ["# Number check", "", f"{n_checked} numbers in the new or replaced text were compared with the generated tables "
            "(`artifacts/tables/**/*.csv`, `numbers.json`, any value rounded to the same number of decimals) and with the original "
            "manuscript. Numbers listed here were not found and must be checked by hand.", "",
            "| Doc | Entry | Number | Context |", "|---|---|---|---|"]
    return "\n".join(head + (rows or ["| | | none | |"])) + "\n"


# ------------------------------------------------------------------------------------------------ driver
def build_one(docx: Path, spec, prefix: str, name: str, title: str) -> List[Dict]:
    media = REV / f"source_media_{prefix}"
    blocks = read_blocks(docx, media)
    entries = assign(blocks, spec.ENTRIES)
    (REV / f"EDIT_GUIDE_{name}.md").write_text(guide(entries, title, docx.name, prefix))
    md = revised_markdown(entries, media.name)
    (REV / f"REVISED_{name}.md").write_text(md)
    ref = ["--reference-doc", str(docx)]
    cmd = ["pandoc", f"REVISED_{name}.md", "-o", f"REVISED_{name}.docx", "--resource-path", str(REV)] + ref
    r = subprocess.run(cmd, cwd=REV, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        r = subprocess.run(cmd[:-2], cwd=REV, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(r.stderr)
    for w in r.stderr.splitlines():
        if "Could not fetch" in w or "not found" in w:
            print("  pandoc:", w)
    n = {a: sum(1 for e in entries if e["action"] == a) for a in ("KEEP", "REPLACE", "DELETE", "NEW")}
    print(f"[{name}] {len(blocks)} blocks covered by {len(entries)} entries {n} -> EDIT_GUIDE_{name}.md, REVISED_{name}.md/.docx")
    return entries


def main() -> int:
    si = load_spec("si_spec")
    m = build_one(MANUSCRIPT_DOCX, MS, "M", "MANUSCRIPT", "Manuscript: line-by-line edit guide")
    s = build_one(SI_DOCX, si, "S", "SI", "Supplementary Information: line-by-line edit guide")
    orig = " ".join(" ".join(e["orig"]) for e in m + s)
    (REV / "NUMBER_CHECK.md").write_text(number_check({"manuscript": m, "SI": s}, orig))
    print("[numbers] see NUMBER_CHECK.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
