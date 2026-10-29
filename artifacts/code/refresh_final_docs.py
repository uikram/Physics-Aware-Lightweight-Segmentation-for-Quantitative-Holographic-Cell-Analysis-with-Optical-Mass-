"""Apply text and table corrections to the edited publication files WITHOUT touching their figures.

    python artifacts/code/refresh_final_docs.py

The Word files in artifacts/manuscript_final/ contain figures inserted by hand (from make_data_figures.py). Rebuilding them
with build_final_docs.py would bring back the placeholders. This script builds fresh documents in a temporary folder and
copies into the edited files only the paragraphs and tables whose text differs; paragraphs that hold a picture are never
replaced. The body structure of both versions must match element by element, otherwise the script stops.
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import docx
from docx.oxml.ns import qn

import build_final_docs as B
import docx_tools as T

FINAL = B.OUTD
PIC = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"


def text(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def transplant(fresh_path: Path, target_path: Path, label: str) -> list:
    fresh = docx.Document(fresh_path)
    target = docx.Document(target_path)
    fb = list(fresh.element.body.iterchildren())
    tb = list(target.element.body.iterchildren())
    # the fresh build may only ADD elements at the end of the body (before sectPr); everything before must align
    n = len(tb) - 1
    if len(fb) < len(tb) or any(a.tag != b.tag for a, b in zip(fb[:n], tb[:n])):
        raise RuntimeError(f"{label}: body structure differs ({len(fb)} vs {len(tb)} elements); refusing to transplant")
    changed = []
    for k, el in enumerate(fb[n:-1]):
        tb[-1].addprevious(copy.deepcopy(el))
        changed.append((n + k, "(new)", text(el)[:120]))
    for i, (a, b) in enumerate(zip(fb[:n], tb[:n])):
        if a.tag not in (qn("w:p"), qn("w:tbl")) or text(a) == text(b):
            continue
        if next(b.iter(PIC), None) is not None or next(a.iter(PIC), None) is not None:
            continue
        b.addprevious(copy.deepcopy(a))
        b.getparent().remove(b)
        changed.append((i, text(b)[:120], text(a)[:120]))
    target.save(target_path)
    return changed


def main() -> int:
    B.extra_facts()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        B.build_ms("yellow", tmp / "ms.docx")
        B.build_si("yellow", tmp / "si.docx")
        log = []
        for fresh, name in ((tmp / "ms.docx", "revised_manuscript_final.docx"),
                            (tmp / "si.docx", "supplementary_information_final.docx")):
            for i, old, new in transplant(fresh, FINAL / name, name):
                log.append(f"| {name} element {i} | refreshed | {old}… | {new}… |")
    B.write_log(FINAL / "change_log.md")
    with open(FINAL / "change_log.md", "a") as fh:
        fh.write("\n## Refresh of the hand-edited files (figures kept)\n\n| Element | Action | Before | After |\n|---|---|---|---|\n")
        fh.write("\n".join(log) + "\n")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
