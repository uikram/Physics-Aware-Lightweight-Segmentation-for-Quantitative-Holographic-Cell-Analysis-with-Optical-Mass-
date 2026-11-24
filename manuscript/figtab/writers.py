"""Writing tables (CSV, Markdown, one XLSX and one DOCX with every table) and figures (PDF, SVG, PNG)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Table:
    number: str                 # "1", "S3"
    slug: str                   # file stem
    title: str
    columns: List[str]
    rows: List[List[str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    group_cols: int = 1         # leading columns that repeat between rows; blanked in the DOCX for readability

    @property
    def section(self) -> str:
        return "supplementary" if self.number.startswith("S") else "main"

    @property
    def label(self) -> str:
        return f"Table {self.number}"


def write_table(t: Table, out: Path) -> List[Path]:
    d = out / "tables" / t.section
    d.mkdir(parents=True, exist_ok=True)
    stem = f"table{t.number}_{t.slug}"
    with open(d / f"{stem}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(t.columns)
        w.writerows(t.rows)
    md = [f"# {t.label}. {t.title}", "", f"**Source:** {', '.join(f'`{s}`' for s in t.sources) or '–'}", "",
          "| " + " | ".join(t.columns) + " |", "|" + "|".join("---" for _ in t.columns) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in t.rows]
    if t.notes:
        md += [""] + [f"* {n}" for n in t.notes]
    (d / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return [d / f"{stem}.csv", d / f"{stem}.md"]


def write_xlsx(tables: List[Table], path: Path) -> Optional[Path]:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter
    except ImportError:
        return None
    wb = Workbook()
    wb.remove(wb.active)
    for t in tables:
        ws = wb.create_sheet(f"Table {t.number}"[:31])
        ws.append([f"{t.label}. {t.title}"])
        ws["A1"].font = Font(bold=True)
        ws.append([])
        ws.append(t.columns)
        for c in ws[3]:
            c.font = Font(bold=True)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for r in t.rows:
            ws.append(list(r))
        ws.append([])
        for n in t.notes:
            ws.append([n])
        ws.append([f"Source: {', '.join(t.sources)}"])
        for i, col in enumerate(t.columns, start=1):
            width = max([len(str(col)) // 2] + [len(str(r[i - 1])) for r in t.rows if i - 1 < len(r)] + [8])
            ws.column_dimensions[get_column_letter(i)].width = min(width + 2, 48)
        ws.freeze_panes = "A4"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def write_docx(tables: List[Table], path: Path, heading: str) -> Optional[Path]:
    """Every table as a Word table with a caption above and notes below, landscape pages, 8 pt body text."""
    try:
        from docx import Document
        from docx.enum.section import WD_ORIENT
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Mm, Pt
    except ImportError:
        return None
    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Mm(297), Mm(210)
    for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, m, Mm(15))
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Arial", Pt(9)
    doc.add_heading(heading, level=1)

    def border_bottom(cell, sz=8):
        tcPr = cell._tc.get_or_add_tcPr()
        b = OxmlElement("w:tcBorders")
        e = OxmlElement("w:bottom")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), "000000")
        b.append(e); tcPr.append(b)

    for k, t in enumerate(tables):
        if k:
            doc.add_page_break()
        p = doc.add_paragraph()
        r = p.add_run(f"{t.label}. "); r.bold = True
        p.add_run(t.title)
        tab = doc.add_table(rows=1, cols=len(t.columns))
        tab.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, c in enumerate(t.columns):
            cell = tab.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(c)); run.bold = True; run.font.size = Pt(7.5)
            border_bottom(cell, 8)
        prev = [None] * t.group_cols
        for row in t.rows:
            cells = tab.add_row().cells
            for i, v in enumerate(row):
                txt = str(v)
                if i < t.group_cols:
                    shown = "" if prev[i] == txt and all(prev[j] == str(row[j]) for j in range(i)) else txt
                else:
                    shown = txt
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(shown); run.font.size = Pt(7.5)
            prev = [str(row[i]) for i in range(t.group_cols)]
        for c in tab.rows[-1].cells:
            border_bottom(c, 8)
        for n in t.notes:
            q = doc.add_paragraph(n); q.runs[0].font.size = Pt(7.5)
        q = doc.add_paragraph(f"Source: {', '.join(t.sources)}"); q.runs[0].font.size = Pt(7); q.runs[0].italic = True
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    return path


def save_figure(fig, out: Path, section: str, stem: str, formats=("pdf", "svg", "png"), dpi: int = 600) -> List[Path]:
    d = out / "figures" / section
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in formats:
        p = d / f"{stem}.{f}"
        kw = {"dpi": dpi} if f in ("png", "tif", "tiff") else {}
        if f in ("tif", "tiff"):
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(p, **kw)
        paths.append(p)
    return paths
