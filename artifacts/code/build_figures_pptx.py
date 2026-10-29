"""Edit the editable figure deck: corrected Fig. 1 labels, placeholders for every image object, labels and legends.

    python artifacts/code/build_figures_pptx.py

Input : artifacts/figures/pptx/Figures_original.pptx (as supplied, not modified)
Output: artifacts/figures/pptx/Figures_revised.pptx

Every picture that comes from the data is replaced by a grey placeholder whose text names the object file written by
make_figure_objects.py (artifacts/figures/objects/<figure>/<name>.png); in PowerPoint: right-click > Change Picture.
Labels, legends, colour-bar numbers and 20 µm scale bars are ordinary, editable shapes.

Slide 1 (Fig. 1), corrections to match the code and the manuscript:
  input standardisation is a per-image z-score (not RGB statistics); the loss sees the background-referenced phase (not raw);
  mask value 2 = spherocyte and 3 = echinocyte (legend was swapped); class colours as in Figs. 6 and 7; readout formulas use
  S = sum(phi_i - phi_bg) dx^2 and m = (lambda / 2 pi alpha) S (Eqs. 9-10); LoRA placement and trainable parts as in
  Section 2.4; "Ttrainable" typo. Equation objects that were edited are converted to plain text (Cambria Math characters),
  so every viewer shows the same, current text.
"""
from __future__ import annotations

import copy
import io
from pathlib import Path

from lxml import etree
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "figures" / "pptx" / "Figures_original.pptx"
OUT = ROOT / "artifacts" / "figures" / "pptx" / "Figures_revised.pptx"
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
      "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
A = "{%s}" % NS["a"]
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
CLASS_HEX = {0: "404040", 1: "0072B2", 2: "009E73", 3: "D55E00", 4: "CC79A7"}
CLASS_NAME = {1: "Discocyte (1)", 2: "Spherocyte (2)", 3: "Echinocyte (3)", 4: "Stomatocyte (4)"}
PX_UM = 0.1441


# ------------------------------------------------------------------------------------------------ placeholders
def placeholder_png(label: str, w=600, h=600) -> io.BytesIO:
    im = Image.new("RGB", (w, h), (200, 200, 198))
    d = ImageDraw.Draw(im)
    d.rectangle((2, 2, w - 3, h - 3), outline=(120, 120, 118), width=4)
    size = max(12, min(w, h) // 14)
    font = ImageFont.truetype(FONT, size)
    lines = label.split("\n") if w >= h else [label.split("/")[-1]]
    y = h / 2 - len(lines) * size * 0.7
    for ln in lines:
        tw = d.textlength(ln, font=font)
        if tw > w - 10 and w < h:
            continue
        d.text(((w - tw) / 2, y), ln, font=font, fill=(60, 60, 60))
        y += size * 1.4
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf


def swap_picture(slide, pic, label: str, w_in=None, h_in=None):
    """Point an existing picture shape at a placeholder image (shape, position and z-order kept)."""
    w_in = w_in or pic.width / 914400
    h_in = h_in or pic.height / 914400
    px = 600
    img = placeholder_png(label, int(px * w_in / max(w_in, h_in)) or 40, int(px * h_in / max(w_in, h_in)) or 40)
    _, rid = slide.part.get_or_add_image_part(img)
    blip = pic._element.find(".//" + A + "blip")
    blip.set("{%s}embed" % NS["r"], rid)
    for ext in blip.findall(A + "extLst"):
        blip.remove(ext)
    src = pic._element.find(".//" + A + "srcRect")
    if src is not None:
        src.getparent().remove(src)
    pic.width, pic.height = Inches(w_in), Inches(h_in)
    pic.name = label.split("\n")[0].split("/")[-1]
    return pic


def add_placeholder(slide, label, x, y, w, h):
    img = placeholder_png(label, int(600 * w / max(w, h)), int(600 * h / max(w, h)))
    pic = slide.shapes.add_picture(img, Inches(x), Inches(y), Inches(w), Inches(h))
    pic.name = label.split("/")[-1]
    return pic


# ------------------------------------------------------------------------------------------------ text helpers
def shapes_by_name(shapes, out=None):
    out = {} if out is None else out
    for s in shapes:
        out.setdefault(s.name, []).append(s)
        if s.shape_type == 6:
            shapes_by_name(s.shapes, out)
    return out


def set_paragraphs(shape, lines):
    """Rewrite the text of a shape line by line, keeping the formatting of the first run of each paragraph."""
    tf = shape.text_frame
    paras = list(tf.paragraphs)
    for i, line in enumerate(lines):
        p = paras[min(i, len(paras) - 1)] if i < len(paras) else None
        if p is None:
            new = copy.deepcopy(paras[-1]._p)
            paras[-1]._p.addnext(new)
            paras = list(tf.paragraphs)
            p = paras[i]
        runs = p.runs
        if not runs:
            continue
        runs[0].text = line
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
    for p in paras[len(lines):]:
        p._p.getparent().remove(p._p)


def replace_words(shape, mapping):
    """Swap whole run texts (e.g. legend entries separated by line breaks) without touching the layout."""
    for r in shape._element.iter(A + "t"):
        if r.text in mapping:
            r.text = mapping[r.text]


def text_box(slide, text, x, y, w, h, size=11, bold=False, color="000000", align=PP_ALIGN.CENTER, rotate=None,
             anchor=MSO_ANCHOR.MIDDLE):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, "Arial"
        r.font.color.rgb = RGBColor.from_string(color)
    if rotate is not None:
        tb.rotation = rotate
    return tb


def rect(slide, x, y, w, h, hex_fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = RGBColor.from_string(hex_fill)
    if line:
        s.line.color.rgb = RGBColor.from_string(line)
        s.line.width = Pt(0.5)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def scale_bar(slide, x, y, panel_w, field_um, label="20 µm"):
    """White 20 µm bar with label, bottom-left inside a panel whose left/bottom edge is (x, y)."""
    bar_w = panel_w * 20.0 / field_um
    rect(slide, x + 0.08, y - 0.14, bar_w, 0.045, "FFFFFF")
    text_box(slide, label, x + 0.06, y - 0.36, 0.7, 0.2, size=10, bold=True, color="FFFFFF", align=PP_ALIGN.LEFT)


# ------------------------------------------------------------------------------------------------ equations
def flatten_math(sp_choice):
    """Turn a14:m equation runs inside a shape into ordinary text runs (same characters and run properties)."""
    for m in sp_choice.findall(".//a14:m", NS):
        runs = []
        for mr in m.findall(".//m:r", NS):
            r = etree.SubElement(etree.Element(A + "dummy"), A + "r")
            rpr = mr.find("a:rPr", NS)
            r.append(copy.deepcopy(rpr) if rpr is not None else etree.Element(A + "rPr"))
            t = etree.SubElement(r, A + "t")
            t.text = "".join(x.text or "" for x in mr.findall("m:t", NS))
            runs.append(r)
        for r in runs:
            m.addprevious(r)
        m.getparent().remove(m)
    return sp_choice


def replace_equation_box(slide, name, runs_spec):
    """Replace an mc:AlternateContent equation box by a plain text box with the given runs.
    runs_spec: list of (text, baseline) with baseline None, 'sub' or 'sup'; formatting copied from the first run."""
    tree = slide.shapes._spTree
    for ac in tree.iter("{%s}AlternateContent" % NS["mc"]):
        nv = ac.find(".//p:cNvPr", NS)
        if nv is None or nv.get("name") != name:
            continue
        sp = copy.deepcopy(ac.find("mc:Choice/p:sp", NS))
        flatten_math(sp)
        p = sp.find(".//a:p", NS)
        old_runs = p.findall("a:r", NS)
        tmpl = copy.deepcopy(old_runs[-1].find("a:rPr", NS))
        for r in old_runs:
            p.remove(r)
        end = p.find("a:endParaRPr", NS)
        for text, base in runs_spec:
            r = etree.Element(A + "r")
            rpr = copy.deepcopy(tmpl)
            rpr.attrib.pop("baseline", None)
            rpr.attrib.pop("spc", None)
            if base == "sub":
                rpr.set("baseline", "-25000")
            elif base == "sup":
                rpr.set("baseline", "30000")
            r.append(rpr)
            t = etree.SubElement(r, A + "t")
            t.text = text
            if end is not None:
                end.addprevious(r)
            else:
                p.append(r)
        ac.addprevious(sp)
        ac.getparent().remove(ac)
        return sp
    raise KeyError(name)


def plain_equation_box(slide, name, text):
    return replace_equation_box(slide, name, [(text, None)])


# ------------------------------------------------------------------------------------------------ slides
def slide1(prs, s):
    by = shapes_by_name(s.shapes)
    set_paragraphs(by["TextBox 30"][0], ["LoRA adapters in encoder attention,", "linear and convolution layers"])
    set_paragraphs(by["TextBox 268"][0], ["Gradients update LoRA adapters, stem, normalisation and decoder"])
    set_paragraphs(by["TextBox 48"][0], ["Trainable"])
    set_paragraphs(by["TextBox 173"][0], ["Integrated phase", "and dry mass"])
    set_paragraphs(by["TextBox 121"][0], ["(z-score per image:", "zero mean, unit variance)"])
    set_paragraphs(by["TextBox 270"][0], ["Trainable (EdgeSAM, r = 32: 21.2%)"])
    replace_words(by["TextBox 62"][0], {"Echinocyte": "Spherocyte"})
    replace_words(by["TextBox 283"][0], {"Spherocyte": "Echinocyte"})
    # class colours: index boxes 0-4 and the small legend squares
    for nm, k in (("Rectangle 52", 0), ("Rectangle 54", 1), ("Rectangle 56", 2), ("Rectangle 58", 3), ("Rectangle 60", 4)):
        for shp in by.get(nm, []):
            shp.fill.solid()
            shp.fill.fore_color.rgb = RGBColor.from_string(CLASS_HEX[k])
    plain_equation_box(s, "TextBox 103", "Background-referenced phase 𝝓 (radians) enters the loss directly")
    plain_equation_box(s, "TextBox 120", "Standardised input 𝚽")
    plain_equation_box(s, "TextBox 43", "W′ = W + (𝜶 / 𝒓) BA")
    replace_equation_box(s, "TextBox 236", [("𝑺 = 𝚺(𝝓", None), ("i", "sub"), (" − 𝝓", None), ("bg", "sub"),
                                            (") · 𝚫𝒙", None), ("2", "sup")])
    plain_equation_box(s, "TextBox 239", "𝒎 = (𝝀 / 𝟐𝝅𝜶) · 𝑺")
    # data pictures -> placeholders (square image + separate colour bar where the old picture had one built in)
    for nm, obj, cbar in (("Picture 18", "Fig1/p1_phase.png", "Fig1/p1_phase_colorbar.png"),
                          ("Picture 23", "Fig1/p2_norm.png", "Fig1/p2_norm_colorbar.png"),
                          ("Picture 83", "Fig1/p4_volume.png", "Fig1/p4_volume_colorbar.png"),
                          ("Picture 51", "Fig1/p3_mask.png", None)):
        pic = by[nm][0]
        h = pic.height / 914400
        x, y = pic.left / 914400, pic.top / 914400
        swap_picture(s, pic, "PLACEHOLDER\n" + obj, h, h)
        if cbar:
            cb = add_placeholder(s, cbar, x + h + 0.04, y, 0.06, h)
            pic._element.addnext(cb._element)


def slide2(prs, s):
    by = shapes_by_name(s.shapes)
    for nm, obj, cbar in (("Picture 50", "Fig1/p1_phase.png", "Fig1/p1_phase_colorbar.png"),
                          ("Picture 70", "Fig1/p2_norm.png", "Fig1/p2_norm_colorbar.png"),
                          ("Picture 68", "Fig1/p4_volume.png", "Fig1/p4_volume_colorbar.png"),
                          ("Picture 45", "Fig1/p3_mask.png", None)):
        pic = by[nm][0]
        h = pic.height / 914400
        x, y = pic.left / 914400, pic.top / 914400
        swap_picture(s, pic, "PLACEHOLDER\n" + obj, h, h)
        if cbar:
            add_placeholder(s, cbar, x + h + 0.04, y, 0.06, h)
    text_box(s, "Objects written by artifacts/code/make_figure_objects.py to artifacts/figures/objects/Fig1/. "
                "Colour-bar limits are listed in objects.json (Fig1); update the numbers next to the colour bars on slide 1.",
             0.3, 6.9, 12.7, 0.35, size=10, color="595959", align=PP_ALIGN.LEFT)


def slide3(prs, s):
    by = shapes_by_name(s.shapes)
    set_paragraphs(by["TextBox 1"][0], ["Figure S-4"])
    set_paragraphs(by["TextBox 5"][0], ["Phase (rad)"])
    set_paragraphs(by["TextBox 9"][0], ["Phase-gradient magnitude"])
    set_paragraphs(by["TextBox 13"][0], ["EdgeSAM (r=32) contours"])
    for nm, obj in (("Picture 2", "FigS4/p1_phase.png"), ("Picture 6", "FigS4/p2_gradient.png"),
                    ("Picture 10", "FigS4/p3_contours.png")):
        swap_picture(s, by[nm][0], "PLACEHOLDER\n" + obj)


def slide4(prs, s):
    """Figure 6 rebuilt: 4 rows (classes) x 6 columns, placeholders, headers, row labels, scale bars and legend."""
    keep = {"TextBox 1"}
    for shp in list(s.shapes):
        if shp.name not in keep:
            shp._element.getparent().remove(shp._element)
    cols = [("c1_phase", "Phase"), ("c2_annotation", "Annotation"), ("c3_mobilenet_unet_r8", "MobileNet-UNet (r=8)"),
            ("c4_mobilesam_r8", "MobileSAM (r=8)"), ("c5_edgesam_r8", "EdgeSAM (r=8)"), ("c6_edgesam_r32", "EdgeSAM (r=32)")]
    rows = [(1, "discocyte", 15), (2, "spherocyte", 47), (3, "echinocyte", 30), (4, "stomatocyte", 19)]
    size, gap, x0, y0 = 1.45, 0.08, 1.35, 0.85
    field_um = 256 * PX_UM
    for c, (_, head) in enumerate(cols):
        text_box(s, head, x0 + c * (size + gap), y0 - 0.32, size, 0.28, size=11)
    for r, (k, cl, day) in enumerate(rows):
        y = y0 + r * (size + gap)
        text_box(s, f"{CLASS_NAME[k]}\nday {day}", x0 - 1.0, y, 0.9, size, size=11, align=PP_ALIGN.CENTER)
        for c, (tag, _) in enumerate(cols):
            x = x0 + c * (size + gap)
            add_placeholder(s, f"Fig6/r{k}_{cl}_{tag}.png", x, y, size, size)
            scale_bar(s, x, y + size, size, field_um)
    ly = y0 + 4 * (size + gap) + 0.08
    items = [(CLASS_HEX[k], CLASS_NAME[k]) for k in (1, 2, 3, 4)] + [("BDBDBD", "outside annotated field")]
    lx = x0
    for hexc, lab in items:
        rect(s, lx, ly + 0.05, 0.28, 0.16, hexc)
        text_box(s, lab, lx + 0.34, ly, 1.75, 0.26, size=11, align=PP_ALIGN.LEFT)
        lx += 0.34 + (1.25 if "outside" not in lab else 2.0)
    text_box(s, "Each panel: 256 × 256 px window (36.89 µm), object 1024 × 1024 px (4× upsampled).\n"
                "Scale bar 20 µm = 54.2% of the panel width. Row days follow objects.json (Fig6);\n"
                "all predictions from training seed 42.", x0 + 6 * (size + gap) + 0.1, y0, 2.6, 1.2, size=9,
             color="808080", align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)


def slide5(prs, s):
    by = shapes_by_name(s.shapes)
    set_paragraphs(by["TextBox 1"][0], ["Figure S-5"])
    title = by["TextBox 2"][0]
    title._element.getparent().remove(title._element)
    days = [0, 8, 19, 27, 47]
    for i, (top, bottom) in enumerate(zip(["Rectangle 3", "Rectangle 7", "Rectangle 11", "Rectangle 15", "Rectangle 19"],
                                          ["Rectangle 23", "Rectangle 26", "Rectangle 29", "Rectangle 32", "Rectangle 35"])):
        for nm, kind in ((top, "phase"), (bottom, "contours")):
            r = by[nm][0]
            pic = add_placeholder(s, f"FigS5/day{days[i]:02d}_{kind}.png", r.left / 914400, r.top / 914400,
                                  r.width / 914400, r.height / 914400)
            r._element.addprevious(pic._element)
            r._element.getparent().remove(r._element)
    text_box(s, "Phase", 0.35, 2.01, 1.0, 2.0, size=11)
    text_box(s, "EdgeSAM (r=32)\ncontours", 0.35, 4.05, 1.0, 2.0, size=11)
    set_paragraphs(by["TextBox 38"][0], ["Panel objects: artifacts/figures/objects/FigS5/ (make_figure_objects.py).",
                                         "Panel image 1536 × 1536 px, upsampled 2× from the 768 × 768 px source array.",
                                         "Pixel pitch 0.1441 μm/px, so one panel spans 768 × 0.1441 = 110.67 μm.",
                                         "Scale bar 20 μm = 18.07% of the panel width."])


def main():
    prs = Presentation(SRC)
    sl = list(prs.slides)
    slide1(prs, sl[0])
    slide2(prs, sl[1])
    slide3(prs, sl[2])
    slide4(prs, sl[3])
    slide5(prs, sl[4])
    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
