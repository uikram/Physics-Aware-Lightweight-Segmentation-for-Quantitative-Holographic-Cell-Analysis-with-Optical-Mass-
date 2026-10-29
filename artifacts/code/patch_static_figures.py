"""Text corrections in the figures that cannot be regenerated from outputs/ (schematics and image panels).

Sources are the images embedded in the final-review manuscript and SI (artifacts/manuscript_in_prog/Claudes_Update/).
Only labels are changed; image content is untouched. Output: artifacts/figures/manuscript/.

Fig. 1  (framework): class legend (mask 2 = spherocyte, 3 = echinocyte, D-25), network input (per-image z-score, not RGB
         statistics), loss input (background-referenced phase, not raw), readout formulas (S with phi_bg, Eq. 9-10),
         LoRA placement and trainable parts (Section 2.4).
Fig. 6  (qualitative): column headers in the "(r=8)" notation.
Fig. S-4 (gradient alignment): no change in content; the clipped right-hand title is shortened.
Fig. S-5 (storage images): the decorative title row is removed.
Fig. S-6 (component counts): "um2" written as "µm²".
"""
from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "manuscript_in_prog" / "Claudes_Update"
MS = SRC / "revised_manuscript_fv - ui - final review.docx"
SI = SRC / "supplementry_info_fv - ui - final review.docx"
OUT = ROOT / "artifacts" / "figures" / "manuscript"
FONTS = Path("/usr/share/fonts/truetype")
SANS = FONTS / "liberation" / "LiberationSans-Regular.ttf"
SANS_B = FONTS / "liberation" / "LiberationSans-Bold.ttf"
SANS_BI = FONTS / "liberation" / "LiberationSans-BoldItalic.ttf"
SERIF_BI = FONTS / "liberation" / "LiberationSerif-BoldItalic.ttf"
SERIF_B = FONTS / "liberation" / "LiberationSerif-Bold.ttf"
DEJAVU = FONTS / "dejavu" / "DejaVuSans.ttf"


def media(docx: Path, name: str) -> Image.Image:
    with zipfile.ZipFile(docx) as z:
        return Image.open(BytesIO(z.read(f"word/media/{name}"))).convert("RGB")


def ink_bbox(im, box, tol=40):
    """Bounding box of the pixels in box that differ from the box's corner colour."""
    a = np.asarray(im.crop(box)).astype(int)
    bg = a[2, 2]
    m = np.abs(a - bg).sum(axis=2) > tol
    ys, xs = np.nonzero(m)
    return (box[0] + xs.min(), box[1] + ys.min(), box[0] + xs.max() + 1, box[1] + ys.max() + 1), tuple(bg)


def draw_runs(d, x, y_base, runs, colour):
    """runs: (text, font, dy) drawn left to right from x on baseline y_base; returns the end x."""
    for text, font, dy in runs:
        d.text((x, y_base + dy), text, font=font, fill=colour, anchor="ls")
        x += d.textlength(text, font=font)
    return x


def replace(im, box, lines, font_path, size, colour, align="center", pad=3, bg=None, line_gap=1.18):
    """Erase the ink inside box and write lines (centred on the erased block, or left-aligned at its left edge)."""
    (x0, y0, x1, y1), bg0 = ink_bbox(im, box)
    bg = bg or bg0
    d = ImageDraw.Draw(im)
    d.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), fill=bg)
    font = ImageFont.truetype(str(font_path), size)
    asc, desc = font.getmetrics()
    h = len(lines) * size * line_gap
    top = (y0 + y1) / 2 - h / 2
    for i, ln in enumerate(lines):
        w = d.textlength(ln, font=font)
        x = (x0 + x1) / 2 - w / 2 if align == "center" else x0
        d.text((x, top + i * size * line_gap + size * 0.9), ln, font=font, fill=colour, anchor="ls")
    return (x0, y0, x1, y1)


def put(im, erase, bg_at, lines, font_path, size, colour, cx=None, max_w=None, line_gap=1.18):
    """Fill the erase rectangle with the colour found at bg_at and write centred lines (shrunk to max_w if needed)."""
    d = ImageDraw.Draw(im)
    bg = im.getpixel(bg_at)
    d.rectangle(erase, fill=bg)
    cx = cx if cx is not None else (erase[0] + erase[2]) / 2
    while max_w and max(d.textlength(l, font=ImageFont.truetype(str(font_path), size)) for l in lines) > max_w:
        size -= 1
    font = ImageFont.truetype(str(font_path), size)
    h = len(lines) * size * line_gap
    top = (erase[1] + erase[3]) / 2 - h / 2
    for i, ln in enumerate(lines):
        w = d.textlength(ln, font=font)
        d.text((cx - w / 2, top + i * size * line_gap + size * 0.92), ln, font=font, fill=colour, anchor="ls")


def fig1():
    im = media(MS, "image1.png")
    grey, navy, teal = (58, 58, 64), (12, 40, 92), (0, 120, 150)
    put(im, (548, 13, 1638, 50), (1090, 14), ["Background-referenced phase φ (radians) bypasses standardisation and enters the loss"],
        SANS_BI, 27, grey, cx=1090, max_w=1070)
    put(im, (62, 829, 345, 859), (52, 844), ["Standardised input Φ"], SANS_B, 26, (255, 255, 255), cx=202)
    replace(im, (60, 872, 340, 940), ["(z-score per image:", "zero mean, unit variance)"], SANS, 24, (40, 40, 40))
    replace(im, (1135, 1185, 1240, 1213), ["Spherocyte"], SANS, 20, (60, 60, 60), align="left")
    replace(im, (1298, 1131, 1400, 1158), ["Echinocyte"], SANS, 20, (60, 60, 60), align="left")
    put(im, (2085, 220, 2440, 288), (2078, 254), ["Integrated phase", "and dry mass"], SANS_B, 28, (255, 255, 255), cx=2260)
    replace(im, (480, 640, 940, 712), ["LoRA adapters in encoder attention,", "linear and convolution layers"], SANS_B, 26, teal)
    put(im, (782, 1297, 1688, 1331), (800, 1300), ["Gradients update LoRA adapters, input stem, normalisation layers and decoder"],
        SANS_BI, 27, grey, cx=1234, max_w=880)
    # readout formulas: S = sum(phi_i - phi_bg) dx^2 and m = (lambda / 2 pi alpha) S
    d = ImageDraw.Draw(im)
    big, small = ImageFont.truetype(str(SERIF_BI), 34), ImageFont.truetype(str(SERIF_BI), 22)
    for box, runs in [
        ((2120, 950, 2420, 1000), [("S = Σ(φ", big, 0), ("i", small, 6), (" − φ", big, 0), ("bg", small, 6),
                                   (")·Δx", big, 0), ("2", small, -16)]),
        ((2120, 1110, 2420, 1162), [("m = (λ/2πα)·S", big, 0)])]:
        (x0, y0, x1, y1), bg = ink_bbox(im, box)
        d.rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), fill=bg)
        draw_runs(d, x0, y1 - 6, runs, navy)
    im.save(OUT / "Fig1_framework.png", dpi=(300, 300))
    print("wrote Fig1_framework")


def fig6():
    im = media(MS, "image8.png")
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(str(DEJAVU), 25)
    heads = ["MobileNet-UNet (r=8)", "MobileSAM (r=8)", "EdgeSAM (r=8)", "EdgeSAM (r=32)"]
    seeds = [42, 43, 43, 43]
    centres = [838, 1141, 1444, 1747]
    d.rectangle((680, 20, im.width - 1, 100), fill=(255, 255, 255))
    for c, h, s in zip(centres, heads, seeds):
        for i, t in enumerate([h, f"seed {s}"]):
            w = d.textlength(t, font=font)
            d.text((c - w / 2, 56 + i * 32), t, font=font, fill=(0, 0, 0), anchor="ls")
    im.save(OUT / "Fig6_qualitative.png", dpi=(300, 300))
    print("wrote Fig6_qualitative")


def si_figs():
    im = media(SI, "image5.png")
    for box, cx, txt in [((40, 10, 820, 68), 412, "Phase map (rad)"), ((880, 10, 1660, 68), 1240, "Phase-gradient magnitude"),
                         ((1660, 10, 2499, 68), 2080, "EdgeSAM (r=8) contours")]:
        put(im, (box[0], 12, box[2], 66), (box[0] + 2, 40), [txt], SANS_B, 38, (0, 0, 0), cx=cx)
    im.save(OUT / "FigS4_gradient_alignment.png", dpi=(300, 300))
    im = media(SI, "image7.png")
    im.crop((0, 80, im.width, im.height)).save(OUT / "FigS5_storage_images.png", dpi=(300, 300))
    im = media(SI, "image9.png")
    im.save(OUT / "FigS6_component_counts.png", dpi=(300, 300))
    print("wrote FigS4, FigS5, FigS6")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    # Fig. 6 and Fig. S-4 to S-6 are data figures: they are regenerated by make_data_figures.py on the machine with
    # data/processed (placeholders in the Word files until then); fig6() and si_figs() only patch the old images' labels.
