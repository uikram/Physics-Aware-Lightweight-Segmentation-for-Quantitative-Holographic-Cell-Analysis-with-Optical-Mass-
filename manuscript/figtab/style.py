"""One visual system for every figure: journal column widths, fonts, line weights and a validated palette.

Widths follow the Springer (Applied Physics B) artwork guide: 84 mm for one column, 174 mm for the full page width,
at most 234 mm tall. All text is Times New Roman, regular weight (8 pt; panel labels "(a)" 9 pt), above the 6 pt
minimum after placement at 100 %. Fonts are embedded as TrueType in the PDF; in the SVG, text is converted to paths so
the file looks the same on every computer (the fonts need not be installed). Variables such as the LoRA rank r are
set in italic through mathtext ($r$).

Colours. Categorical hues were checked with a colour-vision-deficiency validator (OKLab ΔE, all pairs):
- architectures (3 hues): all pairs pass for normal vision and the three CVD types;
- morphology classes (4 hues, deliberately different from the architecture hues so a colour never means two
  things in one manuscript): all pairs pass.
Text is never coloured by series; identity is also carried by labels or marker shape.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4
SINGLE = 84 * MM
ONEHALF = 129 * MM
DOUBLE = 174 * MM
MAX_H = 234 * MM

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#ffffff"

ARCH_COLOR = {"EdgeSAM": "#2a78d6", "MobileSAM": "#eb6834", "MobileNet-UNet": "#1baf7a"}
ARCH_MARKER = {"EdgeSAM": "o", "MobileSAM": "s", "MobileNet-UNet": "D"}
CLASS_COLOR = {"discocyte": "#6250d6", "spherocyte": "#eda100", "echinocyte": "#e87ba4", "stomatocyte": "#008300"}
REFERENCE_COLOR = INK          # the annotation / reference series
HIGHLIGHT = INK                # proposed objective, selected rank


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        # STIXGeneral ships with matplotlib itself, so a computer without Times New Roman (e.g. a Linux server) still
        # gets a Times-style regular font instead of the heavy-looking DejaVu Serif
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "STIXGeneral",
                       "DejaVu Serif"],
        "font.weight": "normal", "axes.titleweight": "normal", "figure.titleweight": "normal",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.title_fontsize": 8,
        "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.2,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        # mathtext in Times style: $r$ is italic, \mathrm{...} upright, \mathcal{L} the script L of the manuscript
        "mathtext.fontset": "stix", "mathtext.default": "it",
    })


def panel_letter(ax, letter: str, x: float = -0.02, y: float = 1.02, **kw):
    """Kept for compatibility: a bracketed, regular-weight letter at a given position."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, va="bottom", ha="right", color=INK, **kw)


def panel_key_title(ax, letter: str, name: str, color: str, pad: float = 22.0, marker: str = "s"):
    """"(a)" centred above the plot and, under it, a small box in the model's colour followed by the model name.
    Used where the colour then identifies the model in the panels below, so the name is written only once."""
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
    from matplotlib.patches import Rectangle
    ax.set_title(f"({letter})", loc="center", fontsize=9, fontweight="normal", color=INK, pad=pad)
    box = DrawingArea(7, 7, 0, 0)
    box.add_artist(Rectangle((0, 0), 7, 7, fc=color, ec="none"))
    txt = TextArea(name, textprops=dict(fontsize=9, color=INK))
    pack = HPacker(children=[box, txt], align="center", pad=0, sep=4)
    ab = AnchoredOffsetbox(loc="lower center", child=pack, pad=0, frameon=False, bbox_to_anchor=(0.5, 1.0),
                           bbox_transform=ax.transAxes, borderpad=0.5)
    ax.add_artist(ab)
    return ab


def panel_title(ax, letter: str, name: str = "", pad: float = 4.0):
    """Panel label centred above the plot: "(a)" on the first line and, if given, the panel's name (for example
    the architecture) centred underneath it, both in regular Times New Roman."""
    ax.set_title(f"({letter})" + (f"\n{name}" if name else ""), loc="center", fontsize=9, fontweight="normal",
                 color=INK, pad=pad, linespacing=1.35)


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def vgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
