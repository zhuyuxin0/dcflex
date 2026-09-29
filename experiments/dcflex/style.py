"""Publication figure style, following the figures4papers house style
(github.com/ChenLiu-1996/figures4papers, scientific-figure-making skill): no top/right spines, frameless
legends, a semantic blue-green-red-neutral palette, black bar edges, tight layout and 300 dpi vector export.
One deliberate difference: figures4papers uses Helvetica/Arial, but this paper's body is Times (newtx), so the
figures use STIX, a Times-compatible face with matching math, and read as part of the same document.

Semantic colors in this paper:
  blue_main       system-optimal schedule (S2) and the model's key results
  blue_secondary  levers and variants of the key result (lighter shades by alpha)
  red_strong      private (tariff-driven) schedules and values; over-crediting
  neutral / grey  business as usual (S0) and background
  teal            the ERCOT market benchmark (S4)
  green_3         rule-based strategy (R)
  violet          learned (machine-learning) components: the physics-guided baseline B5, forecasts
Figures are drawn at the width they are printed (6.5 in = \\linewidth), so font sizes are the printed sizes.
"""
from dataclasses import dataclass
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
    "grey_mid": "#767676", "grey_dark": "#4D4D4D", "ink": "#272727",
    "blue_tint": "#E7EEF7",   # light fill for boxes in conceptual diagrams (tint of blue_main)
    "violet_tint": "#F3E9F1", # light fill for learned components in diagrams (tint of violet)
}
TEXT_WIDTH_IN = 6.5
# diverging map for signed changes: decrease = blue, increase = red
DIVERGING = LinearSegmentedColormap.from_list("f4p_div", [PALETTE["blue_main"], "#FFFFFF", PALETTE["red_strong"]])


@dataclass(frozen=True)
class FigureStyle:
    font_size: float = 9.5
    axes_linewidth: float = 1.1
    font_family: tuple = ("STIXGeneral", "TeX Gyre Termes", "Liberation Serif", "Times New Roman", "DejaVu Serif")


def apply_publication_style(style=None):
    s = style or FigureStyle()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": list(s.font_family), "font.size": s.font_size,
        "axes.labelsize": s.font_size, "axes.titlesize": s.font_size, "legend.fontsize": s.font_size - 1,
        "xtick.labelsize": s.font_size - 0.5, "ytick.labelsize": s.font_size - 0.5,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": s.axes_linewidth,
        "xtick.major.width": s.axes_linewidth, "ytick.major.width": s.axes_linewidth,
        "lines.linewidth": 1.8, "legend.frameon": False, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
    })


def finalize_figure(fig, out_path, pad=0.6, tight=True):
    """tight_layout, then save as vector PDF (300 dpi for any raster content)."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout(pad=pad)
    # no creation date in PDFs, so that an unchanged figure is byte-identical on a rerun
    meta = {"CreationDate": None} if out_path.suffix == ".pdf" else None
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03, metadata=meta)
    plt.close(fig)
    return out_path


apply_publication_style()
