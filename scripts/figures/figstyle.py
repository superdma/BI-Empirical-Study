"""Shared style and export helpers for the report figures.

All figures are drawn with matplotlib only, at a fixed 180 mm width so the
text sizes stay consistent across chapters. Every multi-panel figure runs the
skill's alignment gate before export, and the collision audit is run on the
exported PDF by `make_figures.sh`.
"""
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

SKILL_SCRIPTS = os.environ.get(
    "NATURE_FIGURE_SCRIPTS",
    os.path.expanduser("~/.config/CherryStudio/Data/Agents/.claude/skills/nature-figure/scripts"),
)
if SKILL_SCRIPTS not in sys.path:
    sys.path.insert(0, SKILL_SCRIPTS)

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

CJK = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
if os.path.exists(CJK):
    fm.fontManager.addfont(CJK)

MM = 1 / 25.4
WIDTH = 180 * MM
HALF = 87 * MM

# Restrained palette: one neutral family, engine identities, and a
# pass/fail pair that stays distinguishable in greyscale.
C = {
    "ink": "#1f2933",
    "muted": "#6b7280",
    "grid": "#e3e6ea",
    "vllm": "#4a5aa8",
    "vllm_light": "#9aa5cf",
    "sgl": "#2f8f83",
    "sgl_light": "#94c4bd",
    "pass": "#2f7f77",
    "fail": "#c0563f",
    "warn": "#d9a441",
    "zero": "#b8c0c8",
}


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["WenQuanYi Micro Hei"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": C["muted"],
        "axes.labelcolor": C["ink"],
        "text.color": C["ink"],
        "xtick.color": C["muted"],
        "ytick.color": C["muted"],
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def panel_label(ax, letter, dx=-0.085, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left", color=C["ink"])


def tidy(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def export(fig, name, outdir, multipanel=True, label_check=False, strict=True):
    """Run the alignment gate, then write SVG, PDF and a 600 dpi PNG.

    Alignment artefacts go to scripts/figures/qa so they never ship with the
    site.
    """
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, name)
    qa_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa")
    os.makedirs(qa_dir, exist_ok=True)
    qa = os.path.join(qa_dir, name)
    if multipanel:
        require_matplotlib_panel_alignment(
            fig,
            json_out=qa + ".alignment.json",
            overlay_svg=qa + ".alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=label_check,
            strict=strict,
        )
    fig.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    print("wrote", base + ".png")
