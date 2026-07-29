"""Shared publication style for the PathwayPro main-text figure set.

All six main figures should `from _style import *` so that fonts, colours,
sizing and the storyline-level annotations are visually consistent.  The
palette is colour-blind safe (Wong, Nature 2011, with project-specific
pathway colours).

Story chain (Fig.1 → Fig.6):
    1. Cohort:        ``We have a high-quality paired T1+FLAIR ADNI cohort.``
    2. Biomarkers:    ``Both AD biomarkers and vascular WMH differ between
                       diagnoses — disentangling them is necessary.``
    3. Framework:     ``PathwayPro splits the shared MRI representation into
                       atrophy and vascular pathways with explicit losses.``
    4. Phase-0 head-to-head: ``PathwayPro is the only model that wins on the
                       vascular probe (z_v→WMH) while remaining competitive
                       on classification.``
    5. Explainability: ``PathwayPro attribution concentrates on AD-relevant
                       regions, drops AUC fastest under occlusion, and is the
                       most reproducible across seeds.``
    6. Summary:        ``Capability matrix + 3 headline wins + translational
                       pipeline.``

Use the `STORY` dict below to fetch the one-sentence narrative anchor for the
current figure, and call `style_setup()` once at the top of `main()`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# -----------------------------------------------------------------------------
# 1. Colour palette
# -----------------------------------------------------------------------------
# Method colours – PathwayPro is always the warm Wong "vermillion"; baselines
# get distinct hues so that they are recognisable across all figures.
METHOD_COLORS: Dict[str, str] = {
    "vanilla":   "#999999",
    "betavae":   "#56B4E9",
    "tcvae":     "#009E73",
    "factorvae": "#E69F00",
    "dipvae2":   "#CC79A7",
    "ours":      "#D55E00",
}
METHOD_LABELS: Dict[str, str] = {
    "vanilla":   "Vanilla MTL",
    "betavae":   r"$\beta$-VAE",
    "tcvae":     r"$\beta$-TCVAE",
    "factorvae": "FactorVAE",
    "dipvae2":   "DIP-VAE-II",
    "ours":      "PathwayPro (ours)",
}
METHOD_ORDER = ["vanilla", "betavae", "tcvae", "factorvae", "dipvae2", "ours"]

DIAG_COLORS: Dict[str, str] = {
    "CN":  "#0072B2",
    "MCI": "#E69F00",
    "AD":  "#D55E00",
}

# Pathway colours used in the two-branch overlays and the framework diagram.
PATHWAY: Dict[str, str] = {
    "atrophy":  "#7B4DBA",
    "vascular": "#2CA02C",
    "shared":   "#F5C16C",
}

# Soft accent colours for callouts.
ACCENT = {
    "win":   "#D55E00",
    "neutral": "#444444",
    "muted": "#888888",
    "hi":    "#FFE9D9",  # PathwayPro highlight wash
}

# -----------------------------------------------------------------------------
# 2. Story dictionary — one sentence per figure for inline annotation
# -----------------------------------------------------------------------------
STORY = {
    1: ("PathwayPro is trained on a high-quality, paired T1+FLAIR ADNI cohort "
        "with rich biomarker coverage."),
    2: ("Diagnostic groups differ on both amyloid and vascular markers — "
        "motivating an explicit two-pathway model."),
    3: ("A shared 3-D encoder routes its representation into two pathology-aware "
        "latents ($z_a$ atrophy, $z_v$ vascular) with disentanglement losses."),
    4: ("Head-to-head benchmark: PathwayPro leads on the vascular probe "
        "$z_v\\!\\to\\!$WMH while remaining competitive on classification."),
    5: ("PathwayPro attribution is sharper, more AD-relevant, and more "
        "reproducible than every disentanglement baseline."),
    6: ("Among all tested methods, PathwayPro is the only model that delivers "
        "every required capability for clinical screening and care."),
}

# -----------------------------------------------------------------------------
# 3. Style setup
# -----------------------------------------------------------------------------
MM2IN = 1.0 / 25.4
def mm(v: float) -> float:  # noqa: E704
    return v * MM2IN


_SERIF_CANDIDATES = (
    Path(__file__).resolve().parent.parent / "fonts" / "Times New Roman.ttf",
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"),
    Path.home() / ".fonts" / "Times New Roman.ttf",
)


def style_setup(scale: float = 1.0) -> str:
    """Activate the shared rcParams. Returns the resolved font family."""
    name: Optional[str] = None
    for fp in _SERIF_CANDIDATES:
        if fp.is_file():
            try:
                font_manager.fontManager.addfont(str(fp))
                name = font_manager.FontProperties(fname=str(fp)).get_name()
                break
            except Exception:
                continue
    if name is None:
        available = {f.name for f in font_manager.fontManager.ttflist}
        for n in ("Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"):
            if n in available:
                name = n
                break
        else:
            name = "DejaVu Serif"

    plt.rcParams.update({
        "font.family": name,
        "mathtext.fontset": "stix",
        "font.size":     8.0 * scale,
        "axes.titlesize": 9.5 * scale,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5 * scale,
        "xtick.labelsize": 7.0 * scale,
        "ytick.labelsize": 7.0 * scale,
        "legend.fontsize": 7.5 * scale,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.05,
        "lines.markersize": 3.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "savefig.dpi": 320,
    })
    return name


# -----------------------------------------------------------------------------
# 4. Helpers: panel labels, takeaway boxes, win badges
# -----------------------------------------------------------------------------

def panel_label(ax, text: str, dx: float = -0.10, dy: float = 1.04, fontsize: float = 11.0):
    ax.text(dx, dy, text, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", ha="left", va="bottom")


def figure_title(fig, n: int, title: str, *, y_title: float = 0.965,
                  y_story: Optional[float] = None) -> None:
    """Set the figure header in the Nature-style format used across the chapter.

    The figure number is drawn in the top-left, the title is centred, and the
    one-sentence storyline anchor (from :data:`STORY`) is shown one line below
    in italic muted grey when ``y_story`` is given.
    """
    fig.text(0.012, 0.992, f"Figure {n}", fontsize=12.5,
             fontweight="bold", va="top")
    fig.suptitle(title, fontsize=10.5, y=y_title, x=0.515)
    if y_story is not None and n in STORY:
        fig.text(0.515, y_story, STORY[n], ha="center", va="top",
                  fontsize=8.3, style="italic", color=ACCENT["neutral"])


def takeaway_banner(fig, text: str, *, y: float = 0.018) -> None:
    """Print a one-line ``Take-away:`` banner at the bottom of the figure."""
    fig.text(0.5, y, "Take-away: " + text,
             ha="center", va="bottom",
             fontsize=8.0, style="italic",
             color=ACCENT["win"], fontweight="bold")


def win_badge(ax, text: str = "PathwayPro wins", *,
              loc: str = "upper right",
              fontsize: float = 7.4):
    """Draw a small rounded badge stating that PathwayPro wins this panel."""
    bbox = dict(boxstyle="round,pad=0.3,rounding_size=0.5",
                facecolor=ACCENT["hi"], edgecolor=METHOD_COLORS["ours"],
                linewidth=0.8)
    if loc == "upper right":
        x, y, ha, va = 0.985, 0.95, "right", "top"
    elif loc == "upper left":
        x, y, ha, va = 0.015, 0.95, "left", "top"
    elif loc == "lower right":
        x, y, ha, va = 0.985, 0.06, "right", "bottom"
    else:
        x, y, ha, va = 0.015, 0.06, "left", "bottom"
    ax.text(x, y, text, transform=ax.transAxes,
            ha=ha, va=va, fontsize=fontsize, fontweight="bold",
            color=METHOD_COLORS["ours"], bbox=bbox)


def highlight_xticklabel(ax, label_substr: str = "PathwayPro") -> None:
    """Bold-face & recolour the xtick label matching ``label_substr``."""
    for lab in ax.get_xticklabels():
        if label_substr.lower() in lab.get_text().lower():
            lab.set_fontweight("bold")
            lab.set_color(METHOD_COLORS["ours"])


def highlight_yticklabel(ax, label_substr: str = "PathwayPro") -> None:
    for lab in ax.get_yticklabels():
        if label_substr.lower() in lab.get_text().lower():
            lab.set_fontweight("bold")
            lab.set_color(METHOD_COLORS["ours"])


def annotate_pathwaypro_bar(ax, x: float, height: float, *, dy: float = 0.005,
                             text: Optional[str] = None) -> None:
    """Draw a small downward arrow with a label above the PathwayPro bar."""
    text = text or "best"
    ax.annotate(
        text, xy=(x, height), xytext=(x, height + dy * 4),
        ha="center", va="bottom", fontsize=7.0, fontweight="bold",
        color=METHOD_COLORS["ours"],
        arrowprops=dict(arrowstyle="-|>", color=METHOD_COLORS["ours"],
                         lw=0.7, shrinkA=0, shrinkB=2),
    )


__all__ = [
    "METHOD_COLORS", "METHOD_LABELS", "METHOD_ORDER",
    "DIAG_COLORS", "PATHWAY", "ACCENT", "STORY",
    "MM2IN", "mm", "style_setup",
    "panel_label", "figure_title", "takeaway_banner",
    "win_badge", "highlight_xticklabel", "highlight_yticklabel",
    "annotate_pathwaypro_bar",
]
