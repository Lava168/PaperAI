#!/usr/bin/env python3
"""Journal-style shared defaults for ARA-Net brain figures.

- Times New Roman (Liberation Serif fallback, metric-compatible)
- Dense native-space cuts (less empty black)
- Nature/Neurology-like palettes (Okabe–Ito + RdBu_r / YlOrRd)
- Explicit bg_img=None (no fake MNI underlay halo)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

try:
    import nibabel as nib
    from nilearn import plotting
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"nibabel/nilearn required: {exc}")


# Dense cuts — fill panel width; covers MTL → superior cortex on our crop affine.
CUTS_DENSE = {
    "z": (-36, -28, -20, -12, -4, 4, 12, 20, 28, 36, 44),  # 11 axial
    "y": (-56, -44, -32, -20, -8, 4, 16, 28, 40),  # 9 coronal
    "x": (-40, -28, -16, -8, 0, 8, 16, 28, 40),  # 9 sagittal
}

# Shorter set when many columns share a row.
CUTS_AXIAL_FOCUS = (-32, -24, -16, -8, 0, 8, 16, 24, 32, 40)

# Okabe–Ito (colorblind-safe; common in Nature/Science figures)
OKABE_ITO = {
    "CN": "#0072B2",
    "MCI": "#E69F00",
    "AD": "#D55E00",
    "V6": "#009E73",
    "V7": "#CC79A7",
    "equal": "#56B4E9",
    "atrophy": "#332288",
    "mrf": "#882255",
    "neutral": "#999999",
    "black": "#000000",
}

# Diverging: blue–white–red (classic neuroimaging / Nature Med style)
CMAP_DIV = "RdBu_r"
# On black panels, white-centered diverging maps create a pale “band”.
# Use a black-centered diverging map so near-zero stays invisible.
CMAP_DIV_BLACK = LinearSegmentedColormap.from_list(
    "ara_div_black",
    [
        (0.00, "#2166AC"),
        (0.35, "#67A9CF"),
        (0.50, "#000000"),
        (0.65, "#EF8A62"),
        (1.00, "#B2182B"),
    ],
)
# Sequential positive (group means / q)
CMAP_SEQ = "YlOrRd"
# MRF posterior q in [0,1]
CMAP_Q = "inferno"

CLIM = {
    "atrophy": (-2.5, 2.5),
    "signed": (-0.55, 0.55),
    "q": (0.0, 1.0),
    "assoc": (-0.55, 0.55),
    "delta": (-0.35, 0.35),
}


def apply_journal_style() -> str:
    """Register Times-compatible serif and set rcParams. Returns resolved family name."""
    serif_path = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
    family = "Liberation Serif"
    if serif_path.exists():
        font_manager.fontManager.addfont(str(serif_path))
        for style in ("Bold", "Italic", "BoldItalic"):
            p = serif_path.parent / f"LiberationSerif-{style}.ttf"
            if p.exists():
                font_manager.fontManager.addfont(str(p))
        # Alias so requests for Times New Roman resolve to Liberation Serif metrics.
        try:
            font_manager.fontManager.addfont(str(serif_path))
            prop = font_manager.FontProperties(fname=str(serif_path))
            family = prop.get_name()
        except Exception:
            family = "Liberation Serif"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "mathtext.fontset": "stix",
        }
    )
    return family


def force_serif_on_figure(fig: plt.Figure) -> None:
    """Ensure every text artist uses the journal serif (nilearn often ignores rcParams)."""
    fp = font_manager.FontProperties(
        fname="/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
    )
    for artist in fig.findobj(match=lambda x: isinstance(x, matplotlib.text.Text)):
        try:
            artist.set_fontproperties(fp)
        except Exception:
            artist.set_fontfamily("serif")


def save_fig(fig: plt.Figure, out_path: Path) -> None:
    force_serif_on_figure(fig)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", facecolor="white", dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] -> {out_path.with_suffix('.png')}")


def plot_dense_strip(
    ax,
    img,
    *,
    display_mode: str = "z",
    cut_coords: Sequence[float] = CUTS_AXIAL_FOCUS,
    vmin: float,
    vmax: float,
    cmap,
    threshold: Optional[float] = None,
) -> None:
    """One tight multi-slice strip; black panel, no MNI/anat underlay."""
    # Signed maps: hide near-zero so black stays black (no white-centered band).
    if threshold is None:
        if vmin < 0 < vmax:
            threshold = max(0.04 * max(abs(vmin), abs(vmax)), 1e-3)
        elif vmin >= 0:
            threshold = 1e-4
    plotting.plot_stat_map(
        img,
        display_mode=display_mode,
        cut_coords=tuple(cut_coords),
        axes=ax,
        colorbar=False,
        annotate=False,
        draw_cross=False,
        black_bg=True,
        bg_img=None,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        threshold=threshold,
        radiological=False,
    )


def load_map(maps_dir: Path, name: str):
    path = Path(maps_dir) / f"{name}.nii.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def difference_map(maps_dir: Path, a: str, b: str):
    """a − b on the shared atlas grid (model advantage contrast)."""
    ia = load_map(maps_dir, a)
    ib = load_map(maps_dir, b)
    da = np.asanyarray(ia.dataobj, dtype=np.float32)
    db = np.asanyarray(ib.dataobj, dtype=np.float32)
    return nib.Nifti1Image(da - db, ia.affine, ia.header.copy())


def signed_clim(img, pct: float = 98.0, floor: float = 0.12) -> Tuple[float, float]:
    data = np.asanyarray(img.dataobj, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return (-floor, floor)
    lim = float(np.nanpercentile(np.abs(finite), pct))
    lim = max(lim, floor)
    return (-lim, lim)


def add_cbar(fig, cmap, vmin: float, vmax: float, label: str, rect) -> None:
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def panel_label(ax, text: str, *, x: float = 0.0, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )
