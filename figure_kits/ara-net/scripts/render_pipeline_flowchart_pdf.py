#!/usr/bin/env python3
"""ARA-Net pipeline flowchart with real MRI/atlas tiles + 3D-style cards → PDF.

Uses FastSurfer AIBL exemplars for panels b/g atlas visuals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import (
    FancyBboxPatch, Circle, Wedge, Polygon, Rectangle, FancyArrowPatch, Ellipse,
)
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "reports" / "v6_final_model" / "figures"
ALT = ROOT / "reports" / "brain_figures_fcstyle" / "figures"

from render_atlas_staging_triptych import (  # type: ignore
    brain_bbox_2d,
    fit_panel_square,
    load_fastsurfer_native,
    normalize,
    plane_slice_indices,
    slice_plane,
)

SERIF = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
SERIF_B = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf")

EDGE = "#1F2A37"
MUTED = "#5B6B7C"
AD_KEY = {4, 43, 17, 53, 18, 54}
SCAN_CN = "AIBL_14_bl_I135977"
SCAN_AD = "AIBL_10_bl_I164086"


def apply_style() -> None:
    if SERIF.exists():
        font_manager.fontManager.addfont(str(SERIF))
        if SERIF_B.exists():
            font_manager.fontManager.addfont(str(SERIF_B))
        fam = font_manager.FontProperties(fname=str(SERIF)).get_name()
    else:
        fam = "Times New Roman"
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", fam, "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def force_times(fig: plt.Figure) -> None:
    if not SERIF.exists():
        return
    fp = font_manager.FontProperties(fname=str(SERIF))
    fp_b = font_manager.FontProperties(fname=str(SERIF_B)) if SERIF_B.exists() else fp
    for artist in fig.findobj(match=lambda x: isinstance(x, matplotlib.text.Text)):
        try:
            wt = artist.get_fontweight()
            artist.set_fontproperties(fp_b if str(wt) in ("bold", "700") else fp)
        except Exception:
            artist.set_fontfamily("serif")


# ── visual helpers (3D card look) ───────────────────────────────────────────

def soft_shadow(ax, x, y, w, h, layers=4):
    for i in range(layers, 0, -1):
        d = 0.006 * i
        ax.add_patch(FancyBboxPatch(
            (x + d, y - d), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.025",
            facecolor=(0, 0, 0, 0.04 * i), edgecolor="none",
            transform=ax.transAxes, zorder=1, clip_on=False,
        ))


def card(ax, x, y, w, h, fc="#FFFFFF", ec="#D0D7DE", z=3):
    soft_shadow(ax, x, y, w, h)
    # top highlight strip for “emboss”
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        facecolor=fc, edgecolor=ec, linewidth=0.9,
        transform=ax.transAxes, zorder=z, clip_on=False,
    ))
    ax.add_patch(Rectangle(
        (x + 0.01, y + h - 0.018), w - 0.02, 0.012,
        transform=ax.transAxes, facecolor=(1, 1, 1, 0.45),
        edgecolor="none", zorder=z + 1, clip_on=False,
    ))


def text_card(ax, x, y, w, h, text, fc="#FFFFFF", fs=6.0, weight="normal", tc=EDGE):
    card(ax, x, y, w, h, fc=fc)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, color=tc, transform=ax.transAxes, zorder=10,
            linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color="#3D5A80"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=1.3,
            connectionstyle="arc3,rad=0.0",
            mutation_scale=10,
        ),
        zorder=8,
    )


def icon_person(ax, x, y, color="#5B8DEF", s=0.035):
    ax.add_patch(Circle((x, y + s * 0.9), s * 0.35, transform=ax.transAxes,
                        facecolor=color, edgecolor="white", lw=0.4, zorder=12, clip_on=False))
    ax.add_patch(FancyBboxPatch(
        (x - s * 0.45, y - s * 0.2), s * 0.9, s * 0.85,
        boxstyle="round,pad=0.002,rounding_size=0.01",
        facecolor=color, edgecolor="white", lw=0.3,
        transform=ax.transAxes, zorder=12, clip_on=False,
    ))


def icon_lock(ax, x, y, s=0.04):
    ax.add_patch(FancyBboxPatch(
        (x - s * 0.45, y - s * 0.35), s * 0.9, s * 0.7,
        boxstyle="round,pad=0.002,rounding_size=0.008",
        facecolor="#E57373", edgecolor="white", lw=0.4,
        transform=ax.transAxes, zorder=12, clip_on=False,
    ))
    ax.add_patch(Wedge((x, y + s * 0.15), s * 0.35, 0, 180, width=s * 0.12,
                       transform=ax.transAxes, facecolor="#C62828", zorder=12, clip_on=False))


def icon_shield(ax, x, y, s=0.05):
    verts = np.array([
        [x, y + s], [x + s * 0.7, y + s * 0.55], [x + s * 0.55, y - s * 0.7],
        [x, y - s], [x - s * 0.55, y - s * 0.7], [x - s * 0.7, y + s * 0.55],
    ])
    ax.add_patch(Polygon(verts, closed=True, transform=ax.transAxes,
                         facecolor="#42A5F5", edgecolor="white", lw=0.6, zorder=12, clip_on=False))
    ax.plot([x - s * 0.2, x - s * 0.02, x + s * 0.28],
            [y, y - s * 0.25, y + s * 0.35],
            color="white", lw=1.4, solid_capstyle="round",
            transform=ax.transAxes, zorder=13, clip_on=False)


def icon_target(ax, x, y, s=0.045):
    for r, c in [(s, "#E8F5E9"), (s * 0.65, "#81C784"), (s * 0.35, "#2E7D32")]:
        ax.add_patch(Circle((x, y), r, transform=ax.transAxes, facecolor=c,
                            edgecolor="white", lw=0.3, zorder=12, clip_on=False))


def panel_frame(ax, letter: str, title: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    # outer 3D panel shell
    soft_shadow(ax, 0.01, 0.01, 0.98, 0.98, layers=5)
    ax.add_patch(FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.008,rounding_size=0.03",
        facecolor="#FAFBFC", edgecolor="#CFD8DC", linewidth=1.1,
        transform=ax.transAxes, zorder=0, clip_on=False,
    ))
    # left accent bar
    ax.add_patch(Rectangle((0.01, 0.01), 0.012, 0.98, transform=ax.transAxes,
                           facecolor="#3D5A80", edgecolor="none", zorder=2, clip_on=False))
    ax.text(0.04, 0.965, letter, ha="left", va="top", fontsize=16, fontweight="bold",
            color="black", transform=ax.transAxes, zorder=20)
    ax.text(0.52, 0.965, title, ha="center", va="top", fontsize=8.0, fontweight="bold",
            color=EDGE, transform=ax.transAxes, zorder=20)


# ── real imaging helpers ────────────────────────────────────────────────────

def mri_rgb(mri_2d: np.ndarray) -> np.ndarray:
    g = normalize(mri_2d)
    return np.stack([g, g, g], axis=-1)


def atlas_rgb(mri_2d: np.ndarray, seg_2d: np.ndarray, mode: str = "full") -> np.ndarray:
    g = normalize(mri_2d)
    brain = g > 0.04
    rgb = np.zeros((*g.shape, 3), dtype=np.float32)
    for c in range(3):
        rgb[..., c][brain] = 0.12 + 0.55 * g[brain]
    if mode == "full":
        # colorful parcellation accents
        rng = np.random.default_rng(0)
        labs = [l for l in np.unique(seg_2d) if l > 0]
        for lab in labs:
            col = rng.random(3) * 0.55 + 0.35
            m = seg_2d == lab
            rgb[m] = 0.35 * rgb[m] + 0.65 * col
    else:
        accents = {
            4: (0.25, 0.55, 1.0), 43: (0.25, 0.55, 1.0),
            17: (1.0, 0.25, 0.2), 53: (1.0, 0.25, 0.2),
            18: (1.0, 0.6, 0.1), 54: (1.0, 0.6, 0.1),
        }
        for lab, col in accents.items():
            m = seg_2d == lab
            if np.any(m):
                rgb[m] = col
    rgb[~brain] = 0.02
    return np.clip(rgb, 0, 1)


def get_slice_pair(img, seg, plane: str, which: int = 0, size: int = 96):
    ids = plane_slice_indices(img.shape, plane, n=5, seg=seg)
    idx = ids[min(which, len(ids) - 1)]
    mri_sl = slice_plane(img, plane, idx)
    seg_sl = slice_plane(seg, plane, idx)
    mri_c, seg_c = fit_panel_square(mri_sl, seg_sl, size=size)
    return mri_c, seg_c


def show_img(ax, arr, extent):
    """extent = [x0,x1,y0,y1] in axes fraction."""
    x0, x1, y0, y1 = extent
    # inset-like via fig coords of axes
    soft_shadow(ax, x0, y0, x1 - x0, y1 - y0, layers=3)
    ax_i = ax.inset_axes([x0, y0, x1 - x0, y1 - y0])
    ax_i.imshow(arr, interpolation="bilinear")
    ax_i.set_xticks([])
    ax_i.set_yticks([])
    for s in ax_i.spines.values():
        s.set_color("white")
        s.set_linewidth(1.2)
    return ax_i


# ── panels ──────────────────────────────────────────────────────────────────

def draw_a(ax):
    panel_frame(ax, "a", "Subject-level multi-cohort split")
    rows = [
        (0.70, "Cohort A  ·  Site 1", "#90CAF9", "#1565C0"),
        (0.50, "Cohort B  ·  Site 2", "#FFCC80", "#EF6C00"),
        (0.30, "Cohort C  ·  Site 3", "#F48FB1", "#C2185B"),
    ]
    for y, name, fc, ic in rows:
        card(ax, 0.05, y, 0.40, 0.16, fc=fc)
        for k in range(4):
            icon_person(ax, 0.10 + k * 0.07, y + 0.055, color=ic, s=0.032)
        ax.text(0.38, y + 0.11, name, ha="right", va="center", fontsize=5.8,
                fontweight="bold", color=EDGE, transform=ax.transAxes, zorder=12)
        ax.text(0.38, y + 0.04, "Visit 1…n_i", ha="right", va="center", fontsize=5.0,
                color="black", transform=ax.transAxes, zorder=12)

    icon_shield(ax, 0.56, 0.55, s=0.07)
    ax.text(0.56, 0.42, "No data\nleakage", ha="center", va="top", fontsize=5.8,
            fontweight="bold", color="#1565C0", transform=ax.transAxes, zorder=12)

    outs = [
        (0.72, "Development", "#A5D6A7", "#2E7D32"),
        (0.52, "Validation", "#FFCC80", "#EF6C00"),
        (0.32, "Locked external\ntest", "#F48FB1", "#C2185B"),
    ]
    for y, name, fc, ic in outs:
        card(ax, 0.68, y, 0.28, 0.16, fc=fc)
        for k in range(3):
            icon_person(ax, 0.73 + k * 0.06, y + 0.09, color=ic, s=0.028)
        if "Locked" in name:
            icon_lock(ax, 0.91, y + 0.09, s=0.035)
        ax.text(0.82, y + 0.035, name, ha="center", va="center", fontsize=5.5,
                fontweight="bold", color=EDGE, transform=ax.transAxes, zorder=12)

    arrow(ax, 0.46, 0.58, 0.50, 0.55)
    arrow(ax, 0.62, 0.55, 0.68, 0.80)
    arrow(ax, 0.62, 0.55, 0.68, 0.60)
    arrow(ax, 0.62, 0.55, 0.68, 0.40)
    ax.text(0.5, 0.08, "All visits from one subject stay together.", ha="center",
            fontsize=5.8, color=MUTED, transform=ax.transAxes, zorder=12)


def draw_b(ax, img_cn, seg_cn):
    panel_frame(ax, "b", "Atlas-guided sMRI processing")
    mri_ax, seg_ax = get_slice_pair(img_cn, seg_cn, "axial", which=2, size=110)
    mri_cor, seg_cor = get_slice_pair(img_cn, seg_cn, "coronal", which=2, size=110)
    atl = atlas_rgb(mri_cor, seg_cor, mode="full")
    adk = atlas_rgb(mri_cor, seg_cor, mode="adkey")

    show_img(ax, mri_rgb(mri_ax), [0.05, 0.20, 0.55, 0.82])
    show_img(ax, mri_rgb(mri_cor), [0.22, 0.37, 0.55, 0.82])
    ax.text(0.125, 0.50, "Axial", ha="center", fontsize=5.2, color="black", transform=ax.transAxes, zorder=12)
    ax.text(0.295, 0.50, "Coronal", ha="center", fontsize=5.2, color="black", transform=ax.transAxes, zorder=12)
    ax.text(0.21, 0.44, "T1-weighted sMRI", ha="center", fontsize=5.5, color=MUTED, transform=ax.transAxes, zorder=12)

    arrow(ax, 0.38, 0.70, 0.44, 0.70)
    ax.text(0.41, 0.74, "Registration", ha="center", fontsize=5.2, color=MUTED, transform=ax.transAxes, zorder=12)
    show_img(ax, atl, [0.44, 0.62, 0.52, 0.84])
    ax.text(0.53, 0.48, "Predefined atlas", ha="center", fontsize=5.4, fontweight="bold",
            color=EDGE, transform=ax.transAxes, zorder=12)

    # feature cards
    text_card(ax, 0.66, 0.74, 0.30, 0.12, "Regional volume", fc="#E3F2FD", fs=6.0, weight="bold")
    text_card(ax, 0.66, 0.56, 0.30, 0.14, "Intensity statistics\nMean · Std · Skew · Kurtosis",
              fc="#E8F5E9", fs=5.4, weight="bold")
    xs = np.linspace(-2.5, 2.5, 60)
    ys = np.exp(-0.5 * xs ** 2)
    ax_g = ax.inset_axes([0.70, 0.575, 0.22, 0.06])
    ax_g.fill_between(xs, ys, color="#66BB6A", alpha=0.75)
    ax_g.axis("off")

    show_img(ax, adk, [0.66, 0.90, 0.22, 0.50])
    ax.text(0.78, 0.18, "AD-key: Hipp / Amyg / Vent", ha="center", fontsize=5.2,
            color="black", transform=ax.transAxes, zorder=12)
    cb = ax.inset_axes([0.05, 0.12, 0.45, 0.035])
    cb.imshow(np.linspace(0, 1, 64)[None, :], aspect="auto", cmap="plasma")
    cb.set_yticks([])
    cb.set_xticks([0, 63])
    cb.set_xticklabels(["Low", "High"], fontsize=5)
    ax.text(
        0.5, 0.05,
        r"$x'_{ij}=(x_{ij}-\mu_j^{\mathrm{dev}})/\sigma_j^{\mathrm{dev}}$"
        r"  ·  $m_{ij}=I(x_{ij}\ \mathrm{missing})$",
        ha="center", fontsize=5.4, color=EDGE, transform=ax.transAxes, zorder=12,
    )


def draw_c(ax, img_cn, seg_cn):
    panel_frame(ax, "c", "Atlas-guided multimodal input")
    # build a pseudo feature matrix from real regional volumes-ish random but styled
    rng = np.random.default_rng(2)
    mat = rng.normal(size=(10, 12))
    mat = (mat - mat.min()) / (mat.max() - mat.min())
    soft_shadow(ax, 0.05, 0.38, 0.42, 0.48, layers=3)
    axi = ax.inset_axes([0.05, 0.38, 0.42, 0.48])
    axi.imshow(mat, cmap="Blues", aspect="auto")
    axi.set_xticks([])
    axi.set_yticks([])
    axi.set_xlabel("Features", fontsize=5.5, color="black")
    axi.set_ylabel("Subjects", fontsize=5.5, color="black")
    axi.set_title("Atlas-derived imaging matrix", fontsize=6.0, pad=2)

    clin = [
        ("Age", "#5C6BC0"), ("Sex", "#26A69A"), ("Education", "#7E57C2"),
        ("APOE4", "#EF5350"), ("MMSE", "#42A5F5"), ("CDR-SB", "#FFA726"),
    ]
    ax.text(0.56, 0.88, "Clinical variables", fontsize=6.8, fontweight="bold",
            color=EDGE, transform=ax.transAxes, zorder=12)
    for i, (name, col) in enumerate(clin):
        y = 0.78 - i * 0.08
        card(ax, 0.55, y, 0.40, 0.07, fc="#FFFFFF")
        ax.add_patch(Circle((0.60, y + 0.035), 0.016, transform=ax.transAxes,
                            facecolor=col, edgecolor="white", lw=0.5, zorder=14, clip_on=False))
        ax.text(0.65, y + 0.035, name, fontsize=6.2, va="center", color="black",
                transform=ax.transAxes, zorder=14)

    arrow(ax, 0.26, 0.36, 0.26, 0.28)
    arrow(ax, 0.75, 0.36, 0.50, 0.28)
    # fused strip with 3D bevel
    soft_shadow(ax, 0.10, 0.08, 0.80, 0.14, layers=4)
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, 20))
    for i, col in enumerate(colors):
        ax.add_patch(Rectangle(
            (0.12 + i * 0.038, 0.10), 0.036, 0.10,
            transform=ax.transAxes, facecolor=col, edgecolor="white", lw=0.35, zorder=10, clip_on=False,
        ))
    ax.text(0.5, 0.04, "Fused feature vector (per subject)", ha="center", fontsize=6.2,
            fontweight="bold", color=EDGE, transform=ax.transAxes, zorder=12)


def draw_d(ax):
    panel_frame(ax, "d", "Six heterogeneous probability streams")
    text_card(ax, 0.04, 0.55, 0.18, 0.30, "Fused\nfeature\nvector", fc="#E3F2FD", fs=6.5, weight="bold")
    names = [
        "1 Gradient Boosting", "2 Random Forest", "3 SVM (RBF)",
        "4 Logistic Regression", "5 MLP", "6 k-NN",
    ]
    for i, name in enumerate(names):
        y = 0.78 - i * 0.11
        text_card(ax, 0.28, y, 0.30, 0.09, name, fc="#FFFFFF", fs=5.8, weight="bold")
        arrow(ax, 0.22, 0.70, 0.28, y + 0.045)
        # 3D probability chips
        for j, (lab, col) in enumerate([("CN", "#42A5F5"), ("MCI", "#FFA726"), ("AD", "#EF5350")]):
            x = 0.62 + j * 0.11
            soft_shadow(ax, x, y + 0.015, 0.095, 0.06, layers=2)
            ax.add_patch(FancyBboxPatch(
                (x, y + 0.015), 0.095, 0.06,
                boxstyle="round,pad=0.004,rounding_size=0.01",
                facecolor=col, edgecolor="white", lw=0.5,
                transform=ax.transAxes, zorder=10, clip_on=False,
            ))
            if i == 0:
                ax.text(x + 0.047, 0.90, lab, ha="center", fontsize=5.5, fontweight="bold",
                        color="black", transform=ax.transAxes, zorder=12)
    ax.text(
        0.5, 0.08,
        r"$p_m(x_i)=[p_{m,\mathrm{CN}},\,p_{m,\mathrm{MCI}},\,p_{m,\mathrm{AD}}],\ m=1,\ldots,6$",
        ha="center", fontsize=6.5, color=EDGE, transform=ax.transAxes, zorder=12,
    )


def draw_e(ax):
    panel_frame(ax, "e", "RC-SPE risk-constrained fusion & calibration")
    text_card(ax, 0.04, 0.68, 0.16, 0.18, r"$p_m(x_i)$" + "\n× 6", fc="#FFF8E1", fs=7.0, weight="bold")
    arrow(ax, 0.20, 0.77, 0.25, 0.77)
    text_card(
        ax, 0.25, 0.52, 0.46, 0.36,
        "RC-SPE fusion\n"
        r"$z_{i,k}=\frac{1}{T}\left[\sum_m w_m\log(\max(p_{m,k},\varepsilon))+b_k\right]$"
        "\n"
        r"$w_m\geq 0,\ \sum_m w_m=1$",
        fc="#E3F2FD", fs=5.8, weight="bold",
    )
    for i, (y, t) in enumerate([(0.78, r"Class bias $b_k$"), (0.62, r"Temp. $T$"), (0.46, "Softmax")]):
        text_card(ax, 0.76, y, 0.20, 0.11, t, fc="#F3E5F5", fs=5.8, weight="bold")
        if i == 0:
            arrow(ax, 0.71, 0.78, 0.76, 0.835)
        else:
            arrow(ax, 0.86, y + 0.16, 0.86, y + 0.11)
    ax.text(
        0.5, 0.30,
        r"$\hat{p}_{i,k}=\exp(z_{i,k})/\sum_c\exp(z_{i,c})$",
        ha="center", fontsize=7.0, color=EDGE, transform=ax.transAxes, zorder=12,
    )
    for j, col in enumerate(["#42A5F5", "#FFA726", "#EF5350"]):
        soft_shadow(ax, 0.28 + j * 0.15, 0.10, 0.13, 0.12, layers=3)
        ax.add_patch(FancyBboxPatch(
            (0.28 + j * 0.15, 0.10), 0.13, 0.12,
            boxstyle="round,pad=0.006,rounding_size=0.015",
            facecolor=col, edgecolor="white", lw=0.8,
            transform=ax.transAxes, zorder=10, clip_on=False,
        ))
    ax.text(0.5, 0.05, "Calibrated scan-level probability", ha="center", fontsize=5.8,
            color=MUTED, transform=ax.transAxes, zorder=12)


def draw_f(ax):
    panel_frame(ax, "f", "Repeated-scan subject aggregation")
    ax.text(0.22, 0.88, "Subject s  (real longitudinal visits)", ha="center", fontsize=6.8,
            fontweight="bold", color=EDGE, transform=ax.transAxes, zorder=12)
    for i, lab in enumerate(["Scan 1", "Scan 2", r"Scan $n_s$"]):
        y = 0.70 - i * 0.17
        text_card(ax, 0.05, y, 0.24, 0.13, lab, fc="#FFF8E1", fs=6.2, weight="bold")
        for j, col in enumerate(["#42A5F5", "#FFA726", "#EF5350"]):
            ax.add_patch(FancyBboxPatch(
                (0.32 + j * 0.09, y + 0.03), 0.08, 0.07,
                boxstyle="round,pad=0.004,rounding_size=0.01",
                facecolor=col, edgecolor="white", lw=0.4,
                transform=ax.transAxes, zorder=12, clip_on=False,
            ))
        arrow(ax, 0.60, y + 0.06, 0.66, 0.42)
    text_card(ax, 0.66, 0.34, 0.30, 0.16, "Mean probability\n(per subject)", fc="#E3F2FD", fs=6.5, weight="bold")
    ax.text(
        0.5, 0.18,
        r"$\tilde{p}_{s,k}=\frac{1}{n_s}\sum_i\hat{p}_{i,k}$"
        r"   ·   $\hat{y}_s=\arg\max_k\tilde{p}_{s,k}$",
        ha="center", fontsize=6.2, color=EDGE, transform=ax.transAxes, zorder=12,
    )
    ax.text(
        0.5, 0.08,
        r"$\mathrm{Conf}_s=\max_k\tilde{p}_{s,k}$"
        r"   ·   $\mathrm{Margin}_s=\tilde{p}_{s,(1)}-\tilde{p}_{s,(2)}$",
        ha="center", fontsize=6.2, color=EDGE, transform=ax.transAxes, zorder=12,
    )


def draw_g(ax, img_ad, seg_ad, img_cn, seg_cn):
    panel_frame(ax, "g", "Output & validation")
    text_card(ax, 0.04, 0.68, 0.20, 0.18, "Predicted stage\nCN / MCI / AD", fc="#FFE0B2", fs=6.0, weight="bold")

    # 3D-ish gauge
    ag = ax.inset_axes([0.28, 0.60, 0.28, 0.30])
    ag.set_xlim(-1.2, 1.2)
    ag.set_ylim(-0.2, 1.2)
    ag.axis("off")
    for i in range(40):
        t0 = 180 * (1 - i / 40)
        t1 = 180 * (1 - (i + 1) / 40)
        ag.add_patch(Wedge((0, 0), 1.0, t1, t0, width=0.34, facecolor=plt.cm.turbo(i / 40),
                           edgecolor="none"))
    # rim
    ag.add_patch(Wedge((0, 0), 1.02, 0, 180, width=0.04, facecolor="#37474F", edgecolor="none"))
    ang = np.deg2rad(180 * (1 - 0.76))
    ag.plot([0, 0.85 * np.cos(ang)], [0, 0.85 * np.sin(ang)], color="black", lw=2.0,
            solid_capstyle="round",
            path_effects=[pe.SimpleLineShadow(offset=(0.5, -0.5), alpha=0.35), pe.Normal()])
    ag.text(0, -0.05, r"$\mathrm{Conf}_s=0.76$", ha="center", va="top", fontsize=6.2, fontweight="bold")

    tri = np.array([[0.68, 0.66], [0.94, 0.66], [0.81, 0.92]])
    soft_shadow(ax, 0.68, 0.66, 0.26, 0.26, layers=2)
    ax.add_patch(Polygon(tri, closed=True, fill=True, fc="#ECEFF1", ec=EDGE, lw=1.0,
                         transform=ax.transAxes, zorder=10))
    ax.plot(0.80, 0.76, "o", color="#FFA726", ms=6, transform=ax.transAxes, zorder=12,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.text(0.68, 0.62, "CN", fontsize=5, transform=ax.transAxes, zorder=12, color="black")
    ax.text(0.92, 0.62, "AD", fontsize=5, transform=ax.transAxes, zorder=12, color="black")
    ax.text(0.81, 0.94, "MCI", fontsize=5, ha="center", transform=ax.transAxes, zorder=12, color="black")

    text_card(ax, 0.04, 0.30, 0.20, 0.18, "Locked\nexternal test", fc="#F8BBD0", fs=5.8, weight="bold")
    icon_lock(ax, 0.14, 0.42, s=0.04)
    text_card(ax, 0.28, 0.30, 0.26, 0.18,
              "HC specificity\n" + r"$BAcc=\frac{1}{3}\sum_k Recall_k$",
              fc="#C8E6C9", fs=5.4, weight="bold")
    icon_target(ax, 0.34, 0.42, s=0.035)

    cm = np.array([[148, 6, 0], [2, 24, 9], [0, 4, 23]], dtype=float)
    soft_shadow(ax, 0.60, 0.26, 0.34, 0.28, layers=3)
    acm = ax.inset_axes([0.60, 0.26, 0.34, 0.28])
    acm.imshow(cm, cmap="Blues", aspect="auto")
    acm.set_xticks([0, 1, 2])
    acm.set_yticks([0, 1, 2])
    acm.set_xticklabels(["CN", "MCI", "AD"], fontsize=5, color="black")
    acm.set_yticklabels(["CN", "MCI", "AD"], fontsize=5, color="black")
    acm.tick_params(length=0)
    acm.set_title("Error transition (AIBL heldout)", fontsize=5.5, pad=2, color="black")
    for i in range(3):
        for j in range(3):
            acm.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=5.5,
                     color="white" if cm[i, j] > 40 else "black")

    # real AD vs CN atrophy examples
    mri_ad, seg_ad = get_slice_pair(img_ad, seg_ad, "coronal", which=2, size=90)
    mri_cn, seg_cn = get_slice_pair(img_cn, seg_cn, "coronal", which=2, size=90)
    show_img(ax, atlas_rgb(mri_cn, seg_cn, "adkey"), [0.05, 0.18, 0.04, 0.22])
    show_img(ax, atlas_rgb(mri_ad, seg_ad, "adkey"), [0.20, 0.33, 0.04, 0.22])
    ax.text(0.115, 0.01, "CN", ha="center", fontsize=5.2, color="black", transform=ax.transAxes, zorder=12)
    ax.text(0.265, 0.01, "AD", ha="center", fontsize=5.2, color="black", transform=ax.transAxes, zorder=12)
    ax.text(
        0.70, 0.10,
        r"$d=\dfrac{\mu_{\mathrm{AD}}-\mu_{\mathrm{CN}}}{\sqrt{(\sigma_{\mathrm{AD}}^2+\sigma_{\mathrm{CN}}^2)/2}}$",
        ha="center", fontsize=5.8, color=EDGE, transform=ax.transAxes, zorder=12,
    )
    ax.text(0.70, 0.03, "Atlas–neurodegeneration consistency (real AIBL)", ha="center",
            fontsize=5.2, color=MUTED, transform=ax.transAxes, zorder=12)


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    ALT.mkdir(parents=True, exist_ok=True)

    print("loading FastSurfer exemplars…")
    cn = load_fastsurfer_native(SCAN_CN)
    ad = load_fastsurfer_native(SCAN_AD)
    if cn is None or ad is None:
        raise SystemExit("cannot load FastSurfer volumes")
    img_cn, seg_cn = cn
    img_ad, seg_ad = ad
    print(f"loaded CN={SCAN_CN}  AD={SCAN_AD}")

    fig = plt.figure(figsize=(16.5, 11.2), facecolor="#EEF1F4")
    gs = GridSpec(
        3, 6, figure=fig,
        height_ratios=[1.0, 1.05, 1.08],
        hspace=0.12, wspace=0.14,
        left=0.02, right=0.985, top=0.93, bottom=0.03,
    )
    draw_a(fig.add_subplot(gs[0, 0:2]))
    draw_b(fig.add_subplot(gs[0, 2:4]), img_cn, seg_cn)
    draw_c(fig.add_subplot(gs[0, 4:6]), img_cn, seg_cn)
    draw_d(fig.add_subplot(gs[1, 0:3]))
    draw_e(fig.add_subplot(gs[1, 3:6]))
    draw_f(fig.add_subplot(gs[2, 0:3]))
    draw_g(fig.add_subplot(gs[2, 3:6]), img_ad, seg_ad, img_cn, seg_cn)

    fig.text(
        0.5, 0.975,
        "ARA-Net pipeline  ·  real AIBL FastSurfer exemplars  ·  RC-SPE evidence chain",
        ha="center", va="top", fontsize=12, fontweight="bold", color=EDGE,
    )
    fig.text(
        0.5, 0.948,
        "Panels a–g · 16 pt panel letters · 3D-style cards · real MRI / AD-key atlas tiles",
        ha="center", va="top", fontsize=7.5, color=MUTED,
    )
    force_times(fig)

    pdf = OUT / "figure1_pipeline_flowchart.pdf"
    png = OUT / "figure1_pipeline_flowchart.png"
    fig.savefig(pdf, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(png, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)

    (ALT / pdf.name).write_bytes(pdf.read_bytes())
    (ALT / png.name).write_bytes(png.read_bytes())
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    print(f"copied to {ALT}")


if __name__ == "__main__":
    main()
