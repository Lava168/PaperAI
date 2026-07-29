#!/usr/bin/env python3
"""Rebuild ARA-Net pipeline figure as crisp vector PDF (white bg).

Workflow:
  1) CLAHE contrast-enhance the low-res source (readability)
  2) Rebuild panels a–g as native matplotlib vectors + real FastSurfer tiles
     so zoom stays sharp (unlike upscaled raster).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Wedge
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "reports" / "v6_final_model" / "figures"
ALT = ROOT / "reports" / "brain_figures_fcstyle" / "figures"
SRC = ALT / "image.png"

from render_atlas_staging_triptych import (  # type: ignore
    fit_panel_square,
    load_fastsurfer_native,
    normalize,
    plane_slice_indices,
    slice_plane,
)

SERIF = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
SERIF_B = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf")
EDGE = "#222222"
MUTED = "#555555"
CN, MCI, AD = "#4FC3F7", "#FFB74D", "#EF9A9A"
SCAN_CN, SCAN_AD = "AIBL_14_bl_I135977", "AIBL_10_bl_I164086"


def apply_style():
    if SERIF.exists():
        font_manager.fontManager.addfont(str(SERIF))
        if SERIF_B.exists():
            font_manager.fontManager.addfont(str(SERIF_B))
        fam = font_manager.FontProperties(fname=str(SERIF)).get_name()
    else:
        fam = "Times New Roman"
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", fam, "Liberation Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def force_times(fig):
    if not SERIF.exists():
        return
    fp = font_manager.FontProperties(fname=str(SERIF))
    fpb = font_manager.FontProperties(fname=str(SERIF_B)) if SERIF_B.exists() else fp
    for t in fig.findobj(match=lambda x: isinstance(x, matplotlib.text.Text)):
        try:
            t.set_fontproperties(fpb if str(t.get_fontweight()) in ("bold", "700") else fp)
        except Exception:
            pass


def contrast_enhance(src: Path, out: Path) -> Path:
    im = Image.open(src).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    base = Image.alpha_composite(bg, im).convert("RGB")
    up = base.resize((base.width * 4, base.height * 4), Image.Resampling.LANCZOS)
    arr = cv2.cvtColor(np.asarray(up), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    enh = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enh, (0, 0), 1.2)
    sharp = cv2.addWeighted(enh, 1.55, blur, -0.55, 0)
    rgb = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
    pil = ImageEnhance.Contrast(Image.fromarray(rgb)).enhance(1.2)
    pil = ImageEnhance.Sharpness(pil).enhance(1.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out, dpi=(300, 300))
    return out


def box(ax, x, y, w, h, fc="white", ec=EDGE, lw=0.9):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=fc, edgecolor=ec, linewidth=lw, transform=ax.transAxes, clip_on=False, zorder=2,
    )
    ax.add_patch(p)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0), xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.0, mutation_scale=9), zorder=5,
    )


def panel(ax, letter, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(EDGE)
        s.set_linewidth(1.15)
    ax.set_facecolor("white")
    ax.text(0.015, 0.97, letter, fontsize=16, fontweight="bold", color="black",
            ha="left", va="top", transform=ax.transAxes, zorder=20)
    ax.text(0.07, 0.97, title, fontsize=8.5, fontweight="bold", color=EDGE,
            ha="left", va="top", transform=ax.transAxes, zorder=20)


def mri_rgb(m):
    g = normalize(m)
    return np.stack([g, g, g], -1)


def atlas_full(m, s):
    g = normalize(m)
    brain = g > 0.04
    rgb = np.zeros((*g.shape, 3), np.float32)
    for c in range(3):
        rgb[..., c][brain] = 0.1 + 0.55 * g[brain]
    rng = np.random.default_rng(1)
    for lab in np.unique(s):
        if lab <= 0:
            continue
        col = rng.random(3) * 0.6 + 0.35
        rgb[s == lab] = 0.3 * rgb[s == lab] + 0.7 * col
    rgb[~brain] = 0
    return np.clip(rgb, 0, 1)


def atlas_adkey(m, s):
    g = normalize(m)
    brain = g > 0.04
    rgb = np.zeros((*g.shape, 3), np.float32)
    for c in range(3):
        rgb[..., c][brain] = 0.12 + 0.5 * g[brain]
    cols = {4: (0.3, 0.55, 1), 43: (0.3, 0.55, 1), 17: (1, 0.3, 0.2), 53: (1, 0.3, 0.2),
            18: (1, 0.65, 0.15), 54: (1, 0.65, 0.15)}
    for lab, col in cols.items():
        rgb[s == lab] = col
    rgb[~brain] = 0
    return np.clip(rgb, 0, 1)


def get_pair(img, seg, plane, which=2, size=120):
    ids = plane_slice_indices(img.shape, plane, n=5, seg=seg)
    idx = ids[min(which, len(ids) - 1)]
    return fit_panel_square(slice_plane(img, plane, idx), slice_plane(seg, plane, idx), size=size)


def put_img(ax, arr, x, y, w, h):
    ia = ax.inset_axes([x, y, w, h])
    ia.imshow(arr, interpolation="bilinear")
    ia.set_xticks([])
    ia.set_yticks([])
    for s in ia.spines.values():
        s.set_color(EDGE)
        s.set_linewidth(0.6)
    return ia


def prob_bar(ax, x, y, w=0.16, h=0.06):
    for i, c in enumerate([CN, MCI, AD]):
        ax.add_patch(Rectangle((x + i * w / 3, y), w / 3, h, transform=ax.transAxes,
                               facecolor=c, edgecolor=EDGE, lw=0.4, zorder=6, clip_on=False))


def draw_a(ax):
    panel(ax, "a", "Subject-level multi-cohort split")
    for i, (name, fc) in enumerate([
        ("Cohort A (Site 1)", "#BBDEFB"),
        ("Cohort B (Site 2)", "#FFE0B2"),
        ("Cohort C (Site 3)", "#F8BBD0"),
    ]):
        y = 0.72 - i * 0.22
        box(ax, 0.04, y, 0.42, 0.18, fc=fc, ec="#888")
        ax.text(0.06, y + 0.13, name, fontsize=6.2, fontweight="bold", color="black",
                transform=ax.transAxes, zorder=8)
        for k, lab in enumerate(["Visit 1", "Visit 2", r"Visit $n_i$"]):
            ax.add_patch(Rectangle((0.08 + k * 0.12, y + 0.03), 0.09, 0.08, transform=ax.transAxes,
                                   facecolor="#333", edgecolor="white", lw=0.4, zorder=7, clip_on=False))
            ax.text(0.125 + k * 0.12, y + 0.015, lab, ha="center", fontsize=4.8, color="black",
                    transform=ax.transAxes, zorder=8)
    # shield
    box(ax, 0.50, 0.42, 0.12, 0.22, fc="#E3F2FD", ec="#1976D2")
    ax.text(0.56, 0.58, "No data\nleakage", ha="center", va="center", fontsize=5.8,
            fontweight="bold", color="#1565C0", transform=ax.transAxes, zorder=8)
    for i, (name, fc) in enumerate([
        ("Development set", "#C8E6C9"),
        ("Validation set", "#FFE0B2"),
        ("Locked external\ntest set", "#FFCDD2"),
    ]):
        y = 0.72 - i * 0.22
        box(ax, 0.68, y, 0.28, 0.18, fc=fc)
        ax.text(0.82, y + 0.09, name, ha="center", va="center", fontsize=6.0,
                fontweight="bold", color="black", transform=ax.transAxes, zorder=8)
        arrow(ax, 0.46, 0.53, 0.50, 0.53)
        arrow(ax, 0.62, 0.53, 0.68, y + 0.09)


def draw_b(ax, img, seg):
    panel(ax, "b", "Atlas-guided sMRI processing")
    ma, sa = get_pair(img, seg, "axial", 2)
    mc, sc = get_pair(img, seg, "coronal", 2)
    put_img(ax, mri_rgb(ma), 0.04, 0.62, 0.14, 0.24)
    put_img(ax, mri_rgb(mc), 0.04, 0.34, 0.14, 0.24)
    ax.text(0.11, 0.88, "T1-weighted sMRI", ha="center", fontsize=5.8, color="black", transform=ax.transAxes)
    ax.text(0.11, 0.58, "Axial", ha="center", fontsize=5.2, color="black", transform=ax.transAxes)
    ax.text(0.11, 0.30, "Coronal", ha="center", fontsize=5.2, color="black", transform=ax.transAxes)
    arrow(ax, 0.19, 0.58, 0.26, 0.58)
    ax.text(0.225, 0.62, "Registration", ha="center", fontsize=5.2, color=MUTED, transform=ax.transAxes)
    put_img(ax, atlas_full(mc, sc), 0.26, 0.38, 0.22, 0.42)
    ax.text(0.37, 0.32, "Predefined brain atlas\n(parcellation)", ha="center", fontsize=5.5,
            fontweight="bold", color="black", transform=ax.transAxes)
    box(ax, 0.52, 0.72, 0.44, 0.14, fc="#E3F2FD")
    ax.text(0.74, 0.82, "Regional volume", ha="center", fontsize=6.0, fontweight="bold",
            color="black", transform=ax.transAxes)
    for i, hh in enumerate([0.04, 0.07, 0.05, 0.09, 0.06, 0.08]):
        ax.add_patch(Rectangle((0.58 + i * 0.05, 0.74), 0.035, hh, transform=ax.transAxes,
                               facecolor="#0288D1", zorder=8, clip_on=False))
    box(ax, 0.52, 0.52, 0.44, 0.16, fc="#E8F5E9")
    ax.text(0.74, 0.64, "Intensity statistics", ha="center", fontsize=6.0, fontweight="bold",
            color="black", transform=ax.transAxes)
    ax.text(0.74, 0.56, "Mean · Std · Skew · Kurtosis", ha="center", fontsize=5.2,
            color="black", transform=ax.transAxes)
    box(ax, 0.52, 0.18, 0.44, 0.30, fc="#F3E5F5", ec="#7B1FA2")
    ax.text(0.74, 0.44, "AD-related structural degeneration", ha="center", fontsize=5.6,
            fontweight="bold", color="black", transform=ax.transAxes)
    ad = atlas_adkey(mc, sc)
    put_img(ax, ad, 0.55, 0.22, 0.11, 0.16)
    put_img(ax, ad, 0.68, 0.22, 0.11, 0.16)
    put_img(ax, ad, 0.81, 0.22, 0.11, 0.16)
    ax.text(0.605, 0.195, "Hippocampus", ha="center", fontsize=4.5, color="black", transform=ax.transAxes)
    ax.text(0.735, 0.195, "Amygdala", ha="center", fontsize=4.5, color="black", transform=ax.transAxes)
    ax.text(0.865, 0.195, "Lat. ventricle", ha="center", fontsize=4.5, color="black", transform=ax.transAxes)
    cb = ax.inset_axes([0.55, 0.155, 0.37, 0.025])
    cb.imshow(np.linspace(0, 1, 64)[None, :], aspect="auto", cmap="plasma")
    cb.set_yticks([])
    cb.set_xticks([0, 63])
    cb.set_xticklabels(["Low", "High"], fontsize=5, color="black")
    ax.text(
        0.27, 0.10,
        r"$x'_{ij}=(x_{ij}-\mu_j^{\mathrm{dev}})/\sigma_j^{\mathrm{dev}}$"
        r"   $m_{ij}=I(x_{ij}\ \mathrm{missing})$"
        "\n(fit on development set only)",
        ha="center", fontsize=5.4, color="black", transform=ax.transAxes,
    )


def draw_c(ax):
    panel(ax, "c", "Atlas-guided multimodal input")
    rng = np.random.default_rng(3)
    mat = rng.random((9, 11))
    soft = ax.inset_axes([0.05, 0.38, 0.42, 0.48])
    soft.imshow(mat, cmap="Blues", aspect="auto")
    soft.set_xticks([])
    soft.set_yticks([])
    soft.set_xlabel("Features", fontsize=5.5, color="black")
    soft.set_ylabel("Subjects", fontsize=5.5, color="black")
    soft.set_title("Atlas-derived imaging feature matrix", fontsize=5.8, color="black", pad=2)
    box(ax, 0.54, 0.38, 0.42, 0.50, fc="#FAFAFA")
    ax.text(0.75, 0.84, "Clinical variables", ha="center", fontsize=6.5, fontweight="bold",
            color="black", transform=ax.transAxes)
    for i, name in enumerate(["Age", "Sex", "Education", "APOE4", "MMSE", "CDR-SB"]):
        y = 0.76 - i * 0.06
        ax.add_patch(Circle((0.60, y), 0.012, transform=ax.transAxes, facecolor="#5C6BC0",
                            zorder=8, clip_on=False))
        ax.text(0.64, y, name, va="center", fontsize=6.0, color="black", transform=ax.transAxes)
    arrow(ax, 0.26, 0.36, 0.26, 0.28)
    arrow(ax, 0.75, 0.36, 0.50, 0.28)
    colors = plt.cm.turbo(np.linspace(0.15, 0.9, 22))
    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((0.10 + i * 0.036, 0.10), 0.034, 0.12, transform=ax.transAxes,
                               facecolor=c, edgecolor="white", lw=0.3, zorder=6, clip_on=False))
    ax.text(0.5, 0.05, "Fused feature vector (per subject)", ha="center", fontsize=6.2,
            fontweight="bold", color="black", transform=ax.transAxes)


def draw_d(ax):
    panel(ax, "d", "Six heterogeneous probability streams")
    colors = plt.cm.turbo(np.linspace(0.15, 0.9, 14))
    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((0.03 + i * 0.012, 0.55), 0.011, 0.28, transform=ax.transAxes,
                               facecolor=c, edgecolor="none", zorder=5, clip_on=False))
    ax.text(0.11, 0.48, "Fused feature\nvector", ha="center", fontsize=5.8, color="black",
            transform=ax.transAxes)
    for i in range(6):
        y = 0.78 - i * 0.10
        box(ax, 0.28, y, 0.28, 0.08, fc="#F5F5F5")
        ax.text(0.42, y + 0.04, f"Stream {i+1}", ha="center", va="center", fontsize=6.2,
                fontweight="bold", color="black", transform=ax.transAxes)
        arrow(ax, 0.20, 0.69, 0.28, y + 0.04)
        if i == 0:
            ax.text(0.68, 0.90, "CN", ha="center", fontsize=5.5, color="black", transform=ax.transAxes)
            ax.text(0.78, 0.90, "MCI", ha="center", fontsize=5.5, color="black", transform=ax.transAxes)
            ax.text(0.88, 0.90, "AD", ha="center", fontsize=5.5, color="black", transform=ax.transAxes)
        prob_bar(ax, 0.63, y + 0.01, w=0.30, h=0.06)
    ax.text(
        0.5, 0.08,
        r"$p_m(x_i)=[p_{m,\mathrm{CN}},\,p_{m,\mathrm{MCI}},\,p_{m,\mathrm{AD}}],\ m=1,\ldots,6$",
        ha="center", fontsize=7.0, color="black", transform=ax.transAxes,
    )


def draw_e(ax):
    panel(ax, "e", "RC-SPE risk-constrained fusion & calibration")
    for i in range(6):
        prob_bar(ax, 0.03, 0.78 - i * 0.07, w=0.14, h=0.05)
    arrow(ax, 0.18, 0.60, 0.24, 0.60)
    box(ax, 0.24, 0.42, 0.36, 0.40, fc="#E3F2FD", ec="#1565C0", lw=1.2)
    ax.text(0.42, 0.74, "RC-SPE fusion\n(risk-constrained weighted log-probability)",
            ha="center", va="top", fontsize=5.8, fontweight="bold", color="black", transform=ax.transAxes)
    ax.text(
        0.42, 0.58,
        r"$z_{i,k}=\frac{1}{T}\left[\sum_{m=1}^{6} w_m\log(\max(p_{m,k}(x_i),\varepsilon))+b_k\right]$"
        "\n"
        r"$w_m\geq 0,\ \sum_m w_m=1$",
        ha="center", va="center", fontsize=5.6, color="black", transform=ax.transAxes,
    )
    for i, (y, t, fc) in enumerate([
        (0.70, r"Class bias ($b_k$)", "#FFF3E0"),
        (0.54, r"Temperature scaling ($T$)", "#E8F5E9"),
        (0.38, "Softmax", "#F3E5F5"),
    ]):
        box(ax, 0.66, y, 0.30, 0.12, fc=fc)
        ax.text(0.81, y + 0.06, t, ha="center", va="center", fontsize=6.0, color="black",
                transform=ax.transAxes)
        if i == 0:
            arrow(ax, 0.60, 0.62, 0.66, 0.76)
        else:
            arrow(ax, 0.81, y + 0.14, 0.81, y + 0.12)
    ax.text(
        0.5, 0.22,
        r"$\hat{p}_{i,k}=\frac{\exp(z_{i,k})}{\sum_{c\in\{\mathrm{CN},\mathrm{MCI},\mathrm{AD}\}}\exp(z_{i,c})}$",
        ha="center", fontsize=6.8, color="black", transform=ax.transAxes,
    )
    ax.text(0.5, 0.12, "Calibrated scan-level probability", ha="center", fontsize=5.8,
            color=MUTED, transform=ax.transAxes)
    prob_bar(ax, 0.35, 0.04, w=0.30, h=0.06)


def draw_f(ax, img, seg):
    panel(ax, "f", "Repeated-scan subject aggregation")
    ax.text(0.18, 0.88, r"Subject $s$", ha="center", fontsize=7.0, fontweight="bold",
            color="black", transform=ax.transAxes)
    mc, sc = get_pair(img, seg, "coronal", 2, size=80)
    for i, lab in enumerate(["Scan 1", "Scan 2", r"Scan $n_s$"]):
        y = 0.68 - i * 0.18
        put_img(ax, mri_rgb(mc), 0.04, y, 0.10, 0.14)
        ax.text(0.09, y - 0.02, lab, ha="center", fontsize=5.2, color="black", transform=ax.transAxes)
        prob_bar(ax, 0.16, y + 0.04, w=0.22, h=0.06)
        arrow(ax, 0.40, y + 0.07, 0.48, 0.42)
    box(ax, 0.48, 0.34, 0.22, 0.16, fc="#E3F2FD", ec="#1565C0")
    ax.text(0.59, 0.42, "Mean probability\n(per subject)", ha="center", va="center", fontsize=5.8,
            fontweight="bold", color="black", transform=ax.transAxes)
    arrow(ax, 0.70, 0.42, 0.76, 0.42)
    ax.text(0.86, 0.52, "Subject-level\nprobability", ha="center", fontsize=5.8, fontweight="bold",
            color="black", transform=ax.transAxes)
    prob_bar(ax, 0.76, 0.36, w=0.20, h=0.08)
    ax.text(
        0.5, 0.18,
        r"$\tilde{p}_{s,k}=\frac{1}{n_s}\sum_i\hat{p}_{i,k}$"
        r"   $\hat{y}_s=\arg\max_k\tilde{p}_{s,k}$",
        ha="center", fontsize=6.4, color="black", transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.08,
        r"$\mathrm{Conf}_s=\max_k\tilde{p}_{s,k}$"
        r"   $\mathrm{Margin}_s=\tilde{p}_{s,(1)}-\tilde{p}_{s,(2)}$",
        ha="center", fontsize=6.4, color="black", transform=ax.transAxes,
    )


def draw_g(ax, img_cn, seg_cn, img_ad, seg_ad):
    panel(ax, "g", "Output & validation")
    ax.text(0.14, 0.82, "Predicted stage", ha="center", fontsize=6.0, fontweight="bold",
            color="black", transform=ax.transAxes)
    ax.text(0.05, 0.70, "CN", color="#0288D1", fontsize=8, fontweight="bold", transform=ax.transAxes)
    ax.text(0.12, 0.70, "MCI", color="#EF6C00", fontsize=8, fontweight="bold", transform=ax.transAxes)
    ax.text(0.21, 0.70, "AD", color="#C62828", fontsize=8, fontweight="bold", transform=ax.transAxes)
    # gauge
    ag = ax.inset_axes([0.30, 0.58, 0.28, 0.30])
    ag.set_xlim(-1.15, 1.15)
    ag.set_ylim(-0.15, 1.15)
    ag.axis("off")
    for i in range(36):
        t0, t1 = 180 * (1 - i / 36), 180 * (1 - (i + 1) / 36)
        ag.add_patch(Wedge((0, 0), 1.0, t1, t0, width=0.32, facecolor=plt.cm.turbo(i / 36)))
    ang = np.deg2rad(180 * (1 - 0.76))
    ag.plot([0, 0.85 * np.cos(ang)], [0, 0.85 * np.sin(ang)], "k-", lw=1.8)
    ag.text(0, -0.02, r"$\mathrm{Conf}_s$", ha="center", va="top", fontsize=6, color="black")
    ax.text(0.44, 0.88, "Confidence", ha="center", fontsize=5.8, fontweight="bold",
            color="black", transform=ax.transAxes)
    tri = np.array([[0.70, 0.62], [0.94, 0.62], [0.82, 0.88]])
    ax.add_patch(Polygon(tri, closed=True, fill=False, ec=EDGE, lw=1.0, transform=ax.transAxes))
    ax.plot(0.80, 0.72, "o", color="#EF6C00", ms=5, transform=ax.transAxes)
    ax.text(0.70, 0.58, "CN", fontsize=5, color="black", transform=ax.transAxes)
    ax.text(0.92, 0.58, "AD", fontsize=5, color="black", transform=ax.transAxes)
    ax.text(0.82, 0.90, "MCI", fontsize=5, ha="center", color="black", transform=ax.transAxes)
    ax.text(0.82, 0.52, "Three-class probability simplex", ha="center", fontsize=5.2,
            color="black", transform=ax.transAxes)

    box(ax, 0.04, 0.28, 0.18, 0.16, fc="#FFCDD2")
    ax.text(0.13, 0.36, "Locked\nexternal test", ha="center", va="center", fontsize=5.5,
            fontweight="bold", color="black", transform=ax.transAxes)
    box(ax, 0.26, 0.28, 0.28, 0.16, fc="#C8E6C9")
    ax.text(0.40, 0.36, "HC specificity\n" + r"$BAcc=\frac{1}{3}\sum_k Recall_k$",
            ha="center", va="center", fontsize=5.4, color="black", transform=ax.transAxes)
    cm = np.array([[148, 6, 0], [2, 24, 9], [0, 4, 23]], float)
    acm = ax.inset_axes([0.60, 0.24, 0.34, 0.26])
    acm.imshow(cm, cmap="Blues", aspect="auto")
    acm.set_xticks([0, 1, 2])
    acm.set_yticks([0, 1, 2])
    acm.set_xticklabels(["CN", "MCI", "AD"], fontsize=5, color="black")
    acm.set_yticklabels(["CN", "MCI", "AD"], fontsize=5, color="black")
    acm.tick_params(length=0)
    acm.set_title("Error transition analysis", fontsize=5.5, color="black", pad=2)
    for i in range(3):
        for j in range(3):
            acm.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=5.5,
                     color="white" if cm[i, j] > 40 else "black")

    mc, sc = get_pair(img_cn, seg_cn, "coronal", 2, 70)
    ma, sa = get_pair(img_ad, seg_ad, "coronal", 2, 70)
    put_img(ax, atlas_adkey(mc, sc), 0.04, 0.04, 0.12, 0.16)
    put_img(ax, atlas_adkey(ma, sa), 0.18, 0.04, 0.12, 0.16)
    ax.text(0.10, 0.01, "CN", ha="center", fontsize=5, color="black", transform=ax.transAxes)
    ax.text(0.24, 0.01, "AD", ha="center", fontsize=5, color="black", transform=ax.transAxes)
    ax.text(
        0.70, 0.10,
        "Atlas–neurodegeneration consistency\n"
        r"$d=\frac{\mu_{\mathrm{AD}}-\mu_{\mathrm{CN}}}{\sqrt{(\sigma_{\mathrm{AD}}^2+\sigma_{\mathrm{CN}}^2)/2}}$",
        ha="center", fontsize=5.6, color="black", transform=ax.transAxes,
    )


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    ALT.mkdir(parents=True, exist_ok=True)

    enh = contrast_enhance(SRC, OUT / "figure1_pipeline_contrast_enhanced.png")
    print("contrast-enhanced:", enh)

    print("loading FastSurfer…")
    cn = load_fastsurfer_native(SCAN_CN)
    ad = load_fastsurfer_native(SCAN_AD)
    if cn is None or ad is None:
        raise SystemExit("FastSurfer load failed")
    img_cn, seg_cn = cn
    img_ad, seg_ad = ad

    # Large canvas → sharp when zoomed (vector text + high-res embeds)
    fig = plt.figure(figsize=(18.0, 12.0), facecolor="white")
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.0, 1.05, 1.1],
                  hspace=0.10, wspace=0.12, left=0.02, right=0.985, top=0.94, bottom=0.03)
    draw_a(fig.add_subplot(gs[0, 0:2]))
    draw_b(fig.add_subplot(gs[0, 2:4]), img_cn, seg_cn)
    draw_c(fig.add_subplot(gs[0, 4:6]))
    draw_d(fig.add_subplot(gs[1, 0:3]))
    draw_e(fig.add_subplot(gs[1, 3:6]))
    draw_f(fig.add_subplot(gs[2, 0:3]), img_cn, seg_cn)
    draw_g(fig.add_subplot(gs[2, 3:6]), img_cn, seg_cn, img_ad, seg_ad)

    fig.text(0.5, 0.975, "ARA-Net pipeline (vector rebuild from contrast-enhanced reading)",
             ha="center", va="top", fontsize=12, fontweight="bold", color=EDGE)
    fig.text(0.5, 0.952, "White background · 16 pt panel letters · formulas as vector mathtext · real AIBL FastSurfer tiles",
             ha="center", va="top", fontsize=7.5, color=MUTED)
    force_times(fig)

    pdf = OUT / "figure1_pipeline_vector_hq.pdf"
    png = OUT / "figure1_pipeline_vector_hq.png"
    fig.savefig(pdf, dpi=400, facecolor="white")
    fig.savefig(png, dpi=250, facecolor="white")
    plt.close(fig)
    (ALT / pdf.name).write_bytes(pdf.read_bytes())
    (ALT / png.name).write_bytes(png.read_bytes())
    (ALT / enh.name).write_bytes(enh.read_bytes())
    print("wrote", pdf)
    print("wrote", png)


if __name__ == "__main__":
    main()
