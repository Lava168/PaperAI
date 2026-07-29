#!/usr/bin/env python3
"""Figure 3 — ARA-Net atlas structural directionality validation.

Panels
  a  Multi-slice 21-region atlas + laser outlines on key ROIs
  b  Multi-slice extended pathology (core + Accumbens/Cortex fills) + thin laser on pathology core
  c  Multi-region AD-like z trends (AD-key vs comparison parcels)
  d  Atlas AD-like z distributions (total N annotated)
  e  Cohen's d / regional effects (labels outside bars)

Data: locked AIBL heldout; demo anatomy from a heldout AD FastSurfer case.
Style: Times New Roman (Liberation Serif), tight white layout.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from skimage import measure
import nibabel as nib

from render_atlas_staging_triptych import (  # type: ignore
    atlas_rgb_slice,
    fit_panel_square,
    label_to_index_map,
    load_fastsurfer_native,
    normalize,
    plane_slice_indices,
    slice_plane,
)

OUT = ROOT / "reports" / "v6_final_model" / "figures"
ENRICHED = ROOT / "reports" / "v6_final_model" / "tables" / "final_subject_predictions_enriched.csv"
ROI_MATRIX = ROOT / "reports" / "brain_figures_fcstyle" / "assets" / "matrices" / "methods_roi_matrix.csv"
ROI_VECTORS = ROOT / "reports" / "brain_figures_fcstyle" / "assets" / "matrices" / "roi_vectors.json"

# New demo case (not AIBL_1013 used previously)
DEMO_SCAN_CANDIDATES = [
    "AIBL_1387_bl_I455228",
    "AIBL_1122_bl_I455196",
    "AIBL_567_bl_I161701",
    "AIBL_1100_bl_I133553",
    "AIBL_1504_bl_I444612",
    "AIBL_432_m18_I185470",
]

CLASSES = ["CN", "MCI", "AD"]
CLASS_COLORS = {"CN": "#0072B2", "MCI": "#E69F00", "AD": "#D55E00"}

SERIF_TTF = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
SERIF_BOLD = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf")

AD_KEY_LABELS = {
    17: "#FF1744",  # L Hipp — laser red
    53: "#FF1744",  # R Hipp
    18: "#FF9100",  # L Amyg — laser orange
    54: "#FF9100",  # R Amyg
    4: "#00E5FF",  # L Vent — laser cyan
    43: "#00E5FF",  # R Vent
}

# Extended pathology fills for panel b (from Fig03X panel c)
EXTENDED_FILL = {
    17: (0.92, 0.22, 0.18),  # Hipp
    53: (0.92, 0.22, 0.18),
    18: (1.00, 0.55, 0.12),  # Amyg
    54: (1.00, 0.55, 0.12),
    4: (0.25, 0.50, 0.95),   # Vent
    43: (0.25, 0.50, 0.95),
    26: (0.78, 0.20, 0.90),  # Accumbens (secondary fill)
    58: (0.78, 0.20, 0.90),
    3: (0.45, 0.85, 0.25),   # Cortex (secondary fill)
    42: (0.45, 0.85, 0.25),
}

N_SLICES = 8
PANEL_SIZE = 148


def apply_times_style() -> None:
    if SERIF_TTF.exists():
        font_manager.fontManager.addfont(str(SERIF_TTF))
        if SERIF_BOLD.exists():
            font_manager.fontManager.addfont(str(SERIF_BOLD))
        family = font_manager.FontProperties(fname=str(SERIF_TTF)).get_name()
    else:
        family = "Times New Roman"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", family, "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )


def force_times_on_figure(fig: plt.Figure) -> None:
    if not SERIF_TTF.exists():
        return
    fp = font_manager.FontProperties(fname=str(SERIF_TTF))
    fp_b = font_manager.FontProperties(fname=str(SERIF_BOLD)) if SERIF_BOLD.exists() else fp
    for artist in fig.findobj(match=lambda x: isinstance(x, matplotlib.text.Text)):
        try:
            wt = artist.get_fontweight()
            artist.set_fontproperties(fp_b if str(wt) in ("bold", "700") else fp)
        except Exception:
            artist.set_fontfamily("serif")


def clean_float(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def pick_demo_volume():
    for sid in DEMO_SCAN_CANDIDATES:
        native = load_fastsurfer_native(sid)
        if native is None:
            continue
        img, seg = native
        # Prefer cases with visible AD-key voxels
        key_n = int(np.isin(seg, list(AD_KEY_LABELS)).sum())
        if key_n > 5000:
            print(f"demo case: {sid} (AD-key voxels={key_n})")
            return img, seg, sid
    native = load_fastsurfer_native("AIBL_1013_m18_I190717")
    if native is not None:
        print("demo case fallback: AIBL_1013_m18_I190717")
        return native[0], native[1], "AIBL_1013_m18_I190717"
    atlas = nib.load(str(ROOT / "reports/brain_figures_fcstyle/assets/atlas_template.nii.gz")).get_fdata()
    under = nib.load(str(ROOT / "reports/brain_figures_fcstyle/assets/anat_underlay_mean_t1.nii.gz")).get_fdata()
    return under.astype(np.float32), atlas.astype(np.int32), "template"


def draw_laser_outlines(ax, seg_c: np.ndarray, labels: dict[int, str], lw: float = 0.55) -> None:
    """Thin neon contours on pathology ROIs only (Hipp / Amyg / Vent)."""
    for lab, color in labels.items():
        mask = seg_c == lab
        if mask.sum() < 8:
            continue
        for contour in measure.find_contours(mask.astype(float), 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color="white", lw=lw + 0.55, alpha=0.35, solid_capstyle="round")
            ax.plot(contour[:, 1], contour[:, 0], color=color, lw=lw, alpha=0.95, solid_capstyle="round")


def render_slice_pair(img, seg, plane: str, idx: int, *, mode: str, panel_size: int = PANEL_SIZE):
    """Return (mri gray on black, atlas/adkey RGB on black, seg_c)."""
    mri_sl = slice_plane(img, plane, idx)
    seg_sl = slice_plane(seg, plane, idx) if seg is not None else np.zeros_like(mri_sl)
    mri_c, seg_c = fit_panel_square(mri_sl, seg_sl, size=panel_size)
    g = normalize(mri_c)
    brain = g > 0.04

    if mode == "atlas":
        # Original black-background 21-region atlas (staging-pipeline style)
        rgb = atlas_rgb_slice(seg_c, label_to_index_map(), mri_sl=mri_c).copy()
    else:
        # Extended pathology map (Fig03X panel c style): core + Accumbens/Cortex fills
        rgb = np.zeros((*g.shape, 3), dtype=np.float32)
        tissue = 0.18 + 0.42 * g
        for c in range(3):
            rgb[..., c][brain] = tissue[brain]
        all_labs = list(EXTENDED_FILL)
        other = (seg_c > 0) & ~np.isin(seg_c, all_labs)
        rgb[other] = (0.32, 0.32, 0.35)
        for lab, col in EXTENDED_FILL.items():
            m = seg_c == lab
            if np.any(m):
                rgb[m] = col
        rgb[~brain] = 0.0
    return g, np.clip(rgb, 0, 1), seg_c


def _style_black_ax(ax) -> None:
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_mri_atlas_rows(
    ax_mri_row,
    ax_atlas_row,
    img,
    seg,
    *,
    mode: str,
    plane: str = "coronal",
    n: int = N_SLICES,
) -> None:
    """Top: black-bg original sMRI; bottom: black-bg atlas / AD-key + laser."""
    ids = plane_slice_indices(img.shape, plane, n=n, seg=seg)
    for ax_m, ax_a, idx in zip(ax_mri_row, ax_atlas_row, ids):
        _style_black_ax(ax_m)
        _style_black_ax(ax_a)
        g, rgb, seg_c = render_slice_pair(img, seg, plane, idx, mode=mode)
        ax_m.imshow(g, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
        ax_a.imshow(rgb, interpolation="nearest")
        # Pathology ROIs only (AD-key); thin laser
        draw_laser_outlines(ax_a, seg_c, AD_KEY_LABELS, lw=0.55 if mode == "adkey" else 0.45)
        ax_m.set_xlim(-0.5, g.shape[1] - 0.5)
        ax_m.set_ylim(g.shape[0] - 0.5, -0.5)
        ax_a.set_xlim(-0.5, rgb.shape[1] - 0.5)
        ax_a.set_ylim(rgb.shape[0] - 0.5, -0.5)


def load_aibl_heldout() -> list[dict]:
    rows = []
    with ENRICHED.open() as f:
        for row in csv.DictReader(f):
            if row.get("dataset") == "AIBL" and row.get("split") == "aibl_heldout":
                rows.append(row)
    return rows


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray([x for x in a if math.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if math.isfinite(x)], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(
        ((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2) / (len(a) + len(b) - 2)
    )
    if pooled < 1e-12:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def group_values(rows: list[dict], col: str) -> dict[str, np.ndarray]:
    out = {}
    for lab in CLASSES:
        vals = [clean_float(r[col]) for r in rows if r.get("y_true") == lab]
        out[lab] = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    return out


def load_multiregion_trends() -> tuple[dict[str, list[float]], list[str], list[str]]:
    """CN/MCI/AD mean atrophy z from roi_vectors; return series, adkey names, compare names."""
    blob = json.loads(ROI_VECTORS.read_text())
    regions = blob["regions"]
    cn = np.asarray(blob["vectors"]["atrophy_CN_mean"], float)
    mci = np.asarray(blob["vectors"]["atrophy_MCI_mean"], float)
    ad = np.asarray(blob["vectors"]["atrophy_AD_mean"], float)

    # Bilateral averages for cleaner lines
    pairs = {
        "Hippocampus": ("L-Hippocampus", "R-Hippocampus"),
        "Amygdala": ("L-Amygdala", "R-Amygdala"),
        "Lat. ventricle": ("L-Lat-Ventricle", "R-Lat-Ventricle"),
        "Cortex": ("L-Cortex", "R-Cortex"),
        "Thalamus": ("L-Thalamus", "R-Thalamus"),
        "WM": ("L-WM", "R-WM"),
        "Caudate": ("L-Caudate", "R-Caudate"),
        "Putamen": ("L-Putamen", "R-Putamen"),
        "Pallidum": ("L-Pallidum", "R-Pallidum"),
        "Accumbens": ("L-Accumbens", "R-Accumbens"),
        "Brain-Stem": ("Brain-Stem",),
    }
    name_to_idx = {r: i for i, r in enumerate(regions)}

    def avg(names: tuple[str, ...], vec: np.ndarray) -> float:
        idxs = [name_to_idx[n] for n in names if n in name_to_idx]
        return float(np.mean(vec[idxs])) if idxs else float("nan")

    series: dict[str, list[float]] = {}
    for name, regs in pairs.items():
        series[name] = [avg(regs, cn), avg(regs, mci), avg(regs, ad)]

    adkey = ["Hippocampus", "Amygdala", "Lat. ventricle"]
    compare = [n for n in series if n not in adkey]
    return series, adkey, compare


def load_regional_ad_cn_effects() -> list[tuple[str, float]]:
    if not ROI_MATRIX.exists():
        return []
    with ROI_MATRIX.open() as f:
        rows = {r["row"]: r for r in csv.DictReader(f)}
    ad = rows.get("atrophy_AD_minus_CN", {})
    keep = [
        ("L-Hippocampus", "L Hipp"),
        ("R-Hippocampus", "R Hipp"),
        ("L-Amygdala", "L Amyg"),
        ("R-Amygdala", "R Amyg"),
        ("L-Lat-Ventricle", "L Vent"),
        ("R-Lat-Ventricle", "R Vent"),
        ("L-Cortex", "L Cortex"),
        ("R-Cortex", "R Cortex"),
        ("L-Thalamus", "L Thal"),
        ("R-Thalamus", "R Thal"),
    ]
    out = []
    for key, lab in keep:
        try:
            out.append((lab, float(ad[key])))
        except Exception:
            continue
    return out


def draw_panels_ab(fig, gs_ab, img, seg, demo_id: str) -> None:
    outer = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_ab, height_ratios=[1, 1], hspace=0.18
    )

    def _one(panel_gs, mode: str, title: str, legend_handles):
        gs = GridSpecFromSubplotSpec(
            2, N_SLICES + 1, subplot_spec=panel_gs,
            width_ratios=[1] * N_SLICES + [1.15],
            height_ratios=[1, 1],
            wspace=0.03,
            hspace=0.04,
        )
        ax_mri = [fig.add_subplot(gs[0, j]) for j in range(N_SLICES)]
        ax_atl = [fig.add_subplot(gs[1, j]) for j in range(N_SLICES)]
        draw_mri_atlas_rows(ax_mri, ax_atl, img, seg, mode=mode, plane="coronal", n=N_SLICES)
        ax_mri[0].set_ylabel("Raw sMRI", fontsize=7.2, fontweight="bold", color="#333333")
        ax_atl[0].set_ylabel(
            "21-region atlas" if mode == "atlas" else "Extended pathology",
            fontsize=7.2, fontweight="bold", color="#333333",
        )
        for j, ax in enumerate(ax_mri):
            ax.set_title(f"c{j + 1}", fontsize=6.2, pad=1, color="#78909C")

        # Legend spans both rows on the right
        ax_leg = fig.add_subplot(gs[:, N_SLICES])
        ax_leg.set_facecolor("white")
        ax_leg.axis("off")
        ax_leg.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=2)
        ax_leg.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=False,
            fontsize=6.2,
            labelspacing=0.32,
            handlelength=1.1,
        )
        ax_leg.text(
            0.0, 0.02,
            f"case: {demo_id}\ncoronal ×{N_SLICES}\ntop = black-bg MRI\nbottom = atlas + laser",
            transform=ax_leg.transAxes, ha="left", va="bottom",
            fontsize=5.6, color="#90A4AE",
        )

    _one(
        outer[0],
        "atlas",
        "a  Atlas (21 regions)",
        [
            Patch(facecolor=(0.85, 0.40, 0.72), label="Cortex"),
            Patch(facecolor=(0.95, 0.93, 0.70), label="WM"),
            Patch(facecolor=(0.95, 0.18, 0.18), label="Hippocampus"),
            Patch(facecolor=(1.00, 0.55, 0.10), label="Amygdala"),
            Patch(facecolor=(0.20, 0.45, 0.95), label="Ventricle"),
            Patch(facecolor=(0.55, 0.55, 0.55), label="Other"),
            Patch(facecolor="none", edgecolor="#00E5FF", linewidth=0.8, label="Thin laser (pathology)"),
        ],
    )
    _one(
        outer[1],
        "adkey",
        "b  Extended pathology + laser",
        [
            Patch(facecolor=(0.92, 0.22, 0.18), label="Hippocampus (core)"),
            Patch(facecolor=(1.00, 0.55, 0.12), label="Amygdala (core)"),
            Patch(facecolor=(0.25, 0.50, 0.95), label="Lat. ventricle (core)"),
            Patch(facecolor=(0.78, 0.20, 0.90), label="Accumbens (2°, fill)"),
            Patch(facecolor=(0.45, 0.85, 0.25), label="Cortex (2°, fill)"),
            Patch(facecolor=(0.32, 0.32, 0.35), label="Other"),
            Patch(facecolor="none", edgecolor="#FF1744", linewidth=0.8, label="Thin laser Hipp/Amyg"),
            Patch(facecolor="none", edgecolor="#00E5FF", linewidth=0.8, label="Thin laser Ventricle"),
        ],
    )


def draw_panel_c(ax, series, adkey, compare) -> None:
    ax.set_facecolor("white")
    ax.set_title("c  Multi-region trends vs AD-key", loc="left", fontsize=9, fontweight="bold", pad=2)

    xs = np.arange(len(CLASSES))
    # Comparison parcels — thin grey
    for name in compare:
        ys = series[name]
        ax.plot(xs, ys, "-", color="#B0BEC5", lw=1.0, alpha=0.85, zorder=1, clip_on=True)
        ax.scatter(xs, ys, s=14, color="#90A4AE", zorder=2, linewidths=0, clip_on=True)

    # AD-key — bold colored
    styles = {
        "Hippocampus": ("#C62828", "o"),
        "Amygdala": ("#EF6C00", "s"),
        "Lat. ventricle": ("#1565C0", "^"),
    }
    for name in adkey:
        col, mk = styles[name]
        ys = series[name]
        ax.plot(xs, ys, "-", color=col, lw=2.2, zorder=4, label=name)
        ax.scatter(xs, ys, s=42, color=col, marker=mk, edgecolors="white", linewidths=0.7, zorder=5)

    # one grey handle for legend
    ax.plot([], [], "-", color="#B0BEC5", lw=1.2, label="Other atlas regions")

    ax.set_xticks(xs)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Mean atrophy / AD-like z")
    ax.axhline(0.0, color="#ECEFF1", lw=0.8)
    ax.set_xlim(-0.15, 2.15)
    ax.legend(loc="upper left", frameon=False, fontsize=6.2, ncol=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(
        0.98, 0.02,
        "bold = AD-key · grey = other 21-atlas parcels",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="#78909C",
    )


def draw_panel_d(ax, rows: list[dict]) -> None:
    ax.set_facecolor("white")
    ax.set_title("d  Atlas AD-like z by diagnosis", loc="left", fontsize=9, fontweight="bold", pad=2)

    by = group_values(rows, "atlas_ad_like_z")
    n_total = sum(len(by[lab]) for lab in CLASSES)
    data = [by[lab] for lab in CLASSES]
    parts = ax.violinplot(data, positions=np.arange(1, 4), showmeans=False, showmedians=False, showextrema=False)
    for body, lab in zip(parts["bodies"], CLASSES):
        body.set_facecolor(CLASS_COLORS[lab])
        body.set_alpha(0.32)
        body.set_edgecolor(CLASS_COLORS[lab])

    rng = np.random.default_rng(7)
    for i, lab in enumerate(CLASSES, start=1):
        vals = by[lab]
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            s=7, color=CLASS_COLORS[lab], alpha=0.4, zorder=2, linewidths=0,
        )
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([i - 0.12, i + 0.12], [med, med], color="#212121", lw=1.8, zorder=4)
        ax.vlines(i, q1, q3, color="#424242", lw=3.0, zorder=3)

    # One clear sample-size line (total + strata); no labels under violins
    ax.text(
        0.5, 1.04,
        f"Total N = {n_total}   |   CN n={len(by['CN'])} (μ={by['CN'].mean():.2f})   ·   "
        f"MCI n={len(by['MCI'])} (μ={by['MCI'].mean():.2f})   ·   "
        f"AD n={len(by['AD'])} (μ={by['AD'].mean():.2f})",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3, fontweight="bold", color="#37474F",
    )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("AD-like atlas z")
    ax.axhline(0.0, color="#ECEFF1", lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_panel_e(ax, rows: list[dict], regional: list[tuple[str, float]]) -> None:
    """Original dual horizontal-bar form; values live in side columns (never on bars)."""
    ax.set_facecolor("white")
    ax.set_title("e  Cohen’s d / regional effect sizes", loc="left", fontsize=9, fontweight="bold", pad=2)
    ax.axis("off")

    specs = [
        ("Hippocampus", "atlas_hippocampus_volume"),
        ("Amygdala", "atlas_amygdala_volume"),
        ("Cortex", "atlas_cortex_volume"),
        ("Lat. ventricle", "atlas_lateral_ventricle_volume"),
        ("AD-like z", "atlas_ad_like_z"),
    ]
    by_col = {col: group_values(rows, col) for _, col in specs}
    d_ad = [cohen_d(by_col[col]["AD"], by_col[col]["CN"]) for _, col in specs]
    d_mci = [cohen_d(by_col[col]["MCI"], by_col[col]["CN"]) for _, col in specs]
    labels = [name for name, _ in specs]
    y = np.arange(len(labels))

    # Layout with a clear gap so left values never meet right y-labels
    ax_l = ax.inset_axes([0.00, 0.10, 0.36, 0.82])
    ax_lv = ax.inset_axes([0.365, 0.10, 0.11, 0.82])
    ax_r = ax.inset_axes([0.58, 0.10, 0.28, 0.82])
    ax_rv = ax.inset_axes([0.87, 0.10, 0.12, 0.82])

    h = 0.32
    ax_l.barh(y - h / 2, d_mci, height=h, color="#90CAF9", edgecolor="white",
              linewidth=0.4, label="MCI−CN", zorder=2)
    ax_l.barh(y + h / 2, d_ad, height=h, color="#1565C0", edgecolor="white",
              linewidth=0.4, label="AD−CN", zorder=3)
    ax_l.axvline(0, color="#B0BEC5", lw=0.8, zorder=1)
    finite = [v for v in d_ad + d_mci if math.isfinite(v)]
    xmax = max(abs(min(finite)), abs(max(finite))) * 1.05 if finite else 2.0
    ax_l.set_xlim(-xmax, xmax)
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(labels, fontsize=6.5)
    ax_l.set_xlabel("Cohen’s d", fontsize=7)
    ax_l.invert_yaxis()
    ax_l.legend(loc="lower left", frameon=False, fontsize=5.8)
    for spine in ("top", "right"):
        ax_l.spines[spine].set_visible(False)
    ax_l.text(
        0.0, 1.02, "− atrophy   + enlargement",
        transform=ax_l.transAxes, ha="left", va="bottom", fontsize=5.4, color="#78909C",
    )

    # Values beside left bars (synced y; never drawn on bars)
    ax_lv.set_xlim(0, 1)
    ax_lv.set_ylim(ax_l.get_ylim())
    ax_lv.axis("off")
    ax_lv.text(0.5, 1.02, "AD / MCI", transform=ax_lv.transAxes,
               ha="center", va="bottom", fontsize=5.5, color="#546E7A")
    for yi, da, dm in zip(y, d_ad, d_mci):
        ax_lv.text(0.5, yi + h / 2, f"{da:+.2f}", ha="center", va="center",
                   fontsize=5.7, color="#0D47A1")
        ax_lv.text(0.5, yi - h / 2, f"{dm:+.2f}", ha="center", va="center",
                   fontsize=5.5, color="#5472D3")

    if regional:
        names = [n for n, _ in regional]
        vals = np.asarray([v for _, v in regional], dtype=float)
        yy = np.arange(len(names))
        colors = ["#C62828" if v >= 0 else "#546E7A" for v in vals]
        ax_r.barh(yy, vals, color=colors, edgecolor="white", linewidth=0.4, height=0.68)
        ax_r.axvline(0, color="#B0BEC5", lw=0.8)
        vmax = float(np.nanmax(np.abs(vals))) * 1.05 if len(vals) else 1.0
        ax_r.set_xlim(0, vmax)
        ax_r.set_yticks(yy)
        ax_r.set_yticklabels(names, fontsize=5.5)
        ax_r.tick_params(axis="y", pad=2)
        ax_r.set_xlabel("AD−CN atrophy contrast", fontsize=7)
        ax_r.invert_yaxis()
        for spine in ("top", "right"):
            ax_r.spines[spine].set_visible(False)

        ax_rv.set_xlim(0, 1)
        ax_rv.set_ylim(ax_r.get_ylim())
        ax_rv.axis("off")
        ax_rv.text(0.5, 1.02, "Δ", transform=ax_rv.transAxes,
                   ha="center", va="bottom", fontsize=5.5, color="#546E7A")
        for yi, v in zip(yy, vals):
            ax_rv.text(0.5, yi, f"{v:.2f}", ha="center", va="center",
                       fontsize=5.6, color="#37474F")
    else:
        ax_r.axis("off")
        ax_rv.axis("off")


def main() -> None:
    apply_times_style()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = load_aibl_heldout()
    if not rows:
        raise SystemExit(f"No AIBL heldout rows in {ENRICHED}")
    img, seg, demo_id = pick_demo_volume()
    series, adkey, compare = load_multiregion_trends()
    regional = load_regional_ad_cn_effects()

    fig = plt.figure(figsize=(13.2, 12.4), facecolor="white")
    gs = GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[1.85, 1.0],
        hspace=0.14,
        wspace=0.20,
        left=0.05,
        right=0.988,
        top=0.955,
        bottom=0.055,
    )

    draw_panels_ab(fig, gs[0, :], img, seg, demo_id)

    ax_c = fig.add_subplot(gs[1, 0])
    draw_panel_c(ax_c, series, adkey, compare)

    ax_d = fig.add_subplot(gs[1, 1])
    draw_panel_d(ax_d, rows)

    ax_e = fig.add_subplot(gs[1, 2])
    draw_panel_e(ax_e, rows, regional)

    fig.suptitle(
        "Figure 3. ARA-Net atlas structural directionality validation",
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5, 0.012,
        "AIBL locked heldout (Total N=216). Panel b = extended pathology map (Accumbens/Cortex fills) "
        "with thin laser on pathology core only (Hipp/Amyg/Vent); AD-key trends exceed other parcels.",
        ha="center", va="bottom", fontsize=6.5, color="#607D8B",
    )

    force_times_on_figure(fig)
    png = OUT / "figure3_atlas_structural_directionality.png"
    pdf = OUT / "figure3_atlas_structural_directionality.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="white", pad_inches=0.04)
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="white", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")

    alt = ROOT / "reports" / "brain_figures_fcstyle" / "figures"
    alt.mkdir(parents=True, exist_ok=True)
    (alt / "Fig03_atlas_structural_directionality.png").write_bytes(png.read_bytes())
    (alt / "Fig03_atlas_structural_directionality.pdf").write_bytes(pdf.read_bytes())
    print(f"copied to {alt}")


if __name__ == "__main__":
    main()
