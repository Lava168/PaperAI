#!/usr/bin/env python3
"""Extra figure — extended pathology regions + laser-outline guide.

Companion to Fig03_atlas_structural_directionality.

Panels
  a  What laser outlines are (zoomed MTL; not landmark points)
  b  Extended pathology map: core AD-key + secondary ROIs
  c  Multi-slice coronal gallery with extended lasers
  d  Ranked AD−CN atrophy contrast (why these parcels)

Core AD-key (a priori): Hippocampus, Amygdala, Lat. ventricle (bilateral)
Secondary (data-driven, |Δ| next tier): Accumbens, Cortex (bilateral)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, Patch
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_figure3_atlas_directionality import (  # type: ignore
    DEMO_SCAN_CANDIDATES,
    PANEL_SIZE,
    apply_times_style,
    force_times_on_figure,
    pick_demo_volume,
)
from render_atlas_staging_triptych import (  # type: ignore
    fit_panel_square,
    normalize,
    plane_slice_indices,
    slice_plane,
)

OUT = ROOT / "reports" / "v6_final_model" / "figures"
ALT = ROOT / "reports" / "brain_figures_fcstyle" / "figures"
ROI_MATRIX = ROOT / "reports" / "brain_figures_fcstyle" / "assets" / "matrices" / "methods_roi_matrix.csv"

# Core AD-key (legacy a priori)
CORE = {
    17: "#FF1744",  # L Hipp
    53: "#FF1744",  # R Hipp
    18: "#FF9100",  # L Amyg
    54: "#FF9100",  # R Amyg
    4: "#00E5FF",  # L Vent
    43: "#00E5FF",  # R Vent
}
# Secondary pathology-sensitive (next AD−CN tier on 21-atlas)
SECONDARY = {
    26: "#D500F9",  # L Accumbens — magenta laser
    58: "#D500F9",  # R Accumbens
    3: "#76FF03",  # L Cortex — lime laser
    42: "#76FF03",  # R Cortex
}
# Weak limbic comparator (shown dimmer)
WEAK = {
    10: "#FFD600",  # L Thalamus — amber
    49: "#FFD600",  # R Thalamus
}

FILL = {
    17: (0.92, 0.22, 0.18),
    53: (0.92, 0.22, 0.18),
    18: (1.00, 0.55, 0.12),
    54: (1.00, 0.55, 0.12),
    4: (0.25, 0.50, 0.95),
    43: (0.25, 0.50, 0.95),
    26: (0.78, 0.20, 0.90),
    58: (0.78, 0.20, 0.90),
    3: (0.45, 0.85, 0.25),
    42: (0.45, 0.85, 0.25),
    10: (0.85, 0.75, 0.15),
    49: (0.85, 0.75, 0.15),
}

N_SLICES = 8
SERIF = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
SERIF_B = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf")


# Laser only on core pathology ROIs (not Cortex / other atlas parcels)
PATHOLOGY_LASER = CORE
LASER_LW = 0.55


def draw_laser(ax, seg_c: np.ndarray, labels: dict[int, str], lw: float = LASER_LW, dilate: int = 0) -> None:
    """Thin neon contours; pathology ROIs only."""
    for lab, color in labels.items():
        mask = seg_c == lab
        if mask.sum() < 6:
            continue
        try:
            from scipy import ndimage as ndi

            if dilate > 0:
                mask = ndi.binary_dilation(mask, iterations=dilate)
        except Exception:
            pass
        for contour in measure.find_contours(mask.astype(float), 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color="white", lw=lw + 0.55, alpha=0.35, solid_capstyle="round")
            ax.plot(contour[:, 1], contour[:, 0], color=color, lw=lw, alpha=0.95, solid_capstyle="round")


def render_extended(img, seg, plane: str, idx: int, *, panel_size: int = PANEL_SIZE, include_weak: bool = True):
    mri_sl = slice_plane(img, plane, idx)
    seg_sl = slice_plane(seg, plane, idx)
    mri_c, seg_c = fit_panel_square(mri_sl, seg_sl, size=panel_size)
    g = normalize(mri_c)
    brain = g > 0.04
    rgb = np.zeros((*g.shape, 3), dtype=np.float32)
    tissue = 0.18 + 0.42 * g
    for c in range(3):
        rgb[..., c][brain] = tissue[brain]

    all_labs = set(CORE) | set(SECONDARY) | (set(WEAK) if include_weak else set())
    other = (seg_c > 0) & ~np.isin(seg_c, list(all_labs))
    rgb[other] = (0.32, 0.32, 0.35)
    for lab, col in FILL.items():
        if lab not in all_labs:
            continue
        m = seg_c == lab
        if np.any(m):
            # Weak thalamus: slightly dimmer fill
            if lab in WEAK:
                rgb[m] = tuple(0.55 + 0.45 * x for x in col)
            else:
                rgb[m] = col
    rgb[~brain] = 0.0
    return g, np.clip(rgb, 0, 1), seg_c


def best_mtl_slice(img, seg, plane: str = "coronal") -> int:
    """Pick coronal index maximizing Hipp+Amyg voxels."""
    axis = {"sagittal": 0, "coronal": 1, "axial": 2}[plane]
    n = img.shape[axis]
    want = {17, 53, 18, 54}
    best_i, best_n = n // 2, -1
    for i in range(n):
        if axis == 1:
            sl = seg[:, i, :]
        elif axis == 0:
            sl = seg[i, :, :]
        else:
            sl = seg[:, :, i]
        cnt = int(np.isin(sl, list(want)).sum())
        if cnt > best_n:
            best_n, best_i = cnt, i
    return best_i


def load_ranked_effects() -> list[tuple[str, float, str]]:
    """Return (display_name, value, tier) sorted by |value|."""
    if not ROI_MATRIX.exists():
        return []
    with ROI_MATRIX.open() as f:
        rows = {r["row"]: r for r in csv.DictReader(f)}
    ad = rows.get("atrophy_AD_minus_CN", {})
    core_keys = {
        "L-Hippocampus", "R-Hippocampus", "L-Amygdala", "R-Amygdala",
        "L-Lat-Ventricle", "R-Lat-Ventricle",
    }
    sec_keys = {"L-Accumbens", "R-Accumbens", "L-Cortex", "R-Cortex"}
    weak_keys = {"L-Thalamus", "R-Thalamus"}
    short = {
        "L-Hippocampus": "L Hipp", "R-Hippocampus": "R Hipp",
        "L-Amygdala": "L Amyg", "R-Amygdala": "R Amyg",
        "L-Lat-Ventricle": "L Vent", "R-Lat-Ventricle": "R Vent",
        "L-Accumbens": "L Acc", "R-Accumbens": "R Acc",
        "L-Cortex": "L Cortex", "R-Cortex": "R Cortex",
        "L-Thalamus": "L Thal", "R-Thalamus": "R Thal",
        "L-WM": "L WM", "R-WM": "R WM",
        "L-Caudate": "L Caud", "R-Caudate": "R Caud",
        "L-Putamen": "L Put", "R-Putamen": "R Put",
        "L-Pallidum": "L Pall", "R-Pallidum": "R Pall",
        "Brain-Stem": "BrainStem",
    }
    out = []
    for k, v in ad.items():
        if k == "row" or k not in short:
            continue
        try:
            val = float(v)
        except Exception:
            continue
        if k in core_keys:
            tier = "core"
        elif k in sec_keys:
            tier = "secondary"
        elif k in weak_keys:
            tier = "weak"
        else:
            tier = "other"
        out.append((short[k], val, tier))
    out.sort(key=lambda t: -abs(t[1]))
    return out


def _black(ax):
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_panel_a(fig, gs, img, seg, demo_id: str) -> None:
    """Laser guide: real MRI + atlas/laser zoom + callouts."""
    ax = fig.add_subplot(gs)
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_title(
        "a  What the laser outlines mark",
        loc="left", fontsize=9.5, fontweight="bold", pad=4,
    )

    idx = best_mtl_slice(img, seg, "coronal")
    g, rgb, seg_c = render_extended(img, seg, "coronal", idx, panel_size=220, include_weak=False)

    # Real MRI (left) + atlas/laser (middle)
    ax_mri = ax.inset_axes([0.01, 0.10, 0.26, 0.78])
    _black(ax_mri)
    ax_mri.imshow(g, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    ax_mri.set_xlim(-0.5, g.shape[1] - 0.5)
    ax_mri.set_ylim(g.shape[0] - 0.5, -0.5)
    ax_mri.set_title("Raw sMRI", fontsize=6.8, color="#546E7A", pad=2)

    ax_im = ax.inset_axes([0.28, 0.10, 0.26, 0.78])
    _black(ax_im)
    ax_im.imshow(rgb, interpolation="nearest")
    draw_laser(ax_im, seg_c, PATHOLOGY_LASER)
    ax_im.set_xlim(-0.5, rgb.shape[1] - 0.5)
    ax_im.set_ylim(rgb.shape[0] - 0.5, -0.5)
    ax_im.set_title("Atlas + pathology laser", fontsize=6.8, color="#546E7A", pad=2)

    # explanation box
    box = FancyBboxPatch(
        (0.56, 0.08), 0.42, 0.82,
        transform=ax.transAxes, boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#FAFAFA", edgecolor="#CFD8DC", linewidth=0.9, clip_on=False,
    )
    ax.add_patch(box)
    lines = [
        ("Laser ≠ landmark points", True, "#C62828"),
        ("Neon contours = parcel boundaries", False, "#212121"),
        ("from FastSurfer aseg masks", False, "#455A64"),
        ("(find_contours on ROI voxels).", False, "#455A64"),
        ("", False, "#212121"),
        ("Left = real T1 sMRI;", False, "#212121"),
        ("right = fill shows extended", False, "#212121"),
        ("ROIs; thin laser only on", False, "#212121"),
        ("pathology core (Hipp/Amyg/Vent).", False, "#212121"),
        ("", False, "#212121"),
        ("Laser (pathology only):", True, "#1565C0"),
        ("  red   Hippocampus", False, "#FF1744"),
        ("  orange Amygdala", False, "#FF9100"),
        ("  cyan  Lat. ventricle", False, "#00BCD4"),
        ("Fill only (no laser):", True, "#6A1B9A"),
        ("  magenta Accumbens · lime Cortex", False, "#6A1B9A"),
    ]
    y = 0.86
    for text, bold, col in lines:
        ax.text(
            0.58, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=6.1 if not bold else 6.7,
            fontweight="bold" if bold else "normal", color=col,
        )
        y -= 0.048
    ax.text(
        0.01, 0.02, f"case {demo_id}  ·  y={idx}",
        transform=ax.transAxes, fontsize=5.6, color="#90A4AE",
    )


def draw_panel_b(fig, gs, img, seg) -> None:
    """Side-by-side: each column = raw MRI + atlas/laser (core vs extended)."""
    inner = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs, wspace=0.10, hspace=0.06, height_ratios=[1, 1]
    )
    idx = best_mtl_slice(img, seg, "coronal")
    g, rgb_full, seg_c = render_extended(img, seg, "coronal", idx, panel_size=200, include_weak=False)

    for j, (mode, title) in enumerate([
        ("core", "b1  Core AD-key (Hipp / Amyg / Vent)"),
        ("ext", "b2  Extended (+ Accumbens / Cortex)"),
    ]):
        ax_m = fig.add_subplot(inner[0, j])
        ax_a = fig.add_subplot(inner[1, j])
        _black(ax_m)
        _black(ax_a)
        ax_m.imshow(g, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
        rgb = rgb_full.copy()
        if mode == "core":
            for lab in SECONDARY:
                m = seg_c == lab
                if np.any(m):
                    rgb[m] = (0.32, 0.32, 0.35)
            ax_a.imshow(rgb, interpolation="nearest")
            draw_laser(ax_a, seg_c, PATHOLOGY_LASER)
        else:
            ax_a.imshow(rgb, interpolation="nearest")
            # Fill shows extended ROIs; laser stays on pathology core only
            draw_laser(ax_a, seg_c, PATHOLOGY_LASER)
        for ax_ in (ax_m, ax_a):
            ax_.set_xlim(-0.5, g.shape[1] - 0.5)
            ax_.set_ylim(g.shape[0] - 0.5, -0.5)
        ax_m.set_title(title, fontsize=7.2, fontweight="bold", color="#37474F", pad=2)
        if j == 0:
            ax_m.set_ylabel("Raw sMRI", fontsize=6.5, fontweight="bold", color="#455A64")
            ax_a.set_ylabel("Atlas + laser", fontsize=6.5, fontweight="bold", color="#455A64")


def draw_panel_c(fig, gs, img, seg) -> None:
    """Multi-slice: top raw sMRI, bottom extended atlas + laser."""
    outer = GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs, height_ratios=[0.14, 1.0, 1.0], hspace=0.04
    )
    ax_t = fig.add_subplot(outer[0])
    ax_t.axis("off")
    ax_t.set_title(
        "c  Multi-slice real sMRI (top) + extended pathology / laser (bottom)",
        loc="left", fontsize=9.5, fontweight="bold", pad=2,
    )
    ax_t.legend(
        handles=[
            Patch(facecolor=(0.92, 0.22, 0.18), label="Hipp (core)"),
            Patch(facecolor=(1.00, 0.55, 0.12), label="Amyg (core)"),
            Patch(facecolor=(0.25, 0.50, 0.95), label="Vent (core)"),
            Patch(facecolor=(0.78, 0.20, 0.90), label="Accumbens (2°, fill)"),
            Patch(facecolor=(0.45, 0.85, 0.25), label="Cortex (2°, fill)"),
            Patch(facecolor="none", edgecolor="#FF1744", linewidth=0.8, label="Thin laser = pathology only"),
        ],
        loc="center right", ncol=6, frameon=False, fontsize=5.6,
        handlelength=1.0, columnspacing=0.75,
    )

    row_m = GridSpecFromSubplotSpec(1, N_SLICES, subplot_spec=outer[1], wspace=0.03)
    row_a = GridSpecFromSubplotSpec(1, N_SLICES, subplot_spec=outer[2], wspace=0.03)
    ids = plane_slice_indices(img.shape, "coronal", n=N_SLICES, seg=seg)
    for j, idx in enumerate(ids):
        ax_m = fig.add_subplot(row_m[0, j])
        ax_a = fig.add_subplot(row_a[0, j])
        _black(ax_m)
        _black(ax_a)
        g, rgb, seg_c = render_extended(img, seg, "coronal", idx, panel_size=PANEL_SIZE, include_weak=False)
        ax_m.imshow(g, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
        ax_a.imshow(rgb, interpolation="nearest")
        draw_laser(ax_a, seg_c, PATHOLOGY_LASER)
        for ax_ in (ax_m, ax_a):
            ax_.set_xlim(-0.5, g.shape[1] - 0.5)
            ax_.set_ylim(g.shape[0] - 0.5, -0.5)
        ax_m.set_title(f"c{j + 1}", fontsize=6.0, color="#78909C", pad=1)
        if j == 0:
            ax_m.set_ylabel("Raw sMRI", fontsize=6.2, fontweight="bold", color="#455A64")
            ax_a.set_ylabel("Atlas+laser", fontsize=6.2, fontweight="bold", color="#455A64")


def draw_panel_d(ax, ranked: list[tuple[str, float, str]]) -> None:
    ax.set_facecolor("white")
    ax.set_title(
        "d  Why more regions: ranked AD−CN atrophy contrast",
        loc="left", fontsize=9.5, fontweight="bold", pad=3,
    )
    # show top 14 for readability
    rows = ranked[:14]
    names = [r[0] for r in rows][::-1]
    vals = np.asarray([r[1] for r in rows][::-1], float)
    tiers = [r[2] for r in rows][::-1]
    cmap = {
        "core": "#C62828",
        "secondary": "#6A1B9A",
        "weak": "#F9A825",
        "other": "#90A4AE",
    }
    colors = [cmap[t] for t in tiers]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, edgecolor="white", height=0.72, linewidth=0.4)
    ax.axvline(0, color="#B0BEC5", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.2)
    ax.set_xlabel("AD−CN atrophy contrast  Δ", fontsize=7.5)
    for yi, v in zip(y, vals):
        ax.text(v + (0.03 if v >= 0 else -0.03), yi, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=5.6, color="#37474F")
    xmax = float(np.nanmax(np.abs(vals))) * 1.28
    ax.set_xlim(-0.35, xmax)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=cmap["core"], label="Core AD-key"),
            Patch(facecolor=cmap["secondary"], label="Secondary (new)"),
            Patch(facecolor=cmap["weak"], label="Weak limbic"),
            Patch(facecolor=cmap["other"], label="Other atlas"),
        ],
        loc="lower right", frameon=False, fontsize=6.0,
    )
    ax.text(
        0.01, 1.02,
        "Accumbens & Cortex sit in the next |Δ| tier after Hipp/Amyg/Vent → included as extended pathology.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, color="#607D8B",
    )


def main() -> None:
    apply_times_style()
    OUT.mkdir(parents=True, exist_ok=True)
    ALT.mkdir(parents=True, exist_ok=True)

    img, seg, demo_id = pick_demo_volume()
    ranked = load_ranked_effects()

    fig = plt.figure(figsize=(13.0, 12.6), facecolor="white")
    gs = GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[1.15, 1.45, 1.05],
        hspace=0.26,
        wspace=0.18,
        left=0.05,
        right=0.985,
        top=0.945,
        bottom=0.048,
    )

    draw_panel_a(fig, gs[0, 0], img, seg, demo_id)
    draw_panel_b(fig, gs[0, 1], img, seg)
    draw_panel_c(fig, gs[1, :], img, seg)
    ax_d = fig.add_subplot(gs[2, :])
    draw_panel_d(ax_d, ranked)

    fig.suptitle(
        "Figure 3X. Extended pathology-sensitive atlas regions & laser-outline guide",
        fontsize=12, fontweight="bold", y=0.978,
    )
    fig.text(
        0.5, 0.012,
        "Real AIBL FastSurfer T1 alongside atlas. "
        "Thin laser outlines locate pathology core only (Hipp / Amyg / Lat. ventricle). "
        "Accumbens & Cortex shown as secondary fills (no laser).",
        ha="center", va="bottom", fontsize=6.4, color="#607D8B",
    )

    force_times_on_figure(fig)
    png = OUT / "figure3_extended_pathology_laser_guide.png"
    pdf = OUT / "figure3_extended_pathology_laser_guide.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="white", pad_inches=0.04)
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="white", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")

    (ALT / "Fig03_extended_pathology_laser_guide.png").write_bytes(png.read_bytes())
    (ALT / "Fig03_extended_pathology_laser_guide.pdf").write_bytes(pdf.read_bytes())
    print(f"copied to {ALT}")


if __name__ == "__main__":
    main()
