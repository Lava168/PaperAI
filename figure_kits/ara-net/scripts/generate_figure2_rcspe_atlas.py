#!/usr/bin/env python3
"""Figure 2 — compact Times New Roman redesign.

Changes vs prior:
  - tighter panel spacing
  - panel b glass: 2×2 views, no overlapping nilearn text; legend separated
  - panel c: statistical chart only + empty placeholder box (no exterior arrows)
  - panel d: unchanged metric bars
  - panel e: slope-chart scan→subject form
  - whole figure: Times New Roman (Liberation Serif fallback)
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "deployment"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle, FancyArrowPatch
import nibabel as nib
from nilearn import plotting

from research_inference import (  # type: ignore
    aggregate_subjects,
    ensemble_scan_probabilities,
    read_csv_rows,
)
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
ABLATION = ROOT / "reports" / "v6_algorithm_innovation" / "algorithm_ablation_table.csv"
PROB_CSV = ROOT / "examples" / "probability_input_example.csv"
CONFIG = ROOT / "deployment" / "final_ensemble_config.json"

CLASSES = ["CN", "MCI", "AD"]
CLASS_COLORS = {"CN": "#0072B2", "MCI": "#E69F00", "AD": "#D55E00"}

STREAM_META = [
    ("aibl_adapted_atlas_biomarker_enhanced__hgb", "Atlas+bio", "HGB", "#2E6F9E"),
    ("aibl_adapted_atlas_core_clinical__hgb", "Atlas+clin", "HGB", "#2A9D8F"),
    ("aibl_adapted_clinical_biomarker_only__rf_balanced", "Clin+bio", "RF", "#E3A53B"),
    ("aibl_adapted_clinical_core_only__hgb", "Clin HGB", "HGB", "#7556A7"),
    ("aibl_adapted_clinical_core_only__rf_balanced", "Clin RF", "RF", "#B65C7A"),
    ("rf__logreg", "Cascade", "LR", "#8D6E63"),
]

SERIF_TTF = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
SERIF_BOLD = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf")


def apply_times_style() -> None:
    """Force Times New Roman metrics via Liberation Serif."""
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


def load_ablation_rows() -> dict[str, dict]:
    with ABLATION.open() as f:
        return {row["variant"]: row for row in csv.DictReader(f)}


def pick_demo_volume():
    native = load_fastsurfer_native("AIBL_1013_m18_I190717")
    if native is not None:
        return native
    atlas = nib.load(str(ROOT / "reports/brain_figures_fcstyle/assets/atlas_template.nii.gz")).get_fdata()
    under = nib.load(str(ROOT / "reports/brain_figures_fcstyle/assets/anat_underlay_mean_t1.nii.gz")).get_fdata()
    return under.astype(np.float32), atlas.astype(np.int32)


def as_nifti(vol: np.ndarray) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.asarray(vol, dtype=np.float32), np.eye(4))


def white_anat_slice(ax, img, seg, plane, *, mode="mri", panel_size=200):
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ids = plane_slice_indices(img.shape, plane, n=8, seg=seg)
    idx = ids[len(ids) // 2]
    mri_sl = slice_plane(img, plane, idx)
    seg_sl = slice_plane(seg, plane, idx) if seg is not None else np.zeros_like(mri_sl)
    mri_c, seg_c = fit_panel_square(mri_sl, seg_sl, size=panel_size)

    g = normalize(mri_c)
    brain = g > 0.04
    canvas = np.ones((*g.shape, 3), dtype=np.float32)
    tissue = 0.15 + 0.75 * g
    for c in range(3):
        canvas[..., c][brain] = 1.0 - tissue[brain] * 0.85

    if mode == "mri":
        ax.imshow(np.clip(canvas, 0, 1), interpolation="bilinear")
    elif mode == "atlas":
        rgb = atlas_rgb_slice(seg_c, label_to_index_map(), mri_sl=mri_c).copy()
        rgb[~brain] = 1.0
        ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
    else:  # adkey
        rgb = np.ones((*g.shape, 3), dtype=np.float32)
        tissue_g = 1.0 - (0.15 + 0.75 * g) * 0.55
        for c in range(3):
            rgb[..., c][brain] = tissue_g[brain]
        key = {
            4: (0.25, 0.50, 0.95), 43: (0.25, 0.50, 0.95),
            17: (0.92, 0.22, 0.18), 53: (0.92, 0.22, 0.18),
            18: (1.00, 0.55, 0.12), 54: (1.00, 0.55, 0.12),
        }
        other = (seg_c > 0) & ~np.isin(seg_c, list(key))
        rgb[other] = (0.82, 0.82, 0.84)
        for lab, col in key.items():
            m = seg_c == lab
            if np.any(m):
                rgb[m] = col
        rgb[~brain] = 1.0
        ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
    ax.set_title(plane.capitalize(), fontsize=7.5, pad=1)


def render_glass_quad(seg: np.ndarray):
    """Four separate nilearn glass views without annotate text (avoids overlap)."""
    key_labs = {4, 43, 17, 53, 18, 54}
    mask = np.isin(seg, list(key_labs)).astype(np.float32)
    step = max(1, mask.shape[0] // 64)
    small = mask[::step, ::step, ::step]
    nii = as_nifti(small)
    modes = [("l", "L lat."), ("y", "Coronal"), ("r", "R lat."), ("z", "Axial")]
    tiles = []
    with tempfile.TemporaryDirectory() as td:
        for i, (mode, _) in enumerate(modes):
            fig_g = plt.figure(figsize=(1.55, 1.45), facecolor="white")
            display = plotting.plot_glass_brain(
                nii,
                display_mode=mode,
                colorbar=False,
                black_bg=False,
                cmap="YlOrRd",
                threshold=0.1,
                alpha=0.9,
                annotate=False,
                figure=fig_g,
            )
            tmp = Path(td) / f"g{i}.png"
            fig_g.savefig(
                tmp, dpi=140, facecolor="white", edgecolor="white",
                bbox_inches="tight", pad_inches=0.02,
            )
            plt.close(fig_g)
            display.close()
            tiles.append(plt.imread(str(tmp))[..., :3])

    h = min(t.shape[0] for t in tiles)
    w = min(t.shape[1] for t in tiles)
    tiles = [t[:h, :w] for t in tiles]
    labels = [lab for _, lab in modes]
    # Burn captions into white strips under each tile (guaranteed no overlap with glass)
    strip_h = max(22, h // 8)
    labeled = []
    for tile, lab in zip(tiles, labels):
        strip = np.ones((strip_h, w, 3), dtype=np.float32)
        fig_t = plt.figure(figsize=(w / 100, strip_h / 100), dpi=100, facecolor="white")
        ax_t = fig_t.add_axes([0, 0, 1, 1])
        ax_t.axis("off")
        ax_t.set_xlim(0, 1)
        ax_t.set_ylim(0, 1)
        ax_t.text(0.5, 0.45, lab, ha="center", va="center", fontsize=7, color="#333333")
        with tempfile.NamedTemporaryFile(suffix=".png") as tmpf:
            fig_t.savefig(tmpf.name, dpi=100, facecolor="white")
            plt.close(fig_t)
            cap = plt.imread(tmpf.name)[..., :3]
        cap = np.array(
            [[cap[min(i, cap.shape[0] - 1), min(j, cap.shape[1] - 1)]
              for j in np.linspace(0, cap.shape[1] - 1, w).astype(int)]
             for i in np.linspace(0, cap.shape[0] - 1, strip_h).astype(int)],
            dtype=np.float32,
        )
        labeled.append(np.concatenate([tile, cap], axis=0))
    hh = labeled[0].shape[0]
    gutter = 5
    gap_h = np.ones((hh, gutter, 3), dtype=np.float32)
    gap_v = np.ones((gutter, 2 * w + gutter, 3), dtype=np.float32)
    top = np.concatenate([labeled[0], gap_h, labeled[1]], axis=1)
    bot = np.concatenate([labeled[2], gap_h, labeled[3]], axis=1)
    grid = np.concatenate([top, gap_v, bot], axis=0)
    return grid, labels


def draw_panels_ab(fig, gs_ab, img, seg):
    outer = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_ab, height_ratios=[1, 1], hspace=0.08)

    # a
    gs_a = GridSpecFromSubplotSpec(
        2, 4, subplot_spec=outer[0], width_ratios=[1, 1, 1, 0.95], hspace=0.06, wspace=0.04
    )
    planes = ("axial", "coronal", "sagittal")
    for j, plane in enumerate(planes):
        ax_m = fig.add_subplot(gs_a[0, j])
        ax_t = fig.add_subplot(gs_a[1, j])
        white_anat_slice(ax_m, img, seg, plane, mode="mri")
        white_anat_slice(ax_t, img, seg, plane, mode="atlas")
        if j == 0:
            ax_m.set_ylabel("Original sMRI", fontsize=7.5, fontweight="bold")
            ax_t.set_ylabel("21-region atlas", fontsize=7.5, fontweight="bold")
    ax_leg = fig.add_subplot(gs_a[:, 3])
    ax_leg.set_facecolor("white")
    ax_leg.axis("off")
    ax_leg.set_title("a  Atlas as model input", loc="left", fontsize=9, fontweight="bold", pad=2)
    ax_leg.legend(
        handles=[
            Patch(facecolor=(0.85, 0.40, 0.72), label="Cortex"),
            Patch(facecolor=(0.95, 0.93, 0.70), label="WM"),
            Patch(facecolor=(0.95, 0.18, 0.18), label="Hippocampus"),
            Patch(facecolor=(1.00, 0.55, 0.10), label="Amygdala"),
            Patch(facecolor=(0.20, 0.45, 0.95), label="Ventricle"),
            Patch(facecolor=(0.55, 0.55, 0.55), label="Other"),
        ],
        loc="upper left", frameon=False, fontsize=6.5, labelspacing=0.35,
    )

    # b
    gs_b = GridSpecFromSubplotSpec(
        2, 4, subplot_spec=outer[1], width_ratios=[1, 1, 1, 0.95], hspace=0.06, wspace=0.04
    )
    for j, plane in enumerate(planes):
        ax_m = fig.add_subplot(gs_b[0, j])
        ax_t = fig.add_subplot(gs_b[1, j])
        white_anat_slice(ax_m, img, seg, plane, mode="mri")
        white_anat_slice(ax_t, img, seg, plane, mode="adkey")
        if j == 0:
            ax_m.set_ylabel("Original sMRI", fontsize=7.5, fontweight="bold")
            ax_t.set_ylabel("AD-key prior", fontsize=7.5, fontweight="bold")

    # glass: image on top, legend below — no overlap
    gs_g = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_b[:, 3], height_ratios=[3.2, 1.0], hspace=0.08)
    ax_g = fig.add_subplot(gs_g[0])
    ax_g.set_facecolor("white")
    ax_g.axis("off")
    ax_g.set_title("b  AD-key glass", loc="left", fontsize=9, fontweight="bold", pad=2)
    try:
        grid, _labels = render_glass_quad(seg)
        ax_g.imshow(grid, aspect="equal")
    except Exception as exc:
        ax_g.text(0.5, 0.5, f"glass n/a\n{exc}", ha="center", va="center", fontsize=6)

    ax_bl = fig.add_subplot(gs_g[1])
    ax_bl.set_facecolor("white")
    ax_bl.axis("off")
    ax_bl.legend(
        handles=[
            Patch(facecolor=(0.92, 0.22, 0.18), label="Hippocampus"),
            Patch(facecolor=(1.00, 0.55, 0.12), label="Amygdala"),
            Patch(facecolor=(0.25, 0.50, 0.95), label="Lateral ventricle"),
            Patch(facecolor=(0.82, 0.82, 0.84), label="Other"),
        ],
        loc="upper left", frameon=False, fontsize=6.2, labelspacing=0.3, handlelength=1.0,
    )


def draw_panel_c(ax, config, demo_row):
    """Statistical view only: stacked bars + weight bubbles; empty slot for manual art."""
    ax.set_facecolor("white")
    ax.set_title("c  Six-stream class probabilities (weights encoded)", loc="left",
                 fontsize=9, fontweight="bold", pad=2)

    ax.axis("off")
    ax_stats = ax.inset_axes([0.0, 0.06, 0.60, 0.90])
    ax_empty = ax.inset_axes([0.64, 0.06, 0.36, 0.90])

    weights = np.asarray(config["weights"], dtype=float)
    names = [s[1] for s in STREAM_META]
    mat = np.zeros((6, 3), dtype=float)
    for i, (name, *_rest) in enumerate(STREAM_META):
        for j, cls in enumerate(CLASSES):
            mat[i, j] = float(demo_row[f"{name}__prob_{cls}"])

    y = np.arange(6)
    left = np.zeros(6)
    for j, cls in enumerate(CLASSES):
        ax_stats.barh(
            y, mat[:, j], left=left, height=0.58,
            color=CLASS_COLORS[cls], edgecolor="white", linewidth=0.4, label=cls,
        )
        left += mat[:, j]
    # Weight bubbles fully inside axes (no exterior strokes)
    for i, w in enumerate(weights):
        ax_stats.scatter(
            1.08, i, s=28 + 720 * w, c="#1565C0", alpha=0.88,
            edgecolors="white", linewidths=0.5, zorder=5, clip_on=True,
        )
        ax_stats.text(1.18, i, f"{w:.2f}", va="center", ha="left",
                      fontsize=6.0, color="#1565C0")

    ax_stats.set_yticks(y)
    ax_stats.set_yticklabels(names, fontsize=7)
    ax_stats.set_xlim(0, 1.38)
    ax_stats.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_stats.set_xlabel("Class probability  ·  bubble = stream weight", fontsize=7)
    ax_stats.invert_yaxis()
    ax_stats.legend(loc="lower right", ncol=3, frameon=False, fontsize=6.5)
    for spine in ("top", "right"):
        ax_stats.spines[spine].set_visible(False)
    ax_stats.spines["left"].set_visible(False)
    ax_stats.tick_params(axis="y", length=0)
    ax_stats.set_facecolor("white")
    ax_stats.axvline(1.0, color="#ECEFF1", lw=0.6, ls=":")

    # Reserved empty panel — dashed box only, no exterior connectors
    ax_empty.set_facecolor("white")
    ax_empty.set_xlim(0, 1)
    ax_empty.set_ylim(0, 1)
    ax_empty.set_xticks([])
    ax_empty.set_yticks([])
    for s in ax_empty.spines.values():
        s.set_visible(False)
    ax_empty.add_patch(
        FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#FAFBFC", edgecolor="#CFD8DC", linewidth=1.0, linestyle="--",
            transform=ax_empty.transAxes, clip_on=False,
        )
    )
    ax_empty.text(
        0.5, 0.5, "(reserved)",
        ha="center", va="center", fontsize=7.5, color="#B0BEC5", style="italic",
        transform=ax_empty.transAxes,
    )


def draw_panel_d(ax, ablation):
    """Unchanged metric comparison bars."""
    ax.set_facecolor("white")
    variants = [
        ("Best single base model", "Best single"),
        ("Arithmetic mean ensemble", "Arith. mean"),
        ("Equal log-pooling", "Equal log-pool"),
        ("Full RC-SPE (subject-level)", "Full RC-SPE"),
    ]
    metrics = [
        ("aibl_bacc", "AIBL BAcc"),
        ("aibl_recall_MCI", "MCI recall"),
        ("aibl_recall_AD", "AD recall"),
        ("ixi_cn_retention", "IXI CN ret."),
    ]
    colors = ["#B0BEC5", "#90A4AE", "#607D8B", "#1565C0"]
    x = np.arange(len(metrics))
    width = 0.18
    for i, (key, label) in enumerate(variants):
        vals = [float(ablation[key][m]) for m, _ in metrics]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label,
                      color=colors[i], edgecolor="white", linewidth=0.4)
        if key.startswith("Full RC-SPE"):
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=5.8, fontweight="bold", color="#1565C0")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.axhline(1.0, color="#eeeeee", lw=0.6, ls="--")
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=6.5)
    ax.set_title("d  Probability ensemble comparison", loc="left", fontweight="bold", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_panel_e(ax, scan_rows, scan_probs, subj_row):
    """Slope chart: Scan1 → Scan2 → Subject for CN/MCI/AD."""
    ax.set_facecolor("white")
    ax.set_title("e  Scan → subject probability slopes", loc="left", fontweight="bold", fontsize=9)

    stages = ["Scan 1", "Scan 2", "Subject"]
    xs = [0, 1, 2]
    series = {cls: [] for cls in CLASSES}
    for prob in scan_probs:
        for j, cls in enumerate(CLASSES):
            series[cls].append(float(prob[j]))
    for cls in CLASSES:
        series[cls].append(float(subj_row[f"prob_{cls}"]))

    for cls in CLASSES:
        ys = series[cls]
        ax.plot(xs, ys, "-", color=CLASS_COLORS[cls], lw=1.8, alpha=0.9, label=cls, zorder=2)
        ax.scatter(xs, ys, s=42, color=CLASS_COLORS[cls], edgecolors="white", linewidths=0.8, zorder=3)
        # endpoint label
        ax.text(2.08, ys[-1], f"{ys[-1]:.2f}", va="center", fontsize=6.5, color=CLASS_COLORS[cls])

    ax.set_xticks(xs)
    ax.set_xticklabels(stages)
    ax.set_xlim(-0.25, 2.45)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Probability")
    ax.axvline(1.5, color="#EEEEEE", lw=0.8, ls="--")
    ax.legend(loc="upper left", frameon=False, fontsize=6.5, ncol=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # compact decision chip
    pred = str(subj_row["predicted_label"])
    conf = float(subj_row["confidence"])
    margin = float(subj_row["margin"])
    ax.text(
        0.98, 0.08,
        f"pred={pred} · conf={conf:.3f} · margin={margin:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#E3F2FD", edgecolor="#1565C0", lw=0.7),
    )


def main() -> None:
    apply_times_style()
    OUT.mkdir(parents=True, exist_ok=True)

    config = json.loads(CONFIG.read_text())
    ablation = load_ablation_rows()
    img, seg = pick_demo_volume()
    rows = [r for r in read_csv_rows(PROB_CSV) if r["subject_id"] == "example_subject_002"]
    scan_probs = ensemble_scan_probabilities(rows, config)
    subj = aggregate_subjects(rows, scan_probs, config)[0]

    fig = plt.figure(figsize=(13.2, 10.4), facecolor="white")
    gs = GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[1.40, 0.92, 0.92],
        hspace=0.10,
        wspace=0.08,
        left=0.045,
        right=0.988,
        top=0.960,
        bottom=0.030,
    )

    draw_panels_ab(fig, gs[0, :], img, seg)

    ax_c = fig.add_subplot(gs[1, :])
    draw_panel_c(ax_c, config, rows[0])

    ax_d = fig.add_subplot(gs[2, 0])
    draw_panel_d(ax_d, ablation)

    ax_e = fig.add_subplot(gs[2, 1])
    draw_panel_e(ax_e, rows, scan_probs, subj)

    fig.suptitle(
        "Figure 2. RC-SPE probability ensemble and atlas-guided brain-region visualization",
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )

    force_times_on_figure(fig)
    png = OUT / "figure2_rcspe_atlas_ensemble.png"
    pdf = OUT / "figure2_rcspe_atlas_ensemble.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="white", pad_inches=0.04)
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="white", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")

    alt = ROOT / "reports/brain_figures_fcstyle/figures"
    alt.mkdir(parents=True, exist_ok=True)
    (alt / "Fig02_RC_SPE_atlas_ensemble.png").write_bytes(png.read_bytes())
    (alt / "Fig02_RC_SPE_atlas_ensemble.pdf").write_bytes(pdf.read_bytes())


if __name__ == "__main__":
    main()
