#!/usr/bin/env python3
"""Figure 4 — Atlas structural features across error groups (real AIBL examples).

Highlights AD-like pathology-sensitive parcels on real FastSurfer cases
(hippocampus, amygdala, lateral ventricles, plus Accumbens & Cortex),
colored by AD-like atlas z, and compares group-level structural profiles.

Narrative:
  AD_correct  → larger ventricles, higher AD-like z
  AD→MCI      → weaker structural abnormality (disease-boundary errors)
"""
from __future__ import annotations

import csv
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
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from render_atlas_staging_triptych import (  # type: ignore
    brain_bbox_2d,
    load_fastsurfer_native,
    normalize,
    plane_slice_indices,
    slice_plane,
)

OUT = ROOT / "reports" / "v6_final_model" / "figures"
ENRICHED = ROOT / "reports" / "v6_final_model" / "tables" / "final_subject_predictions_enriched.csv"
GROUP_CSV = ROOT / "reports" / "v6_final_model" / "tables" / "aibl_heldout_error_group_features.csv"

# Real locked AIBL heldout exemplars (FastSurfer-verified)
CASES = [
    {
        "group": "CN correct",
        "key": "CN_correct",
        "scan_id": "AIBL_14_bl_I135977",
        "subject_id": "AIBL_14",
        "note": "reference",
    },
    {
        "group": "AD correct",
        "key": "AD_correct",
        "scan_id": "AIBL_10_bl_I164086",
        "subject_id": "AIBL_10",
        "note": "high vent · high z",
    },
    {
        "group": "AD→MCI",
        "key": "AD_to_MCI",
        "scan_id": "AIBL_1368_bl_I455233",
        "subject_id": "AIBL_1368",
        "note": "weaker structure",
    },
    {
        "group": "MCI→AD",
        "key": "MCI_to_AD",
        "scan_id": "AIBL_1020_bl_I455196",
        "subject_id": "AIBL_1020",
        "note": "AD-like boundary",
    },
]

# Core AD-key + secondary pathology-sensitive parcels (ranked AD−CN contrast)
AD_PATH = {
    4, 43,       # Lat. ventricle L/R
    17, 53,      # Hippocampus L/R
    18, 54,      # Amygdala L/R
    26, 58,      # Accumbens L/R (secondary)
    3, 42,       # Cortex L/R (secondary)
}
AD_KEY = AD_PATH  # backward-compatible alias used below

# Colored outlines identify parcels (fill is shared AD-like z color)
PATH_OUTLINES = {
    (4, 43): np.array([0.25, 0.55, 1.00], dtype=np.float32),   # ventricle — blue
    (17, 53): np.array([1.00, 0.20, 0.15], dtype=np.float32),  # hippocampus — red
    (18, 54): np.array([1.00, 0.55, 0.05], dtype=np.float32),  # amygdala — orange
    (26, 58): np.array([0.85, 0.15, 0.95], dtype=np.float32),  # accumbens — magenta
    (3, 42): np.array([0.35, 0.90, 0.20], dtype=np.float32),   # cortex — lime
}
N_SLICES = 10  # denser coronal montage fills row width
PANEL = 120

SERIF_TTF = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
SERIF_BOLD = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf")


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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )


def force_times(fig: plt.Figure) -> None:
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


def load_subject_features() -> dict[str, dict]:
    out = {}
    with ENRICHED.open() as f:
        for row in csv.DictReader(f):
            if row.get("dataset") == "AIBL" and row.get("split") == "aibl_heldout":
                out[row["subject_id"]] = row
    return out


def load_group_means() -> dict[str, dict]:
    with GROUP_CSV.open() as f:
        return {r["group"]: r for r in csv.DictReader(f)}


ZMIN, ZMAX = -0.8, 2.3  # covers case z ≈ −0.70 … 2.29


def z_to_rgb(z: float, zmin: float = ZMIN, zmax: float = ZMAX) -> tuple[float, float, float]:
    """AD-like z → YlOrRd (same mapping for overlays, chips, and e2)."""
    t = (z - zmin) / (zmax - zmin + 1e-12)
    t = float(np.clip(t, 0, 1))
    return tuple(plt.cm.YlOrRd(0.12 + 0.88 * t)[:3])


def z_colormap():
    """Listed colormap matching z_to_rgb exactly (for colorbars)."""
    from matplotlib.colors import LinearSegmentedColormap

    samples = [z_to_rgb(ZMIN + (ZMAX - ZMIN) * t) for t in np.linspace(0, 1, 256)]
    return LinearSegmentedColormap.from_list("ad_like_z", samples, N=256)


def render_adkey_overlay(mri_c: np.ndarray, seg_c: np.ndarray, z: float) -> np.ndarray:
    """Pathology fill = pure AD-like z color; region ID via colored outlines."""
    from scipy import ndimage as ndi

    g = normalize(mri_c)
    brain = g > 0.04
    rgb = np.zeros((*g.shape, 3), dtype=np.float32)
    for c in range(3):
        rgb[..., c][brain] = 0.16 + 0.50 * g[brain]

    hot = np.asarray(z_to_rgb(z), dtype=np.float32)
    key_mask = np.zeros(seg_c.shape, dtype=bool)
    for lab in AD_PATH:
        key_mask |= seg_c == lab
    rgb[key_mask] = hot

    for labs, col in PATH_OUTLINES.items():
        m = np.zeros(seg_c.shape, dtype=bool)
        for lab in labs:
            m |= seg_c == lab
        if not np.any(m):
            continue
        edge = m & ~ndi.binary_erosion(m, iterations=1)
        rgb[edge] = col

    rgb[~brain] = 0.0
    return rgb


def _trim_lr(mri: np.ndarray, seg: np.ndarray, thr: float = 0.06, pad: int = 1):
    """Drop empty left/right columns so tissue fills the tile (less black between c1…c5)."""
    g = normalize(mri) if mri.max() > 1.5 else mri
    cols = np.where(np.max(g, axis=0) > thr)[0]
    if cols.size == 0:
        return mri, seg
    x0 = max(0, int(cols[0]) - pad)
    x1 = min(mri.shape[1], int(cols[-1]) + pad + 1)
    return mri[:, x0:x1], seg[:, x0:x1]


def _resize_square(arr: np.ndarray, size: int, order: int = 1) -> np.ndarray:
    from scipy.ndimage import zoom

    h, w = arr.shape[:2]
    factor = (size / max(h, 1), size / max(w, 1))
    if arr.ndim == 2:
        out = zoom(arr, factor, order=order)
        return out[:size, :size]
    out = np.stack([zoom(arr[..., c], factor, order=order) for c in range(arr.shape[2])], axis=-1)
    return out[:size, :size, ...]


def _tight_slice_pair(mri_sl: np.ndarray, seg_sl: np.ndarray, z: float, size: int):
    """Tight brain crop → trim empty L/R → square fill."""
    ys, xs = brain_bbox_2d(mri_sl, pad=1)
    mri_c = np.asarray(mri_sl[ys, xs], dtype=np.float32)
    seg_c = np.asarray(seg_sl[ys, xs], dtype=np.int32)
    mri_c, seg_c = _trim_lr(mri_c, seg_c, thr=0.07, pad=1)
    mri_c = _resize_square(mri_c, size, order=1)
    seg_c = _resize_square(seg_c.astype(np.float32), size, order=0).astype(np.int32)
    return normalize(mri_c), render_adkey_overlay(mri_c, seg_c, z)


def draw_case_row(fig, gs_row, case: dict, feat: dict, img, seg, letter: str) -> None:
    z = clean_float(feat.get("atlas_ad_like_z"))
    vent = clean_float(feat.get("atlas_lateral_ventricle_volume"))
    hipp = clean_float(feat.get("atlas_hippocampus_volume"))
    amyg = clean_float(feat.get("atlas_amygdala_volume"))
    ctx = clean_float(feat.get("atlas_cortex_volume"))
    y_true = feat.get("y_true", "")
    y_pred = feat.get("y_pred", "")

    # Left panel letter | slices | info
    gs = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_row,
        width_ratios=[0.32, N_SLICES, 1.40],
        wspace=0.015,
    )
    ax_let = fig.add_subplot(gs[0])
    ax_let.set_facecolor("white")
    ax_let.axis("off")
    ax_let.text(
        0.5, 0.55, letter,
        transform=ax_let.transAxes,
        ha="center", va="center",
        fontsize=16, fontweight="bold", color="black",
        fontfamily="serif",
    )

    gs_img = GridSpecFromSubplotSpec(
        2, N_SLICES, subplot_spec=gs[1],
        height_ratios=[1, 1], hspace=0.015, wspace=-0.06,
    )

    ids = plane_slice_indices(img.shape, "coronal", n=N_SLICES, seg=seg)
    for j, idx in enumerate(ids):
        mri_sl = slice_plane(img, "coronal", idx)
        seg_sl = slice_plane(seg, "coronal", idx)
        mri_p, ad_p = _tight_slice_pair(mri_sl, seg_sl, z, size=PANEL)

        ax_m = fig.add_subplot(gs_img[0, j])
        ax_a = fig.add_subplot(gs_img[1, j])
        for ax in (ax_m, ax_a):
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        ax_m.imshow(mri_p, cmap="gray", vmin=0, vmax=1, interpolation="bilinear", aspect="auto")
        ax_a.imshow(ad_p, interpolation="nearest", aspect="auto")
        if j == 0:
            ax_m.set_ylabel("sMRI", fontsize=7, color="#444", fontweight="bold")
            ax_a.set_ylabel("AD-like", fontsize=7, color="#444", fontweight="bold")
        ax_m.set_title(f"c{j+1}", fontsize=6, color="black", pad=1)

    ax_info = fig.add_subplot(gs[2])
    ax_info.set_facecolor("white")
    ax_info.axis("off")
    ax_info.set_title(case["group"], loc="left", fontsize=9, fontweight="bold", pad=1)
    lines = [
        f"{case['subject_id']}  ·  {y_true}→{y_pred}",
        f"AD-like z = {z:.2f}",
        f"vent = {vent:.4f} · hipp = {hipp:.4f}",
        f"amyg = {amyg:.4f} · cortex = {ctx:.4f}",
        f"({case['note']})",
    ]
    ax_info.text(0.02, 0.82, "\n".join(lines), transform=ax_info.transAxes,
                 ha="left", va="top", fontsize=6.3, linespacing=1.25, color="#37474F",
                 family="serif")
    # Colorbar shifted right; title above bar (not overlapping tick labels)
    ax_info.text(
        0.12, 0.22,
        f"AD-like z → fill color   ▼ {z:.2f}",
        transform=ax_info.transAxes, ha="left", va="bottom",
        fontsize=5.8, color="#455A64",
    )
    grad = np.linspace(ZMIN, ZMAX, 96, dtype=np.float32)
    grad_rgb = np.stack([z_to_rgb(float(v)) for v in grad], axis=0)[None, ...]
    ax_chip = ax_info.inset_axes([0.12, 0.04, 0.82, 0.12])
    ax_chip.imshow(grad_rgb, aspect="auto", extent=[ZMIN, ZMAX, 0, 1], origin="lower")
    ax_chip.axvline(z, color="white", lw=1.2)
    ax_chip.axvline(z, color="#212121", lw=0.6)
    ax_chip.set_xlim(ZMIN, ZMAX)
    ax_chip.set_yticks([])
    ax_chip.set_xticks([ZMIN, 0.0, ZMAX])
    ax_chip.set_xticklabels([f"{ZMIN:.1f}", "0", f"{ZMAX:.1f}"], fontsize=5)
    ax_chip.tick_params(axis="x", pad=1, length=2)
    for s in ax_chip.spines.values():
        s.set_linewidth(0.4)


def draw_group_panel(fig, gs_bottom, groups: dict) -> None:
    """Two heatmaps: (e1) AD-key volumes vs CN; (e2) z / confidence / margin.

    e2 keeps the original 3-column layout, but colors each column on its own
    scale (z: −0.6…2.3; confidence/margin: 0…1) so the colorbar is not mixed.
    """
    order = [
        ("CN_correct", "CN correct"),
        ("MCI_correct", "MCI correct"),
        ("MCI_to_AD", "MCI→AD"),
        ("AD_to_CN_MCI", "AD→MCI/CN"),
        ("AD_correct", "AD correct"),
    ]
    rows_keep = [(k, lab) for k, lab in order if k in groups]
    labels = [f"{lab}\n(n={int(float(groups[k]['n']))})" for k, lab in rows_keep]

    # --- Heatmap 1: volume % change vs CN correct ---
    vol_keys = [
        ("atlas_lateral_ventricle_volume_mean", "Lat. ventricle"),
        ("atlas_hippocampus_volume_mean", "Hippocampus"),
        ("atlas_amygdala_volume_mean", "Amygdala"),
        ("atlas_cortex_volume_mean", "Cortex"),
    ]
    cn = groups["CN_correct"]
    mat1 = np.zeros((len(rows_keep), len(vol_keys)), dtype=float)
    for i, (gk, _) in enumerate(rows_keep):
        for j, (vk, _) in enumerate(vol_keys):
            base = float(cn[vk])
            mat1[i, j] = (float(groups[gk][vk]) / base - 1.0) * 100.0

    # --- Heatmap 2: AD-like z + confidence/margin (column-wise color scales) ---
    feat_keys = [
        ("atlas_ad_like_z_mean", "AD-like z"),
        ("max_prob_mean", "Confidence"),
        ("margin_mean", "Margin"),
    ]
    mat2 = np.zeros((len(rows_keep), len(feat_keys)), dtype=float)
    for i, (gk, _) in enumerate(rows_keep):
        for j, (fk, _) in enumerate(feat_keys):
            mat2[i, j] = float(groups[gk][fk])

    z_cmap = z_colormap()
    rgb = np.zeros((mat2.shape[0], mat2.shape[1], 3), dtype=np.float32)
    # col 0: same z_to_rgb as brain fills; cols 1–2: probability on same ramp [0,1]
    for i in range(mat2.shape[0]):
        rgb[i, 0] = z_to_rgb(mat2[i, 0])
        for j in (1, 2):
            t = float(np.clip(mat2[i, j], 0, 1))
            rgb[i, j] = tuple(plt.cm.YlOrRd(0.12 + 0.88 * t)[:3])

    gs = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_bottom, wspace=0.22, width_ratios=[1.0, 1.0]
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # e1: diverging % volume change
    vlim = float(np.nanmax(np.abs(mat1)))
    im1 = ax1.imshow(mat1, cmap="RdBu_r", aspect="auto", vmin=-vlim, vmax=vlim)
    ax1.set_xticks(np.arange(len(vol_keys)))
    ax1.set_xticklabels([n for _, n in vol_keys], fontsize=7)
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_yticklabels(labels, fontsize=6.5)
    ax1.set_title("e1  Pathology volume vs CN correct (%)", loc="left", fontweight="bold", fontsize=9, pad=2)
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            ax1.text(j, i, f"{mat1[i, j]:+.0f}%", ha="center", va="center", fontsize=6.5,
                     color="white" if abs(mat1[i, j]) > 0.45 * vlim else "#212121")
    cax1 = ax1.inset_axes([1.03, 0.12, 0.035, 0.76])
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.ax.tick_params(labelsize=5.5)
    cb1.set_label("% vs CN", fontsize=6)

    # e2: same 3 columns; z column uses identical colors as AD-key fill
    ax2.imshow(rgb, aspect="auto")
    ax2.set_xticks(np.arange(len(feat_keys)))
    ax2.set_xticklabels([n for _, n in feat_keys], fontsize=7)
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_yticklabels([])
    ax2.set_title("e2  AD-like z / confidence / margin", loc="left", fontweight="bold", fontsize=9, pad=2)
    for i in range(mat2.shape[0]):
        for j in range(mat2.shape[1]):
            val = mat2[i, j]
            if j == 0:
                bright = (val - ZMIN) / (ZMAX - ZMIN) > 0.55
            else:
                bright = val > 0.55
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                     color="white" if bright else "#212121")
    cax_z = ax2.inset_axes([1.02, 0.55, 0.04, 0.38])
    sm_z = plt.cm.ScalarMappable(cmap=z_cmap, norm=Normalize(vmin=ZMIN, vmax=ZMAX))
    sm_z.set_array([])
    cb_z = fig.colorbar(sm_z, cax=cax_z)
    cb_z.ax.tick_params(labelsize=5)
    cb_z.set_label("z (col1)", fontsize=5.5)
    cax_p = ax2.inset_axes([1.02, 0.08, 0.04, 0.38])
    from matplotlib.colors import LinearSegmentedColormap
    p_samples = [tuple(plt.cm.YlOrRd(0.12 + 0.88 * t)[:3]) for t in np.linspace(0, 1, 256)]
    p_cmap = LinearSegmentedColormap.from_list("prob_ylorrd", p_samples, N=256)
    sm_p = plt.cm.ScalarMappable(cmap=p_cmap, norm=Normalize(vmin=0.0, vmax=1.0))
    sm_p.set_array([])
    cb_p = fig.colorbar(sm_p, cax=cax_p)
    cb_p.ax.tick_params(labelsize=5)
    cb_p.set_label("prob. (col2–3)", fontsize=5.5)

    fig.text(
        0.5, 0.018,
        "e1/e2: AD correct shows ↑ ventricle & ↑ AD-like z; AD→MCI/CN is weaker (RC-SPE boundary-like errors). "
        "Pathology fill and e2 col1 share the same z→color map.",
        ha="center", va="bottom", fontsize=6.1, color="#607D8B",
    )


def main() -> None:
    apply_times_style()
    OUT.mkdir(parents=True, exist_ok=True)
    feats = load_subject_features()
    groups = load_group_means()

    loaded = []
    for case in CASES:
        feat = feats.get(case["subject_id"])
        if feat is None:
            raise SystemExit(f"missing features for {case['subject_id']}")
        vol = load_fastsurfer_native(case["scan_id"])
        if vol is None:
            raise SystemExit(f"cannot load FastSurfer for {case['scan_id']}")
        loaded.append((case, feat, vol[0], vol[1]))
        print(f"loaded {case['group']}: {case['scan_id']} z={float(feat['atlas_ad_like_z']):.2f}")

    # Even outer margins (L/R ≈ T/B visual padding)
    fig = plt.figure(figsize=(13.4, 11.2), facecolor="white")
    gs = GridSpec(
        5, 1, figure=fig,
        height_ratios=[1.05, 1.05, 1.05, 1.05, 0.85],
        hspace=0.09,
        left=0.04, right=0.95, top=0.90, bottom=0.05,
    )

    letters = ["a", "b", "c", "d"]
    for i, ((case, feat, img, seg), letter) in enumerate(zip(loaded, letters)):
        outer = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[i])
        draw_case_row(fig, outer[0], case, feat, img, seg, letter=letter)

    draw_group_panel(fig, gs[4], groups)

    fig.suptitle(
        "Figure 4. ARA-Net atlas structural features of RC-SPE error groups (real AIBL heldout)",
        fontsize=12, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.955,
        "Pathology fill color = AD-like atlas z (same map as side/e2 colorbars); "
        "outlines mark Hipp / Amyg / Vent (core) + Accumbens / Cortex (secondary). "
        "AD-correct is warmer than AD→MCI boundary errors.",
        ha="center", va="top", fontsize=6.6, color="#546E7A",
    )

    fig.legend(
        handles=[
            Patch(facecolor=(1.0, 0.20, 0.15), label="Hippocampus"),
            Patch(facecolor=(1.0, 0.55, 0.05), label="Amygdala"),
            Patch(facecolor=(0.25, 0.55, 1.00), label="Ventricle"),
            Patch(facecolor=(0.85, 0.15, 0.95), label="Accumbens"),
            Patch(facecolor=(0.35, 0.90, 0.20), label="Cortex"),
            Patch(facecolor=z_to_rgb(ZMIN), label=f"Fill z = {ZMIN:.1f}"),
            Patch(facecolor=z_to_rgb(ZMAX), label=f"Fill z = {ZMAX:.1f}"),
        ],
        loc="upper center", ncol=7, frameon=False, fontsize=5.9,
        bbox_to_anchor=(0.5, 0.938),
    )

    force_times(fig)
    png = OUT / "figure4_error_group_atlas_structure.png"
    pdf = OUT / "figure4_error_group_atlas_structure.pdf"
    # Fixed canvas (no bbox_inches='tight') so L/R/T/B margins stay even
    fig.savefig(png, dpi=300, facecolor="white", pad_inches=0.0)
    fig.savefig(pdf, dpi=300, facecolor="white", pad_inches=0.0)
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")

    alt = ROOT / "reports" / "brain_figures_fcstyle" / "figures"
    alt.mkdir(parents=True, exist_ok=True)
    (alt / "Fig04_error_group_atlas_structure.png").write_bytes(png.read_bytes())
    (alt / "Fig04_error_group_atlas_structure.pdf").write_bytes(pdf.read_bytes())
    print(f"copied to {alt}")


if __name__ == "__main__":
    main()
