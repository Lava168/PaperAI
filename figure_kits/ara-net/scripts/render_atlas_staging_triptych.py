#!/usr/bin/env python3
"""ARA-Net atlas staging triptych figure.

Layout (one subject):
  Row 1 — raw structural MRI (axial / coronal / sagittal)
  Row 2 — atlas-mapped regionalized representation (21 FreeSurfer-lite parcels;
           AD-key regions emphasized)
  Row 3 — subject-level CN / MCI / AD prediction probabilities (RC-SPE)

Also writes a 3-subject staging board (correct CN / MCI / AD exemplars).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]  # .../ARA-Net
CHAPTER = ROOT.parent  # .../chapter1_foundation
WORKSPACE = CHAPTER.parent  # workspace root

FS_LABELS = [
    2, 3, 4, 10, 11, 12, 13, 16, 17, 18, 26,
    41, 42, 43, 49, 50, 51, 52, 53, 54, 58,
]
REGION_NAMES = [
    "L-WM", "L-Ctx", "L-Vent", "L-Thal", "L-Caud", "L-Put", "L-Pall",
    "BrainStem", "L-Hipp", "L-Amyg", "L-Acc",
    "R-WM", "R-Ctx", "R-Vent", "R-Thal", "R-Caud", "R-Put", "R-Pall",
    "R-Hipp", "R-Amyg", "R-Acc",
]
AD_KEY = {4, 43, 17, 53, 18, 54}  # Vent, Hipp, Amyg bilat

# Full 21-region categorical palette (saturated; AD-key fixed below)
_BASE_RGB = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
_EXTRA = np.array([[0.40, 0.65, 0.45]])
CAT_RGB = np.vstack([_BASE_RGB, _EXTRA])  # 21

OKABE = {"CN": "#0072B2", "MCI": "#E69F00", "AD": "#D55E00"}
PLANES = ("axial", "coronal", "sagittal")

# Collapse full FreeSurfer / DKT aparc+aseg → ARA-Net 21 FS labels (L/R kept).
APARC_TO_FS = {
    # ventricles (incl. inf / 3rd / 4th)
    4: 4, 5: 4, 14: 4, 15: 4, 31: 4,
    43: 43, 44: 43, 63: 43,
    # WM (+ hypointensities / non-WM-hypointensities)
    2: 2, 41: 41, 77: 2, 78: 41, 79: 2,
    # classic aseg cortex (if present)
    3: 3, 42: 42,
    # subcortical
    10: 10, 49: 49, 11: 11, 50: 50, 12: 12, 51: 51,
    13: 13, 52: 52, 16: 16, 17: 17, 53: 53, 18: 18, 54: 54,
    26: 26, 58: 58,
    # ventral DC → thalamus (nearest ARA-Net token)
    28: 10, 60: 49,
    # CSF / vessel-ish → ventricle for display continuity
    24: 4, 30: 4, 62: 43,
    # cerebellum WM/cortex kept via display id below (7/8/46/47)
}


def remap_aparc_to_21(seg: np.ndarray) -> np.ndarray:
    """Map aparc.DKT+aseg (or aseg) labels onto the 21 FreeSurfer-lite IDs."""
    seg = np.asarray(seg, dtype=np.int32)
    out = np.zeros(seg.shape, dtype=np.int32)
    for src, dst in APARC_TO_FS.items():
        out[seg == src] = dst
    # DKT cortical parcels
    out[(seg >= 1000) & (seg < 2000)] = 3   # L-Ctx
    out[(seg >= 2000) & (seg < 3000)] = 42  # R-Ctx
    return out


def resize_vol(vol: np.ndarray, target_shape: tuple[int, ...], order: int) -> np.ndarray:
    from scipy.ndimage import zoom

    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    return zoom(vol, factors, order=order)


def aibl_fastsurfer_mri_dir(scan_id: str) -> Path | None:
    parts = scan_id.split("_")
    if len(parts) < 3 or parts[0] != "AIBL":
        return None
    fs_id = f"AIBL_{parts[1]}_{parts[-1]}"
    mri_dir = WORKSPACE / "data/external/aibl/fastsurfer_seg/fs_subjects" / fs_id / "mri"
    return mri_dir if mri_dir.is_dir() else None


def complete_labels_in_brain(
    seg: np.ndarray,
    brain_mask: np.ndarray,
    max_dist: float = 6.0,
) -> np.ndarray:
    """Nearest-label fill inside brain mask (display-only gap completion)."""
    from scipy import ndimage as ndi

    out = seg.copy()
    labeled = out > 0
    need = brain_mask & ~labeled
    if not np.any(need) or not np.any(labeled):
        return out
    # Distance to nearest labeled voxel + indices of that voxel
    dist, (iz, iy, ix) = ndi.distance_transform_edt(~labeled, return_indices=True)
    fill = need & (dist <= max_dist)
    out[fill] = out[iz[fill], iy[fill], ix[fill]]
    return out


def load_fastsurfer_native(scan_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Native-resolution skull-stripped MRI + complete 21-region atlas."""
    mri_dir = aibl_fastsurfer_mri_dir(scan_id)
    if mri_dir is None:
        return None
    try:
        import nibabel as nib
    except ImportError:
        return None
    aparc = mri_dir / "aparc.DKTatlas+aseg.deep.mgz"
    aseg = mri_dir / "aseg.auto_noCCseg.mgz"
    seg_path = aparc if aparc.exists() else aseg
    if not seg_path.exists():
        return None
    brain_path = None
    for fname in ("orig_nu.mgz", "orig.mgz"):
        if (mri_dir / fname).exists():
            brain_path = mri_dir / fname
            break
    if brain_path is None:
        return None
    img = nib.load(str(brain_path)).get_fdata().astype(np.float32)
    raw = nib.load(str(seg_path)).get_fdata()
    seg21 = remap_aparc_to_21(raw)
    # Display extras (not in 21 tokens): cerebellum → synthetic IDs for coloring
    cereb = np.isin(raw.astype(np.int32), [7, 8, 46, 47])
    seg21 = seg21.copy()
    seg21[cereb & (seg21 == 0)] = 7  # draw as cerebellum green

    # Brain mask from FastSurfer mask.mgz or intensity
    mask_path = mri_dir / "mask.mgz"
    if mask_path.exists():
        brain = nib.load(str(mask_path)).get_fdata() > 0
    else:
        thr = np.percentile(img[img > 0], 15) if np.any(img > 0) else 0
        brain = img > thr
    # Skull-strip MRI so row-1/row-2 share the same brain extent
    img = img * brain.astype(np.float32)
    # Fill all brain-mask gaps (display completeness)
    seg21 = complete_labels_in_brain(seg21, brain, max_dist=20.0)
    # Zero atlas outside brain
    seg21 = seg21 * brain.astype(np.int32)
    return img, seg21


def load_fastsurfer_seg21(scan_id: str, target_shape: tuple[int, ...]) -> np.ndarray | None:
    """Load dense FastSurfer aparc+aseg remapped to 21 regions at NPZ shape."""
    native = load_fastsurfer_native(scan_id)
    if native is None:
        return None
    _, seg21 = native
    return resize_vol(seg21.astype(np.float32), target_shape, order=0).astype(np.int32)


def _orient(sl: np.ndarray) -> np.ndarray:
    return np.flipud(np.rot90(sl))


def slice_plane(vol: np.ndarray, plane: str, idx: int) -> np.ndarray:
    if plane == "axial":
        return _orient(vol[:, :, idx])
    if plane == "coronal":
        return _orient(vol[:, idx, :])
    if plane == "sagittal":
        return _orient(vol[idx, :, :])
    raise ValueError(plane)


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    m = arr > 0
    if not np.any(m):
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[m], [1, 99])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


def brain_bbox_2d(mri_sl: np.ndarray, pad: int = 4) -> tuple[slice, slice]:
    """Tight crop around brain tissue for denser atlas display."""
    g = normalize(mri_sl)
    mask = g > 0.08
    if not np.any(mask):
        return slice(None), slice(None)
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(mri_sl.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mri_sl.shape[1], xs.max() + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def mid_indices(shape: tuple[int, int, int]) -> dict[str, int]:
    x, y, z = shape
    # Slight offset toward MTL for axial/coronal on 96×112×96 crop
    return {
        "axial": int(round(z * 0.48)),
        "coronal": int(round(y * 0.42)),
        "sagittal": int(round(x * 0.50)),
    }


def plane_slice_indices(
    shape: tuple[int, int, int],
    plane: str,
    n: int = 8,
    seg: np.ndarray | None = None,
) -> list[int]:
    """Evenly spaced indices along plane axis on the densest atlas band."""
    axis = {"sagittal": 0, "coronal": 1, "axial": 2}[plane]
    n_axis = shape[axis]

    def _count(i: int) -> int:
        if seg is None:
            return 1
        if axis == 0:
            return int((seg[i, :, :] > 0).sum())
        if axis == 1:
            return int((seg[:, i, :] > 0).sum())
        return int((seg[:, :, i] > 0).sum())

    if seg is not None:
        counts = np.array([_count(i) for i in range(n_axis)], dtype=np.int64)
        peak = int(counts.max()) if counts.size else 0
        if peak > 0:
            good = np.where(counts >= max(1, int(0.55 * peak)))[0]
            if good.size < n:
                good = np.where(counts >= max(1, int(0.40 * peak)))[0]
            if good.size >= 2:
                lo, hi = int(good.min()), int(good.max())
                return [int(round(v)) for v in np.linspace(lo, hi, n)]
    lo = max(8, int(round(n_axis * 0.28)))
    hi = min(n_axis - 9, int(round(n_axis * 0.72)))
    if hi <= lo:
        lo, hi = 10, max(11, n_axis - 11)
    return [int(round(v)) for v in np.linspace(lo, hi, n)]


def coronal_slice_indices(ny: int, n: int = 8, seg: np.ndarray | None = None) -> list[int]:
    """Backward-compatible wrapper."""
    shape = (0, ny, 0) if seg is None else seg.shape
    return plane_slice_indices(shape, "coronal", n=n, seg=seg)


def fit_panel_square(
    mri_sl: np.ndarray,
    seg_sl: np.ndarray,
    size: int = 256,
    pad: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Brain-crop then letterbox-pad/resize so every panel is size×size."""
    from scipy.ndimage import zoom

    ys, xs = brain_bbox_2d(mri_sl, pad=pad)
    mri_c = np.asarray(mri_sl[ys, xs], dtype=np.float32)
    seg_c = np.asarray(seg_sl[ys, xs], dtype=np.int32)
    h, w = mri_c.shape
    side = max(h, w, 1)
    # Pad to square with zeros
    out_m = np.zeros((side, side), dtype=np.float32)
    out_s = np.zeros((side, side), dtype=np.int32)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    out_m[y0 : y0 + h, x0 : x0 + w] = mri_c
    out_s[y0 : y0 + h, x0 : x0 + w] = seg_c
    if side != size:
        factor = size / side
        out_m = zoom(out_m, factor, order=1)
        out_s = zoom(out_s.astype(np.float32), factor, order=0).astype(np.int32)
        # zoom may be off-by-one
        out_m = out_m[:size, :size]
        out_s = out_s[:size, :size]
        if out_m.shape != (size, size):
            tmp_m = np.zeros((size, size), dtype=np.float32)
            tmp_s = np.zeros((size, size), dtype=np.int32)
            hh, ww = out_m.shape
            tmp_m[:hh, :ww] = out_m
            tmp_s[:hh, :ww] = out_s
            out_m, out_s = tmp_m, tmp_s
    return out_m, out_s


def axis_tag(plane: str, idx: int) -> str:
    return {"axial": f"z={idx}", "coronal": f"y={idx}", "sagittal": f"x={idx}"}[plane]


def label_to_index_map() -> dict[int, int]:
    return {lab: i for i, lab in enumerate(FS_LABELS)}


def atlas_rgb_slice(
    seg_sl: np.ndarray,
    lab2i: dict[int, int],
    mri_sl: np.ndarray | None = None,
) -> np.ndarray:
    """Full categorical RGB on black; MRI underlay only in unlabeled voxels."""
    h, w = seg_sl.shape
    out = np.zeros((h, w, 3), dtype=np.float32)

    key_colors = {
        4: np.array([0.20, 0.45, 0.95]),
        43: np.array([0.20, 0.45, 0.95]),
        17: np.array([0.95, 0.18, 0.18]),
        53: np.array([0.95, 0.18, 0.18]),
        18: np.array([1.00, 0.55, 0.10]),
        54: np.array([1.00, 0.55, 0.10]),
    }
    # Dedicated colors for large tissue classes (readability)
    tissue_colors = {
        2: np.array([0.95, 0.93, 0.70]),   # L-WM
        41: np.array([0.90, 0.88, 0.62]),  # R-WM
        3: np.array([0.85, 0.40, 0.72]),   # L-Ctx
        42: np.array([0.35, 0.65, 0.95]),  # R-Ctx
        16: np.array([0.60, 0.35, 0.60]),  # Brain-Stem
        7: np.array([0.30, 0.70, 0.38]),   # Cerebellum (display-only)
    }

    def _brighten(col: np.ndarray, floor: float = 0.28) -> np.ndarray:
        col = np.asarray(col, dtype=np.float32)
        return np.clip(np.maximum(col, floor) * 0.55 + col * 0.45, 0, 1)

    for lab, i in lab2i.items():
        m = seg_sl == lab
        if not np.any(m):
            continue
        if lab in key_colors:
            col = key_colors[lab]
        elif lab in tissue_colors:
            col = tissue_colors[lab]
        else:
            col = _brighten(CAT_RGB[i % len(CAT_RGB)])
        out[m] = col
    m = seg_sl == 7
    if np.any(m):
        out[m] = tissue_colors[7]
    return np.clip(out, 0, 1)


def resolve_npz(scan_id: str, cache_roots: list[Path]) -> Path:
    for root in cache_roots:
        p = root / f"{scan_id}.npz"
        if p.exists():
            return p
    raise FileNotFoundError(f"NPZ not found for {scan_id} in {cache_roots}")


def load_case(scan_id: str, cache_roots: list[Path], prefer_native: bool = True):
    """Load MRI+atlas. Prefer FastSurfer native res for complete atlas display."""
    path = resolve_npz(scan_id, cache_roots)
    native = load_fastsurfer_native(scan_id) if prefer_native else None
    if native is not None:
        img, seg = native
        src = "fastsurfer_native_aparc21"
        print(
            f"  atlas source={src} shape={img.shape} "
            f"labeled%={(seg > 0).mean() * 100:.2f} "
            f"ctx_vox={(seg == 3).sum() + (seg == 42).sum()}"
        )
        return img, seg, path

    z = np.load(path)
    img = np.asarray(z["image"], dtype=np.float32)
    seg_npz = np.asarray(z["seg"], dtype=np.int32)
    seg_fs = load_fastsurfer_seg21(scan_id, img.shape)
    if seg_fs is not None and (seg_fs > 0).sum() >= (seg_npz > 0).sum():
        seg = seg_fs
        src = "fastsurfer_aparc21_resized"
    else:
        seg = seg_npz
        src = "npz_seg"
    print(f"  atlas source={src} labeled%={(seg > 0).mean() * 100:.2f}")
    return img, seg, path


def draw_prob_bars(ax, probs: dict[str, float], y_pred: str) -> None:
    classes = ["CN", "MCI", "AD"]
    vals = [float(probs[c]) for c in classes]
    colors = [OKABE[c] for c in classes]
    bars = ax.barh(classes[::-1], vals[::-1], color=colors[::-1], height=0.62, edgecolor="white", linewidth=0.8)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_xlabel("Subject-level probability (ARA-Net RC-SPE)", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for bar, v, c in zip(bars, vals[::-1], classes[::-1]):
        ax.text(
            min(v + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}" + ("  ← pred" if c == y_pred else ""),
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold" if c == y_pred else "normal",
            color="#222",
        )
    ax.axvline(0.5, color="#bbbbbb", lw=0.6, ls="--")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_facecolor("#fafafa")


def render_subject_figure(
    img: np.ndarray,
    seg: np.ndarray,
    meta: dict,
    out_path: Path,
    title: str | None = None,
    n_slices: int = 8,
    plane: str = "coronal",
    panel_size: int = 256,
) -> None:
    """Same subject, n equal-size slices on one plane; FastSurfer aparc→21."""
    lab2i = label_to_index_map()
    probs = {
        "CN": float(meta["prob_CN"]),
        "MCI": float(meta["prob_MCI"]),
        "AD": float(meta["prob_AD"]),
    }
    y_true = str(meta["y_true"])
    y_pred = str(meta["y_pred"])
    sid = str(meta.get("subject_id", meta.get("scan_id", "")))

    slice_ids = plane_slice_indices(img.shape, plane, n=n_slices, seg=seg)
    ncols = n_slices
    # Square panels → figsize keeps equal physical size
    cell = 1.15
    fig = plt.figure(figsize=(cell * ncols + 1.0, cell * 2 + 1.8), facecolor="white")
    gs = fig.add_gridspec(
        3,
        ncols,
        height_ratios=[1.0, 1.0, 0.70],
        hspace=0.08,
        wspace=0.02,
        left=0.06,
        right=0.995,
        top=0.88,
        bottom=0.07,
    )

    for j, idx in enumerate(slice_ids):
        ax1 = fig.add_subplot(gs[0, j])
        ax2 = fig.add_subplot(gs[1, j])
        mri_raw = slice_plane(img, plane, idx)
        seg_sl = slice_plane(seg, plane, idx)
        mri_raw, seg_sl = fit_panel_square(mri_raw, seg_sl, size=panel_size)
        mri = normalize(mri_raw)
        atlas = atlas_rgb_slice(seg_sl, lab2i, mri_sl=mri_raw)

        ax1.imshow(mri, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
        ax1.set_title(axis_tag(plane, idx), fontsize=7.5, pad=2)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_aspect("equal")
        for s in ax1.spines.values():
            s.set_visible(False)
        if j == 0:
            ax1.set_ylabel("Raw sMRI", fontsize=9, fontweight="bold", labelpad=4)

        ax2.imshow(atlas, interpolation="nearest")
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_aspect("equal")
        for s in ax2.spines.values():
            s.set_visible(False)
        if j == 0:
            ax2.set_ylabel("Atlas regions\n(21 parcels)", fontsize=8.5, fontweight="bold", labelpad=4)

    ax3 = fig.add_subplot(gs[2, :])
    draw_prob_bars(ax3, probs, y_pred)
    ax3.set_ylabel("Staging\nprobability", fontsize=9, fontweight="bold", labelpad=4)

    hdr = title or (
        f"ARA-Net atlas-guided staging  |  {sid}  |  "
        f"{plane} ×{n_slices} equal panels (FastSurfer aparc→21)  |  "
        f"true={y_true} → pred={y_pred}"
    )
    fig.suptitle(hdr, fontsize=11, fontweight="bold", y=0.97)

    legend_elems = [
        Patch(facecolor=(0.85, 0.40, 0.72), edgecolor="none", label="L-Cortex"),
        Patch(facecolor=(0.35, 0.65, 0.95), edgecolor="none", label="R-Cortex"),
        Patch(facecolor=(0.95, 0.93, 0.70), edgecolor="none", label="WM"),
        Patch(facecolor=(0.20, 0.45, 0.95), edgecolor="none", label="Ventricle"),
        Patch(facecolor=(0.95, 0.18, 0.18), edgecolor="none", label="Hippocampus"),
        Patch(facecolor=(1.00, 0.55, 0.10), edgecolor="none", label="Amygdala"),
        Patch(facecolor=(0.30, 0.70, 0.38), edgecolor="none", label="Cerebellum"),
    ]
    fig.legend(
        handles=legend_elems,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.955),
        ncol=7,
        frameon=False,
        fontsize=7,
        columnspacing=0.6,
        handlelength=0.9,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out_path}")


def render_multiplane_figure(
    img: np.ndarray,
    seg: np.ndarray,
    meta: dict,
    out_path: Path,
    n_slices: int = 8,
    panel_size: int = 256,
) -> None:
    """One subject: axial / coronal / sagittal strips, all equal panel size."""
    lab2i = label_to_index_map()
    probs = {
        "CN": float(meta["prob_CN"]),
        "MCI": float(meta["prob_MCI"]),
        "AD": float(meta["prob_AD"]),
    }
    y_true = str(meta["y_true"])
    y_pred = str(meta["y_pred"])
    sid = str(meta.get("subject_id", meta.get("scan_id", "")))

    cell = 1.05
    fig = plt.figure(figsize=(cell * n_slices + 1.2, cell * 6.2 + 1.5), facecolor="white")
    gs = fig.add_gridspec(
        7,
        n_slices,
        height_ratios=[1, 1, 1, 1, 1, 1, 0.85],
        hspace=0.10,
        wspace=0.02,
        left=0.08,
        right=0.995,
        top=0.91,
        bottom=0.05,
    )
    plane_rows = {
        "axial": (0, 1),
        "coronal": (2, 3),
        "sagittal": (4, 5),
    }

    for plane, (r_mri, r_atl) in plane_rows.items():
        ids = plane_slice_indices(img.shape, plane, n=n_slices, seg=seg)
        for j, idx in enumerate(ids):
            ax1 = fig.add_subplot(gs[r_mri, j])
            ax2 = fig.add_subplot(gs[r_atl, j])
            mri_raw, seg_sl = fit_panel_square(
                slice_plane(img, plane, idx),
                slice_plane(seg, plane, idx),
                size=panel_size,
            )
            ax1.imshow(normalize(mri_raw), cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            ax2.imshow(
                atlas_rgb_slice(seg_sl, lab2i, mri_sl=mri_raw),
                interpolation="nearest",
            )
            ax1.set_title(axis_tag(plane, idx), fontsize=6.5, pad=1)
            for ax in (ax1, ax2):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")
                for s in ax.spines.values():
                    s.set_visible(False)
            if j == 0:
                ax1.set_ylabel(f"{plane}\nMRI", fontsize=8, fontweight="bold")
                ax2.set_ylabel(f"{plane}\natlas", fontsize=8, fontweight="bold")

    axp = fig.add_subplot(gs[6, :])
    draw_prob_bars(axp, probs, y_pred)
    axp.set_ylabel("Staging", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"ARA-Net multi-plane staging  |  {sid}  |  "
        f"axial/coronal/sagittal ×{n_slices} equal panels  |  "
        f"true={y_true} → pred={y_pred}",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out_path}")


def render_staging_board(
    cases: list[tuple[str, dict, np.ndarray, np.ndarray]],
    out_path: Path,
    panel_size: int = 256,
) -> None:
    """3 subjects × 3 planes (mid dense slice), equal panel size."""
    n = len(cases)
    planes = ("axial", "coronal", "sagittal")
    fig = plt.figure(figsize=(3.4 * n, 10.5), facecolor="white")
    # rows: 3 planes × (MRI+atlas) = 6, + probs
    gs = fig.add_gridspec(
        7, n,
        height_ratios=[1, 1, 1, 1, 1, 1, 0.9],
        hspace=0.14,
        wspace=0.10,
        left=0.08,
        right=0.98,
        top=0.92,
        bottom=0.04,
    )
    lab2i = label_to_index_map()
    plane_rows = {
        "axial": (0, 1),
        "coronal": (2, 3),
        "sagittal": (4, 5),
    }

    for j, (tag, meta, img, seg) in enumerate(cases):
        probs = {
            "CN": float(meta["prob_CN"]),
            "MCI": float(meta["prob_MCI"]),
            "AD": float(meta["prob_AD"]),
        }
        for plane, (r_mri, r_atl) in plane_rows.items():
            ids = plane_slice_indices(img.shape, plane, n=8, seg=seg)
            idx = ids[len(ids) // 2]
            mri_raw, seg_sl = fit_panel_square(
                slice_plane(img, plane, idx),
                slice_plane(seg, plane, idx),
                size=panel_size,
            )
            ax1 = fig.add_subplot(gs[r_mri, j])
            ax2 = fig.add_subplot(gs[r_atl, j])
            ax1.imshow(normalize(mri_raw), cmap="gray", vmin=0, vmax=1)
            ax2.imshow(atlas_rgb_slice(seg_sl, lab2i, mri_sl=mri_raw), interpolation="nearest")
            if r_mri == 0:
                ax1.set_title(
                    f"{meta['subject_id']}\ntrue={meta['y_true']} → pred={meta['y_pred']}",
                    fontsize=9,
                    color=OKABE.get(str(meta["y_pred"]), "#222"),
                    fontweight="bold",
                )
            for ax in (ax1, ax2):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")
            if j == 0:
                ax1.set_ylabel(f"{plane}\nMRI", fontsize=8, fontweight="bold")
                ax2.set_ylabel(f"{plane}\natlas", fontsize=8, fontweight="bold")

        ax3 = fig.add_subplot(gs[6, j])
        draw_prob_bars(ax3, probs, str(meta["y_pred"]))
        if j == 0:
            ax3.set_ylabel("CN / MCI / AD", fontsize=9, fontweight="bold")

    fig.suptitle(
        "ARA-Net atlas-guided staging (equal panels · axial/coronal/sagittal)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case-picks",
        type=Path,
        default=ROOT / "reports/brain_figures_fcstyle/assets/matrices/case_picks.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports/brain_figures_fcstyle/figures",
    )
    ap.add_argument(
        "--subject",
        default="correct_AD",
        help="Key in case_picks.json, or 'all' for CN/MCI/AD board + each",
    )
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--panel-size", type=int, default=256)
    ap.add_argument(
        "--planes",
        default="axial,coronal,sagittal",
        help="Comma-separated planes to render",
    )
    args = ap.parse_args()
    planes = [p.strip() for p in args.planes.split(",") if p.strip()]

    cache_roots = [
        WORKSPACE / "data/external/aibl/cache_real",
        CHAPTER / "NeuroGate/local_assets/sample_data/cache_real",
        ROOT / "examples",
    ]

    picks = json.loads(args.case_picks.read_text())
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = ["correct_CN", "correct_MCI", "correct_AD"] if args.subject == "all" else [args.subject]

    board_cases = []
    for key in keys:
        if key not in picks:
            raise KeyError(f"{key} not in case_picks")
        meta = picks[key]
        img, seg, npz = load_case(meta["scan_id"], cache_roots)
        print(f"loaded {key}: {npz}")
        for plane in planes:
            out = out_dir / f"Fig_ARANet_atlas_staging_{key}_{plane}.png"
            render_subject_figure(
                img,
                seg,
                meta,
                out,
                n_slices=args.n_slices,
                plane=plane,
                panel_size=args.panel_size,
            )
        # multi-plane composite
        render_multiplane_figure(
            img,
            seg,
            meta,
            out_dir / f"Fig_ARANet_atlas_staging_{key}_multiplane.png",
            n_slices=args.n_slices,
            panel_size=args.panel_size,
        )
        # keep legacy name for coronal
        if "coronal" in planes:
            src = out_dir / f"Fig_ARANet_atlas_staging_{key}_coronal.png"
            (out_dir / f"Fig_ARANet_atlas_staging_{key}.png").write_bytes(src.read_bytes())
        if key in ("correct_CN", "correct_MCI", "correct_AD"):
            board_cases.append((key, meta, img, seg))

    if args.subject == "all" and len(board_cases) == 3:
        render_staging_board(
            board_cases,
            out_dir / "Fig_ARANet_atlas_staging_CN_MCI_AD_board.png",
            panel_size=args.panel_size,
        )

    primary_key = "correct_AD" if args.subject == "all" else args.subject
    for name in (
        f"Fig_ARANet_atlas_staging_{primary_key}_coronal.png",
        f"Fig_ARANet_atlas_staging_{primary_key}_multiplane.png",
    ):
        src = out_dir / name
        if src.exists() and "coronal" in name:
            alias = out_dir / "Fig_ARANet_atlas_staging_pipeline.png"
            alias.write_bytes(src.read_bytes())
            print(f"wrote {alias} (alias of {src.name})")
        if src.exists() and "multiplane" in name:
            alias = out_dir / "Fig_ARANet_atlas_staging_pipeline_multiplane.png"
            alias.write_bytes(src.read_bytes())
            print(f"wrote {alias} (alias of {src.name})")


if __name__ == "__main__":
    main()
