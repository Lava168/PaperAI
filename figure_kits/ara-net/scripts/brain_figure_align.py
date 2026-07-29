#!/usr/bin/env python3
"""Shared helpers to remove anatomy 'white ghost' and avoid fake-MNI glass mismatch.

Root causes (NOT a voxel-grid bug between image/seg — those share 96×112×96):
1. Mean-T1 underlay has ~380k voxels outside the atlas → grayscale rim / 白影.
2. TEMPLATE_AFFINE is only an approximate MNI-like crop. nilearn plot_glass_brain
   always draws a true MNI152 silhouette → overlay looks shifted / oversized.

Affine headers for underlay vs overlay already match; do not 'fix' by resampling
onto MNI with the approximate affine.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

try:
    import nibabel as nib
    from nilearn import plotting
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"nibabel/nilearn required: {exc}")

NiftiLike = Union["nib.Nifti1Image", str]


def _as_img(obj: NiftiLike) -> "nib.Nifti1Image":
    if isinstance(obj, nib.Nifti1Image):
        return obj
    return nib.load(str(obj))


def prepare_anat_underlay(
    underlay: NiftiLike,
    atlas: Optional[NiftiLike] = None,
    *,
    dim: float = 0.40,
    brain_pct: float = 5.0,
    dilate_iter: int = 0,
) -> "nib.Nifti1Image":
    """Mask extracranial air and dim grayscale so overlays sit on a tight brain.

    Default dilate_iter=0: underlay FOV == atlas FOV (kills white halo outside ROIs).
    """
    bg = _as_img(underlay)
    data = np.asanyarray(bg.dataobj, dtype=np.float32)
    pos = data[data > 0]
    thr = float(np.percentile(pos, brain_pct)) if pos.size else 0.0
    brain = data > thr

    if atlas is not None:
        atl = np.asanyarray(_as_img(atlas).dataobj)
        roi = atl > 0
        if dilate_iter > 0:
            try:
                from scipy.ndimage import binary_dilation

                roi = binary_dilation(roi, iterations=int(dilate_iter))
            except Exception:
                pass
        brain = brain & roi

    out = np.where(brain, data * float(dim), 0.0).astype(np.float32)
    return nib.Nifti1Image(out, bg.affine, bg.header.copy())


def prepare_glass_map(
    img: NiftiLike,
    *,
    fwhm: float = 0.0,
) -> "nib.Nifti1Image":
    """Optional light smooth in *native* space (no MNI resample)."""
    src = _as_img(img)
    if fwhm and fwhm > 0:
        from nilearn import image as nimg

        return nimg.smooth_img(src, fwhm=float(fwhm))
    return src


def plot_native_glass(
    img: NiftiLike,
    *,
    axes=None,
    title: str = "",
    cmap: str = "cold_hot",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    threshold: Optional[float] = 1e-4,
    colorbar: bool = True,
    cuts: Optional[Sequence[float]] = None,
) -> object:
    """Glass-*like* multi-view in native crop space (no MNI152 silhouette).

    Uses black background ortho cuts so the map cannot 'miss' a foreign outline.
    """
    src = prepare_glass_map(img, fwhm=0.0)
    cut = list(cuts) if cuts is not None else [-20, 0, 20]
    # Single axial strip is compact; callers that want 4-view use plot_stat_map ortho.
    return plotting.plot_stat_map(
        src,
        display_mode="z",
        cut_coords=cut,
        axes=axes,
        colorbar=colorbar,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        annotate=False,
        draw_cross=False,
        black_bg=True,
        bg_img=None,
        title=title,
    )


def plot_native_ortho_glass(
    img: NiftiLike,
    *,
    axes=None,
    title: str = "",
    cmap: str = "cold_hot",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    threshold: Optional[float] = 1e-4,
    colorbar: bool = True,
) -> object:
    """3-plane ortho on black bg — preferred replacement for plot_glass_brain.

    Critical: pass bg_img=None. Nilearn defaults to MNI152Template, which does not
    match our approximate crop affine and creates the white-shadow / misalignment look.
    """
    src = prepare_glass_map(img, fwhm=0.0)
    return plotting.plot_stat_map(
        src,
        display_mode="ortho",
        cut_coords=(-20, -10, 10),
        axes=axes,
        colorbar=colorbar,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        annotate=True,
        draw_cross=False,
        black_bg=True,
        bg_img=None,
        title=title,
    )
