#!/usr/bin/env python3
"""Clearer real 3D ROI render from cached PathwayPro segmentation.

This version is meant for visual inspection: it avoids the cloudy transparent
whole-brain surface and instead combines a true T1 slice locator with enlarged,
opaque 3D ROI surfaces rendered from the real segmentation cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


ROI_GROUPS = {
    "Hippocampus": {"labels": [67, 75], "color": "#F05A28", "alpha": 1.0},
    "Amygdala": {"labels": [68, 76], "color": "#C43C83", "alpha": 1.0},
    "Thalamus": {"labels": [63, 71], "color": "#27869A", "alpha": 1.0},
}


def robust01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[np.isfinite(arr)], [1, 99])
    arr = np.clip(arr, lo, hi)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)


def add_surface(ax, mask, color, alpha=1.0, step_size=1):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.sum() < 20:
        return None
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, step_size=step_size)
    mesh = Poly3DCollection(verts[faces], linewidth=0.08)
    mesh.set_facecolor(color)
    mesh.set_edgecolor((0, 0, 0, 0.08))
    mesh.set_alpha(alpha)
    ax.add_collection3d(mesh)
    return verts


def roi_bbox(seg):
    mask = np.zeros(seg.shape, dtype=bool)
    for cfg in ROI_GROUPS.values():
        mask |= np.isin(seg, cfg["labels"])
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - 14, 0)
    hi = np.minimum(coords.max(axis=0) + 14, np.asarray(seg.shape))
    return lo, hi, mask


def setup_roi_axis(ax, lo, hi, title, elev, azim):
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_zlim(float(lo[2]), float(hi[2]))
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=2)
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))


def overlay_slice(ax, image, seg, roi_mask):
    # Choose axial slice with largest ROI footprint.
    z = int(np.argmax(roi_mask.sum(axis=(0, 1))))
    img = robust01(image[:, :, z]).T
    ax.imshow(img, cmap="gray", origin="lower")
    rgba = np.zeros((*img.shape, 4), dtype=float)
    for cfg in ROI_GROUPS.values():
        mask2d = np.isin(seg[:, :, z], cfg["labels"]).T
        rgb = matplotlib.colors.to_rgb(cfg["color"])
        rgba[mask2d, :3] = rgb
        rgba[mask2d, 3] = 0.72
    ax.imshow(rgba, origin="lower")
    ax.set_title(f"True T1 axial slice #{z}", fontsize=11)
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_id", default="DEMO_0001")
    parser.add_argument("--cache_dir", default="data/full_cache")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cache = Path(args.cache_dir) / f"{args.scan_id}.npz"
    data = np.load(cache)
    image = data["image"].astype(np.float32)
    seg = data["seg"].astype(np.int16)
    lo, hi, all_roi = roi_bbox(seg)

    fig = plt.figure(figsize=(13.5, 4.7), facecolor="white")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.0, 1.0, 1.0], wspace=0.02)
    ax_slice = fig.add_subplot(gs[0, 0])
    overlay_slice(ax_slice, image, seg, all_roi)

    views = [
        ("Anterior 3D ROI view", 18, -86),
        ("Left-oblique 3D view", 20, -42),
        ("Superior 3D view", 74, -62),
    ]
    for idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(gs[0, idx], projection="3d")
        for cfg in ROI_GROUPS.values():
            add_surface(ax, np.isin(seg, cfg["labels"]), cfg["color"], cfg["alpha"], step_size=1)
        setup_roi_axis(ax, lo, hi, title, elev=elev, azim=azim)

    handles = [Patch(facecolor=cfg["color"], edgecolor="none", label=name) for name, cfg in ROI_GROUPS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"Clear real 3D ROI render from cached T1 segmentation: {args.scan_id}", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0.01, 0.08, 0.99, 0.93))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=280, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
