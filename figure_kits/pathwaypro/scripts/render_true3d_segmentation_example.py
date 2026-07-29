#!/usr/bin/env python3
"""Render a real 3D brain/ROI preview from cached PathwayPro volumes.

This example uses actual cached T1 segmentation (`data/full_cache/{scan}.npz`):
- `seg > 0` for the transparent brain envelope
- remapped FastSurfer labels for hippocampus, amygdala, and thalamus

It is intentionally lightweight and uses only nibabel-free cached arrays,
scikit-image marching cubes, and matplotlib, so it works on the current server
without PyVista/VTK.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


ROI_GROUPS = {
    # Original FreeSurfer labels -> remapped by data/dkt_label_map.json:
    # left/right hippocampus 17/53 -> 67/75
    "Hippocampus": {"labels": [67, 75], "color": "#E65F2B", "alpha": 0.96},
    # left/right amygdala 18/54 -> 68/76
    "Amygdala": {"labels": [68, 76], "color": "#BC3C7D", "alpha": 0.96},
    # left/right thalamus 10/49 -> 63/71
    "Thalamus": {"labels": [63, 71], "color": "#2F7D95", "alpha": 0.90},
}


def add_surface(ax, mask, color, alpha, step_size=2, linewidth=0.0):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.sum() < 20:
        return None
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, step_size=step_size)
    mesh = Poly3DCollection(verts[faces], linewidth=linewidth)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    mesh.set_alpha(alpha)
    ax.add_collection3d(mesh)
    return verts


def setup_axis(ax, mask, title, elev=18, azim=-62):
    coords = np.argwhere(mask)
    lo = coords.min(axis=0) - 4
    hi = coords.max(axis=0) + 4
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(mask.shape))
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_zlim(float(lo[2]), float(hi[2]))
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=2)
    # White panes look cleaner for manuscript previews.
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_id", default="DEMO_0001")
    parser.add_argument("--cache_dir", default="data/full_cache")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cache = Path(args.cache_dir) / f"{args.scan_id}.npz"
    data = np.load(cache)
    seg = data["seg"].astype(np.int16)
    image = data["image"].astype(np.float32)
    brain = seg > 0

    # Cutaway view reveals deep structures instead of hiding everything behind
    # a translucent outer envelope.
    mid_x = seg.shape[0] // 2
    cutaway = brain.copy()
    cutaway[: mid_x - 4, :, :] = False

    fig = plt.figure(figsize=(12, 5.2), facecolor="white")
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    add_surface(ax1, brain, "#C9CED6", 0.10, step_size=2)
    add_surface(ax2, cutaway, "#C9CED6", 0.14, step_size=2)

    for name, cfg in ROI_GROUPS.items():
        roi = np.isin(seg, cfg["labels"])
        add_surface(ax1, roi, cfg["color"], cfg["alpha"], step_size=1)
        add_surface(ax2, roi, cfg["color"], cfg["alpha"], step_size=1)

    setup_axis(ax1, brain, f"{args.scan_id}: transparent brain", elev=18, azim=-62)
    setup_axis(ax2, brain, "cutaway view of deep ROIs", elev=18, azim=-48)

    # Add a compact legend and provenance text.
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=name,
                   markerfacecolor=cfg["color"], markersize=9)
        for name, cfg in ROI_GROUPS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.025))
    fig.text(
        0.5,
        0.005,
        "Real cached T1 segmentation render: brain envelope = seg>0; highlighted ROIs = remapped FastSurfer labels.",
        ha="center",
        va="center",
        fontsize=8,
        color="#222222",
    )
    fig.suptitle("Example real 3D rendering from PathwayPro preprocessing cache", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.92))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
