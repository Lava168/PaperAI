#!/usr/bin/env python3
"""Render real 3D Grad-CAM and ROI-attribution overlays.

Inputs are all real project artifacts:
- T1 image and segmentation from `data/full_cache/{subject_id}.npz`
- 3D Grad-CAM from `analysis_explainability.../gradcam/{subject}_t1_ad_logit.nii.gz`
- ROI ablation attribution from `roi_ablation.csv`

The output is a clear preview figure, not a schematic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def robust01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    lo, hi = np.percentile(finite, [1, 99])
    arr = np.clip(arr, lo, hi)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)


def normalize_cam(cam):
    cam = np.asarray(cam, dtype=np.float32)
    if float(cam.max()) <= float(cam.min()):
        return np.zeros_like(cam, dtype=np.float32)
    lo, hi = np.percentile(cam, [1, 99.8])
    cam = np.clip(cam, lo, hi)
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)


def best_cam_slice(cam, plane="axial"):
    if plane == "axial":
        return int(np.argmax(cam.sum(axis=(0, 1))))
    if plane == "coronal":
        return int(np.argmax(cam.sum(axis=(0, 2))))
    if plane == "sagittal":
        return int(np.argmax(cam.sum(axis=(1, 2))))
    raise ValueError(plane)


def add_surface(ax, mask, color, alpha=1.0, step_size=1, linewidth=0.0):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.sum() < 30:
        return None
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, step_size=step_size)
    mesh = Poly3DCollection(verts[faces], linewidth=linewidth)
    mesh.set_facecolor(color)
    mesh.set_edgecolor((0, 0, 0, 0.04))
    mesh.set_alpha(alpha)
    ax.add_collection3d(mesh)
    return verts


def bbox_from_masks(*masks, pad=10):
    mask = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        mask |= np.asarray(m, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.array([0, 0, 0]), np.asarray(mask.shape)
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad, np.asarray(mask.shape))
    return lo, hi


def setup_axis(ax, lo, hi, title, elev=22, azim=-55):
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_zlim(float(lo[2]), float(hi[2]))
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11, pad=2)
    ax.set_axis_off()
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))


def plot_t1_cam_slice(ax, image, cam, seg):
    z = best_cam_slice(cam, "axial")
    img = robust01(image[:, :, z]).T
    overlay = cam[:, :, z].T
    ax.imshow(img, cmap="gray", origin="lower")
    cmap = plt.get_cmap("inferno")
    rgba = cmap(np.clip(overlay, 0, 1))
    rgba[..., 3] = np.where(overlay > 0.08, np.clip((overlay - 0.08) / 0.55, 0, 0.72), 0)
    ax.imshow(rgba, origin="lower")
    ax.set_title(f"T1 axial Grad-CAM slice #{z}", fontsize=11)
    ax.set_axis_off()


def top_roi_rows(roi_csv: Path, subject_id: str, branch: str, top_k: int):
    df = pd.read_csv(roi_csv)
    df = df[(df["subject_id"].astype(str) == subject_id) & (df["branch"].astype(str) == branch)].copy()
    df["delta_p_ad"] = pd.to_numeric(df["delta_p_ad"], errors="coerce")
    df = df[df["delta_p_ad"] > 0].sort_values("delta_p_ad", ascending=False).head(top_k)
    return df


def add_roi_attribution(ax, seg, roi_rows):
    if roi_rows.empty:
        return []
    vals = roi_rows["delta_p_ad"].to_numpy(dtype=float)
    norm = Normalize(vmin=float(vals.min()), vmax=float(vals.max()) + 1e-12)
    cmap = plt.get_cmap("viridis")
    handles = []
    for _, row in roi_rows.iterrows():
        roi_id = int(row["roi_id"])
        val = float(row["delta_p_ad"])
        color = cmap(norm(val))
        add_surface(ax, seg == roi_id, color=color, alpha=0.96, step_size=1, linewidth=0.04)
        handles.append(Patch(facecolor=color, edgecolor="none", label=f"ROI {roi_id}: ΔP={val:.4f}"))
    return handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", default="DEMO_0008")
    parser.add_argument("--cache_dir", default="data/full_cache")
    parser.add_argument("--gradcam_dir", default="analysis_explainability_factorvae_seed2_full/gradcam")
    parser.add_argument("--roi_csv", default="analysis_explainability_factorvae_seed2_full/roi_ablation.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top_k_roi", type=int, default=8)
    parser.add_argument("--cam_percentile", type=float, default=99.2)
    args = parser.parse_args()

    cache = Path(args.cache_dir) / f"{args.subject_id}.npz"
    data = np.load(cache)
    image = data["image"].astype(np.float32)
    seg = data["seg"].astype(np.int16)
    brain = seg > 0

    cam_path = Path(args.gradcam_dir) / f"{args.subject_id}_t1_ad_logit.nii.gz"
    cam_raw = np.asarray(nib.load(str(cam_path)).get_fdata(), dtype=np.float32)
    cam = normalize_cam(cam_raw) * brain
    if cam.max() <= 0:
        raise ValueError(f"Grad-CAM is empty for {args.subject_id}: {cam_path}")
    cam_threshold = float(np.percentile(cam[brain], args.cam_percentile))
    cam_mask = (cam >= cam_threshold) & brain

    roi_rows = top_roi_rows(Path(args.roi_csv), args.subject_id, "t1", args.top_k_roi)
    roi_mask = np.isin(seg, roi_rows["roi_id"].astype(int).tolist()) if not roi_rows.empty else np.zeros_like(brain)
    lo, hi = bbox_from_masks(brain, cam_mask, roi_mask, pad=8)

    fig = plt.figure(figsize=(14.5, 5.2), facecolor="white")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.0, 1.0, 1.0], wspace=0.03)

    ax_slice = fig.add_subplot(gs[0, 0])
    plot_t1_cam_slice(ax_slice, image, cam, seg)

    ax_cam = fig.add_subplot(gs[0, 1], projection="3d")
    add_surface(ax_cam, brain, "#D8DDE5", alpha=0.08, step_size=3)
    add_surface(ax_cam, cam_mask, "#F05A28", alpha=0.92, step_size=1, linewidth=0.02)
    setup_axis(ax_cam, lo, hi, f"3D T1 Grad-CAM top {100-args.cam_percentile:.1f}%", elev=20, azim=-55)

    ax_roi = fig.add_subplot(gs[0, 2], projection="3d")
    add_roi_attribution(ax_roi, seg, roi_rows)
    roi_lo, roi_hi = bbox_from_masks(roi_mask if roi_mask.any() else brain, pad=12)
    setup_axis(ax_roi, roi_lo, roi_hi, "Top ROI ablation attribution", elev=22, azim=-46)

    ax_combined = fig.add_subplot(gs[0, 3], projection="3d")
    add_roi_attribution(ax_combined, seg, roi_rows.head(5))
    add_surface(ax_combined, cam_mask, "#FFB000", alpha=0.62, step_size=1, linewidth=0.02)
    setup_axis(ax_combined, roi_lo, roi_hi, "Combined ROI + Grad-CAM", elev=28, azim=-36)

    fig.suptitle(f"Real 3D attribution overlay: {args.subject_id} (FactorVAE seed 2, AD logit)", fontsize=14, y=0.98)
    legend_items = [
        Patch(facecolor="#F05A28", edgecolor="none", label="T1 Grad-CAM high-attribution voxels"),
        Patch(facecolor="#FFB000", edgecolor="none", label="Grad-CAM overlay in combined panel"),
    ]
    top3 = roi_rows.head(3)
    for _, row in top3.iterrows():
        legend_items.append(Patch(facecolor="#27869A", alpha=0.0, label=f"Top ROI {int(row['roi_id'])}: ΔP(AD)={float(row['delta_p_ad']):.4f}"))
    fig.legend(handles=legend_items, loc="lower center", ncol=2, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0.01, 0.09, 0.99, 0.92))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=280, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
