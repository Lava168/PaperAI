#!/usr/bin/env python3
"""Render crisp figures from server-side model Grad-CAM outputs.

This consumes real model-side artifacts downloaded from the server:
- T1 and segmentation cache for a subject
- 3D Grad-CAM NIfTI from the trained model
- ROI ablation CSV from the explainability run

No atlas-to-pixel smearing is performed. Pixel/voxel evidence is shown first,
then summarized back to atlas/ROI labels.
"""
from __future__ import annotations

import csv
import gzip
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parent
DL = ROOT / "server_gradcam_download"
OUT = ROOT / "figures_nbe_style"
SUBJECT_ID = "021_S_4718"

T1_NPZ = DL / "data/full_cache/021_S_4718.npz"
CAM_NII = DL / "analysis_explainability_seed0_full/gradcam/021_S_4718_t1_ad_logit.nii.gz"
SUBJECTS_CSV = DL / "analysis_explainability_seed0_full/gradcam_subjects.csv"
ROI_CSV = DL / "analysis_explainability_seed0_full/roi_ablation.csv"

PALETTE = {
    "ink": "#1B1F23",
    "slate": "#4B5563",
    "muted": "#8A94A3",
    "grid": "#D8DEE6",
    "red": "#C9583B",
    "gold": "#E3A53B",
    "green": "#4C9A78",
}


def read_nifti_gz_float32(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        blob = f.read()
    sizeof_hdr = struct.unpack("<i", blob[0:4])[0]
    if sizeof_hdr != 348:
        raise ValueError(f"Unexpected NIfTI header size {sizeof_hdr}")
    dim = struct.unpack("<8h", blob[40:56])
    datatype = struct.unpack("<h", blob[70:72])[0]
    bitpix = struct.unpack("<h", blob[72:74])[0]
    vox_offset = int(struct.unpack("<f", blob[108:112])[0])
    if datatype != 16 or bitpix != 32:
        raise ValueError(f"Expected float32 NIfTI, got datatype={datatype} bitpix={bitpix}")
    shape = tuple(int(v) for v in dim[1:4])
    arr = np.frombuffer(blob[vox_offset:], dtype="<f4", count=int(np.prod(shape))).copy()
    return arr.reshape(shape, order="F")


def robust01(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    vals = x[mask] if mask is not None and np.any(mask) else x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x)
    lo, hi = np.percentile(vals, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)


def normalize_cam(cam: np.ndarray, brain: np.ndarray) -> np.ndarray:
    vals = cam[brain]
    if vals.size == 0 or float(vals.max()) <= float(vals.min()):
        return np.zeros_like(cam)
    lo, hi = np.percentile(vals, [2, 99.9])
    cam = np.clip(cam, lo, hi)
    cam = (cam - lo) / max(hi - lo, 1e-8)
    return np.clip(cam, 0, 1) * brain


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        nxt = out.copy()
        nxt[1:, :, :] &= out[:-1, :, :]
        nxt[:-1, :, :] &= out[1:, :, :]
        nxt[:, 1:, :] &= out[:, :-1, :]
        nxt[:, :-1, :] &= out[:, 1:, :]
        nxt[:, :, 1:] &= out[:, :, :-1]
        nxt[:, :, :-1] &= out[:, :, 1:]
        out = nxt
    return out


def view_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return np.rot90(vol[idx, :, :])
    if axis == 1:
        return np.rot90(vol[:, idx, :])
    return np.rot90(vol[:, :, idx])


def best_slice(cam: np.ndarray, brain: np.ndarray, axis: int) -> int:
    weighted = cam * brain
    if axis == 0:
        return int(np.argmax(weighted.sum(axis=(1, 2))))
    if axis == 1:
        return int(np.argmax(weighted.sum(axis=(0, 2))))
    return int(np.argmax(weighted.sum(axis=(0, 1))))


def subject_info() -> Dict[str, str]:
    with SUBJECTS_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["subject_id"] == SUBJECT_ID and row["branch"] == "t1":
                return row
    return {}


def roi_summary(cam: np.ndarray, seg: np.ndarray, top_n: int = 9) -> List[Tuple[str, float, float]]:
    ablation: Dict[int, float] = {}
    with ROI_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["subject_id"] != SUBJECT_ID or row["branch"] != "t1":
                continue
            ablation[int(row["roi_id"])] = float(row["delta_p_ad"])
    rows = []
    for roi in sorted(int(v) for v in np.unique(seg) if int(v) > 0):
        m = seg == roi
        active = cam[m]
        cam_val = float(np.percentile(active, 98)) if active.size else 0.0
        delta = ablation.get(roi, 0.0)
        combined = cam_val * max(delta, 0.0)
        if cam_val > 0 or delta > 0:
            rows.append((f"ROI {roi}", cam_val, delta, combined))
    rows.sort(key=lambda x: (x[3], x[1]), reverse=True)
    return [(name, cam_val, delta) for name, cam_val, delta, _ in rows[:top_n]]


def crop_bounds(mask2d: np.ndarray, pad: int = 8) -> Tuple[slice, slice]:
    ys, xs = np.where(mask2d)
    if ys.size == 0:
        return slice(None), slice(None)
    y0, y1 = max(0, int(ys.min()) - pad), min(mask2d.shape[0], int(ys.max()) + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(mask2d.shape[1], int(xs.max()) + pad)
    return slice(y0, y1), slice(x0, x1)


def draw_crisp_overlay(ax, image2d: np.ndarray, cam2d: np.ndarray, brain2d: np.ndarray, title: str) -> None:
    sy, sx = crop_bounds(brain2d, pad=8)
    img = image2d[sy, sx]
    cam = cam2d[sy, sx]
    brain = brain2d[sy, sx]
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    vals = cam[brain]
    if vals.size and vals.max() > 0:
        hard = cam >= np.percentile(vals, 98.5)
        soft = cam >= np.percentile(vals, 96.5)
        cmap = LinearSegmentedColormap.from_list("crispred", ["#F6C0B3", PALETTE["red"]])
        rgba_soft = cmap(np.clip(cam, 0, 1))
        rgba_soft[..., 3] = np.where(soft, 0.34 + 0.28 * cam, 0.0)
        rgba_hard = cmap(np.clip(cam, 0, 1))
        rgba_hard[..., 3] = np.where(hard, 0.74, 0.0)
        ax.imshow(rgba_soft, interpolation="nearest")
        ax.imshow(rgba_hard, interpolation="nearest")
        ax.contour(hard.astype(float), levels=[0.5], colors=["#7F1D1D"], linewidths=0.55)
    ax.set_title(title, loc="left", fontsize=8.5, color=PALETTE["slate"], pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_cam_mip(ax, cam: np.ndarray, brain: np.ndarray) -> None:
    mip = cam.max(axis=0)
    bmip = brain.max(axis=0)
    sy, sx = crop_bounds(np.rot90(bmip), pad=8)
    mip2 = np.rot90(mip)[sy, sx]
    bmip2 = np.rot90(bmip)[sy, sx]
    ax.imshow(np.zeros_like(mip2), cmap="gray", vmin=0, vmax=1)
    rgba = plt.get_cmap("inferno")(np.clip(mip2, 0, 1))
    vals = mip2[bmip2]
    th = np.percentile(vals, 92) if vals.size else 1.0
    rgba[..., 3] = np.where(mip2 >= th, 0.28 + 0.66 * mip2, 0.0)
    ax.imshow(rgba, interpolation="nearest")
    ax.set_title("Voxel evidence maximum projection", loc="left", fontsize=8.5, color=PALETTE["slate"], pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_figure() -> None:
    z = np.load(T1_NPZ)
    image = np.asarray(z["image"], dtype=np.float32)
    seg = np.asarray(z["seg"], dtype=np.int16)
    brain = erode(seg > 0, iterations=1)
    image01 = robust01(image, seg > 0)
    cam = normalize_cam(read_nifti_gz_float32(CAM_NII), brain)

    info = subject_info()
    rois = roi_summary(cam, seg, top_n=8)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 7.2, "pdf.fonttype": 42, "ps.fonttype": 42})

    fig = plt.figure(figsize=(8.27, 6.15), facecolor="white")
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 0.70], hspace=0.30, wspace=0.34)
    views = [("Sagittal", 0), ("Coronal", 1), ("Axial", 2)]
    for i, (name, axis) in enumerate(views):
        ax = fig.add_subplot(gs[0, i * 4 : (i + 1) * 4])
        idx = best_slice(cam, brain, axis)
        draw_crisp_overlay(
            ax,
            view_slice(image01, axis, idx),
            view_slice(cam, axis, idx),
            view_slice(brain, axis, idx),
            f"{name} model-side Grad-CAM",
        )

    ax_mip = fig.add_subplot(gs[1, 0:4])
    draw_cam_mip(ax_mip, cam, brain)

    ax_bar = fig.add_subplot(gs[1, 4:8])
    names = [r[0] for r in rois][::-1]
    cam_vals = np.array([r[1] for r in rois][::-1])
    delta_vals = np.array([max(r[2], 0.0) for r in rois][::-1])
    score = cam_vals * (delta_vals / max(float(delta_vals.max()), 1e-8) if delta_vals.max() > 0 else 1)
    yy = np.arange(len(names))
    ax_bar.barh(yy, score, color=PALETTE["red"], alpha=0.88, edgecolor="white", linewidth=0.4)
    ax_bar.set_yticks(yy)
    ax_bar.set_yticklabels(names, fontsize=6.0)
    ax_bar.set_xlabel("Grad-CAM x ROI ablation", labelpad=1)
    ax_bar.set_title("Regional readout after voxel localization", loc="left", fontsize=8.5, color=PALETTE["slate"], pad=3)
    for side in ["top", "right"]:
        ax_bar.spines[side].set_visible(False)
    ax_bar.spines["left"].set_color(PALETTE["slate"])
    ax_bar.spines["bottom"].set_color(PALETTE["slate"])
    ax_bar.tick_params(axis="x", labelsize=6.0)

    ax_logic = fig.add_subplot(gs[1, 8:12])
    ax_logic.set_xlim(0, 1)
    ax_logic.set_ylim(0, 1)
    ax_logic.axis("off")
    p_ad = float(info.get("p_ad", "nan")) if info else float("nan")
    score_raw = float(info.get("score", "nan")) if info else float("nan")
    ax_logic.text(0.02, 0.94, "Server-side model evidence", ha="left", va="top", fontsize=8.5, color=PALETTE["slate"], fontweight="bold")
    lines = [
        f"Subject: {SUBJECT_ID}",
        f"Target: AD logit; p(AD)={p_ad:.3f}",
        f"Model score={score_raw:.3f}",
        "Voxel map: trained-model 3D Grad-CAM",
        "Display: eroded brain mask + top-percentile hotspots",
        "Region map: ROI ablation readout after voxel localization",
    ]
    for i, line in enumerate(lines):
        y = 0.78 - i * 0.105
        ax_logic.add_patch(Circle((0.04, y), 0.012, color=PALETTE["red"] if i < 3 else PALETTE["gold"], alpha=0.9))
        ax_logic.text(0.075, y, line, ha="left", va="center", fontsize=6.2, color=PALETTE["ink" if i < 3 else "slate"])
    ax_logic.text(
        0.02,
        0.05,
        "This is model-side voxel evidence, not atlas-smoothed ATE. It should still be described as attribution evidence, not voxel-level lesion ground truth.",
        ha="left",
        va="bottom",
        fontsize=5.9,
        color=PALETTE["slate"],
        wrap=True,
    )

    fig.suptitle("Crisp model-side dense evidence for A2C/PathwayPro AD prediction", x=0.045, y=0.975, ha="left", fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(left=0.060, right=0.985, top=0.900, bottom=0.075)
    fig.savefig(OUT / "Server_model_side_crisp_dense_evidence.png", dpi=420)
    fig.savefig(OUT / "Server_model_side_crisp_dense_evidence.pdf")
    plt.close(fig)
    print(OUT / "Server_model_side_crisp_dense_evidence.png")


if __name__ == "__main__":
    make_figure()
