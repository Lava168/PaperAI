#!/usr/bin/env python3
"""Crisp gallery from real server-side model Grad-CAM outputs."""
from __future__ import annotations

import csv
import gzip
import struct
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
DL = ROOT / "server_gradcam_download"
OUT = ROOT / "figures_nbe_style"
CASES = ["014_S_4615", "027_S_5197", "018_S_4809"]
PALETTE = {"ink": "#1B1F23", "slate": "#4B5563", "red": "#C9583B", "gold": "#E3A53B"}


def read_nifti_gz_float32(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        blob = f.read()
    dim = struct.unpack("<8h", blob[40:56])
    datatype = struct.unpack("<h", blob[70:72])[0]
    vox_offset = int(struct.unpack("<f", blob[108:112])[0])
    if datatype != 16:
        raise ValueError(f"Expected float32 NIfTI: {path}")
    shape = tuple(int(v) for v in dim[1:4])
    return np.frombuffer(blob[vox_offset:], dtype="<f4", count=int(np.prod(shape))).copy().reshape(shape, order="F")


def case_npz_path(sid: str) -> Path:
    p = DL / "top_cases/full_cache" / f"{sid}.npz"
    if p.exists():
        return p
    p = DL / "top_cases" / f"{sid}.npz"
    if p.exists():
        return p
    return DL / "data/full_cache" / f"{sid}.npz"


def case_cam_path(sid: str) -> Path:
    p = DL / "top_cases/gradcam" / f"{sid}_t1_ad_logit.nii.gz"
    if p.exists():
        return p
    p = DL / "top_cases" / f"{sid}_t1_ad_logit.nii.gz"
    if p.exists():
        return p
    return DL / "analysis_explainability_seed0_full/gradcam" / f"{sid}_t1_ad_logit.nii.gz"


def subject_info() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with (DL / "analysis_explainability_seed0_full/gradcam_subjects.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["branch"] == "t1":
                out[row["subject_id"]] = row
    return out


def robust01(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    vals = x[mask] if mask is not None and np.any(mask) else x[np.isfinite(x)]
    lo, hi = np.percentile(vals, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)


def erode(mask: np.ndarray) -> np.ndarray:
    out = mask.astype(bool)
    nxt = out.copy()
    nxt[1:, :, :] &= out[:-1, :, :]
    nxt[:-1, :, :] &= out[1:, :, :]
    nxt[:, 1:, :] &= out[:, :-1, :]
    nxt[:, :-1, :] &= out[:, 1:, :]
    nxt[:, :, 1:] &= out[:, :, :-1]
    nxt[:, :, :-1] &= out[:, :, 1:]
    return nxt


def normalize_cam(cam: np.ndarray, brain: np.ndarray) -> np.ndarray:
    vals = cam[brain]
    lo, hi = np.percentile(vals, [5, 99.9])
    cam = np.clip((cam - lo) / max(hi - lo, 1e-8), 0, 1)
    return cam * brain


def best_slice(cam: np.ndarray, axis: int) -> int:
    if axis == 0:
        return int(np.argmax(cam.sum(axis=(1, 2))))
    if axis == 1:
        return int(np.argmax(cam.sum(axis=(0, 2))))
    return int(np.argmax(cam.sum(axis=(0, 1))))


def view_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return np.rot90(vol[idx, :, :])
    if axis == 1:
        return np.rot90(vol[:, idx, :])
    return np.rot90(vol[:, :, idx])


def bounds(mask: np.ndarray, pad: int = 5) -> Tuple[slice, slice]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return slice(None), slice(None)
    return (
        slice(max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad)),
        slice(max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad)),
    )


def render(ax, image2d: np.ndarray, cam2d: np.ndarray, brain2d: np.ndarray, title: str) -> None:
    sy, sx = bounds(brain2d, pad=5)
    img = image2d[sy, sx]
    cam = cam2d[sy, sx]
    brain = brain2d[sy, sx]
    ax.set_facecolor("black")
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="hanning")
    vals = cam[brain]
    if vals.size and vals.max() > 0:
        th_soft = np.percentile(vals, 97.2)
        th_hard = np.percentile(vals, 99.0)
        soft = (cam >= th_soft) & brain
        hard = (cam >= th_hard) & brain
        cmap = LinearSegmentedColormap.from_list("hotred", ["#F5B09E", PALETTE["red"]])
        rgba = cmap(np.clip(cam, 0, 1))
        rgba[..., 3] = np.where(soft, 0.30 + 0.42 * cam, 0.0)
        ax.imshow(rgba, interpolation="nearest")
        ax.contour(hard.astype(float), levels=[0.5], colors=["#7F1D1D"], linewidths=0.50)
    ax.set_title(title, loc="left", fontsize=6.9, color=PALETTE["slate"], pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_gallery() -> None:
    info = subject_info()
    plt.rcParams.update({"font.family": "Arial", "font.size": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42})
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.27, 6.10), facecolor="white")
    gs = fig.add_gridspec(len(CASES), 12, wspace=0.055, hspace=0.22)
    axes_names = [("Sag.", 0), ("Cor.", 1), ("Ax.", 2)]
    for r, sid in enumerate(CASES):
        z = np.load(case_npz_path(sid))
        image = z["image"].astype(np.float32)
        seg = z["seg"].astype(np.int16)
        brain = erode(seg > 0)
        image01 = robust01(image, seg > 0)
        cam = normalize_cam(read_nifti_gz_float32(case_cam_path(sid)), brain)
        p_ad = float(info.get(sid, {}).get("p_ad", "nan"))
        score = float(info.get(sid, {}).get("score", "nan"))
        label_ax = fig.add_subplot(gs[r, 0:3])
        label_ax.axis("off")
        label_ax.text(0.02, 0.68, sid, ha="left", va="center", fontsize=8.3, color=PALETTE["ink"], fontweight="bold")
        label_ax.text(0.02, 0.43, f"p(AD)={p_ad:.3f}", ha="left", va="center", fontsize=7.0, color=PALETTE["red"])
        label_ax.text(0.02, 0.22, f"AD logit score={score:.3f}", ha="left", va="center", fontsize=6.1, color=PALETTE["slate"])
        for c, (name, axis) in enumerate(axes_names):
            ax = fig.add_subplot(gs[r, 3 + c * 3 : 6 + c * 3])
            idx = best_slice(cam, axis)
            render(
                ax,
                view_slice(image01, axis, idx),
                view_slice(cam, axis, idx),
                view_slice(brain, axis, idx),
                f"{name} top Grad-CAM",
            )

    fig.suptitle("Server-side model Grad-CAM: crisp voxel evidence before regional aggregation", x=0.045, y=0.975, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(
        0.045,
        0.035,
        "Real trained-model T1 Grad-CAM from server outputs. Display uses eroded brain mask and top-percentile hotspots; regions should be reported after voxel evidence is localized.",
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=PALETTE["slate"],
    )
    fig.subplots_adjust(left=0.045, right=0.990, top=0.91, bottom=0.085)
    fig.savefig(OUT / "Server_model_side_crisp_gradcam_gallery.png", dpi=450)
    fig.savefig(OUT / "Server_model_side_crisp_gradcam_gallery.pdf")
    plt.close(fig)
    print(OUT / "Server_model_side_crisp_gradcam_gallery.png")


if __name__ == "__main__":
    make_gallery()
