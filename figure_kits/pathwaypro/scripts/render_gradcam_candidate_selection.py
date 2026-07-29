#!/usr/bin/env python3
"""Candidate gallery for choosing a publication-grade model-side Grad-CAM case."""
from __future__ import annotations

import csv
import gzip
import os
import struct
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

MPL_CACHE = Path(tempfile.gettempdir()) / "a2c_mpl_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "data/full_cache").exists() else HERE
DL = ROOT / "server_gradcam_download"
OUT = ROOT / "paper/figures" if (ROOT / "paper/figures").exists() else ROOT / "figures_nbe_style"
CASES = ["DEMO_0002", "DEMO_0003", "DEMO_0006", "DEMO_0005", "DEMO_0004"]
PALETTE = {
    "ink": "#1B1F23",
    "slate": "#4B5563",
    "muted": "#8A94A3",
    "red": "#C9583B",
    "red_dark": "#7A1F1F",
    "blue": "#4878A8",
}


def read_nii(path: Path) -> np.ndarray:
    blob = gzip.open(path, "rb").read()
    dim = struct.unpack("<8h", blob[40:56])
    off = int(struct.unpack("<f", blob[108:112])[0])
    shape = tuple(int(v) for v in dim[1:4])
    return np.frombuffer(blob[off:], dtype="<f4", count=int(np.prod(shape))).copy().reshape(shape, order="F")


def case_npz(sid: str) -> Path:
    for p in [
        DL / "top_cases/full_cache" / f"{sid}.npz",
        DL / "top_cases" / f"{sid}.npz",
        DL / "data/full_cache" / f"{sid}.npz",
        ROOT / "data/full_cache" / f"{sid}.npz",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(sid)


def case_cam(sid: str) -> Path:
    for p in [
        DL / "top_cases/gradcam" / f"{sid}_t1_ad_logit.nii.gz",
        DL / "top_cases" / f"{sid}_t1_ad_logit.nii.gz",
        DL / "analysis_explainability_seed0_full/gradcam" / f"{sid}_t1_ad_logit.nii.gz",
        ROOT / "analysis_explainability_seed0_full/gradcam" / f"{sid}_t1_ad_logit.nii.gz",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(sid)


def erode(mask: np.ndarray, n: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(n):
        nxt = out.copy()
        nxt[1:] &= out[:-1]
        nxt[:-1] &= out[1:]
        nxt[:, 1:] &= out[:, :-1]
        nxt[:, :-1] &= out[:, 1:]
        nxt[:, :, 1:] &= out[:, :, :-1]
        nxt[:, :, :-1] &= out[:, :, 1:]
        out = nxt
    return out


def dilate(mask: np.ndarray, n: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(n):
        nxt = out.copy()
        nxt[1:] |= out[:-1]
        nxt[:-1] |= out[1:]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        nxt[:, :, 1:] |= out[:, :, :-1]
        nxt[:, :, :-1] |= out[:, :, 1:]
        out = nxt
    return out


def close_mask(mask: np.ndarray, n: int = 1) -> np.ndarray:
    return erode(dilate(mask, n), n)


def robust01(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = x[mask]
    lo, hi = np.percentile(vals, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)


def normalize_cam(cam: np.ndarray, brain: np.ndarray) -> np.ndarray:
    vals = cam[brain]
    lo, hi = np.percentile(vals, [5, 99.9])
    return np.clip((cam - lo) / max(hi - lo, 1e-8), 0, 1) * brain


def best_slice(cam: np.ndarray, axis: int) -> int:
    if axis == 0:
        return int(np.argmax(cam.sum(axis=(1, 2))))
    if axis == 1:
        return int(np.argmax(cam.sum(axis=(0, 2))))
    return int(np.argmax(cam.sum(axis=(0, 1))))


def view(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return np.rot90(vol[idx])
    if axis == 1:
        return np.rot90(vol[:, idx, :])
    return np.rot90(vol[:, :, idx])


def crop(mask: np.ndarray, pad: int = 8) -> Tuple[slice, slice]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return slice(None), slice(None)
    return (
        slice(max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad)),
        slice(max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad)),
    )


def resize_float(x: np.ndarray, factor: int = 5, resample: int = Image.Resampling.LANCZOS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi <= lo:
        return np.zeros((x.shape[0] * factor, x.shape[1] * factor), dtype=np.float32)
    u8 = np.clip((x - lo) / (hi - lo), 0, 1)
    im = Image.fromarray((u8 * 255).astype(np.uint8))
    im = im.resize((x.shape[1] * factor, x.shape[0] * factor), resample=resample)
    return np.asarray(im, dtype=np.float32) / 255.0 * (hi - lo) + lo


def draw(ax, image2d: np.ndarray, cam2d: np.ndarray, brain2d: np.ndarray, title: str) -> None:
    sy, sx = crop(brain2d, 7)
    img = np.where(brain2d[sy, sx], image2d[sy, sx], 0.0)
    cam = cam2d[sy, sx]
    brain = brain2d[sy, sx]
    vals = cam[brain]
    img_u = resize_float(np.clip(img**0.84, 0, 1), 5, Image.Resampling.LANCZOS)
    brain_u = resize_float(brain.astype(float), 5, Image.Resampling.LANCZOS)
    cam_u = resize_float(cam, 5, Image.Resampling.BICUBIC)
    alpha_brain = np.clip((brain_u - 0.06) / 0.28, 0, 1)
    ax.imshow(img_u, cmap="gray", vmin=0, vmax=1, alpha=alpha_brain, interpolation="nearest")
    ax.contour(alpha_brain, levels=[0.45], colors=["#C7CED8"], linewidths=0.34)
    if vals.size and float(vals.max()) > 0:
        soft = cam >= np.percentile(vals, 97.4)
        hard = cam >= np.percentile(vals, 99.0)
        soft_u = resize_float((soft & brain).astype(float), 5, Image.Resampling.BICUBIC)
        hard_u = resize_float((hard & brain).astype(float), 5, Image.Resampling.NEAREST) > 0.5
        cmap = LinearSegmentedColormap.from_list("hotred", ["#F9B7A5", PALETTE["red"]])
        rgba = cmap(np.clip(cam_u, 0, 1))
        rgba[..., 3] = np.clip(soft_u, 0, 1) * (0.18 + 0.58 * np.clip(cam_u, 0, 1)) * alpha_brain
        ax.imshow(rgba, interpolation="nearest")
        ax.contour(hard_u.astype(float), levels=[0.5], colors=[PALETTE["red_dark"]], linewidths=0.58)
    ax.set_title(title, loc="left", fontsize=6.7, color=PALETTE["slate"], pad=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def subject_info() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    path = next(
        p
        for p in [
            DL / "analysis_explainability_seed0_full/gradcam_subjects.csv",
            ROOT / "analysis_explainability_seed0_full/gradcam_subjects.csv",
        ]
        if p.exists()
    )
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["branch"] == "t1":
                out[row["subject_id"]] = row
    return out


def make() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    info = subject_info()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(8.27, 7.25), facecolor="white")
    gs = fig.add_gridspec(len(CASES), 13, wspace=0.08, hspace=0.18)
    for r, sid in enumerate(CASES):
        z = np.load(case_npz(sid))
        image = z["image"].astype(np.float32)
        display_brain = close_mask(np.abs(image) > 1e-3, 1)
        evidence_brain = erode(display_brain, 1)
        image01 = robust01(image, display_brain)
        cam = normalize_cam(read_nii(case_cam(sid)), evidence_brain)
        p_ad = float(info.get(sid, {}).get("p_ad", "nan"))
        score = float(info.get(sid, {}).get("score", "nan"))
        ax_lab = fig.add_subplot(gs[r, 0:3])
        ax_lab.axis("off")
        ax_lab.text(0.02, 0.70, sid, ha="left", va="center", fontsize=8.0, color=PALETTE["ink"], fontweight="bold")
        ax_lab.text(0.02, 0.48, f"p(AD)={p_ad:.3f}", ha="left", va="center", fontsize=6.5, color=PALETTE["red"])
        ax_lab.text(0.02, 0.30, f"logit={score:.3f}", ha="left", va="center", fontsize=5.8, color=PALETTE["slate"])
        ax_lab.text(0.02, 0.12, "real server Grad-CAM", ha="left", va="center", fontsize=5.5, color=PALETTE["muted"])
        for c, (name, axis, letter) in enumerate([("Sag", 0, "x"), ("Cor", 1, "y"), ("Ax", 2, "z")]):
            idx = best_slice(cam, axis)
            ax = fig.add_subplot(gs[r, 3 + c * 3 : 6 + c * 3])
            draw(ax, view(image01, axis, idx), view(cam, axis, idx), view(display_brain, axis, idx), f"{name} {letter}={idx}")
    fig.suptitle("Candidate real server-side voxel Grad-CAM cases for the A2C figure", x=0.035, y=0.984, ha="left", fontsize=10.2, fontweight="bold")
    fig.text(0.035, 0.030, "White page, continuous T1-derived brain mask, top-percentile AD-logit Grad-CAM constrained to eroded brain tissue.", ha="left", fontsize=5.9, color=PALETTE["slate"])
    fig.subplots_adjust(left=0.035, right=0.990, top=0.940, bottom=0.065)
    fig.savefig(OUT / "Server_gradcam_candidate_selection_crisp.png", dpi=650)
    fig.savefig(OUT / "Server_gradcam_candidate_selection_crisp.pdf")
    plt.close(fig)
    print(OUT / "Server_gradcam_candidate_selection_crisp.png")


if __name__ == "__main__":
    make()
