#!/usr/bin/env python3
"""Publication-style crisp single-case model-side Grad-CAM figure."""
from __future__ import annotations

import csv
import gzip
import os
import struct
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np

MPL_CACHE = Path(tempfile.gettempdir()) / "a2c_mpl_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageEnhance, ImageFilter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "data/full_cache").exists() else HERE
DL = ROOT / "server_gradcam_download"
OUT = ROOT / "paper/figures" if (ROOT / "paper/figures").exists() else ROOT / "figures_nbe_style"
SID = "DEMO_0007"
NPZ = next(
    p
    for p in [
        DL / "best_screened/DEMO_0007.npz",
        ROOT / "data/full_cache/DEMO_0007.npz",
    ]
    if p.exists()
)
CAM = next(
    p
    for p in [
        DL / "best_screened/DEMO_0007_t1_ad_logit.nii.gz",
        ROOT / "analysis_explainability_seed0_full/gradcam/DEMO_0007_t1_ad_logit.nii.gz",
    ]
    if p.exists()
)
SUBJECTS = next(
    p
    for p in [
        DL / "analysis_explainability_seed0_full/gradcam_subjects.csv",
        ROOT / "analysis_explainability_seed0_full/gradcam_subjects.csv",
    ]
    if p.exists()
)
ROI_CSV = next(
    p
    for p in [
        DL / "analysis_explainability_seed0_full/roi_ablation.csv",
        ROOT / "analysis_explainability_seed0_full/roi_ablation.csv",
    ]
    if p.exists()
)
PALETTE = {
    "ink": "#1B1F23",
    "slate": "#4B5563",
    "muted": "#8A94A3",
    "grid": "#D8DEE6",
    "red": "#C9583B",
    "red_dark": "#7A1F1F",
    "gold": "#E3A53B",
    "blue": "#4878A8",
}


def read_nii(path: Path) -> np.ndarray:
    blob = gzip.open(path, "rb").read()
    dim = struct.unpack("<8h", blob[40:56])
    off = int(struct.unpack("<f", blob[108:112])[0])
    shape = tuple(int(v) for v in dim[1:4])
    return np.frombuffer(blob[off:], dtype="<f4", count=int(np.prod(shape))).copy().reshape(shape, order="F")


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


def norm_cam(cam: np.ndarray, brain: np.ndarray) -> np.ndarray:
    vals = cam[brain]
    lo, hi = np.percentile(vals, [5, 99.9])
    return np.clip((cam - lo) / max(hi - lo, 1e-8), 0, 1) * brain


def best_slice(cam: np.ndarray, axis: int) -> int:
    if axis == 0:
        return int(np.argmax(cam.sum(axis=(1, 2))))
    if axis == 1:
        return int(np.argmax(cam.sum(axis=(0, 2))))
    return int(np.argmax(cam.sum(axis=(0, 1))))


def sl(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return np.rot90(vol[idx])
    if axis == 1:
        return np.rot90(vol[:, idx, :])
    return np.rot90(vol[:, :, idx])


def crop(mask: np.ndarray, pad: int = 7) -> Tuple[slice, slice]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return slice(None), slice(None)
    return (
        slice(max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad)),
        slice(max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad)),
    )


def resize_float(x: np.ndarray, factor: int = 6, resample: int = Image.Resampling.LANCZOS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi <= lo:
        return np.zeros((x.shape[0] * factor, x.shape[1] * factor), dtype=np.float32)
    u8 = np.clip((x - lo) / (hi - lo), 0, 1)
    im = Image.fromarray((u8 * 255).astype(np.uint8))
    im = im.resize((x.shape[1] * factor, x.shape[0] * factor), resample=resample)
    out = np.asarray(im, dtype=np.float32) / 255.0
    return out * (hi - lo) + lo


def resize_mri(x: np.ndarray, factor: int = 7) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0, 1)
    im = Image.fromarray((x * 255).astype(np.uint8))
    im = im.resize((x.shape[1] * factor, x.shape[0] * factor), resample=Image.Resampling.LANCZOS)
    im = ImageEnhance.Contrast(im).enhance(1.12)
    im = ImageEnhance.Sharpness(im).enhance(1.55)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=95, threshold=4))
    return np.asarray(im, dtype=np.float32) / 255.0


def draw(ax, img2d: np.ndarray, cam2d: np.ndarray, brain2d: np.ndarray, title: str, coord: str) -> None:
    sy, sx = crop(brain2d, pad=8)
    img = img2d[sy, sx]
    cam = cam2d[sy, sx]
    brain = brain2d[sy, sx]
    img = np.where(brain, img, 0.0)
    vals = cam[brain]
    th_soft = np.percentile(vals, 97.0)
    th_hard = np.percentile(vals, 99.0)
    soft = ((cam >= th_soft) & brain).astype(float)
    hard = ((cam >= th_hard) & brain).astype(float)
    img_u = resize_mri(np.clip(img**0.86, 0, 1), 7)
    cam_u = resize_float(cam, 7, Image.Resampling.BICUBIC)
    brain_u = resize_float(brain.astype(float), 7, Image.Resampling.LANCZOS)
    soft_u = resize_float(soft, 7, Image.Resampling.BICUBIC)
    hard_u = resize_float(hard, 7, Image.Resampling.NEAREST) > 0.5
    alpha_brain = np.clip((brain_u - 0.06) / 0.28, 0, 1)
    ax.set_facecolor("white")
    ax.imshow(img_u, cmap="gray", vmin=0, vmax=1, alpha=alpha_brain, interpolation="nearest")
    ax.contour(alpha_brain, levels=[0.45], colors=["#C7CED8"], linewidths=0.45)
    cmap = LinearSegmentedColormap.from_list("hotred", ["#F9B7A5", PALETTE["red"]])
    rgba = cmap(np.clip(cam_u, 0, 1))
    rgba[..., 3] = np.clip(soft_u, 0, 1) * (0.20 + 0.58 * np.clip(cam_u, 0, 1)) * alpha_brain
    ax.imshow(rgba, interpolation="nearest")
    ax.contour(hard_u.astype(float), levels=[0.5], colors=[PALETTE["red_dark"]], linewidths=0.82)
    ax.set_title(title, loc="left", fontsize=7.8, color=PALETTE["ink"], pad=3, fontweight="bold")
    ax.text(0.995, 1.012, coord, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color=PALETTE["muted"])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def info_row():
    with SUBJECTS.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["subject_id"] == SID and row["branch"] == "t1":
                return row
    return {}


def roi_summary(cam: np.ndarray, seg: np.ndarray, top_n: int = 10):
    ablation = {}
    if ROI_CSV.exists():
        with ROI_CSV.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["subject_id"] == SID and row["branch"] == "t1":
                    ablation[int(row["roi_id"])] = float(row["delta_p_ad"])
    rows = []
    for roi in sorted(int(v) for v in np.unique(seg) if int(v) > 0):
        m = seg == roi
        vals = cam[m]
        cam_score = float(np.percentile(vals, 98.2)) if vals.size else 0.0
        delta = max(ablation.get(roi, 0.0), 0.0)
        combined = cam_score * (1.0 + 8.0 * delta)
        if cam_score > 0:
            rows.append((roi, cam_score, delta, combined))
    rows.sort(key=lambda r: r[3], reverse=True)
    return rows[:top_n]


def draw_roi_bar(ax, rois) -> None:
    vals = np.array([r[3] for r in rois], dtype=float)
    vals = vals / max(float(vals.max()), 1e-8)
    labels = [f"ROI {r[0]}" for r in rois]
    yy = np.arange(len(labels))[::-1]
    ax.barh(yy, vals, height=0.64, color=PALETTE["red"], alpha=0.88, edgecolor="white", linewidth=0.45)
    for y, v, row in zip(yy, vals, rois):
        ax.plot([v, v], [y - 0.27, y + 0.27], color=PALETTE["red_dark"], linewidth=0.55)
        ax.text(min(v + 0.025, 1.02), y, f"{row[1]:.2f}", va="center", ha="left", fontsize=5.6, color=PALETTE["slate"])
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Voxel CAM weighted by ROI ablation", fontsize=6.2, labelpad=2)
    ax.set_title("Regional readout after voxel localization", loc="left", fontsize=8.0, color=PALETTE["ink"], fontweight="bold", pad=4)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.45, alpha=0.65)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#C7CED8")
    ax.spines["bottom"].set_color("#C7CED8")
    ax.tick_params(axis="x", labelsize=5.7, colors=PALETTE["muted"])


def draw_logic(ax, sid: str, p_ad: float, score: float) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Evidence hierarchy for this panel", loc="left", fontsize=8.0, color=PALETTE["ink"], fontweight="bold", pad=4)
    steps = [
        ("trained model", "server-side AD logit"),
        ("voxel map", "3D Grad-CAM inside brain mask"),
        ("region readout", "ROI ablation used only after localization"),
    ]
    xs = [0.12, 0.48, 0.84]
    colors = [PALETTE["blue"], PALETTE["red"], PALETTE["gold"]]
    for i, ((head, sub), x, color) in enumerate(zip(steps, xs, colors)):
        ax.add_patch(Rectangle((x - 0.11, 0.47), 0.22, 0.18, facecolor="white", edgecolor=color, linewidth=1.0))
        ax.text(x, 0.585, head, ha="center", va="center", fontsize=6.6, color=PALETTE["ink"], fontweight="bold")
        ax.text(x, 0.515, sub, ha="center", va="center", fontsize=5.3, color=PALETTE["slate"], wrap=True)
        if i < 2:
            ax.add_patch(FancyArrowPatch((x + 0.13, 0.56), (xs[i + 1] - 0.13, 0.56), arrowstyle="-|>", mutation_scale=7, linewidth=0.75, color=PALETTE["muted"]))
    ax.text(0.02, 0.31, f"Subject {sid}    p(AD)={p_ad:.3f}    AD-logit score={score:.3f}", ha="left", va="center", fontsize=6.4, color=PALETTE["slate"])
    ax.text(0.02, 0.17, "Interpretation: dense attribution evidence is displayed first; region labels are a secondary summary, avoiding atlas-colored blobs as the primary localization.", ha="left", va="center", fontsize=5.9, color=PALETTE["slate"], wrap=True)


def make() -> None:
    z = np.load(NPZ)
    image = z["image"].astype(np.float32)
    seg = z["seg"].astype(np.int16)
    display_brain = close_mask(np.abs(image) > 1e-3, 1)
    evidence_brain = erode(display_brain, 1)
    img = robust01(image, display_brain)
    cam = norm_cam(read_nii(CAM), evidence_brain)
    meta = info_row()
    p_ad = float(meta.get("p_ad", "nan"))
    score = float(meta.get("score", "nan"))
    rois = roi_summary(cam, seg, 10)

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(8.27, 5.15), facecolor="white")
    gs = fig.add_gridspec(2, 12, height_ratios=[1.55, 0.78], hspace=0.16, wspace=0.17)
    for c, (name, axis, letter) in enumerate([("Sagittal", 0, "x"), ("Coronal", 1, "y"), ("Axial", 2, "z")]):
        ax = fig.add_subplot(gs[0, c * 4 : (c + 1) * 4])
        idx = best_slice(cam, axis)
        draw(ax, sl(img, axis, idx), sl(cam, axis, idx), sl(display_brain, axis, idx), name, f"{letter}={idx}")
    ax_roi = fig.add_subplot(gs[1, 0:5])
    draw_roi_bar(ax_roi, rois)
    ax_logic = fig.add_subplot(gs[1, 5:12])
    draw_logic(ax_logic, SID, p_ad, score)
    fig.suptitle("Pixel-first A2C model evidence from the trained server model", x=0.060, y=0.985, ha="left", fontsize=10.6, fontweight="bold")
    fig.text(0.060, 0.030, "Top-percentile AD-logit Grad-CAM hotspots are masked to eroded brain tissue; ROI bars summarize the same voxel evidence with ablation support. Attribution evidence only, not lesion ground truth.", fontsize=5.95, color=PALETTE["slate"], ha="left")
    fig.subplots_adjust(left=0.060, right=0.990, top=0.910, bottom=0.115)
    fig.savefig(OUT / "Best_case_server_gradcam_publication_crisp_v2.png", dpi=720)
    fig.savefig(OUT / "Best_case_server_gradcam_publication_crisp_v2.pdf")
    plt.close(fig)
    print(OUT / "Best_case_server_gradcam_publication_crisp_v2.png")


if __name__ == "__main__":
    make()
