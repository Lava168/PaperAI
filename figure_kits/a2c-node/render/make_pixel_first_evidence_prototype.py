#!/usr/bin/env python3
"""Pixel-first A2C evidence prototype.

This prototype reverses the previous visualization direction:

1. Build a constrained voxel-level evidence candidate map from MRI texture,
   hippocampal/amygdala/ventricular neighborhoods and brain-mask erosion.
2. Aggregate the voxel evidence back to atlas regions for regional reporting.

It is still not a trained voxel-wise A2C model. It is a safer prototype for
testing whether dense localization can be made anatomically plausible before
investing in patch-level model training.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
NPZ = ROOT / "a2c_remote_results/chapter1_foundation/sample_data/cache_real/ADNI_002_S_0413_bl_I40657.npz"
ATE = ROOT / "a2c_remote_results/runs/v4_mm_analysis_npj_5seed/ate_21regions.csv"
OUT = ROOT / "figures_nbe_style"

FS_LABELS = [2, 3, 4, 10, 11, 12, 13, 16, 17, 18, 26, 41, 42, 43, 49, 50, 51, 52, 53, 54, 58]
REGION_SHORT = [
    "L-WM", "L-Ctx", "L-LV", "L-Thal", "L-Caud", "L-Put", "L-Pall",
    "Stem", "L-Hip", "L-Amyg", "L-Acc", "R-WM", "R-Ctx", "R-LV",
    "R-Thal", "R-Caud", "R-Put", "R-Pall", "R-Hip", "R-Amyg", "R-Acc",
]
HIP_AMYG = {17, 18, 53, 54}
VENTRICLES = {4, 43}
CORE_LABELS = HIP_AMYG | VENTRICLES

PALETTE = {
    "ink": "#1B1F23",
    "slate": "#4B5563",
    "muted": "#8A94A3",
    "grid": "#D8DEE6",
    "red": "#C9583B",
    "gold": "#E3A53B",
    "green": "#4C9A78",
}


def normalize_image(im: np.ndarray) -> np.ndarray:
    im = np.asarray(im, dtype=float)
    mask = im > 0
    lo, hi = np.percentile(im[mask], [1, 99]) if np.any(mask) else (float(im.min()), float(im.max()))
    return np.clip((im - lo) / max(hi - lo, 1e-8), 0, 1)


def normalize01(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if mask is None:
        vals = x[np.isfinite(x)]
    else:
        vals = x[mask & np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x)
    lo, hi = np.percentile(vals, [2, 98])
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)


def shift_bool(mask: np.ndarray, axis: int, offset: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    if axis == 0:
        if offset > 0:
            out[offset:, :, :] = mask[:-offset, :, :]
        else:
            out[:offset, :, :] = mask[-offset:, :, :]
    elif axis == 1:
        if offset > 0:
            out[:, offset:, :] = mask[:, :-offset, :]
        else:
            out[:, :offset, :] = mask[:, -offset:, :]
    else:
        if offset > 0:
            out[:, :, offset:] = mask[:, :, :-offset]
        else:
            out[:, :, :offset] = mask[:, :, -offset:]
    return out


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        nxt = out.copy()
        for axis in range(3):
            nxt |= shift_bool(out, axis, 1)
            nxt |= shift_bool(out, axis, -1)
        out = nxt
    return out


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        nxt = out.copy()
        for axis in range(3):
            nxt &= shift_bool(out, axis, 1)
            nxt &= shift_bool(out, axis, -1)
        out = nxt
    return out


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    grads = []
    for axis in range(3):
        g = np.zeros_like(image, dtype=float)
        sl = [slice(None)] * 3
        slp = [slice(None)] * 3
        slm = [slice(None)] * 3
        sl[axis] = slice(1, -1)
        slp[axis] = slice(2, None)
        slm[axis] = slice(None, -2)
        g[tuple(sl)] = 0.5 * (image[tuple(slp)] - image[tuple(slm)])
        grads.append(g)
    return np.sqrt(sum(g * g for g in grads))


def read_ad_ate(path: Path) -> Dict[int, float]:
    out: Dict[int, float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["region_idx"])] = float(row["ate_AD"])
    return out


def label_mask(seg: np.ndarray, labels: Iterable[int]) -> np.ndarray:
    out = np.zeros_like(seg, dtype=bool)
    for label in labels:
        out |= seg == label
    return out


def build_pixel_first_map() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, float]]]:
    z = np.load(NPZ)
    image = normalize_image(z["image"])
    seg = np.asarray(z["seg"])
    brain = seg > 0
    safe_brain = erode(brain, iterations=1)

    hip_amyg = label_mask(seg, HIP_AMYG)
    ventricles = label_mask(seg, VENTRICLES)
    core = dilate(hip_amyg | ventricles, iterations=7) & safe_brain
    hip_rim = dilate(hip_amyg, iterations=3) & safe_brain
    vent_rim = dilate(ventricles, iterations=3) & safe_brain

    grad = normalize01(gradient_magnitude(image), safe_brain)
    dark = normalize01(1.0 - image, safe_brain)

    ate = read_ad_ate(ATE)
    ate_prior = np.zeros_like(image, dtype=float)
    for idx, fs_label in enumerate(FS_LABELS):
        ate_prior[seg == fs_label] = max(0.0, ate.get(idx, 0.0))
    ate_prior = normalize01(ate_prior, brain)

    score = np.zeros_like(image, dtype=float)
    score += 0.42 * grad * core
    score += 0.34 * dark * vent_rim
    score += 0.34 * grad * hip_rim
    score *= (0.62 + 0.38 * ate_prior)
    score *= safe_brain

    nonzero = score[score > 0]
    if nonzero.size:
        threshold = np.percentile(nonzero, 76)
        score = np.where(score >= threshold, score, 0.0)
        score = normalize01(score, score > 0)
    score *= safe_brain

    rows: List[Tuple[str, float]] = []
    for idx, fs_label in enumerate(FS_LABELS):
        m = seg == fs_label
        if not np.any(m):
            continue
        vals = score[m]
        if np.any(vals > 0):
            # Use mean of active voxels so sparse local evidence is not washed
            # out by large cortical or white-matter regions.
            val = float(vals[vals > 0].mean())
        else:
            val = 0.0
        rows.append((REGION_SHORT[idx], val))
    rows.sort(key=lambda x: x[1], reverse=True)
    return image, seg, score, core, rows


def view_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return np.rot90(vol[idx, :, :])
    if axis == 1:
        return np.rot90(vol[:, idx, :])
    return np.rot90(vol[:, :, idx])


def best_slice(score: np.ndarray, axis: int) -> int:
    if axis == 0:
        return int(np.argmax(score.sum(axis=(1, 2))))
    if axis == 1:
        return int(np.argmax(score.sum(axis=(0, 2))))
    return int(np.argmax(score.sum(axis=(0, 1))))


def overlay_pixel_map(ax, image2d: np.ndarray, score2d: np.ndarray, title: str) -> None:
    ax.imshow(image2d, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    alpha = np.where(score2d > 0, 0.20 + 0.70 * score2d, 0.0)
    cmap = LinearSegmentedColormap.from_list("pixelred", ["#FFFFFF", "#F2B6A8", PALETTE["red"]])
    ax.imshow(score2d, cmap=cmap, vmin=0, vmax=1, alpha=alpha, interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=8.2, color=PALETTE["slate"], pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_figure() -> None:
    image, seg, score, core, rows = build_pixel_first_map()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(8.27, 6.4), facecolor="white")
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 0.74], hspace=0.26, wspace=0.36)
    views = [("Sagittal", 0), ("Coronal", 1), ("Axial", 2)]
    for i, (name, axis) in enumerate(views):
        ax = fig.add_subplot(gs[0, i * 4 : (i + 1) * 4])
        idx = best_slice(score, axis)
        overlay_pixel_map(ax, view_slice(image, axis, idx), view_slice(score, axis, idx), f"{name} pixel-first AD evidence")

    ax_bar = fig.add_subplot(gs[1, 0:5])
    top = rows[:8]
    names = [r[0] for r in top][::-1]
    vals = np.array([r[1] for r in top][::-1])
    y = np.arange(len(top))
    colors = [PALETTE["red"] if ("Hip" in n or "Amyg" in n) else PALETTE["gold"] if "LV" in n else PALETTE["green"] for n in names]
    ax_bar.barh(y, vals, color=colors, edgecolor="white", linewidth=0.4)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names, fontsize=6.2)
    ax_bar.set_xlabel("Mean active voxel evidence", labelpad=1)
    ax_bar.set_title("Region summary after pixel-level localization", loc="left", fontsize=8.2, color=PALETTE["slate"], pad=3)
    for side in ["top", "right"]:
        ax_bar.spines[side].set_visible(False)
    ax_bar.spines["left"].set_color(PALETTE["slate"])
    ax_bar.spines["bottom"].set_color(PALETTE["slate"])
    ax_bar.tick_params(axis="x", labelsize=6.2)

    ax_logic = fig.add_subplot(gs[1, 5:12])
    ax_logic.set_xlim(0, 1)
    ax_logic.set_ylim(0, 1)
    ax_logic.axis("off")
    steps = [
        ("1", "Voxel candidates", "MRI gradient + dark CSF boundary"),
        ("2", "Anatomical guardrail", "eroded brain mask + hippocampus/ventricle neighborhood"),
        ("3", "Regional readout", "aggregate active voxels to atlas regions"),
    ]
    xs = [0.12, 0.50, 0.88]
    for i, ((num, head, body), x) in enumerate(zip(steps, xs)):
        ax_logic.scatter(x, 0.66, s=360, color="#FFFFFF", edgecolor=PALETTE["red"], linewidth=1.0, zorder=3)
        ax_logic.text(x, 0.66, num, ha="center", va="center", fontsize=8.4, color=PALETTE["red"], fontweight="bold")
        ax_logic.text(x, 0.43, head, ha="center", va="center", fontsize=7.1, color=PALETTE["ink"], fontweight="bold")
        ax_logic.text(x, 0.25, body, ha="center", va="center", fontsize=6.0, color=PALETTE["slate"], wrap=True)
        if i < len(xs) - 1:
            ax_logic.annotate("", xy=(xs[i + 1] - 0.075, 0.66), xytext=(x + 0.075, 0.66), arrowprops=dict(arrowstyle="->", lw=0.8, color="#AAB4C0"))
    ax_logic.text(0.02, 0.93, "Interpretation boundary", ha="left", va="top", fontsize=8.2, color=PALETTE["slate"], fontweight="bold")
    ax_logic.text(
        0.02,
        0.08,
        "Prototype only: this is a constrained dense evidence map for figure design. It should be written as patch/voxel evidence only after model-side attribution or patch-level training is added.",
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=PALETTE["slate"],
        wrap=True,
    )

    fig.suptitle("Pixel-first dense localization prototype for A2C explanations", x=0.035, y=0.975, ha="left", fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.08)
    fig.savefig(OUT / "Pixel_first_A2C_dense_localization_prototype.png", dpi=360)
    fig.savefig(OUT / "Pixel_first_A2C_dense_localization_prototype.pdf")
    plt.close(fig)
    print(OUT / "Pixel_first_A2C_dense_localization_prototype.png")


if __name__ == "__main__":
    make_figure()
