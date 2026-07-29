#!/usr/bin/env python3
"""Prototype dense A2C evidence overlays from region-level ATE.

This is a visualization prototype, not a new voxel-supervised model. It maps
the existing 21-region AD ATE values back into the atlas mask, then creates a
smoothed and MRI-edge-modulated dense evidence field for visual inspection.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

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

PALETTE = {
    "ink": "#1B1F23",
    "slate": "#4B5563",
    "muted": "#8A94A3",
    "grid": "#D8DEE6",
    "red": "#C9583B",
    "blue": "#2E6F9E",
}


def normalize_image(im: np.ndarray) -> np.ndarray:
    im = np.asarray(im, dtype=float)
    lo, hi = np.percentile(im[im > 0], [1, 99]) if np.any(im > 0) else (im.min(), im.max())
    return np.clip((im - lo) / max(hi - lo, 1e-8), 0, 1)


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x**2) / (2 * sigma * sigma))
    return k / k.sum()


def convolve1d_nearest(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    out = np.zeros_like(arr, dtype=float)
    n = arr.shape[axis]
    offsets = np.arange(-(len(kernel) // 2), len(kernel) // 2 + 1)
    base = np.arange(n)
    for w, off in zip(kernel, offsets):
        idx = np.clip(base + off, 0, n - 1)
        out += float(w) * np.take(arr, idx, axis=axis)
    return out


def smooth3d(arr: np.ndarray, sigma: float) -> np.ndarray:
    k = gaussian_kernel1d(sigma)
    out = np.asarray(arr, dtype=float)
    for axis in range(3):
        out = convolve1d_nearest(out, k, axis)
    return out


def read_ad_ate(path: Path) -> Dict[int, float]:
    out: Dict[int, float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["region_idx"])] = float(row["ate_AD"])
    return out


def build_evidence_fields() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(NPZ)
    image = normalize_image(z["image"])
    seg = np.asarray(z["seg"])
    brain = (seg > 0).astype(float)
    ad_ate = read_ad_ate(ATE)

    raw = np.zeros_like(image, dtype=float)
    for region_idx, fs_label in enumerate(FS_LABELS):
        raw[seg == fs_label] = ad_ate.get(region_idx, 0.0)
    raw = raw / max(float(np.max(np.abs(raw))), 1e-8)

    smooth_num = smooth3d(raw * brain, sigma=1.35)
    smooth_den = smooth3d(brain, sigma=1.35)
    smoothed = np.where(smooth_den > 0.05, smooth_num / np.maximum(smooth_den, 1e-8), 0.0)

    blurred_img = smooth3d(image * brain, sigma=1.05)
    saliency = np.abs(image - blurred_img) * brain
    p98 = float(np.percentile(saliency[brain > 0], 98)) if np.any(brain > 0) else 1.0
    saliency = np.clip(saliency / max(p98, 1e-8), 0, 1)
    dense = smoothed * (0.62 + 0.50 * saliency)
    dense = np.clip(dense, -1, 1) * brain

    return image, seg, raw * brain, smoothed * brain, dense


def view_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return np.rot90(vol[idx, :, :])
    if axis == 1:
        return np.rot90(vol[:, idx, :])
    return np.rot90(vol[:, :, idx])


def best_slice(field: np.ndarray, seg: np.ndarray, axis: int) -> int:
    score = np.abs(field)
    if axis == 0:
        return int(np.argmax(score.sum(axis=(1, 2)) + 0.08 * (seg > 0).sum(axis=(1, 2))))
    if axis == 1:
        return int(np.argmax(score.sum(axis=(0, 2)) + 0.08 * (seg > 0).sum(axis=(0, 2))))
    return int(np.argmax(score.sum(axis=(0, 1)) + 0.08 * (seg > 0).sum(axis=(0, 1))))


def overlay(ax, image2d: np.ndarray, field2d: np.ndarray, title: str, annotate: str | None = None) -> None:
    ax.imshow(image2d, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    alpha = np.clip(np.abs(field2d) ** 0.72, 0, 1) * 0.78
    ax.imshow(field2d, cmap="RdBu_r", vmin=-1, vmax=1, alpha=alpha, interpolation="bilinear")
    ax.set_title(title, loc="left", fontsize=8.0, color=PALETTE["slate"], pad=3)
    if annotate:
        ax.text(
            0.03,
            0.96,
            annotate,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color=PALETTE["ink"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_comparison() -> None:
    image, seg, raw, smoothed, dense = build_evidence_fields()
    OUT.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    view_names = [("Sagittal", 0), ("Coronal", 1), ("Axial", 2)]
    fields = [
        ("Current atlas ATE", raw, "21-region constant value"),
        ("Smoothed atlas evidence", smoothed, "boundary-constrained smoothing"),
        ("Dense evidence prototype", dense, "MRI-edge modulation"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(8.3, 7.4), facecolor="white")
    for r, (view_name, axis) in enumerate(view_names):
        idx = best_slice(dense, seg, axis)
        for c, (title, field, note) in enumerate(fields):
            overlay(
                axes[r, c],
                view_slice(image, axis, idx),
                view_slice(field, axis, idx),
                f"{view_name}: {title}",
                note if r == 0 else None,
            )

    cax = fig.add_axes([0.22, 0.055, 0.56, 0.018])
    cmap = LinearSegmentedColormap.from_list("signed", [PALETTE["blue"], "#FFFFFF", PALETTE["red"]])
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1)), cax=cax, orientation="horizontal")
    cb.set_label("Standardized AD-direction evidence (not voxel-level lesion ground truth)", fontsize=7.0)
    cb.ax.tick_params(labelsize=6.2, length=2)
    fig.suptitle("A2C dense evidence prototype from region-level AD ATE", x=0.035, y=0.985, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(
        0.035,
        0.020,
        "Prototype only: existing 21-region ATE values are redistributed through atlas masks, anatomical smoothing and MRI-edge saliency. No pixel-level supervision or lesion labels are used.",
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=PALETTE["slate"],
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.935, bottom=0.105, wspace=0.035, hspace=0.185)
    fig.savefig(OUT / "Dense_A2C_evidence_prototype_comparison.png", dpi=360)
    fig.savefig(OUT / "Dense_A2C_evidence_prototype_comparison.pdf")
    plt.close(fig)

    sag_idx = best_slice(dense, seg, 0)
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 3.05), facecolor="white")
    for ax, (title, field, note) in zip(axes, fields):
        overlay(ax, view_slice(image, 0, sag_idx), view_slice(field, 0, sag_idx), title, note)
    fig.suptitle("Sagittal fusion: atlas-level ATE versus dense evidence prototype", x=0.035, y=0.96, ha="left", fontsize=10.0, fontweight="bold")
    fig.text(
        0.035,
        0.030,
        "Dense map is an atlas-constrained visualization prototype; formal pixel/patch-level claims require model-side attribution and validation.",
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=PALETTE["slate"],
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.82, bottom=0.11, wspace=0.035)
    fig.savefig(OUT / "Dense_A2C_evidence_sagittal_prototype.png", dpi=360)
    fig.savefig(OUT / "Dense_A2C_evidence_sagittal_prototype.pdf")
    plt.close(fig)

    print(OUT / "Dense_A2C_evidence_prototype_comparison.png")
    print(OUT / "Dense_A2C_evidence_sagittal_prototype.png")


if __name__ == "__main__":
    make_comparison()
