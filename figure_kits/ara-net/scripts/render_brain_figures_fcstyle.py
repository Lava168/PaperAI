#!/usr/bin/env python3
"""Render fcHMRF-comparable ARA-Net brain figures (B1–B4).

Requires assets from build_brain_figure_assets.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import nibabel as nib
    from nilearn import plotting
    from nilearn.image import new_img_like
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"nilearn/nibabel required: {exc}")

from brain_figure_align import (  # noqa: E402
    prepare_anat_underlay,
    prepare_glass_map,
    plot_native_ortho_glass,
)
from train_atlas_feature_baseline import REGION_NAMES  # noqa: E402


METHOD_COLUMNS = [
    "atrophy_AD_mean",
    "atrophy_AD_minus_CN",
    "mrf_q_AD_mean",
    "mrf_q_AD_minus_CN",
    "mrf_q_MCI_minus_CN",
    "assoc_v6_rcspe",
    "assoc_v7_mrf",
    "assoc_equal_logpool",
]

METHOD_TITLES = {
    "atrophy_AD_mean": "Atrophy\nAD mean",
    "atrophy_AD_minus_CN": "Atrophy\nAD−CN",
    "mrf_q_AD_mean": "MRF q\nAD mean",
    "mrf_q_AD_minus_CN": "MRF q\nAD−CN",
    "mrf_q_MCI_minus_CN": "MRF q\nMCI−CN",
    "assoc_v6_rcspe": "Assoc\nV6 RC-SPE",
    "assoc_v7_mrf": "Assoc\nV7+MRF",
    "assoc_equal_logpool": "Assoc\nequal pool",
}

CUTS = {
    "z": (-28, -12, 0, 12, 28, 40),  # axial
    "y": (-52, -28, -8, 12, 32),  # coronal
    "x": (-36, -18, 0, 18, 36),  # sagittal
}

COLOR_LIMITS = {
    "atrophy_AD_mean": (-2.5, 2.5, "cold_hot"),
    "atrophy_AD_minus_CN": (-2.5, 2.5, "cold_hot"),
    "mrf_q_AD_mean": (0.0, 1.0, "plasma"),
    "mrf_q_AD_minus_CN": (-0.6, 0.6, "cold_hot"),
    "mrf_q_MCI_minus_CN": (-0.6, 0.6, "cold_hot"),
    "assoc_v6_rcspe": (-0.6, 0.6, "cold_hot"),
    "assoc_v7_mrf": (-0.6, 0.6, "cold_hot"),
    "assoc_equal_logpool": (-0.6, 0.6, "cold_hot"),
}


def load_map(maps_dir: Path, name: str):
    path = maps_dir / f"{name}.nii.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def plot_mosaic_on_ax(ax, img, display_mode: str, cut_coords, vmin, vmax, cmap: str) -> None:
    # nilearn paints into current figure; we redirect by creating display with axes
    display = plotting.plot_stat_map(
        img,
        display_mode=display_mode,
        cut_coords=cut_coords,
        axes=ax,
        colorbar=False,
        annotate=False,
        draw_cross=False,
        black_bg=True,
        bg_img=None,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        threshold=1e-6 if vmin >= 0 else None,
        radiological=False,
    )
    display.close = lambda: None  # keep axes


def render_b1(maps_dir: Path, out_path: Path) -> None:
    methods = [m for m in METHOD_COLUMNS if (maps_dir / f"{m}.nii.gz").exists()]
    n_m = len(methods)
    fig = plt.figure(figsize=(2.15 * n_m + 1.2, 9.2), dpi=220)
    gs = GridSpec(3, n_m, figure=fig, wspace=0.08, hspace=0.18)

    views = [
        ("z", "Axial", CUTS["z"]),
        ("y", "Coronal", CUTS["y"]),
        ("x", "Sagittal", CUTS["x"]),
    ]
    for col, method in enumerate(methods):
        img = load_map(maps_dir, method)
        vmin, vmax, cmap = COLOR_LIMITS.get(method, (-1, 1, "cold_hot"))
        for row, (mode, label, cuts) in enumerate(views):
            ax = fig.add_subplot(gs[row, col])
            plotting.plot_stat_map(
                img,
                display_mode=mode,
                cut_coords=cuts,
                axes=ax,
                colorbar=False,
                annotate=False,
                draw_cross=False,
                black_bg=True,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                threshold=None,
            )
            if col == 0:
                ax.set_ylabel(label, fontsize=11, color="white")
            if row == 0:
                ax.set_title(METHOD_TITLES.get(method, method), fontsize=9, color="black", pad=6)

    # shared colorbars for two families
    cax1 = fig.add_axes([0.92, 0.55, 0.015, 0.3])
    sm1 = plt.cm.ScalarMappable(cmap="cold_hot", norm=plt.Normalize(vmin=-2.5, vmax=2.5))
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, cax=cax1)
    cb1.set_label("Atrophy / signed maps", fontsize=8)

    cax2 = fig.add_axes([0.92, 0.15, 0.015, 0.3])
    sm2 = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, cax=cax2)
    cb2.set_label("MRF q (AD mean)", fontsize=8)

    fig.suptitle(
        "Figure B1. ARA-Net spatial maps vs method columns (fcHMRF-style montage)",
        fontsize=13,
        y=0.98,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] B1 -> {out_path.with_suffix('.png')}")


def render_b1_panels(maps_dir: Path, out_path: Path) -> None:
    """Three disease-contrast panels × selected methods (closer to fcHMRF Fig.5 layout)."""
    panel_methods = [
        ("atrophy_AD_minus_CN", "Atrophy AD−CN"),
        ("mrf_q_MCI_minus_CN", "MRF q MCI−CN"),
        ("mrf_q_AD_minus_CN", "MRF q AD−CN"),
        ("assoc_v6_rcspe", "V6 assoc"),
        ("assoc_v7_mrf", "V7 assoc"),
        ("assoc_equal_logpool", "Equal pool"),
    ]
    panel_methods = [p for p in panel_methods if (maps_dir / f"{p[0]}.nii.gz").exists()]
    n = len(panel_methods)
    fig = plt.figure(figsize=(2.3 * n + 0.8, 7.8), dpi=220)
    gs = GridSpec(3, n, figure=fig, wspace=0.06, hspace=0.12)
    views = [("z", "Axial", CUTS["z"]), ("y", "Coronal", CUTS["y"]), ("x", "Sagittal", CUTS["x"])]
    for col, (name, title) in enumerate(panel_methods):
        img = load_map(maps_dir, name)
        vmin, vmax, cmap = COLOR_LIMITS.get(name, (-0.6, 0.6, "cold_hot"))
        for row, (mode, label, cuts) in enumerate(views):
            ax = fig.add_subplot(gs[row, col])
            plotting.plot_stat_map(
                img,
                display_mode=mode,
                cut_coords=cuts,
                axes=ax,
                colorbar=False,
                annotate=False,
                draw_cross=False,
                black_bg=True,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=11,
                    rotation=90,
                )
            if row == 0:
                ax.set_title(title, fontsize=10)
    cax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap="cold_hot", norm=plt.Normalize(vmin=-0.6, vmax=0.6))
    fig.colorbar(sm, cax=cax, label="Signed map intensity")
    fig.suptitle("Figure B1b. Disease-contrast and model-association montages", fontsize=13, y=0.99)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] B1b -> {out_path.with_suffix('.png')}")


def render_b2(mat_csv: Path, out_path: Path) -> None:
    import csv

    with mat_csv.open() as handle:
        reader = csv.reader(handle)
        header = next(reader)
        cols = header[1:]
        rows = []
        names = []
        for line in reader:
            names.append(line[0])
            rows.append([float(x) for x in line[1:]])
    mat = np.asarray(rows, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=220)
    # normalize each row for display readability while keeping separate?
    # Show raw values with diverging scale clipped
    vmax = float(np.nanpercentile(np.abs(mat), 95)) if np.any(np.isfinite(mat)) else 1.0
    vmax = max(vmax, 0.2)
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.0, vmax=max(float(mat.max()), 0.1))
    # For signed rows, overlay is imperfect; split display:
    # Re-draw with row-wise colormap families by blocking
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_title("Figure B2. Methods / stages × ROI matrix (MRF & associations)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Value")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # Better B2: three panels for CN/MCI/AD MRF q + one assoc panel
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), dpi=220, gridspec_kw={"height_ratios": [1.2, 1.0]})
    # panel 1: CN/MCI/AD mrf means
    stage_idx = [names.index(n) for n in ["CN", "MCI", "AD"] if n in names]
    stage_mat = mat[stage_idx]
    im0 = axes[0].imshow(stage_mat, aspect="auto", cmap="plasma", vmin=0, vmax=1)
    axes[0].set_yticks(range(len(stage_idx)))
    axes[0].set_yticklabels([names[i] for i in stage_idx])
    axes[0].set_xticks(range(len(cols)))
    axes[0].set_xticklabels(cols, rotation=90, fontsize=7)
    axes[0].set_title("(a) MRF q group means by ROI")
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01, label="q")

    assoc_names = [n for n in ["assoc_v6", "assoc_v7", "assoc_equal", "AD_minus_CN", "MCI_minus_CN"] if n in names]
    assoc_idx = [names.index(n) for n in assoc_names]
    assoc_mat = mat[assoc_idx]
    lim = float(np.nanpercentile(np.abs(assoc_mat), 98)) if assoc_mat.size else 0.6
    lim = max(lim, 0.2)
    im1 = axes[1].imshow(assoc_mat, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
    axes[1].set_yticks(range(len(assoc_idx)))
    axes[1].set_yticklabels(assoc_names)
    axes[1].set_xticks(range(len(cols)))
    axes[1].set_xticklabels(cols, rotation=90, fontsize=7)
    axes[1].set_title("(b) Effects and model–ROI associations")
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.01, label="signed value")
    fig.suptitle("Figure B2. ROI × method/stage heatmaps", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path.with_name(out_path.stem + "_panels.png"), bbox_inches="tight")
    fig.savefig(out_path.with_name(out_path.stem + "_panels.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] B2 -> {out_path.with_suffix('.png')}")


def render_b3(top20_json: Path, out_path: Path) -> None:
    payload = json.loads(top20_json.read_text(encoding="utf-8"))
    regions = payload["regions"]
    values = payload["values"]
    methods = list(values.keys())
    x = np.arange(len(regions))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=220, sharey=False)
    groups = [
        ("mrf_q_AD_minus_CN", "mrf_q_MCI_minus_CN"),
        ("atrophy_AD_minus_CN",),
        ("assoc_v6_rcspe", "assoc_v7_mrf", "assoc_equal_logpool"),
    ]
    titles = [
        "(a) MRF q effects",
        "(b) Atrophy AD−CN",
        "(c) Model–ROI associations",
    ]
    colors = {
        "mrf_q_AD_minus_CN": "#d62728",
        "mrf_q_MCI_minus_CN": "#ff7f0e",
        "atrophy_AD_minus_CN": "#1f77b4",
        "assoc_v6_rcspe": "#2ca02c",
        "assoc_v7_mrf": "#9467bd",
        "assoc_equal_logpool": "#8c564b",
    }
    for ax, keys, title in zip(axes, groups, titles):
        for key in keys:
            if key not in values:
                continue
            ax.scatter(x, values[key], s=46, label=key, color=colors.get(key, None), zorder=3)
            ax.plot(x, values[key], alpha=0.35, color=colors.get(key, "gray"))
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=90, fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.axhline(0.0, color="gray", lw=0.8)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("Value")
    fig.suptitle("Figure B3. Top-20 ROIs by |MRF q AD−CN| with multi-method overlay", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] B3 -> {out_path.with_suffix('.png')}")


def render_b4(maps_dir: Path, out_path: Path) -> None:
    """Native-space ortho panels (replaces fake-MNI glass brain)."""
    maps = [
        ("mrf_q_AD_mean", "MRF q AD mean", "plasma", 0, 1),
        ("mrf_q_AD_minus_CN", "MRF q AD−CN", "cold_hot", -0.6, 0.6),
        ("assoc_v7_mrf", "V7 association", "cold_hot", -0.6, 0.6),
        ("atrophy_AD_minus_CN", "Atrophy AD−CN", "cold_hot", -2.5, 2.5),
    ]
    maps = [m for m in maps if (maps_dir / f"{m[0]}.nii.gz").exists()]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=220)
    axes = axes.ravel()
    for ax, (name, title, cmap, vmin, vmax) in zip(axes, maps):
        img = load_map(maps_dir, name)
        plot_native_ortho_glass(
            img,
            axes=ax,
            title=title,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            threshold=1e-4 if vmin >= 0 else None,
            colorbar=True,
        )
    for ax in axes[len(maps) :]:
        ax.axis("off")
    fig.suptitle(
        "Figure B4. Native-space ortho summary of key ARA-Net spatial maps",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] B4 -> {out_path.with_suffix('.png')}")


def render_b1b_with_underlay(maps_dir: Path, underlay_path: Path, out_path: Path) -> None:
    """Method montages on black bg (no anat underlay — avoids nilearn/MNI halo)."""
    # Keep signature for CLI compatibility; underlay intentionally unused.
    _ = underlay_path
    bg = None
    panel_methods = [
        ("atrophy_AD_minus_CN", "Atrophy AD−CN", -2.5, 2.5, "cold_hot"),
        ("mrf_q_MCI_minus_CN", "MRF q MCI−CN", -0.6, 0.6, "cold_hot"),
        ("mrf_q_AD_minus_CN", "MRF q AD−CN", -0.6, 0.6, "cold_hot"),
        ("assoc_v6_rcspe", "V6 assoc", -0.6, 0.6, "cold_hot"),
        ("assoc_v7_mrf", "V7 assoc", -0.6, 0.6, "cold_hot"),
        ("assoc_equal_logpool", "Equal pool", -0.6, 0.6, "cold_hot"),
    ]
    panel_methods = [p for p in panel_methods if (maps_dir / f"{p[0]}.nii.gz").exists()]
    n = len(panel_methods)
    fig = plt.figure(figsize=(2.45 * n + 1.0, 8.4), dpi=240)
    gs = GridSpec(3, n, figure=fig, wspace=0.04, hspace=0.10)
    views = [("z", "Axial", CUTS["z"]), ("y", "Coronal", CUTS["y"]), ("x", "Sagittal", CUTS["x"])]
    for col, (name, title, vmin, vmax, cmap) in enumerate(panel_methods):
        img = load_map(maps_dir, name)
        for row, (mode, label, cuts) in enumerate(views):
            ax = fig.add_subplot(gs[row, col])
            kwargs = dict(
                display_mode=mode,
                cut_coords=cuts,
                axes=ax,
                colorbar=False,
                annotate=False,
                draw_cross=False,
                black_bg=True,
                bg_img=None,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                threshold=1e-4,
            )
            plotting.plot_stat_map(img, **kwargs)
            if col == 0:
                ax.text(-0.06, 0.5, label, transform=ax.transAxes, va="center", ha="right", fontsize=11, rotation=90, color="black")
            if row == 0:
                ax.set_title(title, fontsize=10, pad=4)
    cax = fig.add_axes([0.93, 0.22, 0.012, 0.55])
    sm = plt.cm.ScalarMappable(cmap="cold_hot", norm=plt.Normalize(vmin=-0.6, vmax=0.6))
    fig.colorbar(sm, cax=cax, label="Signed intensity")
    fig.suptitle(
        "Figure B1. Method montages (native space, no MNI underlay)",
        fontsize=13,
        y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] B1 underlay -> {out_path.with_suffix('.png')}")


def main() -> None:
    """Delegate to journal suite (Times font, dense cuts, group/model contrasts)."""
    from render_brain_figures_journal_suite import main as journal_main

    journal_main()


if __name__ == "__main__":
    main()
