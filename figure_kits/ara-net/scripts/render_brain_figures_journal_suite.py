#!/usr/bin/env python3
"""Re-render the full brain_figures_fcstyle suite in journal style.

Design goals (user request):
  - Times New Roman / Liberation Serif throughout
  - Clear labels, no overlap
  - Dense axial strips (more cuts, less empty black)
  - Explicit CN / MCI / AD group contrasts
  - Model advantage: V6 RC-SPE vs equal-pool (± V7)
  - Nature-like Okabe–Ito + RdBu_r / YlOrRd / inferno palettes

Usage (from repo root or ARA-Net):
  python3 scripts/render_brain_figures_journal_suite.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from brain_figure_style import (
    CMAP_DIV,
    CMAP_DIV_BLACK,
    CMAP_Q,
    CMAP_SEQ,
    CLIM,
    CUTS_AXIAL_FOCUS,
    CUTS_DENSE,
    OKABE_ITO,
    add_cbar,
    apply_journal_style,
    difference_map,
    load_map,
    panel_label,
    plot_dense_strip,
    save_fig,
    signed_clim,
)

SERIF_FP = FontProperties(
    fname="/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
)
SERIF_BOLD = FontProperties(
    fname="/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
)


def _title(ax, text: str, *, fontsize: float = 9, pad: float = 4) -> None:
    ax.set_title(text, fontsize=fontsize, pad=pad, fontproperties=SERIF_FP)


try:
    import nibabel as nib
except Exception as exc:  # pragma: no cover
    raise SystemExit(exc)


def render_b1_group_and_model(maps_dir: Path, out_path: Path) -> None:
    """Primary figure: staging gradients + model spatial advantage (dense axial)."""
    apply_journal_style()
    # Three blocks × three columns — all axial dense strips.
    rows = [
        [
            ("mrf_q_CN_mean", "MRF q · CN", CMAP_Q, CLIM["q"], OKABE_ITO["CN"]),
            ("mrf_q_MCI_mean", "MRF q · MCI", CMAP_Q, CLIM["q"], OKABE_ITO["MCI"]),
            ("mrf_q_AD_mean", "MRF q · AD", CMAP_Q, CLIM["q"], OKABE_ITO["AD"]),
        ],
        [
            ("mrf_q_MCI_minus_CN", "MRF q · MCI−CN", CMAP_DIV_BLACK, CLIM["signed"], OKABE_ITO["MCI"]),
            ("mrf_q_AD_minus_CN", "MRF q · AD−CN", CMAP_DIV_BLACK, CLIM["signed"], OKABE_ITO["AD"]),
            ("atrophy_AD_minus_CN", "Atrophy · AD−CN", CMAP_DIV_BLACK, CLIM["atrophy"], OKABE_ITO["atrophy"]),
        ],
        [
            ("assoc_equal_logpool", "Assoc · equal pool", CMAP_DIV_BLACK, CLIM["assoc"], OKABE_ITO["equal"]),
            ("assoc_v6_rcspe", "Assoc · V6 RC-SPE (ours)", CMAP_DIV_BLACK, CLIM["assoc"], OKABE_ITO["V6"]),
            ("__delta_v6_equal__", "Advantage · V6 − equal", CMAP_DIV_BLACK, CLIM["delta"], OKABE_ITO["V6"]),
        ],
    ]
    row_titles = [
        "(a) Group means — disease staging along MRF affected posterior",
        "(b) Group contrasts — where MCI/AD diverge from CN",
        "(c) Model–ROI association — locked V6 vs equal-pool baseline",
    ]

    fig = plt.figure(figsize=(15.0, 6.4), dpi=300)
    gs = GridSpec(
        3,
        3,
        figure=fig,
        left=0.05,
        right=0.90,
        top=0.88,
        bottom=0.07,
        wspace=0.04,
        hspace=0.38,
    )
    delta_img = None
    if (maps_dir / "assoc_v6_rcspe.nii.gz").exists() and (maps_dir / "assoc_equal_logpool.nii.gz").exists():
        delta_img = difference_map(maps_dir, "assoc_v6_rcspe", "assoc_equal_logpool")

    for r, cols in enumerate(rows):
        for c, (name, title, cmap, clim, _color) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            vmin, vmax = clim
            if name == "__delta_v6_equal__":
                if delta_img is None:
                    ax.axis("off")
                    _title(ax, title + ' (missing)', fontsize=9, pad=8)
                    continue
                img = delta_img
                vmin, vmax = signed_clim(delta_img, pct=97, floor=0.08)
            else:
                if not (maps_dir / f"{name}.nii.gz").exists():
                    ax.axis("off")
                    _title(ax, title + ' (missing)', fontsize=9, pad=8)
                    continue
                img = load_map(maps_dir, name)
            plot_dense_strip(
                ax,
                img,
                display_mode="z",
                cut_coords=CUTS_AXIAL_FOCUS,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            _title(ax, title, fontsize=9, pad=6)
            if c == 0:
                ax.text(
                    -0.02,
                    0.5,
                    row_titles[r].split(" — ")[0],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=8,
                    clip_on=False,
                )

    add_cbar(fig, CMAP_Q, *CLIM["q"], "MRF q", [0.915, 0.70, 0.012, 0.18])
    add_cbar(fig, CMAP_DIV_BLACK, *CLIM["signed"], "Signed effect", [0.915, 0.40, 0.012, 0.18])
    add_cbar(fig, CMAP_DIV_BLACK, *CLIM["delta"], "V6 − equal", [0.915, 0.12, 0.012, 0.18])

    fig.suptitle(
        "Figure B1. Staging gradients and V6 spatial association advantage",
        fontsize=12,
        y=0.97,
    )
    fig.text(
        0.5,
        0.015,
        "Native atlas space · dense axial series · bg_img=None  |  "
        "V6 column is the locked RC-SPE model; Δ highlights ROIs where V6 exceeds equal-pool association.",
        ha="center",
        fontsize=7,
        color="0.35",
    )
    save_fig(fig, out_path)


def render_b1b_multiplane(maps_dir: Path, out_path: Path) -> None:
    """Secondary: AD−CN / V6 / Δ across axial+coronal+sagittal (dense)."""
    apply_journal_style()
    panels = [
        ("mrf_q_AD_minus_CN", "MRF q AD−CN", CMAP_DIV_BLACK, CLIM["signed"]),
        ("atrophy_AD_minus_CN", "Atrophy AD−CN", CMAP_DIV_BLACK, CLIM["atrophy"]),
        ("assoc_v6_rcspe", "V6 RC-SPE", CMAP_DIV_BLACK, CLIM["assoc"]),
        ("__delta_v6_equal__", "V6 − equal", CMAP_DIV_BLACK, CLIM["delta"]),
    ]
    views = [
        ("z", "Axial", CUTS_DENSE["z"]),
        ("y", "Coronal", CUTS_DENSE["y"]),
        ("x", "Sagittal", CUTS_DENSE["x"]),
    ]
    delta_img = None
    if (maps_dir / "assoc_v6_rcspe.nii.gz").exists() and (maps_dir / "assoc_equal_logpool.nii.gz").exists():
        delta_img = difference_map(maps_dir, "assoc_v6_rcspe", "assoc_equal_logpool")

    n = len(panels)
    fig = plt.figure(figsize=(3.1 * n + 1.2, 8.8), dpi=300)
    gs = GridSpec(3, n, figure=fig, left=0.07, right=0.91, top=0.90, bottom=0.05, wspace=0.05, hspace=0.14)
    for c, (name, title, cmap, clim) in enumerate(panels):
        vmin, vmax = clim
        if name == "__delta_v6_equal__":
            img = delta_img
        else:
            img = load_map(maps_dir, name) if (maps_dir / f"{name}.nii.gz").exists() else None
        for r, (mode, label, cuts) in enumerate(views):
            ax = fig.add_subplot(gs[r, c])
            if img is None:
                ax.axis("off")
                continue
            plot_dense_strip(ax, img, display_mode=mode, cut_coords=cuts, vmin=vmin, vmax=vmax, cmap=cmap)
            if r == 0:
                _title(ax, title, fontsize=9, pad=6)
            if c == 0:
                ax.text(
                    -0.04,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=9,
                    clip_on=False,
                )
    add_cbar(fig, CMAP_DIV_BLACK, *CLIM["signed"], "Signed map", [0.935, 0.25, 0.014, 0.5])
    fig.suptitle("Figure B1b. Multi-plane dense montages (group effect vs V6 advantage)", fontsize=12, y=0.97)
    save_fig(fig, out_path)


def render_b2(mat_csv: Path, out_path: Path) -> None:
    apply_journal_style()
    with mat_csv.open() as handle:
        reader = csv.reader(handle)
        header = next(reader)
        cols = header[1:]
        names: List[str] = []
        rows = []
        for line in reader:
            names.append(line[0])
            rows.append([float(x) for x in line[1:]])
    mat = np.asarray(rows, dtype=np.float64)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.5, 7.2),
        dpi=300,
        gridspec_kw={"height_ratios": [1.05, 1.35], "hspace": 0.45},
    )
    stage_names = [n for n in ["CN", "MCI", "AD"] if n in names]
    stage_idx = [names.index(n) for n in stage_names]
    stage_mat = mat[stage_idx]
    im0 = axes[0].imshow(stage_mat, aspect="auto", cmap=CMAP_Q, vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_yticks(range(len(stage_idx)))
    axes[0].set_yticklabels(stage_names)
    axes[0].set_xticks(range(len(cols)))
    axes[0].set_xticklabels(cols, rotation=55, ha="right", fontsize=7)
    _title(axes[0], "(a) MRF q group means by ROI — CN vs MCI vs AD", fontsize=10, pad=8)
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01)
    cb0.set_label("MRF q")

    # Prefer readable association / effect row names
    want = [
        ("MCI_minus_CN", "MRF q MCI−CN"),
        ("AD_minus_CN", "MRF q AD−CN"),
        ("assoc_equal", "Assoc equal pool"),
        ("assoc_v6", "Assoc V6 RC-SPE"),
        ("assoc_v7", "Assoc V7+MRF"),
    ]
    assoc_idx = []
    assoc_labels = []
    for key, lab in want:
        if key in names:
            assoc_idx.append(names.index(key))
            assoc_labels.append(lab)
    assoc_mat = mat[assoc_idx] if assoc_idx else np.zeros((1, len(cols)))
    lim = float(np.nanpercentile(np.abs(assoc_mat), 98)) if assoc_mat.size else 0.55
    lim = max(lim, 0.25)
    im1 = axes[1].imshow(assoc_mat, aspect="auto", cmap=CMAP_DIV, vmin=-lim, vmax=lim, interpolation="nearest")
    axes[1].set_yticks(range(len(assoc_labels)))
    axes[1].set_yticklabels(assoc_labels)
    axes[1].set_xticks(range(len(cols)))
    axes[1].set_xticklabels(cols, rotation=55, ha="right", fontsize=7)
    _title(axes[1], "(b) Group effects and model–ROI associations (V6 highlighted in row order)", fontsize=10, pad=8)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.01)
    cb1.set_label("Signed value")
    # Highlight V6 row
    from matplotlib.patches import Rectangle

    for i, lab in enumerate(assoc_labels):
        if "V6" in lab:
            axes[1].add_patch(
                Rectangle(
                    (-0.5, i - 0.5),
                    len(cols),
                    1.0,
                    fill=False,
                    edgecolor=OKABE_ITO["V6"],
                    lw=2.0,
                    zorder=5,
                )
            )

    fig.suptitle("Figure B2. ROI × stage / method heatmaps", fontsize=12, y=0.98)
    save_fig(fig, out_path)
    # README alias
    from shutil import copyfile

    src = out_path.with_suffix(".png")
    copyfile(src, out_path.with_name(out_path.stem + "_panels.png"))
    copyfile(out_path.with_suffix(".pdf"), out_path.with_name(out_path.stem + "_panels.pdf"))
    print(f"[fig] -> {out_path.with_name(out_path.stem + '_panels.png')}")


def render_b3(top20_json: Path, out_path: Path) -> None:
    apply_journal_style()
    payload = json.loads(top20_json.read_text(encoding="utf-8"))
    regions = payload["regions"]
    values = payload["values"]
    x = np.arange(len(regions))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), dpi=300, sharex=False)
    fig.subplots_adjust(bottom=0.28, top=0.86, wspace=0.28)
    # (a) group effects as paired lollipops
    ax = axes[0]
    for key, color, lab in [
        ("mrf_q_MCI_minus_CN", OKABE_ITO["MCI"], "MCI−CN"),
        ("mrf_q_AD_minus_CN", OKABE_ITO["AD"], "AD−CN"),
    ]:
        if key not in values:
            continue
        y = np.asarray(values[key], dtype=float)
        ax.vlines(x, 0, y, color=color, alpha=0.35, lw=1.0)
        ax.scatter(x, y, s=36, color=color, zorder=3, label=lab, edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="0.5", lw=0.7)
    _title(ax, "(a) MRF q group contrasts", fontsize=10, pad=8)
    ax.set_ylabel("Effect")
    ax.legend(frameon=False, loc="best")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=70, ha="right", fontsize=6.5)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    if "atrophy_AD_minus_CN" in values:
        y = np.asarray(values["atrophy_AD_minus_CN"], dtype=float)
        ax.vlines(x, 0, y, color=OKABE_ITO["atrophy"], alpha=0.35, lw=1.0)
        ax.scatter(x, y, s=36, color=OKABE_ITO["atrophy"], zorder=3, edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="0.5", lw=0.7)
    _title(ax, "(b) Atrophy AD−CN", fontsize=10, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=70, ha="right", fontsize=6.5)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[2]
    width = 0.28
    series = [
        ("assoc_equal_logpool", OKABE_ITO["equal"], "Equal pool", -width),
        ("assoc_v6_rcspe", OKABE_ITO["V6"], "V6 RC-SPE", 0.0),
        ("assoc_v7_mrf", OKABE_ITO["V7"], "V7+MRF", width),
    ]
    for key, color, lab, off in series:
        if key not in values:
            continue
        y = np.asarray(values[key], dtype=float)
        ax.bar(x + off, y, width=width * 0.95, color=color, label=lab, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="0.5", lw=0.7)
    _title(ax, "(c) Model–ROI association (V6 vs baselines)", fontsize=10, pad=8)
    ax.legend(frameon=False, loc="best")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=70, ha="right", fontsize=6.5)
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Figure B3. Top-20 ROIs — group contrast and V6 model advantage", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_path)


def render_b4(maps_dir: Path, out_path: Path) -> None:
    """Dense axial summary strips (replaces sparse glass/ortho empties)."""
    apply_journal_style()
    panels = [
        ("mrf_q_CN_mean", "CN · MRF q", CMAP_Q, CLIM["q"]),
        ("mrf_q_AD_mean", "AD · MRF q", CMAP_Q, CLIM["q"]),
        ("mrf_q_AD_minus_CN", "AD−CN · MRF q", CMAP_DIV_BLACK, CLIM["signed"]),
        ("assoc_equal_logpool", "Equal pool assoc", CMAP_DIV_BLACK, CLIM["assoc"]),
        ("assoc_v6_rcspe", "V6 RC-SPE assoc", CMAP_DIV_BLACK, CLIM["assoc"]),
        ("__delta_v6_equal__", "V6 − equal", CMAP_DIV_BLACK, CLIM["delta"]),
    ]
    delta_img = None
    if (maps_dir / "assoc_v6_rcspe.nii.gz").exists() and (maps_dir / "assoc_equal_logpool.nii.gz").exists():
        delta_img = difference_map(maps_dir, "assoc_v6_rcspe", "assoc_equal_logpool")

    fig = plt.figure(figsize=(14.0, 8.2), dpi=300)
    gs = GridSpec(3, 2, figure=fig, left=0.07, right=0.90, top=0.90, bottom=0.06, wspace=0.08, hspace=0.30)
    for i, (name, title, cmap, clim) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        vmin, vmax = clim
        if name == "__delta_v6_equal__":
            img = delta_img
        else:
            img = load_map(maps_dir, name) if (maps_dir / f"{name}.nii.gz").exists() else None
        if img is None:
            ax.axis("off")
            continue
        plot_dense_strip(ax, img, cut_coords=CUTS_DENSE["z"], vmin=vmin, vmax=vmax, cmap=cmap)
        _title(ax, title, fontsize=9, pad=6)

    add_cbar(fig, CMAP_Q, *CLIM["q"], "MRF q", [0.92, 0.68, 0.012, 0.2])
    add_cbar(fig, CMAP_DIV_BLACK, *CLIM["signed"], "Signed", [0.92, 0.38, 0.012, 0.2])
    add_cbar(fig, CMAP_DIV_BLACK, *CLIM["delta"], "Δ V6−eq", [0.92, 0.10, 0.012, 0.2])
    fig.suptitle("Figure B4. Dense axial summary — staging and V6 advantage", fontsize=12, y=0.97)
    save_fig(fig, out_path)


def render_b5_from_json(dkt_json: Path, out_path: Path) -> None:
    """Papez-proxy figure: staging bars + L/R cortical matrix + anatomical connectome.

    Replaces the previous cartoon schematic (hard to read, low information).
    """
    apply_journal_style()
    if not dkt_json.exists():
        print(f"[skip] B5 cache missing: {dkt_json}")
        return
    payload = json.loads(dkt_json.read_text(encoding="utf-8"))
    ad = payload.get("ad_minus_cn", {})
    mci = payload.get("mci_minus_cn", {})
    meta = payload.get("meta", {})

    node_keys = [
        ("entorhinal", "Entorhinal"),
        ("hippocampus", "Hippocampus"),
        ("parahippocampal", "Parahippocampal"),
        ("cingulate", "Cingulate"),
        ("thalamus", "Thalamus"),
    ]
    labels = [lab for _, lab in node_keys]
    ad_v = np.asarray([float(ad.get(k, 0.0)) for k, _ in node_keys], dtype=float)
    mci_v = np.asarray([float(mci.get(k, 0.0)) for k, _ in node_keys], dtype=float)

    # Cortical L/R matrix (DKT parcels used in Papez proxy)
    cortical_rows = [
        ("Entorhinal", "ctx-lh-entorhinal", "ctx-rh-entorhinal"),
        ("Parahippocampal", "ctx-lh-parahippocampal", "ctx-rh-parahippocampal"),
        ("rACC", "ctx-lh-rostralanteriorcingulate", "ctx-rh-rostralanteriorcingulate"),
        ("cACC", "ctx-lh-caudalanteriorcingulate", "ctx-rh-caudalanteriorcingulate"),
        ("PCC", "ctx-lh-posteriorcingulate", "ctx-rh-posteriorcingulate"),
        ("Isthmus Cing.", "ctx-lh-isthmuscingulate", "ctx-rh-isthmuscingulate"),
    ]
    # Two columns blocks: MCI−CN | AD−CN
    heat = np.zeros((len(cortical_rows), 4), dtype=float)
    for i, (_lab, lh, rh) in enumerate(cortical_rows):
        heat[i, 0] = float(mci.get(lh, np.nan))
        heat[i, 1] = float(mci.get(rh, np.nan))
        heat[i, 2] = float(ad.get(lh, np.nan))
        heat[i, 3] = float(ad.get(rh, np.nan))

    # Anatomical connectome coords (MNI mm, L/R averaged nodes for clarity)
    coords = {
        "Entorhinal": np.array([0.0, -8.0, -28.0]),
        "Hippocampus": np.array([0.0, -22.0, -14.0]),
        "Parahippocampal": np.array([0.0, -32.0, -16.0]),
        "Cingulate": np.array([0.0, -18.0, 38.0]),
        "Thalamus": np.array([0.0, -18.0, 7.0]),
    }
    # Split L/R for connectome (more anatomical)
    node_names = [
        "L-Ent", "R-Ent", "L-Hip", "R-Hip", "L-PHC", "R-PHC", "L-Cing", "R-Cing", "L-Thal", "R-Thal"
    ]
    node_xyz = np.array(
        [
            [-25, -8, -28],
            [25, -8, -28],
            [-26, -22, -14],
            [26, -22, -14],
            [-24, -32, -16],
            [24, -32, -16],
            [-6, -18, 38],
            [6, -18, 38],
            [-11, -18, 7],
            [11, -18, 7],
        ],
        dtype=float,
    )
    # Node values from L/R cortical or bilateral means
    def _get(d, *keys, default=0.0):
        for k in keys:
            if k in d:
                return float(d[k])
        return float(default)

    node_val = np.array(
        [
            _get(ad, "ctx-lh-entorhinal", default=ad_v[0]),
            _get(ad, "ctx-rh-entorhinal", default=ad_v[0]),
            _get(ad, "hippocampus", default=ad_v[1]),
            _get(ad, "hippocampus", default=ad_v[1]),
            _get(ad, "ctx-lh-parahippocampal", default=ad_v[2]),
            _get(ad, "ctx-rh-parahippocampal", default=ad_v[2]),
            _get(ad, "cingulate", default=ad_v[3]),
            _get(ad, "cingulate", default=ad_v[3]),
            _get(ad, "thalamus", default=ad_v[4]),
            _get(ad, "thalamus", default=ad_v[4]),
        ],
        dtype=float,
    )
    # Classic Papez-proxy edges (within-hemisphere loop + commissural)
    edges = [
        (0, 2), (1, 3),  # Ent→Hip
        (2, 4), (3, 5),  # Hip→PHC
        (4, 6), (5, 7),  # PHC→Cing
        (6, 8), (7, 9),  # Cing→Thal
        (8, 2), (9, 3),  # Thal→Hip
        (6, 7), (2, 3),  # cross-hemisphere
    ]
    adj = np.zeros((len(node_names), len(node_names)), dtype=float)
    for i, j in edges:
        w = 0.35 + 0.65 * (0.5 * (abs(node_val[i]) + abs(node_val[j]))) / max(float(np.max(np.abs(node_val))), 1e-6)
        adj[i, j] = w
        adj[j, i] = w

    from nilearn.plotting import plot_connectome

    fig = plt.figure(figsize=(13.6, 5.0), dpi=300)
    gs = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[1.0, 1.25, 1.35],
        wspace=0.32,
        left=0.06,
        right=0.98,
        top=0.84,
        bottom=0.18,
    )

    # (a) Grouped bars: MCI−CN vs AD−CN
    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(labels))
    w = 0.38
    ax0.bar(x - w / 2, mci_v, width=w, color=OKABE_ITO["MCI"], edgecolor="white", label="MCI−CN", zorder=3)
    ax0.bar(x + w / 2, ad_v, width=w, color=OKABE_ITO["AD"], edgecolor="white", label="AD−CN", zorder=3)
    ax0.axhline(0, color="0.45", lw=0.8)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=28, ha="right", fontsize=8, fontproperties=SERIF_FP)
    ax0.set_ylabel("DKT atrophy z\n(higher = more AD-like loss)", fontproperties=SERIF_FP)
    ax0.legend(frameon=False, fontsize=8, prop=SERIF_FP)
    ax0.grid(axis="y", alpha=0.25, zorder=0)
    _title(ax0, "(a) Circuit nodes · staging (AIBL)", fontsize=10, pad=8)

    # (b) L/R cortical heatmap
    ax1 = fig.add_subplot(gs[0, 1])
    vmax = max(float(np.nanmax(np.abs(heat))), 0.5)
    im = ax1.imshow(heat, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax1.set_yticks(range(len(cortical_rows)))
    ax1.set_yticklabels([r[0] for r in cortical_rows], fontsize=8, fontproperties=SERIF_FP)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(["MCI L", "MCI R", "AD L", "AD R"], fontsize=8, fontproperties=SERIF_FP)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i, j]
            if not np.isfinite(v):
                continue
            ax1.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if v > 0.55 * vmax else "0.15",
                fontproperties=SERIF_FP,
            )
    _title(ax1, "(b) Cortical DKT parcels · L/R × stage", fontsize=10, pad=8)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label("Atrophy z", fontsize=8)

    # (c) Anatomical connectome
    ax2 = fig.add_subplot(gs[0, 2])
    vmax_n = max(float(np.max(node_val)), 0.5)
    node_colors = [plt.cm.YlOrRd(0.25 + 0.7 * (v / vmax_n)) for v in node_val]
    node_size = 40 + 180 * (node_val / vmax_n)
    plot_connectome(
        adj,
        node_xyz,
        node_color=node_colors,
        node_size=node_size,
        edge_threshold=0.0,
        edge_cmap="Greys",
        edge_vmin=0.0,
        edge_vmax=1.0,
        colorbar=False,
        axes=ax2,
        display_mode="lzry",
        black_bg=False,
    )
    _title(ax2, "(c) Anatomical Papez-proxy connectome (AD−CN)", fontsize=10, pad=8)

    n_cn = meta.get("cn_hit", "?")
    n_mci = meta.get("mci_hit", "?")
    n_ad = meta.get("ad_hit", "?")
    fig.suptitle(
        "Figure B5. Papez-circuit consistency — AIBL DKT atrophy (CN / MCI / AD)",
        fontsize=12,
        y=0.97,
        fontproperties=SERIF_BOLD,
    )
    fig.text(
        0.5,
        0.03,
        f"Higher z = greater tissue loss vs CN  |  coverage CN={n_cn}, MCI={n_mci}, AD={n_ad}  |  "
        "Supporting anatomical consistency — not Braak staging claim.",
        ha="center",
        fontsize=7,
        color="0.35",
        fontproperties=SERIF_FP,
    )
    save_fig(fig, out_path)


def render_b6_b7_compact(
    *,
    maps_dir: Path,
    assets: Path,
    fig_dir: Path,
    feature_csv: Path,
    pred_aibl: Path,
    pred_oasis: Path,
) -> None:
    """Compact case montages with denser cuts and journal fonts."""
    apply_journal_style()
    from render_brain_figures_phase3 import (
        MRFParams,
        build_roi_graph,
        fit_cn_volume_stats,
        pick_cases,
        read_csv,
        _scan_mrf_map,
        TEMPLATE_AFFINE,
    )

    if not feature_csv.exists() or not pred_aibl.exists():
        print("[skip] B6/B7 missing prediction/feature csv")
        return

    feature_rows = read_csv(feature_csv)
    for r in feature_rows:
        r["label"] = int(float(r["label"]))
    feature_by_scan = {r["scan_id"]: r for r in feature_rows}
    atlas = np.asanyarray(nib.load(str(assets / "atlas_template.nii.gz")).dataobj)
    train_cn = [r for r in feature_rows if r.get("split") == "train" and r["label"] == 0]
    cn_mean, cn_std = fit_cn_volume_stats(train_cn if train_cn else [r for r in feature_rows if r["label"] == 0])
    params = MRFParams()
    edges = build_roi_graph(params.couple_strength, params.anti_strength)

    # ---- B6 ----
    picks = pick_cases(read_csv(pred_aibl))
    keys = [k for k in ["correct_CN", "correct_MCI", "correct_AD", "err_MCI_to_AD", "err_AD_to_MCI", "err_MCI_to_CN"] if k in picks]
    if keys:
        fig = plt.figure(figsize=(2.35 * len(keys) + 0.8, 5.6), dpi=300)
        gs = GridSpec(2, len(keys), figure=fig, height_ratios=[2.4, 1.0], hspace=0.38, wspace=0.12, top=0.86, bottom=0.10, left=0.05, right=0.98)
        cuts = (-28, -16, -4, 8, 20, 32)
        for col, key in enumerate(keys):
            row = picks[key]
            ax0 = fig.add_subplot(gs[0, col])
            ax1 = fig.add_subplot(gs[1, col])
            feat = feature_by_scan.get(row["scan_id"])
            short = str(row["scan_id"])[-16:]
            title = f"{key.replace('_', ' ')}\nT={row['y_true']} → P={row['y_pred']}"
            if feat is None:
                ax0.axis("off")
                _title(ax0, title, fontsize=7, pad=4)
                continue
            qmap = _scan_mrf_map(feat, atlas, cn_mean, cn_std, edges, params)
            img = nib.Nifti1Image(qmap.astype(np.float32), TEMPLATE_AFFINE)
            plot_dense_strip(ax0, img, cut_coords=cuts, vmin=0, vmax=1, cmap=CMAP_Q)
            _title(ax0, title, fontsize=7, pad=4)
            probs = [float(row["prob_CN"]), float(row["prob_MCI"]), float(row["prob_AD"])]
            ax1.bar(["CN", "MCI", "AD"], probs, color=[OKABE_ITO["CN"], OKABE_ITO["MCI"], OKABE_ITO["AD"]], edgecolor="white")
            ax1.set_ylim(0, 1)
            ax1.tick_params(labelsize=7)
            if col == 0:
                ax1.set_ylabel("p", fontsize=8)
            ax1.set_xlabel(short, fontsize=6)
        fig.suptitle("Figure B6. Subject MRF-q cases — correct vs boundary errors", fontsize=11, y=0.97)
        save_fig(fig, fig_dir / "figure_B6_case_montages")

    # ---- B7 ----
    if pred_oasis.exists():
        oasis_rows = read_csv(pred_oasis)
        false_imp, true_cn = [], []
        for r in oasis_rows:
            probs = np.array([float(r["prob_CN"]), float(r["prob_MCI"]), float(r["prob_AD"])])
            pred = int(probs.argmax())
            item = {**r, "_pred": pred, "_p_imp": float(probs[1] + probs[2]), "_conf": float(probs[pred])}
            (false_imp if pred != 0 else true_cn).append(item)
        false_imp = sorted(false_imp, key=lambda r: -r["_p_imp"])[:3]
        true_cn = sorted(true_cn, key=lambda r: -r["_conf"])[:3]
        picks7 = [("false impairment", r) for r in false_imp] + [("true CN", r) for r in true_cn]
        if picks7:
            fig = plt.figure(figsize=(2.55 * len(picks7) + 0.4, 2.8), dpi=300)
            gs = GridSpec(1, len(picks7), figure=fig, wspace=0.08, top=0.72, bottom=0.04, left=0.02, right=0.99)
            cuts = (-32, -24, -16, -8, 0, 8, 16, 24, 32)
            for col, (tag, row) in enumerate(picks7):
                ax = fig.add_subplot(gs[0, col])
                feat = feature_by_scan.get(row["scan_id"])
                title = f"{tag} | pred={['CN','MCI','AD'][row['_pred']]}\n{str(row['scan_id'])[-18:]}"
                if feat is None:
                    ax.axis("off")
                    _title(ax, title, fontsize=7, pad=6)
                    continue
                qmap = _scan_mrf_map(feat, atlas, cn_mean, cn_std, edges, params)
                img = nib.Nifti1Image(qmap.astype(np.float32), TEMPLATE_AFFINE)
                plot_dense_strip(ax, img, cut_coords=cuts, vmin=0, vmax=1, cmap=CMAP_Q)
                _title(ax, title, fontsize=7, pad=6)
            fig.suptitle("Figure B7. OASIS stress-test cases (limitation)", fontsize=11, y=0.95)
            save_fig(fig, fig_dir / "figure_B7_oasis_stress")


def render_smoke(maps_dir: Path, out_path: Path) -> None:
    apply_journal_style()
    fig = plt.figure(figsize=(12.5, 3.2), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    img = load_map(maps_dir, "assoc_v6_rcspe")
    plot_dense_strip(ax, img, cut_coords=CUTS_DENSE["z"], vmin=-0.55, vmax=0.55, cmap=CMAP_DIV_BLACK)
    _title(ax, "V6 RC-SPE association · dense axial (native, no underlay)", fontsize=10, pad=8)
    add_cbar(fig, CMAP_DIV_BLACK, -0.55, 0.55, "ρ", [0.92, 0.2, 0.015, 0.6])
    save_fig(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("chapter1_foundation/ARA-Net/reports/brain_figures_fcstyle"),
    )
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=Path("chapter1_foundation/outputs/v4/atlas_feature_cache_v4.csv"),
    )
    parser.add_argument(
        "--pred-aibl",
        type=Path,
        default=Path(
            "chapter1_foundation/outputs/v4/rescue_probability_subject_no_oasis_tune/balanced_aibl_heldout_predictions.csv"
        ),
    )
    parser.add_argument(
        "--pred-oasis",
        type=Path,
        default=Path(
            "chapter1_foundation/outputs/v4/rescue_probability_subject_no_oasis_tune/balanced_oasis_external_predictions.csv"
        ),
    )
    args = parser.parse_args()

    # Resolve relative to CWD; also try ARA-Net-relative.
    assets_dir = args.assets_dir
    if not assets_dir.exists():
        alt = SCRIPT_DIR.parent / "reports/brain_figures_fcstyle"
        if alt.exists():
            assets_dir = alt

    maps_dir = assets_dir / "assets" / "maps"
    mat_dir = assets_dir / "assets" / "matrices"
    fig_dir = assets_dir / "figures"
    assets = assets_dir / "assets"
    fig_dir.mkdir(parents=True, exist_ok=True)

    apply_journal_style()
    print(f"[style] font family serif -> Times New Roman / Liberation Serif")
    print(f"[out] {fig_dir}")

    # Primary suite
    render_b1_group_and_model(maps_dir, fig_dir / "figure_B1_anatomy_underlay_montage")
    render_b1_group_and_model(maps_dir, fig_dir / "figure_B1_method_montage")
    render_b1b_multiplane(maps_dir, fig_dir / "figure_B1b_contrast_montage")
    render_b2(mat_dir / "methods_roi_matrix.csv", fig_dir / "figure_B2_roi_method_heatmap")
    render_b3(mat_dir / "top20_rois.json", fig_dir / "figure_B3_top_roi_dots")
    render_b4(maps_dir, fig_dir / "figure_B4_glassbrain_summary")
    render_b5_from_json(mat_dir / "dkt_ad_minus_cn.json", fig_dir / "figure_B5_papez_network")
    render_smoke(maps_dir, fig_dir / "mcp_smoke_glassbrain")

    # Resolve feature paths
    feat = args.feature_csv
    if not feat.exists():
        feat = SCRIPT_DIR.parent.parent / "outputs/v4/atlas_feature_cache_v4.csv"
    pred_a = args.pred_aibl
    if not pred_a.exists():
        pred_a = SCRIPT_DIR.parent.parent / "outputs/v4/rescue_probability_subject_no_oasis_tune/balanced_aibl_heldout_predictions.csv"
    pred_o = args.pred_oasis
    if not pred_o.exists():
        pred_o = SCRIPT_DIR.parent.parent / "outputs/v4/rescue_probability_subject_no_oasis_tune/balanced_oasis_external_predictions.csv"

    render_b6_b7_compact(
        maps_dir=maps_dir,
        assets=assets,
        fig_dir=fig_dir,
        feature_csv=feat,
        pred_aibl=pred_a,
        pred_oasis=pred_o,
    )
    print(f"[done] journal suite -> {fig_dir}")


if __name__ == "__main__":
    main()
