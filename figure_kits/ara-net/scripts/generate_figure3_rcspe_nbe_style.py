#!/usr/bin/env python3
"""Generate Figure 3 for the ARA-Net chapter 1 manuscript.

The visual style follows the local A2C NBE-style plotting package: compact
panels, muted colors, thin strokes, editable PDF text, and high information
density. The figure uses locked RC-SPE parameters and aggregate metrics from
the repository; patient-level probability examples are drawn as method
schematics and are not presented as measured patient data.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/aranet_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = ROOT / "reports" / "v6_final_model" / "figures"
CONFIG_PATH = ROOT / "deployment" / "final_ensemble_config.json"
BASE_TABLE_PATH = ROOT / "reports" / "v6_algorithm_innovation" / "base_model_summary_table.csv"

CLASSES = ["CN", "MCI", "AD"]
CLASS_COLORS = {"CN": "#4C9A78", "MCI": "#E3A53B", "AD": "#C9583B"}
PALETTE = {
    "ink": "#1B1F23",
    "slate": "#4B5563",
    "muted": "#8A94A3",
    "grid": "#D8DEE6",
    "panel": "#F7F8FA",
    "blue": "#2E6F9E",
    "teal": "#2A9D8F",
    "green": "#4C9A78",
    "gold": "#E3A53B",
    "red": "#C9583B",
    "purple": "#7556A7",
    "rose": "#B65C7A",
    "brown": "#8D6E63",
}

STREAM_META = {
    "aibl_adapted_atlas_biomarker_enhanced__hgb": {
        "short": "Atlas+bio",
        "feature": "Atlas MRI\n+ biomarkers",
        "learner": "HGB",
        "color": PALETTE["blue"],
    },
    "aibl_adapted_atlas_core_clinical__hgb": {
        "short": "Atlas+clin",
        "feature": "Atlas MRI\n+ clinical",
        "learner": "HGB",
        "color": PALETTE["teal"],
    },
    "aibl_adapted_clinical_biomarker_only__rf_balanced": {
        "short": "Clin+bio",
        "feature": "Clinical\n+ biomarkers",
        "learner": "RF",
        "color": PALETTE["gold"],
    },
    "aibl_adapted_clinical_core_only__hgb": {
        "short": "Clinical HGB",
        "feature": "Clinical\ncore",
        "learner": "HGB",
        "color": PALETTE["purple"],
    },
    "aibl_adapted_clinical_core_only__rf_balanced": {
        "short": "Clinical RF",
        "feature": "Clinical\ncore",
        "learner": "RF",
        "color": PALETTE["rose"],
    },
    "rf__logreg": {
        "short": "Cascade",
        "feature": "RF output\ncascade",
        "learner": "LogReg",
        "color": PALETTE["brown"],
    },
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.2,
            "axes.linewidth": 0.65,
            "axes.edgecolor": PALETTE["slate"],
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 6.4,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def load_config() -> Dict:
    with CONFIG_PATH.open() as f:
        return json.load(f)


def load_base_table() -> Dict[str, Dict[str, str]]:
    with BASE_TABLE_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["run"]: row for row in rows}


def panel_label(ax, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.8,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def clean_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def soft_spines(ax) -> None:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color(PALETTE["slate"])
        ax.spines[side].set_linewidth(0.65)


def box(
    ax,
    xy: Tuple[float, float],
    wh: Tuple[float, float],
    text: str,
    fc: str,
    ec: str | None = None,
    fontsize: float = 7.0,
    lw: float = 0.8,
    radius: float = 0.018,
    color: str | None = None,
):
    ec = ec or PALETTE["ink"]
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color or PALETTE["ink"],
        linespacing=1.05,
    )
    return patch


def arrow(ax, start, end, color=None, lw=0.85, style="-|>", mutation=8, ls="-"):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        linestyle=ls,
        color=color or PALETTE["slate"],
    )
    ax.add_patch(arr)
    return arr


def draw_probability_chips(ax, x: float, y: float, w: float, h: float, labels: Iterable[str]) -> None:
    labels = list(labels)
    gap = 0.006
    chip_w = (w - gap * (len(labels) - 1)) / len(labels)
    for j, cls in enumerate(labels):
        x0 = x + j * (chip_w + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x0, y),
                chip_w,
                h,
                boxstyle="round,pad=0.005,rounding_size=0.012",
                facecolor=CLASS_COLORS[cls],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.96,
            )
        )
        ax.text(
            x0 + chip_w / 2,
            y + h / 2,
            cls,
            ha="center",
            va="center",
            fontsize=5.8,
            color="white",
            fontweight="bold",
        )


def draw_panel_a(ax, config: Dict, base_rows: Dict[str, Dict[str, str]]) -> None:
    clean_axes(ax)
    panel_label(ax, "a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ARA-Net probability streams", loc="left", color=PALETTE["slate"], pad=3)

    y_positions = np.linspace(0.82, 0.14, 6)
    for i, (run, weight, y) in enumerate(zip(config["base_models"], config["weights"], y_positions), start=1):
        meta = STREAM_META[run]
        row = base_rows.get(run, {})
        bacc = float(row.get("aibl_bacc", np.nan))
        color = meta["color"]

        ax.text(0.010, y + 0.002, f"S{i}", ha="left", va="center", fontsize=6.0, color=PALETTE["muted"])
        box(ax, (0.058, y - 0.036), (0.245, 0.072), meta["feature"], "#EEF4F8", color, fontsize=5.5, lw=0.75)
        arrow(ax, (0.310, y), (0.350, y), color=PALETTE["muted"], lw=0.60, mutation=6)
        box(ax, (0.360, y - 0.030), (0.128, 0.060), meta["learner"], "#FFFFFF", color, fontsize=5.9, lw=0.75)
        arrow(ax, (0.497, y), (0.537, y), color=PALETTE["muted"], lw=0.60, mutation=6)
        box(ax, (0.547, y - 0.033), (0.215, 0.066), r"$p_m(CN,MCI,AD)$", "#FFF9EA", color, fontsize=5.0, lw=0.75)
        ax.text(0.790, y + 0.012, f"w={weight:.3f}", fontsize=5.3, color=color, ha="left", va="center")
        ax.text(0.790, y - 0.018, f"BAcc={bacc:.3f}", fontsize=5.2, color=PALETTE["slate"], ha="left", va="center")

    ax.text(
        0.075,
        0.025,
        "Streams stay separate before pooling.",
        fontsize=5.6,
        color=PALETTE["muted"],
        ha="left",
        va="bottom",
    )


def draw_panel_b(ax, config: Dict) -> None:
    clean_axes(ax)
    panel_label(ax, "b")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Three-class uncertainty is preserved", loc="left", color=PALETTE["slate"], pad=3)

    tri = np.array([[0.14, 0.16], [0.86, 0.16], [0.50, 0.80]])
    ax.add_patch(Polygon(tri, closed=True, facecolor="#FFFFFF", edgecolor=PALETTE["grid"], linewidth=0.9))
    label_offsets = {"CN": (0.0, -0.062), "MCI": (-0.030, -0.062), "AD": (0.0, 0.055)}
    for (x, y), label, color in zip(tri, ["CN", "MCI", "AD"], [CLASS_COLORS["CN"], CLASS_COLORS["MCI"], CLASS_COLORS["AD"]]):
        ax.scatter([x], [y], s=52, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        dx, dy = label_offsets[label]
        ax.text(x + dx, y + dy, label, ha="center", va="center", fontsize=6.7, color=PALETTE["ink"])

    schematic_points = np.array(
        [
            [0.35, 0.33],
            [0.43, 0.47],
            [0.59, 0.36],
            [0.50, 0.56],
            [0.66, 0.25],
            [0.29, 0.25],
        ]
    )
    colors = [STREAM_META[run]["color"] for run in config["base_models"]]
    for i, (xy, color) in enumerate(zip(schematic_points, colors), start=1):
        ax.scatter([xy[0]], [xy[1]], s=28, color=color, edgecolor="white", linewidth=0.7, zorder=4)
        ax.text(xy[0] + 0.025, xy[1] + 0.018, f"S{i}", fontsize=5.5, color=color, ha="left", va="center")

    box(ax, (0.19, 0.875), (0.62, 0.070), "probability vector, not hard vote", "#F7F8FA", PALETTE["grid"], fontsize=6.2, lw=0.7)
    ax.text(0.24, 0.055, r"$p_m=[p_{m,CN},p_{m,MCI},p_{m,AD}]$", fontsize=7.0, color=PALETTE["ink"], ha="left")


def draw_panel_c(ax, config: Dict) -> None:
    panel_label(ax, "c")
    weights = np.asarray(config["weights"], dtype=float)
    labels = [STREAM_META[run]["short"] for run in config["base_models"]]
    colors = [STREAM_META[run]["color"] for run in config["base_models"]]

    y = np.arange(len(weights))[::-1]
    ax.barh(y, weights, color=colors, edgecolor="white", height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.8)
    ax.set_xlim(0, 0.40)
    ax.set_xlabel("Locked non-negative weight")
    ax.set_title("Weighted log-probability pooling", loc="left", color=PALETTE["slate"], pad=10)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.55, alpha=0.75)
    soft_spines(ax)
    for yi, wi in zip(y, weights):
        ax.text(wi + 0.008, yi, f"{wi:.3f}", ha="left", va="center", fontsize=5.8, color=PALETTE["slate"])
    ax.text(
        0.01,
        1.16,
        r"$z_c=\sum_m w_m\log(p_{m,c}+\epsilon)$,  $w_m\geq0$,  $\sum_mw_m=1$",
        transform=ax.transAxes,
        fontsize=6.1,
        color=PALETTE["ink"],
        ha="left",
        va="bottom",
    )


def draw_panel_d(ax, config: Dict) -> None:
    panel_label(ax, "d")
    offsets = [float(config["offsets"][cls]) for cls in CLASSES]
    colors = [CLASS_COLORS[cls] for cls in CLASSES]
    ax.bar(CLASSES, offsets, color=colors, edgecolor="white", width=0.66)
    ax.axhline(0, color=PALETTE["slate"], linewidth=0.65)
    ax.set_ylim(-1.10, 1.15)
    ax.set_ylabel("Class offset")
    ax.set_title("Offsets and temperature scaling", loc="left", color=PALETTE["slate"], pad=3)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55, alpha=0.75)
    soft_spines(ax)
    for x, off, color in zip(np.arange(3), offsets, colors):
        va = "bottom" if off >= 0 else "top"
        dy = 0.055 if off >= 0 else -0.055
        ax.text(x, off + dy, f"{off:+.3f}", ha="center", va=va, fontsize=6.2, color=color)

    ax.text(
        0.04,
        0.94,
        rf"$q_c=softmax((z_c+b_c)/T)$",
        transform=ax.transAxes,
        fontsize=6.5,
        color=PALETTE["ink"],
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor=PALETTE["grid"], linewidth=0.6),
    )
    ax.text(
        0.72,
        0.17,
        f"T={float(config['temperature']):.3f}",
        transform=ax.transAxes,
        fontsize=8.0,
        color=PALETTE["blue"],
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#EEF4F8", edgecolor=PALETTE["blue"], linewidth=0.75),
    )


def metric_tile(ax, x, y, w, h, label, value, color, sublabel="", value_size=8.2) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFFFFF", edgecolor=PALETTE["grid"], linewidth=0.65))
    ax.add_patch(Rectangle((x, y), 0.012, h, facecolor=color, edgecolor=color, linewidth=0))
    ax.text(x + 0.025, y + h * 0.66, label, ha="left", va="center", fontsize=5.8, color=PALETTE["slate"])
    ax.text(x + w - 0.022, y + h * 0.36, value, ha="right", va="center", fontsize=value_size, color=color, fontweight="bold")
    if sublabel:
        ax.text(x + 0.025, y + h * 0.20, sublabel, ha="left", va="center", fontsize=5.2, color=PALETTE["muted"])


def draw_panel_e(ax, config: Dict) -> None:
    clean_axes(ax)
    panel_label(ax, "e")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ARA-Net risk-constrained model lock", loc="left", color=PALETTE["slate"], pad=3)

    primary = config["primary_evaluation"]
    recall = primary["recall"]
    box(
        ax,
        (0.05, 0.82),
        (0.90, 0.105),
        "score balances external BAcc, MCI/AD recall,\nAD-to-CN avoidance, calibration and IXI CN retention",
        "#F7F8FA",
        PALETTE["grid"],
        fontsize=5.7,
        lw=0.7,
    )
    tiles = [
        ("AIBL BAcc", f"{primary['balanced_accuracy']:.3f}", PALETTE["blue"], "locked subject endpoint"),
        ("MCI recall", f"{recall['MCI']:.3f}", CLASS_COLORS["MCI"], "minority disease"),
        ("AD recall", f"{recall['AD']:.3f}", CLASS_COLORS["AD"], "disease rescue"),
        ("AD->CN", "0", PALETTE["red"], "endpoint errors"),
        ("IXI CN", "1.000", CLASS_COLORS["CN"], "healthy retention"),
        ("ECE", "0.078", PALETTE["purple"], "calibration"),
    ]
    for idx, item in enumerate(tiles):
        col = idx % 2
        row = idx // 2
        metric_tile(ax, 0.055 + col * 0.47, 0.58 - row * 0.155, 0.405, 0.115, *item)

    ax.text(0.055, 0.090, "ADNI val + AIBL adapt-val + IXI used before lock", fontsize=5.7, color=PALETTE["slate"], ha="left")
    ax.text(0.055, 0.040, "OASIS stress test excluded from tuning", fontsize=5.7, color=PALETTE["red"], ha="left", fontweight="bold")


def draw_panel_f(ax) -> None:
    clean_axes(ax)
    panel_label(ax, "f")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Scan-to-subject probability averaging", loc="left", color=PALETTE["slate"], pad=3)

    ys = [0.72, 0.50, 0.28]
    labels = [r"scan 1: $q_1$", r"scan 2: $q_2$", r"scan k: $q_k$"]
    for y, label in zip(ys, labels):
        box(ax, (0.06, y - 0.055), (0.33, 0.11), label, "#FFFFFF", PALETTE["grid"], fontsize=6.2, lw=0.65)
        draw_probability_chips(ax, 0.425, y - 0.030, 0.245, 0.060, CLASSES)
        arrow(ax, (0.69, y), (0.77, 0.50), color=PALETTE["muted"], lw=0.65, mutation=6)

    box(ax, (0.77, 0.415), (0.18, 0.17), "subject\nmean", "#EEF4F8", PALETTE["blue"], fontsize=6.2, lw=0.75)
    ax.text(
        0.06,
        0.090,
        r"$\bar{q}_{s,c}=\frac{1}{n_s}\sum_{i\in s}q_{i,c}$",
        fontsize=7.8,
        color=PALETTE["ink"],
        ha="left",
        va="center",
    )
    ax.text(0.06, 0.018, "Primary endpoint: subject-level probabilities.", fontsize=5.5, color=PALETTE["muted"], ha="left")


def draw_panel_g(ax, config: Dict) -> None:
    clean_axes(ax)
    panel_label(ax, "g", x=-0.025, y=1.04)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ARA-Net output and reported endpoint", loc="left", color=PALETTE["slate"], pad=3)

    box(ax, (0.040, 0.56), (0.190, 0.190), "subject\nprobabilities", "#FFFFFF", PALETTE["grid"], fontsize=7.0)
    draw_probability_chips(ax, 0.060, 0.48, 0.150, 0.055, CLASSES)
    arrow(ax, (0.245, 0.655), (0.335, 0.655), color=PALETTE["muted"], mutation=8)
    box(ax, (0.350, 0.56), (0.170, 0.190), r"$\hat{y}_s=\arg\max_c\bar{q}_{s,c}$", "#EEF4F8", PALETTE["blue"], fontsize=7.0)
    arrow(ax, (0.535, 0.655), (0.625, 0.655), color=PALETTE["muted"], mutation=8)
    box(ax, (0.640, 0.56), (0.175, 0.190), "confidence\nand margin", "#FFF9EA", PALETTE["gold"], fontsize=7.0)
    arrow(ax, (0.830, 0.655), (0.910, 0.655), color=PALETTE["muted"], mutation=8)
    box(ax, (0.915, 0.56), (0.055, 0.190), "error\nreview", "#F8EEEE", PALETTE["red"], fontsize=6.0)

    primary = config["primary_evaluation"]
    recall = primary["recall"]
    metric_items = [
        ("AIBL subjects", "216", PALETTE["slate"], 8.0),
        ("Accuracy", f"{primary['accuracy']:.3f}", PALETTE["blue"], 8.0),
        ("BAcc", f"{primary['balanced_accuracy']:.3f}", PALETTE["blue"], 8.0),
        ("Macro AUC", f"{primary['macro_auc_ovr']:.3f}", PALETTE["purple"], 8.0),
        ("AD-vs-CN AUC", f"{primary['ad_vs_cn_auc']:.3f}", PALETTE["red"], 8.0),
        ("CN/MCI/AD recall", f"{recall['CN']:.3f}  {recall['MCI']:.3f}  {recall['AD']:.3f}", PALETTE["green"], 5.9),
        ("IXI CN retention", "1.000", PALETTE["green"], 8.0),
        ("AD->CN errors", "0", PALETTE["red"], 8.0),
    ]
    x0s = [0.04, 0.285, 0.530, 0.775]
    for i, (label, value, color, value_size) in enumerate(metric_items):
        row = i // 4
        col = i % 4
        metric_tile(ax, x0s[col], 0.27 - row * 0.145, 0.205, 0.105, label, value, color, value_size=value_size)

    ax.text(
        0.040,
        0.020,
        "Reported output: calibrated CN/MCI/AD subject-level prediction with confidence and decision margin.",
        fontsize=5.8,
        color=PALETTE["muted"],
        ha="left",
    )


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png, dpi=360, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(png)
    print(pdf)


def main() -> None:
    apply_style()
    config = load_config()
    base_rows = load_base_table()

    fig = plt.figure(figsize=(7.2, 8.2))
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.08, 1.0, 0.88], hspace=0.55, wspace=0.55)

    draw_panel_a(fig.add_subplot(gs[0, 0:4]), config, base_rows)
    draw_panel_b(fig.add_subplot(gs[0, 4:8]), config)
    draw_panel_c(fig.add_subplot(gs[0, 8:12]), config)
    draw_panel_d(fig.add_subplot(gs[1, 0:4]), config)
    draw_panel_e(fig.add_subplot(gs[1, 4:8]), config)
    draw_panel_f(fig.add_subplot(gs[1, 8:12]))
    draw_panel_g(fig.add_subplot(gs[2, 0:12]), config)

    fig.suptitle(
        "ARA-Net model architecture with RC-SPE probability ensemble",
        x=0.02,
        y=0.995,
        ha="left",
        va="top",
        fontsize=15.0,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    save(fig, "figure3_rcspe_architecture_nbe_style")


if __name__ == "__main__":
    main()
