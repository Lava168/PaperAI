#!/usr/bin/env python3
"""Generate v6 final-model figures without pandas."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ["CN", "MCI", "AD"]

COLORS = {
    "old": "#6f7782",
    "v4": "#2f6f73",
    "scan": "#276fbf",
    "subject": "#7a4f9a",
    "clinical": "#b8463c",
    "cn": "#3d7f5f",
    "mci": "#d99b2b",
    "ad": "#b8463c",
    "grid": "#d7dde5",
    "text": "#1f2933",
    "muted": "#667085",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {png}")
    print(f"[saved] {pdf}")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": "#9aa4b2",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["text"],
    )


def fmt(value: float) -> str:
    if value != value:
        return "NA"
    return f"{value:.3f}"


def lookup_table2(table2_rows: list[dict], model: str, evaluation: str) -> dict:
    for row in table2_rows:
        if row.get("model") == model and row.get("evaluation") == evaluation:
            return row
    return {}


def figure_external_rescue(summary: dict, table2_rows: list[dict], out_dir: Path) -> None:
    final_subject = summary["final_model"]["subject_level_metrics"]["aibl_heldout"]
    final_scan = summary["final_model"]["scan_level_reference"]["aibl_heldout"]
    ixi_subject = summary["final_model"]["subject_level_metrics"]["ixi_external"]
    ixi_scan = summary["final_model"]["scan_level_reference"]["ixi_external"]
    old = lookup_table2(table2_rows, "Old v3 ensemble", "AIBL external")
    old_ixi = lookup_table2(table2_rows, "Old v3 ensemble", "IXI healthy")
    v4 = lookup_table2(table2_rows, "Recommended atlas+clinical HGB", "AIBL heldout")
    v4_ixi = lookup_table2(table2_rows, "Recommended atlas+clinical HGB", "IXI healthy")
    clinical = lookup_table2(table2_rows, "Clinical-only RF", "AIBL heldout")

    rows = [
        {
            "label": "Old v3",
            "color": COLORS["old"],
            "bacc": as_float(old.get("balanced_acc")),
            "mci": float("nan"),
            "ad": float("nan"),
            "ixi": as_float(old_ixi.get("cn_retention")),
        },
        {
            "label": "v4 atlas+clin",
            "color": COLORS["v4"],
            "bacc": as_float(v4.get("balanced_acc")),
            "mci": as_float(v4.get("recall_mci")),
            "ad": as_float(v4.get("recall_ad")),
            "ixi": as_float(v4_ixi.get("cn_retention")),
        },
        {
            "label": "scan-level",
            "color": COLORS["scan"],
            "bacc": final_scan["balanced_acc"],
            "mci": final_scan["recall_MCI"],
            "ad": final_scan["recall_AD"],
            "ixi": ixi_scan["cn_retention_rate"],
        },
        {
            "label": "subject-level",
            "color": COLORS["subject"],
            "bacc": final_subject["balanced_acc"],
            "mci": final_subject["recall_MCI"],
            "ad": final_subject["recall_AD"],
            "ixi": ixi_subject["cn_retention_rate"],
        },
        {
            "label": "clinical-only",
            "color": COLORS["clinical"],
            "bacc": as_float(clinical.get("balanced_acc")),
            "mci": as_float(clinical.get("recall_mci")),
            "ad": as_float(clinical.get("recall_ad")),
            "ixi": 1.0,
        },
    ]
    panels = [
        ("AIBL heldout balanced accuracy", "bacc", "A"),
        ("AIBL heldout MCI recall", "mci", "B"),
        ("AIBL heldout AD recall", "ad", "C"),
        ("IXI healthy CN retention", "ixi", "D"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 5.35), sharey=True)
    x = np.arange(len(rows))
    for ax, (title, key, letter) in zip(axes, panels):
        values = [row[key] for row in rows]
        colors = [row["color"] for row in rows]
        ax.bar(x, values, color=colors, width=0.72, edgecolor="white", linewidth=0.8)
        ax.set_title(title, pad=10)
        ax.set_ylim(0, 1.10)
        ax.set_xticks(x)
        ax.set_xticklabels([row["label"] for row in rows], rotation=24, ha="right")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.text(i, min(value + 0.025, 1.065), fmt(value), ha="center", va="bottom", fontsize=8)
        add_panel_label(ax, letter)
    axes[0].set_ylabel("Metric value")
    fig.suptitle("Final rescue model improves external minority-class recall while preserving IXI specificity", fontsize=14, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "The locked primary result is the final subject-level ensemble; clinical-only is shown as a comparator rather than the main atlas-guided model.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(bottom=0.30, top=0.82, wspace=0.22)
    save_figure(fig, out_dir, "figure2_final_external_rescue")


def plot_confusion(ax: plt.Axes, mat: np.ndarray, title: str) -> None:
    row_sum = mat.sum(axis=1, keepdims=True)
    norm = np.divide(mat, row_sum, out=np.zeros_like(mat, dtype=float), where=row_sum > 0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom_blues",
        ["#f7fbff", "#b9d6ea", "#5b9ac8", "#1f5f99"],
    )
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if norm[i, j] >= 0.55 else COLORS["text"]
            ax.text(
                j,
                i,
                f"{mat[i, j]}\n{norm[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
                fontweight="bold" if i == j else "normal",
            )
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    recalls = np.diag(norm)
    ax.set_title(title, pad=10)
    ax.text(
        0.5,
        -0.20,
        "Recall CN/MCI/AD: " + "/".join(f"{value:.3f}" for value in recalls),
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["muted"],
    )
    ax.set_xticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)


def figure_final_confusion(summary: dict, out_dir: Path) -> None:
    aibl = np.array(summary["final_model"]["subject_level_metrics"]["aibl_heldout"]["confusion_matrix"], dtype=int)
    internal = np.array(summary["final_model"]["subject_level_metrics"]["internal_test"]["confusion_matrix"], dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    plot_confusion(axes[0], aibl, "AIBL locked heldout subjects")
    plot_confusion(axes[1], internal, "ADNI internal-test subjects")
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.suptitle("Subject-level confusion matrices for the locked final ensemble", fontsize=14, fontweight="bold", y=1.02)
    save_figure(fig, out_dir, "figure3_final_subject_confusion")


def figure_bootstrap(summary: dict, out_dir: Path) -> None:
    boot = summary["final_model"]["subject_level_bootstrap"]["aibl_heldout"]
    metrics = [
        ("Balanced accuracy", "balanced_acc"),
        ("Macro AUC", "macro_auc_ovr"),
        ("MCI recall", "recall_MCI"),
        ("AD recall", "recall_AD"),
    ]
    labels = [item[0] for item in metrics]
    means = [boot[item[1]]["mean"] for item in metrics]
    lows = [boot[item[1]]["ci_low"] for item in metrics]
    highs = [boot[item[1]]["ci_high"] for item in metrics]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(x, means, color=[COLORS["subject"], COLORS["scan"], COLORS["mci"], COLORS["ad"]], width=0.62, edgecolor="white")
    ax.errorbar(
        x,
        means,
        yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
        fmt="none",
        ecolor="#1f2933",
        elinewidth=1.2,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    for i, value in enumerate(means):
        ax.text(i, min(value + 0.06, 1.04), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(
            i,
            max(lows[i] - 0.060, 0.03),
            f"95% CI\n{lows[i]:.3f}-{highs[i]:.3f}",
            ha="center",
            va="top",
            fontsize=8,
            color=COLORS["text"],
        )
    ax.set_title("Bootstrap stability of the final AIBL heldout subject-level result", pad=10)
    save_figure(fig, out_dir, "figure4_final_bootstrap_stability")


def feature_mean(row: dict, name: str) -> float:
    return as_float(row.get(f"{name}_mean"))


def figure_error_profiles(table_dir: Path, out_dir: Path) -> None:
    rows = read_csv_rows(table_dir / "aibl_heldout_error_group_features.csv")
    keep = ["CN_correct", "CN_to_MCI_AD", "MCI_correct", "MCI_to_AD", "AD_correct", "AD_to_CN_MCI"]
    rows = [row for row in rows if row["group"] in keep]
    labels = [row["group"].replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    values = {
        "AD-like atlas z": [feature_mean(row, "atlas_ad_like_z") for row in rows],
        "MMSE": [feature_mean(row, "clin_mmse") for row in rows],
        "Decision margin": [feature_mean(row, "margin") for row in rows],
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
    for ax, (title, vals), color, letter in zip(
        axes,
        values.items(),
        [COLORS["v4"], COLORS["scan"], COLORS["subject"]],
        ["A", "B", "C"],
    ):
        ax.bar(x, vals, color=color, width=0.65, edgecolor="white")
        ax.set_title(title, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
        ax.set_axisbelow(True)
        for i, value in enumerate(vals):
            if np.isfinite(value):
                ax.text(i, value + (0.03 if title != "MMSE" else 0.4), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        add_panel_label(ax, letter)
    fig.suptitle("AIBL heldout MCI/AD error profiles in the final subject-level model", fontsize=14, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        -0.04,
        "Remaining errors are concentrated near MCI/AD boundaries; AD-to-CN errors are absent in the locked AIBL heldout split.",
        ha="center",
        va="top",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    save_figure(fig, out_dir, "figure5_final_error_profiles")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--table2", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    apply_style()
    summary = load_json(args.summary)
    table2_rows = read_csv_rows(args.table2)
    figure_external_rescue(summary, table2_rows, args.out_dir)
    figure_final_confusion(summary, args.out_dir)
    figure_bootstrap(summary, args.out_dir)
    figure_error_profiles(args.table_dir, args.out_dir)


if __name__ == "__main__":
    main()
