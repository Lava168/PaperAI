#!/usr/bin/env python3
"""Generate publication figures for the v4 AD rebuild experiments."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


CLASS_NAMES = ["CN", "MCI", "AD"]

COLORS = {
    "old": "#6f7782",
    "atlas": "#2f6f73",
    "cascade": "#d28b26",
    "main": "#276fbf",
    "clinical": "#7a4f9a",
    "biomarker": "#c44e52",
    "mci": "#d99b2b",
    "ad": "#b8463c",
    "cn": "#3d7f5f",
    "grid": "#d7dde5",
    "text": "#1f2933",
    "muted": "#6b7280",
}


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def metric_value(metrics: dict, *keys: str) -> float:
    for key in keys:
        value = metrics.get(key)
        if value is not None and value != "":
            return float(value)
    return float("nan")


def recall_value(metrics: dict, label: str) -> float:
    return float(metrics.get("per_class", {}).get(label, {}).get("recall") or 0.0)


def label_count_text(item: dict) -> str:
    labels = item.get("labels", {})
    return f"CN/MCI/AD {labels.get('CN', 0)}/{labels.get('MCI', 0)}/{labels.get('AD', 0)}"


def split_text(manifest: dict, split: str, title: str) -> str:
    item = manifest["split_counts"][split]
    return (
        f"{title}\n"
        f"{item['scans']} scans / {item['subjects']} subjects\n"
        f"{label_count_text(item)}"
    )


def split_line(manifest: dict, split: str, title: str) -> str:
    item = manifest["split_counts"][split]
    return f"{title}: {item['scans']} / {item['subjects']}"


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
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["text"],
    )


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    face: str,
    edge: str = "#7d8793",
) -> None:
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(rect)
    ax.text(
        x + 0.012,
        y + h - 0.028,
        title,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COLORS["text"],
    )
    wrapped = "\n".join(textwrap.fill(line, width=28) for line in body.split("\n"))
    ax.text(
        x + 0.012,
        y + h - 0.072,
        wrapped,
        ha="left",
        va="top",
        fontsize=7.4,
        color=COLORS["text"],
        linespacing=1.25,
    )


def arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#667085", lw=1.0, shrinkA=4, shrinkB=4),
    )


def figure_study_design(
    manifest: dict,
    hybrid: dict,
    biomarker: dict,
    out_dir: Path,
) -> None:
    main = hybrid["results"]["aibl_adapted"]["atlas_core_clinical__hgb"]["metrics"]
    aibl = main["aibl_heldout"]
    ixi = main["ixi_external"]
    oasis = main["oasis_external"]
    bio = biomarker["aibl_heldout"]["ad_key_volume_score"]

    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.02,
        0.965,
        "Rebuilt cross-cohort AD modeling workflow",
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
        ha="left",
        va="top",
    )
    ax.text(
        0.02,
        0.925,
        "Subject-level splits, locked external testing, healthy negative controls, and atlas-region biological validation.",
        fontsize=9,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )

    col_x = [0.03, 0.225, 0.42, 0.615, 0.81]
    w = 0.155
    h = 0.16
    ys = [0.70, 0.47, 0.24]

    draw_box(
        ax,
        col_x[0],
        ys[0],
        w,
        h,
        "Cohorts",
        "ADNI discovery\nAIBL adaptation and heldout\nIXI healthy controls\nOASIS transfer stress test",
        "#eef6f6",
        COLORS["atlas"],
    )
    draw_box(
        ax,
        col_x[0],
        ys[1],
        w,
        h,
        "Scale",
        (
            f"ADNI train {manifest['split_counts']['train']['scans']} scans\n"
            f"AIBL heldout {manifest['split_counts']['aibl_heldout']['scans']} scans\n"
            f"IXI {manifest['split_counts']['ixi_external']['scans']} healthy scans"
        ),
        "#f7f8fa",
    )
    draw_box(
        ax,
        col_x[0],
        ys[2],
        w,
        h,
        "Leakage control",
        "All splits are subject-level. AIBL heldout subjects are not used for model fitting or model selection.",
        "#f7f8fa",
    )

    draw_box(
        ax,
        col_x[1],
        ys[0],
        w,
        h,
        "ADNI splits",
        "scans / subjects\n"
        + split_line(manifest, "train", "Train")
        + "\n"
        + split_line(manifest, "val", "Val")
        + "\n"
        + split_line(manifest, "internal_test", "Test")
        + "\nClass mix in Table 1",
        "#eef3fb",
        COLORS["main"],
    )
    draw_box(
        ax,
        col_x[1],
        ys[1],
        w,
        h,
        "AIBL protocol",
        "scans / subjects\n"
        + split_line(manifest, "aibl_adapt_train", "Adapt train")
        + "\n"
        + split_line(manifest, "aibl_adapt_val", "Adapt val")
        + "\n"
        + split_line(manifest, "aibl_heldout", "Heldout"),
        "#eef3fb",
        COLORS["main"],
    )
    draw_box(
        ax,
        col_x[1],
        ys[2],
        w,
        h,
        "External controls",
        "scans / subjects\n"
        + split_line(manifest, "ixi_external", "IXI healthy")
        + "\n"
        + split_line(manifest, "oasis_external", "OASIS")
        + "\nClass mix in Table 1",
        "#eef3fb",
        COLORS["main"],
    )

    draw_box(
        ax,
        col_x[2],
        ys[0],
        w,
        h,
        "Atlas MRI",
        "21 regional volume and intensity features, including hippocampus, amygdala, and lateral ventricles.",
        "#f0f7f2",
        COLORS["cn"],
    )
    draw_box(
        ax,
        col_x[2],
        ys[1],
        w,
        h,
        "Core clinical",
        "Age, sex, education, APOE4, MMSE, and CDR-SB for the main multimodal atlas-guided model.",
        "#f0f7f2",
        COLORS["cn"],
    )
    draw_box(
        ax,
        col_x[2],
        ys[2],
        w,
        h,
        "Sensitivity inputs",
        "Extended cognition and biomarker variables are used only in comparator and sensitivity analyses.",
        "#f7f8fa",
    )

    draw_box(
        ax,
        col_x[3],
        ys[0],
        w,
        h,
        "Model family",
        "Atlas-only baseline\nCascade staging baseline\nAtlas+clinical HGB main model\nClinical-only RF comparator",
        "#fff5e6",
        COLORS["cascade"],
    )
    draw_box(
        ax,
        col_x[3],
        ys[1],
        w,
        h,
        "Selection rule",
        "Models are selected using validation/adaptation performance plus healthy-control specificity, then frozen.",
        "#fff5e6",
        COLORS["cascade"],
    )
    draw_box(
        ax,
        col_x[3],
        ys[2],
        w,
        h,
        "Stability",
        "The key hybrid candidates are repeated across seeds and reported with mean and standard deviation.",
        "#fff5e6",
        COLORS["cascade"],
    )

    draw_box(
        ax,
        col_x[4],
        ys[0],
        w,
        h,
        "External classification",
        (
            f"AIBL heldout BAcc {aibl['balanced_acc']:.3f}\n"
            f"AIBL AUC {aibl['macro_auc_ovr']:.3f}\n"
            f"IXI CN retention {ixi['cn_retention_rate']:.3f}"
        ),
        "#f4eff8",
        COLORS["clinical"],
    )
    draw_box(
        ax,
        col_x[4],
        ys[1],
        w,
        h,
        "Biological validation",
        (
            f"AD-key volume score {bio['ad_key_score']:.3f}\n"
            f"Uniform null {bio['uniform_null']:.3f}\n"
            f"Permutation p={bio['permutation_p_greater']:.4f}"
        ),
        "#f4eff8",
        COLORS["clinical"],
    )
    draw_box(
        ax,
        col_x[4],
        ys[2],
        w,
        h,
        "Boundary",
        f"OASIS remains a transfer stress test: BAcc {oasis['balanced_acc']:.3f}. It is reported as a limitation.",
        "#fbf1ef",
        COLORS["biomarker"],
    )

    for i in range(4):
        for y in [0.78, 0.55, 0.32]:
            arrow(ax, col_x[i] + w + 0.01, y, col_x[i + 1] - 0.01, y)

    save_figure(fig, out_dir, "figure1_revised_study_design")


def figure_external_improvement(
    old_v3: dict,
    feature: dict,
    cascade: dict,
    hybrid: dict,
    out_dir: Path,
) -> None:
    feature_best = feature["results"][feature["best_model"]]["metrics"]
    cascade_best = cascade["results"][cascade["best_model"]]["metrics"]
    h = hybrid["results"]["aibl_adapted"]

    rows = [
        {
            "label": "Old v3",
            "color": COLORS["old"],
            "aibl_bacc": metric_value(old_v3["ensemble"]["aibl"], "balanced_accuracy_present"),
            "aibl_auc": metric_value(old_v3["ensemble"]["aibl"], "macro_auc_ovr_valid"),
            "ixi_retention": metric_value(old_v3["ensemble"]["ixi"], "ixi_cn_retention_rate"),
        },
        {
            "label": "Atlas-only HGB",
            "color": COLORS["atlas"],
            "aibl_bacc": metric_value(feature_best["aibl_heldout"], "balanced_acc"),
            "aibl_auc": metric_value(feature_best["aibl_heldout"], "macro_auc_ovr"),
            "ixi_retention": metric_value(feature_best["ixi_external"], "cn_retention_rate"),
        },
        {
            "label": "Cascade RF-logreg",
            "color": COLORS["cascade"],
            "aibl_bacc": metric_value(cascade_best["aibl_heldout"], "balanced_acc"),
            "aibl_auc": metric_value(cascade_best["aibl_heldout"], "macro_auc_ovr"),
            "ixi_retention": metric_value(cascade_best["ixi_external"], "cn_retention_rate"),
        },
        {
            "label": "Atlas+clinical HGB",
            "color": COLORS["main"],
            "aibl_bacc": metric_value(h["atlas_core_clinical__hgb"]["metrics"]["aibl_heldout"], "balanced_acc"),
            "aibl_auc": metric_value(h["atlas_core_clinical__hgb"]["metrics"]["aibl_heldout"], "macro_auc_ovr"),
            "ixi_retention": metric_value(h["atlas_core_clinical__hgb"]["metrics"]["ixi_external"], "cn_retention_rate"),
        },
        {
            "label": "Clinical-only RF",
            "color": COLORS["clinical"],
            "aibl_bacc": metric_value(h["clinical_core_only__rf_balanced"]["metrics"]["aibl_heldout"], "balanced_acc"),
            "aibl_auc": metric_value(h["clinical_core_only__rf_balanced"]["metrics"]["aibl_heldout"], "macro_auc_ovr"),
            "ixi_retention": metric_value(h["clinical_core_only__rf_balanced"]["metrics"]["ixi_external"], "cn_retention_rate"),
        },
    ]

    panels = [
        ("AIBL heldout balanced accuracy", "aibl_bacc", "A"),
        ("AIBL heldout macro AUC", "aibl_auc", "B"),
        ("IXI healthy CN retention", "ixi_retention", "C"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), sharey=True)
    x = np.arange(len(rows))
    for ax, (title, key, letter) in zip(axes, panels):
        values = [row[key] for row in rows]
        colors = [row["color"] for row in rows]
        ax.bar(x, values, color=colors, width=0.72, edgecolor="white", linewidth=0.8)
        ax.set_title(title, pad=10)
        ax.set_ylim(0, 1.10)
        ax.set_xticks(x)
        ax.set_xticklabels([row["label"] for row in rows], rotation=27, ha="right")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
        ax.set_axisbelow(True)
        ax.axhline(0.5, color="#9aa4b2", lw=0.8, ls="--", zorder=0)
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.text(i, min(value + 0.025, 1.065), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        add_panel_label(ax, letter)
    axes[0].set_ylabel("Metric value")
    fig.suptitle(
        "External validation improvement after the v4 rebuild",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Clinical-only RF is shown as a comparator/upper bound; the main atlas-guided model is Atlas+clinical HGB.",
        ha="center",
        va="top",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    save_figure(fig, out_dir, "figure2_external_classification_improvement")


def confusion_matrix_from_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    mat = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    index = {label: i for i, label in enumerate(CLASS_NAMES)}
    for true_label, pred_label in zip(df["y_true"], df["y_pred"]):
        if true_label in index and pred_label in index:
            mat[index[true_label], index[pred_label]] += 1
    return mat


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


def figure_confusion_matrices(v4_root: Path, out_dir: Path) -> None:
    pred_dir = v4_root / "hybrid_atlas_clinical_baseline"
    main_path = pred_dir / "aibl_adapted_atlas_core_clinical__hgb_aibl_heldout_predictions.csv"
    clinical_path = pred_dir / "aibl_adapted_clinical_core_only__rf_balanced_aibl_heldout_predictions.csv"
    main_mat = confusion_matrix_from_csv(main_path)
    clinical_mat = confusion_matrix_from_csv(clinical_path)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    plot_confusion(axes[0], main_mat, "Atlas+clinical HGB main model")
    plot_confusion(axes[1], clinical_mat, "Clinical-only RF comparator")
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.suptitle("AIBL locked heldout confusion matrices", fontsize=14, fontweight="bold", y=1.02)
    save_figure(fig, out_dir, "figure3_aibl_confusion_matrices")


def short_region(region: str) -> str:
    replacements = {
        "L-Lat-Ventricle": "L lateral\nventricle",
        "R-Lat-Ventricle": "R lateral\nventricle",
        "L-Hippocampus": "L hippocampus",
        "R-Hippocampus": "R hippocampus",
        "L-Amygdala": "L amygdala",
        "R-Amygdala": "R amygdala",
    }
    return replacements.get(region, region.replace("-", " "))


def p_label(p_value: float) -> str:
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def figure_neurodegeneration(biomarker: dict, out_dir: Path) -> None:
    group_order = [
        ("aibl_heldout", "AIBL\nheldout"),
        ("aibl_adapt_heldout", "AIBL adapt\n+ heldout"),
        ("all_labeled_ad", "All labeled\nAD data"),
        ("adni_val_internal_test", "ADNI val\n+ test"),
    ]
    scores = []
    err_low = []
    err_high = []
    p_values = []
    for key, _ in group_order:
        score = biomarker[key]["ad_key_volume_score"]
        ci_low, ci_high = score["bootstrap_ci"]
        value = score["ad_key_score"]
        scores.append(value)
        err_low.append(value - ci_low)
        err_high.append(ci_high - value)
        p_values.append(score["permutation_p_greater"])
    uniform = biomarker["aibl_heldout"]["ad_key_volume_score"]["uniform_null"]

    regions = [
        "L-Lat-Ventricle",
        "R-Lat-Ventricle",
        "L-Hippocampus",
        "R-Hippocampus",
        "L-Amygdala",
        "R-Amygdala",
    ]
    gradients = biomarker["aibl_heldout"]["volume_gradients"]["ad_key"]
    mci_change = []
    ad_change = []
    rhos = []
    rho_p = []
    for region in regions:
        item = gradients[region]
        cn = item["CN"]
        denom = abs(cn) if abs(cn) > 1e-12 else 1.0
        mci_change.append(100.0 * (item["MCI"] - cn) / denom)
        ad_change.append(100.0 * (item["AD"] - cn) / denom)
        rhos.append(item["spearman_rho_label"])
        rho_p.append(item["spearman_p"])

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    x = np.arange(len(group_order))
    colors = [COLORS["atlas"] if p < 0.05 else "#9aa4b2" for p in p_values]
    axes[0].bar(x, scores, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].errorbar(x, scores, yerr=[err_low, err_high], fmt="none", ecolor=COLORS["text"], lw=1.0, capsize=3)
    axes[0].axhline(uniform, color=COLORS["biomarker"], ls="--", lw=1.0, label=f"Uniform null {uniform:.3f}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label for _, label in group_order])
    axes[0].set_ylabel("AD-key volume score")
    axes[0].set_ylim(0, max(scores) + 0.18)
    axes[0].set_title("AD-key signal concentration")
    axes[0].grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    for i, (score, p_value) in enumerate(zip(scores, p_values)):
        axes[0].text(i, score + err_high[i] + 0.025, p_label(p_value), ha="center", va="bottom", fontsize=7.5)
    add_panel_label(axes[0], "A")

    xr = np.arange(len(regions))
    width = 0.36
    axes[1].bar(xr - width / 2, mci_change, width, label="MCI vs CN", color=COLORS["mci"])
    axes[1].bar(xr + width / 2, ad_change, width, label="AD vs CN", color=COLORS["ad"])
    axes[1].axhline(0, color="#5b6570", lw=0.8)
    axes[1].set_xticks(xr)
    axes[1].set_xticklabels([short_region(region) for region in regions], rotation=25, ha="right")
    axes[1].set_ylabel("Volume change vs CN (%)")
    axes[1].set_title("AIBL heldout disease gradient")
    axes[1].grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    axes[1].legend(frameon=False, fontsize=8)
    add_panel_label(axes[1], "B")

    rho_colors = [COLORS["atlas"] if rho > 0 else COLORS["biomarker"] for rho in rhos]
    axes[2].bar(xr, rhos, color=rho_colors, edgecolor="white", linewidth=0.8)
    axes[2].axhline(0, color="#5b6570", lw=0.8)
    axes[2].set_xticks(xr)
    axes[2].set_xticklabels([short_region(region) for region in regions], rotation=25, ha="right")
    axes[2].set_ylabel("Spearman rho with CN-MCI-AD label")
    axes[2].set_title("Monotonic regional trends")
    axes[2].grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    for i, (rho, p_value) in enumerate(zip(rhos, rho_p)):
        va = "bottom" if rho >= 0 else "top"
        y = rho + (0.025 if rho >= 0 else -0.025)
        axes[2].text(i, y, p_label(p_value), ha="center", va=va, fontsize=7.2)
    add_panel_label(axes[2], "C")

    fig.suptitle(
        "Atlas neurodegeneration consistency validation (MRI proxy, not direct Braak staging)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    save_figure(fig, out_dir, "figure4_neurodegeneration_consistency")


def figure_oasis_stress_test(manifest: dict, hybrid: dict, out_dir: Path) -> None:
    runs = [
        ("Atlas+clinical\nHGB", "atlas_core_clinical__hgb", COLORS["main"]),
        ("Clinical-only\nRF", "clinical_core_only__rf_balanced", COLORS["clinical"]),
    ]
    metrics = hybrid["results"]["aibl_adapted"]
    evals = [
        ("AIBL heldout\nBAcc", "aibl_heldout", "balanced_acc"),
        ("IXI healthy\nCN retention", "ixi_external", "cn_retention_rate"),
        ("OASIS stress\nBAcc", "oasis_external", "balanced_acc"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), gridspec_kw={"width_ratios": [1.35, 1.0]})
    x = np.arange(len(evals))
    width = 0.34
    for offset, (label, run, color) in zip([-width / 2, width / 2], runs):
        values = [metrics[run]["metrics"][split][metric] for _, split, metric in evals]
        axes[0].bar(x + offset, values, width, label=label.replace("\n", " "), color=color, edgecolor="white", linewidth=0.8)
        for i, value in enumerate(values):
            axes[0].text(x[i] + offset, min(value + 0.025, 1.07), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].axhspan(0, 0.5, color="#fbf1ef", alpha=0.55, zorder=0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label for label, _, _ in evals])
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Metric value")
    axes[0].set_title("Strong AIBL/IXI results, weak OASIS transfer")
    axes[0].grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=8)
    add_panel_label(axes[0], "A")

    class_colors = [COLORS["cn"], COLORS["mci"], COLORS["ad"]]
    bottom = np.zeros(len(runs))
    x2 = np.arange(len(runs))
    for class_name, color in zip(CLASS_NAMES, class_colors):
        values = []
        for _, run, _ in runs:
            pred = metrics[run]["metrics"]["oasis_external"].get("prediction_distribution", {})
            values.append(int(pred.get(class_name, 0)))
        axes[1].bar(x2, values, bottom=bottom, color=color, label=class_name, edgecolor="white", linewidth=0.8)
        bottom += np.array(values)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([label for label, _, _ in runs])
    axes[1].set_ylabel("Predicted scans")
    oasis_labels = manifest["split_counts"]["oasis_external"]["labels"]
    axes[1].set_title(
        "OASIS predicted-label distribution\n"
        f"True CN/MCI/AD={oasis_labels.get('CN', 0)}/{oasis_labels.get('MCI', 0)}/{oasis_labels.get('AD', 0)}"
    )
    axes[1].grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    add_panel_label(axes[1], "B")

    fig.suptitle("Unresolved external transfer stress test", fontsize=14, fontweight="bold", y=1.02)
    save_figure(fig, out_dir, "figure5_oasis_stress_test")


def write_captions(out_dir: Path) -> None:
    captions = {
        "figure1_revised_study_design": (
            "Revised v4 workflow with subject-level splitting, AIBL adaptation and locked heldout testing, "
            "IXI healthy negative-control evaluation, OASIS stress testing, and atlas-region biological validation."
        ),
        "figure2_external_classification_improvement": (
            "External performance improvement across the failed v3 baseline, atlas-only and cascade baselines, "
            "the main atlas+clinical model, and a clinical-only comparator."
        ),
        "figure3_aibl_confusion_matrices": (
            "AIBL locked heldout confusion matrices for the main atlas+clinical HGB model and the clinical-only RF comparator."
        ),
        "figure4_neurodegeneration_consistency": (
            "MRI neurodegeneration consistency analysis showing AD-key volume signal concentration and AIBL heldout "
            "regional gradients in ventricles, hippocampus, and amygdala."
        ),
        "figure5_oasis_stress_test": (
            "OASIS stress-test results, reported explicitly as an unresolved transfer limitation rather than hidden."
        ),
    }
    path = out_dir / "figure_captions.json"
    path.write_text(json.dumps(captions, indent=2), encoding="utf-8")
    print(f"[saved] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    apply_style()
    root = args.v4_root
    manifest = load_json(root / "manifest_v4_summary.json")
    old_v3 = load_json(root.parent / "analysis" / "external_validation_v3_merged.json")
    feature = load_json(root / "atlas_feature_baseline" / "summary.json")
    cascade = load_json(root / "atlas_cascade_baseline" / "summary.json")
    hybrid = load_json(root / "hybrid_atlas_clinical_baseline" / "summary.json")
    biomarker = load_json(root / "atlas_feature_biomarkers" / "summary.json")

    figure_study_design(manifest, hybrid, biomarker, args.out_dir)
    figure_external_improvement(old_v3, feature, cascade, hybrid, args.out_dir)
    figure_confusion_matrices(root, args.out_dir)
    figure_neurodegeneration(biomarker, args.out_dir)
    figure_oasis_stress_test(manifest, hybrid, args.out_dir)
    write_captions(args.out_dir)


if __name__ == "__main__":
    main()
