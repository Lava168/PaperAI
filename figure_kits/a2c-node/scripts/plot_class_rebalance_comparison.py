#!/usr/bin/env python3
"""Class-imbalance comparison figure (Table tab:collapse) in unified Nature style.

Uses the same rcParams and palette as ``compose_paper_figures.py``.
Outputs:
  paper/figures/results/Figure_class_rebalance.{png,pdf}

* Bars: unweighted row from the manuscript table; rebalanced row from metrics.json.
* Left confusion matrix: schematic constant-MCI collapse (BAcc = 1/3 on this split).
* Right matrix: counts from confusion_matrix.csv, row-normalised to %.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from compose_paper_figures import NAT_PALETTE, apply_nature_rc, panel_label  # noqa: E402

UNWEIGHTED_TABLE = {
    "accuracy": 0.483,
    "balanced_accuracy": 0.333,
    "macro_f1": 0.217,
    "macro_auc": 0.517,
}


def load_confusion_csv(path: Path) -> np.ndarray:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return np.array([[int(x) for x in row] for row in rows[1:]], dtype=float)


def majority_mci_collapse_cm(support: np.ndarray) -> np.ndarray:
    k = len(support)
    cm = np.zeros((k, k), dtype=float)
    for i in range(k):
        cm[i, 1] = support[i]
    return cm


def row_percent(cm: np.ndarray) -> np.ndarray:
    s = cm.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return 100.0 * cm / s


def _cmap_warm() -> object:
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "nat_warm",
        ["#ffffff", NAT_PALETTE["panel"], NAT_PALETTE["mustard"], NAT_PALETTE["sienna"]],
    )


def _cmap_cool() -> object:
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "nat_cool",
        ["#ffffff", "#dce8e2", NAT_PALETTE["sage"], NAT_PALETTE["navy"]],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=PROJ / "paper/figures/results")
    ap.add_argument(
        "--metrics_json",
        type=Path,
        default=PROJ / "runs/v4_mm_analysis_npj_5seed/metrics.json",
    )
    ap.add_argument(
        "--confusion_csv",
        type=Path,
        default=PROJ / "runs/v4_mm_analysis_npj_5seed/confusion_matrix.csv",
    )
    args = ap.parse_args()

    apply_nature_rc()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    metrics = json.loads(args.metrics_json.read_text())
    per = metrics["per_class"]
    macro_f1 = float(np.mean([float(per[c]["F1"]) for c in ("CN", "MCI", "AD")]))

    weighted = {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": macro_f1,
        "macro_auc": float(metrics["macro_auc"]),
    }

    labels = ["CN", "MCI", "AD"]
    cm_data = load_confusion_csv(args.confusion_csv)
    support = cm_data.sum(axis=1)
    pct_collapse = row_percent(majority_mci_collapse_cm(support))
    pct_balanced = row_percent(cm_data)

    cmap_bad = _cmap_warm()
    cmap_good = _cmap_cool()

    fig = plt.figure(figsize=(7.2, 2.85))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.12, 1.0], wspace=0.42)

    # ----- (a) grouped bars -----
    ax0 = fig.add_subplot(gs[0])
    metric_names = ["Accuracy", "Bal. acc.", "Macro-F1", "Macro-AUC"]
    x = np.arange(len(metric_names))
    w = 0.34
    u_vals = [
        UNWEIGHTED_TABLE["accuracy"],
        UNWEIGHTED_TABLE["balanced_accuracy"],
        UNWEIGHTED_TABLE["macro_f1"],
        UNWEIGHTED_TABLE["macro_auc"],
    ]
    b_vals = [
        weighted["accuracy"],
        weighted["balanced_accuracy"],
        weighted["macro_f1"],
        weighted["macro_auc"],
    ]
    ax0.bar(
        x - w / 2,
        u_vals,
        width=w,
        label="Unweighted CE",
        color=NAT_PALETTE["fog"],
        edgecolor=NAT_PALETTE["slate"],
        linewidth=0.55,
        zorder=2,
    )
    ax0.bar(
        x + w / 2,
        b_vals,
        width=w,
        label="Inverse-frequency weight + sampler",
        color=NAT_PALETTE["navy"],
        edgecolor=NAT_PALETTE["ink"],
        linewidth=0.45,
        zorder=2,
    )
    ax0.set_xticks(x)
    ax0.set_xticklabels(metric_names)
    ax0.set_ylabel("Value (accuracy as proportion)")
    ax0.set_ylim(0, 1.02)
    ax0.axhline(
        1.0 / 3.0,
        color=NAT_PALETTE["sienna"],
        ls=(0, (3, 3)),
        lw=0.9,
        alpha=0.9,
        zorder=1,
    )
    ax0.text(
        3.42,
        1.0 / 3.0 + 0.028,
        "chance\n(BAcc)",
        fontsize=6.5,
        color=NAT_PALETTE["sienna"],
        va="bottom",
        ha="left",
    )
    ax0.legend(loc="upper left", ncol=1)
    ax0.set_title(
        "ADNI hold-out test set (n = 121)",
        loc="left",
        pad=5,
        color=NAT_PALETTE["slate"],
    )
    panel_label(ax0, "a", dx=-0.10, dy=1.05)

    # ----- (b) confusion matrices -----
    gs_r = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.38)

    def draw_cm(ax, P: np.ndarray, title: str, cmap) -> None:
        ax.imshow(P, vmin=0, vmax=100, cmap=cmap, aspect="equal")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.tick_params(length=2.5)
        for i in range(3):
            for j in range(3):
                v = P[i, j]
                tc = "#ffffff" if v > 52 else NAT_PALETTE["ink"]
                ax.text(
                    j, i, f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="medium",
                    color=tc,
                )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True class")
        ax.set_title(title, loc="left", pad=4, fontsize=8, color=NAT_PALETTE["slate"])

    ax_l = fig.add_subplot(gs_r[0, 0])
    ax_r = fig.add_subplot(gs_r[0, 1])
    draw_cm(
        ax_l,
        pct_collapse,
        "Unweighted (schematic: predict MCI)",
        cmap_bad,
    )
    draw_cm(
        ax_r,
        pct_balanced,
        "Rebalanced (5-seed ensemble)",
        cmap_good,
    )
    panel_label(ax_l, "b", dx=-0.28, dy=1.08)

    fig.text(
        0.52,
        0.02,
        "Left: row-normalised % under constant MCI (BAcc = 1/3). "
        "Right: empirical ensemble; values are % of each true class.",
        fontsize=6.5,
        color=NAT_PALETTE["slate"],
        ha="left",
        va="bottom",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / "Figure_class_rebalance.png"
    out_pdf = args.out_dir / "Figure_class_rebalance.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[class_rebalance] wrote {out_png} / {out_pdf}")


if __name__ == "__main__":
    main()
