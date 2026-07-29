#!/usr/bin/env python3
"""Figure 6 — diversified visual forms for ensemble strategy comparison.

Panels (same locked metrics; varied chart types):
  a  BAcc — Cleveland lollipop
  b  MCI/AD recall — dumbbell
  c  AD→CN errors — discrete unit / stem tally
  d  IXI CN retention — horizontal range markers
  e  ECE & NLL — dual-axis slope + endpoint labels

Strategies: Best single · Arithmetic mean · Equal log-pooling · Full RC-SPE
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "reports" / "v6_algorithm_innovation" / "algorithm_ablation_table.csv"
OUT = ROOT / "reports" / "v6_final_model" / "figures"

VARIANTS = [
    ("Best single base model", "Best single"),
    ("Arithmetic mean ensemble", "Arith. mean"),
    ("Equal log-pooling", "Equal log-pool"),
    ("Full RC-SPE (subject-level)", "Full RC-SPE"),
]
COLORS = ["#90A4AE", "#78909C", "#546E7A", "#1565C0"]
HIGHLIGHT = "#1565C0"

SERIF_TTF = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
SERIF_BOLD = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf")


def apply_times_style() -> None:
    if SERIF_TTF.exists():
        font_manager.fontManager.addfont(str(SERIF_TTF))
        if SERIF_BOLD.exists():
            font_manager.fontManager.addfont(str(SERIF_BOLD))
        family = font_manager.FontProperties(fname=str(SERIF_TTF)).get_name()
    else:
        family = "Times New Roman"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", family, "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )


def force_times_on_figure(fig: plt.Figure) -> None:
    if not SERIF_TTF.exists():
        return
    fp = font_manager.FontProperties(fname=str(SERIF_TTF))
    fp_b = font_manager.FontProperties(fname=str(SERIF_BOLD)) if SERIF_BOLD.exists() else fp
    for artist in fig.findobj(match=lambda x: isinstance(x, matplotlib.text.Text)):
        try:
            wt = artist.get_fontweight()
            artist.set_fontproperties(fp_b if str(wt) in ("bold", "700") else fp)
        except Exception:
            artist.set_fontfamily("serif")


def load_rows() -> dict[str, dict]:
    with ABLATION.open() as f:
        return {r["variant"]: r for r in csv.DictReader(f)}


def fget(row: dict, key: str) -> float:
    return float(row[key])


def style_ax(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_facecolor("white")


def draw_a(ax, rows: dict) -> None:
    """Dot ranking with RC-SPE reference line and Δ vs Best single."""
    labels = [lab for _, lab in VARIANTS]
    vals = np.asarray([fget(rows[k], "aibl_bacc") for k, _ in VARIANTS], float)
    base = vals[0]
    x = np.arange(len(labels))

    ax.axhline(vals[-1], color=HIGHLIGHT, lw=1.0, ls="--", alpha=0.55, zorder=1)
    ax.fill_between([-0.4, len(labels) - 0.6], vals[-1], 1.0, color="#E3F2FD", alpha=0.55, zorder=0)

    for xi, v, col in zip(x, vals, COLORS):
        ax.vlines(xi, min(vals) - 0.02, v, color="#ECEFF1", lw=1.2, zorder=1)
        ax.scatter([xi], [v], s=120 if col == HIGHLIGHT else 78, color=col,
                   edgecolors="white", linewidths=1.0, zorder=3, marker="o")
        ax.text(xi, v + 0.028, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2,
                fontweight="bold" if col == HIGHLIGHT else "normal",
                color=HIGHLIGHT if col == HIGHLIGHT else "#37474F")
        if xi > 0:
            dlt = v - base
            ax.text(xi, min(vals) - 0.055, f"Δ={dlt:+.3f}", ha="center", va="top",
                    fontsize=5.8, color="#78909C")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_xlim(-0.45, len(labels) - 0.55)
    ax.set_ylim(min(vals) - 0.12, 1.02)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("a  AIBL BAcc (dot rank + RC-SPE line)", loc="left", fontweight="bold", pad=4)
    ax.text(0.98, 0.02, "band / dashed = Full RC-SPE", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.0, color="#78909C")
    style_ax(ax)


def draw_b(ax, rows: dict) -> None:
    """Dumbbell — MCI vs AD recall per strategy."""
    labels = [lab for _, lab in VARIANTS]
    mci = [fget(rows[k], "aibl_recall_MCI") for k, _ in VARIANTS]
    ad = [fget(rows[k], "aibl_recall_AD") for k, _ in VARIANTS]
    y = np.arange(len(labels))[::-1]

    for yi, m, a in zip(y, mci[::-1], ad[::-1]):
        ax.plot([m, a], [yi, yi], color="#CFD8DC", lw=2.0, zorder=1)
        ax.scatter([m], [yi], s=70, color="#E69F00", edgecolors="white", linewidths=0.7, zorder=3)
        ax.scatter([a], [yi], s=70, color="#D55E00", edgecolors="white", linewidths=0.7, zorder=3)
        ax.text(m - 0.03, yi + 0.18, f"{m:.2f}", ha="center", va="bottom", fontsize=6.0, color="#E69F00")
        ax.text(a + 0.03, yi + 0.18, f"{a:.2f}", ha="center", va="bottom", fontsize=6.0, color="#D55E00")

    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Recall")
    ax.set_title("b  MCI ↔ AD recall (dumbbell)", loc="left", fontweight="bold", pad=4)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#E69F00", markersize=7, label="MCI"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#D55E00", markersize=7, label="AD"),
        ],
        loc="lower right", frameon=False, fontsize=6.5,
    )
    style_ax(ax)


def draw_c(ax, rows: dict) -> None:
    """Icon matrix — each AD→CN error as one marker in a fixed 5-slot row."""
    labels = [lab for _, lab in VARIANTS]
    vals = [int(float(rows[k]["aibl_ad_to_cn_errors"])) for k, _ in VARIANTS]
    n_slot = 5
    y = np.arange(len(labels))[::-1]

    ax.set_xlim(-0.5, n_slot + 1.4)
    ax.set_ylim(-0.7, len(labels) - 0.25)
    for yi, v in zip(y, vals[::-1]):
        for i in range(n_slot):
            cx, cy = i + 0.5, yi
            if i < v:
                ax.scatter([cx], [cy], s=160, marker="X", color="#E53935",
                           linewidths=1.6, zorder=3)
            else:
                ax.scatter([cx], [cy], s=70, marker="o", facecolors="white",
                           edgecolors="#B0BEC5", linewidths=1.1, zorder=2)
        if v == 0:
            ax.add_patch(
                FancyBboxPatch(
                    (n_slot + 0.15, yi - 0.28), 1.05, 0.56,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor="#E8F5E9", edgecolor="#2E7D32", lw=1.0, zorder=2,
                )
            )
            ax.text(n_slot + 0.67, yi, "safe", ha="center", va="center",
                    fontsize=7.0, fontweight="bold", color="#2E7D32", zorder=3)
        else:
            ax.text(n_slot + 0.55, yi, f"n={v}", ha="left", va="center",
                    fontsize=7.5, fontweight="bold", color="#C62828")

    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1])
    ax.set_xticks(np.arange(n_slot) + 0.5)
    ax.set_xticklabels([str(i + 1) for i in range(n_slot)], fontsize=6.5)
    ax.set_xlabel("Error slot (X = AD→CN event)")
    ax.set_title("c  AD-to-CN errors (icon matrix)", loc="left", fontweight="bold", pad=4)
    style_ax(ax)


def draw_d(ax, rows: dict) -> None:
    """Horizontal progress / bullet bars for IXI CN retention."""
    labels = [lab for _, lab in VARIANTS]
    vals = np.asarray([fget(rows[k], "ixi_cn_retention") for k, _ in VARIANTS], float)
    y = np.arange(len(labels))[::-1]
    lo, hi = 0.990, 1.002

    # Track background
    for yi in y:
        ax.barh(yi, hi - lo, left=lo, height=0.42, color="#ECEFF1", edgecolor="none", zorder=1)
        ax.barh(yi, 1.0 - 0.995, left=0.995, height=0.42, color="#BBDEFB", edgecolor="none",
                alpha=0.55, zorder=2)

    for yi, v, col in zip(y, vals[::-1], COLORS[::-1]):
        ax.barh(yi, max(v - lo, 0), left=lo, height=0.42, color=col, edgecolor="white",
                linewidth=0.5, zorder=3, alpha=0.95)
        ax.plot([v, v], [yi - 0.28, yi + 0.28], color="white", lw=1.6, zorder=4)
        ax.plot([v, v], [yi - 0.28, yi + 0.28], color=col, lw=1.0, zorder=5)
        ax.text(hi + 0.0004, yi, f"{v:.3f}", va="center", ha="left", fontsize=7.0,
                fontweight="bold" if col == HIGHLIGHT else "normal",
                color=HIGHLIGHT if col == HIGHLIGHT else "#455A64")

    ax.axvline(1.0, color=HIGHLIGHT, lw=0.9, ls="--", alpha=0.75, zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1])
    ax.set_xlim(lo, hi + 0.004)
    ax.set_xlabel("IXI CN retention")
    ax.set_title("d  IXI CN retention (progress bars)", loc="left", fontweight="bold", pad=4)
    ax.text(0.02, 0.98, "blue band ≥0.995 · dashed = 1.0", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.8, color="#78909C")
    style_ax(ax)


def draw_e(ax, rows: dict) -> None:
    """Original dual bar chart: ECE & NLL."""
    labels = [lab for _, lab in VARIANTS]
    ece = [fget(rows[k], "aibl_ece") for k, _ in VARIANTS]
    nll = [fget(rows[k], "aibl_nll") for k, _ in VARIANTS]
    x = np.arange(len(labels))
    w = 0.36
    ax2 = ax.twinx()
    b1 = ax.bar(x - w / 2, ece, width=w, color="#7E57C2", edgecolor="white", label="ECE", zorder=2)
    b2 = ax2.bar(x + w / 2, nll, width=w, color="#26A69A", edgecolor="white", label="NLL", zorder=2)
    for bar in b1:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6.0, color="#5E35B1")
    for bar in b2:
        v = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=6.0, color="#00796B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("ECE", color="#5E35B1")
    ax2.set_ylabel("NLL", color="#00796B")
    ax.set_ylim(0, max(ece) * 1.45)
    ax2.set_ylim(0, max(nll) * 1.35)
    ax.set_title("e  Calibration (ECE & NLL)", loc="left", fontweight="bold", pad=4)
    style_ax(ax)
    ax2.spines["top"].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor="#7E57C2", edgecolor="white", label="ECE (↓ better)"),
            Patch(facecolor="#26A69A", edgecolor="white", label="NLL (↓ better)"),
        ],
        loc="upper right", frameon=False, fontsize=6.5,
    )


def main() -> None:
    apply_times_style()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    for key, _ in VARIANTS:
        if key not in rows:
            raise SystemExit(f"Missing variant in ablation table: {key}")

    fig = plt.figure(figsize=(13.0, 8.6), facecolor="white")
    gs = GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[1.05, 1.0],
        hspace=0.42,
        wspace=0.34,
        left=0.09,
        right=0.97,
        top=0.90,
        bottom=0.12,
    )

    draw_a(fig.add_subplot(gs[0, 0]), rows)
    draw_b(fig.add_subplot(gs[0, 1]), rows)
    draw_c(fig.add_subplot(gs[0, 2]), rows)
    draw_d(fig.add_subplot(gs[1, 0]), rows)
    draw_e(fig.add_subplot(gs[1, 1:]), rows)

    fig.suptitle(
        "Figure 6. External prediction performance across probability ensemble strategies",
        fontsize=12.5,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5, 0.018,
        "AIBL locked heldout (n=216) and IXI healthy controls. Full RC-SPE improves balanced accuracy and "
        "disease recall while eliminating AD→CN errors and preserving IXI CN retention with competitive calibration.",
        ha="center", va="bottom", fontsize=7.2, color="#607D8B",
    )

    force_times_on_figure(fig)
    png = OUT / "figure6_ensemble_external_comparison.png"
    pdf = OUT / "figure6_ensemble_external_comparison.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="white", pad_inches=0.05)
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")

    alt = ROOT / "reports" / "brain_figures_fcstyle" / "figures"
    alt.mkdir(parents=True, exist_ok=True)
    (alt / "Fig06_ensemble_external_comparison.png").write_bytes(png.read_bytes())
    (alt / "Fig06_ensemble_external_comparison.pdf").write_bytes(pdf.read_bytes())
    print(f"copied to {alt}")


if __name__ == "__main__":
    main()
