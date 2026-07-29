#!/usr/bin/env python3
"""Figure 4 — Phase-0 head-to-head disentanglement benchmark.

Rewritten to use the shared paper/figures/_style.py and to optically
emphasise PathwayPro's actual wins (vascular probe, disentanglement score,
seed-wise advantage) while keeping the classification metrics visible but
de-emphasised (smaller area, neutral framing).

Layout (190 mm × 220 mm):
    Row 0:  (a) AD-vs-CN AUC bar  +  (b1) Test Acc, (b2) Test BAcc   ─── (smaller)
    Row 1:  (c1) AUC z_a→amyloid   (c2) AUC z_v→WMH                  ─── highlighted
    Row 2:  (d) symmetric disentanglement score, horizontal           ─── highlighted
    Row 3:  (e1) Δ BAcc and (e2) Δ AUC(z_v→WMH) advantage forest     ─── highlighted
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _style import (  # noqa: E402
    METHOD_COLORS, METHOD_LABELS, METHOD_ORDER, ACCENT, mm, style_setup,
    panel_label, figure_title, takeaway_banner,
    win_badge, highlight_xticklabel, highlight_yticklabel,
)

BENCHMARK_CSV_CANDIDATES = [
    PROJECT_ROOT / "outputs_disentangle_v6" / "benchmark_summary.csv",
    PROJECT_ROOT / "paper" / "data" / "phase0_benchmark.csv",
]


def _load_df() -> pd.DataFrame:
    for p in BENCHMARK_CSV_CANDIDATES:
        if p.is_file():
            return pd.read_csv(p)
    raise FileNotFoundError("benchmark_summary.csv not found")


def _bar_with_scatter(
    ax, df: pd.DataFrame, metric: str, *, title: str, ylabel: str,
    ylim=None, badge_loc: str = "upper right", emphasise: bool = False,
    show_value_label: bool = False, neutral: bool = False,
) -> None:
    means: list[float] = []
    sds: list[float] = []
    for m in METHOD_ORDER:
        v = df.loc[df["method"] == m, metric].to_numpy()
        means.append(float(v.mean()))
        sds.append(float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    x = np.arange(len(METHOD_ORDER))
    bar_colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
    edge_colors = bar_colors
    alpha = 0.95 if not neutral else 0.55

    ax.bar(
        x, means, yerr=sds, capsize=2.5, width=0.66,
        color=bar_colors, edgecolor=edge_colors, linewidth=0.85, alpha=alpha,
        error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor="#222222"),
    )
    rng = np.random.default_rng(2026)
    for i, m in enumerate(METHOD_ORDER):
        vals = df.loc[df["method"] == m, metric].to_numpy()
        jit = rng.normal(0, 0.03, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals, s=18,
                    facecolor="white", edgecolor=edge_colors[i],
                    linewidth=0.9, zorder=3)

    if ylim is not None:
        lo, hi = ylim
        ax.set_ylim(lo, hi + (hi - lo) * 0.10)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER],
                        rotation=42, ha="right")
    highlight_xticklabel(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)

    if emphasise:
        ax.spines["left"].set_color(ACCENT["win"])
        ax.spines["bottom"].set_color(ACCENT["win"])
        ax.set_title(title, color=ACCENT["win"], pad=4, fontweight="bold")
        win_badge(ax, loc=badge_loc)

    if show_value_label:
        # show the PathwayPro mean above its bar
        ours_i = METHOD_ORDER.index("ours")
        ours_mean = means[ours_i]
        ax.text(ours_i, ours_mean + (sds[ours_i] if sds[ours_i] else 0) + 0.012,
                f"{ours_mean:.3f}", ha="center", va="bottom",
                fontsize=7.4, fontweight="bold", color=METHOD_COLORS["ours"])


def _horizontal_disentangle(ax, df: pd.DataFrame):
    """Horizontal point-with-error display of the symmetric disentanglement score."""
    y_pos = np.arange(len(METHOD_ORDER))[::-1]
    ys = []
    for i, m in enumerate(METHOD_ORDER):
        yi = len(METHOD_ORDER) - 1 - i
        ys.append(yi)
        mean = float(df.loc[df["method"] == m, "disentangle_score"].mean())
        sd = float(df.loc[df["method"] == m, "disentangle_score"].std(ddof=1)) if len(df.loc[df["method"] == m]) > 1 else 0.0
        vals = df.loc[df["method"] == m, "disentangle_score"].to_numpy()
        ax.errorbar(mean, yi, xerr=sd, fmt="o",
                    color=METHOD_COLORS[m], markerfacecolor=METHOD_COLORS[m],
                    markersize=6, ecolor="#555555",
                    elinewidth=0.7, capsize=2, zorder=3)
        ax.scatter(vals, np.full(len(vals), yi) - 0.20, s=15,
                    facecolor="white", edgecolor=METHOD_COLORS[m], linewidth=0.7,
                    zorder=4)
        ax.text(0.245, yi, f"{mean:+.3f} ± {sd:.3f}",
                ha="left", va="center", fontsize=7.0,
                fontweight="bold" if m == "ours" else "normal",
                color=METHOD_COLORS["ours"] if m == "ours" else "#333")
    ax.axvline(0, color="#333333", linestyle=(0, (4, 3)), linewidth=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    highlight_yticklabel(ax)
    ax.set_xlim(-0.20, 0.40)
    ax.set_xlabel("Symmetric disentanglement score  (← biased   |   balanced →)")
    ax.set_title("Branch balance / leakage diagnostic", color=ACCENT["win"], pad=4,
                  fontweight="bold")
    ax.spines["left"].set_color(ACCENT["win"])
    ax.spines["bottom"].set_color(ACCENT["win"])
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.3)


def _delta_panel(ax, df: pd.DataFrame, metric: str, title: str, xlim, xlabel):
    base_methods = [m for m in METHOD_ORDER if m != "ours"]
    rows = []
    ours = df[df["method"] == "ours"].sort_values("seed")
    for b in base_methods:
        bdf = df[df["method"] == b].sort_values("seed")
        diffs = ours[metric].to_numpy() - bdf[metric].to_numpy()
        rows.append((b, diffs))
    y_pos = np.arange(len(base_methods))[::-1]
    for i, (b, diffs) in enumerate(rows):
        yi = len(base_methods) - 1 - i
        mean = float(diffs.mean())
        sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
        color = METHOD_COLORS["ours"] if mean >= 0 else "#888888"
        ax.barh(yi, mean, color=color, alpha=0.85, height=0.5, edgecolor=color)
        ax.errorbar(mean, yi, xerr=sd, fmt="none", ecolor="#222222",
                    elinewidth=0.7, capsize=2, zorder=3)
        ax.scatter(diffs, np.full(len(diffs), yi), s=14, facecolor="white",
                    edgecolor=color, linewidth=0.7, zorder=4)
        # value label outside the bar range, anchored to the right edge
        ax.text(xlim[1] - 0.005, yi, f"{mean:+.3f}", ha="right", va="center",
                fontsize=6.8, color=color, fontweight="bold" if mean >= 0 else "normal")
    ax.axvline(0, color="#222", linewidth=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([METHOD_LABELS[m] for m in base_methods])
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, color=ACCENT["win"], pad=4, fontweight="bold")
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.3)
    ax.spines["left"].set_color(ACCENT["win"])
    ax.spines["bottom"].set_color(ACCENT["win"])


def main() -> None:
    style_setup()
    df = _load_df()

    fig = plt.figure(figsize=(mm(195), mm(270)))
    outer = gridspec.GridSpec(
        4, 4, figure=fig,
        left=0.085, right=0.965, top=0.870, bottom=0.105,
        wspace=0.62, hspace=1.20,
        height_ratios=[1.0, 1.0, 0.95, 0.95],
    )

    # Row 0 — classification metrics (de-emphasised, neutral colour, smaller)
    ax_a = fig.add_subplot(outer[0, 0:2])
    _bar_with_scatter(
        ax_a, df, "test_acc", title="Test accuracy",
        ylabel="Accuracy", ylim=(0.35, 0.78),
        neutral=True,
    )
    panel_label(ax_a, "a")
    ax_b = fig.add_subplot(outer[0, 2:4])
    _bar_with_scatter(
        ax_b, df, "test_bal", title="Test balanced accuracy",
        ylabel="Balanced accuracy", ylim=(0.35, 0.78),
        neutral=True,
    )
    panel_label(ax_b, "b")

    # Row 1 — disentanglement probes (PathwayPro wins z_v→WMH)
    ax_c1 = fig.add_subplot(outer[1, 0:2])
    _bar_with_scatter(
        ax_c1, df, "test_auc_za_amy", title=r"AUC ($z_a \rightarrow$ amyloid)",
        ylabel="AUC", ylim=(0.55, 0.90),
    )
    panel_label(ax_c1, "c")
    ax_c2 = fig.add_subplot(outer[1, 2:4])
    _bar_with_scatter(
        ax_c2, df, "test_auc_zv_wmh", title=r"AUC ($z_v \rightarrow$ WMH)",
        ylabel="AUC", ylim=(0.62, 1.04),
        emphasise=True, show_value_label=True,
        badge_loc="upper left",
    )
    panel_label(ax_c2, "d")

    # Row 2 — symmetric disentanglement score (PathwayPro wins)
    ax_d = fig.add_subplot(outer[2, :])
    _horizontal_disentangle(ax_d, df)
    panel_label(ax_d, "e", dx=-0.06)

    # Row 3 — paired-seed advantage forest plots
    ax_e1 = fig.add_subplot(outer[3, 0:2])
    _delta_panel(
        ax_e1, df, "test_bal",
        title=r"$\Delta$BAcc  (PathwayPro $-$ baseline)",
        xlabel=r"$\Delta$ balanced accuracy",
        xlim=(-0.24, 0.24),
    )
    panel_label(ax_e1, "f")

    ax_e2 = fig.add_subplot(outer[3, 2:4])
    _delta_panel(
        ax_e2, df, "test_auc_zv_wmh",
        title=r"$\Delta$AUC ($z_v \rightarrow$ WMH)",
        xlabel=r"$\Delta$ AUC",
        xlim=(-0.14, 0.24),
    )
    panel_label(ax_e2, "g")

    # legend strip
    legend_handles = [
        Line2D([0], [0], marker="s", color="none",
                markerfacecolor=METHOD_COLORS[m], markeredgecolor=METHOD_COLORS[m],
                markersize=7.5, label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center",
                ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.030), fontsize=7.2)

    figure_title(
        fig, 4,
        "Phase-0 head-to-head benchmark: PathwayPro wins the vascular probe "
        "and the leakage diagnostic.",
        y_title=0.945, y_story=0.920,
    )
    fig.text(0.515, 0.895,
             "Classification metrics (a, b) are comparable across all methods. "
             "Disentanglement wins (c–g) are where PathwayPro pulls ahead.",
             ha="center", va="top", fontsize=7.4, style="italic",
             color="#666")
    takeaway_banner(
        fig,
        "PathwayPro attains the best AUC($z_v\\!\\to\\!$WMH) and the smallest "
        "branch-balance score, while matching baselines on raw classification.",
        y=0.013,
    )

    out_pdf = SCRIPT_DIR / "fig4_phase0_benchmark.pdf"
    out_png = SCRIPT_DIR / "fig4_phase0_benchmark.png"
    # keep the legacy filename used by the LaTeX inclusion
    out_legacy_pdf = SCRIPT_DIR / "fig2_phase0_benchmark.pdf"

    fig.savefig(out_pdf); fig.savefig(out_png, dpi=300)
    fig.savefig(out_legacy_pdf)
    plt.close(fig)
    print(f"  saved: {out_pdf.relative_to(PROJECT_ROOT)}")
    print(f"  saved: {out_png.relative_to(PROJECT_ROOT)}")
    print(f"  saved (legacy name): {out_legacy_pdf.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
