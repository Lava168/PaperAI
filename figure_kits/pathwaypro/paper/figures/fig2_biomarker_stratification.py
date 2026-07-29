#!/usr/bin/env python3
"""Figure 2 — Biomarker stratification on the PathwayPro paired cohort.

Six sub-panels:
    (a) CSF Aβ42 violin × diagnosis.
    (b) CSF p-tau violin × diagnosis.
    (c) Amyloid SUVR violin × diagnosis.
    (d) Centiloid distribution and 24-CL positivity threshold.
    (e) WMH burden (log-transformed) × diagnosis.
    (f) Forest plot of standardized effect sizes (Hedges' g) of AD-vs-CN
        contrasts on each biomarker, with 95% CI.

Data: ``data/master_subjects_v2.csv`` joined with paired splits.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MASTER_CSV = DATA_ROOT / "master_subjects_v2.csv"
SPLITS_DIR = DATA_ROOT / "splits_v5_paired"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _style import (  # noqa: E402
    DIAG_COLORS, PATHWAY, ACCENT, mm, style_setup,
    panel_label as _panel_label_shared, figure_title, takeaway_banner,
)

DIAG = {0: "CN", 1: "MCI", 2: "AD"}


def _panel_label(ax, text: str, dx=-0.10, dy=1.05):
    _panel_label_shared(ax, text, dx=dx, dy=dy)


def _load_paired() -> pd.DataFrame:
    splits = {s: pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in ("train", "val", "test")}
    paired = pd.concat(splits.values(), ignore_index=True)
    master = pd.read_csv(MASTER_CSV).drop_duplicates("PTID")
    extra = ["AMY_SUVR", "AMY_CENTILOIDS", "AMY_STATUS",
             "WMH_TOTAL", "CSF_ABETA42", "CSF_PTAU", "CSF_TAU", "AGE", "GENDER"]
    keep = ["PTID"] + [c for c in extra if c in master.columns]
    paired = paired.merge(master[keep], on="PTID", how="left", suffixes=("_p", ""))
    return paired


def _hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, float]:
    """Return (g, se(g)) for the Hedges' g unbiased standardised mean diff."""
    a = group_a[~np.isnan(group_a)]
    b = group_b[~np.isnan(group_b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    s_pool = math.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if s_pool == 0:
        return float("nan"), float("nan")
    d = (a.mean() - b.mean()) / s_pool
    J = 1 - 3 / (4 * (na + nb) - 9)
    g = J * d
    var_g = (na + nb) / (na * nb) + g ** 2 / (2 * (na + nb))
    return g, math.sqrt(var_g)


# -----------------------------------------------------------------------------
# Generic violin panel
# -----------------------------------------------------------------------------

def _violin_diag(ax, df: pd.DataFrame, col: str, title: str, ylabel: str,
                 transform=lambda x: x, log: bool = False):
    if col not in df.columns:
        ax.text(0.5, 0.5, f"{col} unavailable", transform=ax.transAxes,
                ha="center", va="center"); return
    parts = []
    pos = []
    colors = []
    ns = []
    for i, (k, name) in enumerate(DIAG.items()):
        vals = df.loc[df["label"] == k, col].dropna().to_numpy()
        if vals.size == 0:
            continue
        parts.append(transform(vals))
        pos.append(i)
        colors.append(DIAG_COLORS[name])
        ns.append(len(vals))
    if not parts:
        ax.text(0.5, 0.5, f"{col} unavailable", transform=ax.transAxes,
                ha="center", va="center"); return
    vp = ax.violinplot(parts, positions=pos, widths=0.7, showextrema=False, showmeans=False)
    for b, c in zip(vp["bodies"], colors):
        b.set_facecolor(c); b.set_alpha(0.5); b.set_edgecolor(c); b.set_linewidth(0.5)
    bp = ax.boxplot(parts, positions=pos, widths=0.2, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.0), showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor("white"); patch.set_edgecolor(c)

    for i, vals in enumerate(parts):
        jitter = np.random.default_rng(11 + i).normal(0, 0.04, size=len(vals))
        ax.scatter(pos[i] + jitter, vals, s=3.5, color=colors[i], alpha=0.20,
                   linewidths=0, zorder=2)

    ax.set_xticks(pos)
    ax.set_xticklabels([f"{DIAG[k]}\n(n={n})" for k, n in zip([list(DIAG.keys())[p] for p in pos], ns)],
                       fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    if log:
        ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)


# -----------------------------------------------------------------------------
# (d) Centiloid distribution
# -----------------------------------------------------------------------------

def panel_d_centiloid(ax, df: pd.DataFrame):
    col = "AMY_CENTILOIDS"
    if col not in df.columns or df[col].dropna().empty:
        ax.text(0.5, 0.5, "Centiloids unavailable", transform=ax.transAxes,
                ha="center", va="center"); return
    for k, name in DIAG.items():
        vals = df.loc[df["label"] == k, col].dropna().to_numpy()
        if vals.size == 0:
            continue
        ax.hist(vals, bins=30, alpha=0.45, color=DIAG_COLORS[name], label=f"{name} (n={vals.size})",
                histtype="stepfilled", edgecolor=DIAG_COLORS[name])
    ax.axvline(24, color="black", linestyle="--", linewidth=0.8)
    ax.text(24, ax.get_ylim()[1] * 0.95, "24 CL = amyloid+ cut-off",
            ha="left", va="top", fontsize=7, rotation=90)
    ax.set_xlabel("Centiloids")
    ax.set_ylabel("# subjects")
    ax.set_title("Amyloid burden (Centiloids)", pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.8)


# -----------------------------------------------------------------------------
# (f) Forest plot of AD-vs-CN effect sizes
# -----------------------------------------------------------------------------

def panel_f_forest(ax, df: pd.DataFrame):
    biomarkers = [
        ("CSF_ABETA42",   "CSF Aβ42"),
        ("CSF_PTAU",      "CSF p-tau"),
        ("CSF_TAU",       "CSF total tau"),
        ("AMY_SUVR",      "Amyloid SUVR"),
        ("AMY_CENTILOIDS", "Centiloids"),
        ("WMH_TOTAL",     "WMH (Fazekas total)"),
        ("AGE",           "Age"),
    ]
    rows = []
    for col, label in biomarkers:
        if col not in df.columns:
            continue
        a = df.loc[df["label"] == 2, col].dropna().to_numpy()
        c = df.loc[df["label"] == 0, col].dropna().to_numpy()
        g, se = _hedges_g(a, c)
        rows.append((label, g, se, len(a), len(c)))
    if not rows:
        ax.text(0.5, 0.5, "No biomarker overlap", transform=ax.transAxes,
                ha="center", va="center"); return
    ys = list(range(len(rows)))[::-1]
    g_lo = -2.0; g_hi = 2.0
    for y, (label, g, se, na, nc) in zip(ys, rows):
        lo = g - 1.96 * (se if not math.isnan(se) else 0)
        hi = g + 1.96 * (se if not math.isnan(se) else 0)
        col = "#D55E00" if g > 0 else "#0072B2"
        ax.errorbar(g, y, xerr=[[g - lo], [hi - g]],
                    fmt="o", color=col, capsize=2, lw=1.0, markersize=4)
        ax.text(g_hi, y, f"  g={g:+.2f}", ha="left", va="center", fontsize=6.8)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(g_lo, g_hi + 0.6)
    ax.axvline(0, color="#222", linewidth=0.7)
    ax.set_xlabel("Hedges' g  (AD − CN)")
    ax.set_title("Standardised AD-vs-CN effect sizes", pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.4)


def _panel_title_colored(ax, title: str, branch: str) -> None:
    """Title in pathway colour so that AD vs vascular markers are visually grouped."""
    ax.set_title(title, pad=4, color=PATHWAY[branch])


def main() -> None:
    style_setup()
    paired = _load_paired()

    fig = plt.figure(figsize=(mm(190), mm(225)))
    outer = gridspec.GridSpec(3, 2, figure=fig,
                              left=0.080, right=0.965, top=0.860, bottom=0.085,
                              wspace=0.35, hspace=0.85,
                              height_ratios=[1.0, 1.0, 1.0])

    ax_a = fig.add_subplot(outer[0, 0])
    _panel_label(ax_a, "a")
    _violin_diag(ax_a, paired, "CSF_ABETA42", "CSF Aβ42 (pg/mL)", "CSF Aβ42")
    _panel_title_colored(ax_a, "CSF Aβ42 (pg/mL)", "atrophy")

    ax_b = fig.add_subplot(outer[0, 1])
    _panel_label(ax_b, "b")
    _violin_diag(ax_b, paired, "CSF_PTAU", "CSF p-tau (pg/mL)", "CSF p-tau")
    _panel_title_colored(ax_b, "CSF p-tau (pg/mL)", "atrophy")

    ax_c = fig.add_subplot(outer[1, 0])
    _panel_label(ax_c, "c")
    _violin_diag(ax_c, paired, "AMY_SUVR", "Amyloid SUVR", "AMY_SUVR")
    _panel_title_colored(ax_c, "Amyloid SUVR", "atrophy")

    ax_d = fig.add_subplot(outer[1, 1])
    _panel_label(ax_d, "d")
    panel_d_centiloid(ax_d, paired)
    _panel_title_colored(ax_d, "Amyloid burden (Centiloids)", "atrophy")

    ax_e = fig.add_subplot(outer[2, 0])
    _panel_label(ax_e, "e")
    df_plot = paired.copy()
    df_plot["WMH_TOTAL"] = df_plot["WMH_TOTAL"].replace(0, np.nan)
    _violin_diag(ax_e, df_plot, "WMH_TOTAL", "WMH burden (log)", "WMH total", log=True)
    _panel_title_colored(ax_e, "WMH burden (log scale)", "vascular")

    ax_f = fig.add_subplot(outer[2, 1])
    _panel_label(ax_f, "f")
    panel_f_forest(ax_f, paired)

    figure_title(
        fig, 2,
        "Biomarker stratification: amyloid markers strongly separate AD/CN, "
        "while WMH burden runs along a distinct axis.",
        y_title=0.945, y_story=0.918,
    )
    # AD branch / Vascular branch legend on a single line below the story.
    fig.text(0.515, 0.898,
             "AD-branch markers (Aβ, p-tau, amyloid)",
             fontsize=7.6, color=PATHWAY["atrophy"], ha="right", va="top",
             fontweight="bold")
    fig.text(0.520, 0.898,
             "|",
             fontsize=8.0, color="#666", ha="center", va="top")
    fig.text(0.525, 0.898,
             "Vascular-branch marker (WMH)",
             fontsize=7.6, color=PATHWAY["vascular"], ha="left", va="top",
             fontweight="bold")
    takeaway_banner(
        fig,
        "AD biomarkers separate AD/CN (Hedges' g > 1); WMH does not — "
        "vascular pathology is orthogonal to diagnosis and demands a separate latent.",
        y=0.013,
    )

    out_pdf = SCRIPT_DIR / "fig2_biomarker_stratification.pdf"
    out_png = SCRIPT_DIR / "fig2_biomarker_stratification.png"
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  saved: {out_pdf.relative_to(PROJECT_ROOT)}")
    print(f"  saved: {out_png.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
