#!/usr/bin/env python3
"""Figure 1 — Clinical consistency evidence for explainable AD imaging AI.

Panel layout fixes (vs. earlier draft):
  (a) Stacked bars and baseline→latest confusion matrix are side-by-side
      sub-axes — no overlay on the Latest-diagnosis bar.
  (b–d) Extra margins so FDR / correlation annotations do not overlap data.

Data: unique PTIDs in ``data/splits_v5_paired/train.csv`` (n=964) for panel (a);
      scan-level rows from the same training PTIDs for (b–d).

Run:
    python paper/figures/plot_figure1_clinical_consistency.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_PDF = SCRIPT_DIR / "Figure_1_AD_explainability_clinical_consistency.pdf"
OUT_PNG = SCRIPT_DIR / "Figure_1_AD_explainability_clinical_consistency.png"

DX_ORDER = ["CN", "MCI", "AD"]
DX_COLORS = {"CN": "#0072B2", "MCI": "#56B4E9", "AD": "#E69F00"}
DX_TO_IDX = {d: i for i, d in enumerate(DX_ORDER)}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _load_training_cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-PTID table + scan-level rows for the paired training split."""
    master = pd.read_csv(DATA_DIR / "master_subjects_v2.csv")
    train_scans = pd.read_csv(DATA_DIR / "splits_v5_paired" / "train.csv")
    ptids = train_scans["PTID"].unique()
    scans = master[master["PTID"].isin(ptids)].copy()
    subjects = scans.drop_duplicates("PTID").copy()
    return subjects, scans


def _plot_panel_a(fig: plt.Figure, gs_cell: gridspec.SubplotSpec, subjects: pd.DataFrame) -> None:
    inner = gs_cell.subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.55)
    ax_bars = fig.add_subplot(inner[0, 0])
    ax_cm = fig.add_subplot(inner[0, 1])

    bl_counts = subjects["DX_bl"].value_counts().reindex(DX_ORDER, fill_value=0).to_dict()
    lat_counts = subjects["DX_latest"].value_counts().reindex(DX_ORDER, fill_value=0).to_dict()

    # Two grouped stacked bars (not overlaid with the matrix).
    width = 0.36
    xs = np.array([0, 1])
    for j, (counts, xlabel) in enumerate([(bl_counts, "Baseline\ndiagnosis"),
                                          (lat_counts, "Latest\ndiagnosis")]):
        bottom = 0
        for dx in DX_ORDER:
            h = counts[dx]
            ax_bars.bar(xs[j], h, width=width, bottom=bottom, color=DX_COLORS[dx],
                        edgecolor="white", linewidth=0.5)
            if h > 0:
                ax_bars.text(xs[j], bottom + h / 2, str(h), ha="center", va="center",
                             fontsize=7,
                             color="white" if h > 70 else "#222222", zorder=5)
            bottom += h

    ax_bars.set_xticks(xs)
    ax_bars.set_xticklabels(["Baseline diagnosis", "Latest diagnosis"], fontsize=7.5)
    ax_bars.set_ylabel("Unique participants")
    ax_bars.set_xlim(-0.45, 1.45)
    ymax = max(sum(bl_counts.values()), sum(lat_counts.values()))
    ax_bars.set_ylim(0, ymax * 1.08)
    ax_bars.spines[["top", "right"]].set_visible(False)
    ax_bars.set_title("Subject-level training cohort", loc="left", pad=6)

    # Confusion matrix in its own axis (right of bars).
    cm = pd.crosstab(subjects["DX_bl"], subjects["DX_latest"])
    cm = cm.reindex(index=DX_ORDER, columns=DX_ORDER, fill_value=0).to_numpy()
    im = ax_cm.imshow(cm, cmap="Blues", aspect="equal", vmin=0, vmax=cm.max())
    for i in range(3):
        for j in range(3):
            v = int(cm[i, j])
            color = "white" if v > cm.max() * 0.55 else "#222222"
            ax_cm.text(j, i, str(v), ha="center", va="center", fontsize=8, color=color)
    ax_cm.set_xticks(range(3))
    ax_cm.set_xticklabels(DX_ORDER, fontsize=7)
    ax_cm.set_yticks(range(3))
    ax_cm.set_yticklabels(DX_ORDER, fontsize=7)
    ax_cm.set_xlabel("Latest diagnosis", labelpad=4, fontsize=7.5)
    ax_cm.set_ylabel("Baseline diagnosis", labelpad=4, fontsize=7.5)
    ax_cm.set_title("Baseline → latest transition", fontsize=8, pad=6)
    # Panel label on the outer left of the bar axis.
    ax_bars.text(-0.22, 1.06, "A", transform=ax_bars.transAxes,
                 fontsize=11, fontweight="bold", ha="left", va="bottom")

    # Shared diagnosis legend above both sub-panels.
    handles = [plt.Rectangle((0, 0), 1, 1, fc=DX_COLORS[d]) for d in DX_ORDER]
    fig.legend(handles, DX_ORDER, loc="upper left",
               bbox_to_anchor=(0.06, 0.93), ncol=3, frameon=False, fontsize=7)


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def _plot_panel_b(ax: plt.Axes, scans: pd.DataFrame) -> None:
    biomarkers = [
        ("CSF\nAbeta42", "CSF_ABETA42"),
        ("CSF\ntotal tau", "CSF_TAU"),
        ("CSF\np-tau", "CSF_PTAU"),
        ("Amyloid\ncentiloid", "AMY_CENTILOIDS"),
    ]
    positions = []
    pvals = []
    for k, (_, col) in enumerate(biomarkers):
        vals = scans[col].dropna()
        mu, sd = vals.mean(), vals.std(ddof=0)
        groups_z = []
        pos = []
        for i, dx in enumerate(DX_ORDER):
            g = scans.loc[scans["label"] == i, col].dropna()
            z = (g - mu) / sd if sd > 0 else g * 0
            groups_z.append(z.to_numpy())
            pos.append(k * 4 + i + 1)
            parts = ax.violinplot(
                [z.to_numpy()], positions=[pos[-1]], widths=0.78,
                showmeans=False, showmedians=False, showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(DX_COLORS[dx])
                body.set_edgecolor("#333333")
                body.set_alpha(0.85)
                body.set_linewidth(0.5)
        positions.extend(pos)
        if all(len(g) >= 4 for g in groups_z):
            pvals.append(stats.kruskal(*groups_z).pvalue)

    qvals = _fdr_bh(np.array(pvals)) if pvals else np.array([])
    ax.set_xticks([k * 4 + 2 for k in range(len(biomarkers))])
    ax.set_xticklabels([b[0] for b in biomarkers], fontsize=7)
    ax.set_ylabel("Standardized value (z-score)")
    ax.set_title("Disease-stage biomarker gradients", loc="left", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    for k, q in enumerate(qvals):
        x = k * 4 + 2
        y = ax.get_ylim()[1] * 0.97
        ax.text(x, y, "FDR p <1e-4" if q < 1e-4 else f"FDR p {q:.2g}",
                ha="center", va="top", fontsize=6.5, clip_on=False)
    ax.text(-0.12, 1.04, "B", transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="left", va="bottom")


def _plot_panel_c(ax: plt.Axes, scans: pd.DataFrame) -> None:
    var_map = {
        "Disease\nseverity": "label",
        "Age": "AGE",
        "Education": "EDUCATION",
        "APOE e4": "APOE_E4",
        "CSF\nAbeta42": "CSF_ABETA42",
        "CSF tau": "CSF_TAU",
        "CSF p-tau": "CSF_PTAU",
        "WMH": "WMH_TOTAL",
        "Amyloid\nCL": "AMY_CENTILOIDS",
        "Amyloid+": "AMY_STATUS",
    }
    labels = list(var_map.keys())
    mat = scans[list(var_map.values())].copy()
    mat.columns = labels
    corr = mat.corr(method="spearman")
    n = len(labels)
    data = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1):
            data[i, j] = corr.iloc[i, j]

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="equal")
    for i in range(n):
        for j in range(i + 1):
            v = data[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(v) > 0.55 else "#222222")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_title("Clinical-anchor correlation structure", loc="left", pad=6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman rho", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.text(-0.18, 1.04, "C", transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="left", va="bottom")
    ax.text(0.0, -0.22,
            "* FDR-adjusted p < 0.05; disease severity: CN=0, MCI=1, AD=2.",
            transform=ax.transAxes, fontsize=6.5, ha="left", va="top", color="#444444")


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    return float((diff > 0).mean() - (diff < 0).mean())


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 400, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        xs = rng.choice(x, size=len(x), replace=True)
        ys = rng.choice(y, size=len(y), replace=True)
        deltas.append(_cliffs_delta(xs, ys))
    deltas = np.asarray(deltas)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _plot_panel_d(ax: plt.Axes, scans: pd.DataFrame) -> None:
    features = [
        ("Amyloid centiloid", "AMY_CENTILOIDS"),
        ("CSF p-tau", "CSF_PTAU"),
        ("WMH burden", "WMH_TOTAL"),
        ("Age", "AGE"),
        ("CSF Abeta42", "CSF_ABETA42"),
        ("Education", "EDUCATION"),
        ("CSF total tau", "CSF_TAU"),
    ]
    rows = []
    for name, col in features:
        d = scans.dropna(subset=[col])
        ad = d.loc[d["label"] == 2, col].to_numpy()
        cn = d.loc[d["label"] == 0, col].to_numpy()
        if len(ad) < 5 or len(cn) < 5:
            continue
        delta, lo, hi = _bootstrap_ci(ad, cn)
        p = stats.mannwhitneyu(ad, cn, alternative="two-sided").pvalue
        rows.append({"name": name, "delta": delta, "lo": lo, "hi": hi, "p": p})

    df = pd.DataFrame(rows)
    q = _fdr_bh(df["p"].to_numpy())
    y = np.arange(len(df))[::-1]
    for rank, (_, row) in enumerate(df.iterrows()):
        yi = len(df) - 1 - rank
        color = DX_COLORS["AD"] if row["delta"] >= 0 else DX_COLORS["CN"]
        ax.plot([row["lo"], row["hi"]], [yi, yi], color="#333333", lw=1.0, zorder=1)
        ax.scatter(row["delta"], yi, s=36, c=color, edgecolors="#333333",
                   linewidths=0.5, zorder=3)
        ptxt = "FDR p <1e-4" if q[rank] < 1e-4 else (
            f"FDR p {q[rank]:.3f}" if q[rank] >= 0.001 else f"FDR p {q[rank]:.2g}"
        )
        ax.text(1.02, yi, ptxt, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.5, clip_on=False)

    ax.axvline(0, color="#333333", lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels(df["name"], fontsize=7)
    ax.set_xlabel("Cliff's delta, AD vs CN (95% bootstrap CI)")
    ax.set_title("Effect-size anchors for XAI validation", loc="left", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-1.05, 1.05)
    ax.margins(y=0.08)
    ax.text(-0.14, 1.04, "D", transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="left", va="bottom")


def main() -> None:
    subjects, scans = _load_training_cohort()

    fig = plt.figure(figsize=(12.5, 7.8), facecolor="white")
    fig.suptitle(
        "Clinical Consistency Evidence for Explainable Alzheimer's Disease Imaging AI",
        fontsize=12, fontweight="bold", y=0.98,
    )
    gs = fig.add_gridspec(
        2, 2,
        left=0.07, right=0.88, top=0.90, bottom=0.12,
        wspace=0.42, hspace=0.50,
        width_ratios=[1.05, 1.0],
        height_ratios=[1.0, 1.05],
    )

    _plot_panel_a(fig, gs[0, 0], subjects)
    _plot_panel_b(fig.add_subplot(gs[0, 1]), scans)
    _plot_panel_c(fig.add_subplot(gs[1, 0]), scans)
    _plot_panel_d(fig.add_subplot(gs[1, 1]), scans)

    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor="white", pad_inches=0.06, dpi=300)
    plt.close(fig)
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
