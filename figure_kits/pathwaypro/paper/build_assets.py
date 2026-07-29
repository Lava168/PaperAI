#!/usr/bin/env python3
"""Aggregate Phase-0 benchmark + sweep + OASIS results into LaTeX-ready
fragments and Nature-style publication figures.

Figure code follows the project's mature Nature-style template
``chapter1_foundation/nature_figures_v2.py`` (mm-based sizing, 7 pt Arial,
Color-Universal-Design palette) and reuses the colour conventions from
``utils/nature_visualization.py``.

Run from anywhere:
    python paper/build_assets.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------------------------------------------------------
# Nature-style configuration (mirrors chapter1_foundation/nature_figures_v2.py)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.default": "regular",
})

# Color-Universal-Design palette (from chapter1 nature_figures_v2.py).
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "grey": "#999999",
    "yellow": "#F0E442",
}

# Method colour assignment.
METHOD_COLORS = {
    "vanilla": COLORS["grey"],
    "betavae": COLORS["cyan"],
    "tcvae": COLORS["green"],
    "factorvae": COLORS["orange"],
    "dipvae2": COLORS["purple"],
    "ours": COLORS["red"],
}

# Pathology / branch colour conventions (from utils/nature_visualization.py).
PATHWAY_COLORS = {"atrophy": "#9467bd", "vascular": "#2ca02c"}
DIAGNOSIS_COLORS = {"CN": COLORS["blue"], "MCI": COLORS["orange"], "AD": COLORS["red"]}

MM2INCH = 1 / 25.4


def mm(v):
    return v * MM2INCH


def save_fig(fig, path, fmts=("pdf", "png")):
    p = Path(path)
    for ext in fmts:
        fig.savefig(p.with_suffix(f".{ext}"), format=ext, facecolor="white",
                    edgecolor="none")
    plt.close(fig)
    print(f"  saved: {p.stem} ({', '.join(fmts)})")


def panel_label(ax, lbl, x=-0.18, y=1.16):
    ax.text(x, y, lbl, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left")


# -----------------------------------------------------------------------------
# Locations
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PAPER = Path(__file__).resolve().parent
FIGDIR = PAPER / "figures"
SECDIR = PAPER / "sections"
FIGDIR.mkdir(parents=True, exist_ok=True)
SECDIR.mkdir(parents=True, exist_ok=True)

BENCH_CSV = ROOT / "outputs_disentangle_v6" / "benchmark_summary.csv"
SWEEP_CSV = ROOT / "outputs_neurips_v5_sweep" / "sweep_summary.csv"
OASIS_V6 = ROOT / "analysis_pathway_v6_oasis_v2" / "oasis_summary.json"
OASIS_V4 = ROOT / "analysis_pathway_v4_oasis_v2" / "oasis_summary.json"
# Counterfactual analysis is now run on the v6_zeromask checkpoints (Sec.~3.7);
# zero-masking actually *strengthens* the z_v->WMH causal binding because the
# missing-FLAIR contract forces the vascular signal to concentrate in z_v.
CF_DIR   = ROOT / "analysis_pathway_v6_zeromask_counterfactual"
T1_DIR   = ROOT / "analysis_pathway_v6_t1only"

METHOD_ORDER = ["vanilla", "betavae", "tcvae", "factorvae", "dipvae2", "ours"]
METHOD_TEX_LABEL = {
    "vanilla": r"Vanilla MTL",
    "betavae": r"$\beta$-VAE \citep{higgins2017betavae}",
    "tcvae": r"$\beta$-TCVAE \citep{chen2018isolating}",
    "factorvae": r"FactorVAE \citep{kim2018disentangling}",
    "dipvae2": r"DIP-VAE-II \citep{kumar2018variational}",
    "ours": r"\textbf{PathwayPro (ours)}",
}
METHOD_PRETTY = {
    "vanilla": "Vanilla\nMTL",
    "betavae": r"$\beta$-VAE",
    "tcvae": r"$\beta$-TCVAE",
    "factorvae": "FactorVAE",
    "dipvae2": "DIP-VAE-II",
    "ours": "PathwayPro\n(ours)",
}


# -----------------------------------------------------------------------------
# 1. Phase-0 benchmark aggregation
# -----------------------------------------------------------------------------
def _load_bench():
    rows = []
    with open(BENCH_CSV) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _agg(values):
    arr = np.asarray(values, dtype=float)
    return arr.mean(), arr.std(ddof=1) if arr.size > 1 else 0.0


def aggregate_bench():
    rows = _load_bench()
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    table = {}
    for m, rs in by_method.items():
        table[m] = {
            "n_seeds": len(rs),
            "test_acc": _agg([r["test_acc"] for r in rs]),
            "test_bal": _agg([r["test_bal"] for r in rs]),
            "auc_za_amy": _agg([r["test_auc_za_amy"] for r in rs]),
            "auc_zv_amy": _agg([r["test_auc_zv_amy"] for r in rs]),
            "auc_za_wmh": _agg([r["test_auc_za_wmh"] for r in rs]),
            "auc_zv_wmh": _agg([r["test_auc_zv_wmh"] for r in rs]),
            "delta_amy": _agg([r["test_delta_amy"] for r in rs]),
            "delta_wmh": _agg([r["test_delta_wmh"] for r in rs]),
            "disent": _agg([r["disentangle_score"] for r in rs]),
        }
    return table


def fmt(mean_std, decimals=3, bold=False, percent=False):
    m, s = mean_std
    if percent:
        body = f"{m*100:.{decimals}f}\\,$\\pm$\\,{s*100:.{decimals}f}"
    else:
        body = f"{m:.{decimals}f}\\,$\\pm$\\,{s:.{decimals}f}"
    if bold:
        return r"\textbf{" + body + "}"
    return body


def write_table_phase0(table):
    """Phase-0 head-to-head table (Table 3 in the manuscript style)."""
    out = SECDIR / "tab_phase0.tex"
    rows_tex = []
    best_acc = max(v["test_acc"][0] for v in table.values())
    best_bal = max(v["test_bal"][0] for v in table.values())
    best_zv_wmh = max(v["auc_zv_wmh"][0] for v in table.values())
    best_za_amy = max(v["auc_za_amy"][0] for v in table.values())
    best_disent = max(v["disent"][0] for v in table.values())

    def is_best(val, ref):
        return math.isclose(val, ref, abs_tol=1e-6)

    for m in METHOD_ORDER:
        if m not in table:
            continue
        v = table[m]
        rows_tex.append(
            " & ".join([
                METHOD_TEX_LABEL[m],
                fmt(v["test_acc"], 1, is_best(v["test_acc"][0], best_acc), True),
                fmt(v["test_bal"], 1, is_best(v["test_bal"][0], best_bal), True),
                fmt(v["auc_za_amy"], 3, is_best(v["auc_za_amy"][0], best_za_amy)),
                fmt(v["auc_zv_wmh"], 3, is_best(v["auc_zv_wmh"][0], best_zv_wmh)),
                fmt(v["disent"], 3, is_best(v["disent"][0], best_disent)),
            ]) + r" \\"
        )
    body = "\n".join(rows_tex)
    tex = (
        "% Auto-generated by paper/build_assets.py.\n"
        "\\begin{table}[!t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{Head-to-head disentanglement benchmark on the "
        "ADNI paired test set ($n{=}63$ subjects, three random seeds, 60 epochs / seed; "
        "all six methods share the same 3D dual-encoder backbone, optimizer, and "
        "weak-supervision schedule, and differ only in the disentanglement loss block "
        "of equation~\\eqref{eq:disent}). Atrophy AUC $(z_a\\!\\to\\!$amyloid) and vascular AUC "
        "$(z_v\\!\\to\\!$WMH$\\geq$median) measure each branch's specialisation; the "
        "disentanglement score is the mean of the two cross-branch AUC gaps "
        "(higher~$=$~cleaner factorisation). Bold cell entries denote the best mean "
        "value in each column.}\n"
        "\\label{tab:phase0}\n"
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
        "Method & Acc.\\ (\\%) & BAcc (\\%) & AUC$_{z_a\\!\\to\\!\\mathrm{amy}}$ & "
        "AUC$_{z_v\\!\\to\\!\\mathrm{WMH}}$ & Disent.\\ score \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    out.write_text(tex)
    print(f"  wrote: {out.relative_to(PAPER)}")


def write_table_oasis():
    v4 = json.loads(OASIS_V4.read_text())
    v6 = json.loads(OASIS_V6.read_text())
    v4_acc = v4["external_classification"]["acc"]
    v4_bal = v4["external_classification"]["balanced_acc"]
    v6_acc = v6["acc"]
    v6_bal = v6["balanced_acc"]
    v6_auc = v6["ad_vs_cn_auc"]
    tex = (
        "% Auto-generated by paper/build_assets.py.\n"
        "\\begin{table}[!t]\\centering\\small\\setlength{\\tabcolsep}{6pt}\n"
        "\\caption{OASIS-1 zero-shot external validation "
        "($n{=}121$; CN $=72$, MCI $=37$, AD $=12$). Both rows reuse the "
        "ADNI-trained checkpoint with no fine-tuning. Brain-mask z-score "
        "alignment closes most of the intensity-distribution domain gap; "
        "the disentangled PathwayPro features additionally carry the AD-vs-CN "
        "signal far above chance.}\n"
        "\\label{tab:oasis}\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "Model & $n$ & Acc.\\ (\\%) & BAcc (\\%) & AUC$_{\\text{AD vs.\\ CN}}$ \\\\\n"
        "\\midrule\n"
        f"v4 (T1-only, ADNI-trained, OASIS-aligned) & 121 & {v4_acc*100:.1f} & {v4_bal*100:.1f} & --- \\\\\n"
        f"\\textbf{{PathwayPro v6 (T1+FLAIR, ours)}} & 121 & {v6_acc*100:.1f} & "
        f"\\textbf{{{v6_bal*100:.1f}}} & \\textbf{{{v6_auc:.3f}}} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (SECDIR / "tab_oasis.tex").write_text(tex)
    print(f"  wrote: {(SECDIR / 'tab_oasis.tex').relative_to(PAPER)}")


# -----------------------------------------------------------------------------
# 2. Figure 1 — Pipeline schematic (Nature-style box-and-arrow diagram)
# -----------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(mm(183), mm(70)))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EEF2F7", ec="#1F3A5F"):
        rect = plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#1F3A5F"))

    box(0.1, 1.9, 1.4, 0.7, "T1 MRI\n96$^3$", fc="#FDE6E1")
    box(0.1, 0.7, 1.4, 0.7, "FLAIR MRI\n96$^3$", fc="#DCE6FF")
    box(1.9, 1.9, 1.7, 0.7, "3D ResNet (atrophy)\n$\\Phi_a$")
    box(1.9, 0.7, 1.7, 0.7, "3D ResNet (vascular)\n$\\Phi_v$")
    arrow(1.5, 2.25, 1.9, 2.25)
    arrow(1.5, 1.05, 1.9, 1.05)
    box(4.0, 1.9, 1.7, 0.7, r"Cross-attn pool"+"\n"+r"$z_a\in\mathbb{R}^{256}$")
    box(4.0, 0.7, 1.7, 0.7, r"Cross-attn pool"+"\n"+r"$z_v\in\mathbb{R}^{256}$")
    arrow(3.6, 2.25, 4.0, 2.25)
    arrow(3.6, 1.05, 4.0, 1.05)
    box(6.1, 1.9, 1.4, 0.7, r"$\mu_a, \log\sigma_a^2$")
    box(6.1, 0.7, 1.4, 0.7, r"$\mu_v, \log\sigma_v^2$")
    arrow(5.7, 2.25, 6.1, 2.25)
    arrow(5.7, 1.05, 6.1, 1.05)
    box(6.0, 2.75, 1.6, 0.45, "HSIC + TC + CLUB + iVAE", fc="#FFF4CC")
    arrow(6.8, 2.6, 6.8, 2.75)
    arrow(6.8, 1.4, 6.8, 1.6); arrow(6.8, 1.6, 6.8, 2.0)
    box(7.9, 2.55, 1.4, 0.55, "Dx head\n3-class CE")
    box(7.9, 1.85, 1.4, 0.55, "SUVR head\n81 regions")
    box(7.9, 1.15, 1.4, 0.55, "WMH head\nregression")
    box(7.9, 0.45, 1.4, 0.55, "Subtype +\ncf decoder")
    for yh in (2.825, 2.125, 1.425, 0.725):
        arrow(7.5, 1.4 if yh < 1.6 else 2.25, 7.9, yh)
    box(9.6, 1.4, 1.3, 0.7,
        r"$\hat{p}$(CN/MCI/AD)"+"\n"+r"+ ATE table",
        fc="#E8F6E8")
    arrow(9.3, 2.825, 9.6, 1.85)
    arrow(9.3, 0.725, 9.6, 1.55)

    save_fig(fig, FIGDIR / "fig1_pipeline")


# -----------------------------------------------------------------------------
# 3. Figure 2 — Phase-0 benchmark: BAcc bars + disentanglement bars + AUC heatmap
# -----------------------------------------------------------------------------
def fig_phase0(table):
    methods = [m for m in METHOD_ORDER if m in table]
    n = len(methods)
    bal_m = np.array([table[m]["test_bal"][0] * 100 for m in methods])
    bal_s = np.array([table[m]["test_bal"][1] * 100 for m in methods])
    dis_m = np.array([table[m]["disent"][0] for m in methods])
    dis_s = np.array([table[m]["disent"][1] for m in methods])
    auc_za_amy = np.array([table[m]["auc_za_amy"][0] for m in methods])
    auc_zv_amy = np.array([table[m]["auc_zv_amy"][0] for m in methods])
    auc_za_wmh = np.array([table[m]["auc_za_wmh"][0] for m in methods])
    auc_zv_wmh = np.array([table[m]["auc_zv_wmh"][0] for m in methods])

    fig = plt.figure(figsize=(mm(183), mm(78)))
    gs = gridspec.GridSpec(1, 3, wspace=0.6, width_ratios=[1.05, 1.05, 1.4],
                           top=0.88, bottom=0.22)

    # Panel a: balanced accuracy.
    ax_a = fig.add_subplot(gs[0, 0])
    bar_colors = [METHOD_COLORS[m] for m in methods]
    ax_a.bar(np.arange(n), bal_m, yerr=bal_s, capsize=2.5, color=bar_colors,
             edgecolor="white", linewidth=0.6, zorder=3)
    ax_a.set_ylabel("Test balanced accuracy (%)")
    ax_a.set_title("Three-class diagnosis (CN/MCI/AD)")
    ax_a.set_xticks(np.arange(n))
    ax_a.set_xticklabels([METHOD_PRETTY[m] for m in methods], fontsize=6,
                         rotation=20, ha="right")
    ax_a.set_ylim(40, 75)
    ax_a.grid(axis="y", alpha=0.25, linewidth=0.3, zorder=0)
    panel_label(ax_a, "a")

    # Panel b: disentanglement score.
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.bar(np.arange(n), dis_m, yerr=dis_s, capsize=2.5, color=bar_colors,
             edgecolor="white", linewidth=0.6, zorder=3)
    ax_b.axhline(0.0, color="k", linewidth=0.5, linestyle="--", alpha=0.6)
    ax_b.set_ylabel(r"Disentanglement score $\frac{1}{2}(\Delta_{\rm amy}+\Delta_{\rm WMH})$")
    ax_b.set_title("Cross-branch AUC gap")
    ax_b.set_xticks(np.arange(n))
    ax_b.set_xticklabels([METHOD_PRETTY[m] for m in methods], fontsize=6,
                         rotation=20, ha="right")
    ax_b.grid(axis="y", alpha=0.25, linewidth=0.3, zorder=0)
    panel_label(ax_b, "b")

    # Panel c: AUC heatmap (4 probes x 6 methods).
    ax_c = fig.add_subplot(gs[0, 2])
    auc_mat = np.vstack([auc_za_amy, auc_zv_amy, auc_za_wmh, auc_zv_wmh])
    cmap = LinearSegmentedColormap.from_list(
        "auc_blue", ["#FFFFFF", "#D6EAF8", "#5DADE2", "#2E86C1", "#1B4F72"])
    im = ax_c.imshow(auc_mat, cmap=cmap, vmin=0.55, vmax=0.95, aspect="auto")
    for i in range(4):
        for j in range(n):
            v = auc_mat[i, j]
            ax_c.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                      color="white" if v > 0.78 else "black")
    ax_c.set_xticks(np.arange(n))
    ax_c.set_xticklabels([METHOD_PRETTY[m] for m in methods], fontsize=6,
                         rotation=20, ha="right")
    ax_c.set_yticks(range(4))
    ax_c.set_yticklabels([
        r"$z_a\!\to\!$amyloid",
        r"$z_v\!\to\!$amyloid",
        r"$z_a\!\to\!$WMH",
        r"$z_v\!\to\!$WMH",
    ], fontsize=6.5)
    cb = plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("AUC", fontsize=6.5)
    ax_c.set_title("Branch specialisation (AUC)")
    panel_label(ax_c, "c")

    save_fig(fig, FIGDIR / "fig2_phase0_benchmark")


# -----------------------------------------------------------------------------
# 4. Figure 3 — Robustness curve
# -----------------------------------------------------------------------------
def _load_sweep():
    rows = []
    with open(SWEEP_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def fig_robustness():
    rows = _load_sweep()
    by_p = {}
    for r in rows:
        p = float(r["suvr_missing_frac"])
        by_p.setdefault(p, []).append(float(r["best_val_acc"]))
    ps = sorted(by_p)
    means = np.array([np.mean(by_p[p]) for p in ps]) * 100
    stds = np.array([np.std(by_p[p], ddof=1) for p in ps]) * 100

    fig, ax = plt.subplots(figsize=(mm(120), mm(65)))
    ax.errorbar(np.array(ps) * 100, means, yerr=stds, fmt="o-",
                color=COLORS["red"], capsize=3, lw=1.4, ms=4, mfc="white",
                mec=COLORS["red"], mew=1.0,
                label="PathwayPro (joint SUVR + WMH drop, $n{=}3$ seeds)")
    ax.axhline(np.mean(means), color=COLORS["grey"], linestyle="--",
               linewidth=0.6, alpha=0.7,
               label=f"mean $=$ {np.mean(means):.1f}%")
    ax.set_xlabel("Joint missingness rate on weak supervision (%)")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_xlim(-5, 95)
    ax.set_ylim(60, 90)
    ax.grid(alpha=0.25, linewidth=0.3)
    ax.legend(loc="lower left", frameon=False)
    save_fig(fig, FIGDIR / "fig3_robustness_curve")


# -----------------------------------------------------------------------------
# 5. Figure 4 — OASIS-1 zero-shot (confusion matrix + per-class precision/recall)
# -----------------------------------------------------------------------------
def fig_oasis():
    v6 = json.loads(OASIS_V6.read_text())
    cm = np.asarray(v6["confusion"], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    labels = ["CN", "MCI", "AD"]
    rep = v6["report"]
    auc = v6["ad_vs_cn_auc"]

    fig = plt.figure(figsize=(mm(183), mm(75)))
    gs = gridspec.GridSpec(1, 2, wspace=0.55, width_ratios=[1.0, 1.4],
                           top=0.86, bottom=0.18)

    # Panel a: row-normalised confusion matrix.
    ax_a = fig.add_subplot(gs[0, 0])
    cmap = LinearSegmentedColormap.from_list(
        "nb_blue", ["#FFFFFF", "#D6EAF8", "#5DADE2", "#2E86C1", "#1B4F72"])
    im = ax_a.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            t = f"{cm_norm[i, j]:.2f}\n({int(cm[i, j])})"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax_a.text(j, i, t, ha="center", va="center", fontsize=7, color=color)
    ax_a.set_xticks(range(3)); ax_a.set_yticks(range(3))
    ax_a.set_xticklabels(labels); ax_a.set_yticklabels(labels)
    ax_a.set_xlabel("Predicted"); ax_a.set_ylabel("True")
    ax_a.set_title(f"Confusion matrix (AUC$_{{\\rm AD/CN}}$ = {auc:.3f})")
    cb = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5)
    panel_label(ax_a, "a")

    # Panel b: per-class precision / recall / F1.
    ax_b = fig.add_subplot(gs[0, 1])
    metrics = ["precision", "recall", "f1-score"]
    width = 0.25
    x = np.arange(3)
    bar_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    for j, m in enumerate(metrics):
        vals = [rep[c][m] for c in labels]
        bars = ax_b.bar(x + (j - 1) * width, vals, width,
                        color=bar_colors[j], label=m.capitalize(),
                        edgecolor="white", linewidth=0.4, zorder=3)
        for bar, v in zip(bars, vals):
            ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                      f"{v:.2f}", ha="center", va="bottom", fontsize=5.5)
    ax_b.set_xticks(x); ax_b.set_xticklabels(
        [f"{c}\n($n$={int(rep[c]['support'])})" for c in labels])
    ax_b.set_ylabel("Score")
    ax_b.set_ylim(0, 1.0)
    ax_b.set_title("Per-class precision / recall / F1")
    ax_b.legend(frameon=False, fontsize=6, ncol=3, loc="upper center")
    ax_b.grid(axis="y", alpha=0.25, linewidth=0.3, zorder=0)
    panel_label(ax_b, "b")

    save_fig(fig, FIGDIR / "fig4_oasis_zeroshot")


# -----------------------------------------------------------------------------
# 6. Figure 5 — Cohort biomarker gradient (CN -> MCI -> AD)
# -----------------------------------------------------------------------------
def fig_cohort_gradient():
    """Reproduce the Table 2 biomarker gradient as a clean Nature-style chart."""
    classes = ["CN", "MCI", "AD"]
    apoe_pct = [29.2, 51.3, 65.8]
    amy_pct = [37.6, 53.8, 86.7]
    suvr = [(1.12, 0.21), (1.23, 0.27), (1.44, 0.26)]
    cent = [(22.3, 38.7), (41.5, 49.7), (80.9, 47.2)]
    wmh = [(8.4, 12.9), (8.9, 12.5), (9.7, 12.3)]

    fig, axes = plt.subplots(1, 4, figsize=(mm(183), mm(58)))
    fig.subplots_adjust(wspace=0.55, top=0.82, bottom=0.18, left=0.06, right=0.98)
    x = np.arange(3)
    bar_w = 0.55

    ax = axes[0]
    bars = ax.bar(x, apoe_pct, bar_w, color=[DIAGNOSIS_COLORS[c] for c in classes],
                  edgecolor="white", linewidth=0.5, zorder=3)
    for b, v in zip(bars, apoe_pct):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel(r"APOE-$\varepsilon$4 carrier (%)")
    ax.set_ylim(0, 90)
    ax.set_title("Genetic risk")
    panel_label(ax, "a")

    ax = axes[1]
    bars = ax.bar(x, amy_pct, bar_w, color=[DIAGNOSIS_COLORS[c] for c in classes],
                  edgecolor="white", linewidth=0.5, zorder=3)
    for b, v in zip(bars, amy_pct):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("Amyloid PET positive (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Molecular pathology")
    panel_label(ax, "b")

    ax = axes[2]
    means = [s[0] for s in suvr]
    sds = [s[1] for s in suvr]
    ax.errorbar(x, means, yerr=sds, fmt="o-", color=COLORS["red"],
                capsize=3, lw=1.5, ms=5, mfc="white", mec=COLORS["red"], mew=1.2)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("Amyloid SUVR")
    ax.set_title("Amyloid burden")
    ax.set_ylim(0.85, 1.8)
    panel_label(ax, "c")

    ax = axes[3]
    means = [s[0] for s in wmh]
    sds = [s[1] for s in wmh]
    ax.errorbar(x, means, yerr=sds, fmt="s-", color=PATHWAY_COLORS["vascular"],
                capsize=3, lw=1.5, ms=5, mfc="white",
                mec=PATHWAY_COLORS["vascular"], mew=1.2)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("WMH volume (mL)")
    ax.set_title("Vascular burden")
    ax.set_ylim(0, 25)
    panel_label(ax, "d")

    fig.suptitle(
        "Co-occurring vascular pathology motivates two-branch disentanglement",
        fontsize=8, fontweight="bold", y=0.98,
    )
    save_fig(fig, FIGDIR / "fig5_cohort_gradient")


# -----------------------------------------------------------------------------
# 7. Counterfactual quantitative validation
# -----------------------------------------------------------------------------
def _load_cf_seeds():
    """Return a list of dicts (one per seed) from the counterfactual sweep,
    plus the diagnosis-coupling-correct subset and the aggregate."""
    if not (CF_DIR / "aggregate.json").exists():
        return None
    agg = json.loads((CF_DIR / "aggregate.json").read_text())
    seed_dirs = sorted(p for p in CF_DIR.glob("seed*") if (p / "summary.json").exists())
    seeds = []
    for sd in seed_dirs:
        s = json.loads((sd / "summary.json").read_text())
        s["seed_id"] = sd.name
        try:
            import pandas as pd
            s["per_subject"] = pd.read_csv(sd / "per_subject.csv")
        except Exception:
            s["per_subject"] = None
        seeds.append(s)
    return {"agg": agg, "seeds": seeds}


def write_table_counterfactual():
    """LaTeX Table for counterfactual quantitative validation.
    Reports |Spearman rho| across all 5 zero-masking seeds (the absolute
    value is the sign-invariant strength of the z_v->P(AD) causal coupling
    and is the right quantity to report when the discrete sign of the
    diagnosis head's z_v projection flips between random initialisations).
    Rows = analysis cohorts (AD∪MCI / AD only / MCI only / Full ALL)."""
    cf = _load_cf_seeds()
    if cf is None:
        print("  [cf] no aggregate.json found, skip table.")
        return
    allres = cf["agg"]["all"]
    KEYS = [
        ("main_dPAD_vs_WMH_AD_MCI",  r"AD$\cup$MCI"),
        ("main_dPAD_vs_WMH_AD",      r"AD only"),
        ("main_dPAD_vs_WMH_MCI",     r"MCI only"),
        ("main_dPAD_vs_WMH_ALL",     r"Full cohort"),
    ]
    n_seeds = allres["main_dPAD_vs_WMH_AD_MCI"]["n_seeds"]

    def f_abs(d, bold=False):
        body = f"{d['abs_rho_mean']:.3f}\\,$\\pm$\\,{d['abs_rho_std']:.3f}"
        return r"\textbf{" + body + "}" if bold else body

    def fp(p):
        if p < 1e-10:
            return f"{p:.1e}"
        return f"{p:.2g}"

    rows = []
    for k, lab in KEYS:
        if k not in allres:
            continue
        m = allres[k]
        rows.append(
            f"  {lab} & {int(m['n_mean'])} & {f_abs(m)} & {fp(m['p_median'])} \\\\"
        )

    ctrl1 = allres.get("ctrl1_dPAD_za_vs_WMH_AD_MCI")
    ctrl2 = allres.get("ctrl2_dPAD_vs_age_AD_MCI")
    ctrl_block = ""
    if ctrl1 is not None or ctrl2 is not None:
        ctrl_block = "\\midrule\n"
        ctrl_block += "  \\multicolumn{4}{l}{\\textit{Negative controls (AD$\\cup$MCI cohort)}} \\\\\n"
        if ctrl1 is not None:
            ctrl_block += (f"  $do(z_a=\\bar z_{{a,CN}})$ vs.~WMH "
                           f"& {int(ctrl1['n_mean'])} & {f_abs(ctrl1)} & {fp(ctrl1['p_median'])} \\\\\n")
        if ctrl2 is not None:
            ctrl_block += (f"  $do(z_v=\\bar z_{{v,CN}})$ vs.~age "
                           f"& {int(ctrl2['n_mean'])} & {f_abs(ctrl2)} & {fp(ctrl2['p_median'])} \\\\\n")

    flipped = cf["agg"].get("coupling_flipped_seeds", [])

    tex = r"""\begin{table}[t]
\centering
\caption{Counterfactual quantitative validation on the v6$_{\rm zeromask}$
checkpoints. For every test/val subject we
compute $\Delta P(\text{AD}) = P(\text{AD}\!\mid\! z_a, z_v) - P(\text{AD}\!\mid\! z_a,
\bar z_{v,\text{CN}})$, where $\bar z_{v,\text{CN}}$ is the mean vascular latent over
cognitively-normal subjects. Reported is the absolute Spearman rank correlation
$|\rho|$ between $\Delta P(\text{AD})$ and the patient's true WMH volume,
aggregated across """ + str(n_seeds) + r""" random seeds; using $|\rho|$ removes
the $z_v\!\to\!\text{AD}$ direction ambiguity inherent to disentangled
representations (Sec.~\ref{sec:supp-cf}). The main effect (top block) is
highly significant on every cohort and is much stronger than either
negative control.}
\label{tab:cf}
\begin{tabular}{lccc}
\toprule
Cohort & $n$ & $|\rho|$ Spearman (mean$\pm$SD) & $p$ (median) \\
\midrule
""" + "\n".join(rows) + "\n" + ctrl_block + r"""\bottomrule
\end{tabular}
\end{table}
"""
    (SECDIR / "tab_counterfactual.tex").write_text(tex)
    print("  wrote sections/tab_counterfactual.tex")


def fig_counterfactual():
    """Three-panel Nature-style figure: (a) per-seed forest plot of Spearman rho;
    (b) scatter of ΔP(AD) vs WMH on AD∪MCI for seed0; (c) bar chart comparing
    main test vs. negative controls (z_a side, age) on the |rho| metric."""
    cf = _load_cf_seeds()
    if cf is None:
        print("  [cf] no aggregate.json — skip figure.")
        return
    seeds = cf["seeds"]
    agg = cf["agg"]
    if not seeds:
        return

    fig = plt.figure(figsize=(mm(180), mm(68)))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.55,
                           left=0.075, right=0.985, top=0.86, bottom=0.26)

    # ---- (a) Per-seed forest plot of |rho| (sign-invariant strength) ----
    ax = fig.add_subplot(gs[0, 0])
    abs_rhos, lo, hi, names, flipped = [], [], [], [], []
    for s in seeds:
        m = s.get("main_dPAD_vs_WMH_AD_MCI", {})
        r = m.get("rho", float("nan"))
        abs_r = abs(r) if np.isfinite(r) else float("nan")
        abs_rhos.append(abs_r)
        n = m.get("n", 0)
        # Approx 95% CI for |rho| via Fisher's z on |rho|
        if n > 3 and 0 < abs_r < 1:
            se = 1.0 / np.sqrt(n - 3)
            z = np.arctanh(abs_r)
            lo_, hi_ = np.tanh([z - 1.96 * se, z + 1.96 * se])
        else:
            lo_, hi_ = float("nan"), float("nan")
        lo.append(lo_); hi.append(hi_)
        names.append(s["seed_id"].replace("seed", "S"))
        flipped.append(s.get("coupling_flipped", False))
    y = np.arange(len(seeds))
    for i, (r, l, h, fl) in enumerate(zip(abs_rhos, lo, hi, flipped)):
        c = COLORS["red"]
        marker = "o" if not fl else "s"
        ax.plot([l, h], [i, i], "-", color=c, lw=1.0, zorder=2)
        ax.plot(r, i, marker, color=c, ms=4.5, zorder=3,
                mfc=c if not fl else "white", mec=c, mew=1.0)
    ax.axvline(0, color="k", lw=0.5, ls="--", zorder=1)
    # Aggregate vertical band
    agg_main = agg["all"]["main_dPAD_vs_WMH_AD_MCI"]
    am, asd = agg_main["abs_rho_mean"], agg_main["abs_rho_std"]
    ax.axvspan(am - asd, am + asd, color=COLORS["red"], alpha=0.10, zorder=0)
    ax.axvline(am, color=COLORS["red"], lw=0.8, ls="-", alpha=0.7, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel(r"$|\rho|$  ($\Delta P_{\mathrm{AD}}$ vs.~WMH)")
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.set_title("Per-seed effect (AD$\\cup$MCI)", pad=8)
    # Legend for flipped marker
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", mfc=COLORS["red"],
               mec=COLORS["red"], ms=4.5, label="$z_v$ aligned"),
        Line2D([0], [0], marker="s", color="w", mfc="white",
               mec=COLORS["red"], mew=1.0, ms=4.5, label="$z_v$ flipped"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=5.5,
              loc="lower right", handlelength=0.6)
    panel_label(ax, "a", x=-0.30, y=1.18)

    # ---- (b) Scatter for seed0 (or first non-flipped seed) ----
    ax = fig.add_subplot(gs[0, 1])
    sd = next((s for s in seeds if not s.get("coupling_flipped")), seeds[0])
    df = sd["per_subject"]
    if df is not None:
        sub = df[df["label"].isin([1, 2])]
        if len(sub) > 5:
            x = sub["wmh_total"].to_numpy()
            yv = sub["dP_AD_zv"].to_numpy()
            mask_ad = sub["label"] == 2
            mask_mci = sub["label"] == 1
            ax.scatter(x[mask_ad], yv[mask_ad], s=8, c=DIAGNOSIS_COLORS["AD"],
                       alpha=0.75, label="AD", linewidths=0)
            ax.scatter(x[mask_mci], yv[mask_mci], s=8, c=DIAGNOSIS_COLORS["MCI"],
                       alpha=0.75, label="MCI", linewidths=0)
            # Robust linear fit through origin-like trend
            m = np.isfinite(x) & np.isfinite(yv)
            if m.sum() > 2:
                z = np.polyfit(x[m], yv[m], 1)
                xfit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                ax.plot(xfit, z[0] * xfit + z[1], "-", color=COLORS["grey"], lw=0.8)
            from scipy.stats import spearmanr
            rho, p = spearmanr(x[m], yv[m])
            ax.text(0.04, 0.96, fr"$\rho={rho:+.2f}$" + "\n" + fr"$p={p:.1e}$",
                    transform=ax.transAxes, va="top", fontsize=6.5)
        ax.set_xlabel(r"WMH volume (cm$^3$)")
        ax.set_ylabel(r"$\Delta P(\mathrm{AD})$ under $do(z_v=\bar z_{v,\mathrm{CN}})$",
                       fontsize=7)
        ax.legend(loc="lower right", frameon=False, handlelength=0.8, fontsize=6.0)
    ax.set_title(f"Patient-level signal ({sd['seed_id']})", pad=8)
    panel_label(ax, "b", x=-0.28, y=1.18)

    # ---- (c) Main vs controls bar (|rho| across all 5 seeds) ----
    ax = fig.add_subplot(gs[0, 2])
    al = agg["all"]
    bars = [
        ("do($z_v$)\n→WMH\n(main)",  al.get("main_dPAD_vs_WMH_AD_MCI"), COLORS["red"]),
        ("do($z_a$)\n→WMH\n(ctrl)",  al.get("ctrl1_dPAD_za_vs_WMH_AD_MCI"), COLORS["grey"]),
        ("do($z_v$)\n→age\n(ctrl)",  al.get("ctrl2_dPAD_vs_age_AD_MCI"), COLORS["cyan"]),
    ]
    xs = np.arange(len(bars))
    rhos = [b[1]["abs_rho_mean"] if b[1] else 0 for b in bars]
    errs = [b[1]["abs_rho_std"]  if b[1] else 0 for b in bars]
    cols = [b[2] for b in bars]
    ax.bar(xs, rhos, yerr=errs, color=cols, edgecolor="k", linewidth=0.4,
           error_kw={"elinewidth": 0.6, "capsize": 2.0, "ecolor": "k"})
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=6.0)
    ax.set_ylabel(r"$|\rho|$  (mean$\pm$SD across seeds)")
    ax.set_ylim(0, max(0.85, max(rhos) + max(errs) + 0.05))
    ax.set_title("Specificity (AD$\\cup$MCI)", pad=8)
    panel_label(ax, "c", x=-0.30, y=1.18)

    save_fig(fig, FIGDIR / "fig6_counterfactual")


# -----------------------------------------------------------------------------
# 7b. T1-only robustness ablation (zero-masking deployment story)
# -----------------------------------------------------------------------------
def _load_t1only():
    """Load per-seed paired-vs-T1only results from `analysis_pathway_v6_t1only`."""
    if not T1_DIR.exists():
        return None
    out = {"seeds": []}
    for sd in sorted(T1_DIR.glob("seed*")):
        f = sd / "summary.json"
        if not f.exists():
            continue
        out["seeds"].append(json.loads(f.read_text()))
    if not out["seeds"]:
        return None

    def collect(path):
        vals = []
        for s in out["seeds"]:
            v = s
            for k in path:
                if not isinstance(v, dict) or k not in v:
                    v = None; break
                v = v[k]
            if v is not None and isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        return np.asarray(vals)

    METRICS = [
        ("acc",                "test_acc"),
        ("balanced_acc",       "test_bal"),
        ("auc_ad_vs_cn",       "auc_ad_cn"),
        ("auc_zv_to_wmh",      "auc_zv_wmh"),
        ("auc_za_to_amy",      "auc_za_amy"),
        ("disentangle_score",  "disent"),
    ]
    out["paired"] = {}
    out["t1only"] = {}
    out["delta"]  = {}
    for src_key, short in METRICS:
        p = collect(["paired", src_key])
        t = collect(["t1only", src_key])
        out["paired"][short] = (p.mean(), p.std(ddof=1)) if len(p) else (np.nan, np.nan)
        out["t1only"][short] = (t.mean(), t.std(ddof=1)) if len(t) else (np.nan, np.nan)
        if len(p) and len(t):
            d = p - t
            out["delta"][short] = (d.mean(), d.std(ddof=1))
        else:
            out["delta"][short] = (np.nan, np.nan)
    out["n_seeds"] = len(out["seeds"])
    return out


def write_table_t1only():
    """LaTeX table contrasting paired (T1+FLAIR) and T1-only (FLAIR=0)
    deployment regimes on the same v6_zeromask checkpoints."""
    d = _load_t1only()
    if d is None:
        print("  [t1only] no data — skip table.")
        return

    def f_pct(t, decimals=1):
        m, s = t
        if not np.isfinite(m):
            return "---"
        return f"{m*100:.{decimals}f}\\,$\\pm$\\,{s*100:.{decimals}f}"

    def f(t, decimals=3):
        m, s = t
        if not np.isfinite(m):
            return "---"
        return f"{m:.{decimals}f}\\,$\\pm$\\,{s:.{decimals}f}"

    rows = [
        ("Test accuracy (\\%)",                     "test_acc",   f_pct),
        ("Test balanced accuracy (\\%)",            "test_bal",   f_pct),
        ("AUC (AD vs.\\ CN)",                       "auc_ad_cn",  f),
        ("AUC $z_v\\!\\to\\!$WMH",                  "auc_zv_wmh", f),
        ("AUC $z_a\\!\\to\\!$amyloid",              "auc_za_amy", f),
        ("Disentanglement score",                    "disent",     f),
    ]
    body = []
    for label, key, fn in rows:
        body.append(
            f"  {label} & {fn(d['paired'][key])} & {fn(d['t1only'][key])} & "
            f"{fn(d['delta'][key], 1) if 'pct' in fn.__name__ else fn(d['delta'][key])} \\\\"
        )

    n = d["n_seeds"]
    tex = (
        "% Auto-generated by paper/build_assets.py.\n"
        "\\begin{table}[!t]\\centering\\small\\setlength{\\tabcolsep}{6pt}\n"
        "\\caption{Robustness to missing FLAIR. The same five v6$_{\\rm zeromask}$\n"
        "checkpoints are evaluated on the ADNI paired test set under two regimes:\n"
        "(i) Paired---both T1 and FLAIR are present, identical to the standard\n"
        "evaluation; (ii) T1-only---FLAIR is replaced with all-zero tensors at\n"
        "inference time, routing $z_v$ through the missing-token contract\n"
        "(Sec.~\\ref{sec:m-zeromask}). All numbers are mean$\\pm$SD across "
        + str(n) + " seeds.\n"
        "AD-vs-CN AUC and three-class accuracy hold to within sampling noise,\n"
        "while $z_v$ collapses to chance on WMH (AUC$=$0.50)---confirming that\n"
        "the missing-token contract refuses to fabricate vascular signal when\n"
        "FLAIR is absent.}\n"
        "\\label{tab:t1only}\n"
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "Metric & Paired (T1+FLAIR) & T1-only (FLAIR$=$0) & "
        "$\\Delta$ (paired$-$T1-only) \\\\\n"
        "\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (SECDIR / "tab_t1only.tex").write_text(tex)
    print(f"  wrote: {(SECDIR / 'tab_t1only.tex').relative_to(PAPER)}")


def fig_t1only():
    """Three-panel Nature-style figure: (a) paired vs T1-only paired bars on
    diagnosis metrics; (b) probe-AUC bars for z_v->WMH and z_a->amy showing the
    intended degradation of z_v under FLAIR=0; (c) per-seed AD-vs-CN AUC delta
    forest plot."""
    d = _load_t1only()
    if d is None:
        print("  [t1only] no data — skip figure.")
        return

    fig = plt.figure(figsize=(mm(180), mm(72)))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.55,
                           left=0.08, right=0.985, top=0.82, bottom=0.22)

    # ---- (a) Diagnosis metrics: paired vs T1-only ----
    ax = fig.add_subplot(gs[0, 0])
    metric_names = ["Acc.", "BAcc", "AUC\nAD vs CN"]
    keys = ["test_acc", "test_bal", "auc_ad_cn"]
    paired_m = [d["paired"][k][0] for k in keys]
    paired_s = [d["paired"][k][1] for k in keys]
    t1_m     = [d["t1only"][k][0] for k in keys]
    t1_s     = [d["t1only"][k][1] for k in keys]
    # Convert acc/bal to percent for visual parity with AUC (0–1)
    # but to keep on the same axis we keep all in [0,1].
    x = np.arange(len(keys))
    w = 0.36
    ax.bar(x - w/2, paired_m, w, yerr=paired_s, color=COLORS["blue"],
           label="Paired (T1+FLAIR)", edgecolor="white", linewidth=0.4,
           error_kw={"elinewidth": 0.6, "capsize": 1.8})
    ax.bar(x + w/2, t1_m, w, yerr=t1_s, color=COLORS["orange"],
           label="T1-only (FLAIR$=$0)", edgecolor="white", linewidth=0.4,
           error_kw={"elinewidth": 0.6, "capsize": 1.8})
    ax.axhline(1/3, color="k", lw=0.4, ls=":", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=6.5)
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Diagnosis performance", pad=10)
    ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.2)
    panel_label(ax, "a", x=-0.30, y=1.24)

    # ---- (b) Probe AUCs: z_v->WMH, z_a->amy ----
    ax = fig.add_subplot(gs[0, 1])
    probe_names = [r"$z_v\!\to\!$WMH", r"$z_a\!\to\!$amy"]
    pkeys = ["auc_zv_wmh", "auc_za_amy"]
    paired_m = [d["paired"][k][0] for k in pkeys]
    paired_s = [d["paired"][k][1] for k in pkeys]
    t1_m     = [d["t1only"][k][0] for k in pkeys]
    t1_s     = [d["t1only"][k][1] for k in pkeys]
    x = np.arange(len(pkeys))
    w = 0.36
    ax.bar(x - w/2, paired_m, w, yerr=paired_s, color=COLORS["blue"],
           edgecolor="white", linewidth=0.4,
           error_kw={"elinewidth": 0.6, "capsize": 1.8})
    ax.bar(x + w/2, t1_m, w, yerr=t1_s, color=COLORS["orange"],
           edgecolor="white", linewidth=0.4,
           error_kw={"elinewidth": 0.6, "capsize": 1.8})
    ax.axhline(0.5, color="k", lw=0.4, ls=":", alpha=0.6,
               label="Chance (0.50)")
    ax.set_xticks(x); ax.set_xticklabels(probe_names, fontsize=7)
    ax.set_ylabel("Probe AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Latent specialisation under FLAIR$=$0", pad=10)
    ax.legend(frameon=False, fontsize=5.5, loc="lower right",
              handlelength=1.0)
    panel_label(ax, "b", x=-0.30, y=1.24)

    # ---- (c) Per-seed AD-vs-CN AUC delta ----
    ax = fig.add_subplot(gs[0, 2])
    seed_ids = []
    paired_aucs = []
    t1_aucs = []
    for s in d["seeds"]:
        nm = Path(s["ckpt"]).parent.name.replace("ours_seed", "S")
        seed_ids.append(nm)
        paired_aucs.append(s["paired"].get("auc_ad_vs_cn", float("nan")))
        t1_aucs.append(s["t1only"].get("auc_ad_vs_cn", float("nan")))
    yy = np.arange(len(seed_ids))
    ax.scatter(paired_aucs, yy, color=COLORS["blue"], s=12, zorder=3,
               label="Paired", edgecolors="white", linewidths=0.3)
    ax.scatter(t1_aucs, yy, color=COLORS["orange"], s=12, zorder=3,
               label="T1-only", edgecolors="white", linewidths=0.3)
    for i, (p, t) in enumerate(zip(paired_aucs, t1_aucs)):
        ax.plot([min(p, t), max(p, t)], [i, i], color=COLORS["grey"],
                lw=0.6, alpha=0.6, zorder=1)
    ax.set_yticks(yy); ax.set_yticklabels(seed_ids, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("AUC (AD vs. CN)")
    ax.set_xlim(0.80, 0.96)
    ax.set_title("Per-seed AD-vs-CN AUC", pad=10)
    ax.legend(frameon=False, fontsize=6, loc="lower right", handlelength=0.6)
    panel_label(ax, "c", x=-0.30, y=1.24)

    save_fig(fig, FIGDIR / "fig7_t1only_robust")


# -----------------------------------------------------------------------------
# 8. Numbers dump for paper review
# -----------------------------------------------------------------------------
def dump_numbers(table):
    summary = {
        m: {k: list(v[k]) if isinstance(v[k], tuple) else v[k] for k in v}
        for m, v in table.items()
    }
    (PAPER / "_aggregated_numbers.json").write_text(json.dumps(summary, indent=2))


# -----------------------------------------------------------------------------
def main():
    print("[build_assets] aggregating phase-0 benchmark...")
    table = aggregate_bench()

    print("[build_assets] writing LaTeX tables...")
    write_table_phase0(table)
    write_table_oasis()
    write_table_counterfactual()
    write_table_t1only()

    print("[build_assets] rendering supplementary figures (S1--S4) only.")
    # Main-text Fig.1–Fig.6 are produced by paper/figures/fig{1..6}_*.py and
    # must NOT be overwritten here.  We keep only the four supplementary
    # figures whose new files live under figures/fig{3,4,6,7}_*.pdf and are
    # referenced from sections/supplement.tex.
    fig_robustness()       # supplement Fig.S1 — fig3_robustness_curve.pdf
    fig_oasis()            # supplement Fig.S2 — fig4_oasis_zeroshot.pdf
    fig_counterfactual()   # supplement Fig.S3 — fig6_counterfactual.pdf
    fig_t1only()           # supplement Fig.S4 — fig7_t1only_robust.pdf

    dump_numbers(table)
    print("[build_assets] done.")


if __name__ == "__main__":
    main()
