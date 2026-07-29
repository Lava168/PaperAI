import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

SCRIPT_DIR = Path(__file__).resolve().parent

# -----------------------------
# Times New Roman (register TTF if matplotlib cache lacks the name)
# -----------------------------
def setup_times_new_roman() -> str:
    tnr_candidates = [
        SCRIPT_DIR.parent / "fonts" / "Times New Roman.ttf",
        SCRIPT_DIR.parent / "fonts" / "times.ttf",
        Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/times.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"),
        Path.home() / ".fonts/Times New Roman.ttf",
        Path.home() / ".local/share/fonts/Times New Roman.ttf",
        Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"),
    ]
    for fp in tnr_candidates:
        if fp.is_file():
            try:
                font_manager.fontManager.addfont(str(fp))
                prop = font_manager.FontProperties(fname=str(fp))
                name = prop.get_name()
                plt.rcParams["font.family"] = name
                plt.rcParams["font.serif"] = [name, "Times New Roman", "Times"]
                return name
            except Exception:
                continue

    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Times New Roman" in available:
        plt.rcParams["font.family"] = "Times New Roman"
        return "Times New Roman"
    for name in ("Times", "Nimbus Roman No9 L", "Liberation Serif", "DejaVu Serif"):
        if name in available:
            plt.rcParams["font.family"] = name
            print(f"警告: 未找到 Times New Roman.ttf，暂用 {name}")
            return name
    plt.rcParams["font.family"] = "DejaVu Serif"
    return "DejaVu Serif"


# -----------------------------
# Load data
# -----------------------------
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CSV_CANDIDATES = [
    PROJECT_ROOT / "outputs_disentangle_v6" / "benchmark_summary.csv",
    PROJECT_ROOT / "paper" / "data" / "phase0_benchmark.csv",
]
path = next((p for p in CSV_CANDIDATES if p.is_file()), None)
if path is None:
    raise FileNotFoundError(
        "找不到 benchmark CSV。请确认存在其一：\n"
        + "\n".join(f"  - {p}" for p in CSV_CANDIDATES)
    )
OUT_DIR = SCRIPT_DIR
df = pd.read_csv(path)
print(f"Loaded: {path}")

method_order = ["vanilla", "betavae", "tcvae", "factorvae", "dipvae2", "ours"]
method_labels = {
    "vanilla": "Vanilla MTL",
    "betavae": r"$\beta$-VAE",
    "tcvae": r"$\beta$-TCVAE",
    "factorvae": "FactorVAE",
    "dipvae2": "DIP-VAE-II",
    "ours": "PathwayPro (ours)",
}

COL_GREY = "#BDBDBD"
COL_GREY_DARK = "#666666"
COL_OURS = "#E64B35"
COL_FACTOR = "#4DBBD5"
COL_TCVAE = "#00A087"

method_colors = {
    "vanilla": COL_GREY,
    "betavae": COL_GREY,
    "tcvae": COL_TCVAE,
    "factorvae": COL_FACTOR,
    "dipvae2": COL_GREY,
    "ours": COL_OURS,
}
highlight_edge = {
    "vanilla": COL_GREY_DARK,
    "betavae": COL_GREY_DARK,
    "tcvae": COL_TCVAE,
    "factorvae": COL_FACTOR,
    "dipvae2": COL_GREY_DARK,
    "ours": COL_OURS,
}

metrics = ["test_acc", "test_bal", "test_auc_za_amy", "test_auc_zv_wmh", "disentangle_score"]
summary = df.groupby("method")[metrics].agg(["mean", "std"]).reindex(method_order)

base_methods = ["vanilla", "betavae", "tcvae", "factorvae", "dipvae2"]
deltas = []
for base in base_methods:
    ours = df[df["method"] == "ours"].sort_values("seed")
    b = df[df["method"] == base].sort_values("seed")
    for metric in ["test_bal", "test_auc_zv_wmh"]:
        vals = ours[metric].to_numpy() - b[metric].to_numpy()
        for seed, val in zip(ours["seed"], vals):
            deltas.append({"baseline": base, "metric": metric, "seed": seed, "delta": val})
delta_df = pd.DataFrame(deltas)

FONT_NAME = setup_times_new_roman()
print(f"Font: {FONT_NAME}")

plt.rcParams.update({
    "font.family": FONT_NAME,
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.85,
})

# Manuscript order: Fig.1 framework, Fig.2 cohort, Fig.3 = this Phase-0 benchmark
MANUSCRIPT_FIG_NUM = 3

# 2 rows only (no panel e); tight margins, larger plot area
fig = plt.figure(figsize=(15.5, 8.2), facecolor="white")
gs = fig.add_gridspec(
    2,
    4,
    left=0.055,
    right=0.995,
    top=0.90,
    bottom=0.14,
    wspace=0.42,
    hspace=0.48,
    height_ratios=[1.05, 1.0],
)

fig.text(
    0.01,
    0.995,
    f"Figure {MANUSCRIPT_FIG_NUM}",
    ha="left",
    va="top",
    fontsize=15,
    fontweight="bold",
)
fig.suptitle(
    "Phase-0 head-to-head benchmark of six disentanglement objectives",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
fig.text(
    0.5,
    0.935,
    "Shared backbone; only the disentanglement objective differs. Bars: mean ± SD (3 seeds).",
    ha="center",
    fontsize=9,
    style="italic",
    color="#4D4D4D",
)


def panel_label(ax, label):
    ax.text(-0.10, 1.02, f"({label})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="left", va="bottom")


def style_xticks(ax):
    ax.tick_params(axis="x", rotation=48, pad=5, length=3)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")


def add_bars_with_points(ax, metric, ylabel, title, ylim=None, best_method=None):
    x = np.arange(len(method_order))
    means = summary[(metric, "mean")].values
    sds = summary[(metric, "std")].values

    ax.bar(
        x, means, yerr=sds, capsize=3, width=0.64,
        color=[method_colors[m] for m in method_order],
        edgecolor=[highlight_edge[m] for m in method_order],
        linewidth=0.9, alpha=0.9,
        error_kw=dict(elinewidth=0.9, capthick=0.9, ecolor="#222222"),
    )

    rng = np.random.default_rng(2026)
    for i, m in enumerate(method_order):
        vals = df.loc[df["method"] == m, metric].to_numpy()
        jitter = rng.normal(0, 0.028, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals, s=28,
            facecolor="white", edgecolor=highlight_edge[m], linewidth=0.9, zorder=4,
        )

    if ylim:
        lo, hi = ylim
        ax.set_ylim(lo, hi + (hi - lo) * 0.10)
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels[m] for m in method_order])
    style_xticks(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.28)
    ax.set_axisbelow(True)
    ax.margins(x=0.06)


# Row 0: (a) + (b)
ax_a1 = fig.add_subplot(gs[0, 0])
add_bars_with_points(ax_a1, "test_acc", "Accuracy", "Test Acc", ylim=(0.35, 0.78))
panel_label(ax_a1, "a")

ax_a2 = fig.add_subplot(gs[0, 1])
add_bars_with_points(ax_a2, "test_bal", "Balanced accuracy", "Test BAcc", ylim=(0.35, 0.78))
panel_label(ax_a2, "a")

ax_b1 = fig.add_subplot(gs[0, 2])
add_bars_with_points(
    ax_b1, "test_auc_za_amy", "AUC", r"AUC($z_a \rightarrow$ amyloid)", ylim=(0.55, 0.90),
)
panel_label(ax_b1, "b")

ax_b2 = fig.add_subplot(gs[0, 3])
add_bars_with_points(
    ax_b2, "test_auc_zv_wmh", "AUC", r"AUC($z_v \rightarrow$ WMH)", ylim=(0.62, 0.98),
)
panel_label(ax_b2, "b")

# Row 1: (c) + (d)
ax_c = fig.add_subplot(gs[1, 0:2])
panel_label(ax_c, "c")

y_pos = np.arange(len(method_order))[::-1]
label_x = []
for i, m in enumerate(method_order):
    yi = len(method_order) - 1 - i
    mean = summary.loc[m, ("disentangle_score", "mean")]
    sd = summary.loc[m, ("disentangle_score", "std")]
    vals = df.loc[df["method"] == m, "disentangle_score"].to_numpy()
    ax_c.errorbar(
        mean, yi, xerr=sd, fmt="o", markersize=6,
        color=highlight_edge[m], markerfacecolor=method_colors[m],
        markeredgecolor=highlight_edge[m], ecolor="#555555",
        elinewidth=0.9, capsize=3, zorder=3,
    )
    ax_c.scatter(
        vals, np.full(len(vals), yi) - 0.20, s=20,
        facecolor="white", edgecolor=highlight_edge[m], linewidth=0.8, zorder=4,
    )
    x_lab = max(mean + sd, vals.max()) + 0.020
    label_x.append(x_lab)
    ax_c.text(x_lab, yi, f"{mean:+.3f} ± {sd:.3f}", ha="left", va="center", fontsize=8)

ax_c.axvline(0, color="#333333", linestyle=(0, (4, 3)), linewidth=0.9)
ax_c.set_yticks(y_pos)
ax_c.set_yticklabels([method_labels[m] for m in method_order])
ax_c.set_xlim(-0.11, max(0.18, max(label_x) + 0.04))
ax_c.set_xlabel("Symmetric disentanglement score", labelpad=4)
ax_c.set_title("Pathway balance / leakage diagnostic", fontweight="bold", pad=10)
ax_c.spines[["top", "right"]].set_visible(False)
ax_c.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.25)
ax_c.set_axisbelow(True)
ax_c.margins(y=0.10)


def delta_panel(ax, metric, title, xlabel, xlim, panel_label_char=None):
    if panel_label_char:
        panel_label(ax, panel_label_char)
    y = np.arange(len(base_methods))[::-1]
    x_lo, x_hi = xlim
    x_pad = (x_hi - x_lo) * 0.10
    for i, base in enumerate(base_methods):
        yi = len(base_methods) - 1 - i
        vals = delta_df[(delta_df["baseline"] == base) & (delta_df["metric"] == metric)]["delta"].to_numpy()
        mean = vals.mean()
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        color = COL_OURS if mean >= 0 else COL_FACTOR
        ax.barh(yi, mean, color=color, alpha=0.88, height=0.44, edgecolor=color)
        ax.errorbar(mean, yi, xerr=sd, fmt="none", ecolor="#222222", elinewidth=0.9, capsize=3, zorder=3)
        ax.scatter(vals, np.full(len(vals), yi), s=24, facecolor="white", edgecolor=color, linewidth=0.9, zorder=4)
        off = x_pad * 0.30
        if mean >= 0:
            x_txt = min(mean + sd + off, x_hi + x_pad * 0.5)
            ha = "left"
        else:
            x_txt = max(mean - sd - off, x_lo - x_pad * 0.5)
            ha = "right"
        ax.text(x_txt, yi, f"{mean:+.3f}", ha=ha, va="center", fontsize=8, clip_on=True)
    ax.axvline(0, color="#222222", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([method_labels[m] for m in base_methods])
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlim(x_lo - x_pad * 0.35, x_hi + x_pad * 0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(y=0.12)


ax_d1 = fig.add_subplot(gs[1, 2])
delta_panel(
    ax_d1, "test_bal",
    r"$\Delta$BAcc (PathwayPro $-$ baseline)",
    r"$\Delta$ balanced accuracy",
    (-0.20, 0.20),
    panel_label_char="d",
)

ax_d2 = fig.add_subplot(gs[1, 3])
delta_panel(
    ax_d2, "test_auc_zv_wmh",
    r"$\Delta$AUC($z_v \rightarrow$ WMH)",
    r"$\Delta$ AUC",
    (-0.12, 0.22),
)

legend_handles = [
    Line2D([0], [0], marker="s", color="none", markerfacecolor=COL_OURS,
           markeredgecolor=COL_OURS, markersize=8, label="PathwayPro (ours)"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=COL_FACTOR,
           markeredgecolor=COL_FACTOR, markersize=8, label="FactorVAE"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=COL_TCVAE,
           markeredgecolor=COL_TCVAE, markersize=8, label=r"$\beta$-TCVAE"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=COL_GREY,
           markeredgecolor=COL_GREY_DARK, markersize=8, label="Other baselines"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
           markeredgecolor="#444444", markersize=6, label="Individual seed"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 0.02),
    fontsize=9,
)

out_svg = OUT_DIR / "benchmark_summary_nature_quant_vector.svg"
out_pdf = OUT_DIR / "benchmark_summary_nature_quant_vector.pdf"
out_png = OUT_DIR / "benchmark_summary_nature_quant_preview.png"
out_manuscript = OUT_DIR / "fig2_phase0_benchmark.pdf"

for p in (out_svg, out_pdf, out_png, out_manuscript):
    kw = dict(bbox_inches="tight", facecolor="white", pad_inches=0.04)
    if p.suffix == ".png":
        kw["dpi"] = 300
    fig.savefig(p, **kw)
plt.close(fig)

print("Saved:")
print(out_svg)
print(out_pdf)
print(out_png)

print("\nMean ± SD summary:")
for m in method_order:
    row = [
        f"{met}={summary.loc[m, (met, 'mean')]:.4f}±{summary.loc[m, (met, 'std')]:.4f}"
        for met in metrics
    ]
    print(method_labels[m], " | ".join(row))
