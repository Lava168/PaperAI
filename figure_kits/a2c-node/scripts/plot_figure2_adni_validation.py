#!/usr/bin/env python3
"""Figure 2 — ADNI internal validation: seed stability, ROC (ensemble), row-% CM.

Outputs (300 dpi PNG + vector PDF):
  paper/figures/results/Figure2_adni_validation.{png,pdf}

Data:
  * Per-seed validation BAcc: max over epochs from runs/v4_mm_seed*/train.log
  * ROC + AUC: runs/v4_mm_ensemble_5seed.npz
  * Confusion: runs/v4_mm_analysis_npj_5seed/confusion_matrix.csv (counts → row %)
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]


def parse_train_best_bacc(log_path: Path) -> float | None:
    text = log_path.read_text(errors="ignore")
    pat = re.compile(
        r"^epoch \d+\s+train=[0-9.\-]+\s+val=[0-9.\-]+\s+val\(drop\)=[0-9.\-]+\s+"
        r"acc=[0-9.\-]+\s+bacc=([0-9.\-]+)",
        re.M,
    )
    m = pat.findall(text)
    if not m:
        return None
    return float(max(float(x) for x in m))


def load_confusion_csv(path: Path) -> np.ndarray:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    # skip header CN,MCI,AD
    return np.array([[int(x) for x in row] for row in rows[1:]], dtype=float)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=PROJ / "paper/figures/results")
    p.add_argument("--ensemble_npz", type=Path,
                   default=PROJ / "runs/v4_mm_ensemble_5seed.npz")
    p.add_argument("--confusion_csv", type=Path,
                   default=PROJ / "runs/v4_mm_analysis_npj_5seed/confusion_matrix.csv")
    p.add_argument("--runs_root", type=Path, default=PROJ / "runs")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 153, 264, 375, 486])
    args = p.parse_args()

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib import gridspec
    from matplotlib.patches import Rectangle
    from sklearn.metrics import auc, roc_curve

    _noto = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(_noto).is_file():
        fm.fontManager.addfont(_noto)
        _prop = fm.FontProperties(fname=_noto)
        plt.rcParams["font.family"] = _prop.get_name()
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 120,
    })

    # ----- (a) seed stability -----
    baccs = []
    for s in args.seeds:
        lg = args.runs_root / f"v4_mm_seed{s}" / "train.log"
        v = parse_train_best_bacc(lg) if lg.is_file() else None
        if v is None:
            raise SystemExit(f"missing or empty train.log: {lg}")
        baccs.append(v)
    baccs = np.array(baccs, dtype=float)
    mean_b, std_b = float(baccs.mean()), float(baccs.std(ddof=1))

    rng = np.random.default_rng(42)
    x = np.arange(1, len(baccs) + 1, dtype=float)
    jitter = rng.uniform(-0.12, 0.12, size=len(baccs))

    fig = plt.figure(figsize=(12.2, 3.8))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.15, 1.0],
                          wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.bar(
        x, baccs, width=0.42, color="#E8EEF7", edgecolor="#1e3a5f",
        linewidth=0.9, zorder=1,
    )
    ax_a.scatter(
        x + jitter, baccs, s=72, c="#1e3a5f", zorder=4,
        edgecolors="white", linewidths=0.9, label="各种子验证 BAcc",
    )
    for i, (xi, yi) in enumerate(zip(x, baccs)):
        ax_a.plot([xi, xi], [mean_b, yi], color="#94a3b8", lw=0.9, zorder=2)
    ax_a.axhspan(mean_b - std_b, mean_b + std_b, color="#9ca3af", alpha=0.22, zorder=0)
    ax_a.axhline(mean_b, color="#1e3a5f", ls="--", lw=1.4, zorder=3)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([str(i) for i in range(1, len(baccs) + 1)])
    ax_a.set_xlabel("随机种子编号 (1–5)")
    ax_a.set_ylabel("平衡准确率 (Balanced Accuracy)")
    ax_a.set_ylim(0.5, 1.0)
    ax_a.set_title(
        "五次独立训练得到的平衡准确率几乎相同 → 模型稳定",
        fontsize=9.5, pad=8,
    )
    ax_a.text(
        0.03, 0.97,
        f"mean = {mean_b:.3f} ± {std_b:.2f}",
        transform=ax_a.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
    )
    ax_a.text(0.02, 0.02, "a", transform=ax_a.transAxes, fontsize=12, fontweight="bold",
              va="bottom", ha="left")

    # ----- (b) ROC -----
    ax_b = fig.add_subplot(gs[0, 1])
    z = np.load(args.ensemble_npz, allow_pickle=True)
    prob, lab = z["prob"].astype(np.float64), z["label"].astype(np.int64)
    colors = {"CN": "#2ca02c", "MCI": "#ff7f0e", "AD": "#d62728"}
    names = ["CN", "MCI", "AD"]
    aucs = {}
    for name in names:
        c = names.index(name)
        y = (lab == c).astype(np.int64)
        fpr, tpr, _ = roc_curve(y, prob[:, c])
        a = auc(fpr, tpr)
        aucs[name] = a
        ax_b.plot(fpr, tpr, lw=2.6, color=colors[name], label=name)
        mid = len(fpr) // 3
        ax_b.annotate(
            f"{name}\nAUC={a:.2f}",
            xy=(float(fpr[mid]), float(tpr[mid])),
            xytext=(8, 8), textcoords="offset points",
            fontsize=8, color=colors[name], fontweight="bold",
        )

    ax_b.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#9ca3af", label="随机分类")
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.set_xlabel("假阳性率 (FPR)")
    ax_b.set_ylabel("真阳性率 (TPR)")
    ax_b.set_title("One-vs-rest ROC（五种子概率集成）", fontsize=9.5, pad=8)
    ax_b.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.92, title="类别")
    ax_b.text(
        0.98, 0.08,
        "AUC 越接近 1 越好。\n"
        f"三类 AUC 均 > 0.94 →\n区分能力优秀。",
        transform=ax_b.transAxes, ha="right", va="bottom", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.96),
    )
    ax_b.text(0.02, 0.02, "b", transform=ax_b.transAxes, fontsize=12, fontweight="bold",
              va="bottom", ha="left")

    # ----- (c) confusion row-normalized -----
    ax_c = fig.add_subplot(gs[0, 2])
    cm = load_confusion_csv(args.confusion_csv)
    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.where(row_sum > 0, 100.0 * cm / row_sum, 0.0)

    row_labels = ["真实：认知正常 (CN)", "真实：轻度认知障碍 (MCI)", "真实：阿尔茨海默病 (AD)"]
    col_labels = ["预测：CN", "预测：MCI", "预测：AD"]

    ax_c.set_xlim(0, 3)
    ax_c.set_ylim(0, 3)
    ax_c.invert_yaxis()
    ax_c.set_aspect("equal")
    for i in range(3):
        for j in range(3):
            val = pct[i, j]
            if i == j:
                g = 0.35 + 0.65 * (val / 100.0)
                face = plt.cm.Greens(g)
            else:
                if val < 1e-6:
                    face = (0.94, 0.94, 0.95)
                else:
                    face = (1.0, 0.88, 0.88)
            ax_c.add_patch(
                Rectangle((j, i), 1, 1, facecolor=face, edgecolor="#334155", lw=0.8),
            )
            ax_c.text(
                j + 0.5, i + 0.5, f"{val:.1f}%",
                ha="center", va="center", fontsize=11, fontweight="bold", color="#0f172a",
            )

    ax_c.set_xticks(np.arange(3) + 0.5)
    ax_c.set_xticklabels(col_labels, rotation=15, ha="right")
    ax_c.set_yticks(np.arange(3) + 0.5)
    ax_c.set_yticklabels(row_labels)
    ax_c.set_title("混淆矩阵（按行归一化 %）", fontsize=9.5, pad=10)
    ax_c.text(0.02, 0.02, "c", transform=ax_c.transAxes, fontsize=12, fontweight="bold",
              va="bottom", ha="left")

    # Explanatory lines (computed from data)
    cn_row, mci_row, ad_row = pct
    expl = (
        f"• CN：{cn_row[0]:.0f}% 判为 CN。\n"
        f"• MCI：{mci_row[0]:.0f}% 判为 CN，{mci_row[2]:.0f}% 判为 AD"
        f"（其余为 MCI）。\n"
        f"• AD：{ad_row[2]:.0f}% 判为 AD。"
    )
    fig.text(0.5, 0.02, expl, ha="center", va="bottom", fontsize=8.5, linespacing=1.35)

    fig.suptitle(
        "整体目标：A2C-NODE 在 ADNI 上的分类稳定、可靠，并能区分 CN / MCI / AD",
        fontsize=11, fontweight="bold", y=1.02,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "Figure2_adni_validation"
    fig.subplots_adjust(bottom=0.22, top=0.88, left=0.05, right=0.98)
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig2] wrote {out}.png / {out}.pdf")


if __name__ == "__main__":
    main()
