#!/usr/bin/env python3
"""Ablations figure for Table tab:abl (Nature style, English).

Outputs:
  paper/figures/results/Figure_tab_ablations.{png,pdf}
  paper/figures/results/Figure_tab_ablations_data.{csv,json}

Data sources:
  - Full model: runs/v4_mm_analysis_npj_5seed/metrics.json
  - No edges / MLP forecast: runs/baselines/temporal_baseline_summary.json
    (latent_ode and mlp heads trained at seed 42 in shared backbone)
  - No class weights: hard-coded collapse numbers from Table tab:abl
  - Eval-only counterfactual wrapper: identical to full for classification
  - ATE consistency (Spearman rho): hard-coded from Table tab:abl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from compose_paper_figures import NAT_PALETTE, apply_nature_rc, panel_label  # noqa: E402


def _row(
    key: str,
    label: str,
    bacc: float,
    auc: float,
    ate_rho: Optional[float],
    note: str = "",
) -> Dict[str, Any]:
    return {
        "variant_id": key,
        "label": label,
        "bacc": float(bacc),
        "macro_auc": float(auc),
        "ate_consistency_rho": (None if ate_rho is None else float(ate_rho)),
        "note": note,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=PROJ / "paper/figures/results")
    ap.add_argument(
        "--metrics_json",
        type=Path,
        default=PROJ / "runs/v4_mm_analysis_npj_5seed/metrics.json",
    )
    ap.add_argument(
        "--temporal_summary_json",
        type=Path,
        default=PROJ / "runs/baselines/temporal_baseline_summary.json",
    )
    args = ap.parse_args()

    apply_nature_rc()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    full = json.loads(args.metrics_json.read_text())
    tbs = json.loads(args.temporal_summary_json.read_text())

    b_full = float(full["balanced_accuracy"])
    a_full = float(full["macro_auc"])

    # Table tab:abl constants (used when the underlying run artifacts are not saved)
    collapse_bacc, collapse_auc = 0.333, 0.517
    rho_full = 1.00
    rho_no_edges = 0.87
    rho_mlp = 0.87

    b_no_edges = float(tbs["latent_ode"]["BAcc"])
    a_no_edges = float(tbs["latent_ode"]["macro_AUC"])
    b_mlp = float(tbs["mlp"]["BAcc"])
    a_mlp = float(tbs["mlp"]["macro_AUC"])

    rows: List[Dict[str, Any]] = [
        _row("full", "Full (A2C-NODE)", b_full, a_full, rho_full),
        _row("no_edges", "− GCN → per-node ODE (no edges)", b_no_edges, a_no_edges, rho_no_edges,
             "Diagonal latent ODE head (no edges)"),
        _row("mlp_forecast", "− ODE → MLP forecast", b_mlp, a_mlp, rho_mlp,
             "Direct MLP map h(t)=h0+Δ(h0,t)"),
        _row("no_rebalancing", "− class weight & sampler", collapse_bacc, collapse_auc, None,
             "Training collapses to majority class"),
        _row("no_cf_train", "− counterfactual wrapper (eval)", b_full, a_full, None,
             "Classification identical; ATE not applicable"),
    ]

    # ------------------------------ export data
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "Figure_tab_ablations_data.csv"
    out_json = args.out_dir / "Figure_tab_ablations_data.json"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    out_json.write_text(json.dumps({"rows": rows}, indent=2))

    # ------------------------------ plot
    fig = plt.figure(figsize=(7.2, 2.9))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.25, 0.9], wspace=0.42,
        left=0.08, right=0.98, top=0.92, bottom=0.18,
    )

    # (a) BAcc + AUC
    ax_a = fig.add_subplot(gs[0])
    x = np.arange(len(rows), dtype=float)
    w = 0.34
    baccs = [r["bacc"] for r in rows]
    aucs = [r["macro_auc"] for r in rows]

    ax_a.bar(
        x - w / 2,
        baccs,
        width=w,
        label="Balanced accuracy",
        color=NAT_PALETTE["navy"],
        edgecolor=NAT_PALETTE["ink"],
        linewidth=0.45,
    )
    ax_a.bar(
        x + w / 2,
        aucs,
        width=w,
        label="Macro-AUC (OvR)",
        color=NAT_PALETTE["sage"],
        edgecolor=NAT_PALETTE["ink"],
        linewidth=0.45,
    )
    ax_a.axhline(1.0 / 3.0, color=NAT_PALETTE["fog"], ls=(0, (2, 3)), lw=0.8, zorder=0)
    ax_a.set_ylim(0.25, 1.02)
    ax_a.set_ylabel("Score")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([r["variant_id"] for r in rows], fontsize=6.8)
    ax_a.tick_params(axis="x", pad=3)
    ax_a.legend(loc="upper left", ncol=1)
    ax_a.set_title("Ablations on ADNI test split (n = 121)", loc="left",
                   color=NAT_PALETTE["slate"], pad=4)
    panel_label(ax_a, "a", dx=-0.07, dy=1.05)

    # (b) ATE consistency (rho)
    ax_b = fig.add_subplot(gs[1])
    rho = [r["ate_consistency_rho"] for r in rows]
    rho_val = [0.0 if v is None else float(v) for v in rho]
    colors = [
        NAT_PALETTE["navy"] if v is not None else NAT_PALETTE["fog"] for v in rho
    ]
    ax_b.bar(
        x,
        rho_val,
        width=0.52,
        color=colors,
        edgecolor=NAT_PALETTE["ink"],
        linewidth=0.45,
    )
    for xi, v in zip(x, rho):
        if v is None:
            ax_b.text(xi, 0.03, "n/a", ha="center", va="bottom",
                      fontsize=6.5, color=NAT_PALETTE["slate"])
        else:
            ax_b.text(xi, float(v) + 0.03, f"{float(v):.2f}", ha="center", va="bottom",
                      fontsize=6.5, color=NAT_PALETTE["ink"])
    ax_b.set_ylim(0, 1.10)
    ax_b.set_ylabel("ATE consistency\n(Spearman ρ)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([r["variant_id"] for r in rows], rotation=45, ha="right", fontsize=6.2)
    ax_b.set_title("Sanity check: Braak-style ordering", loc="left",
                   color=NAT_PALETTE["slate"], pad=4)
    panel_label(ax_b, "b", dx=-0.16, dy=1.05)

    fig.text(
        0.08,
        0.04,
        "ρ computed on |ATE_AD| ordering (ventricle > hippocampus > amygdala). "
        "Grey bars indicate not applicable / not computed.",
        fontsize=6.5,
        color=NAT_PALETTE["slate"],
        ha="left",
        va="bottom",
    )

    out_png = args.out_dir / "Figure_tab_ablations.png"
    out_pdf = args.out_dir / "Figure_tab_ablations.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[ablations] wrote {out_png} / {out_pdf}")
    print(f"[ablations] data  {out_csv} / {out_json}")


if __name__ == "__main__":
    main()

