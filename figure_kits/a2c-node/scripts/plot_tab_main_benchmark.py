#!/usr/bin/env python3
"""Table~\\ref{tab:main} benchmark: grouped metrics + illustrative 3D cortical panels.

Uses the project's Nature rcParams (:mod:`compose_paper_figures`) and nilearn
fsaverage5 surfaces. Each small-multiple shows the *same* anatomical mesh;
vertex colour encodes that model's balanced accuracy (performance scale only,
not a saliency or attention map).

Outputs:
  paper/figures/results/Figure_tab_main_benchmark.{png,pdf}

Inputs (defaults under repo root):
  runs/v4_mm_analysis_npj_5seed/baseline_summary.json
  runs/baselines/temporal_baseline_summary.json
  runs/baselines/cog_mae_baselines.json
  runs/v4_mm_analysis_npj_5seed/metrics.json
  runs/v4_mm_analysis_npj_5seed/cog_mae.json  (ensemble MMSE MAE)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from compose_paper_figures import NAT_PALETTE, apply_nature_rc, panel_label  # noqa: E402


def _row(
    short: str,
    title: str,
    bacc: float,
    bacc_se: Optional[float],
    auc: float,
    auc_se: Optional[float],
    mae: Optional[float],
) -> Dict[str, Any]:
    return {
        "short": short,
        "title": title,
        "bacc": bacc,
        "bacc_se": bacc_se,
        "auc": auc,
        "auc_se": auc_se,
        "mae": mae,
    }


def load_rows(proj: Path) -> List[Dict[str, Any]]:
    bsl = json.loads(
        (proj / "runs/v4_mm_analysis_npj_5seed/baseline_summary.json").read_text()
    )["models"]
    tbs = json.loads(
        (proj / "runs/baselines/temporal_baseline_summary.json").read_text()
    )
    cog = json.loads((proj / "runs/baselines/cog_mae_baselines.json").read_text())
    metrics = json.loads(
        (proj / "runs/v4_mm_analysis_npj_5seed/metrics.json").read_text()
    )
    cog_ens = json.loads(
        (proj / "runs/v4_mm_analysis_npj_5seed/cog_mae.json").read_text()
    )

    rn = bsl["3D ResNet-18"]

    def se_b(model: dict) -> Tuple[float, Optional[float]]:
        return float(model["test_bacc_mean"]), float(model["test_bacc_std"])

    def se_a(model: dict) -> Tuple[float, Optional[float]]:
        am = model.get("test_macro_auc_mean")
        if am is None:
            return float("nan"), None
        return float(am), float(model.get("test_macro_auc_std") or 0.0)

    b_r, s_r = se_b(rn)
    auc_r, auc_rs = se_a(rn)

    rows: List[Dict[str, Any]] = [
        _row("ResNet", "3D ResNet-18\n(static)", b_r, s_r, auc_r, auc_rs, None),
    ]
    for key, tkey, caption in [
        ("lstm", "lstm", "LSTM\n(6-mo grid)"),
        ("transformer", "transformer", "Time-aware\nTransformer"),
        ("latent_ode", "latent_ode", "STE-ODE\n(latent ODE)"),
        ("mlp", "mlp", "BrainODE\n(MLP head)"),
    ]:
        m = tbs[key]
        ck = cog[tkey]
        rows.append(
            _row(
                key,
                caption,
                float(m["BAcc"]),
                None,
                float(m["macro_AUC"]),
                None,
                float(ck["MAE_calibrated"]),
            )
        )

    rows.append(
        _row(
            "ours",
            "A2C-NODE\n(5-seed ens.)",
            float(metrics["balanced_accuracy"]),
            None,
            float(metrics["macro_auc"]),
            None,
            float(cog_ens["MAE_calibrated"]),
        )
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", type=Path, default=PROJ)
    ap.add_argument(
        "--out_dir", type=Path, default=PROJ / "paper/figures/results"
    )
    args = ap.parse_args()

    apply_nature_rc()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from nilearn import datasets, plotting, surface

    rows = load_rows(args.proj)
    n = len(rows)

    fig = plt.figure(figsize=(7.2, 5.35))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[1.05, 1.0], hspace=0.34,
        left=0.08, right=0.98, top=0.94, bottom=0.07,
    )

    # ----- (a) bars -----
    ax_a = fig.add_subplot(gs[0])
    x = np.arange(n, dtype=float)
    w = 0.34
    baccs = [r["bacc"] for r in rows]
    aucs = [r["auc"] for r in rows]
    eb_b = [r["bacc_se"] if r["bacc_se"] is not None else 0.0 for r in rows]
    eb_a = [r["auc_se"] if r["auc_se"] is not None else 0.0 for r in rows]

    ax_a.bar(
        x - w / 2,
        baccs,
        width=w,
        yerr=eb_b,
        capsize=2.0,
        label="Balanced accuracy",
        color=NAT_PALETTE["navy"],
        edgecolor=NAT_PALETTE["ink"],
        linewidth=0.45,
        ecolor=NAT_PALETTE["slate"],
        error_kw={"elinewidth": 0.6, "capthick": 0.6},
    )
    ax_a.bar(
        x + w / 2,
        aucs,
        width=w,
        yerr=eb_a,
        capsize=2.0,
        label="Macro-AUC (one-vs-rest)",
        color=NAT_PALETTE["sage"],
        edgecolor=NAT_PALETTE["ink"],
        linewidth=0.45,
        ecolor=NAT_PALETTE["slate"],
        error_kw={"elinewidth": 0.6, "capthick": 0.6},
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([r["short"] for r in rows], rotation=18, ha="right")
    ax_a.set_ylabel("Score")
    ax_a.set_ylim(0.25, 1.02)
    ax_a.axhline(1.0 / 3.0, color=NAT_PALETTE["fog"], ls=(0, (2, 3)), lw=0.8, zorder=0)
    ax_a.legend(loc="upper left", ncol=1)
    ax_a.set_title(
        "ADNI three-class prediction at last visit (test n = 121)",
        loc="left",
        color=NAT_PALETTE["slate"],
        pad=4,
    )
    panel_label(ax_a, "a", dx=-0.07, dy=1.06)

    # ----- (b) cortical small multiples -----
    gs_b = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1],
                                            hspace=0.22, wspace=0.08)
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    mesh = fsavg["pial_left"]
    sulc = fsavg["sulc_left"]
    coord, _ = surface.load_surf_mesh(mesh)
    n_v = coord.shape[0]

    vmin, vmax = 0.32, 0.92
    cmap = plt.cm.cividis

    for i, r in enumerate(rows):
        rr, cc = divmod(i, 4)
        ax = fig.add_subplot(gs_b[rr, cc], projection="3d")
        stat = np.full(n_v, r["bacc"], dtype=np.float32)
        plotting.plot_surf_stat_map(
            mesh,
            stat,
            hemi="left",
            view="lateral",
            axes=ax,
            figure=fig,
            bg_map=sulc,
            bg_on_data=False,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            threshold=1e-6,
            colorbar=False,
        )
        ax.set_axis_off()
        mae = r["mae"]
        mae_s = f"\nMMSE MAE {mae:.2f}" if mae is not None else "\n(no MMSE head)"
        ax.set_title(
            f"{r['title']}{mae_s}",
            fontsize=6.2,
            color=NAT_PALETTE["slate"],
            pad=2,
        )

    cax = fig.add_subplot(gs_b[1, 3])
    cax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=cax, fraction=0.92, pad=0.04)
    cb.set_label("Balanced accuracy\n(vertex colour)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    fig.text(
        0.072,
        0.505,
        "b",
        transform=fig.transFigure,
        fontweight="bold",
        fontsize=9,
        color=NAT_PALETTE["ink"],
        ha="left",
        va="top",
    )

    fig.text(
        0.08,
        0.02,
        "Illustrative fsaverage5 pial surfaces (left lateral). "
        "Colour encodes balanced accuracy only; shared anatomy is not a model saliency map.",
        fontsize=6.5,
        color=NAT_PALETTE["slate"],
        ha="left",
        va="bottom",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / "Figure_tab_main_benchmark.png"
    out_pdf = args.out_dir / "Figure_tab_main_benchmark.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[tab_main] wrote {out_png} / {out_pdf}")


if __name__ == "__main__":
    main()