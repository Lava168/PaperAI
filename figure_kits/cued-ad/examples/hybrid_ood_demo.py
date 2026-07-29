#!/usr/bin/env python3
"""
Chapter 4: Hybrid OOD Score & Temporal Uncertainty Demo

1. Hybrid OOD Score:
   - S_recon (Surprise): reconstruction error — ADNI low, HCP high
   - S_uncertainty (Disagreement): MC Dropout variance/entropy — OOD high
   - S_density (Outlier): -log P_GMM(z) — OOD high
   Score_OOD = α·S_recon + β·S_uncertainty + γ·S_density

2. Evaluation: ROC, AUROC (ID=ADNI, OOD=HCP). Goal: AUROC → 1.0

3. Temporal uncertainty (Neural ODE): ADNI band narrow, HCP band explodes.
   "The model expresses high epistemic uncertainty for OOD temporal trajectories."
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import gc

OUTPUT_DIR = SCRIPT_DIR / "hybrid_ood_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Import chapter4 modules
from chapter4_generalization.models.ood_scores import (
    HybridOODScores,
    compute_hybrid_ood_scores,
    evaluate_ood_detection,
    fit_gmm_on_latents,
    score_density,
)
from chapter4_generalization.utils.ood_visualization import (
    set_nature_style,
    plot_ood_roc_curve,
    plot_temporal_uncertainty_comparison,
    DOMAIN_COLORS,
    NATURE_PALETTE,
)


def simulate_hybrid_scores(n_id: int = 80, n_ood: int = 80, seed: int = 42):
    """Simulate S_recon, S_uncertainty, S_density for ID (ADNI) vs OOD (HCP)."""
    np.random.seed(seed)
    # ID: low recon, low uncertainty, high density (low -log P)
    S_recon_id = np.random.exponential(0.1, n_id) + 0.05
    S_unc_id = np.random.exponential(0.2, n_id) + 0.1
    S_den_id = np.random.exponential(0.3, n_id) + 0.2  # -log P moderate

    # OOD: high recon, high uncertainty, low density (high -log P)
    S_recon_ood = np.random.exponential(0.8, n_ood) + 0.5
    S_unc_ood = np.random.exponential(1.2, n_ood) + 0.8
    S_den_ood = np.random.exponential(2.0, n_ood) + 1.5

    return (
        (S_recon_id, S_unc_id, S_den_id),
        (S_recon_ood, S_unc_ood, S_den_ood),
    )


def simulate_temporal_uncertainty(max_month: int = 48, seed: int = 43):
    """ADNI: narrow band; HCP: band explodes after t>0."""
    np.random.seed(seed)
    times = np.linspace(0, max_month, 25)

    # ADNI: mean decays slowly, std grows slowly (tight trumpet)
    adni_mean = 2.0 * np.exp(-0.01 * times) + 0.5
    adni_std = 0.05 + 0.002 * times

    # HCP/OOD: mean similar at t=0, then std explodes
    hcp_mean = 2.0 * np.exp(-0.01 * times) + 0.5 + np.random.randn(len(times)) * 0.05
    hcp_std = 0.05 + 0.15 * (times ** 1.2)
    hcp_std = np.maximum(hcp_std, 0.1)

    return times, adni_mean, adni_std, hcp_mean, hcp_std


def plot_hybrid_summary(
    scores_id,
    scores_ood,
    roc_fpr,
    roc_tpr,
    roc_auc,
    output_dir: Path,
):
    """Nature-style: hybrid score distributions + ROC."""
    set_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # (a) S_recon
    ax = axes[0]
    ax.hist(scores_id.S_recon, bins=20, alpha=0.6, color=DOMAIN_COLORS["ADNI (ID)"], label="ADNI (ID)", density=True)
    ax.hist(scores_ood.S_recon, bins=20, alpha=0.6, color=DOMAIN_COLORS["HCP (OOD)"], label="HCP (OOD)", density=True)
    ax.set_xlabel(r"$S_{recon}$ (Surprise)")
    ax.set_ylabel("Density")
    ax.set_title(r"$\mathbf{a}$  $S_{recon}$", loc="left", fontweight="bold")
    ax.legend(fontsize=8)

    # (b) S_uncertainty
    ax = axes[1]
    ax.hist(scores_id.S_uncertainty, bins=20, alpha=0.6, color=DOMAIN_COLORS["ADNI (ID)"], label="ADNI (ID)", density=True)
    ax.hist(scores_ood.S_uncertainty, bins=20, alpha=0.6, color=DOMAIN_COLORS["HCP (OOD)"], label="HCP (OOD)", density=True)
    ax.set_xlabel(r"$S_{uncertainty}$ (Disagreement)")
    ax.set_ylabel("Density")
    ax.set_title(r"$\mathbf{b}$  $S_{uncertainty}$", loc="left", fontweight="bold")
    ax.legend(fontsize=8)

    # (c) ROC
    ax = axes[2]
    ax.plot(roc_fpr, roc_tpr, color=NATURE_PALETTE[0], linewidth=2.5, label=f"AUROC = {roc_auc:.3f}")
    ax.fill_between(roc_fpr, roc_tpr, color=NATURE_PALETTE[0], alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(r"$\mathbf{c}$  OOD Detection (Hybrid Score)", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Chapter 4: Hybrid OOD Score — ID vs OOD", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "ch4_hybrid_ood_summary.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "ch4_hybrid_ood_summary.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: ch4_hybrid_ood_summary.png")


def main():
    print("=" * 60)
    print("Chapter 4: Hybrid OOD Score & Temporal Uncertainty Demo")
    print("=" * 60)

    # ----- 1. Simulate hybrid score components -----
    (S_recon_id, S_unc_id, S_den_id), (S_recon_ood, S_unc_ood, S_den_ood) = simulate_hybrid_scores()

    # Combine and normalize for global scale
    S_recon_all = np.concatenate([S_recon_id, S_recon_ood])
    S_unc_all = np.concatenate([S_unc_id, S_unc_ood])
    S_den_all = np.concatenate([S_den_id, S_den_ood])

    # Normalize over full (ID+OOD) scale so ROC is comparable
    def _norm_global(a_id, a_ood):
        lo = min(a_id.min(), a_ood.min())
        hi = max(a_id.max(), a_ood.max())
        r = (hi - lo) + 1e-10
        return (a_id - lo) / r, (a_ood - lo) / r
    S_recon_n_id, S_recon_n_ood = _norm_global(S_recon_id, S_recon_ood)
    S_unc_n_id, S_unc_n_ood = _norm_global(S_unc_id, S_unc_ood)
    S_den_n_id, S_den_n_ood = _norm_global(S_den_id, S_den_ood)
    n_id, n_ood = len(S_recon_id), len(S_recon_ood)
    scores_id_arr = 1.0 * S_recon_n_id + 1.0 * S_unc_n_id + 1.0 * S_den_n_id
    scores_ood_arr = 1.0 * S_recon_n_ood + 1.0 * S_unc_n_ood + 1.0 * S_den_n_ood

    sid = compute_hybrid_ood_scores(S_recon_id, S_unc_id, S_den_id, normalize_components=False)
    sod = compute_hybrid_ood_scores(S_recon_ood, S_unc_ood, S_den_ood, normalize_components=False)
    scores_id = HybridOODScores(S_recon=sid.S_recon, S_uncertainty=sid.S_uncertainty,
                               S_density=sid.S_density, Score_OOD=scores_id_arr)
    scores_ood = HybridOODScores(S_recon=sod.S_recon, S_uncertainty=sod.S_uncertainty,
                                 S_density=sod.S_density, Score_OOD=scores_ood_arr)

    # ----- 2. ROC / AUROC -----
    fpr, tpr, auc = evaluate_ood_detection(scores_id_arr, scores_ood_arr)
    print(f"  Hybrid OOD AUROC: {auc:.4f} (goal: → 1.0)")

    # ----- 3. Plot hybrid summary -----
    plot_hybrid_summary(scores_id, scores_ood, fpr, tpr, auc, OUTPUT_DIR)

    # ----- 4. ROC curve only (Nature style) -----
    fig_roc = plot_ood_roc_curve(fpr, tpr, auc, metric_name="Hybrid OOD Score", figsize=(5, 4))
    fig_roc.savefig(OUTPUT_DIR / "ch4_hybrid_roc.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig_roc.savefig(OUTPUT_DIR / "ch4_hybrid_roc.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_roc)
    print(f"  Saved: ch4_hybrid_roc.png")

    # ----- 5. Temporal uncertainty: ADNI vs HCP -----
    times, adni_mean, adni_std, hcp_mean, hcp_std = simulate_temporal_uncertainty()
    fig_temporal = plot_temporal_uncertainty_comparison(
        times, adni_mean, adni_std,
        times, hcp_mean, hcp_std,
        y_label="Hippocampal volume / Cognitive score",
        figsize=(10, 5),
        save_path=OUTPUT_DIR / "ch4_temporal_uncertainty",
    )
    plt.close(fig_temporal)
    print(f"  Saved: ch4_temporal_uncertainty.png")

    gc.collect()
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"Outputs: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
