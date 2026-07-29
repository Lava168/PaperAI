"""
Nature-Style Visualization for OOD Uncertainty (Chapter 4)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


NATURE_PALETTE = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948']
DOMAIN_COLORS = {
    "ADNI (ID)": "#4E79A7",
    "HCP (OOD)": "#E15759",
    "AIBL/NACC (Shift)": "#F28E2B",
    "Artifacts (Noise)": "#7f7f7f",
}


def set_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.framealpha': 0.9,
        'grid.alpha': 0.25,
    })


def plot_uncertainty_distributions(
    uncertainty: Dict[str, np.ndarray],
    title: str = "Predictive Uncertainty Distributions",
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    set_nature_style()
    fig, ax = plt.subplots(figsize=figsize)

    for name, values in uncertainty.items():
        color = DOMAIN_COLORS.get(name, NATURE_PALETTE[0])
        sns.kdeplot(values, ax=ax, label=name, color=color, linewidth=2, fill=True, alpha=0.15)

    ax.set_xlabel("Predictive Entropy")
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    return fig


def plot_ood_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    metric_name: str = "Predictive Entropy",
    figsize: Tuple[float, float] = (5, 4),
) -> plt.Figure:
    set_nature_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color=NATURE_PALETTE[0], linewidth=2.5, label=f"AUC = {auc:.3f}")
    ax.fill_between(fpr, tpr, color=NATURE_PALETTE[0], alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"OOD Detection ({metric_name})", fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    return fig


def plot_3d_uncertainty_space(
    embeddings: np.ndarray,
    uncertainty: np.ndarray,
    domains: List[str],
    title: str = "3D Uncertainty Landscape",
    figsize: Tuple[float, float] = (9, 7),
    elev: float = 20,
    azim: float = 40,
) -> plt.Figure:
    """
    3D latent space with uncertainty coloring (Nature-style depth shading).
    """
    from sklearn.decomposition import PCA

    set_nature_style()
    pca = PCA(n_components=3)
    pts = pca.fit_transform(embeddings)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    norm = Normalize(vmin=np.percentile(uncertainty, 5), vmax=np.percentile(uncertainty, 95))
    cmap = plt.cm.viridis

    for domain in sorted(set(domains)):
        mask = np.array(domains) == domain
        colors = cmap(norm(uncertainty[mask]))
        ax.scatter(
            pts[mask, 0], pts[mask, 1], pts[mask, 2],
            c=colors,
            s=20,
            alpha=0.85,
            depthshade=True,
            edgecolors="white",
            linewidths=0.2,
            label=domain,
        )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title, fontweight="bold")

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label("Predictive Entropy")

    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig


def plot_comprehensive_ood_figure(
    embeddings: np.ndarray,
    uncertainty_by_domain: Dict[str, np.ndarray],
    domains: List[str],
    roc: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    uncertainty_components: Optional[Dict[str, np.ndarray]] = None,
    figsize: Tuple[float, float] = (13, 10),
) -> plt.Figure:
    """
    Nature-style multi-panel summary figure for OOD uncertainty.
    """
    set_nature_style()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

    # (a) 3D uncertainty space
    ax_a = fig.add_subplot(gs[0, :2], projection="3d")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pts = pca.fit_transform(embeddings)
    all_unc = np.concatenate(list(uncertainty_by_domain.values()))
    norm = Normalize(vmin=np.percentile(all_unc, 5), vmax=np.percentile(all_unc, 95))
    cmap = plt.cm.viridis

    for domain in sorted(set(domains)):
        mask = np.array(domains) == domain
        colors = cmap(norm(all_unc[mask]))
        ax_a.scatter(
            pts[mask, 0], pts[mask, 1], pts[mask, 2],
            c=colors, s=18, alpha=0.85, depthshade=True,
            edgecolors="white", linewidths=0.2, label=domain,
        )
    ax_a.view_init(20, 40)
    ax_a.set_title(r'$\mathbf{a}$  3D Uncertainty Landscape', loc='left', fontweight='bold')
    ax_a.set_xlabel("PC1", fontsize=9)
    ax_a.set_ylabel("PC2", fontsize=9)
    ax_a.set_zlabel("PC3", fontsize=9)

    # (b) Uncertainty distributions
    ax_b = fig.add_subplot(gs[0, 2:])
    for name, values in uncertainty_by_domain.items():
        color = DOMAIN_COLORS.get(name, NATURE_PALETTE[0])
        sns.kdeplot(values, ax=ax_b, label=name, color=color, linewidth=2, fill=True, alpha=0.12)
    ax_b.set_xlabel("Predictive Entropy")
    ax_b.set_ylabel("Density")
    ax_b.set_title(r'$\mathbf{b}$  Uncertainty Distributions', loc='left', fontweight='bold')
    ax_b.legend(loc='upper right', fontsize=8)

    # (c) ROC for ID vs HCP OOD
    ax_c = fig.add_subplot(gs[1, :2])
    if "entropy" in roc:
        fpr, tpr, auc = roc["entropy"]
        ax_c.plot(fpr, tpr, color=NATURE_PALETTE[0], linewidth=2.5, label=f"AUC={auc:.3f}")
        ax_c.fill_between(fpr, tpr, color=NATURE_PALETTE[0], alpha=0.15)
    ax_c.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax_c.set_xlabel("False Positive Rate")
    ax_c.set_ylabel("True Positive Rate")
    ax_c.set_title(r'$\mathbf{c}$  OOD Detection ROC', loc='left', fontweight='bold')
    ax_c.legend(loc='lower right', fontsize=8)

    # (d) Uncertainty by domain (boxplot)
    ax_d = fig.add_subplot(gs[1, 2:])
    data = []
    labels = []
    for name, values in uncertainty_by_domain.items():
        data.append(values)
        labels.append(name)
    ax_d.boxplot(data, labels=labels, patch_artist=True,
                 boxprops=dict(facecolor=NATURE_PALETTE[1], alpha=0.5),
                 medianprops=dict(color='black', linewidth=1.5))
    ax_d.set_ylabel("Predictive Entropy")
    ax_d.set_title(r'$\mathbf{d}$  Domain-wise Uncertainty', loc='left', fontweight='bold')
    ax_d.tick_params(axis='x', labelrotation=15)

    # (e) Epistemic vs Aleatoric (if provided)
    ax_e = fig.add_subplot(gs[2, :2])
    if uncertainty_components and "epistemic" in uncertainty_components and "aleatoric" in uncertainty_components:
        ax_e.scatter(uncertainty_components["aleatoric"], uncertainty_components["epistemic"],
                     s=15, alpha=0.6, c=NATURE_PALETTE[2], edgecolors='white', linewidth=0.2)
        ax_e.set_xlabel("Aleatoric (Expected Entropy)")
        ax_e.set_ylabel("Epistemic (Mutual Info)")
        ax_e.set_title(r'$\mathbf{e}$  Uncertainty Decomposition', loc='left', fontweight='bold')
    else:
        ax_e.text(0.5, 0.5, "Epistemic/Aleatoric\nnot provided",
                  ha="center", va="center", transform=ax_e.transAxes)
        ax_e.set_title(r'$\mathbf{e}$  Uncertainty Decomposition', loc='left', fontweight='bold')
        ax_e.axis("off")

    # (f) Summary panel
    ax_f = fig.add_subplot(gs[2, 2:])
    ax_f.axis("off")
    stats_text = "Summary (Synthetic Demo)\n" \
                 "• ID: ADNI older adults\n" \
                 "• OOD: HCP young adults\n" \
                 "• Shift: AIBL/NACC\n" \
                 "• Noise: artifact perturbations\n" \
                 "Goal: High uncertainty on OOD"
    ax_f.text(0.05, 0.9, stats_text, fontsize=10, va="top",
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
    ax_f.set_title(r'$\mathbf{f}$  Study Summary', loc='left', fontweight='bold')

    fig.suptitle("Chapter 4: Bayesian Uncertainty for OOD Detection",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig


def plot_temporal_uncertainty_single(
    times: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ax: Optional[plt.Axes] = None,
    color: str = NATURE_PALETTE[0],
    label: str = "Mean",
    n_sigma: float = 2.0,
    alpha_band: float = 0.35,
) -> plt.Axes:
    """
    Plot trajectory with shaded ±n_sigma band (e.g. ±2σ).
    X: time (months), Y: metric (e.g. hippocampal volume / cognitive score).
    Nature-style: tight band for ID (ADNI), exploding band for OOD (HCP).
    """
    if ax is None:
        set_nature_style()
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, mean, "-", color=color, linewidth=2.5, label=label)
    ax.fill_between(
        times,
        mean - n_sigma * std,
        mean + n_sigma * std,
        color=color,
        alpha=alpha_band,
    )
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Metric (e.g. volume / score)")
    return ax


def plot_temporal_uncertainty_comparison(
    adni_times: np.ndarray,
    adni_mean: np.ndarray,
    adni_std: np.ndarray,
    hcp_times: np.ndarray,
    hcp_mean: np.ndarray,
    hcp_std: np.ndarray,
    y_label: str = "Hippocampal volume / Cognitive score",
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Neural ODE temporal uncertainty: ADNI vs HCP/OOD.

    Figure A (ADNI): narrow confidence band (tight trumpet).
    Figure B (HCP/OOD): band explodes after t>0 — high epistemic uncertainty.

    "The model expresses high epistemic uncertainty for out-of-distribution
     temporal trajectories."
    """
    set_nature_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    plot_temporal_uncertainty_single(
        adni_times, adni_mean, adni_std,
        ax=ax_a, color=DOMAIN_COLORS["ADNI (ID)"], label="ADNI (ID) mean ±2σ", n_sigma=2.0,
    )
    ax_a.set_title(r"$\mathbf{a}$  ADNI (In-Distribution)", loc="left", fontweight="bold")
    ax_a.set_ylabel(y_label)
    ax_a.set_xlim(0, max(adni_times) if len(adni_times) else 48)
    ax_a.legend(loc="upper right", fontsize=8)
    ax_a.grid(True, alpha=0.3)

    plot_temporal_uncertainty_single(
        hcp_times, hcp_mean, hcp_std,
        ax=ax_b, color=DOMAIN_COLORS["HCP (OOD)"], label="HCP (OOD) mean ±2σ", n_sigma=2.0,
    )
    ax_b.set_title(r"$\mathbf{b}$  HCP (Out-of-Distribution)", loc="left", fontweight="bold")
    ax_b.set_xlim(0, max(hcp_times) if len(hcp_times) else 48)
    ax_b.legend(loc="upper right", fontsize=8)
    ax_b.grid(True, alpha=0.3)

    fig.suptitle(
        "Temporal uncertainty: ID vs OOD trajectories (Neural ODE)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in ["png", "pdf"]:
            fig.savefig(save_path.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def save_figure(fig: plt.Figure, path: Path, formats: List[str] = ["png", "pdf"], dpi: int = 300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight", facecolor="white")
