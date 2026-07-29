#!/usr/bin/env python3
"""
Post-hoc Pathway & Subtype Analysis

After training PathwaySubtypeModel, this script:
  1. Extracts z_atrophy, z_vascular for all subjects
  2. Runs SubtypeHead inference -> assigns subtypes
  3. Runs counterfactual decomposition -> pathway attribution
  4. Cross-references subtypes with clinical variables
  5. Generates publication-ready figures
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

NUM_REGIONS = 81

DKT_REGION_NAMES = [
    "Background",
    "L-Caudate", "L-Putamen", "L-Pallidum", "L-Hippocampus", "L-Amygdala",
    "L-Thalamus", "L-Accumbens", "R-Caudate", "R-Putamen", "R-Pallidum",
    "R-Hippocampus", "R-Amygdala", "R-Thalamus", "R-Accumbens",
    "L-BanksSTS", "L-CaudalAnteriorCingulate", "L-CaudalMiddleFrontal",
    "L-Cuneus", "L-Entorhinal", "L-Fusiform", "L-InferiorParietal",
    "L-InferiorTemporal", "L-IsthmusCingulate", "L-LateralOccipital",
    "L-LateralOrbitofrontal", "L-Lingual", "L-MedialOrbitofrontal",
    "L-MiddleTemporal", "L-Parahippocampal", "L-Paracentral",
    "L-ParsOpercularis", "L-ParsOrbitalis", "L-ParsTriangularis",
    "L-Pericalcarine", "L-Postcentral", "L-PosteriorCingulate",
    "L-Precentral", "L-Precuneus", "L-RostralAnteriorCingulate",
    "L-RostralMiddleFrontal", "L-SuperiorFrontal", "L-SuperiorParietal",
    "L-SuperiorTemporal", "L-SupraMarginal", "L-FrontalPole",
    "L-TemporalPole", "L-TransverseTemporal", "L-Insula",
    "R-BanksSTS", "R-CaudalAnteriorCingulate", "R-CaudalMiddleFrontal",
    "R-Cuneus", "R-Entorhinal", "R-Fusiform", "R-InferiorParietal",
    "R-InferiorTemporal", "R-IsthmusCingulate", "R-LateralOccipital",
    "R-LateralOrbitofrontal", "R-Lingual", "R-MedialOrbitofrontal",
    "R-MiddleTemporal", "R-Parahippocampal", "R-Paracentral",
    "R-ParsOpercularis", "R-ParsOrbitalis", "R-ParsTriangularis",
    "R-Pericalcarine", "R-Postcentral", "R-PosteriorCingulate",
    "R-Precentral", "R-Precuneus", "R-RostralAnteriorCingulate",
    "R-RostralMiddleFrontal", "R-SuperiorFrontal", "R-SuperiorParietal",
    "R-SuperiorTemporal", "R-SupraMarginal", "R-FrontalPole",
    "R-TemporalPole", "R-TransverseTemporal", "R-Insula",
]

DX_NAMES = ["CN", "MCI", "AD"]


def load_model(ckpt_path, args):
    from train_pathway import PathwaySubtypeModel

    model = PathwaySubtypeModel(
        feature_dim=args.feature_dim,
        num_subtypes=args.num_subtypes,
        latent_dim=args.latent_dim,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, loader, device):
    """Extract latent features and subtype assignments for all subjects."""
    records = []
    model.to(device)

    for batch in loader:
        img = batch["image"].to(device)
        with torch.amp.autocast("cuda", enabled=True):
            out = model(img)

        B = img.shape[0]
        z_a = out["z_atrophy"].cpu().numpy()
        z_v = out["z_vascular"].cpu().numpy()
        assign = out["subtype_assign"].cpu().numpy()
        dx_logits = out["diagnosis_logits"].cpu().numpy()
        ad_frac = out["cf_ad_fraction"].cpu().numpy()

        for i in range(B):
            records.append({
                "z_a": z_a[i],
                "z_v": z_v[i],
                "subtype_probs": assign[i],
                "subtype": int(assign[i].argmax()),
                "dx_pred": int(dx_logits[i].argmax()),
                "ad_fraction_mean": float(ad_frac[i].mean()),
                "ad_fraction_per_region": ad_frac[i],
                "label": batch["label"][i].item(),
                "amyloid_status": batch["amyloid_status"][i].item(),
                "wmh_total": batch["wmh_total"][i].item(),
                "csf_abeta": batch["csf_abeta"][i].item(),
                "age": batch["age"][i].item(),
                "apoe_e4": batch["apoe_e4"][i].item(),
            })

    return records


def subtype_clinical_analysis(records, output_dir):
    """Cross-reference discovered subtypes with clinical variables."""
    import warnings
    warnings.filterwarnings("ignore")

    df = pd.DataFrame([{
        "subtype": r["subtype"],
        "label": r["label"],
        "dx_pred": r["dx_pred"],
        "ad_fraction_mean": r["ad_fraction_mean"],
        "amyloid_status": r["amyloid_status"] if r["amyloid_status"] >= 0 else np.nan,
        "wmh_total": r["wmh_total"] if not np.isnan(r["wmh_total"]) else np.nan,
        "csf_abeta": r["csf_abeta"] if not np.isnan(r["csf_abeta"]) else np.nan,
        "age": r["age"] if not np.isnan(r["age"]) else np.nan,
        "apoe_e4": r["apoe_e4"] if r["apoe_e4"] >= 0 else np.nan,
        "z_a_norm": np.linalg.norm(r["z_a"]),
        "z_v_norm": np.linalg.norm(r["z_v"]),
    } for r in records])

    report = {}

    # Subtype distribution
    sub_counts = df["subtype"].value_counts().sort_index()
    report["subtype_counts"] = sub_counts.to_dict()

    # Per-subtype clinical profiles
    profiles = {}
    for s in sorted(df["subtype"].unique()):
        sub_df = df[df["subtype"] == s]
        profile = {
            "n": len(sub_df),
            "dx_distribution": sub_df["label"].value_counts().to_dict(),
            "mean_age": float(sub_df["age"].mean()),
            "mean_ad_fraction": float(sub_df["ad_fraction_mean"].mean()),
            "mean_z_a_norm": float(sub_df["z_a_norm"].mean()),
            "mean_z_v_norm": float(sub_df["z_v_norm"].mean()),
            "ad_dominant_ratio": float(
                sub_df["z_a_norm"].mean() /
                (sub_df["z_a_norm"].mean() + sub_df["z_v_norm"].mean() + 1e-8)
            ),
        }

        if sub_df["amyloid_status"].notna().sum() > 0:
            profile["amyloid_positive_rate"] = float(sub_df["amyloid_status"].mean())
        if sub_df["csf_abeta"].notna().sum() > 0:
            profile["mean_csf_abeta"] = float(sub_df["csf_abeta"].mean())
        if sub_df["wmh_total"].notna().sum() > 0:
            profile["mean_wmh_volume"] = float(sub_df["wmh_total"].mean())
        if sub_df["apoe_e4"].notna().sum() > 0:
            profile["apoe_e4_rate"] = float(sub_df["apoe_e4"].mean())

        profiles[f"subtype_{s}"] = profile

    report["subtype_profiles"] = profiles

    # Label interpretation
    for name, prof in profiles.items():
        ratio = prof["ad_dominant_ratio"]
        if ratio > 0.65:
            prof["interpretation"] = "AD-dominant (atrophy-driven)"
        elif ratio < 0.35:
            prof["interpretation"] = "Vascular-dominant"
        else:
            prof["interpretation"] = "Mixed pathology"

    # Statistical tests
    try:
        from scipy.stats import kruskal, chi2_contingency
        stats = {}
        for col in ["age", "z_a_norm", "z_v_norm", "ad_fraction_mean"]:
            groups = [g[col].dropna().values for _, g in df.groupby("subtype")]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) >= 2:
                stat, pval = kruskal(*groups)
                stats[f"kruskal_{col}"] = {"H": float(stat), "p": float(pval)}

        ct = pd.crosstab(df["subtype"], df["label"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = chi2_contingency(ct)
            stats["chi2_subtype_dx"] = {"chi2": float(chi2), "p": float(p), "dof": int(dof)}

        report["statistical_tests"] = stats
    except ImportError:
        pass

    with open(os.path.join(output_dir, "subtype_analysis.json"), "w") as f:
        json.dump(report, f, indent=2, default=lambda x: float(x))

    print("\n=== Subtype Clinical Profiles ===")
    for name, prof in profiles.items():
        dx_str = ", ".join(f"{DX_NAMES[k]}:{v}" for k, v in sorted(prof["dx_distribution"].items()))
        print(f"  {name} (n={prof['n']}): AD-ratio={prof['ad_dominant_ratio']:.2f} | {prof.get('interpretation','')} | {dx_str}")

    return report


def pathway_attribution_analysis(records, output_dir):
    """Analyze regional pathway contributions."""
    ad_fracs = np.array([r["ad_fraction_per_region"] for r in records])
    labels = np.array([r["label"] for r in records])

    results = {}
    for dx_idx, dx_name in enumerate(DX_NAMES):
        mask = labels == dx_idx
        if mask.sum() == 0:
            continue
        mean_frac = ad_fracs[mask].mean(axis=0)
        top_ad = np.argsort(-mean_frac)[:10]
        top_vas = np.argsort(mean_frac)[:10]

        results[dx_name] = {
            "mean_ad_fraction": float(mean_frac.mean()),
            "top_ad_driven_regions": [
                {"region": DKT_REGION_NAMES[min(r+1, len(DKT_REGION_NAMES)-1)],
                 "ad_fraction": float(mean_frac[r])}
                for r in top_ad
            ],
            "top_vascular_driven_regions": [
                {"region": DKT_REGION_NAMES[min(r+1, len(DKT_REGION_NAMES)-1)],
                 "ad_fraction": float(mean_frac[r])}
                for r in top_vas
            ],
        }

    with open(os.path.join(output_dir, "pathway_attribution.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Pathway Attribution ===")
    for dx, info in results.items():
        print(f"  {dx}: mean AD-driven fraction = {info['mean_ad_fraction']:.3f}")
        print(f"    Top AD-driven: {[r['region'] for r in info['top_ad_driven_regions'][:5]]}")
        print(f"    Top Vascular:  {[r['region'] for r in info['top_vascular_driven_regions'][:5]]}")

    return results


def generate_figures(records, output_dir):
    """Generate t-SNE / UMAP and other figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("matplotlib/sklearn not available, skipping figures")
        return

    z_all = np.array([np.concatenate([r["z_a"], r["z_v"]]) for r in records])
    labels = np.array([r["label"] for r in records])
    subtypes = np.array([r["subtype"] for r in records])

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb = tsne.fit_transform(z_all)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors_dx = {0: "#2196F3", 1: "#FF9800", 2: "#F44336"}
    for dx in range(3):
        mask = labels == dx
        axes[0].scatter(emb[mask, 0], emb[mask, 1], c=colors_dx[dx],
                        label=DX_NAMES[dx], alpha=0.6, s=15)
    axes[0].set_title("Disentangled Space (by Diagnosis)")
    axes[0].legend()

    colors_sub = plt.cm.Set2(np.linspace(0, 1, max(subtypes) + 1))
    for s in range(max(subtypes) + 1):
        mask = subtypes == s
        axes[1].scatter(emb[mask, 0], emb[mask, 1], c=[colors_sub[s]],
                        label=f"Subtype {s}", alpha=0.6, s=15)
    axes[1].set_title("Disentangled Space (by Discovered Subtype)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_tsne_subtypes.png"), dpi=200)
    plt.close()

    # Pathway bar chart per subtype
    ad_fracs = np.array([r["ad_fraction_per_region"] for r in records])
    fig, axes = plt.subplots(1, max(subtypes) + 1, figsize=(5 * (max(subtypes) + 1), 5), sharey=True)
    if max(subtypes) == 0:
        axes = [axes]
    for s in range(max(subtypes) + 1):
        mask = subtypes == s
        mean_frac = ad_fracs[mask].mean(axis=0)
        ax = axes[s]
        ax.barh(range(min(20, len(mean_frac))), mean_frac[:20], color=colors_sub[s])
        ax.set_xlabel("AD Fraction")
        ax.set_title(f"Subtype {s} (n={mask.sum()})")
        ax.set_yticks(range(min(20, len(mean_frac))))
        ax.set_yticklabels([DKT_REGION_NAMES[i+1] if i+1 < len(DKT_REGION_NAMES) else f"R{i}"
                            for i in range(min(20, len(mean_frac)))], fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_pathway_by_subtype.png"), dpi=200)
    plt.close()
    print(f"  Figures saved to {output_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="analysis_pathway")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--feature_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--num_subtypes", type=int, default=4)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir = SCRIPT_DIR / "data"
    if args.csv is None:
        args.csv = str(data_dir / "splits_balanced" / "test.csv")
    if args.cache_dir is None:
        args.cache_dir = str(data_dir / "full_cache")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model = load_model(args.checkpoint, args)

    from train_pathway import PathwayDataset
    dataset = PathwayDataset(args.csv, args.cache_dir, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    print("Extracting features...")
    records = extract_features(model, loader, device)
    print(f"  Extracted {len(records)} samples")

    print("\nRunning subtype clinical analysis...")
    subtype_clinical_analysis(records, args.output_dir)

    print("\nRunning pathway attribution analysis...")
    pathway_attribution_analysis(records, args.output_dir)

    print("\nGenerating figures...")
    generate_figures(records, args.output_dir)

    # Save raw features for further analysis
    np.savez_compressed(
        os.path.join(args.output_dir, "features.npz"),
        z_a=np.array([r["z_a"] for r in records]),
        z_v=np.array([r["z_v"] for r in records]),
        subtypes=np.array([r["subtype"] for r in records]),
        labels=np.array([r["label"] for r in records]),
        ad_fractions=np.array([r["ad_fraction_per_region"] for r in records]),
    )
    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
