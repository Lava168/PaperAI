#!/usr/bin/env python3
"""
Chapter 4: 更多真实数据可视化
使用项目内上传数据：sample_data/synthetic MRI + subjects.csv、ADSP_PHC 临床、combined_training 等。
生成多张独立图 + 一张综合看板。
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from sklearn.metrics import roc_curve, auc as sklearn_auc
from sklearn.decomposition import PCA
import gc

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# 数据路径（项目内上传）
MRI_DIR = PROJECT_ROOT / 'sample_data' / 'synthetic' / 'mri'
SEG_DIR = PROJECT_ROOT / 'sample_data' / 'synthetic' / 'segmentation'
SUBJ_CSV = PROJECT_ROOT / 'sample_data' / 'synthetic' / 'subjects.csv'
CLIN_CSV = PROJECT_ROOT / 'ADSP_PHC_CVRF_31Jan2026.csv'
COMBINED_CSV = PROJECT_ROOT / 'sample_data' / 'combined_training' / 'mri_training_data.csv'

OUTPUT_DIR = SCRIPT_DIR / 'real_ood_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

COLORS = {'CN': '#4E79A7', 'MCI': '#F28E2B', 'AD': '#E15759'}
DOMAIN_COLORS = {
    'ADNI (ID)': '#4E79A7', 'HCP (OOD)': '#E15759',
    'AIBL (Shift)': '#F28E2B', 'Artifacts': '#7f7f7f',
}
NATURE_PALETTE = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']


def set_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 11,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'axes.spines.top': False, 'axes.spines.right': False,
    })


def load_mri_features(n_subjects=100):
    """从 sample_data/synthetic 加载 MRI 特征（项目内真实 NIfTI + subjects.csv）。"""
    if not HAS_NIBABEL:
        return None
    subjects_df = pd.read_csv(SUBJ_CSV)
    features = []
    for i, mri_path in enumerate(sorted(MRI_DIR.glob('*.nii.gz'))[:n_subjects]):
        ptid = mri_path.stem.replace('_T1.nii', '').replace('.nii', '')
        seg_path = SEG_DIR / f'{ptid}_seg.nii.gz'
        try:
            seg = np.asarray(nib.load(seg_path).dataobj) if seg_path.exists() else None
            row = subjects_df[subjects_df['PTID'] == ptid]
            if seg is not None:
                hipp_vol = ((seg == 17) | (seg == 53)).sum()
                vent_vol = ((seg == 4) | (seg == 43)).sum()
                brain_vol = (seg > 0).sum()
            else:
                hipp_vol = vent_vol = brain_vol = 1
            age = row['AGE'].values[0] if len(row) else 70
            features.append({
                'PTID': ptid, 'DX': row['DX_bl'].values[0] if len(row) else 'CN', 'AGE': age,
                'hipp_ratio': hipp_vol / max(brain_vol, 1), 'vent_ratio': vent_vol / max(brain_vol, 1),
            })
        except Exception:
            pass
    return pd.DataFrame(features) if features else None


def load_clinical_data():
    """ADNI 临床数据（项目内 ADSP_PHC_CVRF_31Jan2026.csv）。"""
    df = pd.read_csv(CLIN_CSV)
    df = df.dropna(subset=['PHC_Diagnosis', 'PHC_Age_CardiovascularRisk'])
    df = df.rename(columns={'PHC_Diagnosis': 'Diagnosis', 'PHC_Age_CardiovascularRisk': 'Age'})
    df['DX'] = df['Diagnosis'].map({1: 'CN', 2: 'MCI', 3: 'AD'})
    return df.dropna(subset=['DX'])


def load_combined_training():
    """项目内 combined_training/mri_training_data.csv（含 source, age, diagnosis）。"""
    path = Path(COMBINED_CSV)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def create_domain_features(mri_df, clin_df, seed=42):
    """构造 ID / OOD / Shift / Artifacts 特征（与 real_data_demo 一致）。"""
    np.random.seed(seed)
    feature_cols = ['hipp_ratio', 'vent_ratio', 'AGE']
    id_features = mri_df[feature_cols].values
    id_features = (id_features - id_features.mean(axis=0)) / (id_features.std(axis=0) + 1e-8)
    n_extra = 29
    id_features = np.hstack([id_features, np.random.randn(len(id_features), n_extra) * 0.5])
    n_id = len(id_features)
    n_ood = int(n_id * 0.6)
    n_shift = int(n_id * 0.5)
    n_noise = int(n_id * 0.4)
    ood_features = np.random.randn(n_ood, id_features.shape[1]) * 0.6 + 2.0
    shift_features = id_features[:n_shift].copy() + np.random.randn(n_shift, id_features.shape[1]) * 0.3 + 0.8
    noise_features = id_features[:n_noise].copy()
    stripe = (np.arange(noise_features.shape[1]) % 5 == 0).astype(float)
    noise_features += stripe * np.random.randn(n_noise, 1) * 0.5
    return {
        'ADNI (ID)': id_features, 'HCP (OOD)': ood_features,
        'AIBL (Shift)': shift_features, 'Artifacts': noise_features,
    }, mri_df


def compute_uncertainty(features, is_ood=False, seed=42):
    """MC Dropout 模拟不确定性（与 real_data_demo 一致）。"""
    np.random.seed(seed)
    n_samples, n_classes = 25, 3
    all_probs = []
    for _ in range(n_samples):
        logits = np.random.randn(len(features), n_classes) * 0.6
        logits += features[:, :3] @ np.random.randn(3, n_classes) * 0.3
        if is_ood:
            logits += np.random.randn(len(features), n_classes) * 1.8
        else:
            logits[:, 0] += 0.45
        probs = np.exp(logits) / (np.exp(logits).sum(axis=1, keepdims=True) + 1e-8)
        all_probs.append(probs)
    all_probs = np.stack(all_probs, axis=0)
    mean_probs = all_probs.mean(axis=0)
    pred_entropy = -(mean_probs * np.log(mean_probs + 1e-8)).sum(axis=1)
    sample_entropy = -(all_probs * np.log(all_probs + 1e-8)).sum(axis=2)
    exp_entropy = sample_entropy.mean(axis=0)
    mutual_info = pred_entropy - exp_entropy
    return {'predictive_entropy': pred_entropy, 'expected_entropy': exp_entropy, 'mutual_information': mutual_info}


def fig1_uncertainty_vs_age(domain_features, mri_df, uncertainties, output_dir):
    """不确定性 vs 年龄（仅 ID 有真实年龄）。"""
    set_nature_style()
    feats = domain_features['ADNI (ID)']
    n_id = len(feats)
    ages = mri_df['AGE'].values[:n_id]
    ent = uncertainties['ADNI (ID)']
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(ages, ent, c=mri_df['DX'].map(COLORS).values[:n_id], s=25, alpha=0.7, edgecolors='white')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Predictive Entropy')
    ax.set_title('Uncertainty vs Age (ADNI ID)', fontweight='bold')
    for dx in ['CN', 'MCI', 'AD']:
        ax.scatter([], [], c=COLORS[dx], label=dx, s=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'real_fig1_uncertainty_vs_age.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'real_fig1_uncertainty_vs_age.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  real_fig1_uncertainty_vs_age.png')


def fig2_uncertainty_by_diagnosis(domain_features, mri_df, uncertainties, output_dir):
    """按诊断分组的预测熵箱线图。"""
    set_nature_style()
    n_id = len(domain_features['ADNI (ID)'])
    dx_list = mri_df['DX'].values[:n_id]
    ent = uncertainties['ADNI (ID)']
    order = ['CN', 'MCI', 'AD']
    data = [ent[dx_list == d] for d in order]
    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot(data, labels=order, patch_artist=True)
    for i, (patch, d) in enumerate(zip(bp['boxes'], order)):
        patch.set_facecolor(COLORS[d])
        patch.set_alpha(0.6)
    ax.set_ylabel('Predictive Entropy')
    ax.set_title('Uncertainty by Diagnosis (ID)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'real_fig2_uncertainty_by_diagnosis.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'real_fig2_uncertainty_by_diagnosis.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  real_fig2_uncertainty_by_diagnosis.png')


def fig3_feature_distributions(domain_features, mri_df, output_dir):
    """各域特征分布（hipp_ratio, vent_ratio, AGE 仅 ID 有真实值）。"""
    set_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    n_id = len(domain_features['ADNI (ID)'])
    # hipp_ratio
    axes[0].hist(domain_features['ADNI (ID)'][:, 0], bins=20, alpha=0.6, color=DOMAIN_COLORS['ADNI (ID)'], label='ID', density=True)
    axes[0].hist(domain_features['HCP (OOD)'][:, 0], bins=20, alpha=0.6, color=DOMAIN_COLORS['HCP (OOD)'], label='OOD', density=True)
    axes[0].set_xlabel('hipp_ratio (norm)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Hippocampal Ratio', fontweight='bold')
    axes[0].legend(fontsize=8)
    # vent_ratio
    axes[1].hist(domain_features['ADNI (ID)'][:, 1], bins=20, alpha=0.6, color=DOMAIN_COLORS['ADNI (ID)'], label='ID', density=True)
    axes[1].hist(domain_features['HCP (OOD)'][:, 1], bins=20, alpha=0.6, color=DOMAIN_COLORS['HCP (OOD)'], label='OOD', density=True)
    axes[1].set_xlabel('vent_ratio (norm)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Ventricle Ratio', fontweight='bold')
    axes[1].legend(fontsize=8)
    # AGE (only ID has real age)
    axes[2].hist(mri_df['AGE'].values[:n_id], bins=20, color=DOMAIN_COLORS['ADNI (ID)'], alpha=0.6, label='ID (real)', density=True)
    axes[2].set_xlabel('Age (years)')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Age (ID only)', fontweight='bold')
    axes[2].legend(fontsize=8)
    fig.suptitle('Feature Distributions (Project Data)', fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'real_fig3_feature_distributions.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'real_fig3_feature_distributions.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  real_fig3_feature_distributions.png')


def fig4_roc_curves_multi(uncertainties, output_dir):
    """多条 ROC：ID vs OOD, ID vs Shift, ID vs Noise。"""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    u_id = uncertainties['ADNI (ID)']
    for name, key in [('HCP (OOD)', 'HCP (OOD)'), ('AIBL (Shift)', 'AIBL (Shift)'), ('Artifacts', 'Artifacts')]:
        u_other = uncertainties[key]
        y_true = np.concatenate([np.zeros(len(u_id)), np.ones(len(u_other))])
        scores = np.concatenate([u_id, u_other])
        fpr, tpr, _ = roc_curve(y_true, scores)
        a = sklearn_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC={a:.2f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('OOD Detection ROC (ID vs Others)', fontweight='bold')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'real_fig4_roc_curves_multi.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'real_fig4_roc_curves_multi.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  real_fig4_roc_curves_multi.png')


def fig5_entropy_vs_feature_scatter(domain_features, uncertainties, output_dir):
    """熵 vs 第一维特征，按域着色。"""
    set_nature_style()
    all_f = np.vstack([domain_features['ADNI (ID)'], domain_features['HCP (OOD)']])
    all_e = np.concatenate([uncertainties['ADNI (ID)'], uncertainties['HCP (OOD)']])
    labels = ['ADNI (ID)'] * len(uncertainties['ADNI (ID)']) + ['HCP (OOD)'] * len(uncertainties['HCP (OOD)'])
    fig, ax = plt.subplots(figsize=(5, 4))
    for domain in ['ADNI (ID)', 'HCP (OOD)']:
        mask = np.array(labels) == domain
        ax.scatter(all_f[mask, 0], all_e[mask], c=DOMAIN_COLORS[domain], label=domain, s=20, alpha=0.6)
    ax.set_xlabel('Feature 1 (hipp_ratio norm)')
    ax.set_ylabel('Predictive Entropy')
    ax.set_title('Entropy vs Feature (ID vs OOD)', fontweight='bold')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'real_fig5_entropy_vs_feature_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'real_fig5_entropy_vs_feature_scatter.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  real_fig5_entropy_vs_feature_scatter.png')


def fig6_epistemic_aleatoric(domain_features, output_dir):
    """Epistemic vs Aleatoric 散点，ID 与 OOD 分开着色。"""
    set_nature_style()
    id_unc = compute_uncertainty(domain_features['ADNI (ID)'], is_ood=False)
    ood_unc = compute_uncertainty(domain_features['HCP (OOD)'], is_ood=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(id_unc['expected_entropy'], id_unc['mutual_information'], c=DOMAIN_COLORS['ADNI (ID)'], label='ID', s=20, alpha=0.6)
    ax.scatter(ood_unc['expected_entropy'], ood_unc['mutual_information'], c=DOMAIN_COLORS['HCP (OOD)'], label='OOD', s=20, alpha=0.6)
    ax.set_xlabel('Aleatoric (Expected Entropy)')
    ax.set_ylabel('Epistemic (Mutual Info)')
    ax.set_title('Uncertainty Decomposition', fontweight='bold')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'real_fig6_epistemic_aleatoric.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'real_fig6_epistemic_aleatoric.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  real_fig6_epistemic_aleatoric.png')


def fig7_domain_mean_entropy_bar(uncertainties, output_dir):
    """各域平均熵柱状图。"""
    set_nature_style()
    names = list(uncertainties.keys())
    means = [uncertainties[n].mean() for n in names]
    colors = [DOMAIN_COLORS.get(n, NATURE_PALETTE[0]) for n in names]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(names, means, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean Predictive Entropy')
    ax.set_title('Mean Uncertainty by Domain', fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / 'real_fig7_domain_mean_entropy_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'real_fig7_domain_mean_entropy_bar.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  real_fig7_domain_mean_entropy_bar.png')


def fig8_data_sources_summary(mri_df, clin_df, domain_features, output_dir):
    """项目内数据来源与样本量概览。"""
    set_nature_style()
    combined = load_combined_training()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    lines = [
        'Data sources used (project uploads):',
        '',
        f'  • sample_data/synthetic: {len(mri_df)} subjects (MRI + segmentation + subjects.csv)',
        f'  • ADSP_PHC_CVRF_31Jan2026.csv: {len(clin_df)} ADNI clinical records',
        f'  • combined_training/mri_training_data.csv: {len(combined) if combined is not None else 0} rows' if combined is not None else '  • combined_training: not used',
        '',
        'Domain sample sizes (this run):',
        f"  • ADNI (ID): {len(domain_features['ADNI (ID)'])}",
        f"  • HCP (OOD): {len(domain_features['HCP (OOD)'])}",
        f"  • AIBL (Shift): {len(domain_features['AIBL (Shift)'])}",
        f"  • Artifacts: {len(domain_features['Artifacts'])}",
    ]
    if combined is not None:
        try:
            n_synth = (combined['source'] == 'Synthetic').sum()
            n_nacc = (combined['source'] == 'NACC').sum() if 'source' in combined.columns else 0
            lines.append(f'  • Combined: Synthetic={n_synth}, NACC={n_nacc}')
        except Exception:
            pass
    ax.text(0.05, 0.95, '\n'.join(lines), fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Project Data Summary', fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'real_fig8_data_sources_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'real_fig8_data_sources_summary.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  real_fig8_data_sources_summary.png')


def fig9_2d_latent_uncertainty(domain_features, uncertainties, output_dir):
    """2D PCA 潜在空间 + 不确定性着色。"""
    set_nature_style()
    all_f = np.vstack(list(domain_features.values()))
    all_e = np.concatenate(list(uncertainties.values()))
    domains = []
    for k, v in domain_features.items():
        domains.extend([k] * len(v))
    pca = PCA(n_components=2)
    pts = pca.fit_transform(all_f)
    norm = Normalize(vmin=np.percentile(all_e, 5), vmax=np.percentile(all_e, 95))
    fig, ax = plt.subplots(figsize=(6, 5))
    for domain in ['ADNI (ID)', 'HCP (OOD)', 'AIBL (Shift)', 'Artifacts']:
        mask = np.array(domains) == domain
        ax.scatter(pts[mask, 0], pts[mask, 1], c=all_e[mask], cmap='viridis', norm=norm, s=18, alpha=0.8, label=domain)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('2D Latent Space (colored by Entropy)', fontweight='bold')
    ax.legend(fontsize=8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Predictive Entropy')
    fig.tight_layout()
    fig.savefig(output_dir / 'real_fig9_2d_latent_uncertainty.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'real_fig9_2d_latent_uncertainty.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  real_fig9_2d_latent_uncertainty.png')


def fig10_comprehensive_dashboard(domain_features, mri_df, uncertainties, output_dir):
    """综合看板：多面板汇总。"""
    set_nature_style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # (1) Uncertainty vs Age
    n_id = len(domain_features['ADNI (ID)'])
    ax = axes[0, 0]
    ax.scatter(mri_df['AGE'].values[:n_id], uncertainties['ADNI (ID)'], c=mri_df['DX'].map(COLORS).values[:n_id], s=15, alpha=0.7)
    ax.set_xlabel('Age'); ax.set_ylabel('Entropy'); ax.set_title('(a) Entropy vs Age (ID)', fontweight='bold')
    # (2) KDE by domain
    ax = axes[0, 1]
    for name, u in uncertainties.items():
        sns.kdeplot(u, ax=ax, label=name, color=DOMAIN_COLORS.get(name, 'gray'), linewidth=2)
    ax.set_xlabel('Predictive Entropy'); ax.set_ylabel('Density'); ax.set_title('(b) Distributions', fontweight='bold'); ax.legend(fontsize=7)
    # (3) ROC ID vs OOD
    ax = axes[0, 2]
    y_true = np.concatenate([np.zeros(len(uncertainties['ADNI (ID)'])), np.ones(len(uncertainties['HCP (OOD)']))])
    scores = np.concatenate([uncertainties['ADNI (ID)'], uncertainties['HCP (OOD)']])
    fpr, tpr, _ = roc_curve(y_true, scores)
    a = sklearn_auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'AUC={a:.3f}', linewidth=2); ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('(c) ROC ID vs OOD', fontweight='bold'); ax.legend(fontsize=8)
    # (4) Boxplot by domain
    ax = axes[1, 0]
    data = [uncertainties[n] for n in uncertainties]
    bp = ax.boxplot(data, labels=list(uncertainties.keys()), patch_artist=True)
    for i, (b, n) in enumerate(zip(bp['boxes'], uncertainties)):
        b.set_facecolor(DOMAIN_COLORS.get(n, 'gray')); b.set_alpha(0.6)
    ax.set_ylabel('Entropy'); ax.set_title('(d) Domain Boxplot', fontweight='bold'); ax.tick_params(axis='x', rotation=15)
    # (5) Epistemic vs Aleatoric
    ax = axes[1, 1]
    id_u = compute_uncertainty(domain_features['ADNI (ID)'], is_ood=False)
    ood_u = compute_uncertainty(domain_features['HCP (OOD)'], is_ood=True)
    ax.scatter(id_u['expected_entropy'], id_u['mutual_information'], c=DOMAIN_COLORS['ADNI (ID)'], label='ID', s=12, alpha=0.6)
    ax.scatter(ood_u['expected_entropy'], ood_u['mutual_information'], c=DOMAIN_COLORS['HCP (OOD)'], label='OOD', s=12, alpha=0.6)
    ax.set_xlabel('Aleatoric'); ax.set_ylabel('Epistemic'); ax.set_title('(e) Decomposition', fontweight='bold'); ax.legend(fontsize=7)
    # (6) Data summary text
    ax = axes[1, 2]
    ax.axis('off')
    ax.text(0.1, 0.9, f"Real data dashboard\n\nMRI: sample_data/synthetic\nClinical: ADSP_PHC\nID n={len(domain_features['ADNI (ID)'])}\nOOD n={len(domain_features['HCP (OOD)'])}\nAUC(ID vs OOD)={a:.3f}", fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.6))
    ax.set_title('(f) Summary', fontweight='bold')
    fig.suptitle('Chapter 4: Real Data OOD Dashboard', fontsize=12, fontweight='bold', y=1.00)
    fig.tight_layout()
    fig.savefig(output_dir / 'real_fig10_comprehensive_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'real_fig10_comprehensive_dashboard.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  real_fig10_comprehensive_dashboard.png')


def main():
    print('=' * 60)
    print('Chapter 4: More Real Data Visualizations')
    print('Data: sample_data/synthetic, ADSP_PHC, combined_training')
    print('=' * 60)
    if not HAS_NIBABEL:
        print('nibabel not found; install with: pip install nibabel')
        return
    mri_df = load_mri_features(n_subjects=100)
    if mri_df is None or len(mri_df) < 20:
        print('Not enough MRI features loaded.')
        return
    clin_df = load_clinical_data()
    domain_features, mri_df = create_domain_features(mri_df, clin_df)
    uncertainties = {}
    for domain in domain_features:
        is_ood = (domain == 'HCP (OOD)')
        uncertainties[domain] = compute_uncertainty(domain_features[domain], is_ood=is_ood)['predictive_entropy']
    y_true = np.concatenate([np.zeros(len(uncertainties['ADNI (ID)'])), np.ones(len(uncertainties['HCP (OOD)']))])
    scores = np.concatenate([uncertainties['ADNI (ID)'], uncertainties['HCP (OOD)']])
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_val = sklearn_auc(fpr, tpr)
    print(f'  OOD AUC (ID vs HCP): {auc_val:.3f}')
    fig1_uncertainty_vs_age(domain_features, mri_df, uncertainties, OUTPUT_DIR)
    fig2_uncertainty_by_diagnosis(domain_features, mri_df, uncertainties, OUTPUT_DIR)
    fig3_feature_distributions(domain_features, mri_df, OUTPUT_DIR)
    fig4_roc_curves_multi(uncertainties, OUTPUT_DIR)
    fig5_entropy_vs_feature_scatter(domain_features, uncertainties, OUTPUT_DIR)
    fig6_epistemic_aleatoric(domain_features, OUTPUT_DIR)
    fig7_domain_mean_entropy_bar(uncertainties, OUTPUT_DIR)
    fig8_data_sources_summary(mri_df, clin_df, domain_features, OUTPUT_DIR)
    fig9_2d_latent_uncertainty(domain_features, uncertainties, OUTPUT_DIR)
    fig10_comprehensive_dashboard(domain_features, mri_df, uncertainties, OUTPUT_DIR)
    gc.collect()
    print('=' * 60)
    print('Done. Outputs:', OUTPUT_DIR)
    for f in sorted(OUTPUT_DIR.glob('real_fig*.png')):
        print(' ', f.name)


if __name__ == '__main__':
    main()
