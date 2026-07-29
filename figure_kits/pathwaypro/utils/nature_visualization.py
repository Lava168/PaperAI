"""
Nature-Style Visualization Module

按照 Nature 期刊标准的高质量可视化:
- 清晰的配色方案
- 专业的字体设置
- 统一的图例风格
- 高分辨率输出
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

# ============================================================================
# Nature Style Configuration
# ============================================================================

# Nature 期刊推荐配色
NATURE_COLORS = {
    'primary': '#1f77b4',      # 蓝色
    'secondary': '#ff7f0e',    # 橙色
    'tertiary': '#2ca02c',     # 绿色
    'quaternary': '#d62728',   # 红色
    'quinary': '#9467bd',      # 紫色
    'senary': '#8c564b',       # 棕色
    'septenary': '#e377c2',    # 粉色
    'octonary': '#7f7f7f',     # 灰色
}

# Nature 期刊标准配色方案
NATURE_PALETTE = [
    '#4E79A7',  # Blue
    '#F28E2B',  # Orange
    '#E15759',  # Red
    '#76B7B2',  # Teal
    '#59A14F',  # Green
    '#EDC948',  # Yellow
    '#B07AA1',  # Purple
    '#FF9DA7',  # Pink
    '#9C755F',  # Brown
    '#BAB0AC',  # Gray
]

# 诊断组颜色
DIAGNOSIS_COLORS = {
    'CN': '#4E79A7',   # 蓝色 - 正常
    'MCI': '#F28E2B',  # 橙色 - 轻度认知障碍
    'AD': '#E15759',   # 红色 - 阿尔茨海默病
}

# 病理特征颜色
PATHOLOGY_COLORS = {
    'atrophy': '#9467bd',    # 紫色 - 萎缩/神经退行性
    'vascular': '#2ca02c',   # 绿色 - 血管病变
    'amyloid_pos': '#d62728', # 红色 - Amyloid 阳性
    'amyloid_neg': '#1f77b4', # 蓝色 - Amyloid 阴性
}


def set_nature_style():
    """设置 Nature 期刊风格"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.rcParams.update({
        # 字体设置
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        
        # 线条设置
        'lines.linewidth': 1.5,
        'axes.linewidth': 1.0,
        
        # 网格设置
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        
        # 图例设置
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        
        # 保存设置
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # 其他
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    创建 Nature 风格的图形
    
    Nature 期刊推荐:
    - 单栏: 宽度 89mm (3.5 inches)
    - 1.5栏: 宽度 120mm (4.7 inches)
    - 双栏: 宽度 183mm (7.2 inches)
    """
    if figsize is None:
        # 默认双栏宽度
        width = 7.2
        height = width * 0.618 * nrows / ncols  # 黄金比例
        figsize = (width, height)
    
    set_nature_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    
    return fig, axes


# ============================================================================
# Disentanglement Visualization Functions
# ============================================================================

def plot_disentanglement_scatter(
    z_atrophy: np.ndarray,
    z_vascular: np.ndarray,
    labels: np.ndarray,
    label_names: List[str] = ['CN', 'MCI', 'AD'],
    method: str = 'tsne',
    title: str = 'Disentangled Feature Space',
    figsize: Tuple[float, float] = (10, 4),
) -> plt.Figure:
    """
    绘制解耦特征的散点图
    
    左图: Z_atrophy 空间
    右图: Z_vascular 空间
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_atrophy)-1))
    else:
        reducer = PCA(n_components=2)
    
    z_a_2d = reducer.fit_transform(z_atrophy)
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_vascular)-1))
    z_v_2d = reducer.fit_transform(z_vascular)
    
    # Z_atrophy 空间
    for i, name in enumerate(label_names):
        mask = labels == i
        axes[0].scatter(
            z_a_2d[mask, 0], z_a_2d[mask, 1],
            c=DIAGNOSIS_COLORS.get(name, NATURE_PALETTE[i]),
            label=name,
            alpha=0.7,
            s=30,
            edgecolors='white',
            linewidth=0.5,
        )
    
    axes[0].set_xlabel(f'{method.upper()} 1')
    axes[0].set_ylabel(f'{method.upper()} 2')
    axes[0].set_title(r'$\mathbf{Z}_{atrophy}$ Space', fontweight='bold')
    axes[0].legend(loc='upper right', framealpha=0.9)
    
    # Z_vascular 空间
    for i, name in enumerate(label_names):
        mask = labels == i
        axes[1].scatter(
            z_v_2d[mask, 0], z_v_2d[mask, 1],
            c=DIAGNOSIS_COLORS.get(name, NATURE_PALETTE[i]),
            label=name,
            alpha=0.7,
            s=30,
            edgecolors='white',
            linewidth=0.5,
        )
    
    axes[1].set_xlabel(f'{method.upper()} 1')
    axes[1].set_ylabel(f'{method.upper()} 2')
    axes[1].set_title(r'$\mathbf{Z}_{vascular}$ Space', fontweight='bold')
    axes[1].legend(loc='upper right', framealpha=0.9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_correlation_matrix(
    z_atrophy: np.ndarray,
    z_vascular: np.ndarray,
    amyloid: np.ndarray,
    wmh: np.ndarray,
    figsize: Tuple[float, float] = (6, 5),
) -> plt.Figure:
    """
    绘制特征与病理指标的相关性矩阵
    
    验证解耦效果:
    - Z_atrophy 应该与 Amyloid 高相关，与 WMH 低相关
    - Z_vascular 应该与 WMH 高相关，与 Amyloid 低相关
    """
    from scipy.stats import spearmanr
    
    set_nature_style()
    
    # 使用 PCA 将特征压缩为标量
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=1)
    z_a_scalar = pca.fit_transform(z_atrophy).flatten()
    z_v_scalar = pca.fit_transform(z_vascular).flatten()
    
    # 计算相关性矩阵
    features = ['Z_atrophy', 'Z_vascular', 'Amyloid', 'WMH']
    data = np.column_stack([z_a_scalar, z_v_scalar, amyloid, wmh])
    
    n = len(features)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # 处理 NaN
            valid = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
            if valid.sum() > 3:
                corr, p = spearmanr(data[valid, i], data[valid, j])
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p
            else:
                corr_matrix[i, j] = np.nan
                p_matrix[i, j] = 1.0
    
    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    
    # 自定义颜色映射
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # 绘制热力图
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Spearman ρ', fontsize=11)
    
    # 设置刻度
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_yticklabels(features)
    
    # 添加相关系数文本
    for i in range(n):
        for j in range(n):
            if not np.isnan(corr_matrix[i, j]):
                text = f'{corr_matrix[i, j]:.2f}'
                # 显著性标记
                if p_matrix[i, j] < 0.001:
                    text += '***'
                elif p_matrix[i, j] < 0.01:
                    text += '**'
                elif p_matrix[i, j] < 0.05:
                    text += '*'
                
                color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
    
    ax.set_title('Feature-Pathology Correlation Matrix', fontweight='bold', pad=10)
    
    # 添加分隔线
    ax.axhline(y=1.5, color='black', linewidth=2)
    ax.axvline(x=1.5, color='black', linewidth=2)
    
    plt.tight_layout()
    
    return fig


def plot_disentanglement_validation(
    validation_results: Dict[str, float],
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    绘制解耦验证结果（条形图）
    
    展示:
    - 目标预测准确率（应该高）
    - 交叉预测准确率（应该接近随机0.5）
    """
    set_nature_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 准备数据
    target_metrics = {
        r'$Z_{atrophy} \rightarrow$ Amyloid': validation_results.get('z_atrophy_to_amyloid_acc', 0.5),
        r'$Z_{vascular} \rightarrow$ WMH': validation_results.get('z_vascular_to_wmh_acc', 0.5),
    }
    
    cross_metrics = {
        r'$Z_{atrophy} \rightarrow$ WMH': validation_results.get('z_atrophy_to_wmh_acc', 0.5),
        r'$Z_{vascular} \rightarrow$ Amyloid': validation_results.get('z_vascular_to_amyloid_acc', 0.5),
    }
    
    # 合并
    labels = list(target_metrics.keys()) + list(cross_metrics.keys())
    values = list(target_metrics.values()) + list(cross_metrics.values())
    colors = [PATHOLOGY_COLORS['atrophy'], PATHOLOGY_COLORS['vascular'],
              PATHOLOGY_COLORS['atrophy'], PATHOLOGY_COLORS['vascular']]
    
    # 条形图
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 添加随机基线
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Random baseline')
    
    # 分隔目标和交叉
    ax.axvline(x=1.5, color='black', linestyle='-', linewidth=2)
    
    # 标注
    ax.text(0.5, 1.05, 'Target Prediction\n(should be high)', ha='center', 
            transform=ax.get_xaxis_transform(), fontsize=10, fontweight='bold')
    ax.text(2.5, 1.05, 'Cross Prediction\n(should be ~0.5)', ha='center', 
            transform=ax.get_xaxis_transform(), fontsize=10, fontweight='bold')
    
    # 在条形上添加数值
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.0)
    ax.set_title('Disentanglement Validation: Linear Probing', fontweight='bold')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    return fig


def plot_pathology_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    pathology_type: str = 'amyloid',
    figsize: Tuple[float, float] = (10, 4),
) -> plt.Figure:
    """
    绘制病理预测结果
    
    包括:
    - 混淆矩阵
    - ROC 曲线
    - 预测分布
    """
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    
    set_nature_style()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if pathology_type == 'amyloid':
        labels = ['Aβ-', 'Aβ+']
        cmap = 'Reds'
        color = PATHOLOGY_COLORS['amyloid_pos']
    else:
        labels = ['Low WMH', 'High WMH']
        cmap = 'Greens'
        color = PATHOLOGY_COLORS['vascular']
    
    im = axes[0].imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    
    for i in range(2):
        for j in range(2):
            text = f'{cm[i, j]}\n({cm_norm[i, j]:.1%})'
            color_text = 'white' if cm_norm[i, j] > 0.5 else 'black'
            axes[0].text(j, i, text, ha='center', va='center', color=color_text, fontsize=10)
    
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(labels)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix')
    
    # 2. ROC 曲线
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        axes[1].plot(fpr, tpr, color=color, linewidth=2, label=f'AUC = {roc_auc:.3f}')
        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        axes[1].fill_between(fpr, tpr, alpha=0.2, color=color)
        
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc='lower right')
    
    # 3. 预测分布
    if y_prob is not None:
        for label, name in enumerate(labels):
            mask = y_true == label
            axes[2].hist(y_prob[mask], bins=20, alpha=0.6, label=name,
                        color=NATURE_PALETTE[label], edgecolor='white')
        
        axes[2].set_xlabel('Predicted Probability')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Prediction Distribution')
        axes[2].legend()
    
    fig.suptitle(f'{pathology_type.capitalize()} Prediction Results', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_mutual_information_trajectory(
    mi_values: List[float],
    epochs: Optional[List[int]] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> plt.Figure:
    """
    绘制互信息随训练的变化
    """
    set_nature_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if epochs is None:
        epochs = list(range(1, len(mi_values) + 1))
    
    ax.plot(epochs, mi_values, 'o-', color=NATURE_PALETTE[0], 
            linewidth=2, markersize=4, label='MI Estimate')
    
    # 添加趋势线
    z = np.polyfit(epochs, mi_values, 2)
    p = np.poly1d(z)
    ax.plot(epochs, p(epochs), '--', color=NATURE_PALETTE[1], 
            linewidth=1.5, alpha=0.7, label='Trend')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$I(\mathbf{Z}_{atrophy}; \mathbf{Z}_{vascular})$')
    ax.set_title('Mutual Information During Training', fontweight='bold')
    ax.legend()
    
    # 添加目标区域
    ax.axhspan(0, 0.1, alpha=0.1, color='green', label='Target range')
    
    plt.tight_layout()
    
    return fig


def plot_comprehensive_disentanglement_figure(
    z_atrophy: np.ndarray,
    z_vascular: np.ndarray,
    amyloid_true: np.ndarray,
    amyloid_pred: np.ndarray,
    amyloid_prob: np.ndarray,
    wmh_true: np.ndarray,
    wmh_pred: np.ndarray,
    diagnosis_labels: np.ndarray,
    mi_history: List[float],
    validation_results: Dict[str, float],
    figsize: Tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    综合解耦结果图 (Nature 风格主图)
    
    包含所有关键结果的单一图形
    """
    from sklearn.manifold import TSNE
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    
    set_nature_style()
    
    fig = plt.figure(figsize=figsize)
    
    # 创建复杂布局
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # ========== Row 1: Feature Space ==========
    # (a) Z_atrophy t-SNE
    ax_a = fig.add_subplot(gs[0, 0])
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_atrophy)-1))
    z_a_2d = tsne.fit_transform(z_atrophy)
    
    for i, name in enumerate(['CN', 'MCI', 'AD']):
        mask = diagnosis_labels == i
        ax_a.scatter(z_a_2d[mask, 0], z_a_2d[mask, 1],
                    c=DIAGNOSIS_COLORS[name], label=name, alpha=0.6, s=20,
                    edgecolors='white', linewidth=0.3)
    
    ax_a.set_xlabel('t-SNE 1')
    ax_a.set_ylabel('t-SNE 2')
    ax_a.set_title(r'$\mathbf{a}$  $\mathbf{Z}_{atrophy}$ Space', loc='left', fontweight='bold')
    ax_a.legend(loc='upper right', fontsize=7, framealpha=0.9)
    
    # (b) Z_vascular t-SNE
    ax_b = fig.add_subplot(gs[0, 1])
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_vascular)-1))
    z_v_2d = tsne.fit_transform(z_vascular)
    
    for i, name in enumerate(['CN', 'MCI', 'AD']):
        mask = diagnosis_labels == i
        ax_b.scatter(z_v_2d[mask, 0], z_v_2d[mask, 1],
                    c=DIAGNOSIS_COLORS[name], label=name, alpha=0.6, s=20,
                    edgecolors='white', linewidth=0.3)
    
    ax_b.set_xlabel('t-SNE 1')
    ax_b.set_ylabel('t-SNE 2')
    ax_b.set_title(r'$\mathbf{b}$  $\mathbf{Z}_{vascular}$ Space', loc='left', fontweight='bold')
    ax_b.legend(loc='upper right', fontsize=7, framealpha=0.9)
    
    # (c) Correlation Matrix
    ax_c = fig.add_subplot(gs[0, 2:])
    
    pca = PCA(n_components=1)
    z_a_scalar = pca.fit_transform(z_atrophy).flatten()
    z_v_scalar = pca.fit_transform(z_vascular).flatten()
    
    features = [r'$Z_{atrophy}$', r'$Z_{vascular}$', 'Amyloid', 'WMH']
    data = np.column_stack([z_a_scalar, z_v_scalar, amyloid_true, wmh_true])
    
    n = len(features)
    corr_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            valid = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
            if valid.sum() > 3:
                corr, _ = spearmanr(data[valid, i], data[valid, j])
                corr_matrix[i, j] = corr
    
    im = ax_c.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax_c, shrink=0.8, label='Spearman ρ')
    
    ax_c.set_xticks(np.arange(n))
    ax_c.set_yticks(np.arange(n))
    ax_c.set_xticklabels(features, fontsize=9)
    ax_c.set_yticklabels(features, fontsize=9)
    
    for i in range(n):
        for j in range(n):
            color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            ax_c.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                     color=color, fontsize=9)
    
    ax_c.set_title(r'$\mathbf{c}$  Feature-Pathology Correlation', loc='left', fontweight='bold')
    
    # ========== Row 2: Prediction Results ==========
    # (d) Amyloid ROC
    ax_d = fig.add_subplot(gs[1, 0])
    
    valid_amy = ~np.isnan(amyloid_true)
    if valid_amy.sum() > 0:
        fpr, tpr, _ = roc_curve(amyloid_true[valid_amy], amyloid_prob[valid_amy])
        roc_auc = auc(fpr, tpr)
        
        ax_d.plot(fpr, tpr, color=PATHOLOGY_COLORS['amyloid_pos'], linewidth=2)
        ax_d.fill_between(fpr, tpr, alpha=0.2, color=PATHOLOGY_COLORS['amyloid_pos'])
        ax_d.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax_d.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', fontsize=10, fontweight='bold')
    
    ax_d.set_xlabel('False Positive Rate')
    ax_d.set_ylabel('True Positive Rate')
    ax_d.set_title(r'$\mathbf{d}$  Amyloid Prediction ROC', loc='left', fontweight='bold')
    
    # (e) WMH Prediction
    ax_e = fig.add_subplot(gs[1, 1])
    
    valid_wmh = ~np.isnan(wmh_true)
    if valid_wmh.sum() > 0:
        ax_e.scatter(wmh_true[valid_wmh], wmh_pred[valid_wmh], 
                    alpha=0.5, c=PATHOLOGY_COLORS['vascular'], s=20,
                    edgecolors='white', linewidth=0.3)
        
        # 添加回归线
        z = np.polyfit(wmh_true[valid_wmh], wmh_pred[valid_wmh], 1)
        p = np.poly1d(z)
        x_line = np.linspace(wmh_true[valid_wmh].min(), wmh_true[valid_wmh].max(), 100)
        ax_e.plot(x_line, p(x_line), '--', color='black', linewidth=1.5)
        
        # R²
        corr, _ = spearmanr(wmh_true[valid_wmh], wmh_pred[valid_wmh])
        ax_e.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax_e.transAxes,
                 fontsize=10, fontweight='bold', va='top')
    
    ax_e.set_xlabel('True WMH Volume (log)')
    ax_e.set_ylabel('Predicted WMH Volume')
    ax_e.set_title(r'$\mathbf{e}$  WMH Prediction', loc='left', fontweight='bold')
    
    # (f) Disentanglement Validation
    ax_f = fig.add_subplot(gs[1, 2:])
    
    metrics = [
        (r'$Z_{atr} \rightarrow$ Aβ', validation_results.get('z_atrophy_to_amyloid_acc', 0.5), PATHOLOGY_COLORS['atrophy']),
        (r'$Z_{vas} \rightarrow$ WMH', validation_results.get('z_vascular_to_wmh_acc', 0.5), PATHOLOGY_COLORS['vascular']),
        (r'$Z_{atr} \rightarrow$ WMH', validation_results.get('z_atrophy_to_wmh_acc', 0.5), PATHOLOGY_COLORS['atrophy']),
        (r'$Z_{vas} \rightarrow$ Aβ', validation_results.get('z_vascular_to_amyloid_acc', 0.5), PATHOLOGY_COLORS['vascular']),
    ]
    
    x = np.arange(len(metrics))
    bars = ax_f.bar(x, [m[1] for m in metrics], color=[m[2] for m in metrics], 
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax_f.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5)
    ax_f.axvline(x=1.5, color='black', linestyle='-', linewidth=2)
    
    ax_f.set_xticks(x)
    ax_f.set_xticklabels([m[0] for m in metrics], fontsize=9)
    ax_f.set_ylabel('Accuracy')
    ax_f.set_ylim(0, 1)
    ax_f.set_title(r'$\mathbf{f}$  Disentanglement Validation', loc='left', fontweight='bold')
    
    # 标注
    ax_f.text(0.5, 1.02, 'Target', ha='center', transform=ax_f.get_xaxis_transform(), 
             fontsize=9, fontweight='bold')
    ax_f.text(2.5, 1.02, 'Cross', ha='center', transform=ax_f.get_xaxis_transform(),
             fontsize=9, fontweight='bold')
    
    # ========== Row 3: Training Dynamics ==========
    # (g) MI Trajectory
    ax_g = fig.add_subplot(gs[2, :2])
    
    epochs = list(range(1, len(mi_history) + 1))
    ax_g.plot(epochs, mi_history, 'o-', color=NATURE_PALETTE[0], 
             linewidth=2, markersize=4)
    
    ax_g.set_xlabel('Epoch')
    ax_g.set_ylabel(r'$I(\mathbf{Z}_{atrophy}; \mathbf{Z}_{vascular})$')
    ax_g.set_title(r'$\mathbf{g}$  Mutual Information During Training', loc='left', fontweight='bold')
    ax_g.axhspan(-0.5, 0.2, alpha=0.1, color='green')
    
    # (h) Summary Statistics
    ax_h = fig.add_subplot(gs[2, 2:])
    ax_h.axis('off')
    
    # 创建统计表格
    dis_score = validation_results.get('disentanglement_score', 0)
    stats_text = f"""
    Summary Statistics
    ─────────────────────────────
    Disentanglement Score: {dis_score:.3f}
    
    Target Predictions:
      • Amyloid AUC: {f'{roc_auc:.3f}' if 'roc_auc' in dir() and roc_auc is not None else 'N/A'}
      • WMH Correlation: {f'{corr:.3f}' if 'corr' in dir() and corr is not None else 'N/A'}
    
    Cross Independence:
      • Z_atr → WMH: {validation_results.get('z_atrophy_to_wmh_acc', 0.5):.3f}
      • Z_vas → Aβ: {validation_results.get('z_vascular_to_amyloid_acc', 0.5):.3f}
    
    Final MI: {f'{mi_history[-1]:.4f}' if mi_history else 'N/A'}
    """
    
    ax_h.text(0.1, 0.9, stats_text, transform=ax_h.transAxes,
             fontsize=10, family='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_h.set_title(r'$\mathbf{h}$  Summary', loc='left', fontweight='bold')
    
    plt.suptitle('Pathology Feature Disentanglement Results', 
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    formats: List[str] = ['png', 'pdf', 'svg'],
    dpi: int = 300,
):
    """
    以多种格式保存图形 (Nature 投稿要求)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        save_path = path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
