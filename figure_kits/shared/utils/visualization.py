"""
Comprehensive Visualization Utilities for Atlas-Guided Attention

包含以下可视化功能：
1. 脑部 MRI 可视化（多平面、3D）
2. 注意力图可视化
3. 测地距离可视化
4. 训练过程可视化
5. 模型解释性可视化
6. 区域分析可视化
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, Circle
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings


# ============================================================
# 颜色配置
# ============================================================

# FreeSurfer 风格的脑区颜色
FREESURFER_COLORS = {
    # 海马体 - 黄色系
    17: (255, 255, 0),    # Left Hippocampus
    53: (255, 255, 0),    # Right Hippocampus
    # 杏仁核 - 青色
    18: (0, 255, 255),    # Left Amygdala
    54: (0, 255, 255),    # Right Amygdala
    # 内嗅皮层 - 橙色
    1006: (255, 165, 0),
    2006: (255, 165, 0),
    # 海马旁回 - 粉色
    1016: (255, 105, 180),
    2016: (255, 105, 180),
    # 后扣带回 - 绿色
    1023: (0, 255, 0),
    2023: (0, 255, 0),
    # 楔前叶 - 蓝色
    1025: (0, 0, 255),
    2025: (0, 0, 255),
}


def get_region_color(label: int, alpha: float = 1.0) -> Tuple[float, ...]:
    """获取脑区颜色"""
    if label in FREESURFER_COLORS:
        r, g, b = FREESURFER_COLORS[label]
        return (r/255, g/255, b/255, alpha)
    else:
        # 根据标签生成伪随机颜色
        np.random.seed(label)
        return (*np.random.rand(3), alpha)


# ============================================================
# 1. 脑部 MRI 基础可视化
# ============================================================

def plot_brain_slices(
    volume: np.ndarray,
    slices: Optional[List[int]] = None,
    n_slices: int = 7,
    axis: int = 2,
    cmap: str = 'gray',
    title: str = 'Brain MRI',
    figsize: Tuple[int, int] = (16, 4),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    绘制多个脑部切片
    
    Args:
        volume: 3D MRI 体积 (D, H, W)
        slices: 指定的切片索引，None 则自动选择
        n_slices: 切片数量
        axis: 切片轴 (0=矢状面, 1=冠状面, 2=轴位)
        cmap: 颜色映射
        title: 标题
        figsize: 图像大小
        vmin, vmax: 显示范围
    """
    if slices is None:
        # 自动选择切片，避开边缘
        total = volume.shape[axis]
        margin = total // 8
        slices = np.linspace(margin, total - margin, n_slices).astype(int)
    
    n_slices = len(slices)
    fig, axes = plt.subplots(1, n_slices, figsize=figsize)
    
    axis_names = ['Sagittal', 'Coronal', 'Axial']
    
    for i, (ax, slice_idx) in enumerate(zip(axes, slices)):
        # 获取切片
        slicer = [slice(None)] * 3
        slicer[axis] = slice_idx
        img_slice = volume[tuple(slicer)]
        
        # 调整显示方向
        if axis == 0:
            img_slice = np.rot90(img_slice)
        elif axis == 1:
            img_slice = np.rot90(img_slice)
        else:
            img_slice = img_slice.T
        
        im = ax.imshow(img_slice, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f'Slice {slice_idx}')
        ax.axis('off')
    
    fig.suptitle(f'{title} ({axis_names[axis]} View)', fontsize=14)
    plt.tight_layout()
    return fig


def plot_three_views(
    volume: np.ndarray,
    segmentation: Optional[np.ndarray] = None,
    slices: Optional[Tuple[int, int, int]] = None,
    overlay_alpha: float = 0.4,
    figsize: Tuple[int, int] = (15, 5),
    title: str = 'Three Orthogonal Views',
) -> plt.Figure:
    """
    绘制三个正交视图（矢状面、冠状面、轴位）
    
    Args:
        volume: 3D MRI 体积
        segmentation: 可选的分割图谱叠加
        slices: 三个视图的切片索引 (sagittal, coronal, axial)
        overlay_alpha: 叠加透明度
        figsize: 图像大小
        title: 标题
    """
    if slices is None:
        slices = tuple(s // 2 for s in volume.shape)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    views = [
        ('Sagittal (X)', volume[slices[0], :, :].T, 
         segmentation[slices[0], :, :].T if segmentation is not None else None),
        ('Coronal (Y)', volume[:, slices[1], :].T,
         segmentation[:, slices[1], :].T if segmentation is not None else None),
        ('Axial (Z)', volume[:, :, slices[2]].T,
         segmentation[:, :, slices[2]].T if segmentation is not None else None),
    ]
    
    for ax, (name, img, seg) in zip(axes, views):
        ax.imshow(img, cmap='gray', origin='lower')
        
        if seg is not None:
            # 创建彩色叠加
            colored_seg = np.zeros((*seg.shape, 4))
            for label in np.unique(seg):
                if label > 0:
                    mask = seg == label
                    colored_seg[mask] = get_region_color(int(label), overlay_alpha)
            ax.imshow(colored_seg, origin='lower')
        
        ax.set_title(name)
        ax.axis('off')
        
        # 添加十字线
        ax.axhline(y=img.shape[0]//2, color='yellow', linewidth=0.5, alpha=0.5)
        ax.axvline(x=img.shape[1]//2, color='yellow', linewidth=0.5, alpha=0.5)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_brain_mosaic(
    volume: np.ndarray,
    n_rows: int = 4,
    n_cols: int = 6,
    axis: int = 2,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'gray',
) -> plt.Figure:
    """
    绘制脑部切片马赛克图
    """
    total_slices = volume.shape[axis]
    indices = np.linspace(0, total_slices-1, n_rows * n_cols).astype(int)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for ax, idx in zip(axes, indices):
        slicer = [slice(None)] * 3
        slicer[axis] = idx
        img = volume[tuple(slicer)]
        ax.imshow(img.T, cmap=cmap, origin='lower')
        ax.axis('off')
    
    plt.tight_layout()
    return fig


# ============================================================
# 2. 注意力图可视化
# ============================================================

def visualize_attention_heatmap(
    attention_weights: Union[np.ndarray, torch.Tensor],
    volume_shape: Tuple[int, int, int],
    slice_idx: Optional[int] = None,
    axis: int = 2,
    cmap: str = 'hot',
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Attention Heatmap',
) -> plt.Figure:
    """
    将注意力权重可视化为热力图
    
    Args:
        attention_weights: 展平的注意力权重 (N,) 或 3D (D, H, W)
        volume_shape: 原始体积形状
        slice_idx: 切片索引
        axis: 切片轴
        cmap: 颜色映射
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # 重塑为3D
    if attention_weights.ndim == 1:
        # 尝试重塑
        target_size = np.prod(volume_shape)
        if len(attention_weights) != target_size:
            # 需要上采样
            scale = int(round((target_size / len(attention_weights)) ** (1/3)))
            small_shape = tuple(s // scale for s in volume_shape)
            attention_3d = attention_weights.reshape(small_shape)
            from scipy.ndimage import zoom
            attention_3d = zoom(attention_3d, scale, order=1)
        else:
            attention_3d = attention_weights.reshape(volume_shape)
    else:
        attention_3d = attention_weights
    
    if slice_idx is None:
        slice_idx = volume_shape[axis] // 2
    
    # 获取切片
    slicer = [slice(None)] * 3
    slicer[axis] = slice_idx
    attn_slice = attention_3d[tuple(slicer)]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    im = ax.imshow(attn_slice.T, cmap=cmap, origin='lower')
    plt.colorbar(im, ax=ax, label='Attention Weight')
    ax.set_title(f'{title} (Slice {slice_idx})')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def visualize_attention_on_mri(
    mri_volume: np.ndarray,
    attention_weights: np.ndarray,
    segmentation: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    attention_threshold: float = 0.1,
    figsize: Tuple[int, int] = (15, 5),
    title: str = 'Attention Overlay',
) -> plt.Figure:
    """
    在 MRI 上叠加注意力热力图
    
    Args:
        mri_volume: T1 MRI 体积
        attention_weights: 注意力权重（与 MRI 相同尺寸或需要上采样）
        segmentation: 可选的分割图谱
        slice_idx: 切片索引
        axis: 切片轴
        attention_threshold: 显示阈值
    """
    if slice_idx is None:
        slice_idx = mri_volume.shape[axis] // 2
    
    # 确保注意力与 MRI 尺寸匹配
    if attention_weights.shape != mri_volume.shape:
        from scipy.ndimage import zoom
        factors = [m/a for m, a in zip(mri_volume.shape, attention_weights.shape)]
        attention_weights = zoom(attention_weights, factors, order=1)
    
    slicer = [slice(None)] * 3
    slicer[axis] = slice_idx
    
    mri_slice = mri_volume[tuple(slicer)].T
    attn_slice = attention_weights[tuple(slicer)].T
    seg_slice = segmentation[tuple(slicer)].T if segmentation is not None else None
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 原始 MRI
    axes[0].imshow(mri_slice, cmap='gray', origin='lower')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')
    
    # 注意力热力图
    axes[1].imshow(attn_slice, cmap='hot', origin='lower')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    
    # 叠加显示
    axes[2].imshow(mri_slice, cmap='gray', origin='lower')
    
    # 应用阈值
    attn_masked = np.ma.masked_where(attn_slice < attention_threshold, attn_slice)
    im = axes[2].imshow(attn_masked, cmap='hot', alpha=0.7, origin='lower')
    
    # 如果有分割，绘制边界
    if seg_slice is not None:
        from scipy import ndimage
        for label in np.unique(seg_slice):
            if label > 0:
                mask = seg_slice == label
                contours = ndimage.find_objects(mask.astype(int))
                # 简单边界
                edges = ndimage.binary_dilation(mask) ^ mask
                axes[2].contour(edges, colors=[get_region_color(int(label))[:3]], 
                               linewidths=0.5, alpha=0.5)
    
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.colorbar(im, ax=axes[2], label='Attention', shrink=0.8)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def visualize_query_attention_patterns(
    attention_maps: Union[np.ndarray, torch.Tensor],
    query_indices: List[int],
    volume_shape: Tuple[int, int, int],
    slice_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """
    可视化多个 Query Token 的注意力模式
    
    Args:
        attention_maps: 注意力权重 (num_queries, num_keys)
        query_indices: 要可视化的 query 索引
        volume_shape: 空间维度
        slice_idx: 切片索引
    """
    if isinstance(attention_maps, torch.Tensor):
        attention_maps = attention_maps.detach().cpu().numpy()
    
    n_queries = len(query_indices)
    n_cols = min(4, n_queries)
    n_rows = (n_queries + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    if slice_idx is None:
        slice_idx = volume_shape[2] // 2
    
    for i, (ax, q_idx) in enumerate(zip(axes, query_indices)):
        attn = attention_maps[q_idx]  # (num_keys,)
        
        # 重塑
        scale = int(round((np.prod(volume_shape) / len(attn)) ** (1/3)))
        small_shape = tuple(s // scale for s in volume_shape)
        attn_3d = attn.reshape(small_shape)
        
        # 上采样
        from scipy.ndimage import zoom
        attn_3d = zoom(attn_3d, scale, order=1)
        
        # 获取切片
        attn_slice = attn_3d[:, :, min(slice_idx, attn_3d.shape[2]-1)]
        
        im = ax.imshow(attn_slice.T, cmap='hot', origin='lower')
        ax.set_title(f'Query {q_idx}')
        ax.axis('off')
    
    # 隐藏多余的子图
    for ax in axes[n_queries:]:
        ax.axis('off')
    
    fig.suptitle('Query Attention Patterns', fontsize=14)
    plt.tight_layout()
    return fig


def visualize_attention_over_regions(
    attention_weights: np.ndarray,
    segmentation: np.ndarray,
    region_labels: Dict[str, int],
    top_k: int = 10,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    可视化每个脑区接收的注意力总量
    
    Args:
        attention_weights: 注意力权重 (与分割同尺寸)
        segmentation: 脑区分割
        region_labels: 区域名称到标签的映射
        top_k: 显示前 k 个区域
    """
    # 确保尺寸匹配
    if attention_weights.shape != segmentation.shape:
        from scipy.ndimage import zoom
        factors = [s/a for s, a in zip(segmentation.shape, attention_weights.shape)]
        attention_weights = zoom(attention_weights, factors, order=1)
    
    # 计算每个区域的总注意力
    region_attention = {}
    for name, label in region_labels.items():
        mask = segmentation == label
        if mask.sum() > 0:
            total_attn = attention_weights[mask].sum()
            mean_attn = attention_weights[mask].mean()
            region_attention[name] = {
                'total': total_attn,
                'mean': mean_attn,
                'size': mask.sum(),
            }
    
    # 排序
    sorted_regions = sorted(
        region_attention.items(), 
        key=lambda x: x[1]['mean'], 
        reverse=True
    )[:top_k]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 平均注意力
    names = [r[0] for r in sorted_regions]
    mean_attns = [r[1]['mean'] for r in sorted_regions]
    
    colors = [get_region_color(region_labels[n])[:3] for n in names]
    
    axes[0].barh(range(len(names)), mean_attns, color=colors)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel('Mean Attention')
    axes[0].set_title('Mean Attention per Region')
    axes[0].invert_yaxis()
    
    # 总注意力
    total_attns = [r[1]['total'] for r in sorted_regions]
    
    axes[1].barh(range(len(names)), total_attns, color=colors)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names)
    axes[1].set_xlabel('Total Attention')
    axes[1].set_title('Total Attention per Region')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    return fig


# ============================================================
# 3. 测地距离可视化
# ============================================================

def visualize_geodesic_distance_matrix(
    distance_matrix: np.ndarray,
    region_names: List[str],
    figsize: Tuple[int, int] = (14, 12),
    cmap: str = 'viridis',
    annotate: bool = True,
    title: str = 'Geodesic Distance Matrix',
) -> plt.Figure:
    """
    可视化区域间测地距离矩阵
    
    Args:
        distance_matrix: 距离矩阵 (n_regions, n_regions)
        region_names: 区域名称列表
        figsize: 图像大小
        cmap: 颜色映射
        annotate: 是否标注数值
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    im = ax.imshow(distance_matrix, cmap=cmap, aspect='auto')
    
    # 设置标签
    ax.set_xticks(range(len(region_names)))
    ax.set_yticks(range(len(region_names)))
    ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(region_names, fontsize=8)
    
    # 添加数值标注
    if annotate and len(region_names) <= 15:
        for i in range(len(region_names)):
            for j in range(len(region_names)):
                val = distance_matrix[i, j]
                color = 'white' if val > distance_matrix.max()/2 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color=color, fontsize=7)
    
    plt.colorbar(im, ax=ax, label='Geodesic Distance (mm)')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Target Region')
    ax.set_ylabel('Source Region')
    
    plt.tight_layout()
    return fig


def visualize_geodesic_from_region(
    distance_volume: np.ndarray,
    source_region_name: str,
    mri_volume: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    max_distance: float = 100.0,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    可视化从某个区域出发的测地距离
    
    Args:
        distance_volume: 距离体积 (D, H, W)
        source_region_name: 源区域名称
        mri_volume: 可选的 MRI 背景
        slice_idx: 切片索引
        axis: 切片轴
        max_distance: 最大显示距离
    """
    if slice_idx is None:
        slice_idx = distance_volume.shape[axis] // 2
    
    slicer = [slice(None)] * 3
    slicer[axis] = slice_idx
    
    dist_slice = distance_volume[tuple(slicer)].T
    dist_slice = np.clip(dist_slice, 0, max_distance)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 距离图
    im1 = axes[0].imshow(dist_slice, cmap='viridis', origin='lower')
    axes[0].set_title(f'Geodesic Distance from {source_region_name}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], label='Distance (mm)')
    
    # 与 MRI 叠加
    if mri_volume is not None:
        mri_slice = mri_volume[tuple(slicer)].T
        axes[1].imshow(mri_slice, cmap='gray', origin='lower')
        im2 = axes[1].imshow(dist_slice, cmap='hot', alpha=0.5, origin='lower')
        axes[1].set_title('Overlay on MRI')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], label='Distance (mm)')
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def plot_attention_vs_geodesic_distance(
    attention_weights: np.ndarray,
    geodesic_distances: np.ndarray,
    n_bins: int = 20,
    figsize: Tuple[int, int] = (14, 5),
    title: str = 'Attention vs Geodesic Distance',
) -> plt.Figure:
    """
    分析注意力权重与测地距离的关系
    
    理想情况：测地距离越近，注意力越高
    
    Args:
        attention_weights: 注意力矩阵
        geodesic_distances: 距离矩阵
        n_bins: 分箱数量
    """
    # 展平
    attn_flat = attention_weights.flatten()
    dist_flat = geodesic_distances.flatten()
    
    # 移除对角线（自注意力）
    mask = dist_flat > 0
    attn_flat = attn_flat[mask]
    dist_flat = dist_flat[mask]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 散点图
    axes[0].scatter(dist_flat, attn_flat, alpha=0.1, s=2)
    axes[0].set_xlabel('Geodesic Distance (mm)')
    axes[0].set_ylabel('Attention Weight')
    axes[0].set_title('Scatter Plot')
    axes[0].set_xlim(0, np.percentile(dist_flat, 99))
    
    # 分箱平均
    bins = np.linspace(0, np.percentile(dist_flat, 99), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(dist_flat, bins)
    
    bin_means = []
    bin_stds = []
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(attn_flat[mask].mean())
            bin_stds.append(attn_flat[mask].std())
        else:
            bin_means.append(0)
            bin_stds.append(0)
    
    axes[1].bar(bin_centers, bin_means, width=bins[1]-bins[0], alpha=0.7, 
               yerr=bin_stds, capsize=2)
    axes[1].set_xlabel('Geodesic Distance (mm)')
    axes[1].set_ylabel('Mean Attention ± Std')
    axes[1].set_title('Binned Average')
    
    # 相关性热力图
    hist, xedges, yedges = np.histogram2d(dist_flat, attn_flat, bins=30)
    axes[2].imshow(hist.T, origin='lower', aspect='auto', cmap='Blues',
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    axes[2].set_xlabel('Geodesic Distance (mm)')
    axes[2].set_ylabel('Attention Weight')
    axes[2].set_title('2D Histogram')
    
    # 计算相关系数
    corr = np.corrcoef(dist_flat, attn_flat)[0, 1]
    fig.suptitle(f'{title}\n(Pearson r = {corr:.3f})', fontsize=14)
    
    plt.tight_layout()
    return fig


# ============================================================
# 4. 训练过程可视化
# ============================================================

def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (16, 10),
    title: str = 'Training History',
) -> plt.Figure:
    """
    绘制完整的训练历史
    
    Args:
        history: 包含各种指标的字典，如：
            - train_loss, val_loss
            - train_acc, val_acc
            - geodesic_loss, entropy_loss
            - learning_rate
    """
    # 确定子图数量
    metric_groups = {
        'Loss': ['train_loss', 'val_loss'],
        'Accuracy': ['train_acc', 'val_acc'],
        'Geodesic Loss': ['geodesic_loss', 'geo_geodesic'],
        'Attention Losses': ['geo_entropy', 'geo_sparsity', 'geo_atlas_prior'],
        'Learning Rate': ['learning_rate', 'lr'],
    }
    
    # 找出有数据的组
    active_groups = []
    for name, keys in metric_groups.items():
        for key in keys:
            if key in history:
                active_groups.append(name)
                break
    
    n_plots = len(active_groups)
    if n_plots == 0:
        print("No metrics found in history!")
        return None
    
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    colors = plt.cm.tab10.colors
    
    for i, group_name in enumerate(active_groups):
        ax = axes[i]
        keys = metric_groups[group_name]
        
        for j, key in enumerate(keys):
            if key in history:
                epochs = range(1, len(history[key]) + 1)
                ax.plot(epochs, history[key], color=colors[j], 
                       label=key, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(group_name)
        ax.set_title(group_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余子图
    for ax in axes[n_plots:]:
        ax.axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['CN', 'MCI', 'AD'],
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = True,
    title: str = 'Confusion Matrix',
) -> plt.Figure:
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        normalize: 是否归一化
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # 添加数值
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = f'{cm[i, j]:.2f}' if normalize else f'{cm[i, j]}'
            ax.text(j, i, val, ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str] = ['CN', 'MCI', 'AD'],
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    绘制多类别 ROC 曲线
    
    Args:
        y_true: 真实标签 (one-hot 或整数)
        y_scores: 预测概率 (N, n_classes)
        class_names: 类别名称
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(class_names)
    
    # 转换为 one-hot
    if y_true.ndim == 1:
        y_true = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    colors = plt.cm.tab10.colors
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
               label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================
# 5. 模型解释性可视化
# ============================================================

def visualize_region_importance(
    model_weights: Dict[str, float],
    region_names: List[str],
    figsize: Tuple[int, int] = (12, 8),
    title: str = 'Region Importance',
) -> plt.Figure:
    """
    可视化模型学习到的区域重要性
    
    Args:
        model_weights: 区域名称到重要性分数的映射
        region_names: 区域名称列表
    """
    # 排序
    sorted_items = sorted(model_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    colors = ['green' if v > 0 else 'red' for v in values]
    
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def visualize_feature_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    class_names: List[str] = ['CN', 'MCI', 'AD'],
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Feature Embeddings',
) -> plt.Figure:
    """
    可视化特征嵌入（t-SNE 或 UMAP）
    
    Args:
        embeddings: 特征向量 (N, D)
        labels: 类别标签 (N,)
        method: 降维方法 ('tsne' 或 'umap')
        class_names: 类别名称
    """
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
        except ImportError:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            method = 'tsne'
    
    embedded = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    colors = plt.cm.tab10.colors
    
    for i, name in enumerate(class_names):
        mask = labels == i
        ax.scatter(embedded[mask, 0], embedded[mask, 1], 
                  c=[colors[i]], label=name, alpha=0.6, s=50)
    
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig


# ============================================================
# 6. 综合可视化面板
# ============================================================

def create_analysis_dashboard(
    mri_volume: np.ndarray,
    segmentation: np.ndarray,
    attention_weights: np.ndarray,
    prediction_probs: np.ndarray,
    region_labels: Dict[str, int],
    subject_id: str = 'Unknown',
    figsize: Tuple[int, int] = (20, 16),
) -> plt.Figure:
    """
    创建综合分析面板
    
    Args:
        mri_volume: MRI 体积
        segmentation: 分割图谱
        attention_weights: 注意力权重
        prediction_probs: 预测概率 [P(CN), P(MCI), P(AD)]
        region_labels: 区域标签
        subject_id: 被试 ID
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    slice_idx = mri_volume.shape[2] // 2
    
    # 1. 三视图 MRI
    for i, (ax_pos, axis, name) in enumerate([
        ((0, 0), 0, 'Sagittal'),
        ((0, 1), 1, 'Coronal'),
        ((0, 2), 2, 'Axial'),
    ]):
        ax = fig.add_subplot(gs[ax_pos[0], ax_pos[1]])
        slicer = [slice(None)] * 3
        slicer[axis] = mri_volume.shape[axis] // 2
        img = mri_volume[tuple(slicer)]
        ax.imshow(img.T, cmap='gray', origin='lower')
        ax.set_title(name)
        ax.axis('off')
    
    # 2. 预测结果
    ax = fig.add_subplot(gs[0, 3])
    class_names = ['CN', 'MCI', 'AD']
    colors = ['green', 'orange', 'red']
    bars = ax.bar(class_names, prediction_probs, color=colors, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title(f'Prediction\n(Subject: {subject_id})')
    
    # 标注最高概率
    max_idx = np.argmax(prediction_probs)
    ax.annotate(f'{prediction_probs[max_idx]:.1%}', 
               xy=(max_idx, prediction_probs[max_idx]),
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 3. 注意力热力图（三视图）
    if attention_weights.shape != mri_volume.shape:
        from scipy.ndimage import zoom
        factors = [m/a for m, a in zip(mri_volume.shape, attention_weights.shape)]
        attention_weights = zoom(attention_weights, factors, order=1)
    
    for i, (ax_pos, axis, name) in enumerate([
        ((1, 0), 0, 'Attention (Sagittal)'),
        ((1, 1), 1, 'Attention (Coronal)'),
        ((1, 2), 2, 'Attention (Axial)'),
    ]):
        ax = fig.add_subplot(gs[ax_pos[0], ax_pos[1]])
        slicer = [slice(None)] * 3
        slicer[axis] = mri_volume.shape[axis] // 2
        
        mri_slice = mri_volume[tuple(slicer)].T
        attn_slice = attention_weights[tuple(slicer)].T
        
        ax.imshow(mri_slice, cmap='gray', origin='lower')
        im = ax.imshow(attn_slice, cmap='hot', alpha=0.5, origin='lower')
        ax.set_title(name)
        ax.axis('off')
    
    # 4. 区域注意力条形图
    ax = fig.add_subplot(gs[1, 3])
    region_attn = {}
    for name, label in list(region_labels.items())[:8]:  # Top 8
        mask = segmentation == label
        if mask.sum() > 0:
            region_attn[name.replace('left_', 'L-').replace('right_', 'R-')] = \
                attention_weights[mask].mean()
    
    sorted_regions = sorted(region_attn.items(), key=lambda x: x[1], reverse=True)
    names = [r[0] for r in sorted_regions]
    values = [r[1] for r in sorted_regions]
    
    ax.barh(range(len(names)), values, color='coral')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Mean Attention')
    ax.set_title('Region Attention')
    ax.invert_yaxis()
    
    # 5. 分割叠加
    ax = fig.add_subplot(gs[2, 0:2])
    slice_idx = mri_volume.shape[2] // 2
    mri_slice = mri_volume[:, :, slice_idx].T
    seg_slice = segmentation[:, :, slice_idx].T
    
    ax.imshow(mri_slice, cmap='gray', origin='lower')
    
    # 彩色叠加分割
    colored_seg = np.zeros((*seg_slice.shape, 4))
    for label in np.unique(seg_slice):
        if label > 0:
            mask = seg_slice == label
            colored_seg[mask] = get_region_color(int(label), 0.4)
    ax.imshow(colored_seg, origin='lower')
    ax.set_title('Atlas Segmentation Overlay')
    ax.axis('off')
    
    # 6. 图例
    ax = fig.add_subplot(gs[2, 2:4])
    ax.axis('off')
    
    # 创建图例
    legend_items = []
    for name, label in list(region_labels.items())[:10]:
        color = get_region_color(label)[:3]
        legend_items.append(mpatches.Patch(color=color, label=name))
    
    ax.legend(handles=legend_items, loc='center', ncol=2, fontsize=9)
    ax.set_title('Region Legend')
    
    fig.suptitle(f'Atlas-Guided Attention Analysis - {subject_id}', fontsize=16)
    
    return fig


# ============================================================
# 保存工具
# ============================================================

def save_all_figures(
    figures: Dict[str, plt.Figure],
    output_dir: str,
    formats: List[str] = ['png', 'pdf'],
    dpi: int = 150,
):
    """
    批量保存所有图像
    
    Args:
        figures: 图像名称到 Figure 对象的映射
        output_dir: 输出目录
        formats: 保存格式列表
        dpi: 分辨率
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, fig in figures.items():
        if fig is None:
            continue
        for fmt in formats:
            filepath = output_dir / f'{name}.{fmt}'
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f'Saved: {filepath}')
    
    print(f'\nAll figures saved to: {output_dir}')
