"""
Visualization Demo Script

展示所有可视化功能，使用合成数据进行演示。
支持使用新生成的 500 个样本数据集。
"""
import sys
from pathlib import Path

# 自动添加项目根目录到 path（无论从哪里运行）
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
import nibabel as nib
from tqdm import tqdm

# 导入可视化模块
from utils.visualization import (
    # 基础 MRI 可视化
    plot_brain_slices,
    plot_three_views,
    plot_brain_mosaic,
    # 注意力可视化
    visualize_attention_heatmap,
    visualize_attention_on_mri,
    visualize_query_attention_patterns,
    visualize_attention_over_regions,
    # 测地距离可视化
    visualize_geodesic_distance_matrix,
    plot_attention_vs_geodesic_distance,
    # 训练可视化
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curves,
    # 模型解释
    visualize_region_importance,
    visualize_feature_embeddings,
    # 综合面板
    create_analysis_dashboard,
    save_all_figures,
)

# 创建输出目录（使用脚本所在目录）
OUTPUT_DIR = SCRIPT_DIR / 'visualization_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# 合成数据目录
SYNTHETIC_DATA_DIR = PROJECT_ROOT / 'sample_data' / 'synthetic'


def load_real_synthetic_data(subject_id=None, data_dir=None):
    """
    加载真实合成数据集中的样本
    
    Args:
        subject_id: 指定的受试者 ID（如 'SYNTH_0001'），如果为 None 则随机选择
        data_dir: 数据目录，默认使用 SYNTHETIC_DATA_DIR
    
    Returns:
        dict: 包含 mri, segmentation, diagnosis, subject_id 等信息
    """
    if data_dir is None:
        data_dir = SYNTHETIC_DATA_DIR
    
    data_dir = Path(data_dir)
    
    # 加载元数据
    csv_file = data_dir / 'subjects.csv'
    if not csv_file.exists():
        print(f"警告: 找不到数据集 {csv_file}，使用简单合成数据")
        return None
    
    # 读取 CSV
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    # 选择受试者
    if subject_id is None:
        row = df.sample(1).iloc[0]
        subject_id = row['PTID']
    else:
        row = df[df['PTID'] == subject_id].iloc[0]
    
    # 加载 MRI
    mri_path = data_dir / 'mri' / f'{subject_id}_T1.nii.gz'
    seg_path = data_dir / 'segmentation' / f'{subject_id}_seg.nii.gz'
    
    if not mri_path.exists():
        print(f"警告: 找不到 MRI 文件 {mri_path}")
        return None
    
    mri_img = nib.load(str(mri_path))
    mri = mri_img.get_fdata().astype(np.float32)
    
    seg = None
    if seg_path.exists():
        seg_img = nib.load(str(seg_path))
        seg = seg_img.get_fdata().astype(np.int32)
    
    return {
        'mri': mri,
        'segmentation': seg,
        'subject_id': subject_id,
        'diagnosis': row['DX_bl'],
        'age': row['AGE'],
        'sex': row['PTGENDER'],
    }


def load_multiple_subjects(n_subjects=50, data_dir=None, balance_classes=True):
    """
    加载多个受试者数据
    
    Args:
        n_subjects: 要加载的受试者数量
        data_dir: 数据目录
        balance_classes: 是否平衡各类别数量
    
    Returns:
        list: 包含多个受试者数据的列表
    """
    if data_dir is None:
        data_dir = SYNTHETIC_DATA_DIR
    
    data_dir = Path(data_dir)
    csv_file = data_dir / 'subjects.csv'
    
    if not csv_file.exists():
        return None
    
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    if balance_classes:
        # 每个类别取相同数量
        n_per_class = n_subjects // 3
        subjects = []
        for dx in ['CN', 'MCI', 'AD']:
            class_df = df[df['DX_bl'] == dx].sample(min(n_per_class, len(df[df['DX_bl'] == dx])))
            subjects.extend(class_df['PTID'].tolist())
    else:
        subjects = df.sample(min(n_subjects, len(df)))['PTID'].tolist()
    
    data_list = []
    for sid in tqdm(subjects, desc="加载数据"):
        data = load_real_synthetic_data(sid, data_dir)
        if data is not None:
            data_list.append(data)
    
    return data_list


def create_synthetic_mri(shape=(64, 64, 64)):
    """创建合成 MRI 数据（用于演示，当没有真实数据时的备选）"""
    # 创建基础体积
    volume = np.random.randn(*shape) * 0.1
    
    # 添加球形结构模拟脑组织
    center = np.array(shape) // 2
    coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # 大脑轮廓
    brain_radius = min(shape) // 2 - 5
    brain_mask = sum((c - center[i])**2 for i, c in enumerate(coords)) < brain_radius**2
    volume[brain_mask] += 0.5
    
    # 添加一些内部结构
    # 侧脑室
    ventricle_center = center + np.array([0, 5, 0])
    ventricle_mask = sum((c - ventricle_center[i])**2 for i, c in enumerate(coords)) < 8**2
    volume[ventricle_mask] = 0.1
    
    # 灰质/白质边界
    inner_radius = brain_radius - 10
    inner_mask = sum((c - center[i])**2 for i, c in enumerate(coords)) < inner_radius**2
    volume[inner_mask & brain_mask] += 0.2
    
    # 归一化
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    return volume


def create_synthetic_segmentation(shape=(64, 64, 64)):
    """创建合成分割图谱"""
    segmentation = np.zeros(shape, dtype=np.int32)
    center = np.array(shape) // 2
    coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # 定义一些区域
    regions = {
        17: {'center': center + np.array([-15, -5, 0]), 'radius': 5},   # L Hippocampus
        53: {'center': center + np.array([15, -5, 0]), 'radius': 5},    # R Hippocampus
        18: {'center': center + np.array([-12, -8, 0]), 'radius': 4},   # L Amygdala
        54: {'center': center + np.array([12, -8, 0]), 'radius': 4},    # R Amygdala
        1006: {'center': center + np.array([-18, -10, -5]), 'radius': 6},  # L Entorhinal
        2006: {'center': center + np.array([18, -10, -5]), 'radius': 6},   # R Entorhinal
        1023: {'center': center + np.array([0, 15, 10]), 'radius': 7},     # Posterior Cingulate
        1025: {'center': center + np.array([0, 20, 5]), 'radius': 8},      # Precuneus
    }
    
    for label, props in regions.items():
        c = props['center']
        r = props['radius']
        mask = sum((coord - c[i])**2 for i, coord in enumerate(coords)) < r**2
        segmentation[mask] = label
    
    return segmentation


def create_synthetic_attention(shape=(64, 64, 64), focus_regions=None):
    """创建合成注意力图"""
    attention = np.random.rand(*shape) * 0.1
    center = np.array(shape) // 2
    coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # 在海马体区域添加高注意力
    hippocampus_centers = [
        center + np.array([-15, -5, 0]),
        center + np.array([15, -5, 0]),
    ]
    
    for hc in hippocampus_centers:
        dist = np.sqrt(sum((c - hc[i])**2 for i, c in enumerate(coords)))
        attention += np.exp(-dist**2 / 50)
    
    # 归一化
    attention = attention / attention.sum()
    
    return attention


def demo_basic_mri_visualization(use_real_data=True, subject_data=None):
    """演示基础 MRI 可视化"""
    print("\n" + "="*60)
    print("Demo 1: Basic MRI Visualization")
    print("="*60)
    
    if use_real_data and subject_data is not None:
        mri = subject_data['mri']
        seg = subject_data['segmentation']
        subject_id = subject_data['subject_id']
        diagnosis = subject_data['diagnosis']
        title_prefix = f"{subject_id} ({diagnosis})"
    else:
        mri = create_synthetic_mri(shape=(64, 64, 64))
        seg = create_synthetic_segmentation(shape=(64, 64, 64))
        title_prefix = "Synthetic"
    
    # 1. 多切片视图
    fig1 = plot_brain_slices(
        mri, 
        n_slices=7, 
        axis=2,
        title=f'{title_prefix} Brain MRI'
    )
    fig1.savefig(OUTPUT_DIR / 'brain_slices.png', dpi=150)
    print("✓ Saved: brain_slices.png")
    
    # 2. 三视图
    fig2 = plot_three_views(
        mri, 
        segmentation=seg,
        title=f'Three Orthogonal Views - {title_prefix}'
    )
    fig2.savefig(OUTPUT_DIR / 'three_views.png', dpi=150)
    print("✓ Saved: three_views.png")
    
    # 3. 马赛克图
    fig3 = plot_brain_mosaic(mri, n_rows=3, n_cols=5)
    fig3.savefig(OUTPUT_DIR / 'brain_mosaic.png', dpi=150)
    print("✓ Saved: brain_mosaic.png")
    
    plt.close('all')
    return mri, seg


def demo_attention_visualization(mri, seg):
    """演示注意力可视化"""
    print("\n" + "="*60)
    print("Demo 2: Attention Visualization")
    print("="*60)
    
    attention = create_synthetic_attention(shape=mri.shape)
    
    # 1. 注意力热力图
    fig1 = visualize_attention_heatmap(
        attention,
        volume_shape=mri.shape,
        title='Attention Heatmap'
    )
    fig1.savefig(OUTPUT_DIR / 'attention_heatmap.png', dpi=150)
    print("✓ Saved: attention_heatmap.png")
    
    # 2. MRI 上的注意力叠加
    fig2 = visualize_attention_on_mri(
        mri,
        attention,
        segmentation=seg,
        attention_threshold=0.0001,
        title='Attention Overlay on MRI'
    )
    fig2.savefig(OUTPUT_DIR / 'attention_overlay.png', dpi=150)
    print("✓ Saved: attention_overlay.png")
    
    # 3. Query 注意力模式
    # 创建多个 query 的注意力
    num_queries = 8
    num_keys = 8 * 8 * 8  # Downsampled
    query_attention = np.random.rand(num_queries, num_keys)
    
    # 让不同 query 关注不同区域
    for q in range(num_queries):
        focus_idx = q * num_keys // num_queries
        query_attention[q, focus_idx:focus_idx+50] += 1.0
    query_attention = query_attention / query_attention.sum(axis=1, keepdims=True)
    
    fig3 = visualize_query_attention_patterns(
        query_attention,
        query_indices=[0, 2, 4, 6],
        volume_shape=(8, 8, 8),
    )
    fig3.savefig(OUTPUT_DIR / 'query_attention_patterns.png', dpi=150)
    print("✓ Saved: query_attention_patterns.png")
    
    # 4. 区域注意力分析
    region_labels = {
        'Left Hippocampus': 17,
        'Right Hippocampus': 53,
        'Left Amygdala': 18,
        'Right Amygdala': 54,
        'Left Entorhinal': 1006,
        'Right Entorhinal': 2006,
        'Posterior Cingulate': 1023,
        'Precuneus': 1025,
    }
    
    fig4 = visualize_attention_over_regions(
        attention,
        seg,
        region_labels,
        top_k=8
    )
    fig4.savefig(OUTPUT_DIR / 'region_attention.png', dpi=150)
    print("✓ Saved: region_attention.png")
    
    plt.close('all')
    return attention


def demo_geodesic_visualization():
    """演示测地距离可视化"""
    print("\n" + "="*60)
    print("Demo 3: Geodesic Distance Visualization")
    print("="*60)
    
    # 创建合成距离矩阵
    region_names = [
        'L-Hippocampus', 'R-Hippocampus',
        'L-Amygdala', 'R-Amygdala',
        'L-Entorhinal', 'R-Entorhinal',
        'Post-Cingulate', 'Precuneus',
    ]
    n_regions = len(region_names)
    
    # 创建对称距离矩阵
    # 同侧区域距离近，对侧距离远
    distance_matrix = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            if i == j:
                distance_matrix[i, j] = 0
            elif i // 2 == j // 2:  # 同侧
                distance_matrix[i, j] = 10 + np.random.rand() * 5
            elif abs(i - j) == 1 and i // 2 == j // 2:  # 左右对称
                distance_matrix[i, j] = 30 + np.random.rand() * 10
            else:
                distance_matrix[i, j] = 40 + np.random.rand() * 30
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # 1. 距离矩阵
    fig1 = visualize_geodesic_distance_matrix(
        distance_matrix,
        region_names,
        title='Geodesic Distance Matrix'
    )
    fig1.savefig(OUTPUT_DIR / 'geodesic_matrix.png', dpi=150)
    print("✓ Saved: geodesic_matrix.png")
    
    # 2. 注意力 vs 距离
    # 创建符合预期的注意力（距离近 -> 注意力高）
    attention_matrix = np.exp(-distance_matrix / 30)
    attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
    
    fig2 = plot_attention_vs_geodesic_distance(
        attention_matrix,
        distance_matrix,
        title='Attention vs Geodesic Distance (Ideal Pattern)'
    )
    fig2.savefig(OUTPUT_DIR / 'attention_vs_distance.png', dpi=150)
    print("✓ Saved: attention_vs_distance.png")
    
    plt.close('all')


def demo_training_visualization(n_samples=500):
    """演示训练过程可视化"""
    print("\n" + "="*60)
    print("Demo 4: Training Visualization")
    print("="*60)
    
    # 创建合成训练历史
    n_epochs = 100
    
    history = {
        'train_loss': [2.0 * np.exp(-i/30) + 0.3 + np.random.rand()*0.1 for i in range(n_epochs)],
        'val_loss': [2.0 * np.exp(-i/30) + 0.4 + np.random.rand()*0.15 for i in range(n_epochs)],
        'train_acc': [1 - 0.7 * np.exp(-i/20) + np.random.rand()*0.05 for i in range(n_epochs)],
        'val_acc': [1 - 0.75 * np.exp(-i/20) + np.random.rand()*0.08 for i in range(n_epochs)],
        'geo_geodesic': [1.5 * np.exp(-i/40) + 0.2 + np.random.rand()*0.05 for i in range(n_epochs)],
        'geo_entropy': [0.5 * np.exp(-i/50) + 0.1 + np.random.rand()*0.02 for i in range(n_epochs)],
        'learning_rate': [1e-4 * (0.95 ** (i//10)) for i in range(n_epochs)],
    }
    
    # 1. 训练历史
    fig1 = plot_training_history(history, title=f'Training History (n={n_samples})')
    fig1.savefig(OUTPUT_DIR / 'training_history.png', dpi=150)
    print("✓ Saved: training_history.png")
    
    # 2. 混淆矩阵 - 使用更大的样本量
    np.random.seed(42)
    # 模拟更真实的分类结果
    # CN: 200, MCI: 175, AD: 125 (与数据集分布一致)
    y_true = np.concatenate([
        np.zeros(200, dtype=int),      # CN
        np.ones(175, dtype=int),       # MCI
        np.full(125, 2, dtype=int),    # AD
    ])
    
    y_pred = y_true.copy()
    # 添加更真实的错误模式
    # MCI 容易被误判为 CN (约 14%)
    mci_errors = np.random.choice(np.where(y_true == 1)[0], size=25, replace=False)
    y_pred[mci_errors] = 0
    
    # AD 容易被误判为 MCI 或 CN (约 16%)
    ad_errors = np.random.choice(np.where(y_true == 2)[0], size=20, replace=False)
    y_pred[ad_errors[:12]] = 0
    y_pred[ad_errors[12:]] = 1
    
    # CN 也有少量误判 (约 9%)
    cn_errors = np.random.choice(np.where(y_true == 0)[0], size=18, replace=False)
    y_pred[cn_errors[:12]] = 1
    y_pred[cn_errors[12:]] = 2
    
    fig2 = plot_confusion_matrix(y_true, y_pred, normalize=True)
    fig2.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150)
    print("✓ Saved: confusion_matrix.png")
    
    # 3. ROC 曲线
    # 创建更真实的预测概率
    y_scores = np.random.rand(n_samples, 3) * 0.3
    
    # 让预测有较高准确性
    for i in range(n_samples):
        y_scores[i, y_true[i]] += np.random.uniform(0.5, 0.8)
        # MCI 的不确定性更高
        if y_true[i] == 1:
            y_scores[i, 0] += np.random.uniform(0.1, 0.3)
            y_scores[i, 2] += np.random.uniform(0.05, 0.15)
    
    y_scores = y_scores / y_scores.sum(axis=1, keepdims=True)
    
    fig3 = plot_roc_curves(y_true, y_scores)
    fig3.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=150)
    print("✓ Saved: roc_curves.png")
    
    plt.close('all')
    
    return y_true, y_scores


def demo_interpretation_visualization(n_samples=500):
    """演示模型解释性可视化"""
    print("\n" + "="*60)
    print("Demo 5: Model Interpretation Visualization")
    print("="*60)
    
    # 1. 区域重要性 - 添加随机变异
    np.random.seed(42)
    region_importance = {
        'Left Hippocampus': 0.85 + np.random.uniform(-0.05, 0.05),
        'Right Hippocampus': 0.82 + np.random.uniform(-0.05, 0.05),
        'Left Entorhinal': 0.65 + np.random.uniform(-0.05, 0.05),
        'Right Entorhinal': 0.62 + np.random.uniform(-0.05, 0.05),
        'Left Amygdala': 0.45 + np.random.uniform(-0.05, 0.05),
        'Right Amygdala': 0.42 + np.random.uniform(-0.05, 0.05),
        'Posterior Cingulate': 0.55 + np.random.uniform(-0.05, 0.05),
        'Precuneus': 0.50 + np.random.uniform(-0.05, 0.05),
        'Left Frontal': -0.15 + np.random.uniform(-0.05, 0.05),
        'Right Frontal': -0.12 + np.random.uniform(-0.05, 0.05),
    }
    
    fig1 = visualize_region_importance(
        region_importance,
        list(region_importance.keys()),
        title=f'Region Importance for AD Classification (n={n_samples})'
    )
    fig1.savefig(OUTPUT_DIR / 'region_importance.png', dpi=150)
    print("✓ Saved: region_importance.png")
    
    # 2. 特征嵌入 - 使用更多样本，更清晰的分离
    np.random.seed(42)
    
    # 创建有结构的嵌入 - 使用实际数据集的类别分布
    # CN: 200, MCI: 175, AD: 125
    embeddings = []
    labels = []
    
    # CN (蓝色) - 右上角
    n_cn = 200
    cn_cluster = np.random.randn(n_cn, 64) * 0.8
    cn_cluster[:, 0] += 3.5  # x 偏移
    cn_cluster[:, 1] += 7.0  # y 偏移
    embeddings.append(cn_cluster)
    labels.extend([0] * n_cn)
    
    # MCI (橙色) - 中下方
    n_mci = 175
    mci_cluster = np.random.randn(n_mci, 64) * 1.0
    mci_cluster[:, 0] += 3.0  # x 偏移
    mci_cluster[:, 1] += -1.5  # y 偏移
    embeddings.append(mci_cluster)
    labels.extend([1] * n_mci)
    
    # AD (绿色) - 左侧
    n_ad = 125
    ad_cluster = np.random.randn(n_ad, 64) * 0.9
    ad_cluster[:, 0] += -3.0  # x 偏移
    ad_cluster[:, 1] += 1.0  # y 偏移
    embeddings.append(ad_cluster)
    labels.extend([2] * n_ad)
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    fig2 = visualize_feature_embeddings(
        embeddings,
        labels,
        method='tsne',
        title=f'Feature Embeddings (t-SNE, n={n_samples})'
    )
    fig2.savefig(OUTPUT_DIR / 'feature_embeddings.png', dpi=150)
    print("✓ Saved: feature_embeddings.png")
    
    plt.close('all')


def demo_dashboard(use_real_data=True, subject_data=None):
    """演示综合分析面板"""
    print("\n" + "="*60)
    print("Demo 6: Analysis Dashboard")
    print("="*60)
    
    if use_real_data and subject_data is not None:
        mri = subject_data['mri']
        seg = subject_data['segmentation']
        subject_id = subject_data['subject_id']
        diagnosis = subject_data['diagnosis']
        
        # 根据真实诊断生成预测概率
        if diagnosis == 'AD':
            prediction_probs = np.array([0.12, 0.23, 0.65])
        elif diagnosis == 'MCI':
            prediction_probs = np.array([0.25, 0.55, 0.20])
        else:  # CN
            prediction_probs = np.array([0.70, 0.22, 0.08])
        
        # 添加一些随机性
        prediction_probs += np.random.uniform(-0.05, 0.05, 3)
        prediction_probs = np.clip(prediction_probs, 0.01, 0.99)
        prediction_probs = prediction_probs / prediction_probs.sum()
    else:
        # 备用：使用简单合成数据
        mri = create_synthetic_mri(shape=(64, 64, 64))
        seg = create_synthetic_segmentation(shape=(64, 64, 64))
        subject_id = 'DEMO_001'
        prediction_probs = np.array([0.15, 0.25, 0.60])
    
    # 创建注意力图
    attention = create_synthetic_attention(shape=mri.shape)
    
    region_labels = {
        'left_hippocampus': 17,
        'right_hippocampus': 53,
        'left_amygdala': 18,
        'right_amygdala': 54,
        'left_entorhinal': 1006,
        'right_entorhinal': 2006,
        'posterior_cingulate': 1023,
        'precuneus': 1025,
    }
    
    fig = create_analysis_dashboard(
        mri_volume=mri,
        segmentation=seg,
        attention_weights=attention,
        prediction_probs=prediction_probs,
        region_labels=region_labels,
        subject_id=subject_id,
    )
    fig.savefig(OUTPUT_DIR / 'analysis_dashboard.png', dpi=150)
    print("✓ Saved: analysis_dashboard.png")
    
    plt.close('all')


def main(use_real_data=True):
    """运行所有演示"""
    print("\n" + "#"*60)
    print("# Atlas-Guided Attention Visualization Demo")
    print("#"*60)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    
    # 尝试加载真实合成数据
    subject_data = None
    n_samples = 500  # 使用数据集的实际样本数
    
    if use_real_data and SYNTHETIC_DATA_DIR.exists():
        print(f"\n正在从 {SYNTHETIC_DATA_DIR} 加载数据...")
        
        # 加载一个 AD 患者样本作为主要演示
        import pandas as pd
        csv_file = SYNTHETIC_DATA_DIR / 'subjects.csv'
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            n_samples = len(df)
            
            # 选择一个 AD 患者
            ad_subjects = df[df['DX_bl'] == 'AD']['PTID'].tolist()
            if ad_subjects:
                selected_subject = np.random.choice(ad_subjects)
                subject_data = load_real_synthetic_data(selected_subject)
                if subject_data:
                    print(f"✓ 已加载受试者: {subject_data['subject_id']}")
                    print(f"  诊断: {subject_data['diagnosis']}")
                    print(f"  年龄: {subject_data['age']:.1f}")
                    print(f"  性别: {subject_data['sex']}")
                    print(f"  MRI 尺寸: {subject_data['mri'].shape}")
    
    if subject_data is None:
        print("\n使用简单合成数据...")
        use_real_data = False
    
    # Demo 1: 基础 MRI 可视化
    mri, seg = demo_basic_mri_visualization(
        use_real_data=use_real_data, 
        subject_data=subject_data
    )
    
    # Demo 2: 注意力可视化
    attention = demo_attention_visualization(mri, seg)
    
    # Demo 3: 测地距离可视化
    demo_geodesic_visualization()
    
    # Demo 4: 训练过程可视化
    demo_training_visualization(n_samples=n_samples)
    
    # Demo 5: 模型解释性可视化
    demo_interpretation_visualization(n_samples=n_samples)
    
    # Demo 6: 综合面板
    demo_dashboard(use_real_data=use_real_data, subject_data=subject_data)
    
    print("\n" + "="*60)
    print("All visualizations saved!")
    print("="*60)
    print(f"\n数据集信息: {n_samples} 个样本")
    print(f"Check output directory: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main(use_real_data=True)
