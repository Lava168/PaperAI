"""
Demo Script for Atlas-Guided Query Mechanism

This script demonstrates the key components without requiring actual data.
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

from models import AtlasGuidedBrainModel
from losses import GeodesicAttentionLoss


def demo_model_forward():
    """Demonstrate model forward pass with synthetic data"""
    print("=" * 60)
    print("Demo: Model Forward Pass")
    print("=" * 60)
    
    # Create model
    model = AtlasGuidedBrainModel(
        in_channels=1,
        base_channels=16,
        feature_dim=64,
        num_queries=16,
        num_regions=20,
        num_heads=4,
        num_classes=3,
        num_encoder_stages=3,
        num_query_layers=2,
    )
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create synthetic input
    batch_size = 2
    volume_size = 64
    
    # Synthetic MRI volume (random noise with some structure)
    x = torch.randn(batch_size, 1, volume_size, volume_size, volume_size)
    
    # Synthetic segmentation (random regions)
    seg = torch.randint(0, 20, (batch_size, volume_size, volume_size, volume_size))
    
    print(f"\nInput shape: {x.shape}")
    print(f"Segmentation shape: {seg.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x, segmentation=seg, return_attention=True)
    
    print(f"\nOutput shapes:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Query features: {outputs['query_features'].shape}")
    print(f"  Number of attention maps: {len(outputs['attention_maps']) if outputs['attention_maps'] else 0}")
    
    # Show predictions
    probs = torch.softmax(outputs['logits'], dim=1)
    print(f"\nPrediction probabilities (CN, MCI, AD):")
    for i in range(batch_size):
        print(f"  Sample {i}: {probs[i].numpy()}")
    
    return model, outputs


def demo_geodesic_loss():
    """Demonstrate Geodesic Loss computation"""
    print("\n" + "=" * 60)
    print("Demo: Geodesic Loss")
    print("=" * 60)
    
    # Create loss function
    geo_loss = GeodesicAttentionLoss(
        geodesic_weight=1.0,
        atlas_prior_weight=0.5,
        entropy_weight=0.1,
        sparsity_weight=0.01,
        max_geodesic_distance=50.0,
    )
    
    # Create synthetic attention weights
    batch_size = 4
    num_queries = 16
    num_keys = 64
    
    # Random attention weights (softmax normalized)
    attention = torch.softmax(torch.randn(batch_size, num_queries, num_keys), dim=-1)
    
    print(f"Attention shape: {attention.shape}")
    
    # Synthetic geodesic distances (larger for farther positions)
    positions = torch.randn(batch_size, num_keys, 3) * 50  # Random 3D positions
    geodesic_distances = torch.cdist(positions, positions)  # Use Euclidean as proxy
    
    print(f"Distance matrix shape: {geodesic_distances.shape}")
    print(f"Distance range: [{geodesic_distances.min():.1f}, {geodesic_distances.max():.1f}] mm")
    
    # Compute loss
    loss, components = geo_loss(
        attention,
        geodesic_distances=geodesic_distances[:, :num_queries, :],
        return_components=True,
    )
    
    print(f"\nLoss components:")
    for name, value in components.items():
        print(f"  {name}: {value.item():.4f}")
    print(f"Total loss: {loss.item():.4f}")
    
    return geo_loss, loss


def demo_attention_distance_relationship():
    """Show how geodesic loss encourages distance-based attention"""
    print("\n" + "=" * 60)
    print("Demo: Attention-Distance Relationship")
    print("=" * 60)
    
    num_positions = 100
    
    # Create positions on a 1D line (for simplicity)
    positions = torch.linspace(0, 100, num_positions).unsqueeze(1)  # (N, 1)
    
    # Compute distances
    distances = torch.cdist(positions, positions)  # (N, N)
    
    # Create "ideal" attention: higher for closer positions
    ideal_attention = torch.exp(-distances / 20.0)
    ideal_attention = ideal_attention / ideal_attention.sum(dim=-1, keepdim=True)
    
    # Create "bad" attention: uniform (ignores distance)
    bad_attention = torch.ones(num_positions, num_positions)
    bad_attention = bad_attention / bad_attention.sum(dim=-1, keepdim=True)
    
    # Compute geodesic loss for both
    geo_loss = GeodesicAttentionLoss(
        geodesic_weight=1.0,
        atlas_prior_weight=0.0,
        entropy_weight=0.0,
        sparsity_weight=0.0,
        max_geodesic_distance=50.0,
    )
    
    ideal_loss = geo_loss(
        ideal_attention.unsqueeze(0),
        geodesic_distances=distances.unsqueeze(0),
    )
    
    bad_loss = geo_loss(
        bad_attention.unsqueeze(0),
        geodesic_distances=distances.unsqueeze(0),
    )
    
    print(f"Geodesic loss (ideal attention): {ideal_loss.item():.4f}")
    print(f"Geodesic loss (uniform attention): {bad_loss.item():.4f}")
    print(f"\nRatio: {bad_loss.item() / ideal_loss.item():.2f}x higher for uniform attention")
    print("\nThis shows the loss correctly penalizes attending to distant positions!")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Distance matrix
    axes[0].imshow(distances.numpy(), cmap='viridis')
    axes[0].set_title('Distance Matrix')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Position')
    plt.colorbar(axes[0].images[0], ax=axes[0], label='Distance')
    
    # Ideal attention
    axes[1].imshow(ideal_attention.numpy(), cmap='hot')
    axes[1].set_title(f'Ideal Attention\n(Loss: {ideal_loss.item():.4f})')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(axes[1].images[0], ax=axes[1], label='Attention')
    
    # Uniform attention
    axes[2].imshow(bad_attention.numpy(), cmap='hot')
    axes[2].set_title(f'Uniform Attention\n(Loss: {bad_loss.item():.4f})')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    plt.colorbar(axes[2].images[0], ax=axes[2], label='Attention')
    
    plt.tight_layout()
    plt.savefig('geodesic_loss_demo.png', dpi=150)
    print("\nSaved visualization to 'geodesic_loss_demo.png'")
    
    return fig


def demo_atlas_regions():
    """Show example brain atlas region labels"""
    print("\n" + "=" * 60)
    print("Demo: Brain Atlas Regions (FreeSurfer)")
    print("=" * 60)
    
    # Key regions for Alzheimer's Disease research
    ad_related_regions = {
        # Medial Temporal Lobe (earliest affected)
        "Left Hippocampus": 17,
        "Right Hippocampus": 53,
        "Left Entorhinal": 1006,
        "Right Entorhinal": 2006,
        "Left Parahippocampal": 1016,
        "Right Parahippocampal": 2016,
        
        # Temporal Neocortex
        "Left Inferior Temporal": 1009,
        "Right Inferior Temporal": 2009,
        "Left Middle Temporal": 1015,
        "Right Middle Temporal": 2015,
        
        # Posterior Regions
        "Left Posterior Cingulate": 1023,
        "Right Posterior Cingulate": 2023,
        "Left Precuneus": 1025,
        "Right Precuneus": 2025,
        
        # Frontal (later stages)
        "Left Medial Orbitofrontal": 1014,
        "Right Medial Orbitofrontal": 2014,
    }
    
    print("\nAD-Related Brain Regions (FreeSurfer Labels):")
    print("-" * 50)
    
    # Group by hemisphere
    left_regions = {k: v for k, v in ad_related_regions.items() if k.startswith("Left")}
    right_regions = {k: v for k, v in ad_related_regions.items() if k.startswith("Right")}
    
    print("\nLeft Hemisphere:")
    for name, label in left_regions.items():
        print(f"  {name}: {label}")
    
    print("\nRight Hemisphere:")
    for name, label in right_regions.items():
        print(f"  {name}: {label}")
    
    print("\n" + "-" * 50)
    print("These regions are used to compute Atlas Prior and Geodesic Loss.")
    print("The model learns to focus attention on anatomically relevant regions.")


def main():
    """Run all demos"""
    print("\n" + "#" * 60)
    print("# Atlas-Guided Query Mechanism Demo")
    print("# with Geodesic Loss")
    print("#" * 60)
    
    # Demo 1: Model forward pass
    model, outputs = demo_model_forward()
    
    # Demo 2: Geodesic loss
    geo_loss, loss = demo_geodesic_loss()
    
    # Demo 3: Attention-distance relationship
    fig = demo_attention_distance_relationship()
    
    # Demo 4: Atlas regions
    demo_atlas_regions()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. The model uses learnable query tokens guided by brain atlas")
    print("2. Geodesic Loss encourages attention to follow brain anatomy")
    print("3. Closer regions (in geodesic distance) get higher attention")
    print("4. This improves interpretability and biological plausibility")


if __name__ == '__main__':
    main()
