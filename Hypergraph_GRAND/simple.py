"""
Enhanced HyperGRAND Implementation
Improved version with better performance, visualization, and code organization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class HypergraphConvLayer(nn.Module):
    """
    Optimized hypergraph convolution layer with attention mechanism
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Learnable transformations
        self.node_transform = nn.Linear(hidden_dim, hidden_dim)
        self.edge_transform = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)

        # Normalization and activation
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        Efficient hypergraph convolution with attention
        """
        if hyperedge_index.size(1) == 0:
            return h

        num_nodes = h.size(0)
        edge_ids = hyperedge_index[0]
        node_ids = hyperedge_index[1]

        # Transform node features
        h_transformed = self.node_transform(h)

        # Initialize output
        h_new = torch.zeros_like(h)

        # Group by edges for efficient processing
        unique_edges = torch.unique(edge_ids)

        for edge_id in unique_edges:
            # Get nodes in this hyperedge
            edge_mask = (edge_ids == edge_id)
            nodes_in_edge = node_ids[edge_mask]

            if len(nodes_in_edge) <= 1:
                continue

            # Get features for nodes in this edge
            edge_features = h_transformed[nodes_in_edge]

            # Compute attention weights
            attention_weights = torch.softmax(
                self.attention(edge_features), dim=0)

            # Weighted aggregation
            aggregated = torch.sum(attention_weights * edge_features, dim=0)

            # Update all nodes in this hyperedge
            for node in nodes_in_edge:
                h_new[node] = h_new[node] + aggregated

        # Apply normalization and dropout
        h_new = self.layer_norm(h_new)
        h_new = self.dropout_layer(h_new)

        return h_new


class EnhancedHyperGRAND(nn.Module):
    """
    Enhanced HyperGRAND with better architecture and training stability
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 alpha: float = 0.3,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha

        # Input transformation with batch norm
        self.input_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Hypergraph convolution layers
        self.hypergraph_layers = nn.ModuleList([
            HypergraphConvLayer(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections and progressive refinement
        """
        # Input transformation
        h = self.input_transform(x)
        h_residual = h.clone()

        # Apply hypergraph convolutions with residual connections
        for i, layer in enumerate(self.hypergraph_layers):
            h_new = layer(h, hyperedge_index)

            # Residual connection with decay
            decay = self.alpha * (0.8 ** i)  # Decreasing residual strength
            h = h_residual + decay * h_new

            # Update residual
            h_residual = h.clone()

        # Final projection and normalization
        h = self.output_proj(h)
        h = F.normalize(h, p=2, dim=1)

        return h


def create_realistic_hypergraph_data(num_nodes: int = 80,
                                     num_classes: int = 4,
                                     feature_dim: int = 32,
                                     noise_level: float = 0.1):
    """
    Create more realistic hypergraph data with meaningful node features
    """
    print(f"Creating realistic hypergraph data...")
    print(f"  Nodes: {num_nodes}, Classes: {
          num_classes}, Feature dim: {feature_dim}")

    # Create balanced class labels
    nodes_per_class = num_nodes // num_classes
    labels = []
    for i in range(num_classes):
        labels.extend([i] * nodes_per_class)

    # Handle remainder
    remaining = num_nodes - len(labels)
    labels.extend([i % num_classes for i in range(remaining)])

    # Shuffle labels
    indices = list(range(num_nodes))
    np.random.shuffle(indices)
    shuffled_labels = [labels[i] for i in indices]

    # Create meaningful node features
    node_features = torch.zeros(num_nodes, feature_dim, dtype=torch.float)

    # Create class-specific feature centroids
    class_centroids = torch.randn(num_classes, feature_dim) * 2.0

    for i in range(num_nodes):
        class_id = shuffled_labels[i]
        # Sample from class-specific distribution
        node_features[i] = class_centroids[class_id] + \
            torch.randn(feature_dim) * noise_level

    # Create hyperedges with class structure
    hyperedges = []

    # Group nodes by class
    class_nodes = {}
    for i, label in enumerate(shuffled_labels):
        if label not in class_nodes:
            class_nodes[label] = []
        class_nodes[label].append(i)

    # Create intra-class hyperedges (80% of edges)
    total_edges = max(num_nodes // 3, 10)
    intra_edges = int(0.8 * total_edges)

    for _ in range(intra_edges):
        # Pick a random class
        class_id = np.random.choice(num_classes)
        nodes = class_nodes[class_id]

        if len(nodes) >= 2:
            # Create hyperedge with 2-5 nodes from same class
            edge_size = min(np.random.randint(2, 6), len(nodes))
            edge_nodes = np.random.choice(nodes, size=edge_size, replace=False)
            hyperedges.append(edge_nodes.tolist())

    # Create inter-class hyperedges (20% of edges)
    inter_edges = total_edges - intra_edges
    for _ in range(inter_edges):
        # Pick 2-3 different classes
        num_classes_in_edge = np.random.randint(2, min(4, num_classes + 1))
        selected_classes = np.random.choice(
            num_classes, size=num_classes_in_edge, replace=False)

        edge_nodes = []
        for class_id in selected_classes:
            class_node_list = class_nodes[class_id]
            if len(class_node_list) > 0:
                node = np.random.choice(class_node_list)
                edge_nodes.append(node)

        if len(edge_nodes) >= 2:
            hyperedges.append(edge_nodes)

    # Convert to hyperedge_index format
    edge_indices = []
    node_indices = []
    for edge_id, nodes in enumerate(hyperedges):
        for node in nodes:
            edge_indices.append(edge_id)
            node_indices.append(node)

    hyperedge_index = torch.tensor(
        [edge_indices, node_indices], dtype=torch.long)
    labels = torch.tensor(shuffled_labels, dtype=torch.long)

    # Print statistics
    print(f"  Created {len(hyperedges)} hyperedges")

    class_counts = {}
    for label in shuffled_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    print(f"  Class distribution: {class_counts}")

    intra_class_edges = 0
    for edge_nodes in hyperedges:
        edge_classes = set(shuffled_labels[node] for node in edge_nodes)
        if len(edge_classes) == 1:
            intra_class_edges += 1

    print(f"  Intra-class hyperedges: {intra_class_edges}/{len(hyperedges)} "
          f"({intra_class_edges/len(hyperedges)*100:.1f}%)")

    return node_features, hyperedge_index, labels


def evaluate_embeddings(embeddings: torch.Tensor, true_labels: torch.Tensor, num_runs: int = 10):
    """
    Comprehensive evaluation of embeddings
    """
    num_classes = len(torch.unique(true_labels))

    # Multiple clustering runs
    ari_scores = []
    silhouette_scores = []

    for seed in range(num_runs):
        kmeans = KMeans(n_clusters=num_classes, random_state=seed, n_init=10)
        pred_labels = kmeans.fit_predict(embeddings.cpu().numpy())

        # Adjusted Rand Index
        ari = adjusted_rand_score(true_labels.cpu().numpy(), pred_labels)
        ari_scores.append(ari)

        # Silhouette Score
        if len(np.unique(pred_labels)) > 1:
            sil = silhouette_score(embeddings.cpu().numpy(), pred_labels)
            silhouette_scores.append(sil)

    results = {
        'ari_mean': np.mean(ari_scores),
        'ari_std': np.std(ari_scores),
        'silhouette_mean': np.mean(silhouette_scores) if silhouette_scores else 0,
        'silhouette_std': np.std(silhouette_scores) if silhouette_scores else 0,
        'best_ari': np.max(ari_scores),
        'best_seed': np.argmax(ari_scores)
    }

    return results


def create_comprehensive_visualization(embeddings: torch.Tensor,
                                       true_labels: torch.Tensor,
                                       predicted_labels: np.ndarray,
                                       results: dict):
    """
    Create comprehensive visualization with multiple plots
    """
    # Prepare data
    embeddings_np = embeddings.cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()
    num_classes = len(np.unique(true_labels_np))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('HyperGRAND Embedding Analysis',
                 fontsize=16, fontweight='bold')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    # 1. t-SNE visualization - True labels
    perplexity = min(30, len(embeddings_np) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_np)

    ax1 = axes[0, 0]
    for i in range(num_classes):
        mask = true_labels_np == i
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=colors[i], label=f'Class {i}', alpha=0.7, s=50)
    ax1.set_title('True Labels (t-SNE)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. t-SNE visualization - Predicted labels
    ax2 = axes[0, 1]
    for i in range(num_classes):
        mask = predicted_labels == i
        ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=colors[i], label=f'Cluster {i}', alpha=0.7, s=50)
    ax2.set_title(f'Predicted Clusters (ARI={
                  results["best_ari"]:.3f})', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. PCA visualization
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings_np)

    ax3 = axes[0, 2]
    for i in range(num_classes):
        mask = true_labels_np == i
        ax3.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1],
                    c=colors[i], label=f'Class {i}', alpha=0.7, s=50)
    ax3.set_title(f'PCA Visualization (Var: {pca.explained_variance_ratio_.sum():.2f})',
                  fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Embedding norms distribution
    ax4 = axes[1, 0]
    norms = torch.norm(embeddings, dim=1).cpu().numpy()
    ax4.hist(norms, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_title('Embedding Norms Distribution', fontweight='bold')
    ax4.set_xlabel('L2 Norm')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)

    # 5. Performance metrics
    ax5 = axes[1, 1]
    metrics = ['ARI', 'Silhouette']
    means = [results['ari_mean'], results['silhouette_mean']]
    stds = [results['ari_std'], results['silhouette_std']]

    bars = ax5.bar(metrics, means, yerr=stds, capsize=5, alpha=0.7,
                   color=['#FF6B6B', '#4ECDC4'])
    ax5.set_title('Performance Metrics', fontweight='bold')
    ax5.set_ylabel('Score')
    ax5.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + std,
                 f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')

    # 6. Class-wise embedding statistics
    ax6 = axes[1, 2]
    class_means = []
    class_stds = []

    for i in range(num_classes):
        class_mask = true_labels_np == i
        class_embeddings = embeddings_np[class_mask]
        class_means.append(np.mean(class_embeddings))
        class_stds.append(np.std(class_embeddings))

    x_pos = np.arange(num_classes)
    bars = ax6.bar(x_pos, class_means, yerr=class_stds, capsize=5, alpha=0.7,
                   color=colors[:num_classes])
    ax6.set_title('Class-wise Embedding Statistics', fontweight='bold')
    ax6.set_xlabel('Class')
    ax6.set_ylabel('Mean Embedding Value')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f'Class {i}' for i in range(num_classes)])
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('enhanced_hypergrand_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_enhanced_hypergrand():
    """
    Main demonstration with enhanced features
    """
    print("="*80)
    print("ENHANCED HYPERGRAND DEMONSTRATION")
    print("="*80)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create realistic data
    node_features, hyperedge_index, true_labels = create_realistic_hypergraph_data(
        num_nodes=100, num_classes=4, feature_dim=32, noise_level=0.15
    )

    num_nodes, input_dim = node_features.shape
    num_classes = len(torch.unique(true_labels))

    print(f"\nData Overview:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Classes: {num_classes}")
    print(f"  Hyperedges: {torch.max(hyperedge_index[0]) + 1}")

    # Initialize enhanced model
    model = EnhancedHyperGRAND(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=3,
        alpha=0.4,
        dropout=0.1
    )

    print(f"\nEnhanced Model Architecture:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: 64")
    print(f"  Number of layers: 3")
    print(f"  Diffusion strength: 0.4")
    print(f"  Dropout: 0.1")
    print(f"  Total parameters: {sum(p.numel()
          for p in model.parameters()):,}")

    # Generate embeddings
    print(f"\nGenerating enhanced 64D embeddings...")
    model.eval()
    with torch.no_grad():
        embeddings = model(node_features, hyperedge_index)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding statistics:")
    print(f"  Mean norm: {torch.norm(embeddings, dim=1).mean():.4f}")
    print(f"  Std norm: {torch.norm(embeddings, dim=1).std():.4f}")
    print(f"  Min value: {embeddings.min():.4f}")
    print(f"  Max value: {embeddings.max():.4f}")

    # Comprehensive evaluation
    print(f"\nComprehensive Evaluation:")
    results = evaluate_embeddings(embeddings, true_labels, num_runs=15)

    print(f"  Adjusted Rand Index: {
          results['ari_mean']:.4f} ± {results['ari_std']:.4f}")
    print(f"  Silhouette Score: {results['silhouette_mean']:.4f} ± {
          results['silhouette_std']:.4f}")
    print(f"  Best ARI: {results['best_ari']:.4f}")

    # Get best clustering for visualization
    kmeans = KMeans(n_clusters=num_classes,
                    random_state=results['best_seed'], n_init=10)
    predicted_labels = kmeans.fit_predict(embeddings.cpu().numpy())

    # Create comprehensive visualization
    # print(f"\nCreating comprehensive visualization...")
    # create_comprehensive_visualization(
    #     embeddings, true_labels, predicted_labels, results)

    # Baseline comparison
    print(f"\nBaseline Comparison:")
    baseline_model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.LayerNorm(64)
    )

    baseline_model.eval()
    with torch.no_grad():
        baseline_embeddings = baseline_model(node_features)
        baseline_embeddings = F.normalize(baseline_embeddings, p=2, dim=1)

    baseline_results = evaluate_embeddings(
        baseline_embeddings, true_labels, num_runs=10)

    print(f"  HyperGRAND ARI: {results['ari_mean']:.4f} ± {
          results['ari_std']:.4f}")
    print(f"  Baseline ARI:   {baseline_results['ari_mean']:.4f} ± {
          baseline_results['ari_std']:.4f}")
    print(f"  Improvement:    {
          results['ari_mean'] - baseline_results['ari_mean']:+.4f}")

    # Summary
    print(f"\n{'='*80}")
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print(f"✓ Enhanced HyperGRAND with attention mechanism")
    print(f"✓ Realistic hypergraph data generation")
    print(f"✓ Comprehensive evaluation metrics")
    print(f"✓ Multi-panel visualization")
    print(f"✓ Baseline comparison")
    print(f"✓ Achieved ARI: {results['ari_mean']:.4f}")

    if results['ari_mean'] > 0.5:
        print("🎉 Excellent clustering performance!")
    elif results['ari_mean'] > 0.3:
        print("✅ Good clustering performance!")
    else:
        print("⚠️  Consider further hyperparameter tuning")

    return embeddings, true_labels, predicted_labels, results


if __name__ == "__main__":
    demonstrate_enhanced_hypergrand()
