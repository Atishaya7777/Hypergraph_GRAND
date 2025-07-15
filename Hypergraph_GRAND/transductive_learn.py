import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

from models import HypergraphGRAND


def clustering_loss_function(embeddings, true_labels, lambda_sep=1.0, lambda_reg=0.01):
    """
    Improved clustering loss with better separation and regularization
    """
    device = embeddings.device
    unique_labels = torch.unique(true_labels)
    num_clusters = len(unique_labels)

    if num_clusters <= 1:
        return torch.tensor(0.0, device=device)

    # Compute centroids for each cluster
    centroids = []
    intra_cluster_loss = torch.tensor(0.0, device=device)
    total_points = 0

    for label in unique_labels:
        mask = (true_labels == label)
        cluster_size = mask.sum()

        if cluster_size <= 1:
            continue

        cluster_points = embeddings[mask]
        centroid = cluster_points.mean(dim=0)
        centroids.append(centroid)

        # Intra-cluster loss: minimize distances within clusters
        distances_sq = torch.sum((cluster_points - centroid) ** 2, dim=1)
        intra_cluster_loss += distances_sq.sum()
        total_points += cluster_size

    # Normalize intra-cluster loss
    if total_points > 0:
        intra_cluster_loss = intra_cluster_loss / total_points

    # Inter-cluster loss: maximize distances between centroids
    inter_cluster_loss = torch.tensor(0.0, device=device)
    if len(centroids) > 1:
        centroids = torch.stack(centroids)
        # Compute all pairwise distances between centroids
        centroid_distances = torch.cdist(centroids, centroids, p=2)
        # Extract upper triangular part (avoid double counting)
        mask = torch.triu(torch.ones_like(
            centroid_distances, dtype=torch.bool), diagonal=1)
        pairwise_distances = centroid_distances[mask]

        # We want to maximize inter-cluster distances, so minimize negative distances
        # Add small epsilon to avoid numerical issues
        inter_cluster_loss = -torch.mean(pairwise_distances) + 1e-6

    # Regularization term to prevent collapse
    embedding_norm = torch.mean(torch.norm(embeddings, dim=1))
    regularization = lambda_reg * (1.0 / (embedding_norm + 1e-8))

    total_loss = intra_cluster_loss + lambda_sep * \
        inter_cluster_loss + regularization

    return total_loss


def clustering_error_function(model_embeddings, true_labels, n_init=20):
    """
    Improved clustering evaluation with better cluster assignment
    """
    # Convert to numpy
    embeddings_np = model_embeddings.detach().cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()

    # Get number of true clusters
    n_clusters = len(np.unique(true_labels_np))

    if n_clusters <= 1:
        return np.eye(1), 0.0, 0.0

    # Apply K-means with multiple initializations for stability
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,
                    n_init=n_init, max_iter=300)

    try:
        predicted_clusters = kmeans.fit_predict(embeddings_np)
    except:
        # Fallback if K-means fails
        predicted_clusters = np.zeros(len(true_labels_np))

    # Compute ARI (this doesn't require label mapping)
    ari = adjusted_rand_score(true_labels_np, predicted_clusters)

    # For accuracy, we need to find optimal mapping
    cm = confusion_matrix(true_labels_np, predicted_clusters)

    # Find optimal assignment using Hungarian algorithm
    try:
        from scipy.optimize import linear_sum_assignment
        # Convert to maximization problem
        cost_matrix = -cm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Create mapping
        cluster_mapping = {col_indices[i]: row_indices[i]
                           for i in range(len(col_indices))}

        # Apply mapping
        mapped_predictions = np.array(
            [cluster_mapping.get(pred, pred) for pred in predicted_clusters])

        # Compute accuracy
        accuracy = accuracy_score(true_labels_np, mapped_predictions)

        # Final confusion matrix
        final_cm = confusion_matrix(true_labels_np, mapped_predictions)

    except ImportError:
        # Fallback if scipy is not available
        print("Warning: scipy not available, using suboptimal cluster assignment")
        accuracy = accuracy_score(true_labels_np, predicted_clusters)
        final_cm = cm

    return final_cm, accuracy, ari


class ContactDataset:
    """
    Dataset class for contact network data
    """

    def __init__(self, data_path: str, dataset_name: str):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.load_data()

    def load_data(self):
        """
        Load hypergraph data from files
        """
        # Load node labels
        node_labels_file = os.path.join(
            self.data_path, f"node-labels-{self.dataset_name}.txt")
        with open(node_labels_file, 'r') as f:
            labels = [int(line.strip()) for line in f]

        self.num_nodes = len(labels)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.num_classes = len(torch.unique(self.labels))

        # Load hyperedges
        hyperedges_file = os.path.join(
            self.data_path, f"hyperedges-{self.dataset_name}.txt")
        hyperedges = []
        max_node_id = -1

        with open(hyperedges_file, 'r') as f:
            for line in f:
                nodes = [int(x) for x in line.strip().split(',')]
                # Convert to 0-indexed if needed
                nodes = [node - 1 if min(nodes) >
                         0 else node for node in nodes]
                hyperedges.append(nodes)
                max_node_id = max(max_node_id, max(nodes))

        # Verify consistency between node labels and hyperedges
        if max_node_id >= self.num_nodes:
            print(f"Warning: Max node ID in hyperedges ({
                  max_node_id}) >= num_nodes ({self.num_nodes})")
            print("Converting node indices to 0-indexed...")

            # Convert all node indices to 0-indexed
            hyperedges = [[node - 1 for node in edge] for edge in hyperedges]
            max_node_id = max(max(edge) for edge in hyperedges)

            if max_node_id >= self.num_nodes:
                raise ValueError(f"Even after conversion, max node ID ({
                                 max_node_id}) >= num_nodes ({self.num_nodes})")

        self.num_hyperedges = len(hyperedges)

        # Create hyperedge index tensor [2, num_edges]
        edge_indices = []
        node_indices = []

        for edge_id, nodes in enumerate(hyperedges):
            for node_id in nodes:
                # Ensure node_id is within bounds
                if node_id < 0 or node_id >= self.num_nodes:
                    raise ValueError(
                        f"Node ID {node_id} is out of bounds [0, {self.num_nodes-1}]")
                edge_indices.append(edge_id)
                node_indices.append(node_id)

        self.hyperedge_index = torch.tensor(
            [edge_indices, node_indices], dtype=torch.long)

        # Load label names (optional)
        label_names_file = os.path.join(
            self.data_path, f"label-names-{self.dataset_name}.txt")
        try:
            with open(label_names_file, 'r') as f:
                self.label_names = [line.strip() for line in f]
        except FileNotFoundError:
            self.label_names = [f"Class_{i}" for i in range(self.num_classes)]

        # Create node features (using one-hot encoding of node indices as simple features)
        self.node_features = torch.eye(self.num_nodes)

        print(f"Dataset {self.dataset_name} loaded:")
        print(f"  - Nodes: {self.num_nodes}")
        print(f"  - Hyperedges: {self.num_hyperedges}")
        print(f"  - Classes: {self.num_classes}")
        print(
            f"  - Mean hyperedge size: {len(node_indices) / self.num_hyperedges:.2f}")
        print(f"  - Max hyperedge size: {max(len(nodes)
              for nodes in hyperedges)}")
        print(f"  - Node ID range: [0, {max_node_id}]")


def create_transductive_split(labels: torch.Tensor, train_ratio: float = 0.6,
                              val_ratio: float = 0.2, random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create transductive split - stratified by class
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    n_nodes = len(labels)
    unique_classes = torch.unique(labels)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    # Stratified split for each class
    for class_id in unique_classes:
        class_indices = torch.where(labels == class_id)[0]
        n_class = len(class_indices)

        # Shuffle indices for this class
        perm = torch.randperm(n_class)
        class_indices = class_indices[perm]

        # Split indices
        train_end = int(train_ratio * n_class)
        val_end = train_end + int(val_ratio * n_class)

        train_mask[class_indices[:train_end]] = True
        val_mask[class_indices[train_end:val_end]] = True
        test_mask[class_indices[val_end:]] = True

    return train_mask, val_mask, test_mask


class HypergraphTrainer:
    """
    Trainer class for HyperGRAND model
    """

    def __init__(self, model: nn.Module, device: torch.device = torch.device('cpu')):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self, data: ContactDataset, train_mask: torch.Tensor,
                    val_mask: torch.Tensor, optimizer: torch.optim.Optimizer,
                    epoch: int = None, visualize: bool = False) -> Tuple[float, float, float, float]:
        """
        Improved training epoch with optional t-SNE visualization
        """
        self.model.train()

        # Move data to device
        x = data.node_features.to(self.device)
        hyperedge_index = data.hyperedge_index.to(self.device)
        labels = data.labels.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)

        # Forward pass on entire graph
        embeddings = self.model(x, hyperedge_index)

        # Normalize embeddings for better clustering
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute training loss
        train_embeddings = embeddings[train_mask]
        train_labels = labels[train_mask]

        # Use improved loss function with higher separation weight
        train_loss = clustering_loss_function(
            train_embeddings, train_labels, lambda_sep=2.0)

        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        optimizer.step()

        # Validation evaluation
        self.model.eval()
        with torch.no_grad():
            val_embeddings = F.normalize(embeddings[val_mask], p=2, dim=1)
            val_labels = labels[val_mask]
            val_loss = clustering_loss_function(
                val_embeddings, val_labels, lambda_sep=2.0)

            # Compute validation metrics
            _, val_accuracy, val_ari = clustering_error_function(
                val_embeddings, val_labels)

            # Optional visualization
            if visualize and epoch is not None:
                # Get predictions for all nodes for visualization
                all_embeddings = F.normalize(embeddings, p=2, dim=1)
                all_labels = labels

                # Get predicted clusters for all nodes
                _, _, _ = clustering_error_function(all_embeddings, all_labels)

                # Get K-means predictions for visualization
                from sklearn.cluster import KMeans
                embeddings_np = all_embeddings.detach().cpu().numpy()
                n_clusters = len(torch.unique(all_labels))
                kmeans = KMeans(n_clusters=n_clusters,
                                random_state=42, n_init=10)
                predicted_labels = kmeans.fit_predict(embeddings_np)

                # Create visualization
                visualize_embeddings_tsne(
                    all_embeddings, all_labels, predicted_labels,
                    data.dataset_name, epoch
                )

        return train_loss.item(), val_loss.item(), val_accuracy, val_ari

    def train(self, data: ContactDataset, train_mask: torch.Tensor, val_mask: torch.Tensor,
              optimizer: torch.optim.Optimizer, num_epochs: int = 200,
              visualize_epochs: list = None) -> Dict:
        """
        Updated training loop with optional visualization at specific epochs

        Args:
            visualize_epochs: List of epoch numbers to create t-SNE visualizations
                            e.g., [1, 50, 100, 150, 200]
        """
        print(f"Training on {data.dataset_name} dataset:")
        print(f"  - Train nodes: {train_mask.sum().item()}")
        print(f"  - Val nodes: {val_mask.sum().item()}")
        print(f"  - Test nodes: {(~train_mask & ~val_mask).sum().item()}")

        if visualize_epochs:
            print(
                f"  - Will create t-SNE visualizations at epochs: {visualize_epochs}")

        print("-" * 50)

        best_val_ari = -1.0
        best_epoch = 0

        # Add ARI tracking
        self.val_aris = []

        for epoch in range(num_epochs):
            # Check if we should visualize this epoch
            should_visualize = visualize_epochs and (
                epoch + 1) in visualize_epochs

            train_loss, val_loss, val_accuracy, val_ari = self.train_epoch(
                data, train_mask, val_mask, optimizer, epoch + 1, should_visualize
            )

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            self.val_aris.append(val_ari)

            # Track best model based on ARI
            if val_ari > best_val_ari:
                best_val_ari = val_ari
                best_epoch = epoch

            # Print progress every 2 epochs or at the end
            if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_accuracy:.4f} | "
                      f"Val ARI: {val_ari:.4f}")

        print(f"\nBest validation ARI: {
              best_val_ari:.4f} at epoch {best_epoch + 1}")

        return {
            'best_val_accuracy': self.val_accuracies[best_epoch],
            'best_val_ari': best_val_ari,
            'best_epoch': best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_aris': self.val_aris
        }

    def evaluate(self, data: ContactDataset, test_mask: torch.Tensor, visualize: bool = True) -> Dict:
        """
            Updated evaluation with optional t-SNE visualization
            """
        self.model.eval()

        x = data.node_features.to(self.device)
        hyperedge_index = data.hyperedge_index.to(self.device)
        labels = data.labels.to(self.device)
        test_mask = test_mask.to(self.device)

        with torch.no_grad():
            embeddings = self.model(x, hyperedge_index)
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

            test_embeddings = embeddings[test_mask]
            test_labels = labels[test_mask]

            test_loss = clustering_loss_function(
                test_embeddings, test_labels, lambda_sep=2.0)
            confusion_matrix, test_accuracy, test_ari = clustering_error_function(
                test_embeddings, test_labels)

            # Optional visualization of test results
            if visualize:
                # Get predictions for all test nodes
                from sklearn.cluster import KMeans
                embeddings_np = test_embeddings.detach().cpu().numpy()
                n_clusters = len(torch.unique(test_labels))
                kmeans = KMeans(n_clusters=n_clusters,
                                random_state=42, n_init=10)
                predicted_labels = kmeans.fit_predict(embeddings_np)

                # Create visualization
                visualize_embeddings_tsne(
                    test_embeddings, test_labels, predicted_labels,
                    f"{data.dataset_name}_test", "final"
                )

        return {
            'test_loss': test_loss.item(),
            'test_accuracy': test_accuracy,
            'test_ari': test_ari,
            'confusion_matrix': confusion_matrix
        }


def visualize_embeddings_tsne(embeddings, true_labels, predicted_labels, dataset_name, epoch=None):
    """
    Create t-SNE visualization of embeddings with true and predicted labels

    Args:
        embeddings: torch.Tensor of shape [num_nodes, hidden_dim]
        true_labels: torch.Tensor of true class labels
        predicted_labels: numpy array of predicted cluster labels
        dataset_name: str, name of the dataset
        epoch: int, current epoch (optional)
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings

    if isinstance(true_labels, torch.Tensor):
        true_labels_np = true_labels.cpu().numpy()
    else:
        true_labels_np = true_labels

    # Apply t-SNE
    print(f"Applying t-SNE to embeddings...")
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(embeddings_np)-1))
    embeddings_2d = tsne.fit_transform(embeddings_np)

    # Create visualization
    plt.figure(figsize=(15, 6))

    # Define colors for classes
    colors = ['red', 'blue', 'green', 'orange', 'purple',
              'brown', 'pink', 'gray', 'olive', 'cyan']
    num_classes = len(np.unique(true_labels_np))

    # Plot 1: True labels
    plt.subplot(1, 2, 1)
    for i in range(num_classes):
        mask = true_labels_np == i
        if mask.sum() > 0:
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=colors[i % len(colors)], label=f'Class {i}',
                        alpha=0.7, s=30)

    plt.title(f'True Labels - {dataset_name}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Plot 2: Predicted clusters
    plt.subplot(1, 2, 2)
    num_predicted_classes = len(np.unique(predicted_labels))
    for i in range(num_predicted_classes):
        mask = predicted_labels == i
        if mask.sum() > 0:
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=colors[i % len(colors)], label=f'Cluster {i}',
                        alpha=0.7, s=30)

    title = f'Predicted Clusters - {dataset_name}'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save with epoch info if provided
    filename = f'hypergrand_tsne_{dataset_name.replace("-", "_")}'
    if epoch is not None:
        filename += f'_epoch_{epoch}'
    filename += '.png'

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"t-SNE visualization saved as '{filename}'")


def approach_1_transductive_learning():
    """
    Approach 1: Transductive learning on each dataset individually
    """
    print("="*60)
    print("APPROACH 1: TRANSDUCTIVE LEARNING")
    print("="*60)

    # Dataset paths - adjust these to your actual paths
    datasets = {
        'contact-high-school': 'datasets/contact-high-school',
        'contact-primary-school': 'datasets/contact-primary-school'
    }

    results = {}

    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")

        # Load dataset
        data = ContactDataset(dataset_path, dataset_name)

        # Create transductive split
        train_mask, val_mask, test_mask = create_transductive_split(
            data.labels)

        # Initialize model
        model = HypergraphGRAND(
            input_dim=data.num_nodes,  # Using node features as input
            hidden_dim=16,
            num_layers=3,
            alpha=0.1,
            dropout=0.1
        )

        # Initialize trainer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = HypergraphTrainer(model, device)

        # Setup optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, weight_decay=1e-5)

        # Train model
        train_results = trainer.train(
            data, train_mask, val_mask, optimizer, num_epochs=10)

        # Evaluate on test set
        test_results = trainer.evaluate(data, test_mask)

        # Store results
        results[dataset_name] = {
            'train_results': train_results,
            'test_results': test_results,
            'dataset_stats': {
                'num_nodes': data.num_nodes,
                'num_hyperedges': data.num_hyperedges,
                'num_classes': data.num_classes
            }
        }

        print(f"\nFinal Results for {dataset_name}:")
        print(
            f"  - Best Val Accuracy: {train_results['best_val_accuracy']:.4f}")
        print(f"  - Test Accuracy: {test_results['test_accuracy']:.4f}")
        print(f"  - Test Loss: {test_results['test_loss']:.4f}")

    return results


def approach_2_transfer_learning():
    """
    Approach 2: Transfer learning from primary school to high school
    """
    print("\n" + "="*60)
    print("APPROACH 2: TRANSFER LEARNING")
    print("="*60)

    # Load datasets
    print("\nLoading datasets...")
    source_data = ContactDataset(
        'datasets/contact-primary-school', 'contact-primary-school')
    target_data = ContactDataset(
        'datasets/contact-high-school', 'contact-high-school')

    # Create splits
    source_train_mask, source_val_mask, source_test_mask = create_transductive_split(
        source_data.labels)
    target_train_mask, target_val_mask, target_test_mask = create_transductive_split(
        target_data.labels)

    # Phase 1: Pre-train on primary school
    print(f"\n{'='*20} PHASE 1: PRE-TRAINING ON PRIMARY SCHOOL {'='*20}")

    # Initialize model for source domain
    source_model = HypergraphGRAND(
        input_dim=source_data.num_nodes,
        hidden_dim=64,
        num_layers=3,
        alpha=0.05,  # Lower alpha for denser primary school network
        dropout=0.1
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_trainer = HypergraphTrainer(source_model, device)

    # Pre-training with Adam optimizer
    source_optimizer = torch.optim.Adam(
        source_model.parameters(), lr=0.01, weight_decay=1e-5)
    source_results = source_trainer.train(source_data, source_train_mask, source_val_mask,
                                          source_optimizer, num_epochs=10)

    # Evaluate source model
    source_test_results = source_trainer.evaluate(
        source_data, source_test_mask)

    print(f"\nSource Domain Results:")
    print(f"  - Best Val Accuracy: {source_results['best_val_accuracy']:.4f}")
    print(f"  - Test Accuracy: {source_test_results['test_accuracy']:.4f}")

    # Phase 2: Transfer to high school
    print(f"\n{'='*20} PHASE 2: TRANSFER TO HIGH SCHOOL {'='*20}")

    # Initialize target model with different input dimension
    target_model = HypergraphGRAND(
        input_dim=target_data.num_nodes,
        hidden_dim=64,
        num_layers=3,
        alpha=0.1,  # Higher alpha for sparser high school network
        dropout=0.1
    )

    # Transfer weights from source model (except input layer)
    target_state_dict = target_model.state_dict()
    source_state_dict = source_model.state_dict()

    # Transfer all weights except input_transform (different dimensions)
    for name, param in source_state_dict.items():
        if name in target_state_dict and 'input_transform' not in name:
            target_state_dict[name].copy_(param)

    print("Transferred weights from source model (except input layer)")

    # Fine-tuning with different optimizers
    target_trainer = HypergraphTrainer(target_model, device)

    # Strategy 1: Fine-tuning with SGD (lower learning rate)
    print("\nFine-tuning with SGD optimizer...")
    target_optimizer = torch.optim.SGD(
        target_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    target_results = target_trainer.train(target_data, target_train_mask, target_val_mask,
                                          target_optimizer, num_epochs=100)

    # Evaluate target model
    target_test_results = target_trainer.evaluate(
        target_data, target_test_mask)

    print(f"\nTarget Domain Results (Transfer Learning):")
    print(f"  - Best Val Accuracy: {target_results['best_val_accuracy']:.4f}")
    print(f"  - Test Accuracy: {target_test_results['test_accuracy']:.4f}")

    # Baseline: Train from scratch on target domain
    print(f"\n{'='*20} BASELINE: TRAIN FROM SCRATCH ON HIGH SCHOOL {'='*20}")

    baseline_model = HypergraphGRAND(
        input_dim=target_data.num_nodes,
        hidden_dim=64,
        num_layers=3,
        alpha=0.1,
        dropout=0.1
    )

    baseline_trainer = HypergraphTrainer(baseline_model, device)
    baseline_optimizer = torch.optim.Adam(
        baseline_model.parameters(), lr=0.01, weight_decay=1e-5)
    baseline_results = baseline_trainer.train(target_data, target_train_mask, target_val_mask,
                                              baseline_optimizer, num_epochs=200)

    baseline_test_results = baseline_trainer.evaluate(
        target_data, target_test_mask)

    print(f"\nBaseline Results (From Scratch):")
    print(
        f"  - Best Val Accuracy: {baseline_results['best_val_accuracy']:.4f}")
    print(f"  - Test Accuracy: {baseline_test_results['test_accuracy']:.4f}")

    # Compare results
    print(f"\n{'='*20} TRANSFER LEARNING COMPARISON {'='*20}")
    print(f"Transfer Learning Test Accuracy: {
          target_test_results['test_accuracy']:.4f}")
    print(f"From Scratch Test Accuracy: {
          baseline_test_results['test_accuracy']:.4f}")
    improvement = target_test_results['test_accuracy'] - \
        baseline_test_results['test_accuracy']
    print(f"Improvement: {improvement:+.4f}")

    return {
        'source_results': source_results,
        'source_test_results': source_test_results,
        'target_results': target_results,
        'target_test_results': target_test_results,
        'baseline_results': baseline_results,
        'baseline_test_results': baseline_test_results,
        'improvement': improvement
    }


def main():
    """
    Main function to run both approaches
    """
    print("HyperGRAND: Hypergraph Graph Neural Diffusion")
    print("Clustering-based learning on contact network datasets")
    print("="*60)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Approach 1: Transductive learning
        print("Starting Approach 1: Transductive Learning...")
        transductive_results = approach_1_transductive_learning()

        # Approach 2: Transfer learning
        print("\nStarting Approach 2: Transfer Learning...")
        transfer_results = approach_2_transfer_learning()

        # Print final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)

        print("\nApproach 1 - Transductive Learning Results:")
        for dataset_name, results in transductive_results.items():
            print(f"\n{dataset_name}:")
            print(f"  Best Val Accuracy: {
                  results['train_results']['best_val_accuracy']:.4f}")
            print(f"  Test Accuracy: {
                  results['test_results']['test_accuracy']:.4f}")
            print(f"  Test Loss: {results['test_results']['test_loss']:.4f}")

        print("\nApproach 2 - Transfer Learning Results:")
        print(f"  Source (Primary School) Test Accuracy: {
              transfer_results['source_test_results']['test_accuracy']:.4f}")
        print(f"  Target (High School) with Transfer: {
              transfer_results['target_test_results']['test_accuracy']:.4f}")
        print(f"  Target (High School) from Scratch: {
              transfer_results['baseline_test_results']['test_accuracy']:.4f}")
        print(f"  Transfer Learning Improvement: {
              transfer_results['improvement']:+.4f}")

        # Save results to file
        results_summary = {
            'transductive_results': transductive_results,
            'transfer_results': transfer_results,
            'timestamp': str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
        }

        with open('hypergrand_results.json', 'w') as f:
            # Convert tensors to lists for JSON serialization
            import json
            json.dump(results_summary, f, indent=2, default=str)

        print(f"\nResults saved to 'hypergrand_results.json'")

    except FileNotFoundError as e:
        print(f"Error: Could not find dataset files. Please check that the datasets directory exists.")
        print(f"Expected structure:")
        print(f"  datasets/contact-high-school/")
        print(f"    - node-labels-contact-high-school.txt")
        print(f"    - hyperedges-contact-high-school.txt")
        print(f"    - label-names-contact-high-school.txt")
        print(f"  datasets/contact-primary-school/")
        print(f"    - node-labels-contact-primary-school.txt")
        print(f"    - hyperedges-contact-primary-school.txt")
        print(f"    - label-names-contact-primary-school.txt")
        print(f"\nError details: {e}")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
