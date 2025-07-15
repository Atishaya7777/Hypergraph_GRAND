import torch
from typing import Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F

from data import ContactDataset
from .loss import clustering_loss_function, clustering_error_function
from utils import visualize_embeddings_tsne


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
              visualize_epochs: list = []) -> Dict:
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

        if len(visualize_epochs) > 0:
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
