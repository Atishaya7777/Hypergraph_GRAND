import mlflow
import torch
from typing import Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from data import ContactDataset
from .loss import clustering_loss_function, clustering_error_function
from utils import visualize_embeddings_tsne, plot_metric_over_epochs


class HypergraphTrainer:
    """
    Trainer class for the HyperGRAND model, providing training and evaluation routines.
    """

    def __init__(self, model: nn.Module, device: torch.device = torch.device('cpu')):
        """
        Args:
            model (nn.Module): The HyperGRAND model to be trained.
            device (torch.device): The device to run the model on (CPU or CUDA).
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self,
                    data: ContactDataset,
                    train_mask: torch.Tensor,
                    val_mask: torch.Tensor,
                    optimizer: torch.optim.Optimizer,
                    epoch: int = None,
                    visualize: bool = False
                    ) -> Tuple[float, float, float, float]:
        """
        Trains the model for one epoch.

        Args:
            data: Dataset containing features, labels, and hyperedges.
            train_mask: Boolean mask for training nodes.
            val_mask: Boolean mask for validation nodes.
            optimizer: Optimizer used for model training.
            epoch: Current epoch number (used for visualization naming).
            visualize: Whether to create a t-SNE visualization for embeddings.

        Returns:
            Tuple[float, float, float, float]: Training loss, validation loss, validation accuracy, validation ARI.
        """
        self.model.train()

        x = data.node_features.to(self.device)
        hyperedge_index = data.hyperedge_index.to(self.device)
        labels = data.labels.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)

        embeddings = self.model(x, hyperedge_index)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        train_embeddings = embeddings[train_mask]
        train_labels = labels[train_mask]
        train_loss = clustering_loss_function(
            train_embeddings, train_labels, lambda_sep=2.0)

        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        self.model.eval()
        with torch.no_grad():
            val_embeddings = F.normalize(embeddings[val_mask], p=2, dim=1)
            val_labels = labels[val_mask]
            val_loss = clustering_loss_function(
                val_embeddings, val_labels, lambda_sep=2.0)
            _, val_accuracy, val_ari = clustering_error_function(
                val_embeddings, val_labels)

            if visualize and epoch is not None:
                all_embeddings = F.normalize(embeddings, p=2, dim=1)
                all_labels = labels

                embeddings_np = all_embeddings.detach().cpu().numpy()
                n_clusters = len(torch.unique(all_labels))
                kmeans = KMeans(n_clusters=n_clusters,
                                random_state=42, n_init=10)
                predicted_labels = kmeans.fit_predict(embeddings_np)

                visualize_embeddings_tsne(
                    all_embeddings, all_labels, predicted_labels,
                    data.dataset_name, epoch
                )

        return train_loss.item(), val_loss.item(), val_accuracy, val_ari

    def train(self,
              data: ContactDataset,
              train_mask: torch.Tensor,
              val_mask: torch.Tensor,
              optimizer: torch.optim.Optimizer,
              num_epochs: int = 100,
              visualize_epochs: list = []) -> Dict:
        """
        Trains the model for multiple epochs and optionally visualizes at specific epochs.

        Args:
            data (ContactDataset): Dataset containing features, labels, and hyperedges.
            train_mask (torch.Tensor): Boolean mask for training nodes.
            val_mask (torch.Tensor): Boolean mask for validation nodes.
            optimizer (torch.optim.Optimizer): Optimizer used for model training.
            num_epochs (int, optional): Number of training epochs. Default is 200.
            visualize_epochs (list, optional): Epochs at which to visualize embeddings.

        Returns:
            Dict: Dictionary with best validation stats and loss/accuracy history.
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
        self.val_aris = []

        for epoch in range(num_epochs):
            should_visualize = (epoch + 1) in visualize_epochs
            train_loss, val_loss, val_accuracy, val_ari = self.train_epoch(
                data, train_mask, val_mask, optimizer, epoch + 1, should_visualize
            )

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            self.val_aris.append(val_ari)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            mlflow.log_metric("val_ari", val_ari, step=epoch)

            if val_ari > best_val_ari:
                best_val_ari = val_ari
                best_epoch = epoch

            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_accuracy:.4f} | "
                      f"Val ARI: {val_ari:.4f}")

                for metric_name, values in [
                    ("Train Loss", self.train_losses),
                    ("Val Loss", self.val_losses),
                    ("Val Accuracy", self.val_accuracies),
                    ("Val ARI", self.val_aris)
                ]:
                    plot_path = plot_metric_over_epochs(
                        values,
                        name=metric_name,
                        output_dir="mlflow_plots",
                        filename_prefix=f"{data.dataset_name}_epoch_{epoch+1}"
                    )
                    mlflow.log_artifact(plot_path, artifact_path="plots")

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

    def evaluate(self,
                 data: ContactDataset,
                 test_mask: torch.Tensor,
                 visualize: bool = False
                 ) -> Dict:
        """
        Evaluates the trained model on the test set.

        Args:
            data: Dataset containing features, labels, and hyperedges.
            test_mask: Boolean mask for test nodes.
            visualize: Whether to visualize the embeddings using t-SNE.

        Returns:
            Dict: Dictionary containing test loss, accuracy, ARI, and confusion matrix.
        """
        self.model.eval()

        x = data.node_features.to(self.device)
        hyperedge_index = data.hyperedge_index.to(self.device)
        labels = data.labels.to(self.device)
        test_mask = test_mask.to(self.device)

        with torch.no_grad():
            embeddings = self.model(x, hyperedge_index)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            test_embeddings = embeddings[test_mask]
            test_labels = labels[test_mask]

            test_loss = clustering_loss_function(
                test_embeddings, test_labels, lambda_sep=2.0)
            confusion_matrix, test_accuracy, test_ari = clustering_error_function(
                test_embeddings, test_labels)

            if visualize:
                embeddings_np = test_embeddings.detach().cpu().numpy()
                n_clusters = len(torch.unique(test_labels))
                kmeans = KMeans(n_clusters=n_clusters,
                                random_state=42, n_init=10)
                predicted_labels = kmeans.fit_predict(embeddings_np)

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
