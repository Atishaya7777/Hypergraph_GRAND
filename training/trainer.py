import mlflow
from abc import ABC, abstractmethod
import torch
from typing import Tuple, Dict, List, Union
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

from data.dataset import HypergraphData, HypergraphDataset
from .loss import clustering_loss_function, clustering_error_function
from utils import visualize_embeddings_tsne, plot_metric_over_epochs

# TODO: Refactor this file to branch out and code split the various classes

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 1000, min_delta: float = 1e-6, 
                 monitor: str = 'val_accuracy', mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.wait = 0
        self.best_score = None
        self.should_stop = False
        
        if mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            self.monitor_op = np.less
            self.min_delta *= -1
    
    def __call__(self, current_score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            current_score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.should_stop = True
            return True
            
        return False
    
    def reset(self):
        """Reset early stopping state"""
        self.wait = 0
        self.best_score = None
        self.should_stop = False

class BaseHypergraphTrainer(ABC):
    """
    Abstract base class for hypergraph trainers.
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
    
    @abstractmethod
    def train_epoch(self,
                    data: HypergraphData,
                    train_mask: torch.Tensor,
                    val_mask: torch.Tensor,
                    optimizer: torch.optim.Optimizer,
                    epoch: int = None,
                    visualize: bool = False
                    ) -> Tuple[float, float, float, float]:
        """Train the model for one epoch."""
        pass

    @abstractmethod
    def evaluate(self,
                 data: HypergraphData,
                 test_mask: torch.Tensor,
                 visualize: bool = False
                 ) -> Dict:
        """Evaluate the trained model on the test set."""
        pass
    
    def train(self,
              data: HypergraphData,
              train_mask: torch.Tensor,
              val_mask: torch.Tensor,
              optimizer: torch.optim.Optimizer,
              num_epochs: int = 200,
              visualize_epochs: list = []) -> Dict:
        """
        Base training loop that can be used by both clustering and classification trainers.
        """
        self._print_training_info(data, train_mask, val_mask, visualize_epochs)
        
        best_metric = self._get_initial_best_metric()
        best_epoch = 0
        
        for epoch in range(num_epochs):
            should_visualize = (epoch + 1) in visualize_epochs
            metrics = self.train_epoch(
                data, train_mask, val_mask, optimizer, epoch + 1, should_visualize
            )
            
            self._update_metrics(metrics)
            self._log_metrics_to_mlflow(metrics, epoch)
            
            # Update best metric
            current_metric = self._get_current_metric(metrics)
            if self._is_better_metric(current_metric, best_metric):
                best_metric = current_metric
                best_epoch = epoch
            
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                self._print_epoch_progress(epoch + 1, num_epochs, metrics)
                self._plot_and_log_metrics(data, epoch + 1)
        
        self._print_best_results(best_metric, best_epoch + 1)
        return self._get_training_results(best_epoch, best_metric)

    def training_with_early_stopping(self,
                                     data: HypergraphData,
                                     train_mask: torch.Tensor,
                                     val_mask: torch.Tensor,
                                     optimizer: torch.optim.Optimizer,
                                     num_epochs: int=5000,
                                     patience: int =1000,
                                     log_interval=100):

        early_stopping = EarlyStopping(
            patience=patience,
            monitor='val_accuracy' if hasattr(self.base_trainer, 'val_accuracies') else 'val_ari',
            mode='max')

        best_metric = self.base_trainer._get_initial_best_metric()
        best_epoch = 0
        
        # Training loop
        for epoch in range(num_epochs):
            # Train one epoch
            should_visualize = False  # Can be customized
            metrics = self.base_trainer.train_epoch(
                data, train_mask, val_mask, optimizer, epoch + 1, should_visualize
            )
            
            # Update internal metrics
            self.base_trainer._update_metrics(metrics)
            self.base_trainer._log_metrics_to_mlflow(metrics, epoch)
            
            # Get current metric for early stopping
            current_metric = self.base_trainer._get_current_metric(metrics)
            
            # Update best metric
            if self.base_trainer._is_better_metric(current_metric, best_metric):
                best_metric = current_metric
                best_epoch = epoch
            
            # Logging
            if (epoch + 1) % log_interval == 0 or epoch == num_epochs - 1:
                self.base_trainer._print_epoch_progress(epoch + 1, num_epochs, metrics)
                
                # Log additional edge dropout info
                mlflow.log_metrics({
                    'edge_dropout_rate': self.edge_dropout.dropout_rate,
                    'edges_kept_ratio': 1.0 - self.edge_dropout.dropout_rate
                }, step=epoch)
            
            # Check early stopping
            if early_stopping(current_metric):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                print(f"Best metric: {best_metric:.4f} at epoch {best_epoch + 1}")
                break
        
        # Print final results
        self.base_trainer._print_best_results(best_metric, best_epoch + 1)
        
        # Return enhanced results
        base_results = self.base_trainer._get_training_results(best_epoch, best_metric)
        base_results.update({
            'total_epochs': epoch + 1,
            'stopped_early': early_stopping.should_stop,
            'edge_dropout_rate': self.edge_dropout.dropout_rate,
            'patience_used': patience
        })
        
        return base_results

    
    @abstractmethod
    def _print_training_info(self, data, train_mask, val_mask, visualize_epochs):
        """Print training setup information."""
        pass
    
    @abstractmethod
    def _get_initial_best_metric(self):
        """Get initial value for best metric tracking."""
        pass
    
    @abstractmethod
    def _get_current_metric(self, metrics):
        """Extract the main metric from current epoch metrics."""
        pass
    
    @abstractmethod
    def _is_better_metric(self, current, best):
        """Determine if current metric is better than best."""
        pass
    
    @abstractmethod
    def _update_metrics(self, metrics):
        """Update internal metric tracking lists."""
        pass
    
    @abstractmethod
    def _log_metrics_to_mlflow(self, metrics, epoch):
        """Log metrics to MLflow."""
        pass
    
    @abstractmethod
    def _print_epoch_progress(self, epoch, num_epochs, metrics):
        """Print progress for current epoch."""
        pass
    
    @abstractmethod
    def _print_best_results(self, best_metric, best_epoch):
        """Print best results summary."""
        pass
    
    @abstractmethod
    def _get_training_results(self, best_epoch, best_metric):
        """Get final training results dictionary."""
        pass
    
    def _plot_and_log_metrics(self, data, epoch):
        """Common plotting and logging functionality."""
        metric_pairs = self._get_metric_pairs_for_plotting()
        
        for metric_name, values in metric_pairs:
            plot_path = plot_metric_over_epochs(
                values,
                name=metric_name,
                output_dir="mlflow_plots",
                filename_prefix=f"{getattr(data.dataset_info, 'dataset_name', 'dataset')}_epoch_{epoch}"
            )
            mlflow.log_artifact(plot_path, artifact_path="plots")
    
    @abstractmethod
    def _get_metric_pairs_for_plotting(self):
        """Get metric name-value pairs for plotting."""
        pass


class HypergraphClusteringTrainer(BaseHypergraphTrainer):
    """
    Trainer class for the HyperGRAND model for clustering tasks.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = torch.device('cpu')):
        super().__init__(model, device)
        self.val_aris = []
    
    def train_epoch(self,
                    data: HypergraphData,
                    train_mask: torch.Tensor,
                    val_mask: torch.Tensor,
                    optimizer: torch.optim.Optimizer,
                    epoch: int = None,
                    visualize: bool = False
                    ) -> Tuple[float, float, float, float]:
        """Train the model for one epoch."""
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
                    getattr(data.dataset_info, 'dataset_name', 'dataset'), epoch
                )

        return train_loss.item(), val_loss.item(), val_accuracy, val_ari
    
    def evaluate(self,
                 data: HypergraphData,
                 test_mask: torch.Tensor,
                 visualize: bool = False
                 ) -> Dict:
        """Evaluate the trained model on the test set."""
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
            confusion_matrix_result, test_accuracy, test_ari = clustering_error_function(
                test_embeddings, test_labels)

            if visualize:
                embeddings_np = test_embeddings.detach().cpu().numpy()
                n_clusters = len(torch.unique(test_labels))
                kmeans = KMeans(n_clusters=n_clusters,
                                random_state=42, n_init=10)
                predicted_labels = kmeans.fit_predict(embeddings_np)

                visualize_embeddings_tsne(
                    test_embeddings, test_labels, predicted_labels,
                    f"{getattr(data.dataset_info, 'dataset_name', 'dataset')}_test", "final"
                )

        print(f"\nTest Results:")
        print(f"  - Test Loss: {test_loss.item():.4f}")
        print(f"  - Test Accuracy: {test_accuracy:.4f}")
        print(f"  - Test ARI: {test_ari:.4f}")

        return {
            'test_loss': test_loss.item(),
            'test_accuracy': test_accuracy,
            'test_ari': test_ari,
            'confusion_matrix': confusion_matrix_result
        }
    
    def _print_training_info(self, data, train_mask, val_mask, visualize_epochs):
        print(f"Training clustering model:")
        print(f"  - Train nodes: {train_mask.sum().item()}")
        print(f"  - Val nodes: {val_mask.sum().item()}")
        print(f"  - Test nodes: {(~train_mask & ~val_mask).sum().item()}")
        if visualize_epochs:
            print(f"  - Will create t-SNE visualizations at epochs: {visualize_epochs}")
        print("-" * 50)
    
    def _get_initial_best_metric(self):
        return -1.0
    
    def _get_current_metric(self, metrics):
        return metrics[3]  # val_ari
    
    def _is_better_metric(self, current, best):
        return current > best
    
    def _update_metrics(self, metrics):
        train_loss, val_loss, val_accuracy, val_ari = metrics
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        self.val_aris.append(val_ari)
    
    def _log_metrics_to_mlflow(self, metrics, epoch):
        train_loss, val_loss, val_accuracy, val_ari = metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        mlflow.log_metric("val_ari", val_ari, step=epoch)
    
    def _print_epoch_progress(self, epoch, num_epochs, metrics):
        train_loss, val_loss, val_accuracy, val_ari = metrics
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | "
              f"Val ARI: {val_ari:.4f}")
    
    def _print_best_results(self, best_metric, best_epoch):
        print(f"\nBest validation ARI: {best_metric:.4f} at epoch {best_epoch}")
    
    def _get_training_results(self, best_epoch, best_metric):
        return {
            'best_val_accuracy': self.val_accuracies[best_epoch],
            'best_val_ari': best_metric,
            'best_epoch': best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_aris': self.val_aris
        }
    
    def _get_metric_pairs_for_plotting(self):
        return [
            ("Train Loss", self.train_losses),
            ("Val Loss", self.val_losses),
            ("Val Accuracy", self.val_accuracies),
            ("Val ARI", self.val_aris)
        ]


class HypergraphClassificationTrainer(BaseHypergraphTrainer):
    """
    Trainer class for the HyperGRAND model for node classification tasks.
    """

    def __init__(self, model: nn.Module, num_classes: int, device: torch.device = torch.device('cpu')):
        super().__init__(model, device)
        self.num_classes = num_classes
        
        # Add classification head to the model
        if hasattr(model, 'out_dim'):
            self.classifier = nn.Linear(model.out_dim, num_classes).to(device)
        else:
            # Assume last layer output dimension
            self.classifier = nn.Linear(model.layers[-1].out_channels, num_classes).to(device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.val_f1_scores = []

    def train_epoch(self,
                    data: HypergraphData,
                    train_mask: torch.Tensor,
                    val_mask: torch.Tensor,
                    optimizer: torch.optim.Optimizer,
                    epoch: int = None,
                    visualize: bool = False
                    ) -> Tuple[float, float, float, float]:
        """Train the model for one epoch."""
        self.model.train()
        self.classifier.train()

        x = data.node_features.to(self.device)
        hyperedge_index = data.hyperedge_index.to(self.device)
        labels = data.labels.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)

        # Forward pass
        embeddings = self.model(x, hyperedge_index)
        logits = self.classifier(embeddings)
        
        # Training loss
        train_logits = logits[train_mask]
        train_labels = labels[train_mask]
        train_loss = self.criterion(train_logits, train_labels)

        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.classifier.parameters()), 
            max_norm=1.0
        )
        optimizer.step()

        # Validation
        self.model.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            val_logits = logits[val_mask]
            val_labels = labels[val_mask]
            val_loss = self.criterion(val_logits, val_labels)
            
            # Calculate metrics
            val_preds = torch.argmax(val_logits, dim=1)
            val_accuracy = accuracy_score(
                val_labels.cpu().numpy(), 
                val_preds.cpu().numpy()
            )
            val_f1 = f1_score(
                val_labels.cpu().numpy(), 
                val_preds.cpu().numpy(), 
                average='weighted'
            )

            if visualize and epoch is not None:
                # Visualize embeddings (before classification head)
                all_embeddings = embeddings.detach()
                all_labels = labels
                all_preds = torch.argmax(logits, dim=1)

                visualize_embeddings_tsne(
                    all_embeddings, all_labels, all_preds,
                    getattr(data.dataset_info, 'dataset_name', 'dataset'), epoch
                )

        return train_loss.item(), val_loss.item(), val_accuracy, val_f1

    def evaluate(self,
                 data: HypergraphData,
                 test_mask: torch.Tensor,
                 visualize: bool = False
                 ) -> Dict:
        """Evaluate the trained model on the test set."""
        self.model.eval()
        self.classifier.eval()

        x = data.node_features.to(self.device)
        hyperedge_index = data.hyperedge_index.to(self.device)
        labels = data.labels.to(self.device)
        test_mask = test_mask.to(self.device)

        with torch.no_grad():
            embeddings = self.model(x, hyperedge_index)
            logits = self.classifier(embeddings)

            test_logits = logits[test_mask]
            test_labels = labels[test_mask]
            test_embeddings = embeddings[test_mask]

            test_loss = self.criterion(test_logits, test_labels)
            test_preds = torch.argmax(test_logits, dim=1)
            
            # Calculate metrics
            test_accuracy = accuracy_score(
                test_labels.cpu().numpy(), 
                test_preds.cpu().numpy()
            )
            test_f1 = f1_score(
                test_labels.cpu().numpy(), 
                test_preds.cpu().numpy(), 
                average='weighted'
            )
            test_f1_macro = f1_score(
                test_labels.cpu().numpy(), 
                test_preds.cpu().numpy(), 
                average='macro'
            )
            test_confusion_matrix = confusion_matrix(
                test_labels.cpu().numpy(), 
                test_preds.cpu().numpy()
            )

            if visualize:
                visualize_embeddings_tsne(
                    test_embeddings, test_labels, test_preds,
                    f"{getattr(data.dataset_info, 'dataset_name', 'dataset')}_test", "final"
                )

        print(f"\nTest Results:")
        print(f"  - Test Loss: {test_loss.item():.4f}")
        print(f"  - Test Accuracy: {test_accuracy:.4f}")
        print(f"  - Test F1 (weighted): {test_f1:.4f}")
        print(f"  - Test F1 (macro): {test_f1_macro:.4f}")

        return {
            'test_loss': test_loss.item(),
            'test_accuracy': test_accuracy,
            'test_f1_weighted': test_f1,
            'test_f1_macro': test_f1_macro,
            'confusion_matrix': test_confusion_matrix,
            'predictions': test_preds.cpu().numpy(),
            'true_labels': test_labels.cpu().numpy()
        }

    def get_node_embeddings(self, data: HypergraphData) -> torch.Tensor:
        """Get node embeddings from the trained model."""
        self.model.eval()
        
        x = data.node_features.to(self.device)
        hyperedge_index = data.hyperedge_index.to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(x, hyperedge_index)
            
        return embeddings.cpu()

    def predict(self, data: HypergraphData, node_mask: torch.Tensor = None) -> torch.Tensor:
        """Make predictions on nodes."""
        self.model.eval()
        self.classifier.eval()
        
        x = data.node_features.to(self.device)
        hyperedge_index = data.hyperedge_index.to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(x, hyperedge_index)
            logits = self.classifier(embeddings)
            preds = torch.argmax(logits, dim=1)
            
            if node_mask is not None:
                node_mask = node_mask.to(self.device)
                preds = preds[node_mask]
                
        return preds.cpu()
    
    def _print_training_info(self, data, train_mask, val_mask, visualize_epochs):
        print(f"Training classification model:")
        print(f"  - Train nodes: {train_mask.sum().item()}")
        print(f"  - Val nodes: {val_mask.sum().item()}")
        print(f"  - Test nodes: {(~train_mask & ~val_mask).sum().item()}")
        print(f"  - Number of classes: {self.num_classes}")
        if visualize_epochs:
            print(f"  - Will create t-SNE visualizations at epochs: {visualize_epochs}")
        print("-" * 50)
    
    def _get_initial_best_metric(self):
        return 0.0
    
    def _get_current_metric(self, metrics):
        return metrics[2]  # val_accuracy
    
    def _is_better_metric(self, current, best):
        return current > best
    
    def _update_metrics(self, metrics):
        train_loss, val_loss, val_accuracy, val_f1 = metrics
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        self.val_f1_scores.append(val_f1)
    
    def _log_metrics_to_mlflow(self, metrics, epoch):
        train_loss, val_loss, val_accuracy, val_f1 = metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)
    
    def _print_epoch_progress(self, epoch, num_epochs, metrics):
        train_loss, val_loss, val_accuracy, val_f1 = metrics
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | "
              f"Val F1: {val_f1:.4f}")
    
    def _print_best_results(self, best_metric, best_epoch):
        best_f1 = self.val_f1_scores[best_epoch - 1] if best_epoch > 0 else 0.0
        print(f"\nBest validation accuracy: {best_metric:.4f} at epoch {best_epoch}")
        print(f"Best validation F1: {best_f1:.4f} at epoch {best_epoch}")
    
    def _get_training_results(self, best_epoch, best_metric):
        best_f1 = self.val_f1_scores[best_epoch] if best_epoch < len(self.val_f1_scores) else 0.0
        return {
            'best_val_accuracy': best_metric,
            'best_val_f1': best_f1,
            'best_epoch': best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores
        }
    
    def _get_metric_pairs_for_plotting(self):
        return [
            ("Train Loss", self.train_losses),
            ("Val Loss", self.val_losses),
            ("Val Accuracy", self.val_accuracies),
            ("Val F1", self.val_f1_scores)
        ]


def create_hypergraph_trainer(task_type: str, 
                              model: nn.Module, 
                              device: torch.device = torch.device('cpu'),
                              num_classes: int = None) -> BaseHypergraphTrainer:
    """
    Factory function to create appropriate hypergraph trainer based on task type.
    
    Args:
        task_type (str): Either 'classification' or 'clustering'
        model (nn.Module): The HyperGRAND model to be trained
        device (torch.device): The device to run the model on
        num_classes (int): Number of classes (required for classification)
        
    Returns:
        BaseHypergraphTrainer: Appropriate trainer instance
        
    Raises:
        ValueError: If task_type is not supported or required parameters are missing
    """
    task_type = task_type.lower()
    
    if task_type == 'classification':
        if num_classes is None:
            raise ValueError("num_classes must be specified for classification tasks")
        return HypergraphClassificationTrainer(model, num_classes, device)
    
    elif task_type == 'clustering':
        return HypergraphClusteringTrainer(model, device)
    
    else:
        raise ValueError(f"Unsupported task_type: {task_type}. "
                        f"Supported types are: 'classification', 'clustering'")


"""
EXAMPLE USAGE:
# For classification (e.g., Cora dataset)
trainer = create_hypergraph_trainer(
    task_type='classification',
    model=your_model,
    device=torch.device('cuda'),
    num_classes=7
)

# For clustering
trainer = create_hypergraph_trainer(
    task_type='clustering',
    model=your_model,
    device=torch.device('cuda')
)

# Both trainers have the same interface
results = trainer.train(data, train_mask, val_mask, optimizer, num_epochs=200)
test_results = trainer.evaluate(data, test_mask, visualize=True)
"""
