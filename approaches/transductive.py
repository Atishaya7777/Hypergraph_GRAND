import torch
import mlflow
import mlflow.pytorch
import numpy as np
from typing import Dict, Any, Optional, Tuple

from data import ContactDataset, DataSplitter 
from data.dataset import create_hypergraph_dataset
from models import HypergraphGRAND, create_hypergrand_model
from training.trainer import create_hypergraph_trainer, BaseHypergraphTrainer


class EdgeDropout:
    """Edge dropout utility for hypergraphs"""
    
    def __init__(self, dropout_rate: float = 0.5):
        self.dropout_rate = dropout_rate
    
    def __call__(self, hyperedge_index: torch.Tensor, 
                 hyperedge_weight: Optional[torch.Tensor] = None,
                 training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply edge dropout to hypergraph structure
        
        Args:
            hyperedge_index: [2, num_edges] hyperedge connectivity
            hyperedge_weight: Optional edge weights
            training: Whether model is in training mode
            
        Returns:
            Tuple of (dropped_hyperedge_index, dropped_hyperedge_weight)
        """
        if not training or self.dropout_rate == 0.0:
            return hyperedge_index, hyperedge_weight
        
        num_edges = hyperedge_index.size(1)
        if num_edges == 0:
            return hyperedge_index, hyperedge_weight
        
        # Create dropout mask
        keep_prob = 1.0 - self.dropout_rate
        edge_mask = torch.rand(num_edges, device=hyperedge_index.device) < keep_prob
        
        # Apply mask to hyperedge_index
        dropped_hyperedge_index = hyperedge_index[:, edge_mask]
        
        # Apply mask to hyperedge_weight if it exists
        dropped_hyperedge_weight = None
        if hyperedge_weight is not None:
            dropped_hyperedge_weight = hyperedge_weight[edge_mask]
            # Rescale weights to maintain expected sum
            if dropped_hyperedge_weight.numel() > 0:
                dropped_hyperedge_weight = dropped_hyperedge_weight / keep_prob
        
        return dropped_hyperedge_index, dropped_hyperedge_weight


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


def transductive_learning_approach(dataset_name: str, strategy: str = 'clustering', 
                                 edge_dropout_rates: list = [0.5, 0.6, 0.75],
                                 num_epochs: int = 5000, 
                                 patience: int = 1000,
                                 log_detailed_params: bool = True):
    """
    Enhanced transductive learning on each dataset individually with edge dropout and early stopping
    
    Args:
        dataset_name: ['contact', 'planetoid', 'planetoid_cora', 'planetoid_citeseer', 'planetoid_pubmed']
        strategy: ['classification', 'clustering']
        edge_dropout_rates: List of edge dropout rates to try
        num_epochs: Maximum number of training epochs
        patience: Early stopping patience
        log_detailed_params: Whether to log detailed parameters
    """
    print("="*60)
    print("ENHANCED TRANSDUCTIVE LEARNING WITH EDGE DROPOUT")
    print("="*60)

    datasets = {}
    dataset_name = dataset_name.lower()

    # Dataset configuration
    if dataset_name == 'contact':
        datasets.update({
            'contact-high-school': 'datasets/contact-high-school',
            'contact-primary-school': 'datasets/contact-primary-school',
        })
    elif dataset_name == 'planetoid_cora':
        datasets.update({
            'planetoid_cora': 'datasets/cora'
        })
    elif dataset_name == 'planetoid_citeseer':
        datasets.update({
            'planetoid_citeseer': 'datasets/citeseer'
        })
    elif dataset_name == 'planetoid_pubmed':
        datasets.update({
            'planetoid_pubmed': 'datasets/pubmed'
        })

    results = {}
    print(f"Datasets: ", list(datasets.keys()))
    print(f"Edge dropout rates to test: {edge_dropout_rates}")

    # Test different edge dropout rates
    for edge_dropout_rate in edge_dropout_rates:
        print(f"\n{'='*30} EDGE DROPOUT RATE: {edge_dropout_rate} {'='*30}")
        
        for dataset_name, dataset_path in datasets.items():
            print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")

            run_name = f"HyperGRAND_Transductive_{dataset_name}_dropout_{edge_dropout_rate}"
            
            with mlflow.start_run(run_name=run_name):
                # Load dataset
                datasetFactory = create_hypergraph_dataset(dataset_name)
                data = datasetFactory.load_data(dataset_path)

                # Create data splits
                if dataset_name == 'contact':
                    train_mask = data.train_mask
                    val_mask = data.val_mask
                    test_mask = data.test_mask
                else:
                    train_mask, val_mask, test_mask = DataSplitter.create_transductive_split(
                        data.labels
                    )

                # Determine input dimension
                if dataset_name.startswith('planetoid'):
                    input_dim = data.node_features.shape[1]
                else:
                    input_dim = data.num_nodes

                # Hyperparameters
                hyperparams = {
                    "input_dim": input_dim,
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "alpha": 0.02,
                    "dropout": 0.5,
                    "scheme": "explicit",
                    "edge_dropout_rate": edge_dropout_rate,
                    "num_epochs": num_epochs,
                    "patience": patience,
                    "learning_rate": 0.001,
                    "weight_decay": 5e-4,
                    "strategy": strategy,
                    "dataset_name": dataset_name,
                }

                # Integration scheme specific parameters
                scheme_defaults = {
                    "implicit": {"max_iter": 10, "tol": 1e-6},
                    "adaptive": {"min_alpha": 0.01, "max_alpha": 0.5, "tol": 1e-4}
                }
                
                if hyperparams["scheme"] in scheme_defaults:
                    hyperparams.update(scheme_defaults[hyperparams["scheme"]])

                # Log all parameters
                if log_detailed_params:
                    # Model parameters
                    mlflow.log_params(hyperparams)
                    
                    # Dataset statistics
                    dataset_stats = {
                        'num_nodes': data.num_nodes,
                        'num_hyperedges': data.num_hyperedges,
                        'num_classes': data.num_classes,
                        'train_nodes': train_mask.sum().item(),
                        'val_nodes': val_mask.sum().item(),
                        'test_nodes': test_mask.sum().item(),
                    }
                    mlflow.log_params(dataset_stats)
                    
                    # Training configuration
                    training_config = {
                        'early_stopping_monitor': 'val_accuracy',
                        'early_stopping_mode': 'max',
                        'early_stopping_min_delta': 1e-6,
                    }
                    mlflow.log_params(training_config)

                # Create model
                model = create_hypergrand_model(
                    input_dim=hyperparams["input_dim"],
                    hidden_dim=hyperparams["hidden_dim"],
                    num_layers=hyperparams["num_layers"],
                    alpha=hyperparams["alpha"],
                    dropout=hyperparams["dropout"],
                    scheme=hyperparams["scheme"]
                )
                
                # Create trainer
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                trainer = create_hypergraph_trainer(
                    task_type=strategy,
                    model=model,
                    device=device,
                    num_classes=data.num_classes
                )

                # Create optimizer
                optimizer = torch.optim.Adam(
                    list(model.parameters()) + 
                    (list(trainer.classifier.parameters()) if hasattr(trainer, 'classifier') else []), 
                    lr=hyperparams["learning_rate"], 
                    weight_decay=hyperparams["weight_decay"]
                )

                # Create enhanced trainer with edge dropout
                enhanced_trainer = EnhancedHypergraphTrainer(
                    base_trainer=trainer,
                    edge_dropout_rate=edge_dropout_rate
                )

                # Enhanced training loop with edge dropout and early stopping
                train_results = enhanced_trainer.train_with_early_stopping(
                    data=data,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    optimizer=optimizer,
                    num_epochs=num_epochs,
                    patience=patience,
                    log_interval=100
                )

                # Final evaluation
                test_results = enhanced_trainer.evaluate(data, test_mask)

                # Log model
                mlflow.pytorch.log_model(model, artifact_path="model")

                # Log final metrics
                final_metrics = {
                    'final_train_accuracy': train_results.get('final_train_accuracy', 0.0),
                    'best_val_accuracy': train_results['best_val_accuracy'],
                    'test_accuracy': test_results['test_accuracy'],
                    'test_loss': test_results['test_loss'],
                    'total_epochs': train_results['total_epochs'],
                    'stopped_early': train_results['stopped_early'],
                }
                mlflow.log_metrics(final_metrics)

                # Store results
                result_key = f"{dataset_name}_dropout_{edge_dropout_rate}"
                results[result_key] = {
                    'hyperparams': hyperparams,
                    'train_results': train_results,
                    'test_results': test_results,
                    'dataset_stats': dataset_stats
                }

                # Print results
                print(f"\nFinal Results for {dataset_name} (dropout={edge_dropout_rate}):")
                print(f"  - Best Val Accuracy: {train_results['best_val_accuracy']:.4f}")
                print(f"  - Test Accuracy: {test_results['test_accuracy']:.4f}")
                print(f"  - Test Loss: {test_results['test_loss']:.4f}")
                print(f"  - Total Epochs: {train_results['total_epochs']}")
                print(f"  - Stopped Early: {train_results['stopped_early']}")

    return results


class EnhancedHypergraphTrainer:
    """
    Enhanced trainer wrapper that adds edge dropout and early stopping to existing trainers
    """
    
    def __init__(self, base_trainer: BaseHypergraphTrainer, edge_dropout_rate: float = 0.5):
        self.base_trainer = base_trainer
        self.edge_dropout = EdgeDropout(edge_dropout_rate)
        self.original_train_epoch = base_trainer.train_epoch
        
        # Monkey patch the train_epoch method to include edge dropout
        self.base_trainer.train_epoch = self._enhanced_train_epoch
    
    def _enhanced_train_epoch(self, data, train_mask, val_mask, optimizer, epoch=None, visualize=False):
        """Enhanced train_epoch that applies edge dropout during training"""
        
        # Apply edge dropout to the data
        original_hyperedge_index = data.hyperedge_index
        original_hyperedge_weight = getattr(data, 'hyperedge_weight', None)
        
        # Apply dropout for training
        dropped_hyperedge_index, dropped_hyperedge_weight = self.edge_dropout(
            data.hyperedge_index,
            original_hyperedge_weight,
            training=self.base_trainer.model.training
        )
        
        # Temporarily modify the data object
        data.hyperedge_index = dropped_hyperedge_index
        if original_hyperedge_weight is not None:
            data.hyperedge_weight = dropped_hyperedge_weight
        
        try:
            # Call the original train_epoch method with modified data
            result = self.original_train_epoch(data, train_mask, val_mask, optimizer, epoch, visualize)
        finally:
            # Restore original data
            data.hyperedge_index = original_hyperedge_index
            if original_hyperedge_weight is not None:
                data.hyperedge_weight = original_hyperedge_weight
        
        return result
    
    def train_with_early_stopping(self, data, train_mask, val_mask, optimizer, 
                                 num_epochs=5000, patience=1000, log_interval=100):
        early_stopping = EarlyStopping(
            patience=patience,
            monitor='val_accuracy' if hasattr(self.base_trainer, 'val_accuracies') else 'val_ari',
            mode='max'
        )
        
        # Print enhanced training info
        print(f"Enhanced Training Configuration:")
        print(f"  - Max epochs: {num_epochs}")
        print(f"  - Early stopping patience: {patience}")
        print(f"  - Edge dropout rate: {self.edge_dropout.dropout_rate}")
        print(f"  - Log interval: {log_interval}")
        
        # Initialize tracking
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
    
    def evaluate(self, data, test_mask, visualize=False):
        """Delegate evaluation to base trainer (no edge dropout during evaluation)"""
        return self.base_trainer.evaluate(data, test_mask, visualize)
    
    def __getattr__(self, name):
        """Delegate other methods to base trainer"""
        return getattr(self.base_trainer, name)

'''
# Example usage
if __name__ == "__main__":
    # Test different edge dropout rates
    edge_dropout_rates = [0.5, 0.6, 0.75]
    
    results = transductive_learning_approach(
        dataset_name='contact',
        strategy='classification',
        edge_dropout_rates=edge_dropout_rates,
        num_epochs=5000,
        patience=1000,
        log_detailed_params=True
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for result_key, result_data in results.items():
        print(f"\n{result_key}:")
        print(f"  Best Val Acc: {result_data['train_results']['best_val_accuracy']:.4f}")
        print(f"  Test Acc: {result_data['test_results']['test_accuracy']:.4f}")
        print(f"  Epochs: {result_data['train_results']['total_epochs']}")
        print(f"  Early Stop: {result_data['train_results']['stopped_early']}")
'''
