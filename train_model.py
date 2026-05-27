#!/usr/bin/env python3
"""
Training script for HyperGRAND model on validated datasets
Passes validated PyG Data objects to the model for training
Supports task-aware training: classification, clustering, partitioning
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent))

from data.pyg_standardizer import DatasetLoader
from models.hypergrand import create_hypergrand_model
from torch_geometric.data import Data


class FocalLoss(nn.Module):
    """Focal loss for class imbalance (Lin et al. ICCV 2017).

    Down-weights easy (high-confidence correct) examples so the model
    focuses on hard ones near the decision boundary:
        FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    gamma=0 recovers standard cross-entropy. gamma=2 is standard default.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce)
        loss = (1 - p_t) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class TaskAwareHead(nn.Module):
    """Task-specific output head"""
    def __init__(self, input_dim: int, num_classes: int, task_type: str = 'classification'):
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        
        if task_type in ['classification', 'partitioning']:
            # Classification/partitioning: linear layer + softmax
            self.head = nn.Linear(input_dim, num_classes)
        elif task_type == 'clustering':
            # Clustering: use embeddings directly, apply k-means at test time
            self.head = nn.Linear(input_dim, num_classes)  # Still need linear for initialization
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.task_type in ['classification', 'partitioning']:
            return self.head(h)
        else:  # clustering
            return self.head(h)


class HyperGRANDTrainer:
    """Task-aware trainer for HyperGRAND model on PyG Data objects"""
    
    def __init__(
        self,
        model: nn.Module,
        head: TaskAwareHead,
        task_type: str = 'classification',
        device: torch.device = None,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-5,
        label_smoothing: float = 0.1,
        loss: str = 'cross_entropy',
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.head = head.to(self.device)
        self.task_type = task_type
        
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.head.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        if task_type in ['classification', 'partitioning']:
            if loss == 'focal':
                self.criterion = FocalLoss(gamma=2.0)
            else:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif task_type == 'clustering':
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self._last_energy_values = {}
    
    def _move_to_device(self, data: Data):
        """Move data to device"""
        data.x = data.x.to(self.device)
        data.y = data.y.to(self.device)
        data.hyperedge_index = data.hyperedge_index.to(self.device)
        data.train_mask = data.train_mask.to(self.device)
        data.val_mask = data.val_mask.to(self.device)
        data.test_mask = data.test_mask.to(self.device)
        return data
    
    def forward_pass(self, data: Data) -> torch.Tensor:
        """Forward pass through the model"""
        # Get latent representations from HyperGRAND
        h = self.model(data.x, data.hyperedge_index)
        
        # Apply task-specific head
        logits = self.head(h)
        return logits
    
    def train_epoch(self, data: Data) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        self.head.train()
        
        # Forward pass
        logits = self.forward_pass(data)
        
        # Compute loss on training nodes
        loss = self.criterion(logits[data.train_mask], data.y[data.train_mask])
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.head.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        # Log Dirichlet energy per layer (if any layer has track_energy=True)
        energy_values = {}
        if hasattr(self.model, 'diffusion_layers'):
            energies = []
            for i, layer in enumerate(self.model.diffusion_layers):
                e = getattr(layer, 'last_dirichlet_energy', None)
                if e is not None:
                    energy_values[f'energy_layer_{i}'] = e
                    energies.append(e)
            # Compute normalised ratio (energy_layer_i / energy_layer_0)
            if energies and energies[0] is not None and energies[0] > 0:
                e0 = energies[0]
                for i, e in enumerate(energies):
                    energy_values[f'energy_ratio_{i}'] = e / e0
                # First layer where ratio < 0.01 (energy collapse)
                collapse_layer = next(
                    (i for i, e in enumerate(energies) if e / e0 < 0.01),
                    -1
                )
                energy_values['energy_collapse_layer'] = collapse_layer
            # Log to MLflow if available
            try:
                import mlflow
                if mlflow.active_run() is not None and energy_values:
                    mlflow.log_metrics(energy_values)
            except Exception:
                pass  # mlflow not available or not active; skip silently
        self._last_energy_values = energy_values

        # Compute training metric
        with torch.no_grad():
            if self.task_type in ['classification', 'partitioning']:
                train_metric = (logits[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).float().mean()
            else:  # clustering
                train_metric = (logits[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).float().mean()
        
        return loss.item(), train_metric.item()
    
    @torch.no_grad()
    def evaluate(self, data: Data, mask: torch.Tensor) -> Tuple[float, float]:
        """Evaluate on given mask (val or test)"""
        self.model.eval()
        self.head.eval()
        
        logits = self.forward_pass(data)
        
        loss = self.criterion(logits[mask], data.y[mask])
        
        if self.task_type in ['classification', 'partitioning']:
            metric = (logits[mask].argmax(dim=1) == data.y[mask]).float().mean()
        else:  # clustering - use accuracy as proxy
            metric = (logits[mask].argmax(dim=1) == data.y[mask]).float().mean()
        
        return loss.item(), metric.item()
    
    @torch.no_grad()
    def evaluate_clustering(self, data: Data, mask: torch.Tensor) -> Dict:
        """Evaluate clustering with k-means"""
        self.model.eval()
        
        # Get embeddings
        h = self.model(data.x, data.hyperedge_index)
        h_np = h[mask].cpu().numpy()
        y_np = data.y[mask].cpu().numpy()
        
        # Apply k-means
        num_classes = int(data.y.max().item()) + 1
        kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(h_np)
        
        # Compute metrics
        nmi = normalized_mutual_info_score(y_np, pred_labels)
        ari = adjusted_rand_score(y_np, pred_labels)
        
        return {'nmi': nmi, 'ari': ari}
    
    def train(
        self,
        data: Data,
        num_epochs: int = 200,
        patience: int = 50,
        verbose: bool = True
    ) -> Dict:
        """Train the model with early stopping"""
        data = self._move_to_device(data)
        
        best_val_metric = float('inf') if self.task_type != 'clustering' else 0.0
        best_epoch = 0
        patience_counter = 0
        
        results = {
            'train_losses': [],
            'val_losses': [],
            'val_metrics': [],
            'best_val_metric': best_val_metric,
            'best_epoch': 0,
            'final_test_loss': 0.0,
            'final_test_metric': 0.0,
            'task_type': self.task_type,
            'dirichlet_energies': [],  # list of energy_values dicts, one per epoch
        }
        
        pbar = tqdm(range(num_epochs), desc="Training", disable=not verbose)
        
        for epoch in pbar:
            # Train
            train_loss, train_metric = self.train_epoch(data)
            results['train_losses'].append(train_loss)
            results['dirichlet_energies'].append(getattr(self, '_last_energy_values', {}))
            
            # Validate
            val_loss, val_metric = self.evaluate(data, data.val_mask)
            results['val_losses'].append(val_loss)
            results['val_metrics'].append(val_metric)
            
            # Early stopping
            if self.task_type != 'clustering':
                is_better = val_loss < best_val_metric
            else:
                is_better = val_metric > best_val_metric
            
            if is_better:
                best_val_metric = val_loss if self.task_type != 'clustering' else val_metric
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_metric': f'{val_metric:.4f}'
                })
        
        # Test
        test_loss, test_metric = self.evaluate(data, data.test_mask)
        
        # For clustering, also compute NMI/ARI
        if self.task_type == 'clustering':
            cluster_metrics = self.evaluate_clustering(data, data.test_mask)
            results['test_nmi'] = cluster_metrics['nmi']
            results['test_ari'] = cluster_metrics['ari']
        
        results['final_test_loss'] = test_loss
        results['final_test_metric'] = test_metric
        results['best_val_metric'] = best_val_metric
        results['best_epoch'] = best_epoch + 1
        
        return results


class HyperedgeReconstructionPretrainer:
    """Self-supervised pretraining via hyperedge reconstruction.

    Masks a fraction of hyperedges and trains the encoder to reconstruct
    which nodes co-occur in masked edges. Forces the model to learn
    structure-aware representations before seeing any labels.

    Usage:
        pretrainer = HyperedgeReconstructionPretrainer(model, device)
        pretrainer.pretrain(data, num_epochs=50)
        # then fine-tune with HyperGRANDTrainer as usual
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        mask_rate: float = 0.2,
        num_neg_samples: int = 5,
        learning_rate: float = 0.001,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.mask_rate = mask_rate
        self.num_neg_samples = num_neg_samples
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def _mask_hyperedges(self, hyperedge_index: torch.Tensor):
        """Randomly mask a fraction of hyperedge entries."""
        # hyperedge_index is [2, num_entries] where row 0 = edge id, row 1 = node id
        num_edges = int(hyperedge_index[0].max().item()) + 1
        num_mask = max(1, int(self.mask_rate * num_edges))
        masked_edge_ids = torch.randperm(num_edges, device=hyperedge_index.device)[:num_mask]

        # Build a boolean mask over entries
        mask = torch.zeros(hyperedge_index.size(1), dtype=torch.bool, device=hyperedge_index.device)
        for eid in masked_edge_ids:
            mask |= (hyperedge_index[0] == eid)

        kept_index = hyperedge_index[:, ~mask]
        masked_entries = hyperedge_index[:, mask]
        return kept_index, masked_entries, masked_edge_ids

    def _reconstruction_loss(
        self,
        h: torch.Tensor,
        masked_entries: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """BCE loss: positive pairs are co-members of masked edges; negatives are random."""
        if masked_entries.size(1) == 0:
            return torch.tensor(0.0, device=h.device, requires_grad=True)

        # Build positive pairs from masked entries grouped by edge id
        edge_ids = masked_entries[0]
        node_ids = masked_entries[1]
        unique_edges = edge_ids.unique()

        pos_loss = torch.tensor(0.0, device=h.device)
        neg_loss = torch.tensor(0.0, device=h.device)
        count = 0

        for eid in unique_edges:
            members = node_ids[edge_ids == eid]
            if members.size(0) < 2:
                continue
            # All pairs within the edge (positive)
            for i in range(members.size(0)):
                for j in range(i + 1, members.size(0)):
                    score = (h[members[i]] * h[members[j]]).sum()
                    pos_loss = pos_loss + F.binary_cross_entropy_with_logits(
                        score.unsqueeze(0), torch.ones(1, device=h.device)
                    )
                    count += 1

            # Random negative pairs
            for _ in range(min(self.num_neg_samples, members.size(0))):
                neg_node = torch.randint(0, num_nodes, (1,), device=h.device).item()
                ref_node = members[torch.randint(0, members.size(0), (1,)).item()]
                score = (h[ref_node] * h[neg_node]).sum()
                neg_loss = neg_loss + F.binary_cross_entropy_with_logits(
                    score.unsqueeze(0), torch.zeros(1, device=h.device)
                )
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=h.device, requires_grad=True)
        return (pos_loss + neg_loss) / count

    def pretrain(self, data, num_epochs: int = 50, verbose: bool = True):
        """Run self-supervised pretraining on the given dataset.

        Args:
            data: PyG Data object with .x and .hyperedge_index
            num_epochs: Number of pretraining epochs
            verbose: Print progress
        """
        x = data.x.to(self.device)
        hyperedge_index = data.hyperedge_index.to(self.device)
        num_nodes = x.size(0)

        if verbose:
            print(f"[Pretraining] Starting hyperedge reconstruction pretraining for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Mask a fraction of hyperedges
            kept_index, masked_entries, _ = self._mask_hyperedges(hyperedge_index)

            # Encode with the masked hypergraph
            h = self.model(x, kept_index)

            # Reconstruction loss
            loss = self._reconstruction_loss(h, masked_entries, num_nodes)
            loss.backward()
            self.optimizer.step()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"[Pretraining] Epoch {epoch+1}/{num_epochs}  loss={loss.item():.4f}")

        if verbose:
            print("[Pretraining] Done.")


def train_dataset(
    dataset_name: str,
    hidden_dim: int = 32,
    num_epochs: int = 200,
    learning_rate: float = 0.01,
    patience: int = 50,
    verbose: bool = True,
    seed: int = None,
    mlflow_logger = None,
    parent_run_id: str = None,
    config: Dict = None
) -> Dict:
    """
    Train model on a single dataset with task-aware training
    
    Args:
        dataset_name: Name of dataset to train on
        hidden_dim: Hidden dimension for model
        num_epochs: Maximum training epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        verbose: Print training progress
        seed: Random seed for reproducibility
        mlflow_logger: MLFlowLogger instance for logging
        parent_run_id: Parent run ID for nested MLflow runs
        config: Additional configuration dict (for diffusion studies)
    
    Returns:
        Dictionary with training results
    """
    
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Training on {dataset_name}" + (f" (seed={seed})" if seed is not None else ""))
        print(f"{'='*80}")
    
    # Load dataset
    loader = DatasetLoader(base_path="datasets")
    data = loader.load(dataset_name, verbose=False)
    
    # Get dataset info
    num_classes = int(data.y.max().item()) + 1
    input_dim = data.x.shape[1]
    num_nodes = data.num_nodes
    
    # Infer task type from metadata
    task_type = 'classification'  # default
    if hasattr(data, 'metadata'):
        task_type = data.metadata.task_type
    
    if verbose:
        print(f"Dataset Info:")
        print(f"  Nodes: {num_nodes}")
        print(f"  Features: {input_dim}")
        print(f"  Classes: {num_classes}")
        print(f"  Hyperedges: {data.hyperedge_index.shape[1]}")
        print(f"  Task Type: {task_type}")
        print(f"  Train: {data.train_mask.sum().item()} | Val: {data.val_mask.sum().item()} | Test: {data.test_mask.sum().item()}")
    
    # Apply config if provided (for diffusion studies)
    if config:
        hidden_dim = config.get('hidden_dim', hidden_dim)
        num_layers = config.get('num_layers', 3)
        alpha = config.get('alpha', 0.1)
        dropout = config.get('dropout', 0.1)
        scheme = config.get('integration_scheme', 'explicit')
        learning_rate = config.get('lr', learning_rate)
        num_epochs = config.get('epochs', num_epochs)
        patience = config.get('patience', patience)
        label_smoothing = config.get('label_smoothing', 0.1)
        loss_type = config.get('loss', 'cross_entropy')
    else:
        num_layers = 3
        alpha = 0.1
        dropout = 0.1
        scheme = 'explicit'
        label_smoothing = 0.1
        loss_type = 'cross_entropy'
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_hypergrand_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        alpha=alpha,
        dropout=dropout,
        scheme=scheme
    )
    
    # Create task-specific head
    head = TaskAwareHead(hidden_dim, num_classes, task_type=task_type)
    
    # Start nested MLflow run if logger provided
    if mlflow_logger and parent_run_id:
        config_variant = config.get('config_variant', 'default') if config else 'default'
        mlflow_logger.start_child_run(
            dataset_name=dataset_name,
            config_variant=config_variant,
            seed=seed if seed is not None else 42,
            params={
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'alpha': alpha,
                'dropout': dropout,
                'integration_scheme': scheme,
                'learning_rate': learning_rate,
                'epochs': num_epochs,
                'patience': patience,
                'num_nodes': num_nodes,
                'num_features': input_dim,
                'num_classes': num_classes,
                'num_hyperedges': data.hyperedge_index.shape[1],
            },
            tags={'task_type': task_type, 'dataset': dataset_name}
        )
    
    # Self-supervised pretraining (optional, controlled by config['pretrain'])
    if config and config.get('pretrain', False):
        pretrain_epochs = config.get('pretrain_epochs', 50)
        pretrainer = HyperedgeReconstructionPretrainer(
            model=model,
            device=device,
            mask_rate=config.get('pretrain_mask_rate', 0.2),
            num_neg_samples=config.get('pretrain_neg_samples', 5),
        )
        pretrainer.pretrain(data, num_epochs=pretrain_epochs, verbose=verbose)

    # Train
    trainer = HyperGRANDTrainer(
        model=model,
        head=head,
        task_type=task_type,
        device=device,
        learning_rate=learning_rate,
        label_smoothing=label_smoothing,
        loss=loss_type,
    )
    results = trainer.train(
        data,
        num_epochs=num_epochs,
        patience=patience,
        verbose=verbose
    )
    
    # Summarise Dirichlet energy tracking if available
    if verbose and results.get('dirichlet_energies'):
        last_energies = results['dirichlet_energies'][-1]
        if last_energies:
            print(f"\n  Dirichlet Energy (final epoch):")
            for k, v in sorted(last_energies.items()):
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    
    # Log results to MLflow if logger provided
    if mlflow_logger:
        mlflow_logger.log_result(results)
        if parent_run_id:
            mlflow_logger.end_child_run()
    
    if verbose:
        print(f"\nResults:")
        print(f"  Best Epoch: {results['best_epoch']}")
        if task_type in ['classification', 'partitioning']:
            print(f"  Best Val Loss: {results['best_val_metric']:.4f}")
            print(f"  Final Test Loss: {results['final_test_loss']:.4f}")
            print(f"  Final Test Accuracy: {results['final_test_metric']:.4f}")
        elif task_type == 'clustering':
            print(f"  Best Val Loss: {results['best_val_metric']:.4f}")
            print(f"  Final Test NMI: {results.get('test_nmi', 0.0):.4f}")
            print(f"  Final Test ARI: {results.get('test_ari', 0.0):.4f}")
    
    results['dataset_name'] = dataset_name
    results['task_type'] = task_type
    results['num_nodes'] = num_nodes
    results['num_classes'] = num_classes
    results['seed'] = seed if seed is not None else 42
    if config:
        results['config'] = config
    
    return results


def train_all_datasets(
    datasets: list = None,
    hidden_dim: int = 32,
    num_epochs: int = 200,
    learning_rate: float = 0.01,
    patience: int = 50
) -> Dict:
    """Train on multiple datasets and collect results"""
    
    if datasets is None:
        datasets = [
            # Classification
            'cora', 'coauthorship_cora', 'coauthorship_dblp',
            'cocitation_citeseer', 'cocitation_cora', 'cocitation_pubmed',
            'house_committees',
            # Clustering
            'contact_high_school', 'contact_primary_school',
            'walmart_trips', 'news_20w100', 'yelp',
            # Partitioning
            'modelnet40', 'mushroom', 'ntu2012', 'zoo'
        ]
    
    results = {}
    
    for dataset_name in datasets:
        try:
            result = train_dataset(
                dataset_name,
                hidden_dim=hidden_dim,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                patience=patience,
                verbose=True
            )
            results[dataset_name] = result
        except Exception as e:
            print(f"\n❌ Failed to train on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = {'error': str(e)}
    
    return results


def print_summary(results: Dict):
    """Print summary of training results"""
    print(f"\n{'='*100}")
    print("TRAINING SUMMARY")
    print(f"{'='*100}\n")
    
    # Group by task type
    by_task = {}
    for dataset_name, result in sorted(results.items()):
        if 'error' in result:
            task_type = 'unknown'
        else:
            task_type = result.get('task_type', 'classification')
        
        if task_type not in by_task:
            by_task[task_type] = {'success': 0, 'failed': 0, 'results': []}
        
        if 'error' in result:
            print(f"✗ {dataset_name:<30} | ERROR: {result['error'][:50]}")
            by_task[task_type]['failed'] += 1
        else:
            by_task[task_type]['success'] += 1
            metric_val = result.get('final_test_metric', 0.0)
            test_loss = result['final_test_loss']
            
            if task_type == 'clustering':
                nmi = result.get('test_nmi', 0.0)
                ari = result.get('test_ari', 0.0)
                print(f"  {dataset_name:<30} | NMI: {nmi:.4f} | ARI: {ari:.4f} | Loss: {test_loss:.4f}")
            else:
                print(f"  {dataset_name:<30} | Acc: {metric_val:.4f} | Loss: {test_loss:.4f}")
            
            by_task[task_type]['results'].append((dataset_name, metric_val, test_loss))
    
    print(f"\n{'='*100}")
    print("SUMMARY BY TASK TYPE")
    print(f"{'='*100}")
    for task_type, stats in sorted(by_task.items()):
        print(f"\n{task_type.upper()}:")
        print(f"  Successful: {stats['success']} | Failed: {stats['failed']}")
        if stats['results']:
            accs = [m for _, m, _ in stats['results']]
            print(f"  Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    
    total_success = sum(s['success'] for s in by_task.values())
    total_failed = sum(s['failed'] for s in by_task.values())
    print(f"\n{'='*100}")
    print(f"TOTAL: {total_success} successful, {total_failed} failed")
    print(f"{'='*100}\n")


def main():
    """Main training script"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Train HyperGRAND on datasets with task awareness')
    parser.add_argument('--dataset', type=str, default=None, help='Single dataset to train on')
    parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--all', action='store_true', help='Train on all datasets')
    parser.add_argument('--save-results', type=str, default=None, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    if args.dataset:
        # Train single dataset
        result = train_dataset(
            args.dataset,
            hidden_dim=args.hidden_dim,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience
        )
    elif args.all:
        # Train all datasets
        results = train_all_datasets(
            hidden_dim=args.hidden_dim,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience
        )
        print_summary(results)
        
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.save_results}")
    else:
        # Train on a few representative datasets
        representative_datasets = [
            'cora',  # Classification
            'contact_high_school',  # Clustering
            'zoo'  # Partitioning
        ]
        
        results = train_all_datasets(
            datasets=representative_datasets,
            hidden_dim=args.hidden_dim,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience
        )
        print_summary(results)


if __name__ == '__main__':
    main()
