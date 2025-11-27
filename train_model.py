#!/usr/bin/env python3
"""
Training script for HyperGRAND model on validated datasets
Passes validated PyG Data objects to the model for training
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.pyg_standardizer import DatasetLoader
from models.hypergrand import create_hypergrand_model
from torch_geometric.data import Data


class HyperGRANDTrainer:
    """Trainer for HyperGRAND model on PyG Data objects"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-5
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def _move_to_device(self, data: Data):
        """Move data to device"""
        data.x = data.x.to(self.device)
        data.y = data.y.to(self.device)
        data.hyperedge_index = data.hyperedge_index.to(self.device)
        data.train_mask = data.train_mask.to(self.device)
        data.val_mask = data.val_mask.to(self.device)
        data.test_mask = data.test_mask.to(self.device)
        return data
    
    def forward_pass(self, data: Data, num_classes: int) -> torch.Tensor:
        """Forward pass through the model"""
        # Get latent representations from HyperGRAND
        h = self.model(data.x, data.hyperedge_index)
        
        # Add classification head
        logits = nn.Linear(h.shape[1], num_classes, device=self.device)(h)
        return logits
    
    def train_epoch(self, data: Data, num_classes: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Forward pass
        logits = self.forward_pass(data, num_classes)
        
        # Compute loss on training nodes
        loss = self.criterion(logits[data.train_mask], data.y[data.train_mask])
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute training accuracy
        with torch.no_grad():
            train_acc = (logits[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).float().mean()
        
        return loss.item(), train_acc.item()
    
    @torch.no_grad()
    def evaluate(self, data: Data, num_classes: int, mask: torch.Tensor) -> Tuple[float, float]:
        """Evaluate on given mask (val or test)"""
        self.model.eval()
        
        logits = self.forward_pass(data, num_classes)
        
        loss = self.criterion(logits[mask], data.y[mask])
        acc = (logits[mask].argmax(dim=1) == data.y[mask]).float().mean()
        
        return loss.item(), acc.item()
    
    def train(
        self,
        data: Data,
        num_classes: int,
        num_epochs: int = 200,
        patience: int = 50,
        verbose: bool = True
    ) -> Dict:
        """Train the model with early stopping"""
        data = self._move_to_device(data)
        
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        results = {
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'best_val_loss': best_val_loss,
            'best_epoch': 0,
            'final_test_acc': 0.0,
            'final_test_loss': 0.0
        }
        
        pbar = tqdm(range(num_epochs), desc="Training", disable=not verbose)
        
        for epoch in pbar:
            # Train
            train_loss, train_acc = self.train_epoch(data, num_classes)
            results['train_losses'].append(train_loss)
            
            # Validate
            val_loss, val_acc = self.evaluate(data, num_classes, data.val_mask)
            results['val_losses'].append(val_loss)
            results['val_accuracies'].append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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
                    'val_acc': f'{val_acc:.4f}'
                })
        
        # Test
        test_loss, test_acc = self.evaluate(data, num_classes, data.test_mask)
        results['final_test_loss'] = test_loss
        results['final_test_acc'] = test_acc
        results['best_val_loss'] = best_val_loss
        results['best_epoch'] = best_epoch + 1
        
        return results


def train_dataset(
    dataset_name: str,
    hidden_dim: int = 32,
    num_epochs: int = 200,
    learning_rate: float = 0.01,
    patience: int = 50,
    verbose: bool = True
) -> Dict:
    """Train model on a single dataset"""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Training on {dataset_name}")
        print(f"{'='*80}")
    
    # Load dataset
    loader = DatasetLoader(base_path="datasets")
    data = loader.load(dataset_name, verbose=False)
    
    # Get dataset info
    num_classes = int(data.y.max().item()) + 1
    input_dim = data.x.shape[1]
    num_nodes = data.num_nodes
    
    if verbose:
        print(f"Dataset Info:")
        print(f"  Nodes: {num_nodes}")
        print(f"  Features: {input_dim}")
        print(f"  Classes: {num_classes}")
        print(f"  Hyperedges: {data.hyperedge_index.shape[1]}")
        print(f"  Train: {data.train_mask.sum().item()} | Val: {data.val_mask.sum().item()} | Test: {data.test_mask.sum().item()}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_hypergrand_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        alpha=0.1,
        dropout=0.1,
        scheme='explicit'
    )
    
    # Train
    trainer = HyperGRANDTrainer(model, device=device, learning_rate=learning_rate)
    results = trainer.train(
        data,
        num_classes=num_classes,
        num_epochs=num_epochs,
        patience=patience,
        verbose=verbose
    )
    
    if verbose:
        print(f"\nResults:")
        print(f"  Best Epoch: {results['best_epoch']}")
        print(f"  Best Val Loss: {results['best_val_loss']:.4f}")
        print(f"  Final Test Loss: {results['final_test_loss']:.4f}")
        print(f"  Final Test Accuracy: {results['final_test_acc']:.4f}")
    
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
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}\n")
    
    successful = 0
    failed = 0
    
    for dataset_name, result in sorted(results.items()):
        if 'error' in result:
            print(f"✗ {dataset_name:<30} | ERROR: {result['error'][:50]}")
            failed += 1
        else:
            test_acc = result['final_test_acc']
            test_loss = result['final_test_loss']
            print(f"✓ {dataset_name:<30} | Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
            successful += 1
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {successful} successful, {failed} failed")
    print(f"{'='*80}\n")


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train HyperGRAND on datasets')
    parser.add_argument('--dataset', type=str, default=None, help='Single dataset to train on')
    parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--all', action='store_true', help='Train on all datasets')
    
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
