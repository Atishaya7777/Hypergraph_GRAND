#!/usr/bin/env python3
"""
Example: Using PyG Data Objects with HyperGRAND

This script demonstrates how to use the new PyG converter to standardize
dataset loading across the entire codebase. It shows best practices for
working with PyG Data objects in HyperGRAND experiments.
"""

import torch
import argparse
from pathlib import Path

from data.manager import UnifiedDataManager
from models import create_hypergrand_model
from training.trainer import create_hypergraph_trainer
import mlflow


def main():
    parser = argparse.ArgumentParser(
        description="Example: PyG Data-based HyperGRAND Training"
    )
    parser.add_argument('--dataset', type=str, default='planetoid_cora',
                       help='Dataset name')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--hidden_dim', type=int, default=32,
                       help='Hidden dimension')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    
    print("\n" + "="*70)
    print("HYPERGRAND - PyG DATA EXAMPLE")
    print("="*70 + "\n")
    
    # Initialize data manager
    manager = UnifiedDataManager(seed=args.seed)
    
    print(f"Loading {args.dataset}...")
    data = manager.load(args.dataset, verbose=True)
    
    # Access standardized metadata
    print(f"\nDataset Metadata:")
    print(f"  Task Type: {data.metadata.task_type}")
    print(f"  Strategy: {data.metadata.strategy}")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Features: {data.x.shape[1]}")
    print(f"  Classes: {data.metadata.num_classes}")
    print(f"  Hyperedges: {data.metadata.num_hyperedges}")
    
    # Check splits
    if hasattr(data, 'train_mask'):
        print(f"\nData Splits:")
        print(f"  Train: {data.train_mask.sum().item()} nodes")
        print(f"  Val: {data.val_mask.sum().item() if hasattr(data, 'val_mask') else 0} nodes")
        print(f"  Test: {data.test_mask.sum().item() if hasattr(data, 'test_mask') else 0} nodes")
    
    # Now you can use the data with any part of the codebase
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_hypergrand_model(
        input_dim=data.x.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=2,
        alpha=0.2,
        dropout=0.5,
        scheme='explicit'
    ).to(device)
    
    # Create trainer
    trainer = create_hypergraph_trainer(
        task_type=data.metadata.strategy,
        model=model,
        device=device,
        num_classes=data.metadata.num_classes
    )
    
    print(f"\nTraining on {device}...")
    print(f"Task: {data.metadata.strategy}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + 
        (list(trainer.classifier.parameters()) if hasattr(trainer, 'classifier') else []),
        lr=args.learning_rate
    )
    
    # Quick training loop (simplified)
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        # Move data to device
        x = data.x.to(device)
        hyperedge_index = data.hyperedge_index.to(device)
        y = data.y.to(device)
        train_mask = data.train_mask.to(device)
        val_mask = data.val_mask.to(device)
        
        # Training step
        metrics = trainer.train_epoch(
            data, train_mask, val_mask, optimizer, epoch, visualize=False
        )
        
        # Extract metric
        val_acc = metrics[2]
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}: Val Acc = {val_acc:.4f}")
        
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Final evaluation
    test_mask = data.test_mask.to(device)
    test_results = trainer.evaluate(data, test_mask, visualize=False)
    
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_results.get('test_accuracy', 0.0):.4f}")
    if 'test_f1_weighted' in test_results:
        print(f"  Test F1 (weighted): {test_results['test_f1_weighted']:.4f}")
    
    print("\n" + "="*70)


def example_batch_loading():
    """Example: Loading multiple datasets in batch"""
    print("\n" + "="*70)
    print("EXAMPLE: BATCH LOADING DATASETS")
    print("="*70 + "\n")
    
    manager = UnifiedDataManager()
    
    # List all datasets
    print("Available datasets by task type:")
    datasets_by_task = manager.list_datasets()
    for task_type, datasets in datasets_by_task.items():
        print(f"\n{task_type.upper()}:")
        for ds in datasets:
            print(f"  - {ds}")
    
    # Load all classification datasets
    print("\n\nLoading all classification datasets...")
    classification_datasets = manager.load_by_task('classification', verbose=False)
    
    print(f"Loaded {len(classification_datasets)} classification datasets:")
    for name, data in classification_datasets.items():
        print(f"  {name:30s}: {data.num_nodes:5d} nodes, {data.metadata.num_hyperedges:5d} hyperedges, {data.metadata.num_classes:2d} classes")


def example_dataset_info():
    """Example: Accessing dataset information"""
    print("\n" + "="*70)
    print("EXAMPLE: DATASET INFORMATION")
    print("="*70 + "\n")
    
    manager = UnifiedDataManager()
    
    # Get info without loading
    print("Dataset information (without loading):")
    info = manager.get_dataset_info('planetoid_cora')
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Get split information
    print("\n\nDataset split information:")
    data = manager.load('planetoid_cora', verbose=False)
    
    from data.pyg_converter import PyGDataProcessor
    split_info = PyGDataProcessor.get_split_info(data)
    for split, info in split_info.items():
        print(f"  {split.upper()}: {info['count']} nodes ({info['ratio']*100:.1f}%)")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--examples':
        # Run example functions
        example_batch_loading()
        example_dataset_info()
    else:
        # Run main training example
        main()
