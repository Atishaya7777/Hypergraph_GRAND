#!/usr/bin/env python3
"""
Comprehensive evaluation script testing HyperGRAND across all datasets
organized by task type: classification, clustering, and partitioning.

This script validates the hypothesis that HyperGRAND performs well on
clustering tasks but struggles with classification and partitioning tasks.
"""

import argparse
import json
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

from data.dataset import create_hypergraph_dataset, DataSplitter
from models import create_hypergrand_model
from training.trainer import create_hypergraph_trainer
from approaches.transductive import EdgeDropout, EarlyStopping


class DatasetMetadata:
    """Manages dataset metadata and organization by task type"""
    
    def __init__(self, metadata_path: str = "datasets/DATASET_METADATA.json"):
        self.metadata_path = Path(metadata_path)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata from JSON file"""
        with open(self.metadata_path, 'r') as f:
            return json.load(f)
    
    def get_datasets_by_task(self, task_type: str) -> Dict[str, Dict]:
        """Get all datasets for a specific task type"""
        if task_type not in self.metadata:
            raise ValueError(f"Unknown task type: {task_type}. Available: {list(self.metadata.keys())}")
        return self.metadata[task_type]['datasets']
    
    def get_all_task_types(self) -> List[str]:
        """Get all available task types"""
        return [key for key in self.metadata.keys() if key != 'description']
    
    def get_dataset_path(self, dataset_name: str) -> str:
        """Get the filesystem path for a dataset"""
        for task_type in self.get_all_task_types():
            datasets = self.get_datasets_by_task(task_type)
            if dataset_name in datasets:
                return f"datasets/{datasets[dataset_name]['path']}"
        raise ValueError(f"Dataset not found: {dataset_name}")


class ComprehensiveEvaluator:
    """Comprehensive evaluator for testing across datasets and task types"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metadata = DatasetMetadata()
        self.results = {
            'classification': {},
            'clustering': {},
            'partitioning': {}
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def run_all_evaluations(self) -> Dict:
        """Run evaluations for all task types"""
        for task_type in ['classification', 'clustering', 'partitioning']:
            print(f"\n{'='*80}")
            print(f"EVALUATING {task_type.upper()} DATASETS")
            print(f"{'='*80}\n")
            
            datasets = self.metadata.get_datasets_by_task(task_type)
            
            for dataset_name, dataset_info in datasets.items():
                print(f"\n{'-'*60}")
                print(f"Dataset: {dataset_name}")
                print(f"Source: {dataset_info.get('source', 'N/A')}")
                print(f"{'-'*60}\n")
                
                try:
                    result = self.evaluate_dataset(
                        dataset_name=dataset_name,
                        task_type=task_type,
                        dataset_info=dataset_info
                    )
                    self.results[task_type][dataset_name] = result
                except Exception as e:
                    print(f"ERROR evaluating {dataset_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.results[task_type][dataset_name] = {
                        'status': 'ERROR',
                        'error': str(e)
                    }
        
        return self.results
    
    def evaluate_dataset(self, dataset_name: str, task_type: str, dataset_info: Dict) -> Dict:
        """Evaluate a single dataset"""
        
        # Load dataset
        dataset_path = self.metadata.get_dataset_path(dataset_name)
        dataset_loader = create_hypergraph_dataset(dataset_name)
        data = dataset_loader.load_data(dataset_path)
        
        print(f"Loaded: {data.num_nodes} nodes, {data.num_hyperedges} hyperedges, {data.num_classes} classes")
        
        # Determine strategy based on task type
        if task_type == 'classification':
            strategy = 'classification'
        elif task_type == 'clustering':
            strategy = 'clustering'
        else:  # partitioning
            # Treat as clustering for now
            strategy = 'clustering'
        
        # Create data splits
        train_mask, val_mask, test_mask = DataSplitter.create_transductive_split(
            data.labels,
            train_ratio=self.config['train_ratio'],
            val_ratio=self.config['val_ratio'],
            random_state=self.config['seed']
        )
        
        print(f"Split: {train_mask.sum()} train, {val_mask.sum()} val, {test_mask.sum()} test")
        
        # Create model
        input_dim = data.node_features.shape[1] if data.node_features is not None else data.num_nodes
        model = create_hypergrand_model(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            alpha=self.config['alpha'],
            dropout=self.config['dropout'],
            scheme=self.config['scheme']
        )
        
        # Create trainer
        trainer = create_hypergraph_trainer(
            task_type=strategy,
            model=model,
            device=self.device,
            num_classes=data.num_classes
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            list(model.parameters()) + 
            (list(trainer.classifier.parameters()) if hasattr(trainer, 'classifier') else []),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Training with early stopping
        print(f"\nTraining with {strategy} strategy...")
        
        mlflow.set_experiment(f"comprehensive_{task_type}")
        
        with mlflow.start_run(run_name=f"{dataset_name}_{self.config['seed']}"):
            # Log config
            mlflow.log_params({
                'dataset': dataset_name,
                'task_type': task_type,
                'strategy': strategy,
                'hidden_dim': self.config['hidden_dim'],
                'num_layers': self.config['num_layers'],
                'alpha': self.config['alpha'],
                'dropout': self.config['dropout'],
                'scheme': self.config['scheme'],
            })
            
            # Run training
            early_stopping = EarlyStopping(
                patience=self.config['patience'],
                monitor='val_accuracy' if strategy == 'classification' else 'val_ari',
                mode='max'
            )
            
            best_metric = -1.0 if strategy == 'clustering' else 0.0
            best_epoch = 0
            
            for epoch in range(self.config['num_epochs']):
                # Apply edge dropout
                hyperedge_index = data.hyperedge_index.to(self.device)
                dropped_index, _ = EdgeDropout(dropout_rate=self.config['edge_dropout_rate'])(
                    hyperedge_index, training=True
                )
                
                # Train epoch
                metrics = trainer.train_epoch(
                    data, train_mask, val_mask, optimizer, epoch + 1, visualize=False
                )
                
                # Extract metric based on strategy
                if strategy == 'classification':
                    current_metric = metrics[2]  # val_accuracy
                else:
                    current_metric = metrics[3]  # val_ari
                
                # Update best
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                
                # Early stopping
                if early_stopping(current_metric):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if (epoch + 1) % self.config['log_interval'] == 0:
                    print(f"Epoch {epoch + 1}/{self.config['num_epochs']} - "
                          f"Metric: {current_metric:.4f} - Best: {best_metric:.4f}")
            
            # Final evaluation
            test_results = trainer.evaluate(data, test_mask, visualize=False)
            
            # Log final metrics
            mlflow.log_metrics({
                'best_val_metric': best_metric,
                'test_accuracy': test_results.get('test_accuracy', 0.0),
                'test_f1': test_results.get('test_f1_weighted', test_results.get('test_ari', 0.0)),
                'best_epoch': best_epoch,
                'total_epochs': epoch + 1
            })
        
        return {
            'status': 'SUCCESS',
            'dataset_name': dataset_name,
            'task_type': task_type,
            'strategy': strategy,
            'num_nodes': data.num_nodes,
            'num_hyperedges': data.num_hyperedges,
            'num_classes': data.num_classes,
            'best_val_metric': float(best_metric),
            'best_epoch': best_epoch,
            'total_epochs': epoch + 1,
            'test_results': {
                'test_accuracy': float(test_results.get('test_accuracy', 0.0)),
                'test_loss': float(test_results.get('test_loss', 0.0)),
                'test_f1': float(test_results.get('test_f1_weighted', test_results.get('test_ari', 0.0)))
            }
        }
    
    def save_results(self, output_path: str = 'comprehensive_results.json'):
        """Save evaluation results to JSON"""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    def print_summary(self):
        """Print summary of results by task type"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*80 + "\n")
        
        for task_type in ['classification', 'clustering', 'partitioning']:
            print(f"\n{task_type.upper()} RESULTS:")
            print("-" * 60)
            
            results = self.results.get(task_type, {})
            if not results:
                print("  No results")
                continue
            
            success_count = 0
            avg_test_acc = 0.0
            
            for dataset_name, result in results.items():
                if result.get('status') == 'SUCCESS':
                    success_count += 1
                    test_acc = result['test_results'].get('test_accuracy', 0.0)
                    avg_test_acc += test_acc
                    print(f"  {dataset_name:30s}: Test Acc={test_acc:.4f}")
                else:
                    print(f"  {dataset_name:30s}: ERROR - {result.get('error', 'Unknown')}")
            
            if success_count > 0:
                avg_test_acc /= success_count
                print(f"\n  Average Test Accuracy: {avg_test_acc:.4f}")
                print(f"  Successful: {success_count}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive HyperGRAND evaluation across all datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--alpha', type=float, default=0.2, help='Diffusion alpha')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--scheme', type=str, default='explicit', 
                       choices=['explicit', 'implicit', 'multistep', 'adaptive'])
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--edge_dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--train_ratio', type=float, default=0.5)
    parser.add_argument('--val_ratio', type=float, default=0.25)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='comprehensive_results.json')
    parser.add_argument('--skip_tasks', type=str, nargs='+', default=[],
                       help='Task types to skip: classification, clustering, partitioning')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    config = vars(args)
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(config)
    evaluator.run_all_evaluations()
    evaluator.save_results(args.output)
    evaluator.print_summary()


if __name__ == '__main__':
    main()
