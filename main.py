#!/usr/bin/env python3
"""
HyperGRAND: Hypergraph Graph Neural Diffusion
Unified main entry point for testing, training, and evaluation across all datasets.
Integrates MLflow for comprehensive metric logging and experiment tracking.
Supports classification, clustering, and partitioning tasks.
"""

import argparse
import json
import torch
import numpy as np
from typing import Dict, List
from pathlib import Path
import sys

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))

from train_model import train_dataset, train_all_datasets


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="HyperGRAND: Unified testing and training for all datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "train", "validate", "batch"],
        default="train",
        help="Mode: test (structure validation), train (training), validate (full validation), batch (train all datasets)",
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset to train/test. If not specified with --mode=train, trains representative datasets.",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Multiple datasets to train/test",
    )

    # Model architecture
    parser.add_argument(
        "--hidden-dim", type=int, default=32, help="Hidden dimension of the model"
    )
    parser.add_argument(
        "--num-layers", type=int, default=3, help="Number of layers in the model"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Diffusion parameter alpha"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate"
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["explicit", "implicit", "multistep", "adaptive"],
        default="explicit",
        help="Integration scheme for the diffusion process",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=200, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="Early stopping patience"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay for optimizer"
    )

    # MLflow
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="HyperGRAND",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow logging"
    )

    # Output and logging
    parser.add_argument(
        "--output",
        type=str,
        default="hypergrand_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Save training results to JSON file (for batch mode)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()




def validate_dataset(loader, dataset_name: str, task_type: str) -> Dict:
    """Validate a single dataset structure"""
    try:
        data = loader.load(dataset_name)
        if data is None:
            return {
                'valid': False,
                'error': 'Failed to load dataset'
            }
        
        return {
            'valid': True,
            'dataset': dataset_name,
            'task_type': task_type,
            'num_nodes': data.x.shape[0] if data.x is not None else 0,
            'num_features': data.x.shape[1] if data.x is not None else 0,
            'num_classes': int(data.y.max().item()) + 1 if data.y is not None else 0,
            'num_edges': data.hyperedge_index.shape[1] if data.hyperedge_index is not None else 0,
            'has_train_mask': hasattr(data, 'train_mask') and data.train_mask is not None,
            'has_val_mask': hasattr(data, 'val_mask') and data.val_mask is not None,
            'has_test_mask': hasattr(data, 'test_mask') and data.test_mask is not None,
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


class MLFlowLogger:
    """Wrapper for MLflow logging with task-aware metrics"""
    
    def __init__(self, enabled: bool = True, experiment_name: str = "HyperGRAND", tracking_uri: str = None):
        self.enabled = enabled and MLFLOW_AVAILABLE
        if self.enabled:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
    
    def start_run(self, dataset_name: str, task_type: str, params: Dict):
        """Start MLflow run"""
        if not self.enabled:
            return
        
        mlflow.start_run(run_name=f"{dataset_name}_{task_type}")
        mlflow.log_params({
            'dataset': dataset_name,
            'task_type': task_type,
            **{f'param_{k}': v for k, v in params.items()}
        })
    
    def log_metrics(self, metrics: Dict, step: int = None):
        """Log metrics to MLflow"""
        if not self.enabled:
            return
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)
    
    def log_result(self, result: Dict):
        """Log full training result"""
        if not self.enabled:
            return
        
        task_type = result.get('task_type', 'unknown')
        
        # Log common metrics
        mlflow.log_metric('best_epoch', result.get('best_epoch', 0))
        mlflow.log_metric('final_test_loss', result.get('final_test_loss', 0.0))
        
        # Log task-specific metrics
        if task_type in ['classification', 'partitioning']:
            mlflow.log_metric('final_test_accuracy', result.get('final_test_metric', 0.0))
        elif task_type == 'clustering':
            mlflow.log_metric('test_nmi', result.get('test_nmi', 0.0))
            mlflow.log_metric('test_ari', result.get('test_ari', 0.0))
        
        # Log learning curves
        for i, loss in enumerate(result.get('train_losses', [])):
            mlflow.log_metric('train_loss', loss, step=i)
        for i, loss in enumerate(result.get('val_losses', [])):
            mlflow.log_metric('val_loss', loss, step=i)
        for i, metric in enumerate(result.get('val_metrics', [])):
            mlflow.log_metric('val_metric', metric, step=i)
    
    def end_run(self):
        """End MLflow run"""
        if self.enabled:
            mlflow.end_run()


def run_validation_mode(args):
    """Run structure validation on all datasets"""
    print(f"\n{'='*100}")
    print("DATASET STRUCTURE VALIDATION")
    print(f"{'='*100}\n")
    
    from data.pyg_standardizer import DatasetLoader
    
    loader = DatasetLoader(base_path="datasets")
    
    datasets_to_test = {
        'classification': [
            'cora', 'coauthorship_cora', 'coauthorship_dblp',
            'cocitation_citeseer', 'cocitation_cora', 'cocitation_pubmed',
            'house_committees'
        ],
        'clustering': [
            'contact_high_school', 'contact_primary_school',
            'walmart_trips', 'news_20w100', 'yelp'
        ],
        'partitioning': [
            'zoo', 'mushroom', 'ntu2012', 'modelnet40'
        ]
    }
    
    total = 0
    passed = 0
    
    for task_type in sorted(datasets_to_test.keys()):
        print(f"{task_type.upper()}")
        print("-" * 100)
        
        for dataset_name in sorted(datasets_to_test[task_type]):
            total += 1
            try:
                result = validate_dataset(loader, dataset_name, task_type)
                if result['valid']:
                    passed += 1
                    print(f"  ✓ {dataset_name:<30} | nodes={result['num_nodes']:>8} | classes={result['num_classes']:>2} | features={result['num_features']:>5} | edges={result['num_edges']:>8}")
                else:
                    print(f"  ✗ {dataset_name:<30} | {result.get('error', '')[:50]}")
            except Exception as e:
                print(f"  ✗ {dataset_name:<30} | {str(e)[:50]}")
        print()
    
    print(f"{'='*100}")
    print(f"VALIDATION RESULTS: {passed}/{total} datasets valid ({100*passed/total:.0f}%)")
    print(f"{'='*100}\n")


def run_train_mode(args):
    """Run training on specified or representative datasets"""
    print(f"\n{'='*100}")
    print("TRAINING WITH TASK-AWARE LEARNING")
    print(f"{'='*100}\n")
    
    mlflow_logger = MLFlowLogger(
        enabled=not args.no_mlflow,
        experiment_name=args.mlflow_experiment,
        tracking_uri=args.mlflow_tracking_uri if hasattr(args, 'mlflow_tracking_uri') else None
    )
    
    if args.dataset:
        # Train single dataset
        print(f"Training on {args.dataset}...\n")
        result = train_dataset(
            args.dataset,
            hidden_dim=args.hidden_dim,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience,
            verbose=args.verbose
        )
        
        # Log to MLflow
        task_type = result.get('task_type', 'classification')
        mlflow_logger.start_run(args.dataset, task_type, {
            'hidden_dim': args.hidden_dim,
            'epochs': args.epochs,
            'lr': args.lr,
            'patience': args.patience,
        })
        mlflow_logger.log_result(result)
        mlflow_logger.end_run()
        
        return {args.dataset: result}
    else:
        # Train representative datasets
        representative = [
            'cora',  # Classification
            'contact_high_school',  # Clustering
            'zoo'  # Partitioning
        ]
        results = {}
        
        for dataset_name in representative:
            print(f"\nTraining on {dataset_name}...\n")
            try:
                result = train_dataset(
                    dataset_name,
                    hidden_dim=args.hidden_dim,
                    num_epochs=args.epochs,
                    learning_rate=args.lr,
                    patience=args.patience,
                    verbose=args.verbose
                )
                
                # Log to MLflow
                task_type = result.get('task_type', 'classification')
                mlflow_logger.start_run(dataset_name, task_type, {
                    'hidden_dim': args.hidden_dim,
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'patience': args.patience,
                })
                mlflow_logger.log_result(result)
                mlflow_logger.end_run()
                
                results[dataset_name] = result
            except Exception as e:
                print(f"Failed to train {dataset_name}: {e}")
                results[dataset_name] = {'error': str(e)}
        
        return results


def run_batch_mode(args):
    """Run training on all datasets with resume capability"""
    print(f"\n{'='*100}")
    print("BATCH TRAINING ON ALL DATASETS")
    print(f"{'='*100}\n")
    
    mlflow_logger = MLFlowLogger(
        enabled=not args.no_mlflow,
        experiment_name=args.mlflow_experiment,
        tracking_uri=args.mlflow_tracking_uri if hasattr(args, 'mlflow_tracking_uri') else None
    )
    
    # All datasets to train
    all_datasets = [
        # Classification (7)
        'cora', 'coauthorship_cora', 'coauthorship_dblp',
        'cocitation_citeseer', 'cocitation_cora', 'cocitation_pubmed',
        'house_committees',
        # Clustering (5)
        'contact_high_school', 'contact_primary_school',
        'walmart_trips', 'news_20w100', 'yelp',
        # Partitioning (4)
        'modelnet40', 'mushroom', 'ntu2012', 'zoo'
    ]
    
    # Load existing results if resuming
    results_file = args.save_results if args.save_results else 'training_results.json'
    results_path = Path(results_file)
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {results_file}")
        datasets_to_train = [d for d in all_datasets if d not in results]
        print(f"Will train {len(datasets_to_train)} remaining datasets\n")
    else:
        results = {}
        datasets_to_train = all_datasets
        print(f"Will train all {len(datasets_to_train)} datasets\n")
    
    # Train each dataset
    for i, dataset_name in enumerate(datasets_to_train, 1):
        print(f"\n[{i}/{len(datasets_to_train)}] Training {dataset_name}...\n")
        try:
            result = train_dataset(
                dataset_name,
                hidden_dim=args.hidden_dim,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                patience=args.patience,
                verbose=args.verbose if hasattr(args, 'verbose') else True
            )
            results[dataset_name] = result
            
            # Log to MLflow
            task_type = result.get('task_type', 'classification')
            mlflow_logger.start_run(dataset_name, task_type, {
                'hidden_dim': args.hidden_dim,
                'epochs': args.epochs,
                'lr': args.lr,
                'patience': args.patience,
            })
            mlflow_logger.log_result(result)
            mlflow_logger.end_run()
        except Exception as e:
            print(f"\n❌ Failed to train {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = {'error': str(e)}
        
        # Save results after each dataset (incremental save)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved results to {results_file}")
    
    return results


def main():
    """Unified main entry point for all testing and training modes."""
    args = parse_args()
    
    print(f"\n{'='*100}")
    print(f"HyperGRAND: Hypergraph Graph Neural Network with Diffusion")
    print(f"{'='*100}")
    print(f"Mode: {args.mode}")
    print(f"{'='*100}\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    try:
        if args.mode == 'test':
            # Run structure validation (no training)
            run_validation_mode(args)
        
        elif args.mode == 'validate':
            # Full validation (detailed)
            run_validation_mode(args)
        
        elif args.mode == 'train':
            # Train on specified or representative datasets
            results = run_train_mode(args)
            
            # Print summary
            print(f"\n{'='*100}")
            print("TRAINING SUMMARY")
            print(f"{'='*100}")
            for dataset_name, result in results.items():
                if 'error' in result:
                    print(f"{dataset_name}: ERROR - {result['error']}")
                else:
                    task_type = result.get('task_type', 'unknown')
                    best_epoch = result.get('best_epoch', 0)
                    test_loss = result.get('final_test_loss', 0.0)
                    
                    if task_type in ['classification', 'partitioning']:
                        test_metric = result.get('final_test_metric', 0.0)
                        print(f"{dataset_name} ({task_type}): Best Epoch={best_epoch}, Test Acc={test_metric:.4f}, Test Loss={test_loss:.4f}")
                    elif task_type == 'clustering':
                        nmi = result.get('test_nmi', 0.0)
                        ari = result.get('test_ari', 0.0)
                        print(f"{dataset_name} ({task_type}): Best Epoch={best_epoch}, Test NMI={nmi:.4f}, Test ARI={ari:.4f}, Test Loss={test_loss:.4f}")
            print(f"{'='*100}\n")
        
        elif args.mode == 'batch':
            # Train on all datasets
            results = run_batch_mode(args)
            
            # Save results if requested
            if args.save_results:
                import json
                with open(args.save_results, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nResults saved to {args.save_results}")
            
            # Print summary
            print(f"\n{'='*100}")
            print("BATCH TRAINING SUMMARY")
            print(f"{'='*100}")
            for dataset_name, result in results.items():
                if 'error' in result:
                    print(f"{dataset_name}: ERROR - {result['error']}")
                else:
                    task_type = result.get('task_type', 'unknown')
                    best_epoch = result.get('best_epoch', 0)
                    test_loss = result.get('final_test_loss', 0.0)
                    
                    if task_type in ['classification', 'partitioning']:
                        test_metric = result.get('final_test_metric', 0.0)
                        print(f"{dataset_name} ({task_type}): Best Epoch={best_epoch}, Test Acc={test_metric:.4f}, Test Loss={test_loss:.4f}")
                    elif task_type == 'clustering':
                        nmi = result.get('test_nmi', 0.0)
                        ari = result.get('test_ari', 0.0)
                        print(f"{dataset_name} ({task_type}): Best Epoch={best_epoch}, Test NMI={nmi:.4f}, Test ARI={ari:.4f}, Test Loss={test_loss:.4f}")
            print(f"{'='*100}\n")
        
        else:
            print(f"Unknown mode: {args.mode}")
            print("Use --help for usage information")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
