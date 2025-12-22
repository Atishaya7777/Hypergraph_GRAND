"""
Parallelized Diffusion Study Runner
Executes systematic parameter studies across multiple datasets with multiprocessing support.
Integrates with MLflow for comprehensive experiment tracking.
"""

import sys
import torch
import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.diffusion_config import (
    DiffusionStudyConfig,
    DiffusionStudySpace,
    get_study_execution_plan
)
from train_model import train_dataset


# Dataset categorization by task type
TASK_DATASETS = {
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

# Representative datasets for fast mode
REPRESENTATIVE_DATASETS = {
    'classification': 'cora',
    'clustering': 'contact_high_school',
    'partitioning': 'zoo'
}


def run_single_experiment(args: Tuple) -> Dict[str, Any]:
    """
    Run a single training experiment
    
    Args:
        args: Tuple of (dataset_name, config, seed, verbose)
    
    Returns:
        Dictionary with results
    """
    dataset_name, config_dict, seed, verbose = args
    
    try:
        # Convert config dict back to DiffusionStudyConfig
        config = DiffusionStudyConfig.from_dict(config_dict)
        
        # Train
        result = train_dataset(
            dataset_name=dataset_name,
            hidden_dim=config.hidden_dim,
            num_epochs=config.epochs,
            learning_rate=config.lr,
            patience=config.patience,
            verbose=verbose,
            seed=seed,
            mlflow_logger=None,  # Don't use MLflow in worker process
            parent_run_id=None,
            config=config.to_dict()
        )
        
        return {
            'success': True,
            'dataset': dataset_name,
            'config_variant': config.config_variant,
            'study_dimension': config.study_dimension,
            'seed': seed,
            'result': result
        }
    
    except Exception as e:
        import traceback
        return {
            'success': False,
            'dataset': dataset_name,
            'config_variant': config_dict.get('config_variant', 'unknown'),
            'study_dimension': config_dict.get('study_dimension', 'unknown'),
            'seed': seed,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


class DiffusionStudyRunner:
    """Orchestrates parallel execution of diffusion studies"""
    
    def __init__(
        self,
        study_dimensions: List[str] = None,
        datasets: List[str] = None,
        num_seeds: int = 5,
        parallel_workers: int = None,
        fast_mode: bool = False,
        representative_only: bool = False,
        mlflow_logger=None
    ):
        """
        Initialize diffusion study runner
        
        Args:
            study_dimensions: List of dimensions to study
            datasets: List of datasets to use
            num_seeds: Number of random seeds per configuration
            parallel_workers: Number of parallel workers (None = auto-detect)
            fast_mode: Use reduced configuration set
            representative_only: Use only representative datasets
            mlflow_logger: MLFlowLogger instance for tracking
        """
        self.study_dimensions = study_dimensions or ['integration_scheme', 'diffusion_depth', 'attention_mechanism']
        self.num_seeds = num_seeds
        self.mlflow_logger = mlflow_logger
        self.fast_mode = fast_mode
        self.representative_only = representative_only
        
        # Determine number of workers
        if parallel_workers is None:
            # Use half of available CPUs or number of GPUs
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            num_cpus = mp.cpu_count()
            self.parallel_workers = max(1, min(num_gpus if num_gpus > 0 else num_cpus // 2, 8))
        else:
            self.parallel_workers = parallel_workers
        
        # Determine datasets
        if representative_only or datasets == ['representative']:
            self.datasets = list(REPRESENTATIVE_DATASETS.values())
            self.task_types = {ds: task for task, ds in REPRESENTATIVE_DATASETS.items()}
        elif datasets:
            self.datasets = datasets
            # Infer task types
            self.task_types = {}
            for ds in datasets:
                for task, task_datasets in TASK_DATASETS.items():
                    if ds in task_datasets:
                        self.task_types[ds] = task
                        break
        else:
            # Use all datasets
            self.datasets = []
            self.task_types = {}
            for task, task_datasets in TASK_DATASETS.items():
                self.datasets.extend(task_datasets)
                for ds in task_datasets:
                    self.task_types[ds] = task
        
        print(f"\nDiffusion Study Configuration:")
        print(f"  Study Dimensions: {', '.join(self.study_dimensions)}")
        print(f"  Datasets: {len(self.datasets)} ({', '.join(self.datasets[:3])}{'...' if len(self.datasets) > 3 else ''})")
        print(f"  Seeds per config: {self.num_seeds}")
        print(f"  Parallel workers: {self.parallel_workers}")
        print(f"  Fast mode: {self.fast_mode}")
    
    def generate_experiment_queue(self) -> List[Tuple]:
        """
        Generate queue of all experiments to run
        
        Returns:
            List of (dataset_name, config, seed, verbose) tuples
        """
        queue = []
        
        for dataset_name in self.datasets:
            task_type = self.task_types.get(dataset_name, 'classification')
            
            # Get study configurations
            if self.fast_mode:
                study_configs = DiffusionStudySpace.get_fast_mode_configs(task_type)
            else:
                study_configs = DiffusionStudySpace.generate_all_study_configs(
                    task_type=task_type,
                    study_dimensions=self.study_dimensions
                )
            
            # Generate experiments for each config and seed
            for study_dim, configs in study_configs.items():
                for config in configs:
                    for seed in range(42, 42 + self.num_seeds):
                        # Only first experiment is verbose
                        verbose = False
                        queue.append((dataset_name, config.to_dict(), seed, verbose))
        
        return queue
    
    def run_with_mlflow(self, queue: List[Tuple]) -> Dict[str, List[Dict]]:
        """
        Run experiments with MLflow parent/child run tracking
        
        Args:
            queue: List of experiment tuples
        
        Returns:
            Dictionary mapping study_dimension to results
        """
        if not self.mlflow_logger:
            return self.run_without_mlflow(queue)
        
        # Group experiments by study dimension
        by_dimension = {}
        for dataset, config_dict, seed, verbose in queue:
            study_dim = config_dict['study_dimension']
            if study_dim not in by_dimension:
                by_dimension[study_dim] = []
            by_dimension[study_dim].append((dataset, config_dict, seed, verbose))
        
        all_results = {}
        
        # Run each study dimension as a parent run
        for study_dim, experiments in by_dimension.items():
            print(f"\n{'='*100}")
            print(f"Running {study_dim} Study ({len(experiments)} experiments)")
            print(f"{'='*100}\n")
            
            # Start parent run
            parent_run_id = self.mlflow_logger.start_parent_run(
                study_name=f"{study_dim}_study",
                study_dimension=study_dim,
                params={
                    'num_experiments': len(experiments),
                    'num_datasets': len(set(exp[0] for exp in experiments)),
                    'num_seeds': self.num_seeds,
                },
                tags={
                    'study_type': 'diffusion_parameter_study',
                    'fast_mode': str(self.fast_mode),
                }
            )
            
            # Run experiments (can't use multiprocessing with MLflow in child runs)
            # So we'll run sequentially within each parent run
            dimension_results = []
            for dataset, config_dict, seed, _ in tqdm(experiments, desc=f"{study_dim}"):
                # Start child run
                self.mlflow_logger.start_child_run(
                    dataset_name=dataset,
                    config_variant=config_dict['config_variant'],
                    seed=seed,
                    params=config_dict,
                    tags={'study_dimension': study_dim}
                )
                
                # Train
                result_dict = run_single_experiment((dataset, config_dict, seed, False))
                dimension_results.append(result_dict)
                
                # Log to MLflow
                if result_dict['success']:
                    self.mlflow_logger.log_result(result_dict['result'])
                
                # End child run
                self.mlflow_logger.end_child_run()
            
            # Aggregate results for parent run
            self.mlflow_logger.aggregate_child_runs(
                metric_names=['final_test_accuracy', 'final_test_loss', 'test_nmi', 'test_ari']
            )
            
            # End parent run
            self.mlflow_logger.end_run()
            
            all_results[study_dim] = dimension_results
        
        return all_results
    
    def run_without_mlflow(self, queue: List[Tuple]) -> Dict[str, List[Dict]]:
        """
        Run experiments in parallel without MLflow tracking
        
        Args:
            queue: List of experiment tuples
        
        Returns:
            Dictionary mapping study_dimension to results
        """
        print(f"\n{'='*100}")
        print(f"Running {len(queue)} Experiments in Parallel (workers={self.parallel_workers})")
        print(f"{'='*100}\n")
        
        # Run experiments in parallel
        with mp.Pool(processes=self.parallel_workers) as pool:
            results = list(tqdm(
                pool.imap(run_single_experiment, queue),
                total=len(queue),
                desc="Training"
            ))
        
        # Group results by study dimension
        by_dimension = {}
        for result in results:
            study_dim = result['study_dimension']
            if study_dim not in by_dimension:
                by_dimension[study_dim] = []
            by_dimension[study_dim].append(result)
        
        return by_dimension
    
    def run(self) -> Dict[str, List[Dict]]:
        """
        Execute all diffusion studies
        
        Returns:
            Dictionary mapping study_dimension to results
        """
        start_time = time.time()
        
        # Generate experiment queue
        queue = self.generate_experiment_queue()
        
        # Print execution plan
        plan = get_study_execution_plan(
            datasets=self.datasets,
            task_types=self.task_types,
            study_configs={dim: [] for dim in self.study_dimensions},
            num_seeds=self.num_seeds
        )
        
        print(f"\nExecution Plan:")
        print(f"  Total experiments: {len(queue)}")
        print(f"  Estimated time: {plan['estimated_hours']:.1f} hours")
        print(f"  (assuming ~5 min per experiment)\n")
        
        # Run experiments
        if self.mlflow_logger:
            results = self.run_with_mlflow(queue)
        else:
            results = self.run_without_mlflow(queue)
        
        # Print summary
        elapsed_time = time.time() - start_time
        total_success = sum(len([r for r in res if r['success']]) for res in results.values())
        total_failed = sum(len([r for r in res if not r['success']]) for res in results.values())
        
        print(f"\n{'='*100}")
        print(f"DIFFUSION STUDY COMPLETE")
        print(f"{'='*100}")
        print(f"  Total experiments: {len(queue)}")
        print(f"  Successful: {total_success}")
        print(f"  Failed: {total_failed}")
        print(f"  Elapsed time: {elapsed_time/3600:.2f} hours")
        print(f"{'='*100}\n")
        
        return results


def main():
    """CLI for running diffusion studies"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run HyperGRAND diffusion studies')
    parser.add_argument('--study-dimension', type=str, nargs='+',
                       choices=['integration_scheme', 'diffusion_depth', 'attention_mechanism', 'all'],
                       default=['all'],
                       help='Study dimensions to explore')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Datasets to use (default: all)')
    parser.add_argument('--representative-only', action='store_true',
                       help='Use only representative datasets (cora, contact_high_school, zoo)')
    parser.add_argument('--num-seeds', type=int, default=5,
                       help='Number of random seeds per configuration')
    parser.add_argument('--parallel-workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Use reduced configuration set for fast validation')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow tracking')
    parser.add_argument('--mlflow-experiment', type=str, default='HyperGRAND_Diffusion_Studies',
                       help='MLflow experiment name')
    parser.add_argument('--output', type=str, default='diffusion_study_results.json',
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Handle "all" dimension
    if 'all' in args.study_dimension:
        study_dimensions = ['integration_scheme', 'diffusion_depth', 'attention_mechanism']
    else:
        study_dimensions = args.study_dimension
    
    # Create MLflow logger if enabled
    mlflow_logger = None
    if not args.no_mlflow:
        try:
            from main import MLFlowLogger
            mlflow_logger = MLFlowLogger(
                enabled=True,
                experiment_name=args.mlflow_experiment
            )
        except ImportError:
            print("Warning: MLflow not available, running without tracking")
    
    # Create runner
    runner = DiffusionStudyRunner(
        study_dimensions=study_dimensions,
        datasets=args.datasets,
        num_seeds=args.num_seeds,
        parallel_workers=args.parallel_workers,
        fast_mode=args.fast_mode,
        representative_only=args.representative_only,
        mlflow_logger=mlflow_logger
    )
    
    # Run studies
    results = runner.run()
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
