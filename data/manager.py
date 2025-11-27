#!/usr/bin/env python3
"""
Unified Data Manager for HyperGRAND

This module provides a centralized interface for loading all datasets
as standardized PyTorch Geometric Data objects. It serves as the single
entry point for all data loading operations in the codebase.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from torch_geometric.data import Data

from .pyg_converter import PyGDatasetLoader, PyGDataProcessor, DatasetMetadata


class UnifiedDataManager:
    """
    Centralized data manager for all dataset operations in HyperGRAND.
    
    This class provides a unified interface to:
    - Load any dataset as a standardized PyG Data object
    - Manage dataset metadata and organization
    - Create and manipulate data splits
    - Access dataset information
    
    Example usage:
        manager = UnifiedDataManager()
        
        # Load a single dataset
        data = manager.load('planetoid_cora')
        
        # Load multiple datasets
        datasets = manager.load_multiple(['planetoid_cora', 'contact_high_school'])
        
        # Get dataset info
        info = manager.get_dataset_info('planetoid_cora')
        
        # Get all datasets for a task type
        classification_datasets = manager.get_datasets_by_task('classification')
    """
    
    def __init__(
        self,
        base_path: str = "datasets",
        auto_normalize: bool = True,
        seed: int = 42
    ):
        """
        Initialize the data manager
        
        Args:
            base_path: Base path for all datasets
            auto_normalize: Whether to normalize features
            seed: Random seed for reproducibility
        """
        self.base_path = Path(base_path)
        self.seed = seed
        self.loader = PyGDatasetLoader()
        self.processor = PyGDataProcessor()
        self._loaded_datasets = {}
    
    def load(
        self,
        dataset_name: str,
        strategy: Optional[str] = None,
        create_splits: bool = True,
        train_ratio: float = 0.5,
        val_ratio: float = 0.25,
        cache: bool = True,
        verbose: bool = True
    ) -> Data:
        """
        Load a dataset as a PyG Data object
        
        Args:
            dataset_name: Name of the dataset to load
            strategy: Task strategy ('classification' or 'clustering'). If None, inferred
            create_splits: Whether to create train/val/test splits
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            cache: Whether to cache loaded datasets
            verbose: Whether to print loading information
        
        Returns:
            torch_geometric.data.Data: Standardized PyG Data object
        
        Example:
            data = manager.load('planetoid_cora')
            print(data.metadata.task_type)  # 'classification'
        """
        # Check cache
        if cache and dataset_name in self._loaded_datasets:
            if verbose:
                print(f"Loading {dataset_name} from cache...")
            return self._loaded_datasets[dataset_name]
        
        if verbose:
            print(f"Loading {dataset_name}...")
        
        # Get dataset path
        dataset_path = self._get_dataset_path(dataset_name)
        
        # Load dataset
        data = self.loader.load_dataset(
            dataset_name=dataset_name,
            path=str(dataset_path),
            strategy=strategy,
            create_splits=create_splits,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=self.seed
        )
        
        # Cache if requested
        if cache:
            self._loaded_datasets[dataset_name] = data
        
        if verbose:
            self.processor.print_data_info(data)
        
        return data
    
    def load_multiple(
        self,
        dataset_names: List[str],
        create_splits: bool = True,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Data]:
        """
        Load multiple datasets
        
        Args:
            dataset_names: List of dataset names to load
            create_splits: Whether to create train/val/test splits
            verbose: Whether to print information
            **kwargs: Additional arguments to pass to load()
        
        Returns:
            Dictionary mapping dataset names to PyG Data objects
        
        Example:
            datasets = manager.load_multiple(['planetoid_cora', 'contact_high_school'])
        """
        datasets = {}
        
        for dataset_name in dataset_names:
            try:
                data = self.load(
                    dataset_name=dataset_name,
                    create_splits=create_splits,
                    verbose=verbose,
                    **kwargs
                )
                datasets[dataset_name] = data
            except Exception as e:
                if verbose:
                    print(f"✗ Error loading {dataset_name}: {str(e)}")
                continue
        
        return datasets
    
    def load_by_task(
        self,
        task_type: str,
        create_splits: bool = True,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Data]:
        """
        Load all datasets for a specific task type
        
        Args:
            task_type: 'classification', 'clustering', or 'partitioning'
            create_splits: Whether to create train/val/test splits
            verbose: Whether to print information
            **kwargs: Additional arguments
        
        Returns:
            Dictionary of all datasets for the task type
        
        Example:
            clustering_datasets = manager.load_by_task('clustering')
        """
        dataset_names = self.get_datasets_by_task(task_type)
        return self.load_multiple(
            dataset_names=dataset_names,
            create_splits=create_splits,
            verbose=verbose,
            **kwargs
        )
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a dataset without loading it
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Dictionary with dataset information
        """
        return self.loader._get_dataset_metadata(dataset_name) or {}
    
    def get_datasets_by_task(self, task_type: str) -> List[str]:
        """
        Get all dataset names for a specific task type
        
        Args:
            task_type: 'classification', 'clustering', or 'partitioning'
        
        Returns:
            List of dataset names
        
        Example:
            datasets = manager.get_datasets_by_task('clustering')
            # Returns: ['contact_high_school', 'walmart_trips', ...]
        """
        metadata = self.loader._metadata_cache
        if not metadata or task_type not in metadata:
            return []
        
        return list(metadata[task_type].get('datasets', {}).keys())
    
    def list_datasets(self, task_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all available datasets, optionally filtered by task type
        
        Args:
            task_type: Optional filter by task type
        
        Returns:
            Dictionary of datasets organized by task type
        """
        if task_type:
            return {task_type: self.get_datasets_by_task(task_type)}
        
        result = {}
        for task in ['classification', 'clustering', 'partitioning', 'other']:
            datasets = self.get_datasets_by_task(task)
            if datasets:
                result[task] = datasets
        
        return result
    
    def print_available_datasets(self, task_type: Optional[str] = None):
        """Print available datasets"""
        datasets = self.list_datasets(task_type)
        
        print("\n" + "="*70)
        print("AVAILABLE DATASETS")
        print("="*70)
        
        for task, names in datasets.items():
            print(f"\n{task.upper()} ({len(names)} datasets):")
            for name in names:
                print(f"  - {name}")
    
    def _get_dataset_path(self, dataset_name: str) -> Path:
        """Get the filesystem path for a dataset"""
        metadata = self.loader._get_dataset_metadata(dataset_name)
        
        if not metadata:
            # Try standard path conventions
            dataset_path = self.base_path / dataset_name.replace('_', '-')
            if dataset_path.exists():
                return dataset_path
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        return self.base_path / metadata['path']
    
    def verify_datasets(self, dataset_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Verify integrity of datasets
        
        Args:
            dataset_names: List of datasets to verify. If None, verify all
        
        Returns:
            Dictionary of verification results
        """
        if dataset_names is None:
            # Get all datasets
            all_datasets = self.list_datasets()
            dataset_names = []
            for datasets in all_datasets.values():
                dataset_names.extend(datasets)
        
        results = {}
        
        for dataset_name in dataset_names:
            try:
                print(f"Verifying {dataset_name}...", end=" ")
                data = self.load(dataset_name, verbose=False)
                is_valid, issues = self.processor.verify_data_integrity(data)
                
                results[dataset_name] = {
                    'valid': is_valid,
                    'issues': issues
                }
                
                if is_valid:
                    print(" ")
                else:
                    print("✗")
                    for issue in issues:
                        print(f"    - {issue}")
            
            except Exception as e:
                results[dataset_name] = {
                    'valid': False,
                    'error': str(e)
                }
                print(f"✗ Error: {str(e)}")
        
        return results
    
    def clear_cache(self):
        """Clear the loaded datasets cache"""
        self._loaded_datasets.clear()
    
    def get_cached_datasets(self) -> List[str]:
        """Get list of cached datasets"""
        return list(self._loaded_datasets.keys())


# Convenience function for quick dataset loading
def load_dataset(
    dataset_name: str,
    base_path: str = "datasets",
    **kwargs
) -> Data:
    """
    Quick load function for a single dataset
    
    Args:
        dataset_name: Name of the dataset
        base_path: Base path for datasets
        **kwargs: Additional arguments
    
    Returns:
        PyG Data object
    
    Example:
        data = load_dataset('planetoid_cora')
    """
    manager = UnifiedDataManager(base_path=base_path)
    return manager.load(dataset_name, **kwargs)


def load_datasets(
    dataset_names: List[str],
    base_path: str = "datasets",
    **kwargs
) -> Dict[str, Data]:
    """
    Quick load function for multiple datasets
    
    Args:
        dataset_names: List of dataset names
        base_path: Base path for datasets
        **kwargs: Additional arguments
    
    Returns:
        Dictionary of PyG Data objects
    
    Example:
        datasets = load_datasets(['planetoid_cora', 'contact_high_school'])
    """
    manager = UnifiedDataManager(base_path=base_path)
    return manager.load_multiple(dataset_names, **kwargs)


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("\n" + "="*70)
    print("UNIFIED DATA MANAGER - EXAMPLE USAGE")
    print("="*70)
    
    # Initialize manager
    manager = UnifiedDataManager()
    
    # List available datasets
    manager.print_available_datasets()
    
    # Load a single dataset
    print("\n" + "-"*70)
    print("Loading a single dataset...")
    print("-"*70)
    
    # data = manager.load('planetoid_cora')
    # print(f"\nMetadata:")
    # print(f"  Task Type: {data.metadata.task_type}")
    # print(f"  Strategy: {data.metadata.strategy}")
    # print(f"  Classes: {data.metadata.num_classes}")
    
    # Load multiple datasets
    print("\n" + "-"*70)
    print("Loading multiple datasets...")
    print("-"*70)
    
    # datasets = manager.load_multiple(['planetoid_cora', 'contact_high_school'], verbose=True)
    # print(f"\nLoaded {len(datasets)} datasets")
    
    # Verify datasets
    print("\n" + "-"*70)
    print("Verifying dataset integrity...")
    print("-"*70)
    
    # results = manager.verify_datasets(['planetoid_cora'])
    # for dataset_name, result in results.items():
    #     print(f"{dataset_name}: {'  Valid' if result['valid'] else '✗ Invalid'}")
