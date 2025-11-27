#!/usr/bin/env python3
"""
PyTorch Geometric Data Converter for HyperGRAND Datasets

This module provides utilities to convert all hypergraph datasets into
standardized PyTorch Geometric Data objects, ensuring consistency across
the entire codebase regardless of the dataset source or format.

Benefits:
- Standardized data format across all datasets
- Easy integration with PyG-based code
- Automatic feature generation when missing
- Consistent hyperedge indexing
- Metadata preservation
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToUndirected
import json
from dataclasses import dataclass


@dataclass
class DatasetMetadata:
    """Metadata for a dataset"""
    name: str
    task_type: str  # 'classification', 'clustering', 'partitioning'
    strategy: str  # 'classification' or 'clustering'
    num_nodes: int
    num_hyperedges: int
    num_classes: int
    num_features: int
    mean_hyperedge_size: float
    max_hyperedge_size: int
    min_hyperedge_size: int
    source: str = "Unknown"
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary"""
        return {
            'name': self.name,
            'task_type': self.task_type,
            'strategy': self.strategy,
            'num_nodes': self.num_nodes,
            'num_hyperedges': self.num_hyperedges,
            'num_classes': self.num_classes,
            'num_features': self.num_features,
            'mean_hyperedge_size': self.mean_hyperedge_size,
            'max_hyperedge_size': self.max_hyperedge_size,
            'min_hyperedge_size': self.min_hyperedge_size,
            'source': self.source
        }


class HypergraphDataConverter:
    """
    Converts hypergraph data into standardized PyTorch Geometric Data objects.
    
    This class handles:
    - Feature generation/normalization
    - Hyperedge indexing
    - Label encoding
    - Train/val/test split creation
    - Metadata generation
    """
    
    def __init__(self, auto_normalize: bool = True):
        """
        Initialize converter
        
        Args:
            auto_normalize: Whether to normalize features to [0, 1]
        """
        self.auto_normalize = auto_normalize
    
    def convert(
        self,
        node_features: Optional[torch.Tensor],
        hyperedge_index: torch.Tensor,
        labels: torch.Tensor,
        num_nodes: int,
        num_hyperedges: int,
        num_classes: int,
        dataset_name: str,
        task_type: str,
        strategy: str,
        label_names: Optional[List[str]] = None,
        source: str = "Unknown",
        train_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        test_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Data:
        """
        Convert hypergraph data to PyG Data object
        
        Args:
            node_features: Node feature matrix [num_nodes, num_features]
            hyperedge_index: Hyperedge index [2, num_hyperedges_incidences]
            labels: Node labels [num_nodes]
            num_nodes: Total number of nodes
            num_hyperedges: Total number of hyperedges
            num_classes: Number of classes
            dataset_name: Name of the dataset
            task_type: 'classification', 'clustering', or 'partitioning'
            strategy: 'classification' or 'clustering' for training
            label_names: Optional list of class names
            source: Source/description of the dataset
            train_mask: Optional training mask
            val_mask: Optional validation mask
            test_mask: Optional test mask
            **kwargs: Additional metadata
        
        Returns:
            torch_geometric.data.Data: Standardized PyG Data object
        """
        
        # Generate features if not provided
        if node_features is None:
            node_features = torch.eye(num_nodes, dtype=torch.float32)
            num_features = num_nodes
        else:
            num_features = node_features.shape[1]
            # Ensure float32
            if node_features.dtype != torch.float32:
                node_features = node_features.float()
        
        # Normalize features if needed
        if self.auto_normalize and node_features.min() < 0:
            # Min-max normalization
            node_features = self._normalize_features(node_features)
        
        # Ensure labels are long tensors
        labels = labels.long()
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            y=labels,
            num_nodes=num_nodes
        )
        
        # Add hyperedge index as edge_index (for compatibility with PyG)
        # Note: This is the incidence matrix representation [2, total_incidences]
        data.hyperedge_index = hyperedge_index
        
        # Compute and store hyperedge sizes
        hyperedge_sizes = self._compute_hyperedge_sizes(hyperedge_index, num_hyperedges)
        data.hyperedge_sizes = hyperedge_sizes
        
        # Add masks if provided
        if train_mask is not None:
            data.train_mask = train_mask.bool()
        if val_mask is not None:
            data.val_mask = val_mask.bool()
        if test_mask is not None:
            data.test_mask = test_mask.bool()
        
        # Create metadata
        metadata = DatasetMetadata(
            name=dataset_name,
            task_type=task_type,
            strategy=strategy,
            num_nodes=num_nodes,
            num_hyperedges=num_hyperedges,
            num_classes=num_classes,
            num_features=num_features,
            mean_hyperedge_size=float(hyperedge_sizes.float().mean()),
            max_hyperedge_size=int(hyperedge_sizes.max()),
            min_hyperedge_size=int(hyperedge_sizes.min()),
            source=source
        )
        
        # Store metadata as attributes
        data.metadata = metadata
        data.label_names = label_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Store additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(data, key, value)
        
        return data
    
    @staticmethod
    def _normalize_features(features: torch.Tensor) -> torch.Tensor:
        """Normalize features to [0, 1] using min-max scaling"""
        feat_min = features.min(dim=0, keepdim=True)[0]
        feat_max = features.max(dim=0, keepdim=True)[0]
        feat_range = feat_max - feat_min
        feat_range[feat_range == 0] = 1  # Avoid division by zero
        return (features - feat_min) / feat_range
    
    @staticmethod
    def _compute_hyperedge_sizes(
        hyperedge_index: torch.Tensor,
        num_hyperedges: int
    ) -> torch.Tensor:
        """Compute the size (number of nodes) in each hyperedge"""
        sizes = torch.zeros(num_hyperedges, dtype=torch.long)
        if hyperedge_index.numel() > 0:
            hyperedge_ids = hyperedge_index[0]
            sizes = torch.bincount(hyperedge_ids, minlength=num_hyperedges)
        return sizes


class PyGDatasetLoader:
    """
    Standardized dataset loader that outputs PyG Data objects.
    
    This class wraps the existing dataset loading infrastructure and
    converts everything to PyG format for consistency.
    """
    
    def __init__(self):
        self.converter = HypergraphDataConverter(auto_normalize=True)
        self._metadata_cache = self._load_dataset_metadata()
    
    @staticmethod
    def _load_dataset_metadata() -> Dict:
        """Load dataset metadata from JSON"""
        metadata_path = Path("datasets/DATASET_METADATA.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def load_dataset(
        self,
        dataset_name: str,
        path: Union[str, Path],
        strategy: Optional[str] = None,
        create_splits: bool = True,
        train_ratio: float = 0.5,
        val_ratio: float = 0.25,
        seed: int = 42
    ) -> Data:
        """
        Load a dataset and convert to PyG Data object
        
        Args:
            dataset_name: Name of the dataset (e.g., 'planetoid_cora', 'contact_high_school')
            path: Path to the dataset
            strategy: Task strategy ('classification' or 'clustering'). If None, inferred from metadata
            create_splits: Whether to create train/val/test splits
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            seed: Random seed for reproducibility
        
        Returns:
            torch_geometric.data.Data: Standardized PyG Data object with metadata
        """
        from data.dataset import create_hypergraph_dataset, DataSplitter
        
        # Load using existing infrastructure
        dataset_loader = create_hypergraph_dataset(dataset_name)
        hyper_data = dataset_loader.load_data(path)
        
        # Determine task type and strategy
        task_type = self._infer_task_type(dataset_name)
        if strategy is None:
            strategy = self._infer_strategy(dataset_name, task_type)
        
        # Get metadata
        metadata_entry = self._get_dataset_metadata(dataset_name)
        source = metadata_entry.get('source', 'Unknown') if metadata_entry else 'Unknown'
        
        # Create splits if needed and not already present
        train_mask = getattr(hyper_data, 'train_mask', None)
        val_mask = getattr(hyper_data, 'val_mask', None)
        test_mask = getattr(hyper_data, 'test_mask', None)
        
        if create_splits and (train_mask is None or val_mask is None or test_mask is None):
            train_mask, val_mask, test_mask = DataSplitter.create_transductive_split(
                hyper_data.labels,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                random_state=seed
            )
        
        # Convert to PyG Data
        pyg_data = self.converter.convert(
            node_features=hyper_data.node_features,
            hyperedge_index=hyper_data.hyperedge_index,
            labels=hyper_data.labels,
            num_nodes=hyper_data.num_nodes,
            num_hyperedges=hyper_data.num_hyperedges,
            num_classes=hyper_data.num_classes,
            dataset_name=dataset_name,
            task_type=task_type,
            strategy=strategy,
            label_names=hyper_data.label_names,
            source=source,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            **hyper_data.dataset_info
        )
        
        return pyg_data
    
    @staticmethod
    def _infer_task_type(dataset_name: str) -> str:
        """Infer task type from dataset name"""
        dataset_name = dataset_name.lower()
        
        # Classification datasets
        if any(x in dataset_name for x in ['planetoid', 'cora', 'citeseer', 'pubmed', 'house_committees']):
            return 'classification'
        
        # Clustering datasets
        if any(x in dataset_name for x in ['contact', 'walmart', 'stackoverflow', 'amazon']):
            return 'clustering'
        
        # Partitioning datasets
        if any(x in dataset_name for x in ['zoo', 'mushroom', 'ntu', 'modelnet']):
            return 'partitioning'
        
        # Default to clustering for unknown
        return 'clustering'
    
    @staticmethod
    def _infer_strategy(dataset_name: str, task_type: str) -> str:
        """Infer strategy from task type"""
        if task_type == 'classification':
            return 'classification'
        else:  # clustering and partitioning
            return 'clustering'
    
    def _get_dataset_metadata(self, dataset_name: str) -> Optional[Dict]:
        """Get metadata for a dataset from the metadata file"""
        if not self._metadata_cache:
            return None
        
        for task_type in ['classification', 'clustering', 'partitioning', 'other']:
            if task_type in self._metadata_cache:
                datasets = self._metadata_cache[task_type].get('datasets', {})
                if dataset_name in datasets:
                    return datasets[dataset_name]
        
        return None
    
    def batch_load_datasets(
        self,
        dataset_names: List[str],
        base_path: str = "datasets",
        create_splits: bool = True,
        seed: int = 42
    ) -> Dict[str, Data]:
        """
        Load multiple datasets in batch
        
        Args:
            dataset_names: List of dataset names to load
            base_path: Base path for all datasets
            create_splits: Whether to create train/val/test splits
            seed: Random seed
        
        Returns:
            Dictionary mapping dataset names to PyG Data objects
        """
        datasets = {}
        
        for dataset_name in dataset_names:
            print(f"Loading {dataset_name}...", end=" ")
            try:
                path = Path(base_path) / self._get_dataset_metadata(dataset_name)['path']
                data = self.load_dataset(
                    dataset_name=dataset_name,
                    path=str(path),
                    create_splits=create_splits,
                    seed=seed
                )
                datasets[dataset_name] = data
                print(" ")
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                continue
        
        return datasets


class PyGDataProcessor:
    """
    Utilities for processing and analyzing PyG Data objects
    """
    
    @staticmethod
    def print_data_info(data: Data, verbose: bool = False):
        """Print information about a PyG Data object"""
        metadata = data.metadata if hasattr(data, 'metadata') else None
        
        print(f"\n{'='*60}")
        print(f"Dataset: {metadata.name if metadata else 'Unknown'}")
        print(f"{'='*60}")
        print(f"Nodes: {data.num_nodes}")
        print(f"Features: {data.x.shape[1]}")
        print(f"Classes: {data.y.max().item() + 1}")
        
        if hasattr(data, 'hyperedge_index'):
            print(f"Hyperedges: {(data.hyperedge_index[0].max() + 1).item()}")
            print(f"Incidences: {data.hyperedge_index.shape[1]}")
        
        if metadata:
            print(f"Task Type: {metadata.task_type}")
            print(f"Strategy: {metadata.strategy}")
            print(f"Avg Hyperedge Size: {metadata.mean_hyperedge_size:.2f}")
            print(f"Max Hyperedge Size: {metadata.max_hyperedge_size}")
            print(f"Source: {metadata.source}")
        
        if hasattr(data, 'train_mask'):
            n_train = data.train_mask.sum().item()
            n_val = data.val_mask.sum().item() if hasattr(data, 'val_mask') else 0
            n_test = data.test_mask.sum().item() if hasattr(data, 'test_mask') else 0
            print(f"\nSplit: Train={n_train}, Val={n_val}, Test={n_test}")
        
        if verbose and hasattr(data, 'label_names'):
            print(f"\nClass Names: {data.label_names}")
    
    @staticmethod
    def get_split_info(data: Data) -> Dict:
        """Get information about dataset splits"""
        info = {}
        
        if hasattr(data, 'train_mask'):
            info['train'] = {
                'count': data.train_mask.sum().item(),
                'ratio': (data.train_mask.sum().item() / data.num_nodes)
            }
        
        if hasattr(data, 'val_mask'):
            info['val'] = {
                'count': data.val_mask.sum().item(),
                'ratio': (data.val_mask.sum().item() / data.num_nodes)
            }
        
        if hasattr(data, 'test_mask'):
            info['test'] = {
                'count': data.test_mask.sum().item(),
                'ratio': (data.test_mask.sum().item() / data.num_nodes)
            }
        
        return info
    
    @staticmethod
    def verify_data_integrity(data: Data) -> Tuple[bool, List[str]]:
        """
        Verify that a PyG Data object is valid and complete
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        if data.x is None:
            issues.append("Missing node features (x)")
        if data.y is None:
            issues.append("Missing labels (y)")
        if not hasattr(data, 'hyperedge_index'):
            issues.append("Missing hyperedge_index")
        
        # Check consistency
        if data.x is not None and data.x.shape[0] != data.num_nodes:
            issues.append(f"Feature matrix size ({data.x.shape[0]}) != num_nodes ({data.num_nodes})")
        
        if data.y is not None and data.y.shape[0] != data.num_nodes:
            issues.append(f"Label size ({data.y.shape[0]}) != num_nodes ({data.num_nodes})")
        
        # Check metadata
        if not hasattr(data, 'metadata'):
            issues.append("Missing metadata")
        if not hasattr(data, 'label_names'):
            issues.append("Missing label_names")
        
        return (len(issues) == 0, issues)


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the PyG Data converter
    """
    loader = PyGDatasetLoader()
    processor = PyGDataProcessor()
    
    # Load a single dataset
    print("Loading Cora dataset...")
    cora_data = loader.load_dataset(
        dataset_name='planetoid_cora',
        path='datasets/cora',
        strategy='classification'
    )
    
    processor.print_data_info(cora_data, verbose=True)
    
    # Verify integrity
    is_valid, issues = processor.verify_data_integrity(cora_data)
    if is_valid:
        print("\n  Data integrity check passed")
    else:
        print("\n✗ Data integrity issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Load multiple datasets
    print("\n" + "="*60)
    print("Loading multiple datasets...")
    print("="*60)
    
    datasets_to_load = [
        'planetoid_cora',
        'contact_high_school',
        'walmart_trips'
    ]
    
    # batch_datasets = loader.batch_load_datasets(datasets_to_load)
    # for name, data in batch_datasets.items():
    #     processor.print_data_info(data)
