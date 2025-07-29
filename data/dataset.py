from abc import ABC, abstractmethod
from sklearn.utils.validation import validate_data
import torch
import os
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from torch_geometric.datasets import Planetoid


@dataclass
class HypergraphData:
    """Data container for hypergraph datasets"""
    node_features: torch.Tensor
    labels: torch.Tensor
    hyperedge_index: torch.Tensor
    num_nodes: int
    num_hyperedges: int
    num_classes: int
    label_names: List[str]
    dataset_info: Dict[str, Union[str, int, float]]


class DataLoader(ABC):
    """Abstract interface for data loading operations"""

    @abstractmethod
    def load_raw_data(self, path: Union[str, Path]) -> Dict:
        """Load raw data from files"""
        pass

class PlanetoidDataLoader(DataLoader):
    """Data loader for PyTorch Geometric Planetoid datasets"""

    def __init__(self, dataset_name: str = 'Cora'):
        """
        Initialize Planetoid data loader

        Args:
            dataset_name: Name of the dataset ('Cora', 'CiteSeer', 'PubMed')
        """
        self.dataset_name = dataset_name
        valid_datasets = ['Cora', 'CiteSeer', 'PubMed']
        if dataset_name not in valid_datasets:
            raise ValueError(f"Dataset {dataset_name} not supported. Please choose from {valid_datasets}")

    def load_raw_data(self, path: Union[str, Path]) -> Dict:
        """Load Planetoid dataset using PyTorch Geometric"""
        path = Path(path).expanduser()

        dataset = Planetoid(root=str(path), name=self.dataset_name)
        data = dataset[0]

        features = data.x.numpy() if data.x is not None else None
        labels = data.y.numpy()
        edges = data.edge_index.numpy()

        train_mask = data.train_mask.numpy() if hasattr(data, 'train_mask') else None
        val_mask = data.val_mask.numpy() if hasattr(data, 'val_mask') else None
        test_mask = data.test_mask.numpy() if hasattr(data, 'test_mask') else None

        return {
            'features': features,
            'labels': labels,
            'edges': edges,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'num_classes': dataset.num_classes,
            'num_features': dataset.num_features,
            'dataset_name': self.dataset_name.lower(),
            'dataset_info': {
                'num_nodes': data.num_nodes,
                'num_edges': data.num_edges,
                'avg_degree': data.num_edges / data.num_nodes,
                'is_undirected': data.is_undirected(),
                'has_self_loops': data.has_self_loops(),
                'has_isolated_nodes': data.has_isolated_nodes()
            }
        }


class HypergraphConverter(ABC):
    """Abstract interface for converting data to hypergraph format"""

    @abstractmethod
    def convert_to_hypergraph(self, data: Dict, **kwargs) -> List[List[int]]:
        """Convert input data to hyperedge format"""
        pass


class DataValidator:
    """Validates hypergraph data integrity"""

    @staticmethod
    def validate_node_indices(hyperedges: List[List[int]], num_nodes: int) -> bool:
        """Validate that all node indices are within valid range"""
        for edge in hyperedges:
            for node_id in edge:
                if node_id < 0 or node_id >= num_nodes:
                    raise ValueError(f"Node ID {node_id} is out of bounds [0, {num_nodes-1}]")
        return True

    @staticmethod
    def validate_hypergraph_data(data: HypergraphData) -> bool:
        """Validate complete hypergraph data structure"""
        assert data.node_features.shape[0] == data.num_nodes, "Feature matrix size mismatch"
        assert len(data.labels) == data.num_nodes, "Labels size mismatch"
        assert data.hyperedge_index.shape[0] == 2, "Invalid hyperedge index format"
        assert len(data.label_names) == data.num_classes, "Label names count mismatch"
        return True


class ContactDataLoader(DataLoader):
    """Data loader for contact network datasets"""

    def load_raw_data(self, path: Union[str, Path]) -> Dict:
        """Load contact network data from text files"""
        path = Path(path).expanduser()
        dataset_name = path.name if path.is_dir() else path.stem

        node_labels_file = path / f"node-labels-{dataset_name}.txt"

        with open(node_labels_file, 'r') as f:
            labels = [line.strip() for line in f]

        hyperedges_file = path / f"hyperedges-{dataset_name}.txt"
        if not hyperedges_file.exists():
            hyperedges_file = path / "hyperedges.txt"

        hyperedges = []
        with open(hyperedges_file, 'r') as f:
            for line in f:
                nodes = [int(x) for x in line.strip().split(',')]
                hyperedges.append(nodes)

        # Load label names (optional)
        label_names_file = path / f"label-names-{dataset_name}.txt"
        if not label_names_file.exists():
            label_names_file = path / "label-names.txt"

        try:
            with open(label_names_file, 'r') as f:
                label_names = [line.strip() for line in f]
        except FileNotFoundError:
            num_classes = len(set(labels))
            label_names = [f"Class_{i}" for i in range(num_classes)]

        return {
            'labels': labels,
            'hyperedges': hyperedges,
            'label_names': label_names,
            'dataset_name': dataset_name
        }

class CoraDataLoader(DataLoader):
    """Data loader for Cora dataset"""

    def load_raw_data(self, path: Union[str, Path]) -> Dict:
        """Load Cora dataset from content and cites files"""
        path = Path(path).expanduser()

        # Load content file
        content_path = path / 'cora.content'
        if not content_path.exists():
            raise FileNotFoundError(f"Cora content file not found at {content_path}")

        content_data = []
        with open(content_path, 'r') as f:
            for line in f:
                content_data.append(line.strip().split('\t'))

        content_df = pd.DataFrame(content_data)

        # Extract components
        node_ids = content_df.iloc[:, 0].values
        features = content_df.iloc[:, 1:-1].values.astype(float)
        labels = content_df.iloc[:, -1].values

        # Load edges
        cites_path = path / 'cora.cites'
        edges = []

        if cites_path.exists():
            with open(cites_path, 'r') as f:
                for line in f:
                    edge = line.strip().split('\t')
                    edges.append(edge)

        # Create node mapping
        node_id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # Convert edges to indices
        edge_list = []
        for edge in edges:
            if len(edge) == 2 and edge[0] in node_id_map and edge[1] in node_id_map:
                edge_list.append([node_id_map[edge[0]], node_id_map[edge[1]]])

        edge_array = np.array(edge_list).T if edge_list else np.array([[], []]).reshape(2, 0)

        return {
            'features': features,
            'labels': labels,
            'edges': edge_array,
            'node_ids': node_ids
        }


class GraphToHypergraphConverter(HypergraphConverter):
    """Converts graph data to hypergraph format using various strategies"""

    def __init__(self, strategy: str = 'star_expansion'):
        self.strategy = strategy
        self.strategy_map = {
            'star_expansion': self._star_expansion,
            'clique_expansion': self._clique_expansion,
            'neighborhood_expansion': self._neighborhood_expansion
        }

        if strategy not in self.strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategy_map.keys())}")

    def convert_to_hypergraph(self, data: Dict, **kwargs) -> List[List[int]]:
        """Convert graph edges to hyperedges using specified strategy"""
        edge_array = data['edges']
        num_nodes = len(data['labels'])

        converter_func = self.strategy_map[self.strategy]
        return converter_func(edge_array, num_nodes, **kwargs)

    def _star_expansion(self, edge_array: np.ndarray, num_nodes: int, **kwargs) -> List[List[int]]:
        """Convert each edge to a 2-node hyperedge"""
        if edge_array.shape[1] == 0:
            return []

        hyperedges = []
        for i in range(edge_array.shape[1]):
            u, v = int(edge_array[0, i]), int(edge_array[1, i])
            hyperedges.append([u, v])

        return hyperedges

    def _clique_expansion(self, edge_array: np.ndarray, num_nodes: int, 
                         max_clique_size: int = 4, **kwargs) -> List[List[int]]:
        """Create hyperedges from triangles and small cliques"""
        if edge_array.shape[1] == 0:
            return []

        # Build adjacency list
        adj_list = defaultdict(set)
        for i in range(edge_array.shape[1]):
            u, v = int(edge_array[0, i]), int(edge_array[1, i])
            adj_list[u].add(v)
            adj_list[v].add(u)

        hyperedges = []
        processed_cliques = set()

        # Find triangles
        for u in adj_list:
            for v in adj_list[u]:
                if v > u:
                    common_neighbors = adj_list[u] & adj_list[v]
                    for w in common_neighbors:
                        if w > v:
                            triangle = tuple(sorted([u, v, w]))
                            if triangle not in processed_cliques:
                                hyperedges.append(list(triangle))
                                processed_cliques.add(triangle)

        # Fallback to star expansion if no triangles found
        if not hyperedges:
            return self._star_expansion(edge_array, num_nodes)

        return hyperedges

    def _neighborhood_expansion(self, edge_array: np.ndarray, num_nodes: int,
                               max_neighborhood_size: int = 5, **kwargs) -> List[List[int]]:
        """Create hyperedges from node neighborhoods"""
        if edge_array.shape[1] == 0:
            return []
        
        # Build adjacency list
        adj_list = defaultdict(set)
        for i in range(edge_array.shape[1]):
            u, v = int(edge_array[0, i]), int(edge_array[1, i])
            adj_list[u].add(v)
            adj_list[v].add(u)
        
        hyperedges = []
        
        for node in range(num_nodes):
            neighbors = list(adj_list[node])
            if len(neighbors) > 0:
                neighborhood = [node] + neighbors[:max_neighborhood_size-1]
                if len(neighborhood) >= 2:
                    hyperedges.append(sorted(neighborhood))
        
        # Remove duplicates
        unique_hyperedges = []
        seen = set()
        for he in hyperedges:
            he_tuple = tuple(he)
            if he_tuple not in seen:
                unique_hyperedges.append(he)
                seen.add(he_tuple)
        
        return unique_hyperedges


class IdentityHypergraphConverter(HypergraphConverter):
    """Pass-through converter for data already in hypergraph format"""
    
    def convert_to_hypergraph(self, data: Dict, **kwargs) -> List[List[int]]:
        """Return hyperedges as-is, with optional index normalization"""
        hyperedges = data['hyperedges']
        normalize_indices = kwargs.get('normalize_indices', True)
        
        if not normalize_indices:
            return hyperedges
        
        # Normalize to 0-indexed if needed
        min_index = min(min(edge) for edge in hyperedges) if hyperedges else 0
        if min_index > 0:
            hyperedges = [[node - min_index for node in edge] for edge in hyperedges]
        
        return hyperedges


class HypergraphDataset(ABC):
    """Abstract base class for hypergraph datasets"""
    
    def __init__(self, data_loader: DataLoader, converter: HypergraphConverter):
        self.data_loader = data_loader
        self.converter = converter
        self.validator = DataValidator()
        self._data: Optional[HypergraphData] = None
    
    @abstractmethod
    def load_data(self, path: Union[str, Path], **kwargs) -> HypergraphData:
        """Load and process dataset"""
        pass
    
    def get_data(self) -> HypergraphData:
        """Get the loaded hypergraph data"""
        if self._data is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self._data
    
    def _create_hyperedge_index(self, hyperedges: List[List[int]]) -> torch.Tensor:
        """Create hyperedge index tensor from hyperedge list"""
        edge_indices = []
        node_indices = []
        
        for edge_id, nodes in enumerate(hyperedges):
            for node_id in nodes:
                edge_indices.append(edge_id)
                node_indices.append(node_id)
        
        return torch.tensor([edge_indices, node_indices], dtype=torch.long)
    
    def _compute_dataset_stats(self, hyperedges: List[List[int]], 
                              num_nodes: int) -> Dict[str, Union[str, int, float]]:
        """Compute dataset statistics"""
        if not hyperedges:
            return {
                'mean_hyperedge_size': 0.0,
                'max_hyperedge_size': 0,
                'min_hyperedge_size': 0
            }
        
        hyperedge_sizes = [len(edge) for edge in hyperedges]
        return {
            'mean_hyperedge_size': sum(hyperedge_sizes) / len(hyperedge_sizes),
            'max_hyperedge_size': max(hyperedge_sizes),
            'min_hyperedge_size': min(hyperedge_sizes),
            'total_incidences': sum(hyperedge_sizes)
        }


class PlanetoidHypergraphDataset(HypergraphDataset):
    """Dataset class for Planetoid data converted to hypergraph format"""

    def __init__(self, dataset_name: str = 'Cora', hypergraph_strategy: str = 'star_expansion'):
        """
        Initialize Planetoid hypergraph dataset

        Args:
            dataset_name: Name of the Planetoid dataset ('Cora', 'CiteSeer', 'PubMed')
            hypergraph_strategy: Strategy for converting graph to hypergraph ('star_expansion', 'clique_expansion', 'neighbourhood_expansion')
        """
        super().__init__(
            data_loader=PlanetoidDataLoader(dataset_name),
            converter=GraphToHypergraphConverter(hypergraph_strategy)
        )
        self.dataset_name = dataset_name
        self.hypergraph_strategy = hypergraph_strategy

    def load_data(self, path: Union[str, Path], **kwargs) -> HypergraphData:
        """Load Planetoid dataset and convert to hypergraph format"""

        raw_data = self.data_loader.load_raw_data(path)

        labels = torch.tensor(raw_data['labels'], dtype=torch.long)
        num_nodes = len(labels)
        num_classes = raw_data['num_classes']

        label_names = [f"Class_{i}" for i in range(num_classes)]

        if raw_data['features'] is not None:
            node_features = torch.tensor(raw_data['features'], dtype=torch.float32)
        else:
            # Fallback to identity matrix if there are no features
            node_features = torch.eye(num_nodes)

        hyperedges = self.converter.convert_to_hypergraph(raw_data, **kwargs)

        self.validator.validate_node_indices(hyperedges, num_nodes)

        hyperedge_index = self._create_hyperedge_index(hyperedges)

        stats = self._compute_dataset_stats(hyperedges, num_nodes)
        stats.update({
            'dataset_name': raw_data['dataset_name'],
            'dataset_type': 'planetoid_citation_network',
            'hypergraph_strategy': self.hypergraph_strategy,
            'feature_dim': node_features.shape[1],
            'original_num_edges': raw_data['dataset_info']['num_edges'],
            'original_avg_degree': raw_data['dataset_info']['avg_degree'],
            'is_undirected': raw_data['dataset_info']['is_undirected']
        })

        self._data = HypergraphData(
            node_features=node_features,
            labels=labels,
            hyperedge_index=hyperedge_index,
            num_nodes=num_nodes,
            num_hyperedges=len(hyperedges),
            num_classes=num_classes,
            label_names=label_names,
            dataset_info=stats
        )

        self._data.train_mask = torch.tensor(raw_data['train_mask']) if raw_data['train_mask'] is not None else None
        self._data.val_mask = torch.tensor(raw_data['val_mask']) if raw_data['val_mask'] is not None else None
        self._data.test_mask = torch.tensor(raw_data['test_mask']) if raw_data['test_mask'] is not None else None

        self.validator.validate_hypergraph_data(self._data)

        self._print_dataset_info()

        return self._data

    def _print_dataset_info(self):
        """Print dataset information"""
        data = self._data
        info = data.dataset_info
        
        print(f"{self.dataset_name} (Planetoid) hypergraph dataset loaded:")
        print(f"  - Nodes: {data.num_nodes}")
        print(f"  - Node features: {info['feature_dim']}")
        print(f"  - Original edges: {info['original_num_edges']}")
        print(f"  - Hyperedges: {data.num_hyperedges}")
        print(f"  - Classes: {data.num_classes}")
        print(f"  - Mean hyperedge size: {info['mean_hyperedge_size']:.2f}")
        print(f"  - Max hyperedge size: {info['max_hyperedge_size']}")
        print(f"  - Hypergraph strategy: {info['hypergraph_strategy']}")
        print(f"  - Is undirected: {info['is_undirected']}")
        
        # Print train/val/test split info if available
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            print(f"  - Training nodes: {data.train_mask.sum().item()}")
            print(f"  - Validation nodes: {data.val_mask.sum().item()}")
            print(f"  - Test nodes: {data.test_mask.sum().item()}")



class ContactDataset(HypergraphDataset):
    """Dataset class for contact network data"""
    
    def __init__(self):
        super().__init__(
            data_loader=ContactDataLoader(),
            converter=IdentityHypergraphConverter()
        )
    
    def load_data(self, path: Union[str, Path], **kwargs) -> HypergraphData:
        """Load contact network dataset"""
        # Load raw data
        raw_data = self.data_loader.load_raw_data(path)
        
        # Convert to hypergraph format (normalize indices)
        hyperedges = self.converter.convert_to_hypergraph(raw_data, **kwargs)
        
        # Process labels
        labels = torch.tensor(raw_data['labels'], dtype=torch.long)
        num_nodes = len(labels)
        num_classes = len(torch.unique(labels))
        
        # Validate node indices
        self.validator.validate_node_indices(hyperedges, num_nodes)
        
        # Create features (identity matrix)
        node_features = torch.eye(num_nodes)
        
        # Create hyperedge index
        hyperedge_index = self._create_hyperedge_index(hyperedges)
        
        # Compute statistics
        stats = self._compute_dataset_stats(hyperedges, num_nodes)
        stats.update({
            'dataset_name': raw_data['dataset_name'],
            'dataset_type': 'contact_network'
        })
        
        # Create data container
        self._data = HypergraphData(
            node_features=node_features,
            labels=labels,
            hyperedge_index=hyperedge_index,
            num_nodes=num_nodes,
            num_hyperedges=len(hyperedges),
            num_classes=num_classes,
            label_names=raw_data['label_names'],
            dataset_info=stats
        )
        
        # Validate complete data
        self.validator.validate_hypergraph_data(self._data)
        
        self._print_dataset_info()
        return self._data
    
    def _print_dataset_info(self):
        """Print dataset information"""
        data = self._data
        info = data.dataset_info
        
        print(f"Dataset {info['dataset_name']} loaded:")
        print(f"  - Nodes: {data.num_nodes}")
        print(f"  - Hyperedges: {data.num_hyperedges}")
        print(f"  - Classes: {data.num_classes}")
        print(f"  - Mean hyperedge size: {info['mean_hyperedge_size']:.2f}")
        print(f"  - Max hyperedge size: {info['max_hyperedge_size']}")


class DataSplitter:
    """Utility class for creating data splits"""
    
    @staticmethod
    def create_transductive_split(
        labels: torch.Tensor,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        random_state: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create stratified transductive split"""
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        n_nodes = len(labels)
        unique_classes = torch.unique(labels)
        
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        for class_id in unique_classes:
            class_indices = torch.where(labels == class_id)[0]
            n_class = len(class_indices)
            
            # Random permutation for fairness
            perm = torch.randperm(n_class)
            class_indices = class_indices[perm]
            
            # Split indices
            train_end = int(train_ratio * n_class)
            val_end = train_end + int(val_ratio * n_class)
            
            train_mask[class_indices[:train_end]] = True
            val_mask[class_indices[train_end:val_end]] = True
            test_mask[class_indices[val_end:]] = True
        
        return train_mask, val_mask, test_mask


# Factory function for easy dataset creation
def create_hypergraph_dataset(dataset_type: str, **kwargs) -> HypergraphDataset:
    """Factory function to create appropriate dataset instance"""
    if dataset_type.lower() == 'contact':
        return ContactDataset()
    elif dataset_type.lower() in ['planetoid', 'planetoid_cora', 'planetoid_citeseer', 'planetoid_pubmed']:
        # Extract dataset name from type
        if dataset_type.lower() == 'planetoid_cora' or dataset_type.lower() == 'planetoid':
            dataset_name = 'Cora'
        elif dataset_type.lower() == 'planetoid_citeseer':
            dataset_name = 'CiteSeer'
        elif dataset_type.lower() == 'planetoid_pubmed':
            dataset_name = 'PubMed'
        else:
            dataset_name = kwargs.get('dataset_name', 'Cora')

        strategy = kwargs.get('hypergraph_strategy', 'clique_expansion')
        return PlanetoidHypergraphDataset(dataset_name=dataset_name, hypergraph_strategy=strategy)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


'''
EXAMPLE USAGE:

# Load standardized Cora dataset from Planetoid
planetoid_cora = create_hypergraph_dataset('planetoid_cora', hypergraph_strategy='clique_expansion')
data = planetoid_cora.load_data('/tmp/cora_data')  # Downloads automatically if not present

# Compare with your raw implementation
raw_cora = create_hypergraph_dataset('cora', hypergraph_strategy='clique_expansion')
raw_data = raw_cora.load_data('/path/to/raw/cora/files')

print(f"Planetoid Cora hyperedges: {data.num_hyperedges}")
print(f"Raw Cora hyperedges: {raw_data.num_hyperedges}")

# Use standard train/val/test splits
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

# Other Planetoid datasets
citeseer_dataset = create_hypergraph_dataset('planetoid_citeseer', hypergraph_strategy='neighborhood_expansion')
pubmed_dataset = create_hypergraph_dataset('planetoid_pubmed', hypergraph_strategy='star_expansion')
'''
