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
            'neighborhood_expansion': self._neighborhood_expansion,
            'co_citation_expansion': self._co_citation_expansion,
            'citation_expansion': self._citation_expansion,
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
    
    def _co_citation_expansion(self, edge_array: np.ndarray, num_nodes: int, **kwargs) -> List[List[int]]:
        """
        Convert citation edges to co-citation hyperedges.

        For each paper, create a hyperedge containing all papers it cites.
        This works for Cora, Citeseer, and Pubmed
        This follows the method described in Yadati et al., 2019 Link: https://arxiv.org/pdf/2207.06680#cite.yadati2019hypergcn
        """

        if edge_array.shape[1] == 0:
            return []

        # Build outgoing citation lists for each paper
        # outgoing_citations[paper_id] = list of papers that paper_id cites
        outgoing_citations = defaultdict(list)

        for i in range(edge_array.shape[1]):
            citing_paper = int(edge_array[0, i])
            cited_paper = int(edge_array[1, i])
            outgoing_citations[citing_paper].append(cited_paper)

        hyperedges = []

        for citing_paper in range(num_nodes):
            cited_papers = outgoing_citations.get(citing_paper, [])

            # Only create hyperedge if paper cites at least 2 others
            # Single citations don't really create any meaningful hyperedges
            if len(cited_papers) >= 2:
                hyperedge = sorted(cited_papers)
                hyperedges.append(hyperedge)

        unique_hyperedges = []
        seen = set()
        for he in hyperedges:
            he_tuple = tuple(he)
            if he_tuple not in seen:
                unique_hyperedges.append(he)
                seen.add(he_tuple)

        return unique_hyperedges
        
    def _citation_expansion(self, edge_array: np.ndarray, num_nodes: int, **kwargs) -> List[List[int]]:
        """Citing papers method: papers citing the same target form a hyperedge"""
        if edge_array.shape[1] == 0:
            return []
        
        incoming_citations = defaultdict(list)
        
        for i in range(edge_array.shape[1]):
            citing_paper = int(edge_array[0, i])
            cited_paper = int(edge_array[1, i])
            incoming_citations[cited_paper].append(citing_paper)
        
        hyperedges = []
        min_citers = kwargs.get('min_citers', 2)  # Minimum citers to form hyperedge
        
        for cited_paper in range(num_nodes):
            citing_papers = incoming_citations.get(cited_paper, [])
            
            if len(citing_papers) >= min_citers:
                hyperedge = sorted(citing_papers)
                hyperedges.append(hyperedge)
        
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

    def __init__(self, dataset_name: str = 'Cora', hypergraph_strategy: str = 'star_expansion', normalize_features: bool = True):
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
        self.normalize_features=normalize_features

    def load_raw_data(self, path: Union[str, Path]) -> Dict:
        """Overriding the default load_raw_data to use the one which conserves the tensors"""
        path = Path(path).expanduser()

        dataset = Planetoid(root=str(path), name=self.dataset_name)
        data = dataset[0]

        features = data.x

        if features is not None and self.normalize_features:
            features = self._normalize_features(features)
        elif features is None:
            features = None

        labels = data.y
        edges = data.edge_index

        train_mask = data.train_mask if hasattr(data, 'train_mask') else None
        val_mask = data.val_mask if hasattr(data, 'val_mask') else None
        test_mask = data.test_mask if hasattr(data, 'test_mask') else None

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

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply row-wise L1 normalization to features (same as NormalizeFeatures transform)"""
        row_sum = features.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum))

        return features/row_sum

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
        
        # TODO: Rework this section so that if we use our custom stratified training/test/val split, it reflects accordingly
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            print(f"  - Training nodes: {data.train_mask.sum().item()}")
            print(f"  - Validation nodes: {data.val_mask.sum().item()}")
            print(f"  - Test nodes: {data.test_mask.sum().item()}")



class GenericHypergraphDataLoader(DataLoader):
    """Generic loader for hypergraph datasets with standard file format"""

    def load_raw_data(self, path: Union[str, Path]) -> Dict:
        """Load generic hypergraph data from standardized files"""
        path = Path(path).expanduser()
        dataset_name = path.name if path.is_dir() else path.stem

        # Try to load hyperedges
        hyperedges_file = path / f"hyperedges-{dataset_name}.txt"
        if not hyperedges_file.exists():
            hyperedges_file = path / "hyperedges.txt"
        
        if not hyperedges_file.exists():
            # Try with .edges extension
            hyperedges_file = path / f"{dataset_name}.edges"
        
        if not hyperedges_file.exists():
            raise FileNotFoundError(f"Hyperedges file not found in {path}")

        hyperedges = []
        with open(hyperedges_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle both space and comma separated formats
                    if ',' in line:
                        nodes = [int(x.strip()) for x in line.split(',')]
                    else:
                        nodes = [int(x.strip()) for x in line.split()]
                    if nodes:  # Only add non-empty hyperedges
                        hyperedges.append(nodes)

        # Try to load node labels
        node_labels_file = path / f"node-labels-{dataset_name}.txt"
        if not node_labels_file.exists():
            node_labels_file = path / "node-labels.txt"
        
        if not node_labels_file.exists():
            # Try with .content extension (like Cora/CiteSeer)
            node_labels_file = path / f"{dataset_name}.content"

        labels = None
        label_names = []
        
        if node_labels_file.exists():
            with open(node_labels_file, 'r') as f:
                labels = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Handle comma-separated labels (multi-label)
                        if ',' in line:
                            label_indices = [int(x.strip()) for x in line.split(',')]
                            # For multi-label, we'll use the first label for single-label tasks
                            labels.append(label_indices[0] if label_indices else 0)
                        else:
                            try:
                                labels.append(int(line))
                            except ValueError:
                                # String label
                                labels.append(line)

        # Try to load label names
        label_names_file = path / f"label-names-{dataset_name}.txt"
        if not label_names_file.exists():
            label_names_file = path / "label-names.txt"

        if label_names_file.exists():
            with open(label_names_file, 'r') as f:
                label_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if not label_names and labels:
            num_classes = len(set(labels))
            label_names = [f"Class_{i}" for i in range(num_classes)]

        return {
            'labels': labels,
            'hyperedges': hyperedges,
            'label_names': label_names,
            'dataset_name': dataset_name
        }


class GenericHypergraphDataset(HypergraphDataset):
    """Generic dataset class for hypergraph data with standard format"""
    
    def __init__(self):
        super().__init__(
            data_loader=GenericHypergraphDataLoader(),
            converter=IdentityHypergraphConverter()
        )
    
    def load_data(self, path: Union[str, Path], **kwargs) -> HypergraphData:
        """Load generic hypergraph dataset"""
        raw_data = self.data_loader.load_raw_data(path)
        
        hyperedges = self.converter.convert_to_hypergraph(raw_data, **kwargs)
        
        # Convert labels
        raw_labels = raw_data['labels']
        if raw_labels is None or len(raw_labels) == 0:
            # No labels - create dummy labels for all nodes
            # Infer number of nodes from hyperedges
            num_nodes = max(max(edge) for edge in hyperedges) + 1 if hyperedges else 1
            labels = torch.zeros(num_nodes, dtype=torch.long)
            num_classes = 1
            raw_data['label_names'] = ['Unknown']
        else:
            if isinstance(raw_labels[0], str):
                # Use LabelEncoder to convert string labels to integers
                label_encoder = LabelEncoder()
                label_indices = label_encoder.fit_transform(raw_labels)
                labels = torch.tensor(label_indices, dtype=torch.long)
                # Create label names
                unique_labels = sorted(set(raw_labels))
                raw_data['label_names'] = unique_labels
            else:
                labels = torch.tensor(raw_labels, dtype=torch.long)
        
        num_nodes = len(labels)
        num_classes = len(torch.unique(labels))
        
        self.validator.validate_node_indices(hyperedges, num_nodes)
        
        # Use identity features (one-hot encoding)
        node_features = torch.eye(num_nodes)
        
        hyperedge_index = self._create_hyperedge_index(hyperedges)
        
        stats = self._compute_dataset_stats(hyperedges, num_nodes)
        stats.update({
            'dataset_name': raw_data['dataset_name'],
            'dataset_type': 'generic_hypergraph'
        })
        
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


class ContactDataset(HypergraphDataset):
    """Dataset class for contact network data"""
    
    def __init__(self):
        super().__init__(
            data_loader=ContactDataLoader(),
            converter=IdentityHypergraphConverter()
        )
    
    def load_data(self, path: Union[str, Path], **kwargs) -> HypergraphData:
        """Load contact network dataset"""
        raw_data = self.data_loader.load_raw_data(path)
        
        hyperedges = self.converter.convert_to_hypergraph(raw_data, **kwargs)
        
        # Convert string labels to integers if needed
        raw_labels = raw_data['labels']
        if isinstance(raw_labels[0], str):
            # Use LabelEncoder to convert string labels to integers
            label_encoder = LabelEncoder()
            label_indices = label_encoder.fit_transform(raw_labels)
            labels = torch.tensor(label_indices, dtype=torch.long)
            # Update label_names to match the encoded labels
            if 'label_names' in raw_data:
                # Create mapping from encoded index to original label name
                unique_labels = sorted(set(raw_labels))
                raw_data['label_names'] = unique_labels
        else:
            labels = torch.tensor(raw_labels, dtype=torch.long)
        
        num_nodes = len(labels)
        num_classes = len(torch.unique(labels))
        
        self.validator.validate_node_indices(hyperedges, num_nodes)
        
        node_features = torch.eye(num_nodes)
        
        hyperedge_index = self._create_hyperedge_index(hyperedges)
        
        stats = self._compute_dataset_stats(hyperedges, num_nodes)
        stats.update({
            'dataset_name': raw_data['dataset_name'],
            'dataset_type': 'contact_network'
        })
        
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
        train_ratio: float = 0.5,
        val_ratio: float = 0.25,
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
    """
    Factory function to create appropriate dataset instance
    
    Args:
        dataset_type: Type of dataset. Can be:
            - 'contact': Contact networks
            - 'planetoid_cora', 'planetoid_citeseer', 'planetoid_pubmed': Planetoid datasets
            - Generic dataset names: 'contact_high_school', 'contact_primary_school', 'walmart_trips',
              'stackoverflow_answers', 'amazon_reviews', 'zoo', 'mushroom', 'ntu2012', 'modelnet40',
              'house_committees', '20newsW100', 'coauthorship', 'cocitation', 'yelp'
        **kwargs: Additional arguments for dataset configuration
    
    Returns:
        HypergraphDataset: Appropriate dataset instance
    """
    dataset_type = dataset_type.lower()
    
    # Contact networks (clustering)
    if dataset_type in ['contact', 'contact_high_school', 'contact_primary_school']:
        return ContactDataset()
    
    # Planetoid datasets (classification)
    elif dataset_type in ['planetoid', 'planetoid_cora', 'planetoid_citeseer', 'planetoid_pubmed']:
        # Extract dataset name from type
        if dataset_type == 'planetoid_cora' or dataset_type == 'planetoid':
            dataset_name = 'Cora'
        elif dataset_type == 'planetoid_citeseer':
            dataset_name = 'CiteSeer'
        elif dataset_type == 'planetoid_pubmed':
            dataset_name = 'PubMed'
        else:
            dataset_name = kwargs.get('dataset_name', 'Cora')

        strategy = kwargs.get('hypergraph_strategy', 'co_citation_expansion')
        return PlanetoidHypergraphDataset(dataset_name=dataset_name, hypergraph_strategy=strategy, normalize_features=True)
    
    # Generic hypergraph datasets (clustering and others)
    elif dataset_type in [
        'walmart_trips',
        'stackoverflow_answers',
        'amazon_reviews',
        'zoo',
        'mushroom',
        'ntu2012',
        'modelnet40',
        'house_committees',
        '20newsW100',
        'coauthorship',
        'cocitation',
        'yelp'
    ]:
        return GenericHypergraphDataset()
    
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Supported types: 'contact', 'planetoid_cora', 'planetoid_citeseer', 'planetoid_pubmed', "
            f"'walmart_trips', 'stackoverflow_answers', 'amazon_reviews', 'zoo', 'mushroom', 'ntu2012', "
            f"'modelnet40', 'house_committees', '20newsW100', 'coauthorship', 'cocitation', 'yelp'"
        )


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
