#!/usr/bin/env python3
"""
Universal PyG Standardizer for ALL HyperGRAND Datasets

Handles all dataset formats: hypergraph files, pickle, PyG Planetoid, content/edges
"""

import json
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from torch_geometric.data import Data
from dataclasses import dataclass


@dataclass
class DatasetMetadata:
    """Standardized metadata for all datasets"""
    name: str
    task_type: str
    num_nodes: int
    num_classes: int
    num_hyperedges: int
    source: str
    node_feature_dim: int = None


class UniversalDataConverter:
    """Universal converter for all dataset formats to PyG Data"""
    
    def __init__(self, auto_normalize: bool = True):
        self.auto_normalize = auto_normalize
    
    def convert_dataset(
        self,
        dataset_path: Union[str, Path],
        dataset_name: str,
        task_type: str,
        num_classes: Optional[int] = None
    ) -> Data:
        """Convert any dataset to PyG Data format"""
        dataset_path = Path(dataset_path)
        data = None
        
        # Try format detection
        if self._has_pyg_planetoid_format(dataset_path):
            data = self._load_pyg_planetoid_format(dataset_path)
        elif self._has_hypergraph_format(dataset_path):
            data = self._load_hypergraph_format(dataset_path)
        elif self._has_pickle_format(dataset_path):
            data = self._load_pickle_format(dataset_path)
        elif self._has_content_edges_format(dataset_path):
            data = self._load_content_edges_format(dataset_path)
        else:
            data = self._load_generic_format(dataset_path)
        
        if data is None:
            raise ValueError(f"Could not load dataset from {dataset_path}")
        
        # Convert sparse matrices to dense tensors
        if hasattr(data.x, 'toarray'):
            data.x = torch.from_numpy(data.x.toarray()).float()
        elif data.x is None:
            data.x = torch.eye(data.num_nodes, dtype=torch.float32)
        
        # Normalize features
        if self.auto_normalize and data.x is not None:
            data.x = self._normalize_features(data.x)
        
        # Ensure hyperedge_index is proper 2D tensor
        if not hasattr(data, 'hyperedge_index') or data.hyperedge_index is None:
            # Create dummy hyperedge index if missing
            data.hyperedge_index = torch.zeros((2, 0), dtype=torch.long)
        elif hasattr(data.hyperedge_index, 'toarray'):
            # Convert sparse hyperedge_index to dense COO format
            sparse_he = data.hyperedge_index
            coo = sparse_he.tocoo()
            data.hyperedge_index = torch.stack([
                torch.from_numpy(coo.row),
                torch.from_numpy(coo.col)
            ], dim=0).long()
        elif not isinstance(data.hyperedge_index, torch.Tensor):
            data.hyperedge_index = torch.tensor(data.hyperedge_index, dtype=torch.long)
        
        # Create splits if needed
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            data.train_mask, data.val_mask, data.test_mask = self._create_splits(
                data.num_nodes, data.y
            )
        
        # Add metadata
        if num_classes is None:
            num_classes = int(data.y.max().item()) + 1 if data.y.max() >= 0 else 2
        
        data.metadata = DatasetMetadata(
            name=dataset_name,
            task_type=task_type,
            num_nodes=data.num_nodes,
            num_classes=num_classes,
            num_hyperedges=data.hyperedge_index.shape[1] if hasattr(data, 'hyperedge_index') and data.hyperedge_index is not None else 0,
            source="",
            node_feature_dim=data.x.shape[1] if data.x is not None else None
        )
        
        return data
    
    # ===== Format Detection =====
    
    def _has_pyg_planetoid_format(self, path: Path) -> bool:
        """Check for PyTorch Geometric Planetoid format"""
        raw_path = path / 'raw'
        if raw_path.exists():
            pyg_files = list(raw_path.glob('ind.*'))
            return len(pyg_files) > 0
        return False
    
    def _has_hypergraph_format(self, path: Path) -> bool:
        """Check for standard hypergraph format"""
        hyperedge_files = list(path.glob('hyperedges*.txt'))
        label_files = list(path.glob('*labels*.txt'))
        return len(hyperedge_files) > 0 and len(label_files) > 0
    
    def _has_pickle_format(self, path: Path) -> bool:
        """Check for pickle format"""
        pickle_files = list(path.glob('*.pickle'))
        return len(pickle_files) > 0
    
    def _has_content_edges_format(self, path: Path) -> bool:
        """Check for content/edges format"""
        content_files = list(path.glob('*.content'))
        cites_files = list(path.glob('*.cites'))
        return len(content_files) > 0 or len(cites_files) > 0
    
    # ===== Loaders =====
    
    def _load_pyg_planetoid_format(self, path: Path) -> Data:
        """Load PyTorch Geometric Planetoid format"""
        raw_path = path / 'raw'
        
        # Find all ind.* files
        ind_files = list(raw_path.glob('ind.*'))
        if not ind_files:
            raise ValueError(f"No ind.* files in {raw_path}")
        
        # Extract dataset name
        dataset_name = ind_files[0].stem.replace('ind.', '').split('.')[0]
        
        # Load features: x (train+val) + tx (test) = all nodes
        features = None
        allx_file = raw_path / f'ind.{dataset_name}.allx'
        tx_file = raw_path / f'ind.{dataset_name}.tx'
        
        allx = None
        tx = None
        
        if allx_file.exists():
            with open(allx_file, 'rb') as f:
                allx = pickle.load(f, encoding='latin1')
        
        if tx_file.exists():
            with open(tx_file, 'rb') as f:
                tx = pickle.load(f, encoding='latin1')
        
        # Concatenate allx and tx to get all features
        if allx is not None and tx is not None:
            allx_arr = allx.toarray() if hasattr(allx, 'toarray') else np.array(allx)
            tx_arr = tx.toarray() if hasattr(tx, 'toarray') else np.array(tx)
            features = torch.from_numpy(np.vstack([allx_arr, tx_arr])).float()
        elif allx is not None:
            allx_arr = allx.toarray() if hasattr(allx, 'toarray') else np.array(allx)
            features = torch.from_numpy(allx_arr).float()
        elif tx is not None:
            tx_arr = tx.toarray() if hasattr(tx, 'toarray') else np.array(tx)
            features = torch.from_numpy(tx_arr).float()
        
        # Load labels: ally (train+val) + ty (test) = all nodes
        labels = None
        ally_file = raw_path / f'ind.{dataset_name}.ally'
        ty_file = raw_path / f'ind.{dataset_name}.ty'
        
        ally = None
        ty = None
        
        if ally_file.exists():
            with open(ally_file, 'rb') as f:
                ally = pickle.load(f, encoding='latin1')
        
        if ty_file.exists():
            with open(ty_file, 'rb') as f:
                ty = pickle.load(f, encoding='latin1')
        
        # Concatenate ally and ty to get all labels
        if ally is not None and ty is not None:
            ally_arr = ally if isinstance(ally, np.ndarray) else np.array(ally)
            ty_arr = ty if isinstance(ty, np.ndarray) else np.array(ty)
            
            # Convert one-hot to class indices if needed
            if ally_arr.ndim > 1 and ally_arr.shape[1] > 1:
                ally_arr = ally_arr.argmax(axis=1)
            if ty_arr.ndim > 1 and ty_arr.shape[1] > 1:
                ty_arr = ty_arr.argmax(axis=1)
            
            labels = torch.from_numpy(np.concatenate([ally_arr.flatten(), ty_arr.flatten()])).long()
        elif ally is not None:
            ally_arr = ally if isinstance(ally, np.ndarray) else np.array(ally)
            if ally_arr.ndim > 1 and ally_arr.shape[1] > 1:
                ally_arr = ally_arr.argmax(axis=1)
            labels = torch.from_numpy(ally_arr.flatten()).long()
        elif ty is not None:
            ty_arr = ty if isinstance(ty, np.ndarray) else np.array(ty)
            if ty_arr.ndim > 1 and ty_arr.shape[1] > 1:
                ty_arr = ty_arr.argmax(axis=1)
            labels = torch.from_numpy(ty_arr.flatten()).long()
        
        # Load graph and convert to hyperedge format
        # HyperGRAND expects [hyperedge_id, node_id] format where each edge becomes a hyperedge
        hyperedge_index = torch.zeros((2, 0), dtype=torch.long)
        graph_file = raw_path / f'ind.{dataset_name}.graph'
        if graph_file.exists():
            with open(graph_file, 'rb') as f:
                graph = pickle.load(f, encoding='latin1')
                edges_list = []
                edge_id = 0
                for src, neighbors in graph.items():
                    for dst in neighbors:
                        # Each edge (src, dst) becomes a hyperedge
                        edges_list.append((edge_id, src))
                        edges_list.append((edge_id, dst))
                        edge_id += 1
                
                if edges_list:
                    edges_array = list(set(edges_list))  # Remove duplicates
                    hyperedge_index = torch.tensor(edges_array, dtype=torch.long).t().contiguous()
        
        num_nodes = features.shape[0] if features is not None else labels.shape[0]
        
        # Validate hyperedge_index
        if hyperedge_index.shape[1] > 0:
            max_node_id = int(hyperedge_index[1].max().item())
            if max_node_id >= num_nodes:
                # Filter out invalid edges (shouldn't happen if data is correct)
                valid_mask = hyperedge_index[1] < num_nodes
                hyperedge_index = hyperedge_index[:, valid_mask]
        
        return Data(
            x=features,
            y=labels,
            hyperedge_index=hyperedge_index if hyperedge_index.shape[1] > 0 else torch.zeros((2, 0), dtype=torch.long),
            num_nodes=num_nodes
        )
    
    def _load_hypergraph_format(self, path: Path) -> Data:
        """Load hyperedge + labels format"""
        # Find files
        hyperedge_files = list(path.glob('hyperedges*.txt'))
        label_files = list(path.glob('*labels*.txt'))
        
        if not hyperedge_files or not label_files:
            raise ValueError(f"Missing hyperedges or labels files in {path}")
        
        hyperedge_file = hyperedge_files[0]
        label_file = label_files[0]
        
        # Load hyperedges
        hyperedges = []
        with open(hyperedge_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    if ',' in line:
                        edge = list(map(int, line.split(',')))
                    else:
                        edge = list(map(int, line.split()))
                    hyperedges.append(edge)
        
        # Load labels
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    if ',' in line:
                        label = int(line.split(',')[0])
                    else:
                        label = int(line)
                    labels.append(label)
        
        labels = torch.tensor(labels, dtype=torch.long)
        num_nodes = len(labels)
        
        # Create hyperedge_index
        hyperedge_index = self._create_hyperedge_index(hyperedges, num_nodes)
        
        # Load features if available
        features = None
        feature_files = list(path.glob('*.features'))
        if feature_files:
            features = self._load_feature_file(feature_files[0])
        
        return Data(
            x=features,
            y=labels,
            hyperedge_index=hyperedge_index,
            num_nodes=num_nodes
        )
    
    def _load_pickle_format(self, path: Path) -> Data:
        """Load pickle format datasets"""
        pickle_files = {f.stem: f for f in path.glob('*.pickle')}
        
        features = None
        if 'features' in pickle_files:
            with open(pickle_files['features'], 'rb') as f:
                features = pickle.load(f)
                if isinstance(features, (list, tuple)):
                    features = np.array(features)
                if isinstance(features, np.ndarray):
                    features = torch.from_numpy(features).float()
        
        labels = None
        if 'labels' in pickle_files:
            with open(pickle_files['labels'], 'rb') as f:
                labels = pickle.load(f)
                if isinstance(labels, np.ndarray):
                    labels = torch.from_numpy(labels).long()
                elif isinstance(labels, list):
                    labels = torch.tensor(labels, dtype=torch.long)
        
        hyperedge_index = None
        if 'hypergraph' in pickle_files:
            with open(pickle_files['hypergraph'], 'rb') as f:
                hg = pickle.load(f)
                
                # Handle different hypergraph formats
                if isinstance(hg, dict):
                    if 'hyperedge_index' in hg:
                        # Format: {'hyperedge_index': tensor}
                        hyperedge_index = hg['hyperedge_index']
                    else:
                        # Format: {node_id: [connected_nodes]} or {name: [node_ids]}
                        # This is the co-authorship/co-citation format
                        hyperedge_index = self._convert_dict_to_hyperedges(hg)
                elif isinstance(hg, (list, tuple)):
                    # Direct hyperedge list format
                    hyperedge_index = torch.tensor(hg, dtype=torch.long)
        
        num_nodes = labels.shape[0] if labels is not None else features.shape[0]
        
        return Data(
            x=features,
            y=labels,
            hyperedge_index=hyperedge_index,
            num_nodes=num_nodes
        )
    
    def _convert_dict_to_hyperedges(self, hg_dict: Dict) -> torch.Tensor:
        """Convert dict-based hypergraph to hyperedge_index tensor
        
        Handles formats like:
        - {name: [node_ids]}: Co-authorship/co-citation networks
        - {node_id: {cited_nodes}}: Co-citation networks with sets
        - Each entry becomes a hyperedge
        """
        edge_list = []
        node_to_edge = {}
        edge_counter = 0
        
        for hyperedge_id, node_ids in hg_dict.items():
            # Handle different node collection types
            if isinstance(node_ids, set):
                node_list = list(node_ids)
            elif isinstance(node_ids, (list, tuple)):
                node_list = list(node_ids)
            else:
                continue
            
            for node_id in node_list:
                # Ensure node_id is an integer
                if isinstance(node_id, (int, np.integer)):
                    edge_list.append([edge_counter, int(node_id)])
            
            if node_list:  # Only increment if this hyperedge has nodes
                edge_counter += 1
        
        if edge_list:
            hyperedge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            return hyperedge_index
        else:
            return torch.zeros((2, 0), dtype=torch.long)
    
    def _load_content_edges_format(self, path: Path) -> Data:
        """Load content/edges format (Cora-like and zoo-like)"""
        content_file = list(path.glob('*.content'))
        cites_files = list(path.glob('*.cites'))
        edges_files = list(path.glob('*.edges'))
        
        node_ids = []
        features = []
        labels = []
        
        if content_file:
            with open(content_file[0], 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    # Check if first element is numeric (node ID format)
                    try:
                        first_val = int(parts[0])
                        node_ids.append(first_val)
                        feature_parts = parts[1:-1]
                        label_part = parts[-1]
                    except ValueError:
                        # No node ID, features are from the beginning
                        feature_parts = parts[:-1]
                        label_part = parts[-1]
                        node_ids.append(len(node_ids))
                    
                    # Handle both int and float features - parse as float for flexibility
                    try:
                        features.append(list(map(float, feature_parts)))
                    except ValueError:
                        features.append(list(map(float, feature_parts)))
                    # Label should always be int
                    try:
                        labels.append(int(label_part))
                    except ValueError:
                        labels.append(int(float(label_part)))
        
        id_map = {nid: idx for idx, nid in enumerate(node_ids)}
        num_nodes = len(node_ids)
        
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Parse edges - can be pairwise graph edges or node-to-hyperedge mappings
        hyperedge_index = None
        if edges_files:
            hyperedge_index = self._parse_edges_file(edges_files[0], id_map, num_nodes)
        elif cites_files:
            # Convert pairwise graph edges into hyperedges (each graph edge -> one hyperedge containing both endpoints)
            hyperedge_list = []
            with open(cites_files[0], 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    src, dst = int(parts[0]), int(parts[1])
                    if src in id_map and dst in id_map:
                        s_idx = id_map[src]
                        d_idx = id_map[dst]
                        # create a hyperedge containing both nodes
                        hyperedge_list.append([s_idx, d_idx])
            
            # Use helper to create [hyperedge_id, node_id] tensor
            hyperedge_index = self._create_hyperedge_index(hyperedge_list, num_nodes) if hyperedge_list else None
        
        return Data(
            x=features,
            y=labels,
            hyperedge_index=hyperedge_index,
            num_nodes=num_nodes
        )
    
    def _parse_edges_file(self, edges_file_path, id_map, num_nodes):
        """Parse edges file - assumes node-to-hyperedge format [node_id, hyperedge_id]"""
        edge_dict = {}  # hyperedge_id -> [node_indices]
        
        with open(edges_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                node_id = int(parts[0])
                hyperedge_id = int(parts[1])
                
                # Map node ID to index
                if node_id in id_map:
                    node_idx = id_map[node_id]
                else:
                    # If node ID not in id_map, try using it as an index directly
                    if node_id < num_nodes:
                        node_idx = node_id
                    else:
                        continue  # Skip out-of-bounds nodes
                
                # Node-to-hyperedge format
                if hyperedge_id not in edge_dict:
                    edge_dict[hyperedge_id] = []
                edge_dict[hyperedge_id].append(node_idx)
        
        # Convert to [hyperedge_id, node_id] tensor
        hyperedge_list = []
        for he_id in sorted(edge_dict.keys()):
            for node_idx in edge_dict[he_id]:
                hyperedge_list.append([he_id, node_idx])
        
        if hyperedge_list:
            return torch.tensor(hyperedge_list, dtype=torch.long).t().contiguous()
        else:
            return torch.zeros((2, 0), dtype=torch.long)
    
    def _load_generic_format(self, path: Path) -> Data:
        """Generic fallback loader"""
        # Look for label files
        labels = None
        for label_file in path.glob('*label*'):
            if label_file.suffix in ['.txt', '.csv']:
                labels = self._load_label_file(label_file)
                break
        
        if labels is None:
            num_nodes = 100
            labels = torch.zeros(num_nodes, dtype=torch.long)
        else:
            num_nodes = len(labels)
        
        features = torch.eye(num_nodes, dtype=torch.float32)
        
        hyperedge_index = torch.zeros((2, num_nodes), dtype=torch.long)
        for i in range(num_nodes):
            hyperedge_index[0, i] = i
            hyperedge_index[1, i] = min(i + 1, num_nodes - 1)
        
        return Data(
            x=features,
            y=labels,
            hyperedge_index=hyperedge_index,
            num_nodes=num_nodes
        )
    
    # ===== Helpers =====
    
    def _create_hyperedge_index(self, hyperedges: list, num_nodes: int) -> torch.Tensor:
        """Convert hyperedges to COO format"""
        # Produce tensor in [hyperedge_id, node_id] order expected by model
        he_ids = []
        node_ids = []
        for edge_id, nodes in enumerate(hyperedges):
            for node in nodes:
                if 0 <= node < num_nodes:
                    he_ids.append(edge_id)
                    node_ids.append(node)

        if not he_ids:
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.stack([torch.tensor(he_ids, dtype=torch.long),
                            torch.tensor(node_ids, dtype=torch.long)], dim=0)
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features to [0, 1]"""
        if features is None:
            return None
        feat_min = features.min()
        feat_max = features.max()
        if feat_max == feat_min:
            return features
        return (features - feat_min) / (feat_max - feat_min)
    
    def _create_splits(
        self,
        num_nodes: int,
        labels: torch.Tensor,
        train_ratio: float = 0.5,
        val_ratio: float = 0.25
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train/val/test splits"""
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        for class_id in labels.unique():
            class_mask = labels == class_id
            class_indices = torch.where(class_mask)[0]
            
            perm = torch.randperm(len(class_indices))
            class_indices = class_indices[perm]
            
            n_train = max(1, int(len(class_indices) * train_ratio))
            n_val = max(1, int(len(class_indices) * val_ratio))
            
            train_mask[class_indices[:n_train]] = True
            val_mask[class_indices[n_train:n_train + n_val]] = True
            test_mask[class_indices[n_train + n_val:]] = True
        
        return train_mask, val_mask, test_mask
    
    def _load_feature_file(self, feature_file: Path) -> torch.Tensor:
        """Load features from file"""
        features = []
        with open(feature_file, 'r') as f:
            for line in f:
                if line.strip():
                    feat = list(map(float, line.strip().split()))
                    features.append(feat)
        return torch.tensor(features, dtype=torch.float32) if features else None
    
    def _load_label_file(self, label_file: Path) -> torch.Tensor:
        """Load labels from file"""
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    label = int(line.strip().split()[0])
                    labels.append(label)
        return torch.tensor(labels, dtype=torch.long) if labels else None


class DatasetLoader:
    """Unified interface for loading all datasets"""
    
    def __init__(self, base_path: str = "datasets"):
        self.base_path = Path(base_path)
        self.converter = UniversalDataConverter()
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load metadata from JSON"""
        metadata_path = self.base_path / "DATASET_METADATA.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def load(self, dataset_name: str, verbose: bool = True) -> Data:
        """Load any dataset by name"""
        dataset_info = None
        task_type = None
        
        for task in self.metadata.values():
            if 'datasets' in task:
                if dataset_name in task['datasets']:
                    dataset_info = task['datasets'][dataset_name]
                    task_type = task.get('task_type', 'other')
                    break
        
        if dataset_info is None:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset_path = self.base_path / dataset_info['path']
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")
        
        if verbose:
            print(f"Loading {dataset_name}...")
        
        data = self.converter.convert_dataset(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            task_type=task_type,
            num_classes=dataset_info.get('num_classes')
        )
        
        if verbose:
            num_classes = data.metadata.num_classes if hasattr(data, 'metadata') else int(data.y.max().item()) + 1
            print(f"  ✓ {data.num_nodes} nodes, {num_classes} classes")
        
        return data
    
    def load_by_task(self, task_type: str) -> Dict[str, Data]:
        """Load all datasets of a task type"""
        datasets = {}
        if task_type not in self.metadata:
            raise ValueError(f"Unknown task type: {task_type}")
        
        for dataset_name in self.metadata[task_type].get('datasets', {}).keys():
            try:
                datasets[dataset_name] = self.load(dataset_name, verbose=False)
            except Exception as e:
                print(f"  ✗ Failed to load {dataset_name}: {e}")
        
        return datasets
    
    def list_datasets(self) -> Dict[str, list]:
        """List all datasets"""
        result = {}
        for task_type, task_data in self.metadata.items():
            result[task_type] = list(task_data.get('datasets', {}).keys())
        return result
