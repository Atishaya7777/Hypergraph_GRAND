import torch
import os
import numpy as np
from typing import Tuple


class ContactDataset:
    """
    Dataset class for contact network data
    """

    def __init__(self, data_path: str, dataset_name: str):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.load_data()

    def load_data(self):
        """
        Load hypergraph data from files
        """
        node_labels_file = os.path.join(
            self.data_path, f"node-labels-{self.dataset_name}.txt")
        with open(node_labels_file, 'r') as f:
            labels = [int(line.strip()) for line in f]

        self.num_nodes = len(labels)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.num_classes = len(torch.unique(self.labels))

        hyperedges_file = os.path.join(
            self.data_path, f"hyperedges-{self.dataset_name}.txt")
        hyperedges = []
        max_node_id = -1

        with open(hyperedges_file, 'r') as f:
            for line in f:
                nodes = [int(x) for x in line.strip().split(',')]
                # Convert to 0-indexed if needed
                nodes = [node - 1 if min(nodes) >
                         0 else node for node in nodes]
                hyperedges.append(nodes)
                max_node_id = max(max_node_id, max(nodes))

        if max_node_id >= self.num_nodes:
            print(f"Warning: Max node ID in hyperedges ({
                  max_node_id}) >= num_nodes ({self.num_nodes})")
            print("Converting node indices to 0-indexed...")

            # Convert all node indices to 0-indexed
            hyperedges = [[node - 1 for node in edge] for edge in hyperedges]
            max_node_id = max(max(edge) for edge in hyperedges)

            if max_node_id >= self.num_nodes:
                raise ValueError(f"Even after conversion, max node ID ({
                                 max_node_id}) >= num_nodes ({self.num_nodes})")

        self.num_hyperedges = len(hyperedges)

        # Create hyperedge index tensor [2, num_edges]
        edge_indices = []
        node_indices = []

        for edge_id, nodes in enumerate(hyperedges):
            for node_id in nodes:
                if node_id < 0 or node_id >= self.num_nodes:
                    raise ValueError(
                        f"Node ID {node_id} is out of bounds [0, {self.num_nodes-1}]")
                edge_indices.append(edge_id)
                node_indices.append(node_id)

        self.hyperedge_index = torch.tensor(
            [edge_indices, node_indices], dtype=torch.long)

        label_names_file = os.path.join(
            self.data_path, f"label-names-{self.dataset_name}.txt")
        try:
            with open(label_names_file, 'r') as f:
                self.label_names = [line.strip() for line in f]
        except FileNotFoundError:
            self.label_names = [f"Class_{i}" for i in range(self.num_classes)]

        # Using one hot encoding, basically just an identity matrix
        self.node_features = torch.eye(self.num_nodes)

        print(f"Dataset {self.dataset_name} loaded:")
        print(f"  - Nodes: {self.num_nodes}")
        print(f"  - Hyperedges: {self.num_hyperedges}")
        print(f"  - Classes: {self.num_classes}")
        print(
            f"  - Mean hyperedge size: {len(node_indices) / self.num_hyperedges:.2f}")
        print(f"  - Max hyperedge size: {max(len(nodes)
              for nodes in hyperedges)}")
        print(f"  - Node ID range: [0, {max_node_id}]")


def create_transductive_split(
    labels: torch.Tensor,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create transductive split - stratified by class
    """
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

        # Randomly permute the indices for the classes so that there is no visible bias
        perm = torch.randperm(n_class)
        class_indices = class_indices[perm]

        # Split indices
        train_end = int(train_ratio * n_class)
        val_end = train_end + int(val_ratio * n_class)

        train_mask[class_indices[:train_end]] = True
        val_mask[class_indices[train_end:val_end]] = True
        test_mask[class_indices[val_end:]] = True

    return train_mask, val_mask, test_mask
