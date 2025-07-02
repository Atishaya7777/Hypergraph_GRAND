import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os


def load_single_hypergraph_dataset(label_path, hyperedge_path, dataset_name):
    """
    Load a single hypergraph dataset from label and hyperedge files

    Args:
        label_path: Path to node labels file
        hyperedge_path: Path to hyperedges file
        dataset_name: Name identifier for the dataset

    Returns:
        Dictionary containing dataset information
    """
    # Load node labels
    with open(label_path, "r") as f:
        y = [int(line.strip()) for line in f]
    y = torch.tensor(y, dtype=torch.long)

    # Load hyperedges
    hyperedge_list = []
    max_node_id = -1
    hyperedge_nodes = {}  # Store which nodes belong to each hyperedge

    with open(hyperedge_path, "r") as f:
        for eid, line in enumerate(f):
            node_ids = list(map(int, line.strip().split(",")))
            max_node_id = max(max_node_id, max(node_ids))
            hyperedge_nodes[eid] = node_ids
            for node in node_ids:
                hyperedge_list.append([eid, node])

    hyperedge_index = torch.tensor(
        hyperedge_list, dtype=torch.long).t().contiguous()
    num_nodes = max(len(y), max_node_id + 1)

    # Pad labels if necessary
    if len(y) < num_nodes:
        padded_y = torch.full((num_nodes,), -1, dtype=torch.long)
        padded_y[:len(y)] = y
        y = padded_y

    return {
        'name': dataset_name,
        'num_nodes': num_nodes,
        'labels': y,
        'hyperedge_index': hyperedge_index,
        'hyperedge_nodes': hyperedge_nodes,
        'max_node_id': max_node_id
    }


def merge_hypergraph_datasets(datasets):
    """
    Merge multiple hypergraph datasets into a single combined dataset

    Args:
        datasets: List of dataset dictionaries from load_single_hypergraph_dataset

    Returns:
        Combined dataset dictionary
    """
    combined_labels = []
    combined_hyperedge_list = []
    combined_hyperedge_nodes = {}

    total_nodes = 0
    total_hyperedges = 0
    dataset_node_offsets = {}
    dataset_hyperedge_offsets = {}

    print("Merging datasets:")

    for i, dataset in enumerate(datasets):
        dataset_name = dataset['name']
        num_nodes = dataset['num_nodes']

        dataset_node_offsets[dataset_name] = total_nodes
        dataset_hyperedge_offsets[dataset_name] = total_hyperedges

        print(f"  {dataset_name}: {num_nodes} nodes, {
              len(dataset['hyperedge_nodes'])} hyperedges")

        labels = dataset['labels']
        combined_labels.extend(labels.tolist())

        # Add hyperedges with appropriate offsets
        hyperedge_index = dataset['hyperedge_index']
        for j in range(hyperedge_index.size(1)):
            edge_id = hyperedge_index[0, j].item() + total_hyperedges
            node_id = hyperedge_index[1, j].item() + total_nodes
            combined_hyperedge_list.append([edge_id, node_id])

        # Update hyperedge_nodes mapping
        for edge_id, nodes in dataset['hyperedge_nodes'].items():
            new_edge_id = edge_id + total_hyperedges
            new_nodes = [node + total_nodes for node in nodes]
            combined_hyperedge_nodes[new_edge_id] = new_nodes

        total_nodes += num_nodes
        total_hyperedges += len(dataset['hyperedge_nodes'])

    # Convert to tensors
    combined_labels = torch.tensor(combined_labels, dtype=torch.long)
    combined_hyperedge_index = torch.tensor(
        combined_hyperedge_list, dtype=torch.long).t().contiguous()

    print(f"Combined dataset: {total_nodes} nodes, {
          total_hyperedges} hyperedges")

    return {
        'name': 'merged_dataset',
        'num_nodes': total_nodes,
        'labels': combined_labels,
        'hyperedge_index': combined_hyperedge_index,
        'hyperedge_nodes': combined_hyperedge_nodes,
        'dataset_offsets': {
            'nodes': dataset_node_offsets,
            'hyperedges': dataset_hyperedge_offsets
        }
    }


def create_enhanced_node_features(dataset, feature_dim=128):
    """
    Create enhanced node features instead of simple identity matrix

    Args:
        dataset: Dataset dictionary
        feature_dim: Dimension of node features

    Returns:
        Enhanced node feature matrix
    """
    num_nodes = dataset['num_nodes']
    hyperedge_index = dataset['hyperedge_index']
    labels = dataset['labels']

    # Initialize features
    features = torch.zeros(num_nodes, feature_dim)

    # 1. One-hot encoding for first part (up to num_nodes if feature_dim allows)
    identity_dim = min(num_nodes, feature_dim // 2)
    if identity_dim > 0:
        features[:identity_dim, :identity_dim] = torch.eye(identity_dim)

    # 2. Degree-based features
    node_degrees = torch.zeros(num_nodes)
    hyperedge_degrees = torch.zeros(num_nodes)

    for i in range(hyperedge_index.size(1)):
        node_id = hyperedge_index[1, i].item()
        node_degrees[node_id] += 1

    # Count unique hyperedges per node
    for node in range(num_nodes):
        hyperedges = (hyperedge_index[1] == node)
        hyperedge_degrees[node] = hyperedge_index[0][hyperedges].unique(
        ).numel()

    # Add degree features
    degree_start = identity_dim
    if degree_start < feature_dim:
        # Normalize degrees
        max_degree = max(node_degrees.max().item(), 1)
        max_hyperedge_degree = max(hyperedge_degrees.max().item(), 1)

        features[:, degree_start] = node_degrees / max_degree
        if degree_start + 1 < feature_dim:
            features[:, degree_start + 1] = hyperedge_degrees / \
                max_hyperedge_degree

    # 3. Label-based features (if available)
    label_start = degree_start + 2
    if label_start < feature_dim:
        # Use label information where available
        unique_labels = labels[labels != -1].unique()
        for i, label in enumerate(unique_labels):
            if label_start + i < feature_dim:
                mask = (labels == label)
                features[mask, label_start + i] = 1.0

    # 4. Random features for remaining dimensions
    remaining_start = min(feature_dim, label_start + len(unique_labels))
    if remaining_start < feature_dim:
        features[:, remaining_start:] = torch.randn(
            num_nodes, feature_dim - remaining_start) * 0.1

    return features


def create_membership_function(hyperedge_index, num_nodes, sparsity=0.1, membership_type='enhanced'):
    """
    Create membership function for hyperedges with different strategies

    Args:
        hyperedge_index: Hyperedge connectivity tensor
        num_nodes: Total number of nodes
        sparsity: Fraction of connections to modify
        membership_type: Type of membership ('basic', 'enhanced', 'weighted')

    Returns:
        membership: Tensor of shape [num_hyperedges, num_nodes]
    """
    num_hyperedges = hyperedge_index[0].max().item() + 1
    membership = torch.zeros(num_hyperedges, num_nodes)

    # Basic membership assignment
    for i in range(hyperedge_index.size(1)):
        e_idx = hyperedge_index[0, i]
        n_idx = hyperedge_index[1, i]
        membership[e_idx, n_idx] = 1.0

    if membership_type == 'basic':
        # Simple random negative assignments
        mask = torch.rand_like(membership) < sparsity
        membership[mask] = -1.0

    elif membership_type == 'enhanced':
        # More sophisticated membership assignment

        # 1. Create some partial memberships
        partial_mask = torch.rand_like(membership) < sparsity * 0.5
        membership[partial_mask] = torch.rand(
            partial_mask.sum()) * 0.8 + 0.2  # 0.2 to 1.0

        # 2. Create some negative memberships (opposing nodes)
        negative_mask = torch.rand_like(membership) < sparsity * 0.3
        membership[negative_mask] = - \
            torch.rand(negative_mask.sum()) * 0.5 - 0.1  # -0.6 to -0.1

        # 3. Add weak connections between nearby hyperedges
        for e1 in range(num_hyperedges):
            nodes_e1 = (membership[e1] > 0).nonzero().flatten()
            if len(nodes_e1) > 0:
                for e2 in range(e1 + 1, min(e1 + 5, num_hyperedges)):  # Check nearby hyperedges
                    nodes_e2 = (membership[e2] > 0).nonzero().flatten()
                    # If hyperedges share nodes, create weak cross-memberships
                    shared_nodes = set(nodes_e1.tolist()) & set(
                        nodes_e2.tolist())
                    if len(shared_nodes) > 0 and torch.rand(1).item() < 0.3:
                        # Add weak membership
                        for node in nodes_e1:
                            if membership[e2, node] == 0 and torch.rand(1).item() < 0.2:
                                membership[e2, node] = torch.rand(
                                    1).item() * 0.3  # Weak membership

    elif membership_type == 'weighted':
        # Weight-based membership using node degrees
        node_degrees = torch.zeros(num_nodes)
        for i in range(hyperedge_index.size(1)):
            node_id = hyperedge_index[1, i].item()
            node_degrees[node_id] += 1

        # Normalize degrees
        max_degree = node_degrees.max().item()
        normalized_degrees = node_degrees / max_degree if max_degree > 0 else node_degrees

        # Adjust membership based on node importance
        for e_idx in range(num_hyperedges):
            nodes_in_edge = (membership[e_idx] > 0).nonzero().flatten()
            for node in nodes_in_edge:
                # Higher degree nodes get stronger membership
                membership[e_idx, node] = 0.5 + 0.5 * normalized_degrees[node]

        # Add some random negative memberships
        negative_mask = torch.rand_like(membership) < sparsity * 0.2
        membership[negative_mask] = -torch.rand(negative_mask.sum()) * 0.5

    return membership


def load_and_split_datasets(dataset_paths, train_ratio=0.6, val_ratio=0.3, test_ratio=0.1,
                            feature_dim=128, membership_type='enhanced', random_seed=42):
    """
    Load multiple datasets, merge them, and create train/val/test splits

    Args:
        dataset_paths: Dictionary with dataset names as keys and (label_path, hyperedge_path) as values
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set  
        test_ratio: Fraction for test set
        feature_dim: Dimension of node features
        membership_type: Type of membership function to create
        random_seed: Random seed for reproducible splits

    Returns:
        Dictionary containing train/val/test splits and metadata
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print("Loading individual datasets...")
    datasets = []

    for dataset_name, (label_path, hyperedge_path) in dataset_paths.items():
        if os.path.exists(label_path) and os.path.exists(hyperedge_path):
            dataset = load_single_hypergraph_dataset(
                label_path, hyperedge_path, dataset_name)
            datasets.append(dataset)
            print(f"Loaded {dataset_name}: {dataset['num_nodes']} nodes")
        else:
            print(f"Warning: Dataset files not found for {dataset_name}")
            print(f"  Label path: {label_path}")
            print(f"  Hyperedge path: {hyperedge_path}")

    if not datasets:
        raise ValueError("No valid datasets found!")

    # Merge datasets
    print("\nMerging datasets...")
    merged_dataset = merge_hypergraph_datasets(datasets)

    # Create enhanced node features
    print(f"Creating enhanced node features (dim={feature_dim})...")
    x = create_enhanced_node_features(merged_dataset, feature_dim)

    # Create membership function
    print(f"Creating {membership_type} membership function...")
    membership = create_membership_function(
        merged_dataset['hyperedge_index'],
        merged_dataset['num_nodes'],
        membership_type=membership_type
    )

    # Create train/val/test splits based on nodes
    num_nodes = merged_dataset['num_nodes']
    node_indices = np.arange(num_nodes)

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        node_indices, test_size=(val_ratio + test_ratio), random_state=random_seed
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=(1 - val_size), random_state=random_seed
    )

    # Convert to tensors
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    print(f"\nDataset splits:")
    print(f"  Training: {len(train_indices)} nodes ({
          len(train_indices)/num_nodes*100:.1f}%)")
    print(f"  Validation: {len(val_indices)} nodes ({
          len(val_indices)/num_nodes*100:.1f}%)")
    print(f"  Test: {len(test_indices)} nodes ({
          len(test_indices)/num_nodes*100:.1f}%)")

    return {
        'x': x,
        'hyperedge_index': merged_dataset['hyperedge_index'],
        'membership': membership,
        'labels': merged_dataset['labels'],
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'num_nodes': num_nodes,
        'num_hyperedges': merged_dataset['hyperedge_index'][0].max().item() + 1,
        'dataset_info': {
            'original_datasets': [d['name'] for d in datasets],
            'feature_dim': feature_dim,
            'membership_type': membership_type,
            'split_ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
            'dataset_offsets': merged_dataset['dataset_offsets']
        }
    }


def load_highschool_hypergraph(label_path, hyperedge_path):
    """
    Legacy function for backward compatibility
    Load hypergraph dataset from label and hyperedge files
    """
    dataset = load_single_hypergraph_dataset(
        label_path, hyperedge_path, "legacy")

    # Create simple identity features for backward compatibility
    x = torch.eye(dataset['num_nodes'])

    return x, dataset['hyperedge_index'], dataset['labels']


def create_membership_function(hyperedge_index, num_nodes, sparsity=0.1, membership_type='enhanced'):
    """
    Create membership function for hyperedges with different strategies

    Args:
        hyperedge_index: Hyperedge connectivity tensor
        num_nodes: Total number of nodes
        sparsity: Fraction of connections to modify
        membership_type: Type of membership ('basic', 'enhanced', 'weighted')

    Returns:
        membership: Tensor of shape [num_hyperedges, num_nodes]
    """
    num_hyperedges = hyperedge_index[0].max().item() + 1
    membership = torch.zeros(num_hyperedges, num_nodes)

    # Basic membership assignment
    for i in range(hyperedge_index.size(1)):
        e_idx = hyperedge_index[0, i]
        n_idx = hyperedge_index[1, i]
        membership[e_idx, n_idx] = 1.0

    if membership_type == 'basic':
        # Simple random negative assignments
        mask = torch.rand_like(membership) < sparsity
        membership[mask] = -1.0

    elif membership_type == 'enhanced':
        # More sophisticated membership assignment

        # 1. Create some partial memberships
        partial_mask = torch.rand_like(membership) < sparsity * 0.5
        membership[partial_mask] = torch.rand(
            partial_mask.sum()) * 0.8 + 0.2  # 0.2 to 1.0

        # 2. Create some negative memberships (opposing nodes)
        negative_mask = torch.rand_like(membership) < sparsity * 0.3
        membership[negative_mask] = - \
            torch.rand(negative_mask.sum()) * 0.5 - 0.1  # -0.6 to -0.1

        # 3. Add weak connections between nearby hyperedges
        for e1 in range(num_hyperedges):
            nodes_e1 = (membership[e1] > 0).nonzero().flatten()
            if len(nodes_e1) > 0:
                for e2 in range(e1 + 1, min(e1 + 5, num_hyperedges)):  # Check nearby hyperedges
                    nodes_e2 = (membership[e2] > 0).nonzero().flatten()
                    # If hyperedges share nodes, create weak cross-memberships
                    shared_nodes = set(nodes_e1.tolist()) & set(
                        nodes_e2.tolist())
                    if len(shared_nodes) > 0 and torch.rand(1).item() < 0.3:
                        # Add weak membership
                        for node in nodes_e1:
                            if membership[e2, node] == 0 and torch.rand(1).item() < 0.2:
                                membership[e2, node] = torch.rand(
                                    1).item() * 0.3  # Weak membership

    elif membership_type == 'weighted':
        node_degrees = torch.zeros(num_nodes)
        for i in range(hyperedge_index.size(1)):
            node_id = hyperedge_index[1, i].item()
            node_degrees[node_id] += 1

        max_degree = node_degrees.max().item()
        normalized_degrees = node_degrees / max_degree if max_degree > 0 else node_degrees

        for e_idx in range(num_hyperedges):
            nodes_in_edge = (membership[e_idx] > 0).nonzero().flatten()
            for node in nodes_in_edge:
                # Higher degree nodes get stronger membership
                membership[e_idx, node] = 0.5 + 0.5 * normalized_degrees[node]

        # Add some random negative memberships
        negative_mask = torch.rand_like(membership) < sparsity * 0.2

    return membership
