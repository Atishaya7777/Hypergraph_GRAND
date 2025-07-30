import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from data.dataset import create_hypergraph_dataset

# Load standard dataset with normalization
standard_dataset = Planetoid(root='/tmp/Cora_standard', name='Cora', transform=NormalizeFeatures())
standard_data = standard_dataset[0]

# Load your updated implementation
your_data = create_hypergraph_dataset('planetoid_cora', normalize_features=True).load_data('/tmp/cora_data')

print("=== FEATURE COMPARISON ===")
print(f"Standard features shape: {standard_data.x.shape}")
print(f"Your features shape: {your_data.node_features.shape}")
print(f"Standard features mean: {standard_data.x.mean():.6f}")
print(f"Your features mean: {your_data.node_features.mean():.6f}")
print(f"Standard features std: {standard_data.x.std():.6f}")
print(f"Your features std: {your_data.node_features.std():.6f}")

# Check if they match exactly
features_match = torch.allclose(standard_data.x, your_data.node_features, atol=1e-6)
print(f"Features match exactly: {features_match}")

if not features_match:
    diff = torch.abs(standard_data.x - your_data.node_features)
    print(f"Max difference: {diff.max():.8f}")
    print(f"Mean difference: {diff.mean():.8f}")

print("\n=== LABELS COMPARISON ===")
labels_match = torch.equal(standard_data.y, your_data.labels)
print(f"Labels match exactly: {labels_match}")

print("\n=== MASKS COMPARISON ===")
train_match = torch.equal(standard_data.train_mask, your_data.train_mask)
val_match = torch.equal(standard_data.val_mask, your_data.val_mask)
test_match = torch.equal(standard_data.test_mask, your_data.test_mask)
print(f"Train mask match: {train_match}")
print(f"Val mask match: {val_match}")
print(f"Test mask match: {test_match}")

print("\n=== EDGES COMPARISON ===")
print(f"Standard edges shape: {standard_data.edge_index.shape}")
print(f"Standard num edges: {standard_data.edge_index.shape[1]}")
print(f"Your original edges: {your_data.dataset_info['original_num_edges']}")

# Test without normalization for comparison
print("\n=== WITHOUT NORMALIZATION ===")
your_data_unnorm = create_hypergraph_dataset('planetoid_cora', normalize_features=False).load_data('/tmp/cora_data')
print(f"Unnormalized features mean: {your_data_unnorm.node_features.mean():.6f}")

# Load standard without normalization
standard_dataset_unnorm = Planetoid(root='/tmp/Cora_standard_unnorm', name='Cora')
standard_data_unnorm = standard_dataset_unnorm[0]
print(f"Standard unnormalized mean: {standard_data_unnorm.x.mean():.6f}")

unnorm_match = torch.allclose(standard_data_unnorm.x, your_data_unnorm.node_features, atol=1e-6)
print(f"Unnormalized features match: {unnorm_match}")
