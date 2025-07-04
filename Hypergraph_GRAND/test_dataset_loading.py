#!/usr/bin/env python3
"""
Quick test script to verify dataset loading works correctly
"""

import torch
import os
import sys

# Add current directory to path to import main
sys.path.append('.')

# Import the dataset loading function from main
from main import load_hypergraph_dataset

def test_dataset_loading():
    """Test dataset loading for both datasets"""
    
    datasets = [
        './datasets/contact-primary-school/',
        './datasets/contact-high-school/'
    ]
    
    for dataset_path in datasets:
        print(f"\n{'='*60}")
        print(f"Testing dataset: {os.path.basename(dataset_path)}")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            data = load_hypergraph_dataset(dataset_path)
            
            # Verify shapes and basic properties
            print(f"\nDataset loaded successfully:")
            print(f"  Nodes: {data['num_nodes']}")
            print(f"  Hyperedges: {data['num_hyperedges']}")
            print(f"  Clusters: {data['num_clusters']}")
            print(f"  Node features shape: {data['x'].shape}")
            print(f"  Hyperedge index shape: {data['hyperedge_index'].shape}")
            print(f"  Membership matrix shape: {data['membership'].shape}")
            print(f"  Node labels shape: {data['node_labels'].shape}")
            
            # Check if node indices are valid (0-indexed)
            hyperedge_index = data['hyperedge_index']
            if hyperedge_index.size(1) > 0:
                max_node_in_edges = hyperedge_index[1, :].max().item()
                min_node_in_edges = hyperedge_index[1, :].min().item()
                print(f"  Node index range in hyperedges: [{min_node_in_edges}, {max_node_in_edges}]")
                print(f"  Expected range: [0, {data['num_nodes']-1}]")
                
                if max_node_in_edges >= data['num_nodes']:
                    print(f"  ❌ ERROR: Node index {max_node_in_edges} >= num_nodes {data['num_nodes']}")
                elif min_node_in_edges < 0:
                    print(f"  ❌ ERROR: Node index {min_node_in_edges} < 0")
                else:
                    print(f"  ✅ Node indices are valid")
            
            # Check cluster distribution
            labels = data['node_labels']
            unique_labels = torch.unique(labels)
            print(f"  Cluster distribution:")
            for label in unique_labels:
                count = (labels == label).sum().item()
                cluster_name = data['cluster_names'][unique_labels.tolist().index(label.item())] if unique_labels.tolist().index(label.item()) < len(data['cluster_names']) else f"Cluster_{label.item()}"
                print(f"    {cluster_name}: {count} nodes")
                
        except Exception as e:
            print(f"❌ ERROR loading dataset: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_dataset_loading()
