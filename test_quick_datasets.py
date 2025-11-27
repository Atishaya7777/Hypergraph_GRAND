#!/usr/bin/env python3
"""
Quick Dataset Compatibility Test - Test key datasets
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.pyg_standardizer import DatasetLoader


def test_quick():
    """Quick test of key datasets"""
    print("\n" + "="*80)
    print("QUICK DATASET COMPATIBILITY TEST")
    print("="*80 + "\n")
    
    loader = DatasetLoader(base_path="datasets")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test a few key datasets
    test_datasets = {
        'classification': ['cora', 'house_committees'],
        'clustering': ['contact_high_school', 'contact_primary_school', 'walmart_trips', 'yelp'],
        'partitioning': ['zoo', 'mushroom']
    }
    
    passed = 0
    total = 0
    
    for task_type, datasets in test_datasets.items():
        print(f"{task_type.upper()}")
        print("-" * 80)
        
        for dataset_name in datasets:
            total += 1
            try:
                # Load dataset
                print(f"  {dataset_name:<30}", end=" ... ", flush=True)
                data = loader.load(dataset_name, verbose=False)
                
                # Validate
                assert data.x is not None, "Features are None"
                assert data.y is not None, "Labels are None"
                assert isinstance(data.x, torch.Tensor), f"Features not tensor: {type(data.x)}"
                assert isinstance(data.y, torch.Tensor), f"Labels not tensor: {type(data.y)}"
                assert data.train_mask is not None, "No train mask"
                assert data.val_mask is not None, "No val mask"
                assert data.test_mask is not None, "No test mask"
                
                # Test model forward pass
                from models import create_hypergrand_model
                input_dim = data.x.shape[1]
                num_classes = int(data.y.max().item()) + 1 if data.y.max() >= 0 else 2
                
                model = create_hypergrand_model(
                    input_dim=input_dim,
                    hidden_dim=16,
                    num_classes=num_classes,
                    dropout=0.0
                )
                model = model.to(device)
                x = data.x.to(device)
                hyperedge_index = data.hyperedge_index.to(device) if hasattr(data, 'hyperedge_index') and data.hyperedge_index is not None else torch.zeros((2, 0), dtype=torch.long, device=device)
                
                with torch.no_grad():
                    output = model(x, hyperedge_index)
                
                assert output.shape[0] == data.num_nodes, "Output shape mismatch"
                assert output.shape[1] == num_classes, "Output classes mismatch"
                
                print(f"✓ PASS | nodes={data.num_nodes:>5} | classes={num_classes:>2} | features={data.x.shape[1]:>4}")
                passed += 1
            
            except Exception as e:
                print(f"✗ FAIL")
                print(f"      Error: {str(e)[:100]}")
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed}/{total} datasets passed ({100*passed/total:.0f}%)")
    print("="*80 + "\n")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(test_quick())
