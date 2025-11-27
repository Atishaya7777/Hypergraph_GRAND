#!/usr/bin/env python3
"""
Dataset Structure Validation - Tests dataset loading without models
"""

import sys
import torch
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent))

from data.pyg_standardizer import DatasetLoader


def validate_dataset(loader: DatasetLoader, dataset_name: str, task_type: str) -> Dict:
    """Validate a single dataset structure"""
    try:
        # Load dataset
        data = loader.load(dataset_name, verbose=False)
        
        # Check basic properties
        issues = []
        
        if data.x is None or not isinstance(data.x, torch.Tensor):
            issues.append(f"Features issue: {type(data.x)}")
        elif data.x.dim() != 2:
            issues.append(f"Features dims: {data.x.shape}")
        
        if data.y is None or not isinstance(data.y, torch.Tensor):
            issues.append(f"Labels issue: {type(data.y)}")
        elif data.y.dim() != 1:
            issues.append(f"Labels dims: {data.y.shape}")
        
        if data.num_nodes <= 0:
            issues.append(f"Invalid nodes: {data.num_nodes}")
        
        if not hasattr(data, 'hyperedge_index') or data.hyperedge_index is None:
            issues.append("Missing hyperedge_index")
        elif not isinstance(data.hyperedge_index, torch.Tensor) or data.hyperedge_index.dim() != 2:
            issues.append(f"hyperedge_index issue: {type(data.hyperedge_index)}")
        
        # Check consistency
        if data.x is not None and data.x.shape[0] != data.num_nodes:
            issues.append(f"Feature mismatch: {data.x.shape[0]} != {data.num_nodes}")
        
        if data.y is not None and data.y.shape[0] != data.num_nodes:
            issues.append(f"Label mismatch: {data.y.shape[0]} != {data.num_nodes}")
        
        # Check splits
        if not (hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask')):
            issues.append("Missing train/val/test masks")
        
        if issues:
            return {'valid': False, 'errors': issues}
        
        # Calculate stats
        num_classes = int(data.y.max().item()) + 1 if data.y.max() >= 0 else 2
        num_edges = data.hyperedge_index.shape[1]
        
        return {
            'valid': True,
            'num_nodes': data.num_nodes,
            'num_classes': num_classes,
            'num_features': data.x.shape[1],
            'num_edges': num_edges,
            'train_pct': f"{100*data.train_mask.sum().item()/data.num_nodes:.0f}%",
            'val_pct': f"{100*data.val_mask.sum().item()/data.num_nodes:.0f}%",
            'test_pct': f"{100*data.test_mask.sum().item()/data.num_nodes:.0f}%"
        }
    
    except Exception as e:
        return {'valid': False, 'error': str(e)[:80]}


def main():
    """Run validation tests"""
    loader = DatasetLoader(base_path="datasets")
    
    # Datasets to test (avoiding very large ones)
    datasets_to_test = {
        'classification': [
            'cora', 'coauthorship_cora', 'coauthorship_dblp',
            'cocitation_citeseer', 'cocitation_cora', 'cocitation_pubmed',
            'house_committees'
        ],
        'clustering': [
            'contact_high_school', 'contact_primary_school',
            'walmart_trips', 'news_20w100', 'yelp'
        ],
        'partitioning': [
            'zoo', 'mushroom', 'ntu2012', 'modelnet40'
        ]
    }
    
    print("\n" + "="*100)
    print("DATASET STRUCTURE VALIDATION TEST")
    print("="*100 + "\n")
    
    total = 0
    passed = 0
    results = {}
    
    for task_type in sorted(datasets_to_test.keys()):
        print(f"{task_type.upper()}")
        print("-" * 100)
        
        results[task_type] = {}
        
        for dataset_name in sorted(datasets_to_test[task_type]):
            total += 1
            result = validate_dataset(loader, dataset_name, task_type)
            results[task_type][dataset_name] = result
            
            if result['valid']:
                passed += 1
                print(f"  ✓ {dataset_name:<30} | nodes={result['num_nodes']:>8} | classes={result['num_classes']:>2} | features={result['num_features']:>5} | edges={result['num_edges']:>8}")
                print(f"    {'':30}   splits: train {result['train_pct']:>5} | val {result['val_pct']:>5} | test {result['test_pct']:>5}")
            else:
                errors = result.get('errors', [])
                error_msg = result.get('error', '')
                if errors:
                    error_msg = '; '.join(errors[:2])
                print(f"  ✗ {dataset_name:<30} | {error_msg}")
        
        print()
    
    # Summary
    print("=" * 100)
    print(f"RESULTS: {passed}/{total} datasets valid ({100*passed/total:.0f}%)")
    print("=" * 100 + "\n")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
