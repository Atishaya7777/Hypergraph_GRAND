#!/usr/bin/env python3
"""
Quick Dataset Structure Validation Test
Tests that all datasets load and have valid structure - no model involved
"""

import sys
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from data.pyg_standardizer import DatasetLoader


class DatasetStructureValidator:
    """Validates dataset structure without using the model"""
    
    def __init__(self):
        self.loader = DatasetLoader(base_path="datasets")
        self.results = {}
    
    def validate_all(self) -> Dict:
        """Validate all datasets"""
        all_datasets = self.loader.list_datasets()
        
        print("\n" + "="*100)
        print("DATASET STRUCTURE VALIDATION TEST")
        print("="*100)
        print(f"\nValidating {sum(len(v) for v in all_datasets.values())} datasets...\n")
        
        total = 0
        valid = 0
        
        for task_type in sorted(all_datasets.keys()):
            print(f"\n{task_type.upper()}")
            print("-" * 100)
            
            self.results[task_type] = {}
            
            for dataset_name in sorted(all_datasets[task_type]):
                total += 1
                result = self.validate_dataset(dataset_name, task_type)
                self.results[task_type][dataset_name] = result
                
                if result['valid']:
                    valid += 1
                    self._print_success(dataset_name, result)
                else:
                    self._print_failure(dataset_name, result)
        
        self._print_summary(total, valid)
        return self.results
    
    def validate_dataset(self, dataset_name: str, task_type: str) -> Dict:
        """Validate a single dataset structure"""
        try:
            # Step 1: Load dataset
            try:
                data = self.loader.load(dataset_name, verbose=False)
            except Exception as e:
                return {
                    'valid': False,
                    'error': f"Failed to load: {str(e)[:80]}"
                }
            
            # Step 2: Check basic PyG properties
            issues = []
            
            if data.x is None:
                issues.append("Features (x) is None")
            elif not isinstance(data.x, torch.Tensor):
                issues.append(f"Features not tensor: {type(data.x)}")
            elif data.x.dim() != 2:
                issues.append(f"Features wrong dims: {data.x.shape}")
            
            if data.y is None:
                issues.append("Labels (y) is None")
            elif not isinstance(data.y, torch.Tensor):
                issues.append(f"Labels not tensor: {type(data.y)}")
            elif data.y.dim() != 1:
                issues.append(f"Labels wrong dims: {data.y.shape}")
            
            if data.num_nodes is None or data.num_nodes <= 0:
                issues.append(f"Invalid num_nodes: {data.num_nodes}")
            
            if not hasattr(data, 'hyperedge_index') or data.hyperedge_index is None:
                issues.append("Missing hyperedge_index")
            elif not isinstance(data.hyperedge_index, torch.Tensor):
                issues.append(f"hyperedge_index not tensor: {type(data.hyperedge_index)}")
            elif data.hyperedge_index.dim() != 2:
                issues.append(f"hyperedge_index wrong dims: {data.hyperedge_index.shape}")
            elif data.hyperedge_index.shape[0] != 2:
                issues.append(f"hyperedge_index not 2xE: {data.hyperedge_index.shape}")
            
            # Step 3: Check consistency
            if data.x is not None and data.num_nodes is not None:
                if data.x.shape[0] != data.num_nodes:
                    issues.append(f"Feature nodes mismatch: {data.x.shape[0]} != {data.num_nodes}")
            
            if data.y is not None and data.num_nodes is not None:
                if data.y.shape[0] != data.num_nodes:
                    issues.append(f"Label nodes mismatch: {data.y.shape[0]} != {data.num_nodes}")
            
            # Step 4: Check splits
            if not hasattr(data, 'train_mask') or data.train_mask is None:
                issues.append("Missing train_mask")
            elif data.train_mask.shape[0] != data.num_nodes:
                issues.append(f"train_mask size mismatch: {data.train_mask.shape[0]} != {data.num_nodes}")
            
            if not hasattr(data, 'val_mask') or data.val_mask is None:
                issues.append("Missing val_mask")
            elif data.val_mask.shape[0] != data.num_nodes:
                issues.append(f"val_mask size mismatch: {data.val_mask.shape[0]} != {data.num_nodes}")
            
            if not hasattr(data, 'test_mask') or data.test_mask is None:
                issues.append("Missing test_mask")
            elif data.test_mask.shape[0] != data.num_nodes:
                issues.append(f"test_mask size mismatch: {data.test_mask.shape[0]} != {data.num_nodes}")
            
            # Step 5: Compute statistics
            num_classes = int(data.y.max().item()) + 1 if data.y.max() >= 0 else 2
            num_edges = data.hyperedge_index.shape[1] if data.hyperedge_index is not None else 0
            train_nodes = data.train_mask.sum().item() if data.train_mask is not None else 0
            val_nodes = data.val_mask.sum().item() if data.val_mask is not None else 0
            test_nodes = data.test_mask.sum().item() if data.test_mask is not None else 0
            
            if issues:
                return {
                    'valid': False,
                    'errors': issues,
                    'num_nodes': data.num_nodes,
                    'num_features': data.x.shape[1] if data.x is not None else 0,
                    'num_classes': num_classes,
                    'num_hyperedges': num_edges
                }
            
            return {
                'valid': True,
                'num_nodes': data.num_nodes,
                'num_features': data.x.shape[1] if data.x is not None else 0,
                'num_classes': num_classes,
                'num_hyperedges': num_edges,
                'train_nodes': train_nodes,
                'val_nodes': val_nodes,
                'test_nodes': test_nodes,
                'train_pct': f"{100*train_nodes/data.num_nodes:.1f}%",
                'val_pct': f"{100*val_nodes/data.num_nodes:.1f}%",
                'test_pct': f"{100*test_nodes/data.num_nodes:.1f}%"
            }
        
        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation error: {str(e)[:80]}"
            }
    
    def _print_success(self, name: str, result: Dict):
        """Print successful validation"""
        print(f"  ✓ {name:<30} | nodes={result['num_nodes']:>8} | classes={result['num_classes']:>2} | features={result['num_features']:>5} | edges={result['num_hyperedges']:>8}")
        print(f"    {'':30}   splits: train {result['train_pct']:>6} | val {result['val_pct']:>6} | test {result['test_pct']:>6}")
    
    def _print_failure(self, name: str, result: Dict):
        """Print failed validation"""
        print(f"  ✗ {name:<30}")
        if 'error' in result:
            print(f"    Error: {result['error']}")
        elif 'errors' in result:
            for error in result['errors']:
                print(f"    • {error}")
    
    def _print_summary(self, total: int, valid: int):
        """Print summary statistics"""
        print("\n" + "="*100)
        print("VALIDATION SUMMARY")
        print("="*100)
        
        print(f"\nTotal datasets:    {total}")
        print(f"Valid:             {valid}")
        print(f"Failed:            {total - valid}")
        print(f"Success rate:      {100*valid/total:.1f}%")
        
        # Group by status
        status_counts = {}
        for task_results in self.results.values():
            for dataset_result in task_results.values():
                status = 'valid' if dataset_result['valid'] else 'invalid'
                status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\nStatus breakdown:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count}")
        
        if valid == total:
            print(f"\n✅ ALL DATASETS PASSED VALIDATION!")
            print(f"All {total} datasets have valid structure and can be used for training.\n")
        else:
            print(f"\n⚠️  {total - valid} dataset(s) failed validation.")
            print(f"See details above for issues.\n")
        
        return valid == total


def main():
    """Main entry point"""
    validator = DatasetStructureValidator()
    results = validator.validate_all()
    
    all_valid = all(
        result['valid']
        for task_results in results.values()
        for result in task_results.values()
    )
    
    return 0 if all_valid else 1


if __name__ == '__main__':
    sys.exit(main())
