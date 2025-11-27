#!/usr/bin/env python3
"""
Dataset Compatibility Test - Verify all datasets load and work with the model
"""

import sys
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from data.pyg_standardizer import DatasetLoader


class DatasetCompatibilityTester:
    """Test all datasets for compatibility with the model"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.loader = DatasetLoader(base_path="datasets")
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_all_datasets(self) -> Dict[str, Any]:
        """Test all datasets comprehensively"""
        all_datasets = self.loader.list_datasets()
        total_count = sum(len(v) for v in all_datasets.values())
        tested_count = 0
        passed_count = 0
        
        # List of small datasets to test first (to avoid OOM)
        small_datasets = {
            'classification': ['cora', 'house_committees', 'coauthorship_cora', 'coauthorship_dblp', 'cocitation_citeseer', 'cocitation_cora', 'cocitation_pubmed'],
            'clustering': ['contact_high_school', 'contact_primary_school', 'walmart_trips', 'news_20w100', 'yelp'],
            'partitioning': ['zoo', 'mushroom', 'ntu2012', 'modelnet40']
        }
        
        print("\n" + "="*80)
        print("DATASET COMPATIBILITY TEST - ALL DATASETS")
        print("="*80)
        print(f"\nTotal datasets to test: {total_count}")
        print(f"Device: {self.device}\n")
        print("Note: Testing smaller datasets first to manage memory. Large datasets (stackoverflow_answers, amazon_reviews)")
        print("      may be skipped if memory constraints are detected.\n")
        
        # Test by task type
        for task_type in ['classification', 'clustering', 'partitioning']:
            if task_type not in all_datasets:
                continue
            
            print(f"\n{'-'*80}")
            print(f"{task_type.upper()} ({len(all_datasets[task_type])} datasets)")
            print(f"{'-'*80}\n")
            
            self.results[task_type] = {}
            
            for dataset_name in all_datasets[task_type]:
                # Skip very large datasets to avoid OOM
                skip_large = dataset_name in ['stackoverflow_answers', 'amazon_reviews']
                if skip_large:
                    print(f"  ⊘ {dataset_name:<30} SKIPPED (very large dataset)")
                    continue
                
                tested_count += 1
                try:
                    result = self.test_dataset(dataset_name, task_type)
                    self.results[task_type][dataset_name] = result
                    
                    if result['status'] == 'PASS':
                        passed_count += 1
                        self._print_result(dataset_name, result, passed=True)
                    else:
                        self._print_result(dataset_name, result, passed=False)
                
                except Exception as e:
                    self.results[task_type][dataset_name] = {
                        'status': 'ERROR',
                        'error': str(e)
                    }
                    self._print_error(dataset_name, str(e))
        
        # Summary
        self._print_summary(tested_count, passed_count)
        
        return self.results
    
    def test_dataset(self, dataset_name: str, task_type: str) -> Dict[str, Any]:
        """Test a single dataset for compatibility"""
        if self.verbose:
            print(f"  Testing {dataset_name}...", end=" ", flush=True)
        
        # Load dataset
        try:
            data = self.loader.load(dataset_name, verbose=False)
        except Exception as e:
            return {
                'status': 'LOAD_FAILED',
                'error': str(e)
            }
        
        # Validate PyG Data object
        try:
            self._validate_pyg_data(data)
        except Exception as e:
            return {
                'status': 'INVALID_FORMAT',
                'error': str(e)
            }
        
        # Validate shapes and compatibility
        try:
            self._validate_compatibility(data, task_type)
        except Exception as e:
            return {
                'status': 'INCOMPATIBLE',
                'error': str(e)
            }
        
        # Test model compatibility
        try:
            self._test_model_forward_pass(data)
        except Exception as e:
            return {
                'status': 'MODEL_FORWARD_FAILED',
                'error': str(e)
            }
        
        return {
            'status': 'PASS',
            'num_nodes': int(data.num_nodes),
            'num_classes': int(data.y.max().item()) + 1 if data.y is not None else 0,
            'num_features': int(data.x.shape[1]) if data.x is not None else 0,
            'num_hyperedges': int(data.hyperedge_index.shape[1]) if hasattr(data, 'hyperedge_index') and data.hyperedge_index is not None else 0,
            'task_type': task_type
        }
    
    def _validate_pyg_data(self, data):
        """Validate that data is proper PyG Data object"""
        required_attrs = ['num_nodes', 'x', 'y']
        for attr in required_attrs:
            if not hasattr(data, attr):
                raise ValueError(f"Missing required attribute: {attr}")
        
        if data.x is None:
            raise ValueError("Features (x) cannot be None")
        
        if data.y is None:
            raise ValueError("Labels (y) cannot be None")
        
        if not isinstance(data.x, torch.Tensor):
            raise ValueError(f"Features must be tensor, got {type(data.x)}")
        
        if not isinstance(data.y, torch.Tensor):
            raise ValueError(f"Labels must be tensor, got {type(data.y)}")
    
    def _validate_compatibility(self, data, task_type: str):
        """Validate dataset compatibility with model"""
        # Check shapes
        if data.x.shape[0] != data.num_nodes:
            raise ValueError(
                f"Feature shape mismatch: {data.x.shape[0]} != {data.num_nodes}"
            )
        
        if data.y.shape[0] != data.num_nodes:
            raise ValueError(
                f"Label shape mismatch: {data.y.shape[0]} != {data.num_nodes}"
            )
        
        # Check train/val/test masks
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            raise ValueError("Missing train_mask")
        
        if not hasattr(data, 'val_mask') or data.val_mask is None:
            raise ValueError("Missing val_mask")
        
        if not hasattr(data, 'test_mask') or data.test_mask is None:
            raise ValueError("Missing test_mask")
        
        # Validate masks
        total_masked = (data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()).item()
        if total_masked == 0:
            raise ValueError("No nodes in any split")
        
        # Check for reasonable split sizes
        train_size = data.train_mask.sum().item()
        if train_size < 1:
            raise ValueError("Train set is empty")
    
    def _test_model_forward_pass(self, data):
        """Test that data works with a simple forward pass"""
        try:
            # Import model
            from models import create_hypergrand_model
            
            # Create model
            input_dim = data.x.shape[1]
            num_classes = int(data.y.max().item()) + 1 if data.y.max() >= 0 else 2
            model = create_hypergrand_model(
                input_dim=input_dim,
                hidden_dim=16,
                num_classes=num_classes,
                dropout=0.0
            )
            
            # Move to device
            model = model.to(self.device)
            x = data.x.to(self.device)
            hyperedge_index = data.hyperedge_index.to(self.device) if hasattr(data, 'hyperedge_index') and data.hyperedge_index is not None else torch.zeros((2, 0), dtype=torch.long, device=self.device)
            
            # Forward pass
            with torch.no_grad():
                output = model(x, hyperedge_index)
            
            # Validate output
            if output.shape[0] != data.num_nodes:
                raise ValueError(f"Output shape mismatch: {output.shape[0]} != {data.num_nodes}")
            
            if output.shape[1] != num_classes:
                raise ValueError(f"Output classes mismatch: {output.shape[1]} != {num_classes}")
        
        except Exception as e:
            raise Exception(f"Model forward pass failed: {str(e)}")
    
    def _print_result(self, name: str, result: Dict, passed: bool = True):
        """Print test result"""
        status_sym = "✓" if passed else "✗"
        
        if passed:
            print(f"{status_sym} PASS")
            print(f"    Nodes: {result['num_nodes']:>8} | Classes: {result['num_classes']:>3} | Features: {result['num_features']:>5} | Hyperedges: {result['num_hyperedges']:>8}")
        else:
            print(f"{status_sym} FAIL - {result.get('status', 'UNKNOWN')}")
            if 'error' in result:
                print(f"    Error: {result['error'][:100]}")
    
    def _print_error(self, name: str, error: str):
        """Print error"""
        print(f"✗ ERROR")
        print(f"    {error[:100]}")
    
    def _print_summary(self, total: int, passed: int):
        """Print summary"""
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Total datasets tested: {total}")
        print(f"Passed:                {passed}")
        print(f"Failed:                {total - passed}")
        print(f"Success rate:          {100*passed/total:.1f}%")
        
        # Group results by status
        status_counts = {}
        for task_type in self.results.values():
            for dataset_result in task_type.values():
                status = dataset_result.get('status', 'UNKNOWN')
                status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\nStatus breakdown:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count}")
        
        if passed == total:
            print(f"\n🎉 ALL TESTS PASSED! All {total} datasets are compatible with the model.")
        else:
            print(f"\n⚠️  {total - passed} dataset(s) failed. See details above.")
        
        print(f"\n{'='*80}\n")
    
    def save_results(self, output_file: str = "dataset_test_results.json"):
        """Save test results to JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")


def main():
    """Main test entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test all datasets for model compatibility')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output', default='dataset_test_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    # Run tests
    tester = DatasetCompatibilityTester(verbose=args.verbose)
    results = tester.test_all_datasets()
    
    # Save results
    tester.save_results(args.output)
    
    # Return exit code based on results
    all_passed = all(
        result.get('status') == 'PASS'
        for task_type_results in results.values()
        for result in task_type_results.values()
    )
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
