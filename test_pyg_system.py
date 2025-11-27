#!/usr/bin/env python3
"""
Quick test script to verify the PyG data system works correctly
"""

import sys
import torch

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from data import (
            UnifiedDataManager,
            load_dataset,
            load_datasets,
            HypergraphDataConverter,
            PyGDatasetLoader,
            PyGDataProcessor
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_loading():
    """Test that metadata loads correctly"""
    print("\nTesting metadata loading...")
    try:
        from data.pyg_converter import PyGDatasetLoader
        loader = PyGDatasetLoader()
        
        metadata = loader._metadata_cache
        if not metadata:
            print("✗ Metadata cache is empty")
            return False
        
        # Check structure
        expected_tasks = ['classification', 'clustering', 'partitioning']
        for task in expected_tasks:
            if task not in metadata:
                print(f"✗ Missing task type: {task}")
                return False
        
        # Count datasets
        total_datasets = sum(
            len(metadata[task].get('datasets', {}))
            for task in expected_tasks
        )
        
        print(f"✓ Metadata loaded: {total_datasets} datasets found")
        return True
    except Exception as e:
        print(f"✗ Metadata loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_manager_creation():
    """Test that UnifiedDataManager can be created"""
    print("\nTesting UnifiedDataManager creation...")
    try:
        from data import UnifiedDataManager
        manager = UnifiedDataManager()
        
        # Test listing datasets
        datasets = manager.list_datasets()
        if not datasets:
            print("✗ No datasets found")
            return False
        
        print(f"✓ Manager created successfully")
        print(f"  Available task types: {list(datasets.keys())}")
        
        for task_type, names in datasets.items():
            print(f"  {task_type}: {len(names)} datasets")
        
        return True
    except Exception as e:
        print(f"✗ Manager creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test loading a small dataset"""
    print("\nTesting dataset loading...")
    try:
        from data import load_dataset
        
        # Try to load contact high school (one of the smaller ones)
        print("  Loading contact_high_school...")
        data = load_dataset('contact_high_school', verbose=False)
        
        # Check structure
        if not hasattr(data, 'x') or not hasattr(data, 'y'):
            print("✗ Data missing x or y")
            return False
        
        if not hasattr(data, 'metadata'):
            print("✗ Data missing metadata")
            return False
        
        # Check sizes
        if data.x.shape[0] != data.num_nodes:
            print(f"✗ Feature matrix size mismatch")
            return False
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Features: {data.x.shape[1]}")
        print(f"  Classes: {data.metadata.num_classes}")
        print(f"  Task Type: {data.metadata.task_type}")
        print(f"  Strategy: {data.metadata.strategy}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_pyg_processor():
    """Test PyG processor utilities"""
    print("\nTesting PyGDataProcessor...")
    try:
        from data import load_dataset
        from data.pyg_converter import PyGDataProcessor
        
        data = load_dataset('contact_high_school', verbose=False)
        processor = PyGDataProcessor()
        
        # Test integrity check
        is_valid, issues = processor.verify_data_integrity(data)
        
        if is_valid:
            print("✓ Data integrity check passed")
        else:
            print(f"⚠ Data integrity issues found: {issues}")
        
        # Test split info
        if hasattr(data, 'train_mask'):
            split_info = processor.get_split_info(data)
            print(f"✓ Split info retrieved: {list(split_info.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Processor test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("HYPERGRAND PyG DATA SYSTEM - VERIFICATION TESTS")
    print("="*70)
    
    tests = [
        test_imports,
        test_metadata_loading,
        test_manager_creation,
        test_dataset_loading,
        test_pyg_processor
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
        print("\nThe PyG data system is working correctly!")
        print("\nYou can now use:")
        print("  from data import load_dataset, UnifiedDataManager")
        return 0
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total} passed)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
