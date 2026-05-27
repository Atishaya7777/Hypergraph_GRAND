#!/usr/bin/env python3
"""
Quick test to verify diffusion study framework is working correctly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from experiments.diffusion_config import DiffusionStudyConfig, DiffusionStudySpace
        print("  ✓ diffusion_config imported")
    except Exception as e:
        print(f"  ✗ Failed to import diffusion_config: {e}")
        return False
    
    try:
        from experiments.run_diffusion_studies import DiffusionStudyRunner
        print("  ✓ run_diffusion_studies imported")
    except Exception as e:
        print(f"  ✗ Failed to import run_diffusion_studies: {e}")
        return False
    
    try:
        from experiments.analyze_diffusion_results import DiffusionStudyAnalyzer
        print("  ✓ analyze_diffusion_results imported")
    except Exception as e:
        print(f"  ✗ Failed to import analyze_diffusion_results: {e}")
        return False
    
    try:
        from main import MLFlowLogger
        print("  ✓ MLFlowLogger imported")
    except Exception as e:
        print(f"  ✗ Failed to import MLFlowLogger: {e}")
        return False
    
    return True


def test_config_generation():
    """Test that configuration generation works"""
    print("\nTesting configuration generation...")
    
    try:
        from experiments.diffusion_config import DiffusionStudySpace
        
        # Test baseline config
        baseline = DiffusionStudySpace.get_baseline_config('classification')
        assert baseline.hidden_dim == 64
        assert baseline.integration_scheme == 'implicit'
        print("  ✓ Baseline config generated")
        
        # Test study config generation
        configs = DiffusionStudySpace.generate_all_study_configs('classification')
        assert 'integration_scheme' in configs
        assert 'diffusion_depth' in configs
        assert 'attention_mechanism' in configs
        print(f"  ✓ Study configs generated ({sum(len(v) for v in configs.values())} total)")
        
        # Test fast mode configs
        fast_configs = DiffusionStudySpace.get_fast_mode_configs('clustering')
        assert 'baseline' in fast_configs
        assert 'integration_scheme' in fast_configs
        print(f"  ✓ Fast mode configs generated ({sum(len(v) for v in fast_configs.values())} total)")
        
        return True
    except Exception as e:
        print(f"  ✗ Config generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mlflow_logger():
    """Test MLflow logger basic functionality"""
    print("\nTesting MLflow logger...")
    
    try:
        from main import MLFlowLogger
        
        # Test without actual MLflow connection
        logger = MLFlowLogger(enabled=False)
        
        # These should not raise errors even when disabled
        logger.start_run("test_dataset", "classification", {})
        logger.log_metrics({"test_metric": 0.5})
        logger.end_run()
        
        print("  ✓ MLFlowLogger basic operations work")
        return True
    except Exception as e:
        print(f"  ✗ MLFlowLogger test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_train_dataset_signature():
    """Test that train_dataset has correct signature"""
    print("\nTesting train_dataset signature...")
    
    try:
        from train_model import train_dataset
        import inspect
        
        sig = inspect.signature(train_dataset)
        params = list(sig.parameters.keys())
        
        required_params = ['dataset_name', 'seed', 'mlflow_logger', 'parent_run_id', 'config']
        for param in required_params:
            assert param in params, f"Missing parameter: {param}"
        
        print(f"  ✓ train_dataset has all required parameters: {', '.join(required_params)}")
        return True
    except Exception as e:
        print(f"  ✗ train_dataset signature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("DIFFUSION STUDY FRAMEWORK - SMOKE TESTS")
    print("="*80)
    
    tests = [
        test_imports,
        test_config_generation,
        test_mlflow_logger,
        test_train_dataset_signature,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*80)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
        print("="*80)
        print("\nFramework is ready to use!")
        print("\nQuick start:")
        print("  make diffusion-fast    # Test on 3 datasets (~4 hours)")
        print("  make mlflow            # View results")
        return 0
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total} passed)")
        print("="*80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
