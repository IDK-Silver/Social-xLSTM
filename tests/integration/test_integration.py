#!/usr/bin/env python3
"""
Test script to verify the integrated functionality works correctly.

This script tests the integrated functions from the file reorganization to ensure
backward compatibility and proper functionality.
"""

import sys
import tempfile
import warnings
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
# Add project root to path for scripts imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_data_stabilization():
    """Test data stabilization integration."""
    print("🧪 Testing Data Stabilization Integration")
    print("=" * 50)
    
    try:
        from social_xlstm.dataset.storage.h5_converter import TrafficFeatureExtractor
        
        # Test that the class exists and has the expected methods
        assert hasattr(TrafficFeatureExtractor, 'validate_dataset_quality')
        assert hasattr(TrafficFeatureExtractor, 'stabilize_dataset')
        print("✅ TrafficFeatureExtractor methods available")
        
        # Test backward compatibility
        from scripts.fix.simple_data_fix import create_stable_dataset, test_stable_data_quality
        print("✅ Backward compatibility functions importable")
        
        # Test CLI interface
        from scripts.utils.data_stability_tools import validate_dataset, stabilize_dataset
        print("✅ CLI interface functions available")
        
        print("✅ Data stabilization integration test PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Data stabilization integration test FAILED: {e}\n")
        return False


def test_diagnostics_integration():
    """Test diagnostics integration."""
    print("🧪 Testing Diagnostics Integration")
    print("=" * 50)
    
    try:
        from social_xlstm.evaluation.evaluator import DatasetDiagnostics
        
        # Test that the class exists and has the expected methods
        diagnostics = DatasetDiagnostics()
        assert hasattr(diagnostics, 'analyze_h5_dataset')
        assert hasattr(diagnostics, 'analyze_data_splits')
        assert hasattr(diagnostics, 'analyze_model_complexity')
        assert hasattr(diagnostics, 'create_diagnostic_plots')
        assert hasattr(diagnostics, 'comprehensive_diagnosis')
        print("✅ DatasetDiagnostics methods available")
        
        # Test backward compatibility
        from scripts.debug.overfitting_diagnosis import analyze_h5_dataset, analyze_data_splits
        print("✅ Backward compatibility functions importable")
        
        # Test CLI interface
        from scripts.utils.diagnostic_tools import run_dataset_analysis, run_comprehensive_diagnosis
        print("✅ CLI interface functions available")
        
        print("✅ Diagnostics integration test PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Diagnostics integration test FAILED: {e}\n")
        return False


def test_config_generation_integration():
    """Test configuration generation integration."""
    print("🧪 Testing Configuration Generation Integration")
    print("=" * 50)
    
    try:
        from social_xlstm.dataset.core.processor import TrafficConfigGenerator
        
        # Test that the class exists and has the expected methods
        assert hasattr(TrafficConfigGenerator, 'create_optimized_configs')
        assert hasattr(TrafficConfigGenerator, 'create_development_configs')
        assert hasattr(TrafficConfigGenerator, 'create_production_configs')
        print("✅ TrafficConfigGenerator methods available")
        
        # Test backward compatibility
        from scripts.fix.simple_data_fix import create_fixed_configs
        print("✅ Backward compatibility functions importable")
        
        # Test CLI interface
        from scripts.utils.config_generator import create_optimized_configs, create_development_configs
        print("✅ CLI interface functions available")
        
        # Test actual configuration generation (with temporary directory)
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_h5_path = "/tmp/test_dataset.h5"  # Non-existent file is OK for config generation
            
            # This should work even if the H5 file doesn't exist since we're only generating configs
            configs = TrafficConfigGenerator.create_optimized_configs(test_h5_path, tmp_dir)
            
            # Check that config files were created
            assert len(configs) == 2  # LSTM and xLSTM configs
            for config_path in configs:
                assert Path(config_path).exists()
                print(f"✅ Configuration file created: {Path(config_path).name}")
        
        print("✅ Configuration generation integration test PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Configuration generation integration test FAILED: {e}\n")
        return False


def test_deprecation_warnings():
    """Test that deprecation warnings are properly issued."""
    print("🧪 Testing Deprecation Warnings")
    print("=" * 50)
    
    try:
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test data stabilization deprecation
            from scripts.fix.simple_data_fix import create_stable_dataset
            
            # This should generate a deprecation warning
            try:
                create_stable_dataset("/tmp/test.h5", "/tmp/test_out.h5")
            except:
                pass  # Expected to fail due to missing file, but warning should be issued
            
            # Check that warning was issued
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0
            print("✅ Deprecation warning issued for create_stable_dataset")
            
        print("✅ Deprecation warnings test PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Deprecation warnings test FAILED: {e}\n")
        return False


def main():
    """Run all integration tests."""
    print("🚀 INTEGRATION TEST SUITE")
    print("=" * 60)
    print("Testing the integrated functionality from file reorganization")
    print("=" * 60)
    
    tests = [
        test_data_stabilization,
        test_diagnostics_integration,
        test_config_generation_integration,
        test_deprecation_warnings
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"✅ Tests passed: {passed}")
    print(f"❌ Tests failed: {failed}")
    print(f"📈 Success rate: {passed / (passed + failed) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ File reorganization integration is working correctly")
        return 0
    else:
        print(f"\n⚠️  {failed} INTEGRATION TESTS FAILED!")
        print("❌ Some aspects of the integration need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())