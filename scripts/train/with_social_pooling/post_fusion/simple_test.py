#!/usr/bin/env python3
"""
Simple Post-Fusion Integration Test

This is a minimal test to verify that the Post-Fusion Social Pooling
components can be imported and basic functionality works.

Usage:
    python simple_test.py

Author: Social-xLSTM Project Team
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test Social Pooling core imports
        from social_xlstm.models.social_pooling_config import SocialPoolingConfig, create_social_pooling_config
        print("✅ SocialPoolingConfig imported successfully")
        
        from social_xlstm.models.social_pooling import SocialPooling
        print("✅ SocialPooling imported successfully")
        
        from social_xlstm.models.social_traffic_model import SocialTrafficModel, create_social_traffic_model
        print("✅ SocialTrafficModel imported successfully")
        
        # Test base model imports
        from social_xlstm.models.lstm import TrafficLSTM
        print("✅ TrafficLSTM imported successfully")
        
        from social_xlstm.models.xlstm import TrafficXLSTM, TrafficXLSTMConfig
        print("✅ TrafficXLSTM imported successfully")
        
        # Test trainer imports
        from social_xlstm.training.without_social_pooling.single_vd_trainer import SingleVDTrainer, SingleVDTrainingConfig
        print("✅ SingleVDTrainer imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_social_pooling_config():
    """Test Social Pooling configuration creation."""
    print("\nTesting Social Pooling configuration...")
    
    try:
        # Test default configuration
        config = SocialPoolingConfig()
        print(f"✅ Default config created: {config}")
        
        # Test preset configurations
        urban_config = SocialPoolingConfig.urban_preset()
        highway_config = SocialPoolingConfig.highway_preset()
        mixed_config = SocialPoolingConfig.mixed_preset()
        
        print(f"✅ Urban preset: radius={urban_config.pooling_radius}m, neighbors={urban_config.max_neighbors}")
        print(f"✅ Highway preset: radius={highway_config.pooling_radius}m, neighbors={highway_config.max_neighbors}")
        print(f"✅ Mixed preset: radius={mixed_config.pooling_radius}m, neighbors={mixed_config.max_neighbors}")
        
        # Test factory function
        factory_config = create_social_pooling_config("urban", pooling_radius=800.0)
        print(f"✅ Factory config: radius={factory_config.pooling_radius}m")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_coordinate_loading():
    """Test coordinate data loading."""
    print("\nTesting coordinate data loading...")
    
    try:
        import json
        
        # Create test coordinate data
        test_coords = {
            "VD-TEST-01": [121.5654, 25.0330],
            "VD-TEST-02": [121.5643, 25.0315],
            "VD-TEST-03": [121.5632, 25.0345]
        }
        
        # Test coordinate validation
        for vd_id, coord in test_coords.items():
            assert isinstance(coord, list) and len(coord) == 2
            assert all(isinstance(x, (int, float)) for x in coord)
        
        print(f"✅ Coordinate validation passed for {len(test_coords)} VDs")
        
        # Test coordinate format conversion
        coordinates = {}
        for vd_id, coords in test_coords.items():
            x, y = float(coords[0]), float(coords[1])
            coordinates[str(vd_id)] = (x, y)
        
        print(f"✅ Coordinate format conversion successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Coordinate loading test failed: {e}")
        return False


def test_model_creation():
    """Test basic model creation without data dependencies."""
    print("\nTesting model creation...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Test TrafficLSTM creation
        lstm_model = TrafficLSTM.create_single_vd_model(
            input_size=3,
            output_size=3,
            hidden_size=64,
            num_layers=2,
            dropout=0.1
        )
        print(f"✅ TrafficLSTM created: {lstm_model.get_model_info()['total_parameters']} parameters")
        
        # Test Social Pooling configuration
        social_config = SocialPoolingConfig.urban_preset(pooling_radius=500.0)
        print(f"✅ Social config created: {social_config}")
        
        # Test SocialTrafficModel creation
        social_model = create_social_traffic_model(
            base_model=lstm_model,
            social_config=social_config,
            model_type="post_fusion_lstm",
            scenario="urban"
        )
        
        model_info = social_model.get_model_info()
        print(f"✅ SocialTrafficModel created: {model_info['total_parameters']} parameters")
        print(f"   Base model: {model_info['base_model_type']}")
        print(f"   Fusion strategy: {model_info['fusion_strategy']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test basic forward pass without real data."""
    print("\nTesting forward pass...")
    
    try:
        import torch
        
        # Create simple models
        lstm_model = TrafficLSTM.create_single_vd_model(3, 3, 32, 1, 0.0)
        social_config = SocialPoolingConfig(pooling_radius=500.0, max_neighbors=3)
        social_model = create_social_traffic_model(lstm_model, social_config, "test", "urban")
        
        # Create mock input
        batch_size, seq_len, features = 2, 5, 3
        input_seq = torch.randn(batch_size, seq_len, features)
        coordinates = torch.tensor([[0.0, 0.0], [100.0, 100.0]], dtype=torch.float32)
        vd_ids = ["VD-01", "VD-02"]
        
        # Test forward pass
        with torch.no_grad():
            output = social_model(input_seq, coordinates, vd_ids)
        
        expected_shape = (batch_size, 1, features)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert torch.isfinite(output).all(), "Output contains infinite values"
        
        print(f"✅ Forward pass successful: input {input_seq.shape} → output {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Post-Fusion Social Pooling - Simple Integration Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_social_pooling_config),
        ("Coordinate Loading Test", test_coordinate_loading),
        ("Model Creation Test", test_model_creation),
        ("Forward Pass Test", test_forward_pass)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running {test_name}...")
        print(f"{'-' * 40}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"{'=' * 60}")
    
    if passed == total:
        print("🎉 All tests passed! Post-Fusion integration is working correctly.")
        print("Ready for full training with real data.")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the errors above.")
        print("Fix issues before proceeding with training.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)