#!/usr/bin/env python3
"""
Test script for Task 2.2: Social Pooling integration.

Tests the integrated spatial social pooling functionality in DistributedSocialXLSTMModel.
"""

import torch
import sys
from collections import OrderedDict

# Add project root to path
sys.path.append('/home/GP/repo/Social-xLSTM/src')

def test_legacy_mode():
    """Test that legacy mode (no spatial pooling) still works."""
    print("=== Testing Legacy Mode (enable_spatial_pooling=False) ===")
    
    try:
        from social_xlstm.models.xlstm import TrafficXLSTMConfig
        from social_xlstm.models.distributed_social_xlstm import DistributedSocialXLSTMModel
        
        # Create config
        xlstm_config = TrafficXLSTMConfig()
        xlstm_config.input_size = 3
        xlstm_config.hidden_size = 32
        xlstm_config.num_blocks = 4
        xlstm_config.sequence_length = 12
        xlstm_config.prediction_length = 3
        
        # Create model without spatial pooling (legacy mode)
        model = DistributedSocialXLSTMModel(
            xlstm_config=xlstm_config,
            num_features=3,
            hidden_dim=xlstm_config.embedding_dim,
            prediction_length=3,
            social_pool_type="mean",
            enable_gradient_checkpointing=False,
            enable_spatial_pooling=False  # Legacy mode
        )
        
        print(f"✓ Legacy model created with spatial_pooling={model.enable_spatial_pooling}")
        
        # Test forward pass without positions
        vd_inputs = OrderedDict({
            "VD_001": torch.randn(2, 12, 3),
            "VD_002": torch.randn(2, 12, 3),
            "VD_003": torch.randn(2, 12, 3)
        })
        
        outputs = model(vd_inputs)  # No positions provided
        
        print(f"✓ Legacy forward pass successful")
        print(f"  Output VDs: {list(outputs.keys())}")
        print(f"  Output shapes: {[tensor.shape for tensor in outputs.values()]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Legacy mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spatial_mode():
    """Test spatial pooling mode with position data."""
    print("\\n=== Testing Spatial Mode (enable_spatial_pooling=True) ===")
    
    try:
        from social_xlstm.models.xlstm import TrafficXLSTMConfig
        from social_xlstm.models.distributed_social_xlstm import DistributedSocialXLSTMModel
        from social_xlstm.pooling.xlstm_pooling import create_mock_positions
        
        # Create config
        xlstm_config = TrafficXLSTMConfig()
        xlstm_config.input_size = 3
        xlstm_config.hidden_size = 32
        xlstm_config.num_blocks = 4
        xlstm_config.sequence_length = 12
        xlstm_config.prediction_length = 3
        
        # Create model with spatial pooling
        model = DistributedSocialXLSTMModel(
            xlstm_config=xlstm_config,
            num_features=3,
            hidden_dim=xlstm_config.embedding_dim,
            prediction_length=3,
            social_pool_type="mean",
            enable_gradient_checkpointing=False,
            enable_spatial_pooling=True,  # Spatial mode
            spatial_radius=2.0
        )
        
        print(f"✓ Spatial model created with spatial_pooling={model.enable_spatial_pooling}")
        print(f"  Spatial radius: {model.spatial_radius}")
        print(f"  Social pooling type: {type(model.social_pooling).__name__}")
        
        # Create test data
        vd_ids = ["VD_001", "VD_002", "VD_003"]
        batch_size = 2
        seq_len = 12
        
        vd_inputs = OrderedDict({
            vd_id: torch.randn(batch_size, seq_len, 3) for vd_id in vd_ids
        })
        
        # Create mock position data
        positions = create_mock_positions(
            vd_ids=vd_ids,
            batch_size=batch_size,
            seq_len=seq_len,
            spatial_range=5.0,
            device=torch.device('cpu')
        )
        
        print(f"✓ Mock positions created")
        print(f"  Position keys: {list(positions.keys())}")
        print(f"  Position shapes: {[pos.shape for pos in positions.values()]}")
        
        # Test forward pass with positions
        outputs = model(vd_inputs, positions=positions)
        
        print(f"✓ Spatial forward pass successful")
        print(f"  Output VDs: {list(outputs.keys())}")
        print(f"  Output shapes: {[tensor.shape for tensor in outputs.values()]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Spatial mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spatial_mode_without_positions():
    """Test spatial mode when positions are not provided (should fallback to legacy)."""
    print("\\n=== Testing Spatial Mode Fallback (no positions provided) ===")
    
    try:
        from social_xlstm.models.xlstm import TrafficXLSTMConfig
        from social_xlstm.models.distributed_social_xlstm import DistributedSocialXLSTMModel
        
        # Create config
        xlstm_config = TrafficXLSTMConfig()
        xlstm_config.input_size = 3
        xlstm_config.hidden_size = 32
        xlstm_config.num_blocks = 4
        xlstm_config.sequence_length = 12
        xlstm_config.prediction_length = 3
        
        # Create model with spatial pooling enabled
        model = DistributedSocialXLSTMModel(
            xlstm_config=xlstm_config,
            num_features=3,
            hidden_dim=xlstm_config.embedding_dim,
            prediction_length=3,
            social_pool_type="mean",
            enable_gradient_checkpointing=False,
            enable_spatial_pooling=True,  # Spatial enabled but no positions provided
            spatial_radius=2.0
        )
        
        print(f"✓ Spatial model created (will fallback to legacy without positions)")
        
        # Test forward pass without positions (should fallback to legacy)
        vd_inputs = OrderedDict({
            "VD_001": torch.randn(2, 12, 3),
            "VD_002": torch.randn(2, 12, 3),
            "VD_003": torch.randn(2, 12, 3)
        })
        
        outputs = model(vd_inputs)  # No positions = fallback to legacy
        
        print(f"✓ Fallback to legacy pooling successful")
        print(f"  Output VDs: {list(outputs.keys())}")
        print(f"  Output shapes: {[tensor.shape for tensor in outputs.values()]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Spatial fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_info():
    """Test model information reporting."""
    print("\\n=== Testing Model Information ===")
    
    try:
        from social_xlstm.models.xlstm import TrafficXLSTMConfig
        from social_xlstm.models.distributed_social_xlstm import DistributedSocialXLSTMModel
        
        # Create config
        xlstm_config = TrafficXLSTMConfig()
        xlstm_config.input_size = 3
        xlstm_config.hidden_size = 32
        xlstm_config.num_blocks = 4
        xlstm_config.sequence_length = 12
        xlstm_config.prediction_length = 3
        
        # Test both modes
        for enable_spatial in [False, True]:
            print(f"\\n--- Testing model info (spatial={enable_spatial}) ---")
            
            model = DistributedSocialXLSTMModel(
                xlstm_config=xlstm_config,
                num_features=3,
                hidden_dim=xlstm_config.embedding_dim,
                prediction_length=3,
                social_pool_type="weighted_mean",
                enable_gradient_checkpointing=False,
                enable_spatial_pooling=enable_spatial,
                spatial_radius=1.5
            )
            
            # Get model info
            info = model.get_model_info()
            
            print(f"  Total parameters: {info['total_parameters']:,}")
            print(f"  Trainable parameters: {info['trainable_parameters']:,}")
            print(f"  Hidden dim: {info['hidden_dim']}")
            print(f"  Prediction length: {info['prediction_length']}")
            
            # Test social pooling layer info
            if hasattr(model.social_pooling, 'get_info'):
                pooling_info = model.social_pooling.get_info()
                print(f"  Social pooling: {pooling_info}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model info test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Task 2.2 Integration Test: Social Pooling Integration")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Legacy Mode", test_legacy_mode()))
    results.append(("Spatial Mode", test_spatial_mode()))
    results.append(("Spatial Fallback", test_spatial_mode_without_positions()))
    results.append(("Model Info", test_model_info()))
    
    # Summary
    print("\\n" + "=" * 60)
    print("Integration Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All integration tests PASSED! Task 2.2 successfully integrated!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()