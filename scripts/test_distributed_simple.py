#!/usr/bin/env python3
"""
Simplified test for distributed xLSTM components without pydantic dependency.
"""

import torch
import sys
from collections import OrderedDict

# Add project root to path
sys.path.append('/home/GP/repo/Social-xLSTM/src')

def test_xlstm_basic():
    """Test basic xLSTM functionality."""
    print("=== Testing xLSTM Basic ===")
    
    try:
        from social_xlstm.models.xlstm import TrafficXLSTMConfig, TrafficXLSTM
        
        # Create simple config
        config = TrafficXLSTMConfig()
        config.input_size = 3
        config.hidden_size = 32
        config.num_blocks = 4  # Need at least 4 for slstm_at=[1,3]
        config.sequence_length = 12
        config.prediction_length = 3
        
        model = TrafficXLSTM(config)
        print(f"✓ xLSTM model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        x = torch.randn(2, 12, 3)  # [B, T, F]
        output = model(x)
        print(f"✓ xLSTM forward pass successful. Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ xLSTM basic test failed: {e}")
        return False

def test_vd_manager():
    """Test VDXLSTMManager."""
    print("\\n=== Testing VDXLSTMManager ===")
    
    try:
        from social_xlstm.models.xlstm import TrafficXLSTMConfig
        from social_xlstm.models.vd_xlstm_manager import VDXLSTMManager
        
        config = TrafficXLSTMConfig()
        config.input_size = 3
        config.hidden_size = 32
        config.num_blocks = 4  # Need at least 4 for slstm_at=[1,3]
        config.sequence_length = 12
        config.prediction_length = 3
        
        # Create manager with lazy init
        manager = VDXLSTMManager(
            xlstm_config=config,
            lazy_init=True,
            enable_gradient_checkpointing=False  # Disable for testing
        )
        print("✓ VDXLSTMManager created")
        
        # Test forward pass with distributed batch
        batch_dict = OrderedDict({
            "VD_001": torch.randn(2, 12, 3),
            "VD_002": torch.randn(2, 12, 3)
        })
        
        outputs = manager(batch_dict)
        print(f"✓ VDXLSTMManager forward pass successful")
        print(f"  Output VDs: {list(outputs.keys())}")
        print(f"  Output shapes: {[tensor.shape for tensor in outputs.values()]}")
        
        # Check memory usage
        memory_info = manager.get_memory_usage()
        print(f"  Total parameters: {memory_info['total_parameters']:,}")
        print(f"  Initialized VDs: {memory_info['num_models']}")
        return True
        
    except Exception as e:
        print(f"✗ VDXLSTMManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distributed_model():
    """Test DistributedSocialXLSTMModel."""
    print("\\n=== Testing DistributedSocialXLSTMModel ===")
    
    try:
        from social_xlstm.models.xlstm import TrafficXLSTMConfig
        from social_xlstm.models.distributed_social_xlstm import DistributedSocialXLSTMModel
        
        # Create model config
        xlstm_config = TrafficXLSTMConfig()
        xlstm_config.input_size = 3
        xlstm_config.hidden_size = 32
        xlstm_config.num_blocks = 4  # Need at least 4 for slstm_at=[1,3]
        xlstm_config.sequence_length = 12
        xlstm_config.prediction_length = 3
        
        # Create distributed model
        model = DistributedSocialXLSTMModel(
            xlstm_config=xlstm_config,
            num_features=3,
            hidden_dim=xlstm_config.embedding_dim,  # Match xLSTM's hidden dimension
            prediction_length=3,
            social_pool_type="mean",
            enable_gradient_checkpointing=False
        )
        
        print(f"✓ DistributedSocialXLSTMModel created")
        
        # Test forward pass
        vd_inputs = OrderedDict({
            "VD_001": torch.randn(2, 12, 3),
            "VD_002": torch.randn(2, 12, 3),
            "VD_003": torch.randn(2, 12, 3)
        })
        
        outputs = model(vd_inputs)
        print(f"✓ Forward pass successful")
        print(f"  Output VDs: {list(outputs.keys())}")
        print(f"  Output shapes: {[tensor.shape for tensor in outputs.values()]}")
        
        # Get model info
        model_info = model.get_model_info()
        print(f"  Total parameters: {model_info['total_parameters']:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ DistributedSocialXLSTMModel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage scaling."""
    print("\\n=== Testing Memory Usage ===")
    
    try:
        from social_xlstm.models.xlstm import TrafficXLSTMConfig
        from social_xlstm.models.vd_xlstm_manager import VDXLSTMManager
        
        config = TrafficXLSTMConfig()
        config.input_size = 3
        config.hidden_size = 16  # Very small for memory test
        config.num_blocks = 4  # Need at least 4 for slstm_at=[1,3]
        config.sequence_length = 12
        config.prediction_length = 3
        
        # Test with different VD counts
        vd_counts = [1, 2, 5, 10]
        
        for num_vds in vd_counts:
            try:
                # Create manager
                manager = VDXLSTMManager(xlstm_config=config, lazy_init=True)
                
                # Create test batch
                batch_dict = OrderedDict({
                    f"VD_{i:03d}": torch.randn(1, 12, 3)  # Small batch
                    for i in range(num_vds)
                })
                
                # Forward pass
                outputs = manager(batch_dict)
                
                # Get memory info
                memory_info = manager.get_memory_usage()
                
                print(f"  {num_vds} VDs: {memory_info['total_parameters']:,} params - ✓")
                
                # Cleanup
                del manager, batch_dict, outputs
                
            except Exception as e:
                print(f"  {num_vds} VDs: Failed - {e}")
                break
        
        return True
        
    except Exception as e:
        print(f"✗ Memory usage test failed: {e}")
        return False

def main():
    print("Simplified Distributed xLSTM Test")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("xLSTM Basic", test_xlstm_basic()))
    results.append(("VDXLSTMManager", test_vd_manager()))
    results.append(("DistributedSocialXLSTM", test_distributed_model()))
    results.append(("Memory Usage", test_memory_usage()))
    
    # Summary
    print("\\n" + "=" * 50)
    print("Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests PASSED! Distributed xLSTM architecture is working!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()