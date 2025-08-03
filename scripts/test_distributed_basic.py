#!/usr/bin/env python3
"""
Basic test for distributed xLSTM components.
Simple verification that the core components work together.
"""

import torch
import sys
import os
from collections import OrderedDict

# Add project root to path
sys.path.append('/home/GP/repo/Social-xLSTM/src')

# Basic imports to test
try:
    from social_xlstm.interfaces.tensor_spec import TensorSpec, validate_distributed_batch
    print("✓ TensorSpec import successful")
except Exception as e:
    print(f"✗ TensorSpec import failed: {e}")
    sys.exit(1)

try:
    from social_xlstm.models.xlstm import TrafficXLSTMConfig, TrafficXLSTM
    print("✓ xLSTM import successful")
except Exception as e:
    print(f"✗ xLSTM import failed: {e}")

try:
    from social_xlstm.models.vd_xlstm_manager import VDXLSTMManager
    print("✓ VDXLSTMManager import successful")
except Exception as e:
    print(f"✗ VDXLSTMManager import failed: {e}")

def test_tensor_spec():
    """Test tensor specification validation."""
    print("\\n=== Testing TensorSpec ===")
    
    # Create tensor spec
    spec = TensorSpec(batch_size=2, time_steps=12, num_vds=3, feature_dim=5)
    
    # Test centralized tensor validation
    centralized_tensor = torch.randn(2, 12, 3, 5)  # [B, T, N, F]
    try:
        spec.validate_centralized_tensor(centralized_tensor)
        print("✓ Centralized tensor validation passed")
    except Exception as e:
        print(f"✗ Centralized tensor validation failed: {e}")
    
    # Test distributed batch validation
    distributed_batch = OrderedDict({
        "VD_001": torch.randn(2, 12, 5),  # [B, T, F]
        "VD_002": torch.randn(2, 12, 5),
        "VD_003": torch.randn(2, 12, 5)
    })
    
    try:
        spec.validate_distributed_batch(distributed_batch, ["VD_001", "VD_002", "VD_003"])
        print("✓ Distributed batch validation passed")
    except Exception as e:
        print(f"✗ Distributed batch validation failed: {e}")

def test_xlstm_basic():
    """Test basic xLSTM functionality."""
    print("\\n=== Testing xLSTM Basic ===")
    
    # Create simple config
    config = TrafficXLSTMConfig(
        input_size=3,
        hidden_size=32,
        num_layers=2,
        sequence_length=12,
        prediction_length=3
    )
    
    try:
        model = TrafficXLSTM(config)
        print(f"✓ xLSTM model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        x = torch.randn(2, 12, 3)  # [B, T, F]
        output = model(x)
        print(f"✓ xLSTM forward pass successful. Output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ xLSTM basic test failed: {e}")

def test_vd_manager():
    """Test VDXLSTMManager."""
    print("\\n=== Testing VDXLSTMManager ===")
    
    config = TrafficXLSTMConfig(
        input_size=3,
        hidden_size=32,
        num_layers=2,
        sequence_length=12,
        prediction_length=3
    )
    
    try:
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
        
    except Exception as e:
        print(f"✗ VDXLSTMManager test failed: {e}")

def test_distributed_workflow():
    """Test complete distributed workflow."""
    print("\\n=== Testing Complete Workflow ===")
    
    try:
        # 1. Create synthetic centralized data [B, T, N, F]
        batch_size, seq_len, num_vds, num_features = 2, 12, 3, 3
        centralized_data = torch.randn(batch_size, seq_len, num_vds, num_features)
        
        # 2. Transform to distributed format
        vd_ids = ["VD_001", "VD_002", "VD_003"]
        distributed_batch = OrderedDict()
        
        for i, vd_id in enumerate(vd_ids):
            distributed_batch[vd_id] = centralized_data[:, :, i, :]  # [B, T, F]
        
        print(f"✓ Data transformation: [B,T,N,F] -> per-VD format")
        print(f"  Centralized shape: {centralized_data.shape}")
        print(f"  Distributed VDs: {len(distributed_batch)}")
        
        # 3. Validate with TensorSpec
        spec = TensorSpec(
            batch_size=batch_size,
            time_steps=seq_len,
            num_vds=num_vds,
            feature_dim=num_features
        )
        
        spec.validate_centralized_tensor(centralized_data)
        spec.validate_distributed_batch(distributed_batch, vd_ids)
        print("✓ Tensor specification validation passed")
        
        # 4. Process through VDXLSTMManager
        config = TrafficXLSTMConfig(
            input_size=num_features,
            hidden_size=16,  # Small for testing
            num_layers=1,
            sequence_length=seq_len,
            prediction_length=3
        )
        
        manager = VDXLSTMManager(xlstm_config=config, lazy_init=True)
        hidden_states = manager(distributed_batch)
        
        print("✓ VDXLSTMManager processing successful")
        print(f"  Hidden state shapes: {[h.shape for h in hidden_states.values()]}")
        
        # 5. Validate hidden states format
        spec.validate_hidden_states({
            vd_id: hidden[:, -1, :].unsqueeze(0).transpose(0, 1)  # [T=1, H]
            for vd_id, hidden in hidden_states.items()
        })
        print("✓ Hidden states validation passed")
        
        print("\\n🎉 Complete distributed workflow test PASSED!")
        
    except Exception as e:
        print(f"✗ Complete workflow test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Distributed xLSTM Architecture Test")
    print("=" * 50)
    
    # Run all tests
    test_tensor_spec()
    test_xlstm_basic()
    test_vd_manager()
    test_distributed_workflow()
    
    print("\\n" + "=" * 50)
    print("Test Summary Complete")

if __name__ == "__main__":
    main()