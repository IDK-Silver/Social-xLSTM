#!/usr/bin/env python3
"""
Test script for the new Spatial-Only Social Pooling configuration system.

This script validates:
1. Configuration loading and validation
2. Model instantiation with spatial pooling enabled/disabled
3. Forward pass functionality
"""

import sys
import torch
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from social_xlstm.models.distributed_config import (
    DistributedSocialXLSTMConfig, 
    SocialPoolingConfig,
    load_distributed_config_from_dict
)
from social_xlstm.models.xlstm import TrafficXLSTMConfig
from social_xlstm.models.distributed_social_xlstm import DistributedSocialXLSTMModel


def test_config_validation():
    """Test configuration validation logic."""
    print("🧪 Testing configuration validation...")
    
    # Test 1: Valid configuration with social pooling enabled
    try:
        social_config = SocialPoolingConfig(
            enabled=True,
            radius=2.0,
            aggregation="weighted_mean", 
            hidden_dim=64
        )
        print("✅ Valid enabled config passed")
    except Exception as e:
        print(f"❌ Valid enabled config failed: {e}")
        return False
    
    # Test 2: Valid configuration with social pooling disabled
    try:
        social_config = SocialPoolingConfig(
            enabled=False,
            radius=2.0,
            aggregation="mean",  # Not validated when disabled
            hidden_dim=64
        )
        print("✅ Valid disabled config passed")
    except Exception as e:
        print(f"❌ Valid disabled config failed: {e}")
        return False
    
    # Test 3: Invalid aggregation method (should fail)
    try:
        social_config = SocialPoolingConfig(
            enabled=True,
            radius=2.0,
            aggregation="invalid_method",  # Invalid
            hidden_dim=64
        )
        print("❌ Invalid aggregation method should have failed")
        return False
    except ValueError as e:
        print(f"✅ Invalid aggregation correctly rejected: {e}")
    
    # Test 4: Invalid radius (should fail)
    try:
        social_config = SocialPoolingConfig(
            enabled=True,
            radius=-1.0,  # Invalid
            aggregation="mean",
            hidden_dim=64
        )
        print("❌ Invalid radius should have failed")
        return False
    except ValueError as e:
        print(f"✅ Invalid radius correctly rejected: {e}")
    
    return True


def test_model_instantiation():
    """Test model instantiation with new configuration system."""
    print("\n🧪 Testing model instantiation...")
    
    # Create minimal xLSTM config
    xlstm_config = TrafficXLSTMConfig(
        input_size=3,
        embedding_dim=32,
        num_blocks=2,
        output_size=3,
        sequence_length=12,
        prediction_length=3,
        slstm_at=[0],
        slstm_backend="vanilla",
        mlstm_backend="chunkwise",
        context_length=64,
        dropout=0.1
    )
    
    # Test 1: Model with social pooling enabled
    try:
        social_config_enabled = SocialPoolingConfig(
            enabled=True,
            radius=2.0,
            aggregation="mean",
            hidden_dim=32
        )
        
        distributed_config = DistributedSocialXLSTMConfig(
            xlstm=xlstm_config,
            num_features=3,
            prediction_length=3,
            learning_rate=0.001,
            enable_gradient_checkpointing=False,
            social=social_config_enabled
        )
        
        model = DistributedSocialXLSTMModel(distributed_config)
        print(f"✅ Model with enabled social pooling created: {model.social_pooling is not None}")
        
        # Check model info
        info = model.get_model_info()
        print(f"   Social pooling enabled: {info['social_pooling_enabled']}")
        if info['social_pooling_config']:
            print(f"   Social config: {info['social_pooling_config']}")
        
    except Exception as e:
        print(f"❌ Model with enabled social pooling failed: {e}")
        return False
    
    # Test 2: Model with social pooling disabled
    try:
        social_config_disabled = SocialPoolingConfig(
            enabled=False,
            radius=2.0,
            aggregation="mean",
            hidden_dim=32
        )
        
        distributed_config = DistributedSocialXLSTMConfig(
            xlstm=xlstm_config,
            num_features=3,
            prediction_length=3,
            learning_rate=0.001,
            enable_gradient_checkpointing=False,
            social=social_config_disabled
        )
        
        model = DistributedSocialXLSTMModel(distributed_config)
        print(f"✅ Model with disabled social pooling created: {model.social_pooling is None}")
        
        # Check model info
        info = model.get_model_info()
        print(f"   Social pooling enabled: {info['social_pooling_enabled']}")
        print(f"   Social config: {info['social_pooling_config']}")
        
    except Exception as e:
        print(f"❌ Model with disabled social pooling failed: {e}")
        return False
    
    return True


def test_config_loading():
    """Test configuration loading from dictionary."""
    print("\n🧪 Testing configuration loading from dict...")
    
    # Test configuration dictionary
    config_dict = {
        "model": {
            "xlstm": {
                "input_size": 3,
                "embedding_dim": 32,
                "num_blocks": 2,
                "output_size": 3,
                "sequence_length": 12,
                "prediction_length": 3,
                "slstm_at": [0],
                "slstm_backend": "vanilla",
                "mlstm_backend": "chunkwise", 
                "context_length": 64,
                "dropout": 0.1
            },
            "distributed_social": {
                "num_features": 3,
                "prediction_length": 3,
                "learning_rate": 0.001,
                "enable_gradient_checkpointing": False,
                "social": {
                    "enabled": True,
                    "radius": 2.5,
                    "aggregation": "max",
                    "hidden_dim": 32
                }
            }
        }
    }
    
    try:
        config = load_distributed_config_from_dict(config_dict)
        print("✅ Configuration loaded from dictionary")
        print(f"   Social enabled: {config.social.enabled}")
        print(f"   Social radius: {config.social.radius}")
        print(f"   Social aggregation: {config.social.aggregation}")
        print(f"   Social hidden_dim: {config.social.hidden_dim}")
        
        # Test model creation
        model = DistributedSocialXLSTMModel(config)
        print("✅ Model created from loaded configuration")
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False
    
    return True


def test_forward_pass():
    """Test forward pass with both enabled and disabled social pooling."""
    print("\n🧪 Testing forward pass...")
    
    # Create test data
    batch_size, num_vds, seq_len, num_features = 2, 3, 12, 3
    
    # Mock VD inputs
    vd_inputs = {
        f"VD-{i:03d}": torch.randn(batch_size, seq_len, num_features)
        for i in range(num_vds)
    }
    
    # Mock positions for spatial pooling
    positions = {
        f"VD-{i:03d}": torch.randn(batch_size, seq_len, 2) * 10  # Random positions
        for i in range(num_vds)
    }
    
    # Test configuration
    xlstm_config = TrafficXLSTMConfig(
        input_size=3, embedding_dim=32, num_blocks=2,
        output_size=3, sequence_length=12, prediction_length=3,
        slstm_at=[0], slstm_backend="vanilla", mlstm_backend="chunkwise",
        context_length=64, dropout=0.1
    )
    
    # Test 1: Forward pass with enabled social pooling
    try:
        social_config = SocialPoolingConfig(
            enabled=True, radius=5.0, aggregation="weighted_mean", hidden_dim=32
        )
        config = DistributedSocialXLSTMConfig(
            xlstm=xlstm_config, num_features=3, prediction_length=3,
            learning_rate=0.001, enable_gradient_checkpointing=False, social=social_config
        )
        
        model = DistributedSocialXLSTMModel(config)
        
        # Forward pass with positions
        try:
            outputs = model(vd_inputs, positions=positions)
            print(f"✅ Forward pass with enabled social pooling: {len(outputs)} VDs")
        except Exception as e:
            print(f"❌ Forward pass error details: {e}")
            # Debug model architecture
            print(f"   Model fusion input dim: {model.fusion_layer[0].in_features}")
            print(f"   Model fusion output dim: {model.fusion_layer[0].out_features}")
            print(f"   Model has social_projection: {model.social_projection is not None}")
            
            # Debug forward pass dimensions
            print("   Debugging forward pass dimensions...")
            vd_ids = list(vd_inputs.keys())
            individual_hidden_states = model.vd_manager(vd_inputs)
            print(f"   Individual hidden states shape: {[individual_hidden_states[vd_id].shape for vd_id in vd_ids[:1]]}")
            
            if model.social_pooling is not None:
                social_contexts = model.social_pooling(
                    agent_hidden_states=individual_hidden_states,
                    agent_positions=positions,
                    target_agent_ids=vd_ids
                )
                print(f"   Social contexts shape: {[social_contexts[vd_id].shape for vd_id in vd_ids[:1]]}")
                
                # Test concatenation
                individual_hidden = individual_hidden_states[vd_ids[0]][:, -1, :]
                social_context = social_contexts[vd_ids[0]]
                print(f"   Individual hidden (last): {individual_hidden.shape}")
                print(f"   Social context: {social_context.shape}")
                fused = torch.cat([individual_hidden, social_context], dim=-1)
                print(f"   Fused features: {fused.shape}")
            
            raise
        
        # Check output shapes
        for vd_id, output in outputs.items():
            expected_shape = (batch_size, 3 * 3)  # prediction_length * num_features
            if output.shape == expected_shape:
                print(f"   ✅ {vd_id} output shape: {output.shape}")
            else:
                print(f"   ❌ {vd_id} wrong shape: {output.shape}, expected: {expected_shape}")
                return False
        
    except Exception as e:
        print(f"❌ Forward pass with enabled social pooling failed: {e}")
        return False
    
    # Test 2: Forward pass with disabled social pooling
    try:
        social_config = SocialPoolingConfig(
            enabled=False, radius=5.0, aggregation="mean", hidden_dim=32
        )
        config = DistributedSocialXLSTMConfig(
            xlstm=xlstm_config, num_features=3, prediction_length=3,
            learning_rate=0.001, enable_gradient_checkpointing=False, social=social_config
        )
        
        model = DistributedSocialXLSTMModel(config)
        
        # Forward pass without positions (should work with disabled social pooling)
        outputs = model(vd_inputs, positions=None)
        print(f"✅ Forward pass with disabled social pooling: {len(outputs)} VDs")
        
        # Check output shapes
        for vd_id, output in outputs.items():
            expected_shape = (batch_size, 3 * 3)  # prediction_length * num_features
            if output.shape == expected_shape:
                print(f"   ✅ {vd_id} output shape: {output.shape}")
            else:
                print(f"   ❌ {vd_id} wrong shape: {output.shape}, expected: {expected_shape}")
                return False
        
    except Exception as e:
        print(f"❌ Forward pass with disabled social pooling failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("🚀 Testing Spatial-Only Social Pooling Configuration System\n")
    
    tests = [
        ("Configuration Validation", test_config_validation),
        ("Model Instantiation", test_model_instantiation), 
        ("Configuration Loading", test_config_loading),
        ("Forward Pass", test_forward_pass)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 All tests passed! Spatial-Only configuration system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)