#!/usr/bin/env python3
"""
Test script to investigate actual xLSTM output dimensions and behavior.
Understanding whether we have sLSTM vector states vs mLSTM matrix states.
"""

import sys
import torch
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from social_xlstm.models.xlstm import TrafficXLSTM, TrafficXLSTMConfig


def investigate_xlstm_dimensions():
    """Test what xLSTM actually returns for different configurations."""
    print("🧪 Investigating xLSTM Output Dimensions...")
    
    # Test configuration with different embedding_dim vs what should be hidden_size
    config = TrafficXLSTMConfig(
        input_size=3,
        embedding_dim=32,      # xLSTM representation dimension (output dimension)
        num_blocks=2,
        output_size=3,
        sequence_length=12,
        prediction_length=3,
        slstm_at=[0],          # First block is sLSTM
        slstm_backend="vanilla",
        mlstm_backend="chunkwise",
        context_length=64,
        dropout=0.1
    )
    
    print(f"Config: embedding_dim={config.embedding_dim} (xLSTM output dimension)")
    print(f"Blocks: {config.num_blocks}, sLSTM at: {config.slstm_at}")
    
    # Create model
    model = TrafficXLSTM(config)
    
    # Test input
    batch_size, seq_len, input_size = 2, 12, 3
    x = torch.randn(batch_size, seq_len, input_size)
    
    print(f"\nInput shape: {x.shape}")
    
    # Test forward pass
    with torch.no_grad():
        # Get final output
        output = model(x)
        print(f"Forward output shape: {output.shape}")
        
        # Get hidden states
        hidden_states = model.get_hidden_states(x)
        print(f"Hidden states shape: {hidden_states.shape}")
        
        # Let's also check internal components
        embedded = model.input_embedding(x)
        print(f"After input embedding: {embedded.shape}")
        
        xlstm_output = model.xlstm_stack(embedded)
        print(f"After xLSTM stack: {xlstm_output.shape}")
        
        # Check if there's a difference between different time steps
        print(f"Hidden states at t=0: {hidden_states[:, 0, :].shape}")
        print(f"Hidden states at t=-1: {hidden_states[:, -1, :].shape}")


def investigate_xlstm_internals():
    """Try to peek into xLSTM internal structure."""
    print("\n🔍 Investigating xLSTM Internal Structure...")
    
    config = TrafficXLSTMConfig(
        input_size=3, embedding_dim=32, num_blocks=2,
        output_size=3, sequence_length=12, prediction_length=3,
        slstm_at=[0], slstm_backend="vanilla", mlstm_backend="chunkwise",
        context_length=64, dropout=0.1
    )
    
    model = TrafficXLSTM(config)
    
    # Inspect the xLSTM stack
    print("xLSTM Stack Structure:")
    print(f"  Type: {type(model.xlstm_stack)}")
    
    if hasattr(model.xlstm_stack, 'blocks'):
        print(f"  Number of blocks: {len(model.xlstm_stack.blocks)}")
        for i, block in enumerate(model.xlstm_stack.blocks):
            print(f"  Block {i}: {type(block).__name__}")
            
            # Check if we can access layer configs
            if hasattr(block, 'mlstm') and block.mlstm is not None:
                print(f"    mLSTM layer present")
            if hasattr(block, 'slstm') and block.slstm is not None:
                print(f"    sLSTM layer present")
    
    # Check model parameters and their shapes
    print("\nModel Parameter Shapes:")
    total_params = 0
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
        total_params += param.numel()
    print(f"Total parameters: {total_params:,}")


def test_different_configurations():
    """Test with different embedding_dim vs hidden_size to see the difference."""
    print("\n⚖️  Testing Different Configurations...")
    
    configurations = [
        {"embedding_dim": 32, "desc": "small embedding_dim"},
        {"embedding_dim": 64, "desc": "large embedding_dim"},
        {"embedding_dim": 128, "desc": "very large embedding_dim"},
    ]
    
    for config_params in configurations:
        print(f"\n{config_params['desc']}:")
        try:
            config = TrafficXLSTMConfig(
                input_size=3,
                embedding_dim=config_params["embedding_dim"],
                num_blocks=2, output_size=3, sequence_length=12, prediction_length=3,
                slstm_at=[0], slstm_backend="vanilla", mlstm_backend="chunkwise",
                context_length=64, dropout=0.1
            )
            
            model = TrafficXLSTM(config)
            x = torch.randn(2, 12, 3)
            
            with torch.no_grad():
                hidden_states = model.get_hidden_states(x)
                print(f"  Hidden states shape: {hidden_states.shape}")
                print(f"  Expected dimension: {config_params['embedding_dim']} (embedding_dim)")
                print(f"  Actual last dimension: {hidden_states.shape[-1]}")
                
                if hidden_states.shape[-1] == config_params["embedding_dim"]:
                    print("  ✅ Returns embedding_dim (correct)")
                else:
                    print("  ❌ Unexpected dimension")
                    
        except Exception as e:
            print(f"  ❌ Configuration failed: {e}")


def main():
    """Run all investigations."""
    print("🚀 xLSTM Dimension Investigation\n")
    
    investigate_xlstm_dimensions()
    investigate_xlstm_internals()
    test_different_configurations()
    
    print("\n📋 Summary:")
    print("- This test reveals whether xLSTM returns embedding_dim or hidden_size")
    print("- It shows the actual internal structure of sLSTM vs mLSTM blocks")  
    print("- It helps determine if our social pooling should use a different dimension")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)