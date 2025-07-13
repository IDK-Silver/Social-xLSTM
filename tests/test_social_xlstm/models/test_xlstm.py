"""
Unit tests for TrafficXLSTM model.

Test coverage includes:
- Model initialization
- Configuration validation
- Forward pass functionality
- Model information retrieval
- Device management
"""

import pytest
import torch
import numpy as np
from social_xlstm.models import TrafficXLSTM, TrafficXLSTMConfig


class TestTrafficXLSTMConfig:
    """Test TrafficXLSTMConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrafficXLSTMConfig()
        
        assert config.input_size == 3
        assert config.embedding_dim == 128
        assert config.num_blocks == 6
        assert config.slstm_at == [1, 3]
        assert config.slstm_backend == "vanilla"
        assert config.mlstm_backend == "vanilla"
        assert config.context_length == 256
        assert config.dropout == 0.1
        assert config.multi_vd_mode is False
        assert config.num_vds is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrafficXLSTMConfig(
            embedding_dim=64,
            num_blocks=4,
            slstm_at=[0, 2],
            dropout=0.2,
            multi_vd_mode=True,
            num_vds=5
        )
        
        assert config.embedding_dim == 64
        assert config.num_blocks == 4
        assert config.slstm_at == [0, 2]
        assert config.dropout == 0.2
        assert config.multi_vd_mode is True
        assert config.num_vds == 5
    
    def test_multi_vd_validation(self):
        """Test validation for multi-VD mode."""
        # Should raise error when multi_vd_mode=True but num_vds=None
        with pytest.raises(ValueError, match="num_vds must be specified"):
            TrafficXLSTMConfig(multi_vd_mode=True, num_vds=None)
    
    def test_slstm_position_validation(self):
        """Test validation for sLSTM positions."""
        # Should raise error when sLSTM positions exceed num_blocks
        with pytest.raises(ValueError, match="sLSTM positions .* exceed num_blocks"):
            TrafficXLSTMConfig(num_blocks=4, slstm_at=[0, 1, 5])


class TestTrafficXLSTM:
    """Test TrafficXLSTM model."""
    
    def test_model_initialization(self):
        """Test basic model initialization."""
        config = TrafficXLSTMConfig()
        model = TrafficXLSTM(config)
        
        assert isinstance(model, TrafficXLSTM)
        assert hasattr(model, 'input_embedding')
        assert hasattr(model, 'xlstm_stack')
        assert hasattr(model, 'output_projection')
        assert hasattr(model, 'dropout')
    
    def test_forward_pass_single_batch(self):
        """Test forward pass with single batch."""
        config = TrafficXLSTMConfig()
        model = TrafficXLSTM(config)
        
        batch_size, seq_len, input_size = 1, 12, 3
        x = torch.randn(batch_size, seq_len, input_size)
        
        output = model(x)
        
        assert output.shape == (batch_size, 1, 3)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_forward_pass_multiple_batches(self):
        """Test forward pass with multiple batches."""
        config = TrafficXLSTMConfig()
        model = TrafficXLSTM(config)
        
        batch_size, seq_len, input_size = 4, 12, 3
        x = torch.randn(batch_size, seq_len, input_size)
        
        output = model(x)
        
        assert output.shape == (batch_size, 1, 3)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        config = TrafficXLSTMConfig()
        model = TrafficXLSTM(config)
        
        for seq_len in [6, 12, 24, 48]:
            batch_size, input_size = 2, 3
            x = torch.randn(batch_size, seq_len, input_size)
            
            output = model(x)
            
            assert output.shape == (batch_size, 1, 3)
            assert not torch.isnan(output).any()
    
    def test_custom_embedding_dim(self):
        """Test model with custom embedding dimension."""
        config = TrafficXLSTMConfig(embedding_dim=64)
        model = TrafficXLSTM(config)
        
        batch_size, seq_len, input_size = 2, 12, 3
        x = torch.randn(batch_size, seq_len, input_size)
        
        output = model(x)
        
        assert output.shape == (batch_size, 1, 3)
        assert not torch.isnan(output).any()
    
    def test_input_validation(self):
        """Test input validation in forward pass."""
        config = TrafficXLSTMConfig()
        model = TrafficXLSTM(config)
        
        # Test wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 3D input"):
            x = torch.randn(12, 3)  # 2D instead of 3D
            model(x)
        
        # Test wrong input size
        with pytest.raises(ValueError, match="Expected input_size=3"):
            x = torch.randn(2, 12, 5)  # input_size=5 instead of 3
            model(x)
    
    def test_model_info(self):
        """Test model information retrieval."""
        config = TrafficXLSTMConfig()
        model = TrafficXLSTM(config)
        
        info = model.get_model_info()
        
        assert info["model_type"] == "TrafficXLSTM"
        assert info["num_blocks"] == 6
        assert info["embedding_dim"] == 128
        assert info["slstm_positions"] == [1, 3]
        assert isinstance(info["total_parameters"], int)
        assert isinstance(info["trainable_parameters"], int)
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0
        assert info["multi_vd_mode"] is False
    
    def test_device_management(self):
        """Test device management functionality."""
        config = TrafficXLSTMConfig()
        model = TrafficXLSTM(config)
        
        # Test default device
        if torch.cuda.is_available():
            assert config.device == "cuda"
        else:
            assert config.device == "cpu"
        
        # Test explicit device setting
        model.to_device("cpu")
        assert config.device == "cpu"
    
    def test_reproducibility(self):
        """Test model reproducibility with fixed seed."""
        torch.manual_seed(42)
        config1 = TrafficXLSTMConfig()
        model1 = TrafficXLSTM(config1)
        
        torch.manual_seed(42)
        config2 = TrafficXLSTMConfig()
        model2 = TrafficXLSTM(config2)
        
        # Models should have same parameter initialization
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)
    
    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        config = TrafficXLSTMConfig()
        model = TrafficXLSTM(config)
        
        batch_size, seq_len, input_size = 2, 12, 3
        x = torch.randn(batch_size, seq_len, input_size, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for model parameters
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_training_mode(self):
        """Test training vs evaluation mode."""
        config = TrafficXLSTMConfig(dropout=0.5)
        model = TrafficXLSTM(config)
        
        batch_size, seq_len, input_size = 4, 12, 3
        x = torch.randn(batch_size, seq_len, input_size)
        
        # Training mode - outputs should vary due to dropout
        model.train()
        output1 = model(x)
        output2 = model(x)
        
        # Evaluation mode - outputs should be identical
        model.eval()
        with torch.no_grad():
            output3 = model(x)
            output4 = model(x)
        
        # In training mode with dropout, outputs should differ
        assert not torch.allclose(output1, output2, atol=1e-6)
        
        # In eval mode, outputs should be identical
        assert torch.allclose(output3, output4, atol=1e-6)


class TestTrafficXLSTMIntegration:
    """Integration tests for TrafficXLSTM."""
    
    def test_xlstm_block_configuration(self):
        """Test that xLSTM blocks are configured correctly."""
        config = TrafficXLSTMConfig(
            num_blocks=4,
            slstm_at=[1, 3],
            embedding_dim=64
        )
        model = TrafficXLSTM(config)
        
        # Verify xLSTM stack exists and is configured
        assert model.xlstm_stack is not None
        assert hasattr(model.xlstm_stack, 'blocks')
        
        # Test with input data
        x = torch.randn(2, 12, 3)
        output = model(x)
        assert output.shape == (2, 1, 3)
    
    def test_sLSTM_mLSTM_positions(self):
        """Test that sLSTM and mLSTM are at correct positions."""
        config = TrafficXLSTMConfig(
            num_blocks=6,
            slstm_at=[1, 3],  # sLSTM at positions 1 and 3
        )
        model = TrafficXLSTM(config)
        
        # Basic functionality test
        x = torch.randn(2, 12, 3)
        output = model(x)
        assert output.shape == (2, 1, 3)
        
        # Verify model info reflects configuration
        info = model.get_model_info()
        assert info["slstm_positions"] == [1, 3]
        assert info["num_blocks"] == 6


if __name__ == "__main__":
    pytest.main([__file__])