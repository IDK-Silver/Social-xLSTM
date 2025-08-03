"""
Unit tests for xLSTM Social Pooling (Task 2.1)

Tests the xlstm_hidden_states_aggregation algorithm and XLSTMSocialPoolingLayer.
"""

import pytest
import torch
import numpy as np
from collections import OrderedDict

from src.social_xlstm.pooling.xlstm_pooling import (
    xlstm_hidden_states_aggregation,
    XLSTMSocialPoolingLayer,
    create_mock_positions,
    validate_spatial_inputs
)


class TestXLSTMHiddenStatesAggregation:
    """Test the core xlstm_hidden_states_aggregation function."""
    
    def setup_method(self):
        """Set up test data."""
        self.batch_size = 2
        self.seq_len = 5
        self.hidden_dim = 32
        self.device = torch.device('cpu')
        
        # Create test agents in a spatial arrangement
        self.agent_ids = ['VD_001', 'VD_002', 'VD_003', 'VD_004']
        
        # Create hidden states
        self.hidden_states = OrderedDict()
        for i, agent_id in enumerate(self.agent_ids):
            # Make each agent's hidden state slightly different for testing
            base_value = (i + 1) * 0.1
            self.hidden_states[agent_id] = torch.full(
                (self.batch_size, self.seq_len, self.hidden_dim),
                base_value,
                dtype=torch.float32,
                device=self.device
            )
        
        # Create positions in a line: [0,0], [1,0], [2,0], [5,0]
        # So VD_001 and VD_002 are close (distance=1)
        # VD_002 and VD_003 are close (distance=1)  
        # VD_003 and VD_004 are far (distance=3)
        self.positions = OrderedDict()
        base_positions = [
            [0.0, 0.0],  # VD_001
            [1.0, 0.0],  # VD_002  
            [2.0, 0.0],  # VD_003
            [5.0, 0.0]   # VD_004
        ]
        
        for i, agent_id in enumerate(self.agent_ids):
            pos = base_positions[i]
            # Create constant positions over time
            self.positions[agent_id] = torch.tensor(
                [[pos] * self.seq_len] * self.batch_size,
                dtype=torch.float32,
                device=self.device
            )  # [B, T, 2]
    
    def test_basic_aggregation_mean(self):
        """Test basic mean aggregation with radius=1.5."""
        # With radius=1.5, VD_002 should have VD_001 and VD_003 as neighbors
        result = xlstm_hidden_states_aggregation(
            agent_hidden_states=self.hidden_states,
            agent_positions=self.positions,
            target_agent_id='VD_002',
            radius=1.5,
            pool_type='mean'
        )
        
        # Expected: mean of VD_001 (0.1) and VD_003 (0.3) = 0.2
        expected = torch.full((self.batch_size, self.hidden_dim), 0.2, device=self.device)
        
        assert result.shape == (self.batch_size, self.hidden_dim)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    
    def test_no_neighbors(self):
        """Test behavior when no neighbors are within radius."""
        # With radius=0.5, VD_004 should have no neighbors
        result = xlstm_hidden_states_aggregation(
            agent_hidden_states=self.hidden_states,
            agent_positions=self.positions,
            target_agent_id='VD_004',
            radius=0.5,
            pool_type='mean'
        )
        
        # Expected: zero tensor
        expected = torch.zeros((self.batch_size, self.hidden_dim), device=self.device)
        
        assert result.shape == (self.batch_size, self.hidden_dim)
        torch.testing.assert_close(result, expected)
    
    def test_max_pooling(self):
        """Test max pooling aggregation."""
        # With radius=1.5, VD_002 should have VD_001 (0.1) and VD_003 (0.3) as neighbors
        result = xlstm_hidden_states_aggregation(
            agent_hidden_states=self.hidden_states,
            agent_positions=self.positions,
            target_agent_id='VD_002',
            radius=1.5,
            pool_type='max'
        )
        
        # Expected: max of VD_001 (0.1) and VD_003 (0.3) = 0.3
        expected = torch.full((self.batch_size, self.hidden_dim), 0.3, device=self.device)
        
        assert result.shape == (self.batch_size, self.hidden_dim)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    
    def test_weighted_mean_pooling(self):
        """Test weighted mean pooling with inverse distance weighting."""
        # With radius=1.5, VD_002 should have VD_001 (dist=1) and VD_003 (dist=1) as neighbors
        # Both at same distance, so weights should be equal -> same as mean
        result = xlstm_hidden_states_aggregation(
            agent_hidden_states=self.hidden_states,
            agent_positions=self.positions,
            target_agent_id='VD_002',
            radius=1.5,
            pool_type='weighted_mean'
        )
        
        # Expected: weighted mean of VD_001 (0.1) and VD_003 (0.3) 
        # With equal distances, this should equal regular mean = 0.2
        expected = torch.full((self.batch_size, self.hidden_dim), 0.2, device=self.device)
        
        assert result.shape == (self.batch_size, self.hidden_dim)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    
    def test_large_radius_all_neighbors(self):
        """Test with large radius that includes all agents."""
        # With radius=10.0, VD_002 should have all other agents as neighbors
        result = xlstm_hidden_states_aggregation(
            agent_hidden_states=self.hidden_states,
            agent_positions=self.positions,
            target_agent_id='VD_002',
            radius=10.0,
            pool_type='mean'
        )
        
        # Expected: mean of VD_001 (0.1), VD_003 (0.3), VD_004 (0.4) = 0.267
        expected_value = (0.1 + 0.3 + 0.4) / 3
        expected = torch.full((self.batch_size, self.hidden_dim), expected_value, device=self.device)
        
        assert result.shape == (self.batch_size, self.hidden_dim)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    
    def test_missing_target_agent(self):
        """Test error handling for missing target agent."""
        with pytest.raises(ValueError, match="Target agent 'MISSING' not found"):
            xlstm_hidden_states_aggregation(
                agent_hidden_states=self.hidden_states,
                agent_positions=self.positions,
                target_agent_id='MISSING',
                radius=1.0,
                pool_type='mean'
            )
    
    def test_invalid_pool_type(self):
        """Test error handling for invalid pool type."""
        with pytest.raises(ValueError, match="Unknown pool_type: invalid"):
            xlstm_hidden_states_aggregation(
                agent_hidden_states=self.hidden_states,
                agent_positions=self.positions,
                target_agent_id='VD_001',
                radius=1.0,
                pool_type='invalid'
            )


class TestXLSTMSocialPoolingLayer:
    """Test the XLSTMSocialPoolingLayer neural network module."""
    
    def setup_method(self):
        """Set up test data."""
        self.batch_size = 2
        self.seq_len = 5
        self.hidden_dim = 32
        self.device = torch.device('cpu')
        
        # Create test data using helper function
        self.agent_ids = ['VD_001', 'VD_002', 'VD_003']
        
        self.hidden_states = OrderedDict()
        for i, agent_id in enumerate(self.agent_ids):
            self.hidden_states[agent_id] = torch.randn(
                self.batch_size, self.seq_len, self.hidden_dim,
                device=self.device
            )
        
        self.positions = create_mock_positions(
            vd_ids=self.agent_ids,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            spatial_range=5.0,
            device=self.device
        )
    
    def test_layer_initialization(self):
        """Test layer initialization with different parameters."""
        # Test basic initialization
        layer = XLSTMSocialPoolingLayer(
            hidden_dim=self.hidden_dim,
            radius=2.0,
            pool_type='mean'
        )
        
        assert layer.hidden_dim == self.hidden_dim
        assert float(layer.radius) == 2.0
        assert layer.pool_type == 'mean'
        assert not isinstance(layer.radius, torch.nn.Parameter)
        
        # Test learnable radius
        layer_learnable = XLSTMSocialPoolingLayer(
            hidden_dim=self.hidden_dim,
            radius=3.0,
            pool_type='weighted_mean',
            learnable_radius=True
        )
        
        assert isinstance(layer_learnable.radius, torch.nn.Parameter)
        assert float(layer_learnable.radius) == 3.0
    
    def test_forward_pass(self):
        """Test forward pass through the layer."""
        layer = XLSTMSocialPoolingLayer(
            hidden_dim=self.hidden_dim,
            radius=2.0,
            pool_type='mean'
        )
        
        result = layer(
            agent_hidden_states=self.hidden_states,
            agent_positions=self.positions
        )
        
        # Check output format
        assert isinstance(result, OrderedDict)
        assert set(result.keys()) == set(self.agent_ids)
        
        for agent_id, social_context in result.items():
            assert social_context.shape == (self.batch_size, self.hidden_dim)
            assert social_context.device == self.device
    
    def test_specific_target_agents(self):
        """Test forward pass with specific target agents."""
        layer = XLSTMSocialPoolingLayer(
            hidden_dim=self.hidden_dim,
            radius=2.0,
            pool_type='mean'
        )
        
        # Only process VD_001
        result = layer(
            agent_hidden_states=self.hidden_states,
            agent_positions=self.positions,
            target_agent_ids=['VD_001']
        )
        
        # Check output format
        assert isinstance(result, OrderedDict)
        assert set(result.keys()) == {'VD_001'}
        assert result['VD_001'].shape == (self.batch_size, self.hidden_dim)
    
    def test_get_info(self):
        """Test layer information retrieval."""
        layer = XLSTMSocialPoolingLayer(
            hidden_dim=64,
            radius=1.5,
            pool_type='max',
            learnable_radius=True
        )
        
        info = layer.get_info()
        
        assert info['layer_type'] == 'XLSTMSocialPoolingLayer'
        assert info['hidden_dim'] == 64
        assert info['radius'] == 1.5
        assert info['pool_type'] == 'max'
        assert info['learnable_radius'] is True


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_mock_positions(self):
        """Test mock position generation."""
        vd_ids = ['VD_001', 'VD_002', 'VD_003']
        batch_size = 3
        seq_len = 10
        spatial_range = 8.0
        
        positions = create_mock_positions(
            vd_ids=vd_ids,
            batch_size=batch_size,
            seq_len=seq_len,
            spatial_range=spatial_range,
            device=torch.device('cpu')
        )
        
        assert isinstance(positions, OrderedDict)
        assert set(positions.keys()) == set(vd_ids)
        
        for vd_id, pos_tensor in positions.items():
            assert pos_tensor.shape == (batch_size, seq_len, 2)
            assert pos_tensor.dtype == torch.float32
    
    def test_validate_spatial_inputs_valid(self):
        """Test validation with valid inputs."""
        hidden_states = {
            'A': torch.randn(2, 5, 32),
            'B': torch.randn(2, 5, 32)
        }
        positions = {
            'A': torch.randn(2, 5, 2),
            'B': torch.randn(2, 5, 2)
        }
        
        # Should not raise any exception
        validate_spatial_inputs(hidden_states, positions)
    
    def test_validate_spatial_inputs_missing_positions(self):
        """Test validation with missing position data."""
        hidden_states = {
            'A': torch.randn(2, 5, 32),
            'B': torch.randn(2, 5, 32)
        }
        positions = {
            'A': torch.randn(2, 5, 2)
            # Missing 'B'
        }
        
        with pytest.raises(ValueError, match="Missing positions for agents"):
            validate_spatial_inputs(hidden_states, positions)
    
    def test_validate_spatial_inputs_shape_mismatch(self):
        """Test validation with incompatible shapes."""
        hidden_states = {
            'A': torch.randn(2, 5, 32)
        }
        positions = {
            'A': torch.randn(3, 5, 2)  # Different batch size
        }
        
        with pytest.raises(ValueError, match="incompatible batch/time dimensions"):
            validate_spatial_inputs(hidden_states, positions)
        
        # Test wrong spatial dimension
        positions_wrong_dim = {
            'A': torch.randn(2, 5, 3)  # Should be 2D positions
        }
        
        with pytest.raises(ValueError, match="positions must have shape"):
            validate_spatial_inputs(hidden_states, positions_wrong_dim)


if __name__ == '__main__':
    pytest.main([__file__])