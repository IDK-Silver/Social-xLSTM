"""
Test VD ID selection functionality in Single VD Trainer.

This test ensures that VD ID to index mapping is correct and consistent
across different scenarios.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import logging

from social_xlstm.training.without_social_pooling.single_vd_trainer import (
    SingleVDTrainer, 
    SingleVDTrainingConfig
)


@pytest.mark.unit


class TestVDSelection:
    """Test VD ID selection functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        # Create a dummy parameter to avoid empty parameter list
        dummy_param = torch.nn.Parameter(torch.randn(1))
        model.parameters.return_value = [dummy_param]
        return model
    
    @pytest.fixture
    def base_config(self):
        """Create base training configuration."""
        return SingleVDTrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            experiment_name="test_vd_selection",
            save_dir="/tmp",
            plot_training_curves=False,
            plot_predictions=False
        )
    
    @pytest.fixture
    def mock_data_loaders(self):
        """Create mock data loaders."""
        return Mock(), Mock(), Mock()
    
    def create_batch(self, vdids_structure, batch_size=2, seq_len=5, num_vds=None, num_features=3):
        """
        Create a mock batch with specified VD structure.
        
        Args:
            vdids_structure: List of VD ID lists for each sample
            batch_size: Size of the batch
            seq_len: Sequence length
            num_vds: Number of VDs (inferred from vdids_structure if None)
            num_features: Number of features
        """
        if num_vds is None:
            num_vds = len(vdids_structure[0]) if vdids_structure else 1
        
        return {
            'input_seq': torch.randn(batch_size, seq_len, num_vds, num_features),
            'target_seq': torch.randn(batch_size, 15, num_vds, num_features),
            'vdids': vdids_structure
        }
    
    def test_single_vd_selection_basic(self, mock_model, base_config, mock_data_loaders):
        """Test basic single VD selection."""
        config = base_config
        config.select_vd_id = "VD-A"
        
        trainer = SingleVDTrainer(mock_model, config, *mock_data_loaders)
        
        # Create batch with single VD
        batch = self.create_batch([['VD-A'], ['VD-A']], num_vds=1)
        
        inputs, targets = trainer.prepare_batch(batch)
        
        # Verify shapes
        assert inputs.shape == (2, 5, 3)  # [batch, seq, features]
        assert targets.shape == (2, 1, 3)  # [batch, pred_steps, features]
        
        # Verify we're using the correct VD data
        expected_inputs = batch['input_seq'][:, :, 0, :]  # First (and only) VD
        torch.testing.assert_close(inputs, expected_inputs)
    
    def test_multi_vd_selection_first(self, mock_model, base_config, mock_data_loaders):
        """Test selecting first VD from multi-VD batch."""
        config = base_config
        config.select_vd_id = "VD-A"
        
        trainer = SingleVDTrainer(mock_model, config, *mock_data_loaders)
        
        # Create batch with multiple VDs
        batch = self.create_batch([['VD-A', 'VD-B', 'VD-C'], ['VD-A', 'VD-B', 'VD-C']], num_vds=3)
        
        inputs, targets = trainer.prepare_batch(batch)
        
        # Should select VD-A (index 0)
        expected_inputs = batch['input_seq'][:, :, 0, :]
        torch.testing.assert_close(inputs, expected_inputs)
    
    def test_multi_vd_selection_middle(self, mock_model, base_config, mock_data_loaders):
        """Test selecting middle VD from multi-VD batch."""
        config = base_config
        config.select_vd_id = "VD-B"
        
        trainer = SingleVDTrainer(mock_model, config, *mock_data_loaders)
        
        # Create batch with multiple VDs
        batch = self.create_batch([['VD-A', 'VD-B', 'VD-C'], ['VD-A', 'VD-B', 'VD-C']], num_vds=3)
        
        inputs, targets = trainer.prepare_batch(batch)
        
        # Should select VD-B (index 1)
        expected_inputs = batch['input_seq'][:, :, 1, :]
        torch.testing.assert_close(inputs, expected_inputs)
    
    def test_multi_vd_selection_last(self, mock_model, base_config, mock_data_loaders):
        """Test selecting last VD from multi-VD batch."""
        config = base_config
        config.select_vd_id = "VD-C"
        
        trainer = SingleVDTrainer(mock_model, config, *mock_data_loaders)
        
        # Create batch with multiple VDs
        batch = self.create_batch([['VD-A', 'VD-B', 'VD-C'], ['VD-A', 'VD-B', 'VD-C']], num_vds=3)
        
        inputs, targets = trainer.prepare_batch(batch)
        
        # Should select VD-C (index 2)
        expected_inputs = batch['input_seq'][:, :, 2, :]
        torch.testing.assert_close(inputs, expected_inputs)
    
    def test_vd_not_found_fallback(self, mock_model, base_config, mock_data_loaders):
        """Test fallback to first VD when selected VD is not found."""
        config = base_config
        config.select_vd_id = "VD-NOT-EXIST"
        
        trainer = SingleVDTrainer(mock_model, config, *mock_data_loaders)
        
        batch = self.create_batch([['VD-A', 'VD-B'], ['VD-A', 'VD-B']], num_vds=2)
        
        with patch('social_xlstm.training.without_social_pooling.single_vd_trainer.logger') as mock_logger:
            inputs, targets = trainer.prepare_batch(batch)
            
            # Should log warning and use first VD
            mock_logger.warning.assert_called_once()
            assert "VD-NOT-EXIST" in str(mock_logger.warning.call_args)
            
            # Should use first VD (index 0)
            expected_inputs = batch['input_seq'][:, :, 0, :]
            torch.testing.assert_close(inputs, expected_inputs)
    
    def test_no_vd_selection_uses_first(self, mock_model, base_config, mock_data_loaders):
        """Test that no VD selection defaults to first VD."""
        config = base_config
        config.select_vd_id = None  # No selection
        
        trainer = SingleVDTrainer(mock_model, config, *mock_data_loaders)
        
        batch = self.create_batch([['VD-A', 'VD-B', 'VD-C'], ['VD-A', 'VD-B', 'VD-C']], num_vds=3)
        
        inputs, targets = trainer.prepare_batch(batch)
        
        # Should use first VD (index 0)
        expected_inputs = batch['input_seq'][:, :, 0, :]
        torch.testing.assert_close(inputs, expected_inputs)
    
    def test_different_batch_sizes(self, mock_model, base_config, mock_data_loaders):
        """Test VD selection works with different batch sizes."""
        config = base_config
        config.select_vd_id = "VD-B"
        
        trainer = SingleVDTrainer(mock_model, config, *mock_data_loaders)
        
        for batch_size in [1, 2, 4, 8]:
            vdids_structure = [['VD-A', 'VD-B', 'VD-C'] for _ in range(batch_size)]
            batch = self.create_batch(vdids_structure, batch_size=batch_size, num_vds=3)
            
            inputs, targets = trainer.prepare_batch(batch)
            
            # Should select VD-B (index 1) regardless of batch size
            expected_inputs = batch['input_seq'][:, :, 1, :]
            torch.testing.assert_close(inputs, expected_inputs)
            
            # Verify correct output shapes
            assert inputs.shape == (batch_size, 5, 3)
            assert targets.shape == (batch_size, 1, 3)
    
    def test_data_consistency_across_samples(self, mock_model, base_config, mock_data_loaders):
        """Test that VD selection is consistent across all samples in batch."""
        config = base_config
        config.select_vd_id = "VD-B"
        
        trainer = SingleVDTrainer(mock_model, config, *mock_data_loaders)
        
        # Create batch where all samples have the same VD structure
        batch = self.create_batch([['VD-A', 'VD-B', 'VD-C'], ['VD-A', 'VD-B', 'VD-C']], num_vds=3)
        
        # Set different values for each VD to verify correct selection
        for vd_idx in range(3):
            batch['input_seq'][:, :, vd_idx, :] = vd_idx * 10  # VD-A=0, VD-B=10, VD-C=20
        
        inputs, targets = trainer.prepare_batch(batch)
        
        # All values should be 10 (from VD-B, index 1)
        assert torch.all(inputs == 10.0)
    
    def test_edge_case_empty_vdids(self, mock_model, base_config, mock_data_loaders):
        """Test handling of edge case with empty or malformed vdids."""
        config = base_config
        config.select_vd_id = "VD-A"
        
        trainer = SingleVDTrainer(mock_model, config, *mock_data_loaders)
        
        # Test empty vdids
        batch = {
            'input_seq': torch.randn(2, 5, 1, 3),
            'target_seq': torch.randn(2, 15, 1, 3),
            'vdids': []
        }
        
        with patch('social_xlstm.training.without_social_pooling.single_vd_trainer.logger') as mock_logger:
            inputs, targets = trainer.prepare_batch(batch)
            
            # Should log warning and use default
            mock_logger.warning.assert_called()
            
            # Should use first VD (index 0)
            expected_inputs = batch['input_seq'][:, :, 0, :]
            torch.testing.assert_close(inputs, expected_inputs)
    
    def test_nested_list_format_handling(self, mock_model, base_config, mock_data_loaders):
        """Test correct handling of nested list format from DataLoader."""
        config = base_config
        config.select_vd_id = "VD-B"
        
        trainer = SingleVDTrainer(mock_model, config, *mock_data_loaders)
        
        # This is the actual format from DataLoader collate
        batch = self.create_batch([['VD-A', 'VD-B'], ['VD-A', 'VD-B']], num_vds=2)
        
        inputs, targets = trainer.prepare_batch(batch)
        
        # Should correctly extract from nested format and select VD-B (index 1)
        expected_inputs = batch['input_seq'][:, :, 1, :]
        torch.testing.assert_close(inputs, expected_inputs)
    
    def test_prediction_steps_parameter(self, mock_model, base_config, mock_data_loaders):
        """Test that prediction_steps parameter affects target shape correctly."""
        config = base_config
        config.select_vd_id = "VD-A"
        config.prediction_steps = 3  # Test with 3 prediction steps
        
        trainer = SingleVDTrainer(mock_model, config, *mock_data_loaders)
        
        batch = self.create_batch([['VD-A'], ['VD-A']], num_vds=1)
        
        inputs, targets = trainer.prepare_batch(batch)
        
        # Targets should be limited to prediction_steps
        assert targets.shape == (2, 3, 3)  # [batch, 3_pred_steps, features]
    
    @pytest.mark.parametrize("vd_structure,select_vd,expected_idx", [
        ([['VD-X']], "VD-X", 0),
        ([['VD-A', 'VD-B']], "VD-A", 0),
        ([['VD-A', 'VD-B']], "VD-B", 1),
        ([['VD-1', 'VD-2', 'VD-3', 'VD-4']], "VD-3", 2),
        ([['VD-1', 'VD-2', 'VD-3', 'VD-4']], "VD-4", 3),
    ])
    def test_parametrized_vd_selection(self, mock_model, base_config, mock_data_loaders, 
                                       vd_structure, select_vd, expected_idx):
        """Parametrized test for various VD selection scenarios."""
        config = base_config
        config.select_vd_id = select_vd
        
        trainer = SingleVDTrainer(mock_model, config, *mock_data_loaders)
        
        num_vds = len(vd_structure[0])
        batch_structure = [vd_structure[0], vd_structure[0]]  # Duplicate for batch
        batch = self.create_batch(batch_structure, num_vds=num_vds)
        
        # Set unique values for each VD to verify correct selection
        for vd_idx in range(num_vds):
            batch['input_seq'][:, :, vd_idx, :] = vd_idx * 100
        
        inputs, targets = trainer.prepare_batch(batch)
        
        # Should select the correct VD
        expected_value = expected_idx * 100
        assert torch.all(inputs == expected_value), f"Expected {expected_value}, got {inputs[0,0,0].item()}"


class TestVDSelectionIntegration:
    """Integration tests with real data structures."""
    
    def test_real_batch_format_simulation(self):
        """Test with simulated real batch format from TrafficDataModule."""
        # Simulate the exact format from TrafficTimeSeries
        batch = {
            'input_seq': torch.randn(2, 5, 3, 5),  # [B, T, N, F] - B=批次, T=時間步, N=VD數量, F=特徵
            'target_seq': torch.randn(2, 15, 3, 5),
            'vdids': [
                ['VD-11-0020-002-001', 'VD-28-0740-000-001', 'VD-13-0660-000-002'],
                ['VD-11-0020-002-001', 'VD-28-0740-000-001', 'VD-13-0660-000-002']
            ]
        }
        
        config = SingleVDTrainingConfig(
            epochs=1,
            select_vd_id="VD-28-0740-000-001",  # Middle VD
            experiment_name="integration_test"
        )
        
        model = Mock()
        dummy_param = torch.nn.Parameter(torch.randn(1))
        model.parameters.return_value = [dummy_param]
        
        trainer = SingleVDTrainer(model, config, Mock(), Mock(), Mock())
        
        # Set unique values for verification
        batch['input_seq'][:, :, 0, :] = 100  # VD-11-0020-002-001
        batch['input_seq'][:, :, 1, :] = 200  # VD-28-0740-000-001 (target)
        batch['input_seq'][:, :, 2, :] = 300  # VD-13-0660-000-002
        
        inputs, targets = trainer.prepare_batch(batch)
        
        # Should select VD-28-0740-000-001 (index 1, value 200)
        assert torch.all(inputs == 200.0)
        assert inputs.shape == (2, 5, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])