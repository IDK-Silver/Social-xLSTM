"""
Tests for TrafficLSTM output parsing methods.

This module tests the static methods added to TrafficLSTM for parsing
multi-VD model outputs back to structured formats.
"""

import pytest
import torch
from unittest.mock import patch

from social_xlstm.models.lstm import TrafficLSTM


class TestMultiVDOutputParsing:
    """Test cases for multi-VD output parsing methods."""
    
    def test_parse_multi_vd_output_basic(self):
        """Test basic multi-VD output parsing functionality."""
        # Arrange
        batch_size, seq_len = 4, 1
        num_vds, num_features = 3, 5
        total_features = num_vds * num_features
        
        flat_output = torch.randn(batch_size, seq_len, total_features)
        
        # Act
        structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
        
        # Assert
        expected_shape = (batch_size, seq_len, num_vds, num_features)
        assert structured.shape == expected_shape
        assert isinstance(structured, torch.Tensor)
    
    def test_parse_multi_vd_output_different_dimensions(self):
        """Test parsing with different input dimensions."""
        test_cases = [
            # (batch_size, seq_len, num_vds, num_features)
            (1, 1, 2, 3),      # Minimal case
            (10, 5, 4, 5),     # Typical case
            (32, 12, 5, 7),    # Larger case
        ]
        
        for batch_size, seq_len, num_vds, num_features in test_cases:
            total_features = num_vds * num_features
            flat_output = torch.randn(batch_size, seq_len, total_features)
            
            structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
            
            expected_shape = (batch_size, seq_len, num_vds, num_features)
            assert structured.shape == expected_shape, f"Failed for dimensions: batch={batch_size}, seq={seq_len}, vds={num_vds}, feat={num_features}"
    
    def test_parse_multi_vd_output_preserves_values(self):
        """Test that parsing preserves the original values correctly."""
        # Arrange - create known values for easy verification
        batch_size, seq_len = 2, 1
        num_vds, num_features = 2, 3
        
        # Create flat output with known pattern: [0, 1, 2, 3, 4, 5]
        flat_output = torch.arange(batch_size * seq_len * num_vds * num_features, dtype=torch.float32)
        flat_output = flat_output.view(batch_size, seq_len, num_vds * num_features)
        
        # Act
        structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
        
        # Assert - check specific values
        # First sample, first sequence, first VD should be [0, 1, 2]
        assert torch.allclose(structured[0, 0, 0, :], torch.tensor([0., 1., 2.]))
        # First sample, first sequence, second VD should be [3, 4, 5]  
        assert torch.allclose(structured[0, 0, 1, :], torch.tensor([3., 4., 5.]))
    
    def test_parse_multi_vd_output_dimension_mismatch_error(self):
        """Test error handling for dimension mismatches."""
        # Arrange
        flat_output = torch.randn(4, 1, 12)  # 12 features
        num_vds, num_features = 3, 5  # Expected: 3 × 5 = 15 features
        
        # Act & Assert
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
    
    def test_parse_multi_vd_output_zero_vds_error(self):
        """Test error handling for zero VDs."""
        flat_output = torch.randn(4, 1, 15)
        
        with pytest.raises(ValueError):
            TrafficLSTM.parse_multi_vd_output(flat_output, num_vds=0, num_features=5)
    
    def test_parse_multi_vd_output_zero_features_error(self):
        """Test error handling for zero features."""
        flat_output = torch.randn(4, 1, 15)
        
        with pytest.raises(ValueError):
            TrafficLSTM.parse_multi_vd_output(flat_output, num_vds=3, num_features=0)
    
    @patch('social_xlstm.models.lstm.logger')
    def test_parse_multi_vd_output_logs_debug(self, mock_logger):
        """Test that debug logging works correctly."""
        flat_output = torch.randn(2, 1, 6)
        TrafficLSTM.parse_multi_vd_output(flat_output, num_vds=2, num_features=3)
        
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "Parsed multi-VD output" in call_args


class TestVDPredictionExtraction:
    """Test cases for VD prediction extraction methods."""
    
    def test_extract_vd_prediction_basic(self):
        """Test basic VD prediction extraction."""
        # Arrange
        batch_size, seq_len, num_vds, num_features = 4, 1, 3, 5
        structured_output = torch.randn(batch_size, seq_len, num_vds, num_features)
        vd_index = 1
        
        # Act
        vd_prediction = TrafficLSTM.extract_vd_prediction(structured_output, vd_index)
        
        # Assert
        expected_shape = (batch_size, seq_len, num_features)
        assert vd_prediction.shape == expected_shape
        
        # Verify it's the correct slice
        expected_values = structured_output[:, :, vd_index, :]
        assert torch.allclose(vd_prediction, expected_values)
    
    def test_extract_vd_prediction_all_indices(self):
        """Test extraction for all valid VD indices."""
        batch_size, seq_len, num_vds, num_features = 2, 1, 4, 3
        structured_output = torch.randn(batch_size, seq_len, num_vds, num_features)
        
        for vd_index in range(num_vds):
            vd_prediction = TrafficLSTM.extract_vd_prediction(structured_output, vd_index)
            
            expected_shape = (batch_size, seq_len, num_features)
            assert vd_prediction.shape == expected_shape
            
            # Verify correctness
            expected = structured_output[:, :, vd_index, :]
            assert torch.allclose(vd_prediction, expected)
    
    def test_extract_vd_prediction_index_out_of_range(self):
        """Test error handling for out-of-range VD indices."""
        structured_output = torch.randn(2, 1, 3, 5)  # 3 VDs (indices 0, 1, 2)
        
        # Test upper bound
        with pytest.raises(IndexError, match="VD index 3 out of range"):
            TrafficLSTM.extract_vd_prediction(structured_output, vd_index=3)
        
        # Test way out of range
        with pytest.raises(IndexError, match="VD index 10 out of range"):
            TrafficLSTM.extract_vd_prediction(structured_output, vd_index=10)
    
    def test_extract_vd_prediction_negative_index(self):
        """Test error handling for negative VD indices."""
        structured_output = torch.randn(2, 1, 3, 5)
        
        with pytest.raises(IndexError, match="VD index -1 out of range"):
            TrafficLSTM.extract_vd_prediction(structured_output, vd_index=-1)
    
    def test_extract_vd_prediction_preserves_values(self):
        """Test that extraction preserves original values."""
        # Arrange - create structured output with known values
        batch_size, seq_len, num_vds, num_features = 2, 1, 3, 2
        structured_output = torch.zeros(batch_size, seq_len, num_vds, num_features)
        
        # Set specific values for VD 1
        structured_output[:, :, 1, :] = torch.tensor([[[10., 20.]], [[30., 40.]]])
        
        # Act
        vd_1_prediction = TrafficLSTM.extract_vd_prediction(structured_output, vd_index=1)
        
        # Assert
        expected = torch.tensor([[[10., 20.]], [[30., 40.]]])
        assert torch.allclose(vd_1_prediction, expected)
    
    @patch('social_xlstm.models.lstm.logger')
    def test_extract_vd_prediction_logs_debug(self, mock_logger):
        """Test that debug logging works correctly."""
        structured_output = torch.randn(2, 1, 3, 5)
        TrafficLSTM.extract_vd_prediction(structured_output, vd_index=1)
        
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "Extracted VD_001 prediction" in call_args


class TestIntegrationScenarios:
    """Integration tests for combined parsing scenarios."""
    
    def test_full_parsing_pipeline(self):
        """Test the complete parsing pipeline from flat to individual VDs."""
        # Arrange
        batch_size, seq_len = 3, 1
        num_vds, num_features = 4, 5
        total_features = num_vds * num_features
        
        # Create flat output with distinctive patterns for each VD
        flat_output = torch.zeros(batch_size, seq_len, total_features)
        for vd_idx in range(num_vds):
            start_idx = vd_idx * num_features
            end_idx = start_idx + num_features
            flat_output[:, :, start_idx:end_idx] = (vd_idx + 1) * 10  # VD0=10, VD1=20, etc.
        
        # Act - Full pipeline
        structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
        
        individual_vds = {}
        for vd_idx in range(num_vds):
            individual_vds[f'VD_{vd_idx}'] = TrafficLSTM.extract_vd_prediction(structured, vd_idx)
        
        # Assert
        assert len(individual_vds) == num_vds
        
        for vd_idx in range(num_vds):
            vd_prediction = individual_vds[f'VD_{vd_idx}']
            expected_value = (vd_idx + 1) * 10
            
            # All values for this VD should be the expected value
            assert torch.allclose(vd_prediction, torch.full_like(vd_prediction, expected_value))
    
    def test_parse_and_extract_consistency(self):
        """Test that parse + extract gives same result as direct indexing."""
        # Arrange
        batch_size, seq_len, num_vds, num_features = 2, 1, 3, 4
        flat_output = torch.randn(batch_size, seq_len, num_vds * num_features)
        
        # Act
        structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
        
        for vd_idx in range(num_vds):
            # Method 1: Using extract_vd_prediction
            vd_pred_method1 = TrafficLSTM.extract_vd_prediction(structured, vd_idx)
            
            # Method 2: Direct indexing
            vd_pred_method2 = structured[:, :, vd_idx, :]
            
            # Assert they are identical
            assert torch.allclose(vd_pred_method1, vd_pred_method2)
    
    def test_real_model_integration(self):
        """Test integration with actual TrafficLSTM model output."""
        # Arrange
        num_vds, num_features = 2, 3
        model = TrafficLSTM.create_multi_vd_model(
            num_vds=num_vds,
            input_size=num_features,
            hidden_size=32,
            num_layers=1
        )
        
        batch_size, seq_len = 2, 12
        inputs = torch.randn(batch_size, seq_len, num_vds * num_features)
        
        # Act
        model.eval()
        with torch.no_grad():
            flat_output = model(inputs)
        
        # Test parsing
        structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
        
        # Test extraction
        for vd_idx in range(num_vds):
            vd_prediction = TrafficLSTM.extract_vd_prediction(structured, vd_idx)
            assert vd_prediction.shape == (batch_size, 1, num_features)
    
    def test_batch_processing_scenario(self):
        """Test parsing with multiple prediction steps (batch scenario)."""
        # Arrange
        batch_size, seq_len = 5, 3  # Multiple prediction steps
        num_vds, num_features = 3, 4
        
        flat_output = torch.randn(batch_size, seq_len, num_vds * num_features)
        
        # Act
        structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
        
        # Assert - check all VDs across all time steps
        for vd_idx in range(num_vds):
            vd_prediction = TrafficLSTM.extract_vd_prediction(structured, vd_idx)
            assert vd_prediction.shape == (batch_size, seq_len, num_features)
            
            # Verify that each time step is correctly extracted
            for t in range(seq_len):
                expected = structured[:, t, vd_idx, :]
                actual = vd_prediction[:, t, :]
                assert torch.allclose(actual, expected)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_batch_single_sequence(self):
        """Test minimal case: single batch, single sequence."""
        flat_output = torch.randn(1, 1, 6)  # 2 VDs × 3 features
        
        structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds=2, num_features=3)
        assert structured.shape == (1, 1, 2, 3)
        
        vd_0 = TrafficLSTM.extract_vd_prediction(structured, 0)
        assert vd_0.shape == (1, 1, 3)
    
    def test_single_feature_per_vd(self):
        """Test case with only one feature per VD."""
        batch_size, seq_len = 4, 1
        num_vds, num_features = 5, 1
        
        flat_output = torch.randn(batch_size, seq_len, num_vds)
        
        structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
        assert structured.shape == (batch_size, seq_len, num_vds, num_features)
        
        for vd_idx in range(num_vds):
            vd_pred = TrafficLSTM.extract_vd_prediction(structured, vd_idx)
            assert vd_pred.shape == (batch_size, seq_len, 1)
    
    def test_single_vd_multi_features(self):
        """Test case with single VD but multiple features."""
        batch_size, seq_len = 3, 1
        num_vds, num_features = 1, 10
        
        flat_output = torch.randn(batch_size, seq_len, num_features)
        
        structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
        assert structured.shape == (batch_size, seq_len, 1, num_features)
        
        vd_0 = TrafficLSTM.extract_vd_prediction(structured, 0)
        assert vd_0.shape == (batch_size, seq_len, num_features)
    
    def test_large_dimensions(self):
        """Test with larger, more realistic dimensions."""
        batch_size, seq_len = 64, 1
        num_vds, num_features = 10, 8
        
        flat_output = torch.randn(batch_size, seq_len, num_vds * num_features)
        
        structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
        assert structured.shape == (batch_size, seq_len, num_vds, num_features)
        
        # Test random VD extraction
        mid_vd = num_vds // 2
        vd_pred = TrafficLSTM.extract_vd_prediction(structured, mid_vd)
        assert vd_pred.shape == (batch_size, seq_len, num_features)


class TestParameterValidation:
    """Test parameter validation and type checking."""
    
    def test_parse_multi_vd_output_invalid_tensor_type(self):
        """Test error handling for non-tensor input."""
        with pytest.raises(AttributeError):
            TrafficLSTM.parse_multi_vd_output([[1, 2, 3]], num_vds=1, num_features=3)
    
    def test_parse_multi_vd_output_wrong_tensor_dimensions(self):
        """Test error handling for wrong tensor dimensions."""
        # 2D tensor instead of 3D
        with pytest.raises(ValueError):
            TrafficLSTM.parse_multi_vd_output(torch.randn(4, 15), num_vds=3, num_features=5)
        
        # 4D tensor instead of 3D
        with pytest.raises(ValueError):
            TrafficLSTM.parse_multi_vd_output(torch.randn(4, 1, 3, 5), num_vds=3, num_features=5)
    
    def test_extract_vd_prediction_invalid_tensor_type(self):
        """Test error handling for non-tensor input in extraction."""
        with pytest.raises(AttributeError):
            TrafficLSTM.extract_vd_prediction([[[[1]]]], vd_index=0)
    
    def test_extract_vd_prediction_wrong_tensor_dimensions(self):
        """Test error handling for wrong tensor dimensions in extraction."""
        # 3D tensor instead of 4D
        with pytest.raises(ValueError):
            TrafficLSTM.extract_vd_prediction(torch.randn(4, 1, 15), vd_index=0)
        
        # 2D tensor instead of 4D
        with pytest.raises(ValueError):
            TrafficLSTM.extract_vd_prediction(torch.randn(4, 15), vd_index=0)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])