"""Tests for TrafficDataProcessor."""

import pytest
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from social_xlstm.dataset.core.processor import TrafficDataProcessor


class TestTrafficDataProcessor:
    """Test TrafficDataProcessor class."""
    
    def test_normalize_features_standard(self):
        """Test standard normalization."""
        # Create test data [T, N, F]
        data = np.array([
            [[1.0, 2.0], [3.0, 4.0]],  # Time 0
            [[5.0, 6.0], [7.0, 8.0]],  # Time 1
            [[9.0, 10.0], [11.0, 12.0]]  # Time 2
        ])
        
        normalized, scaler = TrafficDataProcessor.normalize_features(
            data, method='standard'
        )
        
        assert isinstance(scaler, StandardScaler)
        assert normalized.shape == data.shape
        # Check that mean is close to 0 and std is close to 1
        flat_data = normalized.reshape(-1, 2)
        assert np.allclose(np.mean(flat_data, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(flat_data, axis=0), 1, atol=1e-10)
    
    def test_normalize_features_minmax(self):
        """Test min-max normalization."""
        data = np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]]
        ])
        
        normalized, scaler = TrafficDataProcessor.normalize_features(
            data, method='minmax'
        )
        
        assert isinstance(scaler, MinMaxScaler)
        assert normalized.shape == data.shape
        # Check that values are in [0, 1] range
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1 + 1e-10)  # Allow small floating point errors
    
    def test_normalize_features_with_nan(self):
        """Test normalization with NaN values."""
        data = np.array([
            [[1.0, np.nan], [3.0, 4.0]],
            [[np.nan, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]]
        ])
        
        normalized, scaler = TrafficDataProcessor.normalize_features(
            data, method='standard'
        )
        
        assert normalized.shape == data.shape
        # NaN values should remain NaN
        assert np.isnan(normalized[0, 0, 1])
        assert np.isnan(normalized[1, 0, 0])
        # Valid values should be normalized
        assert not np.isnan(normalized[2, 1, 0])
    
    def test_normalize_features_with_existing_scaler(self):
        """Test normalization with pre-fitted scaler."""
        data1 = np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ])
        
        data2 = np.array([
            [[9.0, 10.0], [11.0, 12.0]]
        ])
        
        # Fit scaler on first dataset
        _, scaler = TrafficDataProcessor.normalize_features(
            data1, method='standard', fit_scaler=True
        )
        
        # Apply same scaler to second dataset
        normalized2, _ = TrafficDataProcessor.normalize_features(
            data2, method='standard', scaler=scaler, fit_scaler=False
        )
        
        assert normalized2.shape == data2.shape
        # The scaler should be the same object
        assert _ is scaler
    
    def test_handle_missing_values_zero(self):
        """Test zero filling for missing values."""
        data = np.array([
            [[1.0, np.nan], [3.0, 4.0]],
            [[np.nan, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]]
        ])
        
        result = TrafficDataProcessor.handle_missing_values(data, method='zero')
        
        assert result.shape == data.shape
        assert result[0, 0, 1] == 0.0
        assert result[1, 0, 0] == 0.0
        assert result[0, 0, 0] == 1.0  # Non-NaN values unchanged
    
    def test_handle_missing_values_forward(self):
        """Test forward fill for missing values."""
        data = np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[np.nan, 6.0], [7.0, np.nan]],
            [[9.0, np.nan], [11.0, 12.0]]
        ])
        
        result = TrafficDataProcessor.handle_missing_values(data, method='forward')
        
        assert result.shape == data.shape
        assert result[1, 0, 0] == 1.0  # Forward filled from time 0
        assert result[2, 0, 1] == 6.0  # Forward filled from time 1
        assert result[1, 1, 1] == 4.0  # Forward filled from time 0
    
    def test_handle_missing_values_interpolate(self):
        """Test linear interpolation for missing values."""
        data = np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[np.nan, 6.0], [7.0, np.nan]],
            [[9.0, 10.0], [11.0, 12.0]]
        ])
        
        result = TrafficDataProcessor.handle_missing_values(data, method='interpolate')
        
        assert result.shape == data.shape
        # Check interpolation: (1 + 9) / 2 = 5
        assert result[1, 0, 0] == 5.0
        # Check interpolation: (4 + 12) / 2 = 8
        assert result[1, 1, 1] == 8.0
    
    def test_create_time_features(self):
        """Test time feature creation."""
        timestamps = [
            "2023-01-01T12:30:00",
            "2023-01-02T06:15:00",
            "2023-12-31T23:59:00"
        ]
        
        features = TrafficDataProcessor.create_time_features(timestamps)
        
        assert features.shape == (3, 9)  # 3 timestamps, 9 features
        assert features.dtype == np.float32
        
        # Check first timestamp (2023-01-01T12:30:00)
        # hour = 12/23 ≈ 0.52, minute = 30/59 ≈ 0.51
        assert np.isclose(features[0, 0], 12/23)  # hour
        assert np.isclose(features[0, 1], 30/59)  # minute
        assert np.isclose(features[0, 2], 6/6)    # day_of_week (Sunday = 6)
        assert np.isclose(features[0, 3], 0/30)   # day_of_month (1st - 1 = 0)
        assert np.isclose(features[0, 4], 0/11)   # month (Jan = 0)
        
        # Check cyclical features are in valid range
        assert -1 <= features[0, 5] <= 1  # hour_sin
        assert -1 <= features[0, 6] <= 1  # hour_cos
        assert -1 <= features[0, 7] <= 1  # dow_sin
        assert -1 <= features[0, 8] <= 1  # dow_cos
    
    def test_create_time_features_with_z_suffix(self):
        """Test time feature creation with Z suffix."""
        timestamps = ["2023-01-01T12:30:00Z"]
        
        features = TrafficDataProcessor.create_time_features(timestamps)
        
        assert features.shape == (1, 9)
        assert features.dtype == np.float32
    
    def test_invalid_normalization_method(self):
        """Test invalid normalization method."""
        data = np.array([[[1.0, 2.0]]])
        
        with pytest.raises(ValueError, match="Unknown normalization method"):
            TrafficDataProcessor.normalize_features(data, method='invalid')
    
    def test_invalid_missing_value_method(self):
        """Test invalid missing value handling method."""
        data = np.array([[[1.0, np.nan]]])
        
        with pytest.raises(ValueError, match="Unknown missing value handling method"):
            TrafficDataProcessor.handle_missing_values(data, method='invalid')