"""Data preprocessing utilities for traffic data."""

import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TrafficDataProcessor:
    """Data preprocessing utilities for traffic data."""
    
    @staticmethod
    def normalize_features(data: np.ndarray, method: str = 'standard', 
                          scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None,
                          fit_scaler: bool = True) -> Tuple[np.ndarray, Union[StandardScaler, MinMaxScaler]]:
        """Normalize features across time and space dimensions."""
        original_shape = data.shape  # [T, N, F]
        
        # Reshape to [T*N, F] for fitting scaler
        data_reshaped = data.reshape(-1, data.shape[-1])
        
        # Remove NaN values for fitting
        valid_mask = ~np.isnan(data_reshaped).any(axis=1)
        valid_data = data_reshaped[valid_mask]
        
        if scaler is None:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        
        if fit_scaler and len(valid_data) > 0:
            scaler.fit(valid_data)
        elif fit_scaler and len(valid_data) == 0:
            # If no valid data for fitting, create a dummy scaler
            print("Warning: No valid data for fitting scaler. Using dummy scaler.")
            dummy_data = np.zeros((1, data.shape[-1]))
            scaler.fit(dummy_data)
        
        # Transform all data (including NaN)
        normalized_data = np.full_like(data_reshaped, np.nan)
        if len(valid_data) > 0:
            normalized_data[valid_mask] = scaler.transform(valid_data)
        
        # Reshape back to original shape
        normalized_data = normalized_data.reshape(original_shape)
        
        return normalized_data, scaler
    
    @staticmethod
    def handle_missing_values(data: np.ndarray, method: str = 'interpolate') -> np.ndarray:
        """Handle missing values in time series data."""
        if method == 'zero':
            return np.nan_to_num(data, nan=0.0)
        
        elif method == 'forward':
            # Forward fill along time axis
            result = data.copy()
            for vd_idx in range(data.shape[1]):
                for feat_idx in range(data.shape[2]):
                    series = result[:, vd_idx, feat_idx]
                    mask = ~np.isnan(series)
                    if mask.any():
                        # Forward fill
                        valid_indices = np.where(mask)[0]
                        for i in range(len(series)):
                            if np.isnan(series[i]) and len(valid_indices) > 0:
                                # Find the last valid value
                                prev_valid = valid_indices[valid_indices < i]
                                if len(prev_valid) > 0:
                                    series[i] = series[prev_valid[-1]]
            return result
        
        elif method == 'interpolate':
            # Linear interpolation along time axis
            result = data.copy()
            for vd_idx in range(data.shape[1]):
                for feat_idx in range(data.shape[2]):
                    series = result[:, vd_idx, feat_idx]
                    valid_mask = ~np.isnan(series)
                    if valid_mask.sum() > 1:
                        valid_indices = np.where(valid_mask)[0]
                        valid_values = series[valid_mask]
                        # Interpolate
                        interpolated = np.interp(
                            np.arange(len(series)), 
                            valid_indices, 
                            valid_values
                        )
                        result[:, vd_idx, feat_idx] = interpolated
            return result
        
        else:
            raise ValueError(f"Unknown missing value handling method: {method}")
    
    @staticmethod
    def create_time_features(timestamps: List[str]) -> np.ndarray:
        """Create time-based features from timestamps."""
        time_features = []
        
        for ts_str in timestamps:
            # Handle empty or invalid timestamps
            if not ts_str or ts_str.strip() == '':
                # Use NaN features for empty timestamps
                time_features.append([np.nan] * 9)  # 9 features total
                continue
                
            try:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except:
                try:
                    # Fallback parsing
                    dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
                except:
                    # If all parsing fails, use NaN features
                    time_features.append([np.nan] * 9)
                    continue
            
            # Extract time features
            hour = dt.hour / 23.0  # Normalize to [0, 1]
            minute = dt.minute / 59.0
            day_of_week = dt.weekday() / 6.0
            day_of_month = (dt.day - 1) / 30.0
            month = (dt.month - 1) / 11.0
            
            # Cyclical encoding for hour and day of week
            hour_sin = np.sin(2 * np.pi * hour)
            hour_cos = np.cos(2 * np.pi * hour)
            dow_sin = np.sin(2 * np.pi * day_of_week)
            dow_cos = np.cos(2 * np.pi * day_of_week)
            
            time_features.append([
                hour, minute, day_of_week, day_of_month, month,
                hour_sin, hour_cos, dow_sin, dow_cos
            ])
        
        return np.array(time_features, dtype=np.float32)