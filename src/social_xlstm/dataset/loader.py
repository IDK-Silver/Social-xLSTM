import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pytorch_lightning as pl

from .json_utils import VDInfo, VDLiveList
from .h5_utils import TrafficHDF5Reader


@dataclass
class TrafficDatasetConfig:
    """Configuration for traffic dataset."""
    hdf5_path: Path
    sequence_length: int = 60           # Input sequence length (minutes)
    prediction_length: int = 15         # Prediction sequence length (minutes)
    selected_vdids: Optional[List[str]] = None
    selected_features: Optional[List[str]] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    normalize: bool = True
    normalization_method: str = 'standard'  # 'standard', 'minmax'
    fill_missing: str = 'interpolate'   # 'zero', 'forward', 'interpolate'
    stride: int = 1                     # Sliding window stride
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        self.hdf5_path = Path(self.hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")
        
        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")


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
        
        # Transform all data (including NaN)
        normalized_data = np.full_like(data_reshaped, np.nan)
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
            try:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except:
                # Fallback parsing
                dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
            
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


class TrafficTimeSeries(Dataset):
    """Time series traffic dataset."""
    
    def __init__(self, config: TrafficDatasetConfig, split: str = 'train'):
        self.config = config
        self.split = split
        
        # Load HDF5 data
        self.reader = TrafficHDF5Reader(config.hdf5_path)
        self.metadata = self.reader.get_metadata()
        
        # Get data dimensions
        self.timestamps = self.reader.get_timestamps()
        self.all_vdids = self.metadata['vdids']
        self.all_features = self.metadata['feature_names']
        
        # Select VDs and features
        self.selected_vdids = config.selected_vdids or self.all_vdids
        self.selected_features = config.selected_features or self.all_features
        
        # Get indices for selection
        self.vd_indices = [self.all_vdids.index(vdid) for vdid in self.selected_vdids 
                          if vdid in self.all_vdids]
        self.feature_indices = [self.all_features.index(feat) for feat in self.selected_features 
                               if feat in self.all_features]
        
        if not self.vd_indices:
            raise ValueError("No valid VDIDs found in dataset")
        if not self.feature_indices:
            raise ValueError("No valid features found in dataset")
        
        # Load selected data
        self.data = self.reader.get_features(
            vd_indices=self.vd_indices,
            feature_indices=self.feature_indices
        )  # Shape: [T, N, F]
        
        # Calculate split indices
        total_length = len(self.timestamps)
        train_end = int(total_length * config.train_ratio)
        val_end = int(total_length * (config.train_ratio + config.val_ratio))
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = train_end
        elif split == 'val':
            self.start_idx = train_end
            self.end_idx = val_end
        elif split == 'test':
            self.start_idx = val_end
            self.end_idx = total_length
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Create valid sample indices
        self.valid_indices = self._create_sample_indices()
        
        # Data preprocessing
        self.scaler = None
        self._preprocess_data()
        
        print(f"{split.upper()} dataset: {len(self.valid_indices)} samples, "
              f"VDs: {len(self.selected_vdids)}, Features: {len(self.selected_features)}")
    
    def _create_sample_indices(self) -> List[int]:
        """Create valid sample starting indices."""
        valid_indices = []
        min_length = self.config.sequence_length + self.config.prediction_length
        
        for i in range(self.start_idx, self.end_idx - min_length + 1, self.config.stride):
            valid_indices.append(i)
        
        return valid_indices
    
    def _preprocess_data(self):
        """Preprocess the data."""
        # Handle missing values
        self.data = TrafficDataProcessor.handle_missing_values(
            self.data, method=self.config.fill_missing
        )
        
        # Normalize features
        if self.config.normalize:
            if self.split == 'train':
                # Fit scaler on training data only
                train_data = self.data[self.start_idx:self.end_idx]
                self.data, self.scaler = TrafficDataProcessor.normalize_features(
                    self.data, method=self.config.normalization_method, fit_scaler=True
                )
            else:
                # Use pre-fitted scaler (should be passed from training dataset)
                # For now, we'll fit on available data (this should be improved)
                self.data, self.scaler = TrafficDataProcessor.normalize_features(
                    self.data, method=self.config.normalization_method, fit_scaler=True
                )
        
        # Create time features
        self.time_features = TrafficDataProcessor.create_time_features(self.timestamps)
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        start_idx = self.valid_indices[idx]
        input_end = start_idx + self.config.sequence_length
        target_end = input_end + self.config.prediction_length
        
        # Extract sequences
        input_seq = self.data[start_idx:input_end]  # [seq_len, num_vds, num_features]
        target_seq = self.data[input_end:target_end]  # [pred_len, num_vds, num_features]
        
        # Extract time features
        input_time_feat = self.time_features[start_idx:input_end]  # [seq_len, time_feat_dim]
        target_time_feat = self.time_features[input_end:target_end]  # [pred_len, time_feat_dim]
        
        # Create masks (1 for valid data, 0 for missing/invalid)
        input_mask = ~np.isnan(input_seq).any(axis=-1)  # [seq_len, num_vds]
        target_mask = ~np.isnan(target_seq).any(axis=-1)  # [pred_len, num_vds]
        
        # Replace any remaining NaN with 0
        input_seq = np.nan_to_num(input_seq, nan=0.0)
        target_seq = np.nan_to_num(target_seq, nan=0.0)
        
        return {
            'input_seq': torch.FloatTensor(input_seq),
            'target_seq': torch.FloatTensor(target_seq),
            'input_time_feat': torch.FloatTensor(input_time_feat),
            'target_time_feat': torch.FloatTensor(target_time_feat),
            'input_mask': torch.BoolTensor(input_mask),
            'target_mask': torch.BoolTensor(target_mask),
            'timestamps': self.timestamps[start_idx:target_end],
            'vdids': self.selected_vdids
        }
    
    def get_scaler(self):
        """Get the fitted scaler for inverse transformation."""
        return self.scaler
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data."""
        if self.scaler is None:
            return data
        
        original_shape = data.shape
        data_reshaped = data.reshape(-1, data.shape[-1])
        
        # Only transform non-zero values (assuming 0 means missing)
        non_zero_mask = (data_reshaped != 0).any(axis=1)
        result = data_reshaped.copy()
        
        if non_zero_mask.any():
            result[non_zero_mask] = self.scaler.inverse_transform(data_reshaped[non_zero_mask])
        
        return result.reshape(original_shape)


class TrafficDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for traffic data."""
    
    def __init__(self, config: TrafficDatasetConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.shared_scaler = None
    
    def setup(self, stage: str = None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = TrafficTimeSeries(self.config, split='train')
            self.val_dataset = TrafficTimeSeries(self.config, split='val')
            
            # Share scaler from training to validation
            self.shared_scaler = self.train_dataset.get_scaler()
            if self.shared_scaler is not None:
                self.val_dataset.scaler = self.shared_scaler
        
        if stage == 'test' or stage is None:
            self.test_dataset = TrafficTimeSeries(self.config, split='test')
            
            # Share scaler to test dataset
            if self.shared_scaler is not None:
                self.test_dataset.scaler = self.shared_scaler
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )
    
    def get_data_info(self) -> Dict[str, any]:
        """Get dataset information."""
        if self.train_dataset is None:
            self.setup('fit')
        
        return {
            'num_vds': len(self.train_dataset.selected_vdids),
            'num_features': len(self.train_dataset.selected_features),
            'time_feat_dim': self.train_dataset.time_features.shape[1],
            'sequence_length': self.config.sequence_length,
            'prediction_length': self.config.prediction_length,
            'vdids': self.train_dataset.selected_vdids,
            'features': self.train_dataset.selected_features,
            'scaler': self.shared_scaler
        }


class TrafficeFeature:
    """Legacy class for backward compatibility."""
    
    @staticmethod
    def create_dataloader(config: TrafficDatasetConfig, split: str = 'train') -> DataLoader:
        """Create a single dataloader."""
        dataset = TrafficTimeSeries(config, split=split)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == 'train'),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )


# Convenience functions
def create_traffic_datamodule(hdf5_path: Union[str, Path], **kwargs) -> TrafficDataModule:
    """Create traffic data module with default configuration."""
    config = TrafficDatasetConfig(hdf5_path=hdf5_path, **kwargs)
    return TrafficDataModule(config)


def create_traffic_dataloaders(hdf5_path: Union[str, Path], **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders."""
    data_module = create_traffic_datamodule(hdf5_path, **kwargs)
    data_module.setup()
    
    return (
        data_module.train_dataloader(),
        data_module.val_dataloader(), 
        data_module.test_dataloader()
    )