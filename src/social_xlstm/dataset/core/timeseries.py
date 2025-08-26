"""Time series dataset implementation for traffic data."""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List

from ..config import TrafficDatasetConfig
from .processor import TrafficDataProcessor
from ..storage.h5_reader import TrafficHDF5Reader


class TrafficTimeSeries(Dataset):
    """Time series traffic dataset."""
    
    def __init__(self, config: TrafficDatasetConfig, split: str = 'train', scaler=None):
        self.config = config
        self.split = split
        self.external_scaler = scaler  # For sharing scaler between splits
        
        # Load HDF5 data
        self.reader = TrafficHDF5Reader(config.hdf5_path)
        self.metadata = self.reader.get_metadata()
        
        # Get data dimensions
        self.timestamps = self.reader.get_timestamps()
        self.all_vdids = self.metadata['vdids']
        self.all_features = self.metadata['feature_names']
        
        # Filter out invalid timestamps and corresponding data
        valid_indices = self._find_valid_timesteps()
        self.timestamps = [self.timestamps[i] for i in valid_indices]
        
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
        
        # Load selected data and filter to valid timesteps
        all_data = self.reader.get_features(
            vd_indices=self.vd_indices,
            feature_indices=self.feature_indices
        )  # Shape: [T, N, F]
        
        # Filter data to only include valid timesteps
        self.data = all_data[valid_indices]  # Shape: [T_valid, N, F]
        
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
    
    def _find_valid_timesteps(self) -> List[int]:
        """Find indices of valid timesteps (non-empty timestamps)."""
        valid_indices = []
        for i, timestamp in enumerate(self.timestamps):
            if timestamp and timestamp.strip():  # Non-empty timestamp
                valid_indices.append(i)
        
        if not valid_indices:
            raise ValueError("No valid timestamps found in dataset")
        
        print(f"Found {len(valid_indices)} valid timesteps out of {len(self.timestamps)} total")
        return valid_indices
    
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
                # FIXED: Fit scaler on training data and apply to entire dataset in ONE step
                # Extract training portion for fitting
                train_data = self.data[self.start_idx:self.end_idx]
                
                # Fit scaler on training data only, but apply to entire dataset
                self.data, self.scaler = TrafficDataProcessor.normalize_features(
                    self.data, method=self.config.normalization_method, 
                    scaler=None, fit_scaler=True, fit_on_subset=(self.start_idx, self.end_idx)
                )
            else:
                # Use pre-fitted scaler from training dataset
                if self.external_scaler is not None:
                    self.scaler = self.external_scaler
                    self.data, _ = TrafficDataProcessor.normalize_features(
                        self.data, method=self.config.normalization_method, 
                        scaler=self.scaler, fit_scaler=False
                    )
                else:
                    # Fallback: fit on available data (should be avoided)
                    print(f"WARNING: No external scaler provided for {self.split} split. Fitting on current data.")
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