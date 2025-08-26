"""
Emergency Data Quality Cleanup Module for Social-xLSTM

This module implements Phase 1 emergency fixes for critical data quality issues:
- 49.8% NaN values in 3/5 features (avg_speed, avg_occupancy, speed_std) 
- 100% constant feature (lane_count = 2.0)
- Resulting in model training failure (R² = -1000.6)

Emergency Strategy:
1. Remove constant features automatically
2. Implement robust NaN handling (interpolation + forward fill)
3. Basic normalization to prevent numerical instability
4. Return clean data suitable for model training

Author: Social-xLSTM Project Team
Created: 2025-08-04 (Phase 1 Emergency Response)
"""

import logging
import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


class EmergencyDataCleaner:
    """
    Emergency data cleaner for critical quality issues.
    
    This class implements immediate fixes for the worst data quality problems
    that prevent model training from succeeding.
    """
    
    def __init__(self, 
                 nan_threshold: float = 0.95,
                 constant_threshold: float = 1e-6,
                 min_valid_features: int = 2):
        """
        Initialize emergency data cleaner.
        
        Args:
            nan_threshold: Remove features with NaN percentage above this (0.95 = 95%)
            constant_threshold: Remove features with variance below this
            min_valid_features: Minimum features required after cleaning
        """
        self.nan_threshold = nan_threshold
        self.constant_threshold = constant_threshold
        self.min_valid_features = min_valid_features
        
        # Track cleaning statistics
        self.cleaning_stats = {
            'original_shape': None,
            'final_shape': None,
            'features_removed': [],
            'nan_counts_before': {},
            'nan_counts_after': {},
            'constant_features': [],
            'normalization_params': {}
        }
    
    def emergency_cleanup(self, data: np.ndarray, 
                         feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Emergency cleanup pipeline for critical data quality issues.
        
        Args:
            data: Input data array, shape (time_steps, num_vds, num_features)
            feature_names: Optional list of feature names for logging
            
        Returns:
            Tuple of (cleaned_data, cleaning_stats)
        """
        logger.info("🚨 Starting emergency data cleanup...")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[-1])]
        
        self.cleaning_stats['original_shape'] = data.shape
        original_data = data.copy()
        
        # Step 1: Identify and log data quality issues
        logger.info("📊 Analyzing data quality issues...")
        self._analyze_quality_issues(data, feature_names)
        
        # Step 2: Remove constant features
        logger.info("🔧 Removing constant features...")
        data, feature_names = self._remove_constant_features(data, feature_names)
        
        # Step 3: Handle NaN values with robust interpolation
        logger.info("🩹 Handling NaN values...")
        data = self._handle_nan_values(data, feature_names)
        
        # Step 4: Remove features with excessive NaN (if any remain)
        logger.info("✂️ Removing features with excessive NaN...")
        data, feature_names = self._remove_excessive_nan_features(data, feature_names)
        
        # Step 5: Basic normalization for numerical stability
        logger.info("📏 Applying basic normalization...")
        data = self._apply_basic_normalization(data, feature_names)
        
        # Step 6: Final validation
        logger.info("✅ Validating cleaned data...")
        self._validate_cleaned_data(data, feature_names)
        
        self.cleaning_stats['final_shape'] = data.shape
        self.cleaning_stats['remaining_features'] = feature_names
        
        logger.info(f"🎉 Emergency cleanup complete: {original_data.shape} → {data.shape}")
        
        return data, self.cleaning_stats
    
    def _analyze_quality_issues(self, data: np.ndarray, feature_names: List[str]):
        """Analyze and log current data quality issues."""
        num_features = data.shape[-1]
        
        for i in range(num_features):
            feature_data = data[:, :, i].flatten()
            nan_count = np.isnan(feature_data).sum()
            total_count = len(feature_data)
            nan_pct = (nan_count / total_count) * 100
            
            self.cleaning_stats['nan_counts_before'][feature_names[i]] = {
                'count': int(nan_count),
                'percentage': float(nan_pct)
            }
            
            # Check if constant
            valid_data = feature_data[~np.isnan(feature_data)]
            if len(valid_data) > 0:
                variance = np.var(valid_data)
                is_constant = variance < self.constant_threshold
                
                if is_constant:
                    self.cleaning_stats['constant_features'].append({
                        'name': feature_names[i],
                        'value': float(valid_data[0]),
                        'variance': float(variance)
                    })
            
            logger.info(f"  {feature_names[i]}: {nan_pct:.1f}% NaN, "
                       f"{'CONSTANT' if feature_names[i] in [cf['name'] for cf in self.cleaning_stats['constant_features']] else 'VARIABLE'}")
    
    def _remove_constant_features(self, data: np.ndarray, 
                                feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Remove features that are constant across all samples."""
        features_to_keep = []
        indices_to_keep = []
        
        for i, name in enumerate(feature_names):
            feature_data = data[:, :, i].flatten()
            valid_data = feature_data[~np.isnan(feature_data)]
            
            if len(valid_data) > 0:
                variance = np.var(valid_data)
                if variance >= self.constant_threshold:
                    features_to_keep.append(name)
                    indices_to_keep.append(i)
                else:
                    logger.warning(f"  Removing constant feature: {name} (variance={variance:.2e})")
                    self.cleaning_stats['features_removed'].append({
                        'name': name,
                        'reason': 'constant',
                        'variance': float(variance)
                    })
        
        if indices_to_keep:
            cleaned_data = data[:, :, indices_to_keep]
            logger.info(f"  Kept {len(features_to_keep)}/{len(feature_names)} features")
            return cleaned_data, features_to_keep
        else:
            raise ValueError("All features are constant! Cannot proceed with training.")
    
    def _handle_nan_values(self, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Handle NaN values using robust interpolation strategies."""
        cleaned_data = data.copy()
        
        for i, name in enumerate(feature_names):
            feature_data = cleaned_data[:, :, i]
            
            # Count NaNs before cleaning
            nan_count_before = np.isnan(feature_data).sum()
            if nan_count_before == 0:
                continue
            
            logger.info(f"  Processing {name}: {nan_count_before} NaN values")
            
            # Strategy 1: Forward fill for each VD independently
            for vd_idx in range(feature_data.shape[1]):
                vd_series = feature_data[:, vd_idx]
                
                # Forward fill
                mask = ~np.isnan(vd_series)
                if mask.any():
                    # Find first valid value
                    first_valid_idx = np.where(mask)[0][0]
                    last_valid_idx = np.where(mask)[0][-1]
                    
                    # Forward fill from first valid value
                    filled_series = vd_series.copy()
                    for t in range(first_valid_idx + 1, len(vd_series)):
                        if np.isnan(filled_series[t]):
                            filled_series[t] = filled_series[t-1]
                    
                    # Backward fill before first valid value
                    if first_valid_idx > 0:
                        filled_series[:first_valid_idx] = filled_series[first_valid_idx]
                    
                    cleaned_data[:, vd_idx, i] = filled_series
            
            # Strategy 2: If still NaNs, use global mean
            remaining_nans = np.isnan(cleaned_data[:, :, i]).sum()
            if remaining_nans > 0:
                global_mean = np.nanmean(cleaned_data[:, :, i])
                if not np.isnan(global_mean):
                    cleaned_data[:, :, i] = np.nan_to_num(cleaned_data[:, :, i], nan=global_mean)
                    logger.warning(f"  Used global mean ({global_mean:.2f}) for remaining NaN in {name}")
            
            # Final check
            final_nan_count = np.isnan(cleaned_data[:, :, i]).sum()
            logger.info(f"  {name}: {nan_count_before} → {final_nan_count} NaN values")
        
        return cleaned_data
    
    def _remove_excessive_nan_features(self, data: np.ndarray, 
                                     feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Remove features that still have excessive NaN after cleaning."""
        features_to_keep = []
        indices_to_keep = []
        
        for i, name in enumerate(feature_names):
            feature_data = data[:, :, i].flatten()
            nan_count = np.isnan(feature_data).sum()
            total_count = len(feature_data)
            nan_pct = nan_count / total_count
            
            if nan_pct <= self.nan_threshold:
                features_to_keep.append(name)
                indices_to_keep.append(i)
                
                self.cleaning_stats['nan_counts_after'][name] = {
                    'count': int(nan_count),
                    'percentage': float(nan_pct * 100)
                }
            else:
                logger.warning(f"  Removing feature with excessive NaN: {name} ({nan_pct*100:.1f}%)")
                self.cleaning_stats['features_removed'].append({
                    'name': name,
                    'reason': 'excessive_nan',
                    'nan_percentage': float(nan_pct * 100)
                })
        
        if len(features_to_keep) < self.min_valid_features:
            raise ValueError(f"Only {len(features_to_keep)} valid features remaining, "
                           f"minimum required: {self.min_valid_features}")
        
        if indices_to_keep:
            cleaned_data = data[:, :, indices_to_keep]
            logger.info(f"  Kept {len(features_to_keep)} features after NaN filtering")
            return cleaned_data, features_to_keep
        else:
            raise ValueError("No features passed NaN threshold check!")
    
    def _apply_basic_normalization(self, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Apply basic standardization for numerical stability."""
        normalized_data = data.copy()
        
        for i, name in enumerate(feature_names):
            feature_data = normalized_data[:, :, i]
            
            # Calculate robust statistics (ignoring any remaining NaN)
            mean_val = np.nanmean(feature_data)
            std_val = np.nanstd(feature_data)
            
            # Avoid division by zero
            if std_val < 1e-8:
                std_val = 1.0
                logger.warning(f"  {name}: Very small std ({std_val:.2e}), using 1.0")
            
            # Standardize
            normalized_data[:, :, i] = (feature_data - mean_val) / std_val
            
            self.cleaning_stats['normalization_params'][name] = {
                'mean': float(mean_val),
                'std': float(std_val)
            }
            
            logger.info(f"  {name}: normalized (mean={mean_val:.2f}, std={std_val:.2f})")
        
        return normalized_data
    
    def _validate_cleaned_data(self, data: np.ndarray, feature_names: List[str]):
        """Validate that cleaned data is suitable for training."""
        # Check for any remaining NaN
        total_nan = np.isnan(data).sum()
        if total_nan > 0:
            raise ValueError(f"Cleaned data still contains {total_nan} NaN values!")
        
        # Check for infinite values
        total_inf = np.isinf(data).sum()
        if total_inf > 0:
            raise ValueError(f"Cleaned data contains {total_inf} infinite values!")
        
        # Check feature count
        num_features = data.shape[-1]
        if num_features < self.min_valid_features:
            raise ValueError(f"Only {num_features} features remaining, "
                           f"minimum required: {self.min_valid_features}")
        
        # Check data variability
        for i, name in enumerate(feature_names):
            feature_std = np.std(data[:, :, i])
            if feature_std < 1e-6:
                logger.warning(f"Feature {name} has very low variability (std={feature_std:.2e})")
        
        logger.info(f"✅ Data validation passed: {data.shape}, {num_features} features")


def emergency_cleanup(data_path: str, 
                     output_suffix: str = "_cleaned",
                     **cleaner_kwargs) -> Tuple[str, Dict]:
    """
    Convenience function for emergency data cleanup.
    
    Args:
        data_path: Path to HDF5 data file
        output_suffix: Suffix for cleaned data file
        **cleaner_kwargs: Arguments for EmergencyDataCleaner
        
    Returns:
        Tuple of (cleaned_data_path, cleaning_stats)
    """
    import h5py
    from pathlib import Path
    
    logger.info(f"🚨 Emergency cleanup starting for: {data_path}")
    
    # Load original data
    with h5py.File(data_path, 'r') as f:
        data = f['data/features'][:]
        feature_names = [name.decode() if isinstance(name, bytes) else str(name) 
                        for name in f['metadata/feature_names'][:]]
        vdids = [vid.decode() if isinstance(vid, bytes) else str(vid) 
                for vid in f['metadata/vdids'][:]]
        timestamps = f['metadata/timestamps'][:]
    
    logger.info(f"Original data: {data.shape}, features: {feature_names}")
    
    # Clean data
    cleaner = EmergencyDataCleaner(**cleaner_kwargs)
    cleaned_data, stats = cleaner.emergency_cleanup(data, feature_names)
    
    # Save cleaned data
    data_path_obj = Path(data_path)
    cleaned_path = str(data_path_obj.parent / f"{data_path_obj.stem}{output_suffix}.h5")
    
    remaining_features = stats['remaining_features']
    
    with h5py.File(cleaned_path, 'w') as f:
        # Save cleaned data
        data_group = f.create_group('data')
        data_group.create_dataset('features', data=cleaned_data)
        
        # Save updated metadata
        metadata_group = f.create_group('metadata')
        metadata_group.create_dataset('feature_names', 
                                     data=[name.encode() for name in remaining_features])
        metadata_group.create_dataset('vdids', data=[vid.encode() for vid in vdids])
        metadata_group.create_dataset('timestamps', data=timestamps)
        
        # Save cleaning statistics
        import json
        stats_str = json.dumps(stats, indent=2, default=str)
        metadata_group.create_dataset('cleaning_stats', data=stats_str.encode())
    
    logger.info(f"✅ Cleaned data saved to: {cleaned_path}")
    logger.info(f"📊 Final shape: {cleaned_data.shape}")
    logger.info(f"🏷️ Remaining features: {remaining_features}")
    
    return cleaned_path, stats


if __name__ == "__main__":
    # Emergency cleanup for development dataset
    logging.basicConfig(level=logging.INFO)
    
    dev_data_path = "blob/dataset/pre-processed/h5/traffic_features_dev.h5"
    cleaned_path, stats = emergency_cleanup(dev_data_path)
    
    print(f"\n🎉 Emergency cleanup completed!")
    print(f"Original: {stats['original_shape']}")
    print(f"Cleaned: {stats['final_shape']}")
    print(f"Features removed: {len(stats['features_removed'])}")
    print(f"Cleaned data: {cleaned_path}")