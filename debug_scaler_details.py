#!/usr/bin/env python3
"""
Detailed investigation of the scaler application process.
"""

import sys
sys.path.append('.')

import numpy as np
import torch
from pathlib import Path

from src.social_xlstm.dataset.core.timeseries import TrafficTimeSeries
from src.social_xlstm.dataset.config.base import TrafficDatasetConfig

def investigate_scaler_application():
    """Investigate how the scaler is applied in TrafficTimeSeries."""
    
    print("🔬 Deep Investigation of Scaler Application")
    print("=" * 60)
    
    # Create minimal config
    config = TrafficDatasetConfig(
        hdf5_path='blob/dataset/processed/pems_bay.h5',
        sequence_length=12,
        prediction_length=1,
        selected_vdids=None,
        selected_features=None,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        normalize=True,
        normalization_method='standard',
        fill_missing='interpolate',
        stride=1,
        batch_size=8,
        num_workers=0,
        pin_memory=False
    )
    
    print("📊 Creating TRAIN dataset and analyzing normalization...")
    
    # Create train dataset (this fits the scaler)
    train_dataset = TrafficTimeSeries(config, split='train')
    train_scaler = train_dataset.get_scaler()
    
    print(f"✅ Train dataset created with {len(train_dataset)} samples")
    print(f"🎯 Scaler parameters:")
    print(f"   Mean (first 3): {train_scaler.mean_[:3]}")
    print(f"   Scale (first 3): {train_scaler.scale_[:3]}")
    
    # Check raw data statistics before normalization
    print("\n📈 Raw data analysis (before normalization):")
    
    # We need to create a dataset without normalization to see raw data
    raw_config = TrafficDatasetConfig(
        hdf5_path='blob/dataset/processed/pems_bay.h5',
        sequence_length=12,
        prediction_length=1,
        selected_vdids=None,
        selected_features=None,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        normalize=False,  # Disable normalization
        normalization_method='standard',
        fill_missing='interpolate',
        stride=1,
        batch_size=8,
        num_workers=0,
        pin_memory=False
    )
    
    raw_train_dataset = TrafficTimeSeries(raw_config, split='train')
    raw_val_dataset = TrafficTimeSeries(raw_config, split='val')
    
    # Get a sample to check raw values
    raw_train_sample = raw_train_dataset[0]
    raw_val_sample = raw_val_dataset[0]
    
    train_raw_targets = raw_train_sample['target_seq']  # [pred_len, num_vds, num_features]
    val_raw_targets = raw_val_sample['target_seq']
    
    print(f"📊 Raw train targets stats:")
    print(f"   Shape: {train_raw_targets.shape}")
    print(f"   Mean: {train_raw_targets.mean():.4f}")
    print(f"   Std: {train_raw_targets.std():.4f}")
    print(f"   Min: {train_raw_targets.min():.4f}")
    print(f"   Max: {train_raw_targets.max():.4f}")
    
    print(f"📊 Raw val targets stats:")
    print(f"   Shape: {val_raw_targets.shape}")
    print(f"   Mean: {val_raw_targets.mean():.4f}")
    print(f"   Std: {val_raw_targets.std():.4f}")
    print(f"   Min: {val_raw_targets.min():.4f}")
    print(f"   Max: {val_raw_targets.max():.4f}")
    
    print("\n🔬 Normalized data analysis:")
    
    # Now create validation dataset WITH shared scaler
    val_dataset = TrafficTimeSeries(config, split='val', scaler=train_scaler)
    
    # Get samples from normalized datasets
    norm_train_sample = train_dataset[0]
    norm_val_sample = val_dataset[0]
    
    train_norm_targets = norm_train_sample['target_seq']
    val_norm_targets = norm_val_sample['target_seq']
    
    print(f"📊 Normalized train targets stats:")
    print(f"   Shape: {train_norm_targets.shape}")
    print(f"   Mean: {train_norm_targets.mean():.4f}")
    print(f"   Std: {train_norm_targets.std():.4f}")
    print(f"   Min: {train_norm_targets.min():.4f}")
    print(f"   Max: {train_norm_targets.max():.4f}")
    
    print(f"📊 Normalized val targets stats:")
    print(f"   Shape: {val_norm_targets.shape}")
    print(f"   Mean: {val_norm_targets.mean():.4f}")
    print(f"   Std: {val_norm_targets.std():.4f}")
    print(f"   Min: {val_norm_targets.min():.4f}")
    print(f"   Max: {val_norm_targets.max():.4f}")
    
    # Check if there's inconsistency
    train_val_mean_diff = abs(train_norm_targets.mean() - val_norm_targets.mean())
    print(f"\n⚠️  Normalized mean difference: {train_val_mean_diff:.6f}")
    
    if train_val_mean_diff > 1.0:
        print("🚨 CRITICAL: Large normalized mean difference!")
        print("🚨 This suggests different normalization was applied!")
    
    # Let's also check what happens when we manually transform the raw data
    print("\n🧪 Manual transformation verification:")
    
    # Convert to numpy and reshape for scaler
    raw_train_np = train_raw_targets.numpy().reshape(-1, train_raw_targets.shape[-1])
    raw_val_np = val_raw_targets.numpy().reshape(-1, val_raw_targets.shape[-1])
    
    # Apply the same scaler to both
    manual_train_transformed = train_scaler.transform(raw_train_np)
    manual_val_transformed = train_scaler.transform(raw_val_np)
    
    print(f"📊 Manual train transformed:")
    print(f"   Mean: {manual_train_transformed.mean():.6f}")
    print(f"   Std: {manual_train_transformed.std():.6f}")
    
    print(f"📊 Manual val transformed:")
    print(f"   Mean: {manual_val_transformed.mean():.6f}")
    print(f"   Std: {manual_val_transformed.std():.6f}")
    
    manual_diff = abs(manual_train_transformed.mean() - manual_val_transformed.mean())
    print(f"⚠️  Manual transform mean difference: {manual_diff:.6f}")
    
    if manual_diff < 0.1 and train_val_mean_diff > 1.0:
        print("🎯 DIAGNOSIS: The scaler itself is fine!")
        print("🎯 The problem is in HOW it's being applied in TrafficTimeSeries!")
        
        # Let's investigate the specific application in TrafficTimeSeries
        print("\n🔍 Investigating TrafficTimeSeries internal data:")
        
        # Check the internal normalized data arrays
        print(f"📊 Train dataset internal data stats:")
        print(f"   Full data shape: {train_dataset.data.shape}")
        print(f"   Data mean: {train_dataset.data.mean():.6f}")
        print(f"   Data std: {train_dataset.data.std():.6f}")
        
        print(f"📊 Val dataset internal data stats:")
        print(f"   Full data shape: {val_dataset.data.shape}")
        print(f"   Data mean: {val_dataset.data.mean():.6f}")
        print(f"   Data std: {val_dataset.data.std():.6f}")
        
        internal_diff = abs(train_dataset.data.mean() - val_dataset.data.mean())
        print(f"⚠️  Internal data mean difference: {internal_diff:.6f}")
        
        if internal_diff > 1.0:
            print("🚨 FOUND THE BUG: Different datasets have different internal data!")
            print("🚨 This suggests the normalization process is inconsistent!")

if __name__ == "__main__":
    investigate_scaler_application()