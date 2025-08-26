#!/usr/bin/env python3
"""
Debug script to investigate the data pipeline scaling bug.
Compares train/val target distributions and scaler parameters.
"""

import sys
sys.path.append('.')

import numpy as np
import torch
from pathlib import Path

from src.social_xlstm.utils.yaml import load_profile_config
from src.social_xlstm.dataset.core.datamodule import TrafficDataModule
from src.social_xlstm.dataset.config.base import TrafficDatasetConfig

def debug_scaling_consistency():
    """Debug the scaling consistency between train/val splits."""
    
    print("🔍 Debugging Data Pipeline Scaling Issue")
    print("=" * 50)
    
    # Use available PEMS-BAY dataset
    try:
        config = {
            'data': {
                'path': 'blob/dataset/processed/pems_bay.h5',
                'sequence_length': 12,
                'prediction_length': 1,
                'selected_vdids': None,  # Use all VDs
                'selected_features': None,  # Use all features
                'split': {'train': 0.8, 'val': 0.1, 'test': 0.1},
                'normalize': True,
                'normalization_method': 'standard',
                'fill_missing': 'interpolate',
                'stride': 1,
                'loader': {'batch_size': 8, 'num_workers': 0, 'pin_memory': False}
            }
        }
        print("✅ Config created successfully")
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return
    
    # Create dataset configuration
    try:
        dataset_config = TrafficDatasetConfig(
            hdf5_path=config['data']['path'],
            sequence_length=config['data']['sequence_length'],
            prediction_length=config['data']['prediction_length'],
            selected_vdids=config['data'].get('selected_vdids'),
            selected_features=config['data']['selected_features'],
            train_ratio=config['data']['split']['train'],
            val_ratio=config['data']['split']['val'],
            test_ratio=config['data']['split']['test'],
            normalize=config['data']['normalize'],
            normalization_method=config['data']['normalization_method'],
            fill_missing=config['data']['fill_missing'],
            stride=config['data']['stride'],
            batch_size=config['data']['loader']['batch_size'],
            num_workers=config['data']['loader']['num_workers'],
            pin_memory=config['data']['loader']['pin_memory'],
        )
        dataset_config.batch_format = 'distributed'
        print("✅ Dataset config created")
    except Exception as e:
        print(f"❌ Dataset config failed: {e}")
        return
    
    # Create data module and setup
    try:
        datamodule = TrafficDataModule(dataset_config)
        datamodule.setup(stage='fit')
        print("✅ Data module setup complete")
    except Exception as e:
        print(f"❌ Data module setup failed: {e}")
        return
    
    # Get scalers and check consistency
    print("\n📊 Scaler Analysis:")
    train_scaler = datamodule.train_dataset.get_scaler()
    val_scaler = datamodule.val_dataset.get_scaler()
    shared_scaler = datamodule.shared_scaler
    
    print(f"  Train scaler ID: {id(train_scaler)}")
    print(f"  Val scaler ID:   {id(val_scaler)}")
    print(f"  Shared scaler ID: {id(shared_scaler)}")
    
    scaler_consistent = (id(train_scaler) == id(val_scaler) == id(shared_scaler))
    print(f"  ✅ Scaler instances identical: {scaler_consistent}")
    
    if train_scaler and hasattr(train_scaler, 'mean_'):
        print(f"  📈 Scaler mean (first 3 features): {train_scaler.mean_[:3]}")
        print(f"  📈 Scaler scale (first 3 features): {train_scaler.scale_[:3]}")
    
    # Get sample batches and analyze target distributions
    print("\n🎯 Target Distribution Analysis:")
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    # Get first batch from each loader
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    # Extract targets (assuming distributed format)
    if 'targets' in train_batch:
        train_targets = train_batch['targets']
        val_targets = val_batch['targets']
    elif 'features' in train_batch and 'targets' in train_batch:
        train_targets = train_batch['targets']
        val_targets = val_batch['targets']
    else:
        print("❌ Could not find targets in batch")
        print(f"   Train batch keys: {list(train_batch.keys())}")
        return
    
    # For distributed format, targets will be dict[VD_ID, Tensor]
    if isinstance(train_targets, dict):
        # Get targets from first VD for analysis
        first_vd = next(iter(train_targets.keys()))
        train_target_tensor = train_targets[first_vd]  # [B, pred_len, F]
        val_target_tensor = val_targets[first_vd]
        
        print(f"  📊 Analyzing VD: {first_vd}")
        print(f"  📊 Train targets shape: {train_target_tensor.shape}")
        print(f"  📊 Val targets shape: {val_target_tensor.shape}")
        
        # Calculate statistics
        train_mean = train_target_tensor.mean().item()
        train_std = train_target_tensor.std().item()
        val_mean = val_target_tensor.mean().item()
        val_std = val_target_tensor.std().item()
        
        print(f"  🎯 Train targets - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
        print(f"  🎯 Val targets   - Mean: {val_mean:.4f}, Std: {val_std:.4f}")
        
        # Check for the reported issue
        mean_diff = abs(train_mean - val_mean)
        ratio_diff = abs(train_mean) / max(abs(val_mean), 1e-8)
        
        print(f"  ⚠️  Mean difference: {mean_diff:.4f}")
        print(f"  ⚠️  Mean ratio: {ratio_diff:.2f}x")
        
        if mean_diff > 0.5:
            print("  🚨 CRITICAL: Large mean difference detected!")
        
        if ratio_diff > 5.0:
            print("  🚨 CRITICAL: Large ratio difference detected!")
        
        # Calculate zero-prediction MSE (baseline)
        train_zero_mse = (train_target_tensor ** 2).mean().item()
        val_zero_mse = (val_target_tensor ** 2).mean().item()
        zero_mse_ratio = train_zero_mse / max(val_zero_mse, 1e-8)
        
        print(f"  📏 Train zero-pred MSE: {train_zero_mse:.4f}")
        print(f"  📏 Val zero-pred MSE: {val_zero_mse:.4f}")
        print(f"  📏 Zero-pred MSE ratio: {zero_mse_ratio:.2f}x")
        
        if zero_mse_ratio > 100:
            print("  🚨 CRITICAL: Massive MSE ratio detected! (>100x)")
            print("  🚨 This matches the reported 10,525x issue!")
            
    else:
        print(f"❌ Expected dict format, got {type(train_targets)}")
    
    print("\n" + "=" * 50)
    print("🔍 Debug analysis complete")

if __name__ == "__main__":
    debug_scaling_consistency()