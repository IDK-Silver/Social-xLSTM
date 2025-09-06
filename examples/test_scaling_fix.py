#!/usr/bin/env python3
"""
Test script to verify the scaling bug fix.
"""

import sys
sys.path.append('.')

import numpy as np
import torch
from pathlib import Path

from src.social_xlstm.dataset.core.datamodule import TrafficDataModule
from src.social_xlstm.dataset.config.base import TrafficDatasetConfig

def test_scaling_fix():
    """Test that the scaling fix resolves the consistency issue."""
    
    print("🧪 Testing Scaling Bug Fix")
    print("=" * 50)
    
    # Create dataset configuration
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
    
    config.batch_format = 'distributed'
    
    print("📊 Creating data module with fixed implementation...")
    
    try:
        datamodule = TrafficDataModule(config)
        datamodule.setup(stage='fit')
        
        print("✅ Data module setup successful")
        
        # Test scaler consistency
        train_scaler = datamodule.train_dataset.get_scaler()
        val_scaler = datamodule.val_dataset.get_scaler()
        shared_scaler = datamodule.shared_scaler
        
        scaler_consistent = (id(train_scaler) == id(val_scaler) == id(shared_scaler))
        print(f"✅ Scaler instances consistent: {scaler_consistent}")
        
        # Test data distribution consistency
        print("\n🎯 Testing target distribution consistency...")
        
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        # For distributed format
        if 'targets' in train_batch:
            train_targets = train_batch['targets']
            val_targets = val_batch['targets']
        else:
            print("❌ Could not find targets in batch")
            return False
        
        # Get first VD for analysis
        first_vd = next(iter(train_targets.keys()))
        train_tensor = train_targets[first_vd]
        val_tensor = val_targets[first_vd]
        
        train_mean = train_tensor.mean().item()
        val_mean = val_tensor.mean().item()
        
        print(f"📊 Train targets mean: {train_mean:.6f}")
        print(f"📊 Val targets mean: {val_mean:.6f}")
        
        mean_diff = abs(train_mean - val_mean)
        print(f"📊 Mean difference: {mean_diff:.6f}")
        
        # Calculate zero-prediction MSE ratio
        train_zero_mse = (train_tensor ** 2).mean().item()
        val_zero_mse = (val_tensor ** 2).mean().item()
        mse_ratio = train_zero_mse / max(val_zero_mse, 1e-8)
        
        print(f"📏 Train zero-pred MSE: {train_zero_mse:.6f}")
        print(f"📏 Val zero-pred MSE: {val_zero_mse:.6f}")
        print(f"📏 MSE ratio: {mse_ratio:.2f}x")
        
        # Success criteria
        success_criteria = {
            'scaler_consistent': scaler_consistent,
            'mean_diff_small': mean_diff < 0.5,  # Should be much smaller now
            'mse_ratio_reasonable': mse_ratio < 10.0,  # Should be much smaller now
            'train_mean_normalized': abs(train_mean) < 1.0,  # Should be close to 0
            'val_mean_normalized': abs(val_mean) < 1.0   # Should be close to 0
        }
        
        print("\n🎯 Success Criteria Check:")
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {criterion}: {status}")
        
        all_passed = all(success_criteria.values())
        
        print(f"\n{'🎉 SUCCESS' if all_passed else '❌ FAILED'}: Scaling fix {'works!' if all_passed else 'needs more work'}")
        
        if all_passed:
            print("✅ The data pipeline scaling bug has been fixed!")
            print("✅ Train/val data now use consistent normalization!")
        else:
            print("❌ Some issues remain. Further investigation needed.")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_scaling_fix()
    exit(0 if success else 1)