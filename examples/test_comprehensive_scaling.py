#!/usr/bin/env python3
"""
Comprehensive test to validate scaling fix across different scenarios.
"""

import sys
sys.path.append('.')

import numpy as np
import torch
from pathlib import Path

from src.social_xlstm.dataset.core.timeseries import TrafficTimeSeries
from src.social_xlstm.dataset.config.base import TrafficDatasetConfig

def test_edge_cases():
    """Test edge cases and boundary conditions for the scaling fix."""
    
    print("🧪 Comprehensive Scaling Fix Validation")
    print("=" * 60)
    
    results = {}
    
    # Test different data splits
    test_configs = [
        {"name": "Standard Split", "train": 0.8, "val": 0.1, "test": 0.1},
        {"name": "Small Train", "train": 0.6, "val": 0.2, "test": 0.2},
        {"name": "Large Train", "train": 0.9, "val": 0.05, "test": 0.05}
    ]
    
    base_config = {
        'hdf5_path': 'blob/dataset/processed/pems_bay.h5',
        'sequence_length': 12,
        'prediction_length': 1,
        'selected_vdids': None,
        'selected_features': None,
        'normalize': True,
        'normalization_method': 'standard',
        'fill_missing': 'interpolate',
        'stride': 1,
        'batch_size': 8,
        'num_workers': 0,
        'pin_memory': False
    }
    
    for test_config in test_configs:
        print(f"\n🔬 Testing {test_config['name']} ({test_config['train']:.0%}/{test_config['val']:.0%}/{test_config['test']:.0%})...")
        
        # Create config without base_config spread to avoid duplicate keys
        config = TrafficDatasetConfig(
            hdf5_path=base_config['hdf5_path'],
            sequence_length=base_config['sequence_length'],
            prediction_length=base_config['prediction_length'],
            selected_vdids=base_config['selected_vdids'],
            selected_features=base_config['selected_features'],
            train_ratio=test_config['train'],
            val_ratio=test_config['val'],
            test_ratio=test_config['test'],
            normalize=base_config['normalize'],
            normalization_method=base_config['normalization_method'],
            fill_missing=base_config['fill_missing'],
            stride=base_config['stride'],
            batch_size=base_config['batch_size'],
            num_workers=base_config['num_workers'],
            pin_memory=base_config['pin_memory']
        )
        
        try:
            # Create datasets
            train_dataset = TrafficTimeSeries(config, split='train')
            train_scaler = train_dataset.get_scaler()
            
            val_dataset = TrafficTimeSeries(config, split='val', scaler=train_scaler)
            test_dataset = TrafficTimeSeries(config, split='test', scaler=train_scaler)
            
            # Check scaler consistency
            scaler_consistent = (
                id(train_scaler) == id(val_dataset.get_scaler()) == 
                id(test_dataset.get_scaler())
            )
            
            # Get sample data from each split
            train_sample = train_dataset[0]['target_seq']  # [pred_len, num_vds, features]
            val_sample = val_dataset[0]['target_seq'] 
            test_sample = test_dataset[0]['target_seq']
            
            # Calculate statistics
            train_mean = train_sample.mean().item()
            val_mean = val_sample.mean().item()
            test_mean = test_sample.mean().item()
            
            train_std = train_sample.std().item()
            val_std = val_sample.std().item()
            test_std = test_sample.std().item()
            
            # Check consistency criteria
            max_mean_diff = max(
                abs(train_mean - val_mean),
                abs(train_mean - test_mean),
                abs(val_mean - test_mean)
            )
            
            max_std_diff = max(
                abs(train_std - val_std),
                abs(train_std - test_std),
                abs(val_std - test_std)
            )
            
            test_results = {
                'scaler_consistent': scaler_consistent,
                'all_means_normalized': max(abs(train_mean), abs(val_mean), abs(test_mean)) < 0.5,
                'mean_consistency': max_mean_diff < 0.5,
                'std_consistency': max_std_diff < 0.5
            }
            
            results[test_config['name']] = test_results
            
            status = "✅ PASS" if all(test_results.values()) else "❌ FAIL"
            print(f"  {status} - Max mean diff: {max_mean_diff:.4f}, Max std diff: {max_std_diff:.4f}")
            
            if not all(test_results.values()):
                print(f"    ❌ Failed criteria: {[k for k, v in test_results.items() if not v]}")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results[test_config['name']] = {'error': str(e)}
    
    # Test different normalization methods
    print(f"\n🔬 Testing Different Normalization Methods...")
    
    norm_methods = ['standard', 'minmax']
    for method in norm_methods:
        print(f"\n  Testing {method} normalization...")
        
        config = TrafficDatasetConfig(
            hdf5_path=base_config['hdf5_path'],
            sequence_length=base_config['sequence_length'],
            prediction_length=base_config['prediction_length'],
            selected_vdids=base_config['selected_vdids'],
            selected_features=base_config['selected_features'],
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            normalize=base_config['normalize'],
            normalization_method=method,
            fill_missing=base_config['fill_missing'],
            stride=base_config['stride'],
            batch_size=base_config['batch_size'],
            num_workers=base_config['num_workers'],
            pin_memory=base_config['pin_memory']
        )
        
        try:
            train_dataset = TrafficTimeSeries(config, split='train')
            val_dataset = TrafficTimeSeries(config, split='val', scaler=train_dataset.get_scaler())
            
            train_sample = train_dataset[0]['target_seq']
            val_sample = val_dataset[0]['target_seq']
            
            mean_diff = abs(train_sample.mean().item() - val_sample.mean().item())
            
            success = mean_diff < 0.5
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"    {status} - Mean diff: {mean_diff:.6f}")
            
            results[f'{method}_normalization'] = success
            
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            results[f'{method}_normalization'] = False
    
    # Summary
    print(f"\n📊 Test Summary:")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if 
                      (isinstance(result, dict) and all(result.values()) and 'error' not in result) or 
                      (isinstance(result, bool) and result))
    
    for test_name, result in results.items():
        if isinstance(result, dict):
            if 'error' in result:
                print(f"  {test_name}: ❌ ERROR")
            else:
                status = "✅ PASS" if all(result.values()) else "❌ FAIL"
                print(f"  {test_name}: {status}")
        else:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
    
    success_rate = passed_tests / total_tests * 100
    print(f"\n🎯 Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT: Scaling fix is robust across all scenarios!")
    elif success_rate >= 75:
        print("✅ GOOD: Scaling fix works well with minor issues")
    else:
        print("❌ NEEDS WORK: Scaling fix has significant issues")
    
    return success_rate >= 90

if __name__ == "__main__":
    success = test_edge_cases()
    exit(0 if success else 1)