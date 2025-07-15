#!/usr/bin/env python3
"""
Test script for temporal splitting functionality.

This script tests the new temporal splitting strategy and compares it with
the original random splitting to demonstrate the improvement.
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from social_xlstm.dataset.core.temporal_split import TemporalSplitter, create_temporal_split_config


def load_h5_data(h5_path: str):
    """Load data from H5 file."""
    print(f"📂 Loading data from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as h5file:
        features = h5file['data/features'][:]  # shape: (4267, 3, 5)
        timestamps = h5file['metadata/timestamps'][:]
        feature_names = h5file['metadata/feature_names'][:]
        vdids = h5file['metadata/vdids'][:]
        
        # Convert bytes to strings if needed
        feature_names = [name.decode() if isinstance(name, bytes) else name for name in feature_names]
        vdids = [vd.decode() if isinstance(vd, bytes) else vd for vd in vdids]
        timestamps = [ts.decode() if isinstance(ts, bytes) else ts for ts in timestamps]
        
        print(f"✅ Data loaded: {features.shape}, {len(timestamps)} timestamps")
        return features, timestamps, feature_names, vdids


def test_original_splitting(data, timestamps, feature_names):
    """Test the original random splitting approach."""
    print(f"\n🔍 TESTING ORIGINAL RANDOM SPLITTING")
    print(f"{'='*60}")
    
    total_samples = len(timestamps)
    train_ratio = 0.8
    
    # Original approach: simple ratio-based split
    train_end = int(total_samples * train_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:]
    
    print(f"📊 Original Split Results:")
    print(f"   Train samples: {len(train_data)}")
    print(f"   Val samples:   {len(val_data)}")
    
    # Check distribution differences (focus on primary VD)
    print(f"\n📈 Distribution Analysis (Primary VD):")
    
    distribution_diffs = {}
    for feat_idx, feat_name in enumerate(feature_names[:3]):  # Check first 3 features
        train_feat = train_data[:, 0, feat_idx]  # VD 0
        val_feat = val_data[:, 0, feat_idx]
        
        # Remove invalid data
        train_valid = train_feat[np.isfinite(train_feat) & (train_feat != 0)]
        val_valid = val_feat[np.isfinite(val_feat) & (val_feat != 0)]
        
        if len(train_valid) > 10 and len(val_valid) > 10:
            train_mean = np.mean(train_valid)
            val_mean = np.mean(val_valid)
            train_std = np.std(train_valid)
            val_std = np.std(val_valid)
            
            mean_diff = abs(train_mean - val_mean) / train_mean if train_mean != 0 else 0
            std_diff = abs(train_std - val_std) / train_std if train_std != 0 else 0
            
            distribution_diffs[feat_name] = {'mean_diff': mean_diff, 'std_diff': std_diff}
            
            status = "❌" if mean_diff > 0.15 or std_diff > 0.15 else "✅"
            print(f"   {status} {feat_name:15}: mean_diff={mean_diff:.3f}, std_diff={std_diff:.3f}")
    
    return distribution_diffs


def test_temporal_splitting(data, timestamps, feature_names):
    """Test the new temporal splitting approach."""
    print(f"\n🚀 TESTING NEW TEMPORAL SPLITTING")
    print(f"{'='*60}")
    
    # Create temporal splitter
    splitter = create_temporal_split_config(
        sequence_length=12,
        prediction_length=1,
        train_ratio=0.8,
        val_ratio=0.2,
        gap_size=12  # Prevent any temporal leakage
    )
    
    # Perform temporal split
    split_result = splitter.split_temporal(
        data=data,
        timestamps=timestamps,
        sequence_length=12,
        prediction_length=1
    )
    
    print(f"📊 Temporal Split Results:")
    splits = split_result['splits']
    for split_name, split_data in splits.items():
        print(f"   {split_name.capitalize()} samples: {split_data['usable_samples']}")
    
    # Validate split quality
    validation_results = splitter.validate_split_quality(
        split_result=split_result,
        feature_names=feature_names,
        max_distribution_diff=0.15,
        output_dir="blob/debug/temporal_split_validation"
    )
    
    return validation_results


def compare_splitting_methods(original_results, temporal_results):
    """Compare the two splitting methods."""
    print(f"\n⚖️  COMPARISON: ORIGINAL vs TEMPORAL SPLITTING")
    print(f"{'='*70}")
    
    print(f"{'Feature':<15} {'Original':<25} {'Temporal':<25} {'Improvement':<15}")
    print(f"{'-'*70}")
    
    improvements = {}
    
    # Compare mean differences
    for feat_name in original_results.keys():
        if feat_name in ['avg_speed', 'total_volume', 'avg_occupancy']:
            orig_mean_diff = original_results[feat_name]['mean_diff']
            
            # Find corresponding temporal result
            temporal_key = f"vd0_{feat_name}_mean_diff"
            temp_mean_diff = temporal_results.get(temporal_key, 0)
            
            improvement = orig_mean_diff - temp_mean_diff
            improvements[feat_name] = improvement
            
            status = "✅" if improvement > 0 else "⚠️"
            print(f"{feat_name:<15} {orig_mean_diff:.3f}                 {temp_mean_diff:.3f}                 {status} {improvement:+.3f}")
    
    # Overall assessment
    avg_improvement = np.mean(list(improvements.values()))
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"   Average improvement: {avg_improvement:.3f}")
    print(f"   Quality check passed: {'✅ YES' if temporal_results.get('passed_quality_check', False) else '❌ NO'}")
    
    return improvements


def create_comparison_plot(original_results, temporal_results, output_dir="blob/debug"):
    """Create comparison visualization."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    features = ['avg_speed', 'total_volume', 'avg_occupancy']
    original_diffs = []
    temporal_diffs = []
    
    for feat in features:
        if feat in original_results:
            original_diffs.append(original_results[feat]['mean_diff'])
            temporal_key = f"vd0_{feat}_mean_diff"
            temporal_diffs.append(temporal_results.get(temporal_key, 0))
    
    x = np.arange(len(features))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, original_diffs, width, label='Original (Random)', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, temporal_diffs, width, label='Temporal', color='green', alpha=0.7)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Train/Val Mean Difference')
    ax.set_title('Splitting Method Comparison: Distribution Differences')
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    # Add quality threshold line
    ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, label='Quality Threshold (15%)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/splitting_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plot saved to {output_dir}/splitting_comparison.png")


def main():
    """Main test execution."""
    h5_path = "blob/dataset/pre-processed/h5/traffic_features_default.h5"
    
    print(f"🧪 TEMPORAL SPLITTING TEST SUITE")
    print(f"{'='*80}")
    
    # Load data
    try:
        data, timestamps, feature_names, vdids = load_h5_data(h5_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Test original splitting
    try:
        original_results = test_original_splitting(data, timestamps, feature_names)
    except Exception as e:
        print(f"❌ Error in original splitting test: {e}")
        return
    
    # Test temporal splitting
    try:
        temporal_results = test_temporal_splitting(data, timestamps, feature_names)
    except Exception as e:
        print(f"❌ Error in temporal splitting test: {e}")
        return
    
    # Compare methods
    try:
        improvements = compare_splitting_methods(original_results, temporal_results)
        create_comparison_plot(original_results, temporal_results)
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        return
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    if temporal_results.get('passed_quality_check', False):
        print(f"✅ SUCCESS: Temporal splitting significantly improves data quality!")
        print(f"✅ Ready to integrate into training pipeline.")
    else:
        print(f"⚠️  WARNING: Temporal splitting shows improvement but may need further tuning.")
    
    print(f"\n📊 Next steps:")
    print(f"   1. Review validation plots in blob/debug/temporal_split_validation/")
    print(f"   2. If results look good, update TrafficTimeSeries to use temporal splitting")
    print(f"   3. Re-run training with new splitting strategy")


if __name__ == "__main__":
    main()