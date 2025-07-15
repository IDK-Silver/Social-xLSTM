#!/usr/bin/env python3
"""
Deep analysis of the temporal patterns in the traffic data.
This script investigates why temporal splitting doesn't improve distribution consistency.
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_and_analyze_temporal_patterns(h5_path: str):
    """Analyze temporal patterns in the data."""
    print(f"🔍 DEEP TEMPORAL PATTERN ANALYSIS")
    print(f"{'='*60}")
    
    with h5py.File(h5_path, 'r') as h5file:
        features = h5file['data/features'][:]  # shape: (4267, 3, 5)
        timestamps = h5file['metadata/timestamps'][:]
        feature_names = h5file['metadata/feature_names'][:]
        vdids = h5file['metadata/vdids'][:]
        
        # Convert bytes to strings
        feature_names = [name.decode() if isinstance(name, bytes) else name for name in feature_names]
        vdids = [vd.decode() if isinstance(vd, bytes) else vd for vd in vdids]
        timestamps = [ts.decode() if isinstance(ts, bytes) else ts for ts in timestamps]
    
    print(f"📊 Data Shape: {features.shape}")
    print(f"📅 Time Range: {timestamps[0]} -> {timestamps[-1]}")
    
    # 1. Analyze data completeness over time
    print(f"\n1️⃣ DATA COMPLETENESS OVER TIME:")
    
    for vd_idx, vd_name in enumerate(vdids):
        print(f"\n📈 {vd_name}:")
        
        for feat_idx, feat_name in enumerate(feature_names[:3]):  # First 3 features
            vd_data = features[:, vd_idx, feat_idx]
            
            # Calculate completeness in chunks
            chunk_size = 500  # ~8 hours of data (assuming 1min intervals)
            completeness_over_time = []
            time_chunks = []
            
            for i in range(0, len(vd_data), chunk_size):
                chunk = vd_data[i:i+chunk_size]
                valid_count = np.sum(np.isfinite(chunk) & (chunk != 0))
                completeness = valid_count / len(chunk) * 100
                completeness_over_time.append(completeness)
                time_chunks.append(i // chunk_size)
            
            # Check for patterns
            completeness_array = np.array(completeness_over_time)
            mean_completeness = np.mean(completeness_array)
            std_completeness = np.std(completeness_array)
            
            print(f"   {feat_name:15}: mean={mean_completeness:5.1f}%, std={std_completeness:5.1f}%")
            
            # Check if early data is different from late data
            if len(completeness_over_time) >= 4:
                early_comp = np.mean(completeness_over_time[:len(completeness_over_time)//2])
                late_comp = np.mean(completeness_over_time[len(completeness_over_time)//2:])
                diff = abs(early_comp - late_comp)
                
                if diff > 20:
                    print(f"      ⚠️  Temporal quality shift: early={early_comp:.1f}%, late={late_comp:.1f}% (diff={diff:.1f}%)")
    
    # 2. Analyze value distributions over time
    print(f"\n2️⃣ VALUE DISTRIBUTION DRIFT:")
    
    primary_vd_data = features[:, 0, :]  # Use primary VD
    total_samples = len(primary_vd_data)
    
    # Split into early, middle, late periods
    period_size = total_samples // 3
    periods = {
        'early': primary_vd_data[:period_size],
        'middle': primary_vd_data[period_size:2*period_size],
        'late': primary_vd_data[2*period_size:]
    }
    
    for feat_idx, feat_name in enumerate(feature_names[:3]):
        print(f"\n📊 {feat_name} Distribution Drift:")
        
        period_stats = {}
        for period_name, period_data in periods.items():
            feat_data = period_data[:, feat_idx]
            valid_data = feat_data[np.isfinite(feat_data) & (feat_data != 0)]
            
            if len(valid_data) > 10:
                period_stats[period_name] = {
                    'mean': np.mean(valid_data),
                    'std': np.std(valid_data),
                    'count': len(valid_data)
                }
        
        # Compare periods
        if len(period_stats) >= 2:
            early_mean = period_stats.get('early', {}).get('mean', 0)
            late_mean = period_stats.get('late', {}).get('mean', 0)
            
            if early_mean > 0:
                drift = abs(early_mean - late_mean) / early_mean * 100
                print(f"   Early->Late drift: {drift:.1f}%")
                
                for period_name, stats in period_stats.items():
                    print(f"   {period_name:6}: mean={stats['mean']:7.2f}, std={stats['std']:6.2f}, n={stats['count']:4d}")
                
                if drift > 30:
                    print(f"   🚨 SEVERE TEMPORAL DRIFT DETECTED!")
    
    # 3. Check for systematic patterns
    print(f"\n3️⃣ SYSTEMATIC PATTERN DETECTION:")
    
    # Check if the 80/20 split happens to fall on a pattern boundary
    train_ratio = 0.8
    split_point = int(total_samples * train_ratio)
    
    print(f"📍 Split Analysis (80/20 at index {split_point}):")
    
    # Look at data around the split point
    window = 100  # Look at 100 samples before and after split
    pre_split = primary_vd_data[max(0, split_point-window):split_point]
    post_split = primary_vd_data[split_point:min(total_samples, split_point+window)]
    
    for feat_idx, feat_name in enumerate(feature_names[:3]):
        pre_data = pre_split[:, feat_idx]
        post_data = post_split[:, feat_idx]
        
        pre_valid = pre_data[np.isfinite(pre_data) & (pre_data != 0)]
        post_valid = post_data[np.isfinite(post_data) & (post_data != 0)]
        
        if len(pre_valid) > 10 and len(post_valid) > 10:
            pre_mean = np.mean(pre_valid)
            post_mean = np.mean(post_valid)
            
            if pre_mean > 0:
                boundary_diff = abs(pre_mean - post_mean) / pre_mean * 100
                print(f"   {feat_name:15}: boundary change = {boundary_diff:.1f}%")
                
                if boundary_diff > 15:
                    print(f"      🚨 Sharp change at split boundary!")
    
    return features, timestamps, feature_names, vdids


def investigate_data_generation_issues(features, timestamps, feature_names, vdids):
    """Investigate potential data generation or collection issues."""
    print(f"\n4️⃣ DATA GENERATION ISSUE INVESTIGATION:")
    print(f"{'='*60}")
    
    # Check for artificial patterns that suggest synthetic or corrupted data
    primary_vd_data = features[:, 0, :]  # Focus on primary VD
    
    for feat_idx, feat_name in enumerate(feature_names[:3]):
        feat_data = primary_vd_data[:, feat_idx]
        valid_data = feat_data[np.isfinite(feat_data) & (feat_data != 0)]
        
        if len(valid_data) > 100:
            print(f"\n🔍 {feat_name} Artificial Pattern Check:")
            
            # Check for too many rounded values
            rounded_values = np.sum(valid_data == np.round(valid_data))
            rounded_ratio = rounded_values / len(valid_data) * 100
            print(f"   Rounded values: {rounded_ratio:.1f}%")
            
            # Check for repeated sequences
            sequence_length = 10
            repeated_count = 0
            total_sequences = len(valid_data) - sequence_length
            
            for i in range(total_sequences - 1):
                seq1 = valid_data[i:i+sequence_length]
                for j in range(i+1, min(i+20, total_sequences)):  # Check next 20 sequences
                    seq2 = valid_data[j:j+sequence_length]
                    if np.allclose(seq1, seq2, rtol=1e-6):
                        repeated_count += 1
                        break
            
            repeat_rate = repeated_count / total_sequences * 100
            print(f"   Repeated sequences: {repeat_rate:.1f}%")
            
            # Check for sudden jumps (potential data switching)
            diffs = np.diff(valid_data)
            mean_diff = np.mean(np.abs(diffs))
            large_jumps = np.sum(np.abs(diffs) > 5 * mean_diff)
            jump_rate = large_jumps / len(diffs) * 100
            print(f"   Large jumps: {jump_rate:.1f}%")
            
            # Overall assessment
            issues = []
            if rounded_ratio > 80:
                issues.append("too many rounded values")
            if repeat_rate > 10:
                issues.append("excessive repeated sequences")
            if jump_rate > 5:
                issues.append("too many large jumps")
            
            if issues:
                print(f"   🚨 Issues detected: {', '.join(issues)}")
            else:
                print(f"   ✅ No obvious artificial patterns")


def create_temporal_analysis_plots(features, timestamps, feature_names, output_dir="blob/debug"):
    """Create detailed temporal analysis plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Data completeness over time
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    chunk_size = 200  # Samples per chunk
    
    for vd_idx in range(min(3, features.shape[1])):
        completeness_data = []
        time_indices = []
        
        for i in range(0, len(features), chunk_size):
            chunk = features[i:i+chunk_size, vd_idx, 0]  # Use avg_speed
            valid_count = np.sum(np.isfinite(chunk) & (chunk != 0))
            completeness = valid_count / len(chunk) * 100
            completeness_data.append(completeness)
            time_indices.append(i)
        
        axes[vd_idx].plot(time_indices, completeness_data, 'o-', alpha=0.7)
        axes[vd_idx].set_title(f'Data Completeness Over Time - VD {vd_idx}')
        axes[vd_idx].set_ylabel('Completeness (%)')
        axes[vd_idx].grid(True, alpha=0.3)
        
        # Mark 80% split point
        split_point = int(len(features) * 0.8)
        axes[vd_idx].axvline(x=split_point, color='red', linestyle='--', alpha=0.7, label='80% split')
        axes[vd_idx].legend()
    
    axes[-1].set_xlabel('Time Index')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/temporal_completeness_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Value distributions in different time periods
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    primary_vd_data = features[:, 0, :]
    total_samples = len(primary_vd_data)
    
    # Define periods
    periods = {
        'Early (0-25%)': primary_vd_data[:total_samples//4],
        'Mid-Early (25-50%)': primary_vd_data[total_samples//4:total_samples//2],
        'Mid-Late (50-75%)': primary_vd_data[total_samples//2:3*total_samples//4],
        'Late (75-100%)': primary_vd_data[3*total_samples//4:]
    }
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for feat_idx in range(min(4, len(feature_names))):
        ax = axes[feat_idx]
        
        for i, (period_name, period_data) in enumerate(periods.items()):
            feat_data = period_data[:, feat_idx]
            valid_data = feat_data[np.isfinite(feat_data) & (feat_data != 0)]
            
            if len(valid_data) > 10:
                ax.hist(valid_data, bins=30, alpha=0.5, label=period_name, 
                       color=colors[i], density=True)
        
        ax.set_title(f'{feature_names[feat_idx]} Distribution by Time Period')
        ax.set_xlabel(feature_names[feat_idx])
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/temporal_distribution_drift.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Temporal analysis plots saved to {output_dir}/")


def main():
    """Main analysis execution."""
    h5_path = "blob/dataset/pre-processed/h5/traffic_features_default.h5"
    
    print(f"🔬 DEEP DATA TEMPORAL ANALYSIS")
    print(f"{'='*80}")
    
    try:
        # Load and analyze data
        features, timestamps, feature_names, vdids = load_and_analyze_temporal_patterns(h5_path)
        
        # Investigate data generation issues
        investigate_data_generation_issues(features, timestamps, feature_names, vdids)
        
        # Create analysis plots
        create_temporal_analysis_plots(features, timestamps, feature_names)
        
        # Conclusions and recommendations
        print(f"\n🎯 CONCLUSIONS & RECOMMENDATIONS:")
        print(f"{'='*60}")
        print(f"1. If temporal drift is > 30%: The data has inherent temporal instability")
        print(f"2. If artificial patterns detected: Consider data cleaning/regeneration")
        print(f"3. If severe drift at split boundary: Try different split ratios")
        print(f"4. If issues persist: Focus on model regularization instead of data splitting")
        
        print(f"\n📊 Next steps:")
        print(f"   1. Review temporal analysis plots in blob/debug/")
        print(f"   2. Based on findings, choose appropriate fix strategy")
        print(f"   3. Consider data cleaning if patterns suggest data quality issues")
        
    except Exception as e:
        print(f"❌ Error in temporal analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()