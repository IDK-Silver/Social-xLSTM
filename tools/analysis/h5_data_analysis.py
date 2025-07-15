#!/usr/bin/env python3
"""
Corrected data analysis for the actual H5 format
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

def analyze_actual_h5_data():
    h5_path = "blob/dataset/pre-processed/h5/traffic_features_default.h5"
    
    print(f"🔍 Analyzing actual H5 data format")
    
    with h5py.File(h5_path, 'r') as h5file:
        # Get the actual data
        features = h5file['data/features'][:]  # shape: (4267, 3, 5)
        timestamps = h5file['metadata/timestamps'][:]
        feature_names = h5file['metadata/feature_names'][:]
        vdids = h5file['metadata/vdids'][:]
        
        print(f"📊 Data shape: {features.shape}")
        print(f"📊 Feature names: {[name.decode() if isinstance(name, bytes) else name for name in feature_names]}")
        print(f"📊 VD IDs: {[vd.decode() if isinstance(vd, bytes) else vd for vd in vdids]}")
        print(f"📊 Time samples: {len(timestamps)}")
        
        # Analyze data quality for each VD
        print(f"\n🔍 Data Quality Analysis:")
        for vd_idx in range(features.shape[1]):  # 3 VDs
            vd_name = vdids[vd_idx].decode() if isinstance(vdids[vd_idx], bytes) else vdids[vd_idx]
            print(f"\n📈 VD {vd_idx}: {vd_name}")
            
            vd_data = features[:, vd_idx, :]  # shape: (4267, 5)
            
            for feat_idx in range(features.shape[2]):  # 5 features
                feat_name = feature_names[feat_idx].decode() if isinstance(feature_names[feat_idx], bytes) else feature_names[feat_idx]
                feat_data = vd_data[:, feat_idx]
                
                # Check for missing data (NaN or very abnormal values)
                valid_mask = ~np.isnan(feat_data) & (feat_data != 0) & np.isfinite(feat_data)
                valid_count = np.sum(valid_mask)
                total_count = len(feat_data)
                valid_rate = valid_count / total_count * 100
                
                if valid_count > 0:
                    mean_val = np.mean(feat_data[valid_mask])
                    std_val = np.std(feat_data[valid_mask])
                    min_val = np.min(feat_data[valid_mask])
                    max_val = np.max(feat_data[valid_mask])
                    
                    print(f"   {feat_name:15}: {valid_rate:6.1f}% valid, mean={mean_val:8.3f}, std={std_val:8.3f}, range=[{min_val:6.2f}, {max_val:6.2f}]")
                    
                    # Check for suspicious patterns
                    if valid_rate < 50:
                        print(f"      ⚠️  WARNING: Low data quality!")
                    
                    # Check for repeated values (potential data issues)
                    unique_count = len(np.unique(feat_data[valid_mask]))
                    if unique_count < valid_count * 0.1:  # Less than 10% unique values
                        print(f"      ⚠️  WARNING: Too many repeated values ({unique_count} unique out of {valid_count})")
                    
                    # Check for extreme outliers
                    if std_val > 0:
                        outlier_threshold = mean_val + 4 * std_val
                        outliers = np.sum(feat_data[valid_mask] > outlier_threshold)
                        if outliers > valid_count * 0.05:  # More than 5% outliers
                            print(f"      ⚠️  WARNING: Many outliers ({outliers} out of {valid_count})")
                else:
                    print(f"   {feat_name:15}: {valid_rate:6.1f}% valid - NO VALID DATA!")
        
        # Analyze the train/val split implications
        print(f"\n🔄 Train/Val Split Analysis:")
        total_samples = features.shape[0]
        train_size = int(total_samples * 0.8)
        val_size = total_samples - train_size
        
        print(f"📊 Total time samples: {total_samples}")
        print(f"📊 Train samples: {train_size}")
        print(f"📊 Val samples: {val_size}")
        
        # For single VD mode (using VD index 0 as example)
        primary_vd_data = features[:, 0, :]  # First VD
        
        train_data = primary_vd_data[:train_size]
        val_data = primary_vd_data[train_size:]
        
        print(f"\n📈 Primary VD Distribution Comparison (Train vs Val):")
        for feat_idx in range(features.shape[2]):
            feat_name = feature_names[feat_idx].decode() if isinstance(feature_names[feat_idx], bytes) else feature_names[feat_idx]
            
            train_feat = train_data[:, feat_idx]
            val_feat = val_data[:, feat_idx]
            
            # Remove invalid data
            train_valid = train_feat[~np.isnan(train_feat) & np.isfinite(train_feat) & (train_feat != 0)]
            val_valid = val_feat[~np.isnan(val_feat) & np.isfinite(val_feat) & (val_feat != 0)]
            
            if len(train_valid) > 0 and len(val_valid) > 0:
                train_mean = np.mean(train_valid)
                val_mean = np.mean(val_valid)
                train_std = np.std(train_valid)
                val_std = np.std(val_valid)
                
                mean_diff = abs(train_mean - val_mean) / train_mean * 100 if train_mean != 0 else 0
                std_diff = abs(train_std - val_std) / train_std * 100 if train_std != 0 else 0
                
                print(f"   {feat_name:15}: mean_diff={mean_diff:6.1f}%, std_diff={std_diff:6.1f}%")
                
                if mean_diff > 20 or std_diff > 30:
                    print(f"      🚨 CRITICAL: Large distribution shift between train and val!")
            else:
                print(f"   {feat_name:15}: Insufficient valid data for comparison")
        
        # Check for data leakage patterns
        print(f"\n🔍 Data Leakage Detection:")
        
        # Check if there are exact duplicate sequences
        sequence_length = 12  # As used in training
        
        for vd_idx in range(min(2, features.shape[1])):  # Check first 2 VDs
            vd_name = vdids[vd_idx].decode() if isinstance(vdids[vd_idx], bytes) else vdids[vd_idx]
            vd_data = features[:, vd_idx, 0]  # Just check first feature (avg_speed)
            
            duplicates = 0
            total_sequences = len(vd_data) - sequence_length
            
            for i in range(total_sequences - 1):
                seq1 = vd_data[i:i+sequence_length]
                for j in range(i+1, min(i+50, total_sequences)):  # Check next 50 sequences
                    seq2 = vd_data[j:j+sequence_length]
                    if np.allclose(seq1, seq2, rtol=1e-6, equal_nan=True):
                        duplicates += 1
                        break
            
            duplicate_rate = duplicates / total_sequences * 100 if total_sequences > 0 else 0
            print(f"   VD {vd_name}: {duplicate_rate:.2f}% duplicate sequences")
            
            if duplicate_rate > 5:
                print(f"      🚨 WARNING: High duplicate rate suggests data issues!")
        
        return features, timestamps, feature_names, vdids

def create_data_quality_plots():
    """Create plots to visualize data quality issues"""
    features, timestamps, feature_names, vdids = analyze_actual_h5_data()
    
    # Create time series plots for first VD
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    primary_vd_data = features[:, 0, :]  # First VD
    feature_indices = [0, 1, 2]  # avg_speed, total_volume, avg_occupancy
    
    for i, feat_idx in enumerate(feature_indices):
        feat_name = feature_names[feat_idx].decode() if isinstance(feature_names[feat_idx], bytes) else feature_names[feat_idx]
        feat_data = primary_vd_data[:, feat_idx]
        
        # Sample data for visualization
        sample_indices = np.arange(0, len(feat_data), max(1, len(feat_data)//1000))
        sample_data = feat_data[sample_indices]
        
        axes[i].plot(sample_indices, sample_data, alpha=0.7, linewidth=0.5)
        axes[i].set_title(f'{feat_name} - Primary VD')
        axes[i].set_ylabel(feat_name)
        axes[i].grid(True, alpha=0.3)
        
        # Highlight potential issues
        valid_mask = ~np.isnan(sample_data) & np.isfinite(sample_data) & (sample_data != 0)
        if not np.all(valid_mask):
            invalid_indices = sample_indices[~valid_mask]
            axes[i].scatter(invalid_indices, np.zeros(len(invalid_indices)), 
                          c='red', s=2, alpha=0.8, label='Invalid/Missing')
            axes[i].legend()
    
    axes[-1].set_xlabel('Time Index')
    plt.tight_layout()
    plt.savefig('blob/debug/data_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Data quality plot saved to blob/debug/data_quality_analysis.png")

if __name__ == "__main__":
    analyze_actual_h5_data()
    create_data_quality_plots()