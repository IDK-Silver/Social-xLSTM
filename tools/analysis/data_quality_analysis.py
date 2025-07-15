#!/usr/bin/env python3
"""
Data cleaning solution for temporal quality issues.

Based on deep analysis, this script creates a cleaned dataset by:
1. Removing low-quality early data periods
2. Using only the stable, high-quality later periods
3. Creating a balanced train/val split from clean data
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def analyze_data_quality_timeline(features, timestamps, feature_names, vdids):
    """Analyze data quality over time to find stable periods."""
    print(f"🔍 ANALYZING DATA QUALITY TIMELINE")
    print(f"{'='*50}")
    
    chunk_size = 200  # ~3 hours of data
    total_chunks = len(features) // chunk_size
    
    quality_timeline = {}
    
    for vd_idx, vd_name in enumerate(vdids):
        print(f"\n📈 {vd_name}:")
        quality_timeline[vd_name] = {}
        
        for feat_idx, feat_name in enumerate(feature_names[:3]):
            chunk_qualities = []
            
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(features))
                
                chunk_data = features[start_idx:end_idx, vd_idx, feat_idx]
                valid_count = np.sum(np.isfinite(chunk_data) & (chunk_data != 0))
                quality = valid_count / len(chunk_data) * 100
                chunk_qualities.append(quality)
            
            quality_timeline[vd_name][feat_name] = chunk_qualities
            
            # Find stable high-quality region
            stable_threshold = 85  # 85% data quality threshold
            stable_chunks = [i for i, q in enumerate(chunk_qualities) if q >= stable_threshold]
            
            if stable_chunks:
                stable_start = min(stable_chunks)
                stable_end = max(stable_chunks)
                stable_ratio = len(stable_chunks) / len(chunk_qualities) * 100
                
                print(f"   {feat_name:15}: stable region chunks {stable_start}-{stable_end} ({stable_ratio:.1f}% of data)")
            else:
                print(f"   {feat_name:15}: no stable high-quality region found")
    
    return quality_timeline


def find_optimal_data_window(quality_timeline, min_data_quality=85, min_duration_ratio=0.4):
    """Find the optimal time window with consistent high data quality."""
    print(f"\n🎯 FINDING OPTIMAL DATA WINDOW")
    print(f"{'='*40}")
    
    # Find common stable region across all VDs and features
    all_stable_chunks = []
    
    for vd_name, vd_quality in quality_timeline.items():
        for feat_name, chunk_qualities in vd_quality.items():
            stable_chunks = [i for i, q in enumerate(chunk_qualities) if q >= min_data_quality]
            if stable_chunks:
                all_stable_chunks.append((min(stable_chunks), max(stable_chunks)))
    
    if not all_stable_chunks:
        print("❌ No stable high-quality regions found!")
        return None
    
    # Find intersection of all stable regions
    latest_start = max(start for start, _ in all_stable_chunks)
    earliest_end = min(end for _, end in all_stable_chunks)
    
    if latest_start <= earliest_end:
        stable_duration = earliest_end - latest_start + 1
        total_chunks = len(next(iter(next(iter(quality_timeline.values())).values())))
        duration_ratio = stable_duration / total_chunks
        
        print(f"📊 Optimal window found:")
        print(f"   Chunk range: {latest_start} - {earliest_end}")
        print(f"   Duration: {stable_duration} chunks ({duration_ratio:.1%} of total data)")
        
        if duration_ratio >= min_duration_ratio:
            chunk_size = 200  # From analyze_data_quality_timeline
            return {
                'start_sample': latest_start * chunk_size,
                'end_sample': min((earliest_end + 1) * chunk_size, len(quality_timeline)),
                'chunk_start': latest_start,
                'chunk_end': earliest_end,
                'duration_ratio': duration_ratio,
                'chunk_size': chunk_size
            }
        else:
            print(f"⚠️  Window too short ({duration_ratio:.1%} < {min_duration_ratio:.1%})")
            return None
    else:
        print("❌ No overlapping stable region found!")
        return None


def create_cleaned_dataset(features, timestamps, feature_names, vdids, optimal_window, output_path):
    """Create cleaned dataset using only the high-quality time window."""
    print(f"\n🧹 CREATING CLEANED DATASET")
    print(f"{'='*40}")
    
    start_idx = optimal_window['start_sample']
    end_idx = optimal_window['end_sample']
    
    # Extract clean data
    clean_features = features[start_idx:end_idx]
    clean_timestamps = timestamps[start_idx:end_idx]
    
    print(f"📊 Clean dataset:")
    print(f"   Original samples: {len(features)}")
    print(f"   Clean samples: {len(clean_features)}")
    print(f"   Reduction: {(1 - len(clean_features)/len(features))*100:.1f}%")
    
    # Verify data quality in clean dataset
    print(f"\n✅ Clean dataset quality verification:")
    for vd_idx, vd_name in enumerate(vdids):
        print(f"   {vd_name}:")
        for feat_idx, feat_name in enumerate(feature_names[:3]):
            feat_data = clean_features[:, vd_idx, feat_idx]
            valid_count = np.sum(np.isfinite(feat_data) & (feat_data != 0))
            quality = valid_count / len(feat_data) * 100
            print(f"      {feat_name:15}: {quality:5.1f}% valid")
    
    # Save cleaned dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as h5file:
        # Save data
        data_group = h5file.create_group('data')
        data_group.create_dataset('features', data=clean_features)
        
        # Save metadata
        metadata_group = h5file.create_group('metadata')
        metadata_group.create_dataset('feature_names', data=[name.encode() for name in feature_names])
        metadata_group.create_dataset('timestamps', data=[ts.encode() for ts in clean_timestamps])
        metadata_group.create_dataset('vdids', data=[vd.encode() for vd in vdids])
        
        # Save cleaning info
        cleaning_info = metadata_group.create_group('cleaning_info')
        cleaning_info.attrs['original_samples'] = len(features)
        cleaning_info.attrs['clean_samples'] = len(clean_features)
        cleaning_info.attrs['start_sample'] = start_idx
        cleaning_info.attrs['end_sample'] = end_idx
        cleaning_info.attrs['quality_threshold'] = 85
    
    print(f"💾 Cleaned dataset saved to: {output_path}")
    return output_path


def test_cleaned_data_splitting(clean_h5_path):
    """Test train/val splitting on cleaned data."""
    print(f"\n🧪 TESTING CLEANED DATA SPLITTING")
    print(f"{'='*45}")
    
    with h5py.File(clean_h5_path, 'r') as h5file:
        features = h5file['data/features'][:]
        timestamps = h5file['metadata/timestamps'][:]
        feature_names = h5file['metadata/feature_names'][:]
        
        # Convert back to strings
        feature_names = [name.decode() for name in feature_names]
        timestamps = [ts.decode() for ts in timestamps]
    
    # Test 80/20 split on cleaned data
    total_samples = len(features)
    train_end = int(total_samples * 0.8)
    
    train_data = features[:train_end]
    val_data = features[train_end:]
    
    print(f"📊 Clean data split results:")
    print(f"   Total clean samples: {total_samples}")
    print(f"   Train samples: {len(train_data)}")
    print(f"   Val samples: {len(val_data)}")
    
    # Check distribution consistency on cleaned data
    print(f"\n📈 Distribution consistency (primary VD):")
    
    improvements = {}
    
    for feat_idx, feat_name in enumerate(feature_names[:3]):
        train_feat = train_data[:, 0, feat_idx]  # Primary VD
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
            
            improvements[feat_name] = {'mean_diff': mean_diff, 'std_diff': std_diff}
            
            status = "✅" if mean_diff <= 0.15 and std_diff <= 0.15 else "⚠️"
            print(f"   {status} {feat_name:15}: mean_diff={mean_diff:.3f}, std_diff={std_diff:.3f}")
    
    # Overall quality assessment
    all_diffs = []
    for feat_data in improvements.values():
        all_diffs.extend([feat_data['mean_diff'], feat_data['std_diff']])
    
    max_diff = max(all_diffs) if all_diffs else 0
    avg_diff = np.mean(all_diffs) if all_diffs else 0
    passed = all(d <= 0.15 for d in all_diffs)
    
    print(f"\n🎯 Cleaned data quality assessment:")
    print(f"   Max difference: {max_diff:.3f}")
    print(f"   Avg difference: {avg_diff:.3f}")
    print(f"   Quality passed: {'✅ YES' if passed else '❌ NO'}")
    
    return passed, improvements


def create_model_configs_for_clean_data(clean_h5_path, output_dir="cfgs/cleaned"):
    """Create updated model configurations for the cleaned dataset."""
    print(f"\n⚙️  CREATING UPDATED MODEL CONFIGS")
    print(f"{'='*45}")
    
    # Load original config
    original_config_path = "cfgs/snakemake/default.yaml"
    with open(original_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths to use cleaned dataset
    config['h5_file_path'] = clean_h5_path
    
    # Apply model simplifications based on smaller, cleaner dataset
    if 'lstm_config' in config:
        config['lstm_config']['hidden_size'] = 32  # Reduced from 128
        config['lstm_config']['dropout'] = 0.5    # Increased from 0.2
    
    if 'xlstm_config' in config:
        config['xlstm_config']['hidden_size'] = 32  # Reduced from 128
        config['xlstm_config']['dropout'] = 0.5     # Increased regularization
    
    # Adjust training parameters for cleaner, smaller dataset
    config['training']['early_stopping_patience'] = 10  # More conservative
    config['training']['batch_size'] = 16  # Smaller batches for small dataset
    
    # Save cleaned config
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    clean_config_path = f"{output_dir}/cleaned_data.yaml"
    
    with open(clean_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"💾 Updated config saved to: {clean_config_path}")
    print(f"📋 Key changes:")
    print(f"   - Dataset: {clean_h5_path}")
    print(f"   - Hidden size: 128 → 32")
    print(f"   - Dropout: 0.2 → 0.5")
    print(f"   - Batch size: 32 → 16")
    
    return clean_config_path


def main():
    """Main cleaning execution."""
    print(f"🧹 DATA CLEANING FOR OVERFITTING SOLUTION")
    print(f"{'='*80}")
    
    # Input and output paths
    input_h5 = "blob/dataset/pre-processed/h5/traffic_features_default.h5"
    output_h5 = "blob/dataset/pre-processed/h5/traffic_features_cleaned.h5"
    
    # Load original data
    print(f"📂 Loading data from: {input_h5}")
    with h5py.File(input_h5, 'r') as h5file:
        features = h5file['data/features'][:]
        timestamps = h5file['metadata/timestamps'][:]
        feature_names = h5file['metadata/feature_names'][:]
        vdids = h5file['metadata/vdids'][:]
        
        # Convert bytes to strings
        feature_names = [name.decode() if isinstance(name, bytes) else name for name in feature_names]
        vdids = [vd.decode() if isinstance(vd, bytes) else vd for vd in vdids]
        timestamps = [ts.decode() if isinstance(ts, bytes) else ts for ts in timestamps]
    
    # Step 1: Analyze quality timeline
    quality_timeline = analyze_data_quality_timeline(features, timestamps, feature_names, vdids)
    
    # Step 2: Find optimal window
    optimal_window = find_optimal_data_window(quality_timeline)
    
    if optimal_window is None:
        print("❌ Cannot find suitable data window. Consider relaxing quality requirements.")
        return
    
    # Step 3: Create cleaned dataset
    clean_h5_path = create_cleaned_dataset(features, timestamps, feature_names, vdids, 
                                         optimal_window, output_h5)
    
    # Step 4: Test cleaned data
    quality_passed, improvements = test_cleaned_data_splitting(clean_h5_path)
    
    # Step 5: Create updated configs
    if quality_passed:
        clean_config_path = create_model_configs_for_clean_data(clean_h5_path)
        
        print(f"\n🎉 SUCCESS! Data cleaning completed successfully!")
        print(f"✅ Ready for training with improved data quality")
        print(f"\n🚀 Next steps:")
        print(f"   1. Train models using: {clean_config_path}")
        print(f"   2. Compare results with original overfitting models")
        print(f"   3. Monitor for improved train/val consistency")
    else:
        print(f"\n⚠️  Data cleaning improved quality but distribution differences remain.")
        print(f"💡 Consider further model regularization or different time windows.")


if __name__ == "__main__":
    main()