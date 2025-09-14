#!/usr/bin/env python3
"""
Convert PEMS-BAY CSV data to hierarchical HDF5 format compatible with Social-xLSTM.

Input:  PEMS-BAY.csv (52,117×326), PEMS-BAY-META.csv (325×18)
Output: Hierarchical HDF5 with data/features [T,N,F] structure

Features (F=6):
- avg_speed: Speed data in mph（不再做 mph→km/h 轉換，原樣保存）
- lanes: Number of lanes (from META, broadcast over time)
- length: Sensor length (from META, broadcast)
- latitude: Latitude coordinate (from META, broadcast)
- longitude: Longitude coordinate (from META, broadcast)
- direction: Traffic direction (N/S/E/W → degrees, broadcast)
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm


def direction_to_degrees(direction):
    """Convert direction string to degrees."""
    direction_map = {
        'N': 0.0, 'S': 180.0, 'E': 90.0, 'W': 270.0,
        'n': 0.0, 's': 180.0, 'e': 90.0, 'w': 270.0
    }
    return direction_map.get(str(direction).strip(), np.nan)


def convert_pems_bay_to_hdf5(data_csv_path, meta_csv_path, output_h5_path, 
                           compress_level=4, chunk_size=1024):
    """
    Convert PEMS-BAY CSV files to hierarchical HDF5 format.
    
    Args:
        data_csv_path: Path to PEMS-BAY.csv
        meta_csv_path: Path to PEMS-BAY-META.csv  
        output_h5_path: Output HDF5 file path
        compress_level: gzip compression level (0-9)
        chunk_size: HDF5 chunk size for time dimension
    """
    print("Loading PEMS-BAY data...")
    
    # Load CSV files
    data_df = pd.read_csv(data_csv_path)
    meta_df = pd.read_csv(meta_csv_path) 
    
    print(f"Data shape: {data_df.shape}")
    print(f"Meta shape: {meta_df.shape}")
    
    # Extract dimensions
    T = len(data_df) - 1  # Skip header: 52,116 time steps
    N = len(data_df.columns) - 1  # Skip time column: 325 sensors
    F = 6  # 6 features
    
    print(f"Output tensor shape: [{T}, {N}, {F}]")
    
    # Initialize feature tensor
    features = np.full((T, N, F), np.nan, dtype=np.float32)
    
    # Extract sensor IDs from data column names (skip first column which is timestamp)
    sensor_cols = data_df.columns[1:]  # Remove 'Unnamed: 0' or timestamp column
    
    # Build sensor ID to index mapping
    print("Building sensor mappings...")
    
    # Extract timestamps (skip header row)
    timestamps_str = data_df.iloc[1:, 0].values  # Skip first row (header)
    
    # Convert timestamps to Unix epoch
    print("Converting timestamps...")
    timestamps = np.array([
        int(datetime.strptime(str(ts).strip(), "%Y-%m-%d %H:%M:%S").timestamp())
        for ts in tqdm(timestamps_str, desc="Timestamps")
    ], dtype=np.int64)
    
    # Process speed data (Feature 0: avg_speed)
    print("Processing speed data (mph, saved as-is)...")
    speed_data = data_df.iloc[1:, 1:].values.astype(np.float32)  # Skip header and timestamp
    features[:, :, 0] = speed_data  # keep mph
    
    # Create VD ID list matching column order
    vdids = [str(col).strip() for col in sensor_cols]
    
    # Process metadata features (Features 1-5: broadcast over time)  
    print("Processing metadata features...")
    meta_df_indexed = meta_df.set_index('sensor_id')
    
    for j, sensor_col in enumerate(tqdm(sensor_cols, desc="Metadata")):
        try:
            sensor_id = int(sensor_col)
            
            if sensor_id in meta_df_indexed.index:
                meta_row = meta_df_indexed.loc[sensor_id]
                
                # Broadcast metadata features across all time steps
                features[:, j, 1] = float(meta_row.get('Lanes', np.nan))
                features[:, j, 2] = float(meta_row.get('Length', np.nan))  
                features[:, j, 3] = float(meta_row.get('Latitude', np.nan))
                features[:, j, 4] = float(meta_row.get('Longitude', np.nan))
                features[:, j, 5] = direction_to_degrees(meta_row.get('Dir', ''))
            else:
                print(f"WARNING: Sensor {sensor_id} not found in metadata")
                # Features 1-5 remain NaN for missing metadata
                
        except (ValueError, TypeError) as e:
            print(f"WARNING: Error processing sensor {sensor_col}: {e}")
            continue
    
    # Define feature names
    feature_names = ['avg_speed', 'lanes', 'length', 'latitude', 'longitude', 'direction']
    
    # Write hierarchical HDF5
    print(f"Writing HDF5 to {output_h5_path}...")
    
    with h5py.File(output_h5_path, 'w') as f:
        # Root attributes
        f.attrs['dataset_name'] = 'pems_bay'
        f.attrs['feature_set'] = 'pems_bay_v1' 
        f.attrs['feature_schema_version'] = '1.0'
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['description'] = 'PEMS-BAY traffic data converted to Social-xLSTM format'
        f.attrs['source_data'] = str(data_csv_path)
        f.attrs['source_meta'] = str(meta_csv_path)
        f.attrs['num_timesteps'] = T
        f.attrs['num_locations'] = N
        f.attrs['num_features'] = F
        
        # Create groups
        data_group = f.create_group('data')
        metadata_group = f.create_group('metadata')
        
        # Write features with chunking and compression
        chunk_t = min(chunk_size, T)
        chunk_n = min(64, N) 
        
        features_dset = data_group.create_dataset(
            'features',
            data=features,
            chunks=(chunk_t, chunk_n, F),
            compression='gzip',
            compression_opts=compress_level,
            shuffle=True
        )
        
        # Write metadata  
        metadata_group.create_dataset('vdids', data=np.array(vdids, dtype='S64'))
        metadata_group.create_dataset('feature_names', data=np.array(feature_names, dtype='S32'))  
        
        # Store both int64 timestamps and string timestamps for compatibility
        metadata_group.create_dataset('timestamps', data=np.array([
            datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps
        ], dtype='S19'))
        metadata_group.create_dataset('timestamps_epoch', data=timestamps, dtype='int64')
        metadata_group.create_dataset('frequency', data=b'5min')
        
        # Units information as JSON string
        units_json = '{"avg_speed": "mph", "lanes": "count", "length": "miles", "latitude": "degrees", "longitude": "degrees", "direction": "degrees"}'
        metadata_group.create_dataset('units', data=units_json.encode('utf-8'))
        metadata_group.create_dataset('source', data=b'PEMS-BAY 2017-01 to 2017-06')

        # Unified spec: write per-VD info with lat/lon to metadata/vd_info/<vdid>
        vd_info_group = metadata_group.create_group('vd_info')
        meta_df_indexed = meta_df.set_index('sensor_id')
        created_count = 0
        for sensor_col in sensor_cols:
            try:
                sensor_id = int(sensor_col)
                vdid = str(sensor_id)
                vd_subgroup = vd_info_group.create_group(vdid)
                if sensor_id in meta_df_indexed.index:
                    meta_row = meta_df_indexed.loc[sensor_id]
                    lat = float(meta_row.get('Latitude', np.nan))
                    lon = float(meta_row.get('Longitude', np.nan))
                    # store minimal required fields
                    vd_subgroup.attrs['position_lat'] = lat
                    vd_subgroup.attrs['position_lon'] = lon
                    # optional extra context (best-effort)
                    if 'Lanes' in meta_row:
                        try:
                            vd_subgroup.attrs['lanes'] = float(meta_row.get('Lanes', np.nan))
                        except Exception:
                            pass
                    if 'Length' in meta_row:
                        try:
                            vd_subgroup.attrs['length'] = float(meta_row.get('Length', np.nan))
                        except Exception:
                            pass
                    if 'Dir' in meta_row:
                        try:
                            vd_subgroup.attrs['direction'] = str(meta_row.get('Dir', '')).strip()
                        except Exception:
                            pass
                created_count += 1
            except Exception:
                # Skip malformed id/row and continue
                continue
        # Optionally store CRS hint
        vd_info_group.attrs['coord_crs'] = 'EPSG:4326'

    print("Conversion completed!")
    return output_h5_path


def validate_hdf5_structure(hdf5_path):
    """Validate the converted HDF5 structure."""
    print(f"Validating HDF5 structure: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check required structure
        required_paths = [
            'data/features',
            'metadata/vdids', 
            'metadata/timestamps',
            'metadata/feature_names'
        ]
        
        for path in required_paths:
            assert path in f, f"Missing required path: {path}"
        
        # Check dimensions
        T, N, F = f['data/features'].shape
        assert len(f['metadata/vdids']) == N, f"VDIDs length mismatch: {len(f['metadata/vdids'])} vs {N}"
        assert len(f['metadata/timestamps']) == T, f"Timestamps length mismatch: {len(f['metadata/timestamps'])} vs {T}"  
        assert len(f['metadata/feature_names']) == F, f"Feature names length mismatch: {len(f['metadata/feature_names'])} vs {F}"
        
        # Check timestamp monotonicity using epoch timestamps
        if 'metadata/timestamps_epoch' in f:
            timestamps_epoch = f['metadata/timestamps_epoch'][:]
            assert np.all(np.diff(timestamps_epoch) > 0), "Timestamps must be strictly increasing"
        else:
            # Fallback for legacy files with string timestamps
            timestamps = f['metadata/timestamps'][:]
            if timestamps.dtype.kind in ['S', 'U']:
                print("Warning: String timestamps detected, skipping monotonicity check")
            else:
                assert np.all(np.diff(timestamps) > 0), "Timestamps must be strictly increasing"
        
        # Check for reasonable data ranges
        features = f['data/features'][:]
        speed_data = features[:, :, 0]  # avg_speed
        valid_speeds = speed_data[~np.isnan(speed_data)]
        if len(valid_speeds) > 0:
            assert valid_speeds.min() >= 0, "Speed cannot be negative"
            assert valid_speeds.max() <= 200, "Speed seems unreasonably high (>200 mph)"
        
        # Report statistics
        feature_names = [s.decode('utf-8') for s in f['metadata/feature_names'][:]]
        print(f"Shape: [{T}, {N}, {F}]")
        print(f"Features: {feature_names}")
        print(f"Time range: {T} steps (5-min intervals)")
        print(f"Sensors: {N} locations")
        
        # Feature statistics
        for i, name in enumerate(feature_names):
            feat_data = features[:, :, i]
            valid_data = feat_data[~np.isnan(feat_data)]
            if len(valid_data) > 0:
                print(f"  {name}: {valid_data.min():.2f} - {valid_data.max():.2f} (valid: {len(valid_data):,})")
            else:
                print(f"  {name}: All NaN")
    
    print("Structure validation passed!")


def main():
    parser = argparse.ArgumentParser(description='Convert PEMS-BAY CSV to hierarchical HDF5')
    parser.add_argument('--data-csv', required=True, help='Path to PEMS-BAY.csv')
    parser.add_argument('--meta-csv', required=True, help='Path to PEMS-BAY-META.csv')  
    parser.add_argument('--output-h5', required=True, help='Output HDF5 file path')
    parser.add_argument('--compress-level', type=int, default=4, help='gzip compression level (0-9)')
    parser.add_argument('--chunk-size', type=int, default=1024, help='HDF5 chunk size for time dimension')
    parser.add_argument('--validate', action='store_true', help='Validate output structure after conversion')
    
    args = parser.parse_args()
    
    # Validate input files
    data_path = Path(args.data_csv)
    meta_path = Path(args.meta_csv)
    output_path = Path(args.output_h5)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta CSV not found: {meta_path}")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert
    result_path = convert_pems_bay_to_hdf5(
        data_path, meta_path, output_path,
        compress_level=args.compress_level,
        chunk_size=args.chunk_size
    )
    
    # Validate if requested
    if args.validate:
        validate_hdf5_structure(result_path)
    
    print(f"Output: {result_path}")


if __name__ == '__main__':
    main()
