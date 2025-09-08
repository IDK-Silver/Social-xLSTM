#!/usr/bin/env python3
"""
Convert METR-LA CSV data to hierarchical HDF5 format compatible with Social-xLSTM,
but only keeping features: avg_speed, latitude, longitude.

Input:  metr-la.csv (), metr-la_sensors_location.csv (325×18)
Output: Hierarchical HDF5 with data/features [T,N,F=3] structure

Features (F=3):
- avg_speed: Speed data (mph→km/h conversion)
- latitude: Latitude coordinate 
- longitude: Longitude coordinate 
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

"""
python scripts/dataset/pre_process/metr_la/convert_metr_la_to_hdf5.py \
    --data-csv blob/dataset/raw/METR-LA/metr-la.csv \
    --meta-csv blob/dataset/raw/METR-LA/metr-la_sensor_locations.csv \
    --output-h5 blob/dataset/processed/metr_la.h5 \
    --validate 
"""
def convert_metr_la_to_hdf5(data_csv_path, meta_csv_path, output_h5_path, 
                              compress_level=4, chunk_size=1024):
    print("Loading METR-LA data...")
    
    data_df = pd.read_csv(data_csv_path)
    meta_df = pd.read_csv(meta_csv_path)
    
    T = len(data_df) - 1
    N = len(data_df.columns) - 1
    F = 3  # avg_speed, latitude, longitude
    
    print(f"Output tensor shape: [{T}, {N}, {F}]")
    features = np.full((T, N, F), np.nan, dtype=np.float32)
    
    sensor_cols = data_df.columns[1:]
    
    # timestamps
    timestamps_str = data_df.iloc[1:, 0].values
    print("Converting timestamps...")
    timestamps = np.array([
        int(datetime.strptime(str(ts).strip(), "%Y-%m-%d %H:%M:%S").timestamp())
        for ts in tqdm(timestamps_str, desc="Timestamps")
    ], dtype=np.int64)
    
    # avg_speed (Feature 0)
    print("Processing speed data (mph → km/h)...")
    speed_data = data_df.iloc[1:, 1:].values.astype(np.float32)
    # 將 0 值去掉 (設為 NaN)
    print("Before replace, zero count:", np.sum(speed_data == 0))
    speed_data[speed_data == 0] = np.nan
    #print("After replace, zero count:", np.sum(speed_data == 0))
    #print("NaN count:", np.isnan(speed_data).sum())

    # 計算每個 sensor 的平均速度（忽略 NaN）
    col_means = np.nanmean(speed_data, axis=0)

    # 將 NaN 補上對應 sensor 的平均值
    inds = np.where(np.isnan(speed_data))
    speed_data[inds] = np.take(col_means, inds[1])

    # 單位轉換 mph → km/h
    features[:, :, 0] = speed_data * 1.609344
    #看看還有沒有NaN
    print("NaN count after filling:", np.isnan(speed_data).sum())

    # metadata: latitude and longitude (Feature 1-2)
    print("Processing metadata features (latitude, longitude)...")
    meta_df_indexed = meta_df.set_index('sensor_id')
    for j, sensor_col in enumerate(tqdm(sensor_cols, desc="Metadata")):
        try:
            sensor_id = int(sensor_col)
            if sensor_id in meta_df_indexed.index:
                meta_row = meta_df_indexed.loc[sensor_id]
                features[:, j, 1] = float(meta_row.get('latitude', np.nan))
                features[:, j, 2] = float(meta_row.get('longitude', np.nan))
            else:
                print(f"WARNING: Sensor {sensor_id} not found in metadata")
        except (ValueError, TypeError) as e:
            print(f"WARNING: Error processing sensor {sensor_col}: {e}")
            continue
    
    feature_names = ['avg_speed', 'latitude', 'longitude']
    vdids = [str(col).strip() for col in sensor_cols]
    
    print(f"Writing HDF5 to {output_h5_path}...")
    with h5py.File(output_h5_path, 'w') as f:
        f.attrs['dataset_name'] = 'metr_la'
        f.attrs['feature_set'] = 'metr_la_v1_reduced'
        f.attrs['feature_schema_version'] = '1.0'
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['description'] = 'METR-LA traffic data (3 features only) converted to HDF5'
        f.attrs['source_data'] = str(data_csv_path)
        f.attrs['source_meta'] = str(meta_csv_path)
        f.attrs['num_timesteps'] = T
        f.attrs['num_locations'] = N
        f.attrs['num_features'] = F
        
        data_group = f.create_group('data')
        metadata_group = f.create_group('metadata')
        
        chunk_t = min(chunk_size, T)
        chunk_n = min(64, N)
        
        data_group.create_dataset(
            'features',
            data=features,
            chunks=(chunk_t, chunk_n, F),
            compression='gzip',
            compression_opts=compress_level,
            shuffle=True
        )
        
        metadata_group.create_dataset('vdids', data=np.array(vdids, dtype='S64'))
        metadata_group.create_dataset('feature_names', data=np.array(feature_names, dtype='S32'))
        metadata_group.create_dataset('timestamps', data=np.array([
            datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps
        ], dtype='S19'))
        metadata_group.create_dataset('timestamps_epoch', data=timestamps, dtype='int64')
        metadata_group.create_dataset('frequency', data=b'5min')
        units_json = '{"avg_speed": "km/h", "latitude": "degrees", "longitude": "degrees"}'
        metadata_group.create_dataset('units', data=units_json.encode('utf-8'))
        metadata_group.create_dataset('source', data=b'METR-LA 2017-01 to 2017-06')
    
    print("Conversion completed!")
    return output_h5_path

def validate_hdf5_structure(hdf5_path):
    """Validate HDF5 structure and report feature statistics."""
    print(f"Validating HDF5 structure: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        required_paths = [
            'data/features',
            'metadata/vdids',
            'metadata/timestamps',
            'metadata/feature_names'
        ]
        for path in required_paths:
            assert path in f, f"Missing required path: {path}"
        
        T, N, F = f['data/features'].shape
        assert len(f['metadata/vdids']) == N, "VDIDs length mismatch"
        assert len(f['metadata/timestamps']) == T, "Timestamps length mismatch"
        assert len(f['metadata/feature_names']) == F, "Feature names length mismatch"
        
        timestamps_epoch = f['metadata/timestamps_epoch'][:]
        assert np.all(np.diff(timestamps_epoch) > 0), "Timestamps must be strictly increasing"
        
        features = f['data/features'][:]
        feature_names = [s.decode('utf-8') for s in f['metadata/feature_names'][:]]
        
        print(f"Shape: [{T}, {N}, {F}]")
        print(f"Features: {feature_names}")
        print(f"Time range: {T} steps (5-min intervals)")
        print(f"Sensors: {N} locations")
        
        for i, name in enumerate(feature_names):
            data = features[:, :, i]
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                print(f"  {name}: min={valid_data.min():.2f}, max={valid_data.max():.2f}, valid={len(valid_data)}")
            else:
                print(f"  {name}: All NaN")
    
    print("HDF5 validation passed!")

def main():
    parser = argparse.ArgumentParser(description='Convert METR-LA CSV to HDF5 (3 features)')
    parser.add_argument('--data-csv', required=True, help='Path to METR-LA.csv')
    parser.add_argument('--meta-csv', required=True, help='Path to METR-LA_sensor_location.csv')  
    parser.add_argument('--output-h5', required=True, help='Output HDF5 file path')
    parser.add_argument('--compress-level', type=int, default=4, help='gzip compression level (0-9)')
    parser.add_argument('--chunk-size', type=int, default=1024, help='HDF5 chunk size for time dimension')
    parser.add_argument('--validate', action='store_true', help='Validate output structure after conversion')
    args = parser.parse_args()
    
    data_path = Path(args.data_csv)
    meta_path = Path(args.meta_csv)
    output_path = Path(args.output_h5)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta CSV not found: {meta_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    convert_metr_la_to_hdf5(
        data_path, meta_path, output_path,
        compress_level=args.compress_level,
        chunk_size=args.chunk_size
    )

    # Validate if requested
    if args.validate:
        validate_hdf5_structure(output_path)
    
    
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
