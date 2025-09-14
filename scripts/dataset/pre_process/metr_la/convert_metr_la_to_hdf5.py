#!/usr/bin/env python3
"""
Convert METR-LA CSV data to hierarchical HDF5 format compatible with Social-xLSTM.

Input:  metr-la.csv, metr-la_sensor_locations.csv
Output: Hierarchical HDF5 with data/features [T,N,F=3] structure

Features (F=3):
- avg_speed: Speed data (converted mph→km/h)
- latitude: Latitude coordinate (broadcast over time)
- longitude: Longitude coordinate (broadcast over time)
"""

""" 
python scripts/dataset/pre_process/metr_la/convert_metr_la_to_hdf5.py \
  --data-csv blob/dataset/raw/METR-LA/metr-la.csv \
  --meta-csv blob/dataset/raw/METR-LA/metr-la_sensor_locations.csv \
  --output-h5 blob/dataset/processed/metr_la.h5 \
  --validate
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm


def convert_metr_la_to_hdf5(data_csv_path, meta_csv_path, output_h5_path,
                            compress_level=4, chunk_size=1024):
    print("Loading METR-LA data...")
    data_df = pd.read_csv(data_csv_path)
    meta_df = pd.read_csv(meta_csv_path)

    # Extract dimensions
    T = len(data_df) - 1
    N = len(data_df.columns) - 1
    F = 3  # avg_speed, latitude, longitude
    print(f"Output tensor shape: [{T}, {N}, {F}]")

    # Initialize features tensor
    features = np.full((T, N, F), np.nan, dtype=np.float32)

    # Extract sensor IDs
    sensor_cols = data_df.columns[1:]

    # Extract timestamps (skip first row)
    timestamps_str = data_df.iloc[1:, 0].values
    print("Converting timestamps...")
    timestamps = np.array([
        int(datetime.strptime(str(ts).strip(), "%Y-%m-%d %H:%M:%S").timestamp())
        for ts in tqdm(timestamps_str, desc="Timestamps")
    ], dtype=np.int64)

    # Process avg_speed (mph, saved as-is)
    print("Processing speed data (mph, saved as-is)...")
    speed_data = data_df.iloc[1:, 1:].values.astype(np.float32)

    # 計算每一 row 的零值比例
    zero_ratio = (speed_data == 0).sum(axis=1) / speed_data.shape[1]

    # 設定閾值，例如 0.5 → 超過 50% sensor 為 0 才刪掉
    threshold = 0.5
    mask = zero_ratio < threshold

    # 過濾資料
    speed_data = speed_data[mask]
    timestamps = timestamps[mask]
    T = len(speed_data)

    print(f"原始筆數: {data_df.shape[0] - 1}, 保留筆數: {T}, 刪掉筆數: {(~mask).sum()} (threshold={threshold})")

    # 初始化 features tensor（因為 T 更新了）
    features = np.full((T, N, F), np.nan, dtype=np.float32)
    features[:, :, 0] = speed_data  # 保持 mph


    # Broadcast latitude and longitude from metadata
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
        except Exception as e:
            print(f"WARNING: Error processing sensor {sensor_col}: {e}")
            continue

    # Define metadata
    feature_names = ['avg_speed', 'latitude', 'longitude']
    vdids = [str(col).strip() for col in sensor_cols]

    # Write HDF5
    print(f"Writing HDF5 to {output_h5_path}...")
    with h5py.File(output_h5_path, 'w') as f:
        # Root attrs
        f.attrs['dataset_name'] = 'metr_la'
        f.attrs['feature_set'] = 'metr_la_v1'
        f.attrs['feature_schema_version'] = '1.0'
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['description'] = 'METR-LA traffic data converted to Social-xLSTM format'
        f.attrs['source_data'] = str(data_csv_path)
        f.attrs['source_meta'] = str(meta_csv_path)
        f.attrs['num_timesteps'] = T
        f.attrs['num_locations'] = N
        f.attrs['num_features'] = F

        # Groups
        data_group = f.create_group('data')
        metadata_group = f.create_group('metadata')

        # Data: features
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

        # Metadata
        metadata_group.create_dataset('vdids', data=np.array(vdids, dtype='S64'))
        metadata_group.create_dataset('feature_names', data=np.array(feature_names, dtype='S32'))
        metadata_group.create_dataset('timestamps', data=np.array([
            datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps
        ], dtype='S19'))
        metadata_group.create_dataset('timestamps_epoch', data=timestamps, dtype='int64')
        metadata_group.create_dataset('frequency', data=b'5min')
        units_json = '{"avg_speed": "mph", "latitude": "degrees", "longitude": "degrees"}'
        metadata_group.create_dataset('units', data=units_json.encode('utf-8'))
        metadata_group.create_dataset('source', data=b'METR-LA 2012-03 to 2012-06')

        # Per-VD info
        vd_info_group = metadata_group.create_group('vd_info')
        created_count = 0
        for sensor_col in sensor_cols:
            try:
                sensor_id = int(sensor_col)
                vdid = str(sensor_id)
                vd_subgroup = vd_info_group.create_group(vdid)
                if sensor_id in meta_df_indexed.index:
                    meta_row = meta_df_indexed.loc[sensor_id]
                    lat = float(meta_row.get('latitude', np.nan))
                    lon = float(meta_row.get('longitude', np.nan))
                    vd_subgroup.attrs['position_lat'] = lat
                    vd_subgroup.attrs['position_lon'] = lon
                created_count += 1
            except Exception:
                continue
        vd_info_group.attrs['coord_crs'] = 'EPSG:4326'

    print("Conversion completed!")
    return output_h5_path


def validate_hdf5_structure(hdf5_path):
    """Validate HDF5 structure."""
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
        assert len(f['metadata/vdids']) == N
        assert len(f['metadata/timestamps']) == T
        assert len(f['metadata/feature_names']) == F

        timestamps_epoch = f['metadata/timestamps_epoch'][:]
        assert np.all(np.diff(timestamps_epoch) > 0), "Timestamps not strictly increasing"

        # Stats
        features = f['data/features'][:]
        feature_names = [s.decode('utf-8') for s in f['metadata/feature_names'][:]]
        print(f"Shape: [{T}, {N}, {F}]")
        print(f"Features: {feature_names}")
        for i, name in enumerate(feature_names):
            data = features[:, :, i]
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                print(f"  {name}: min={valid_data.min():.2f}, max={valid_data.max():.2f}, valid={len(valid_data)}")
            else:
                print(f"  {name}: All NaN")
    print("Validation passed!")


def main():
    parser = argparse.ArgumentParser(description='Convert METR-LA CSV to hierarchical HDF5')
    parser.add_argument('--data-csv', required=True, help='Path to METR-LA.csv')
    parser.add_argument('--meta-csv', required=True, help='Path to METR-LA_sensor_locations.csv')
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

    result_path = convert_metr_la_to_hdf5(
        data_path, meta_path, output_path,
        compress_level=args.compress_level,
        chunk_size=args.chunk_size
    )

    if args.validate:
        validate_hdf5_structure(result_path)

    print(f"Output: {result_path}")


if __name__ == '__main__':
    main()
