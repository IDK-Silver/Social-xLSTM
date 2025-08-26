# PEMS-BAY Data Conversion

Convert PEMS-BAY CSV data to hierarchical HDF5 format compatible with Social-xLSTM.

## Usage

```bash
python scripts/dataset/pre_process/pems_bay/convert_pems_bay_to_hdf5.py \
    --data-csv blob/dataset/raw/PEMS-BAY/PEMS-BAY.csv \
    --meta-csv blob/dataset/raw/PEMS-BAY/PEMS-BAY-META.csv \
    --output-h5 blob/dataset/processed/pems_bay.h5 \
    --validate
```

## Input Files

- **PEMS-BAY.csv**: Traffic speed data (52,117×326)
  - Format: [timestamp, sensor1_speed, sensor2_speed, ...]
  - Units: mph (converted to km/h)
  
- **PEMS-BAY-META.csv**: Sensor metadata (325×18)  
  - Fields: sensor_id, Lanes, Length, Latitude, Longitude, Dir, ...

## Output

**Hierarchical HDF5 Structure**:
```
/data/
  └── features: [52116, 325, 6] float32, gzip compressed
/metadata/
  ├── vdids: [325] string, sensor IDs
  ├── timestamps: [52116] int64, Unix epoch seconds
  ├── feature_names: [6] string, feature names
  ├── frequency: "5min"  
  ├── units: JSON format units definition
  └── source: "PEMS-BAY 2017-01 to 2017-06"
```

**6 Features**:
1. **avg_speed**: Speed (mph → km/h, ×1.609344)
2. **lanes**: Number of lanes (broadcast from META)
3. **length**: Sensor length in miles (broadcast from META) 
4. **latitude**: Latitude coordinate (broadcast from META)
5. **longitude**: Longitude coordinate (broadcast from META)
6. **direction**: Traffic direction N/S/E/W → 0/180/90/270 degrees (broadcast from META)

## Data Quality

**PEMS-BAY has excellent data quality**:
- ✅ Zero missing values in speed data
- ✅ Complete metadata for all 325 sensors
- ✅ Continuous timestamps (5-min intervals)  
- ✅ Reasonable speed range (0-85.1 mph)
- ⚠️ 521 zero speeds (likely traffic congestion - preserved)

## Missing Value Handling

**Current Strategy**: Preserve NaN values, do not handle in conversion script.

Missing values (if any) are handled downstream by:
- TrafficDataModule with `fill_missing: "interpolate"`  
- Or separate preprocessing scripts if needed

## Compatibility

Output HDF5 is fully compatible with:
- `TrafficHDF5Reader` - reads hierarchical structure
- `TrafficTimeSeries` - provides [T,N,F] tensors
- `DistributedCollator` - converts to {"VD_ID": [B,T,F]} format
- Existing Social-xLSTM training pipeline

## Validation

Run with `--validate` to check:
- ✅ Required HDF5 structure paths
- ✅ Dimension consistency  
- ✅ Timestamp monotonicity
- ✅ Reasonable data ranges
- ✅ Feature statistics summary