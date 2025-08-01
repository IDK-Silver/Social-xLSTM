# Data Format Reference

Complete reference for traffic data formats and structures used in Social-xLSTM.

## Overview

The project processes Taiwan traffic department Vehicle Detector (VD) data through multiple format transformations:

```
ZIP Archives → XML Files → JSON Format → HDF5 Format
```

## Raw Data Structure

### Directory Structure
```
blob/dataset/pre-processed/unzip_to_json/
├── 2025-03-18_00-49-00/
│   ├── VDList.json          # VD metadata (coordinates, lane count, etc.)
│   └── VDLiveList.json      # VD real-time traffic data
├── 2025-03-18_00-50-00/
└── ...
```

## JSON Data Structures

### VDList.json (VD Metadata)
```json
{
  "UpdateInfo": {
    "UpdateTime": "2025-03-18T00:46:11+08:00",
    "UpdateInterval": 60,
    "Version": "1.0"
  },
  "VDList": [
    {
      "VDID": "VD-11-0020-002-001",
      "PositionLon": 121.459501,
      "PositionLat": 25.149709,
      "RoadID": "300020",
      "RoadName": "臺2線",
      "LaneNum": 3,
      "Authority": "THB"
    }
  ]
}
```

### VDLiveList.json (Real-time Traffic Data)
```json
{
  "LiveTrafficData": [
    {
      "VDID": "VD-11-0020-002-001",
      "DataCollectTime": "2025-03-18T00:46:11+08:00", 
      "UpdateInterval": 60,
      "AuthorityCode": "THB",
      "LinkFlows": [                    // 📍 Key: Data is here
        {
          "LinkID": "3000200000176F",
          "Lanes": [                    // 📍 Each Link contains multiple Lanes  
            {
              "LaneID": 0,
              "LaneType": 1,
              "Speed": 59.0,            // Speed (km/h)
              "Occupancy": 1.0,         // Occupancy (%)
              "Vehicles": [             // 📍 Vehicle data by type
                {
                  "VehicleType": "L",   // L=Large, S=Small
                  "Volume": 0           // Volume for this vehicle type
                },
                {
                  "VehicleType": "S", 
                  "Volume": 1
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

## Data Access Patterns

### Correct Data Path
```
VD → LinkFlows → [LinkID] → Lanes → [LaneID] → Speed/Occupancy/Vehicles
```

### Code Example
```python
for vd in vd_live_list.LiveTrafficData:
    print(f"VD ID: {vd.VDID}")
    
    for link_flow in vd.LinkFlows:
        print(f"  Link ID: {link_flow.LinkID}")
        
        for lane in link_flow.Lanes:
            print(f"    Lane {lane.LaneID}: Speed={lane.Speed}, Occ={lane.Occupancy}")
            
            for vehicle in lane.Vehicles:
                print(f"      {vehicle.VehicleType}: {vehicle.Volume} vehicles")
```

## Error Codes and Handling

### Common Error Codes
| Error Code | Meaning | Handling |
|------------|---------|----------|
| `-99` | Sensor failure or missing data | Convert to NaN |
| `-1` | Initialization error or unknown state | Convert to NaN |
| `255` | Value overflow or system error | Convert to NaN |
| `0` | Possible true zero or no vehicles | Keep (needs context) |

### Validation Function
```python
def is_valid_traffic_value(value, feature_type):
    if value is None or np.isnan(value):
        return False
    
    # Error code filtering
    if value in [-99, -1, 255]:
        return False
    
    # Range checking
    if feature_type == 'speed':
        return 0 <= value <= 200      # km/h
    elif feature_type == 'occupancy':
        return 0 <= value <= 100      # %
    elif feature_type == 'volume':
        return value >= 0             # Non-negative
    
    return True
```

## Feature Definitions

### Extracted Traffic Features
| Feature Name | English Name | Calculation | Normal Range | Unit |
|--------------|-------------|-------------|--------------|------|
| 平均速度 | `avg_speed` | Average of all lane speeds | 0-200 | km/h |
| 總流量 | `total_volume` | Sum of all lane volumes (all vehicle types) | ≥0 | vehicles/minute |
| 平均佔有率 | `avg_occupancy` | Average of all lane occupancies | 0-100 | % |
| 速度標準差 | `speed_std` | Standard deviation of lane speeds | ≥0 | km/h |
| 車道數 | `lane_count` | Total effective lanes for this VD | 1-10 | count |

### Aggregation Rules
```python
# Single VD may have multiple LinkFlows, each with multiple Lanes
all_lanes = []
for link_flow in vd.LinkFlows:
    all_lanes.extend(link_flow.Lanes)

# Feature calculations
avg_speed = np.mean([lane.Speed for lane in all_lanes if is_valid(lane.Speed)])
total_volume = np.sum([sum(v.Volume for v in lane.Vehicles) for lane in all_lanes])
avg_occupancy = np.mean([lane.Occupancy for lane in all_lanes if is_valid(lane.Occupancy)])
```

## HDF5 Format Structure

### Data Organization
```python
# HDF5 file structure
{
    'data/features': (timestamps, num_vds, num_features),  # Main feature array
    'data/timestamps': (timestamps,),                      # Timestamp strings
    'data/vd_ids': (num_vds,),                            # VD identifier strings
    'metadata/feature_names': (num_features,),             # Feature name strings
    'metadata/creation_time': scalar,                      # File creation timestamp
    'metadata/source_info': dict                           # Source data information
}
```

### Feature Array Shape
```python
# Typical shape: (4267, 3, 5)
# - 4267 timestamps
# - 3 VDs 
# - 5 features per VD [avg_speed, total_volume, avg_occupancy, speed_std, lane_count]
```

### Reading HDF5 Data
```python
import h5py

with h5py.File('traffic_features.h5', 'r') as f:
    features = f['data/features'][:]      # Shape: (T, N, F)
    timestamps = f['data/timestamps'][:]   # Shape: (T,)
    vd_ids = f['data/vd_ids'][:]          # Shape: (N,)
    feature_names = f['metadata/feature_names'][:]  # Shape: (F,)
```

## Model Input/Output Formats

### Single VD Mode
```python
# Input format
input_shape = [batch_size, sequence_length, num_features]
# Example: [4, 12, 5] - 4 samples, 12 time steps, 5 features

# Output format  
output_shape = [batch_size, prediction_length, num_features]
# Example: [4, 1, 5] - 4 samples, 1 prediction step, 5 features
```

#### Single VD Output Parsing
```python
# Direct access to predictions (no parsing needed)
predictions = model(inputs)  # [4, 1, 5]

# Extract specific feature (e.g., speed prediction)
speed_predictions = predictions[:, :, 1]  # [4, 1] - index 1 corresponds to speed
```

### Multi-VD Mode (Flattened)
```python
# Input format (flattened)
input_shape = [batch_size, sequence_length, num_vds * num_features]
# Example: [4, 12, 15] - 4 samples, 12 time steps, 15 flattened features (3VD × 5features)

# Output format (flattened)
output_shape = [batch_size, prediction_length, num_vds * num_features] 
# Example: [4, 1, 15] - 4 samples, 1 prediction step, 15 flattened features
```

#### Flattening Order
```python
# For 3 VDs with 5 features each, output indices:
[
    # VD_000 features
    VD_000_volume,     # index 0
    VD_000_speed,      # index 1
    VD_000_occupancy,  # index 2
    VD_000_density,    # index 3
    VD_000_flow,       # index 4
    
    # VD_001 features
    VD_001_volume,     # index 5
    VD_001_speed,      # index 6
    VD_001_occupancy,  # index 7
    VD_001_density,    # index 8
    VD_001_flow,       # index 9
    
    # VD_002 features
    VD_002_volume,     # index 10
    VD_002_speed,      # index 11
    VD_002_occupancy,  # index 12
    VD_002_density,    # index 13
    VD_002_flow,       # index 14
]
```

#### Multi-VD Output Parsing

Multi-VD mode requires restructuring flattened output to structured format:

##### Basic Restructuring
```python
# Restructure flattened output to 4D tensor
flat_output = model(inputs)  # [4, 1, 15]
structured = TrafficLSTM.parse_multi_vd_output(
    flat_output, 
    num_vds=3, 
    num_features=5
)
print(structured.shape)  # [4, 1, 3, 5]
```

##### Extract Specific VD
```python
# Extract specific VD prediction (e.g., VD_001)
vd_001_prediction = TrafficLSTM.extract_vd_prediction(structured, vd_index=1)
print(vd_001_prediction.shape)  # [4, 1, 5]

# Extract VD_001 speed prediction
vd_001_speed = vd_001_prediction[:, :, 1]  # [4, 1]
```

##### Complete Prediction and Parsing Workflow
```python
def predict_and_parse_multi_vd(model, inputs, num_vds, num_features):
    """
    Complete multi-VD prediction and parsing workflow
    
    Args:
        model: TrafficLSTM model (multi_vd_mode=True)
        inputs: Input tensor [batch, seq, num_vds * num_features]
        num_vds: Number of VDs
        num_features: Features per VD
    
    Returns:
        Dict[str, torch.Tensor]: Predictions for each VD
    """
    model.eval()
    with torch.no_grad():
        # Model prediction
        flat_output = model(inputs)
        
        # Restructure to structured format
        structured = TrafficLSTM.parse_multi_vd_output(
            flat_output, num_vds, num_features
        )
        
        # Separate predictions for each VD
        vd_predictions = {}
        for vd_idx in range(num_vds):
            vd_pred = TrafficLSTM.extract_vd_prediction(structured, vd_idx)
            vd_predictions[f'VD_{vd_idx:03d}'] = vd_pred
    
    return vd_predictions

# Usage example
vd_results = predict_and_parse_multi_vd(model, inputs, num_vds=3, num_features=5)
print(f"VD_000 prediction shape: {vd_results['VD_000'].shape}")  # [4, 1, 5]
print(f"VD_001 speed prediction: {vd_results['VD_001'][:, :, 1]}")  # [4, 1]
```

### Design Rationale for Flattened Output

#### Technical Reasons
1. **Unified Training Logic**: Different aggregation modes (flatten, attention, pooling) can produce consistent output format
2. **Simplified Loss Calculation**: Flattened predictions and targets can directly compute MSE and other losses
3. **LSTM Architecture Compatibility**: LSTM naturally outputs 2D tensors, flattening avoids extra reshape operations

#### Trade-offs
**Advantages**:
- ✅ Simple unified training process
- ✅ Easy to implement different aggregation strategies
- ✅ Direct loss computation
- ✅ Natural fit with existing LSTM architecture

**Disadvantages**:
- ❌ Requires additional parsing steps
- ❌ Output semantics not intuitive
- ❌ Increases usage complexity

## Data Quality Metrics

### Expected Quality Indicators
| Metric | Normal Range | Description |
|--------|-------------|-------------|
| VD Match Rate | 80-90% | ID matching rate between VDList and VDLiveList |
| Valid Data Ratio | 60-90% | Valid values after error code filtering |
| NaN Ratio | 10-40% | Including error codes and missing data |
| Time Coverage | >95% | Data completeness across continuous time steps |

### Quality Check Example
```python
def check_data_quality(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        features = f['data/features']
        
        # Calculate validity ratio for each feature
        for i, feature_name in enumerate(['avg_speed', 'total_volume', ...]):
            feature_data = features[:, :, i]
            valid_ratio = np.sum(~np.isnan(feature_data)) / feature_data.size
            print(f"{feature_name}: {valid_ratio:.1%} valid")
```

## Common Issues and Solutions

### Q1: Why isn't VD matching 100%?
**A**: Normal behavior due to:
- Some VDs may be offline during certain periods (maintenance, failure)
- VDList contains all registered VDs, but not all operate in real-time
- VDLiveList may include test VDs not yet in official list

### Q2: Why so many NaN values?
**A**: Main causes:
- Error code filtering (-99, 255, etc.)
- Sensor failure or maintenance periods  
- Very low traffic periods (e.g., late night)
- New or decommissioned VDs

### Q3: Is Speed=0 normal?
**A**: Requires context:
- If Volume=0 too: Likely genuine no-traffic state
- If Volume>0 but Speed=0: Possibly traffic jam or sensor anomaly
- Recommend cross-checking with Occupancy

### Q4: How to handle inconsistent lane counts?
**A**: 
- VDList LaneNum is theoretical lane count
- Actual LinkFlows Lanes is currently operating lanes
- Recommend using actual operating lane count as feature

## Data Processing Recommendations

### Pre-training Processing
1. **Error Code Filtering**: Remove -99, -1, 255, etc.
2. **Outlier Detection**: Check values outside reasonable ranges
3. **Missing Value Handling**: Choose interpolation or masking strategy based on model needs
4. **Normalization**: Apply appropriate normalization per feature type
5. **Time Alignment**: Ensure timestamp consistency

### Spatial Analysis Recommendations
- Use VD longitude/latitude coordinates for spatial correlation analysis
- Consider road network structure (VDs with same RoadID may be correlated)
- Note different directions may have different LinkIDs

### Time Series Analysis
- Data update interval typically 1-5 minutes
- Consider traffic cyclical patterns (hourly, daily, weekly)
- Account for holidays and special events

## Related Components

- **H5 Converter**: `src/social_xlstm/dataset/storage/h5_converter.py`
- **Data Reader**: `src/social_xlstm/dataset/storage/h5_reader.py`
- **Data Processor**: `src/social_xlstm/dataset/core/processor.py`
- **Feature Definition**: `src/social_xlstm/dataset/storage/feature.py`