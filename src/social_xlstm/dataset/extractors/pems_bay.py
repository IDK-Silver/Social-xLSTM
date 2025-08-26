"""
PEMS Bay Feature Extractor

Implements feature extraction for PEMS Bay Area traffic data.
Handles flat HDF5 structure created by other team member's converter:
- features: (time, sensors, 6_features) array
- sensors: sensor ID list  
- dates: timestamp list

Features: [speed, lanes, length, latitude, longitude, direction]
"""

from typing import List, Any, Dict
import numpy as np
import h5py
from pathlib import Path
from .base import BaseFeatureExtractor


class PemsBayExtractor(BaseFeatureExtractor):
    """Feature extractor for PEMS Bay traffic data with flat HDF5 format."""
    
    # PEMS Bay supports these 6 features from the flat HDF5 structure
    SUPPORTED_FEATURES = [
        'speed',      # Traffic speed (first feature in array)
        'lanes',      # Number of lanes (static metadata)
        'length',     # Sensor length (static metadata)  
        'latitude',   # Sensor latitude (static metadata)
        'longitude',  # Sensor longitude (static metadata)
        'direction'   # Traffic direction (static metadata, encoded as float)
    ]
    
    def __init__(self, dataset_name: str = "pems_bay", feature_set: str = "pems_bay_v1"):
        super().__init__(dataset_name, feature_set)
    
    def get_supported_features(self) -> List[str]:
        """Get list of supported features."""
        return self.SUPPORTED_FEATURES.copy()
    
    def validate_feature_names(self, feature_names: List[str]) -> bool:
        """Validate that all requested features are supported."""
        return set(feature_names).issubset(set(self.SUPPORTED_FEATURES))
    
    def extract_features(self, raw_data: Any, feature_names: List[str]) -> List[float]:
        """
        Extract features from PEMS Bay data.
        
        Args:
            raw_data: Either a numpy array [6 features] or dict with sensor info
            feature_names: List of feature names to extract
            
        Returns:
            List of feature values in the same order as feature_names
        """
        if isinstance(raw_data, np.ndarray):
            # Direct feature array from HDF5: [speed, lanes, length, lat, lon, direction]
            return self._extract_from_array(raw_data, feature_names)
        elif isinstance(raw_data, dict):
            # Sensor metadata dict
            return self._extract_from_metadata(raw_data, feature_names)
        else:
            # Unknown format, return NaN for all features
            return [np.nan] * len(feature_names)
    
    def _extract_from_array(self, feature_array: np.ndarray, feature_names: List[str]) -> List[float]:
        """Extract features from the 6-element feature array."""
        if len(feature_array) != 6:
            return [np.nan] * len(feature_names)
        
        # Map feature names to array indices
        feature_map = {
            'speed': 0,      # Speed value
            'lanes': 1,      # Number of lanes
            'length': 2,     # Sensor length
            'latitude': 3,   # Latitude coordinate
            'longitude': 4,  # Longitude coordinate  
            'direction': 5   # Direction encoding
        }
        
        extracted = []
        for feature_name in feature_names:
            if feature_name in feature_map:
                idx = feature_map[feature_name]
                value = self._safe_float_conversion(feature_array[idx])
                
                # Apply PEMS-specific validation
                if feature_name == 'speed' and not self._is_valid_pems_speed(value):
                    value = np.nan
                elif feature_name == 'lanes' and not self._is_valid_pems_lanes(value):
                    value = np.nan
                elif feature_name in ['latitude', 'longitude'] and not self._is_valid_pems_coordinate(value):
                    value = np.nan
                    
                extracted.append(value)
            else:
                extracted.append(np.nan)
        
        return extracted
    
    def _extract_from_metadata(self, metadata: Dict[str, Any], feature_names: List[str]) -> List[float]:
        """Extract features from sensor metadata dictionary."""
        extracted = []
        for feature_name in feature_names:
            value = metadata.get(feature_name, np.nan)
            value = self._safe_float_conversion(value)
            
            # Apply feature-specific validation
            if feature_name == 'speed' and not self._is_valid_pems_speed(value):
                value = np.nan
            elif feature_name == 'lanes' and not self._is_valid_pems_lanes(value):
                value = np.nan
            elif feature_name in ['latitude', 'longitude'] and not self._is_valid_pems_coordinate(value):
                value = np.nan
                
            extracted.append(value)
        
        return extracted
    
    def _is_valid_pems_speed(self, value: float) -> bool:
        """Check if speed value is valid for PEMS data (0-120 mph range)."""
        return self._is_valid_value(value, min_val=0, max_val=120)
    
    def _is_valid_pems_lanes(self, value: float) -> bool:
        """Check if lane count is valid for PEMS data (1-10 lanes typical)."""
        return self._is_valid_value(value, min_val=1, max_val=10)
    
    def _is_valid_pems_coordinate(self, value: float) -> bool:
        """Check if coordinate is valid for Bay Area (rough bounds)."""
        # Bay Area approximate bounds: lat 36.5-38.5, lon -123 to -121
        return not (np.isnan(value) or np.isinf(value))


class PemsBayHDF5Adapter:
    """
    Adapter to read PEMS Bay flat HDF5 files and convert to standard format.
    
    This adapter reads the flat HDF5 structure created by the other team member:
    - features: (time, sensors, 6) array
    - sensors: sensor ID array
    - dates: timestamp array
    
    And provides methods to access data in a standardized way.
    """
    
    def __init__(self, hdf5_path: str):
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"PEMS Bay HDF5 file not found: {hdf5_path}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from PEMS Bay HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Read basic structure
            features_shape = f['features'].shape
            sensors = [s.decode('utf-8') if isinstance(s, bytes) else str(s) 
                      for s in f['sensors'][:]]
            dates = [d.decode('utf-8') if isinstance(d, bytes) else str(d) 
                    for d in f['dates'][:]]
            
            return {
                'dataset_name': 'pems_bay',
                'feature_set': 'pems_bay_v1', 
                'feature_schema_version': '1.0',
                'data_shape': features_shape,
                'vdids': sensors,  # Use sensor IDs as VD IDs for compatibility
                'feature_names': ['speed', 'lanes', 'length', 'latitude', 'longitude', 'direction'],
                'timestamps': dates,
                'num_timesteps': len(dates),
                'num_sensors': len(sensors),
                'num_features': 6
            }
    
    def get_sensor_timeseries(self, sensor_id: str) -> np.ndarray:
        """Get complete time series for a specific sensor."""
        metadata = self.get_metadata()
        try:
            sensor_index = metadata['vdids'].index(sensor_id)
            with h5py.File(self.hdf5_path, 'r') as f:
                # Return time series for this sensor: (time, features)
                return f['features'][:, sensor_index, :]
        except ValueError:
            raise ValueError(f"Sensor {sensor_id} not found in dataset")
    
    def get_features_at_timestep(self, timestep: int) -> np.ndarray:
        """Get features for all sensors at a specific timestep."""
        with h5py.File(self.hdf5_path, 'r') as f:
            return f['features'][timestep, :, :]  # (sensors, features)
    
    def convert_to_standard_format(self, output_path: str, 
                                  selected_sensors: List[str] = None) -> None:
        """
        Convert flat PEMS Bay HDF5 to standard hierarchical format.
        
        Args:
            output_path: Path for output HDF5 file
            selected_sensors: Optional list of sensors to include
        """
        from ..storage.h5_reader import TrafficHDF5Reader
        
        metadata = self.get_metadata()
        
        if selected_sensors:
            sensor_indices = [metadata['vdids'].index(s) for s in selected_sensors 
                            if s in metadata['vdids']]
            vdids = selected_sensors
        else:
            sensor_indices = list(range(len(metadata['vdids'])))
            vdids = metadata['vdids']
        
        with h5py.File(self.hdf5_path, 'r') as src, h5py.File(output_path, 'w') as dst:
            # Create standard structure
            metadata_group = dst.create_group('metadata')
            data_group = dst.create_group('data')
            
            # Add standardized metadata
            dst.attrs['dataset_name'] = 'pems_bay'
            dst.attrs['feature_set'] = 'pems_bay_v1'
            dst.attrs['feature_schema_version'] = '1.0'
            dst.attrs['creation_date'] = '2025-01-01T00:00:00'
            
            # Store VD information
            metadata_group.create_dataset(
                'vdids', 
                data=np.array(vdids, dtype=h5py.string_dtype(encoding='utf-8'))
            )
            metadata_group.create_dataset(
                'feature_names',
                data=np.array(metadata['feature_names'], dtype=h5py.string_dtype(encoding='utf-8'))
            )
            metadata_group.create_dataset(
                'timestamps',
                data=np.array(metadata['timestamps'], dtype=h5py.string_dtype(encoding='utf-8'))
            )
            
            # Convert and store feature data
            if selected_sensors:
                features_data = src['features'][:, sensor_indices, :]
            else:
                features_data = src['features'][:]
            
            data_group.create_dataset('features', data=features_data)
            
        print(f"✅ PEMS Bay data converted to standard format: {output_path}")