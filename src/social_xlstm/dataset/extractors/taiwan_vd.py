"""
Taiwan VD Feature Extractor

Implements feature extraction for Taiwan VD (Vehicle Detector) traffic data.
Handles the specific data structure: VDLiveDetail -> LinkFlows -> Lanes -> Vehicles
"""

from typing import List, Any
import numpy as np
from .base import BaseFeatureExtractor


class TaiwanVDExtractor(BaseFeatureExtractor):
    """Feature extractor for Taiwan VD traffic data."""
    
    # Taiwan VD supports these 5 core traffic features
    SUPPORTED_FEATURES = [
        'avg_speed',      # Average speed across all lanes (km/h)
        'total_volume',   # Total volume across all lanes (vehicles)
        'avg_occupancy',  # Average occupancy across all lanes (%)
        'speed_std',      # Standard deviation of speed (km/h)
        'lane_count'      # Number of lanes
    ]
    
    def __init__(self, dataset_name: str = "taiwan_vd", feature_set: str = "traffic_core_v1"):
        super().__init__(dataset_name, feature_set)
    
    def get_supported_features(self) -> List[str]:
        """Get list of supported features."""
        return self.SUPPORTED_FEATURES.copy()
    
    def validate_feature_names(self, feature_names: List[str]) -> bool:
        """Validate that all requested features are supported."""
        return set(feature_names).issubset(set(self.SUPPORTED_FEATURES))
    
    def extract_features(self, raw_data: Any, feature_names: List[str]) -> List[float]:
        """
        Extract features from Taiwan VD data.
        
        Args:
            raw_data: VDLiveDetail object with LinkFlows -> Lanes -> Vehicles structure
            feature_names: List of feature names to extract
            
        Returns:
            List of feature values in the same order as feature_names
        """
        # Get all lanes from all LinkFlows
        all_lanes = []
        for link_flow in raw_data.LinkFlows:
            all_lanes.extend(link_flow.Lanes)
        
        if not all_lanes:
            return [np.nan] * len(feature_names)
        
        # Use vectorized aggregation for performance
        return self._vectorized_lane_aggregation(all_lanes, feature_names)
    
    def _vectorized_lane_aggregation(self, all_lanes: List[Any], feature_names: List[str]) -> List[float]:
        """Vectorized aggregation of lane features for Taiwan VD data."""
        if not all_lanes:
            return [np.nan] * len(feature_names)
        
        num_lanes = len(all_lanes)
        
        # Pre-allocate arrays for all lane data
        speeds = np.full(num_lanes, np.nan, dtype=np.float32)
        occupancies = np.full(num_lanes, np.nan, dtype=np.float32)
        volumes = np.full(num_lanes, 0.0, dtype=np.float32)
        
        # Vectorized data extraction with Taiwan-specific error filtering
        for i, lane in enumerate(all_lanes):
            # Process speed with Taiwan-specific validation
            speed_val = self._safe_float_conversion(lane.Speed)
            if self._is_valid_taiwan_speed(speed_val):
                speeds[i] = speed_val
            
            # Process occupancy with Taiwan-specific validation
            occupancy_val = self._safe_float_conversion(lane.Occupancy)
            if self._is_valid_taiwan_occupancy(occupancy_val):
                occupancies[i] = occupancy_val
            
            # Process volume (sum all vehicle volumes in lane)
            if hasattr(lane, 'Vehicles') and lane.Vehicles:
                lane_volume = sum(
                    self._safe_float_conversion(v.Volume, 0.0) 
                    for v in lane.Vehicles 
                    if self._is_valid_taiwan_volume(self._safe_float_conversion(v.Volume, 0.0))
                )
                volumes[i] = lane_volume
        
        # Compute requested features
        aggregated = []
        for feature_name in feature_names:
            if feature_name == 'avg_speed':
                valid_speeds = speeds[~np.isnan(speeds)]
                aggregated.append(np.mean(valid_speeds) if len(valid_speeds) > 0 else np.nan)
            
            elif feature_name == 'total_volume':
                aggregated.append(np.sum(volumes))
            
            elif feature_name == 'avg_occupancy':
                valid_occupancies = occupancies[~np.isnan(occupancies)]
                aggregated.append(np.mean(valid_occupancies) if len(valid_occupancies) > 0 else np.nan)
            
            elif feature_name == 'speed_std':
                valid_speeds = speeds[~np.isnan(speeds)]
                aggregated.append(np.std(valid_speeds) if len(valid_speeds) > 1 else np.nan)
            
            elif feature_name == 'lane_count':
                aggregated.append(float(num_lanes))
            
            else:
                # Should not reach here if validation is done properly
                aggregated.append(np.nan)
        
        return aggregated
    
    def _is_valid_taiwan_speed(self, value: float) -> bool:
        """Check if speed value is valid for Taiwan VD data (0-200 km/h range)."""
        return self._is_valid_value(value, min_val=0, max_val=200)
    
    def _is_valid_taiwan_occupancy(self, value: float) -> bool:
        """Check if occupancy value is valid for Taiwan VD data (0-100% range)."""
        return self._is_valid_value(value, min_val=0, max_val=100)
    
    def _is_valid_taiwan_volume(self, value: float) -> bool:
        """Check if volume value is valid for Taiwan VD data (non-negative)."""
        return self._is_valid_value(value, min_val=0)