from dataclasses import dataclass


@dataclass
class TrafficFeature:
    """
    Data class representing traffic features for a single vehicle detector (VD).
    
    This class stores common traffic metrics and provides utilities for field name access
    and dictionary conversion, which is useful for data processing and model training.
    
    Attributes:
        avg_speed (float): Average speed of vehicles in km/h or mph
        total_volume (float): Total number of vehicles passed through the detector
        avg_occupancy (float): Average lane occupancy percentage (0-100)
        speed_std (float): Standard deviation of vehicle speeds
        lane_count (float): Number of lanes at this detector location
    """
    avg_speed: float
    total_volume: float
    avg_occupancy: float
    speed_std: float
    lane_count: float
    
    # Class attributes for field names - used for consistent string references
    # This avoids hardcoding strings throughout the codebase and provides autocomplete
    AVG_SPEED = "avg_speed"
    TOTAL_VOLUME = "total_volume"
    AVG_OCCUPANCY = "avg_occupancy"
    SPEED_STD = "speed_std"
    LANE_COUNT = "lane_count"
    
    @classmethod
    def get_field_names(cls):
        """
        Get all field names as a list.
        
        Returns:
            list[str]: List of all field names in the order they are defined
        """
        return [cls.AVG_SPEED, cls.TOTAL_VOLUME, cls.AVG_OCCUPANCY, cls.SPEED_STD, cls.LANE_COUNT]
    
    def to_dict(self):
        """
        Convert to dictionary with field names as keys.
        
        Returns:
            dict[str, float]: Dictionary mapping field names to their values
        """
        return {
            self.AVG_SPEED: self.avg_speed,
            self.TOTAL_VOLUME: self.total_volume,
            self.AVG_OCCUPANCY: self.avg_occupancy,
            self.SPEED_STD: self.speed_std,
            self.LANE_COUNT: self.lane_count
        }

