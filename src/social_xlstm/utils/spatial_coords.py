import math
from typing import Union, Tuple, Optional


class CoordinateSystem:
    """
    A coordinate system class that handles both geographic (lat/lon) and 
    relative planar coordinates with conversion capabilities.
    """
    
    def __init__(self, lat_origin: float = 23.9150, lon_origin: float = 120.6846, radius: float = 6378137):
        """
        Initialize coordinate system with origin point.
        
        Args:
            lat_origin (float): Origin latitude (default: Taiwan Nantou)
            lon_origin (float): Origin longitude (default: Taiwan Nantou)
            radius (float): Earth radius in meters (default: WGS84)
        """
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.radius = radius
        
        # Current coordinate storage
        self._latitude: Optional[float] = None
        self._longitude: Optional[float] = None
        self._x: Optional[float] = None
        self._y: Optional[float] = None
    
    def from_latlon(self, latitude: float, longitude: float) -> 'CoordinateSystem':
        """
        Create coordinate from latitude/longitude.
        
        Args:
            latitude (float): Latitude in degrees
            longitude (float): Longitude in degrees
            
        Returns:
            CoordinateSystem: Self for method chaining
        """
        if latitude is None or longitude is None:
            raise ValueError("Latitude and longitude cannot be None")
        
        self._latitude = latitude
        self._longitude = longitude
        
        # Calculate corresponding x, y coordinates
        self._x, self._y = self._latlon_to_xy(latitude, longitude)
        
        return self
    
    def from_xy(self, x: float, y: float) -> 'CoordinateSystem':
        """
        Create coordinate from relative x/y position.
        
        Args:
            x (float): X coordinate in meters relative to origin
            y (float): Y coordinate in meters relative to origin
            
        Returns:
            CoordinateSystem: Self for method chaining
        """
        if x is None or y is None:
            raise ValueError("X and Y coordinates cannot be None")
        
        self._x = x
        self._y = y
        
        # Calculate corresponding lat, lon coordinates
        self._latitude, self._longitude = self._xy_to_latlon(x, y)
        
        return self
    
    def to_latlon(self) -> Tuple[float, float]:
        """
        Convert to latitude/longitude.
        
        Returns:
            tuple: (latitude, longitude) in degrees
        """
        if self._latitude is None or self._longitude is None:
            raise ValueError("Coordinate not initialized")
        
        return self._latitude, self._longitude
    
    def to_xy(self) -> Tuple[float, float]:
        """
        Convert to relative x/y coordinates.
        
        Returns:
            tuple: (x, y) in meters relative to origin
        """
        if self._x is None or self._y is None:
            raise ValueError("Coordinate not initialized")
        
        return self._x, self._y
    
    def _latlon_to_xy(self, latitude: float, longitude: float) -> Tuple[float, float]:
        """
        Convert lat/lon to x/y using Mercator projection.
        
        Args:
            latitude (float): Latitude in degrees
            longitude (float): Longitude in degrees
            
        Returns:
            tuple: (x, y) in meters
        """
        # Convert to radians
        lat_rad = math.radians(latitude)
        lon_rad = math.radians(longitude)
        lat0_rad = math.radians(self.lat_origin)
        lon0_rad = math.radians(self.lon_origin)
        
        # Mercator projection
        x = self.radius * (lon_rad - lon0_rad)
        y = self.radius * (math.log(math.tan(math.pi / 4 + lat_rad / 2)) -
                          math.log(math.tan(math.pi / 4 + lat0_rad / 2)))
        
        return x, y
    
    def _xy_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert x/y to lat/lon using inverse Mercator projection.
        
        Args:
            x (float): X coordinate in meters
            y (float): Y coordinate in meters
            
        Returns:
            tuple: (latitude, longitude) in degrees
        """
        lat0_rad = math.radians(self.lat_origin)
        lon0_rad = math.radians(self.lon_origin)
        
        # Inverse Mercator projection
        lon_rad = lon0_rad + x / self.radius
        lat_rad = 2 * (math.atan(math.exp(y / self.radius + 
                                         math.log(math.tan(math.pi / 4 + lat0_rad / 2)))) - math.pi / 4)
        
        latitude = math.degrees(lat_rad)
        longitude = math.degrees(lon_rad)
        
        return latitude, longitude
    
    def distance_to(self, other: 'CoordinateSystem') -> float:
        """
        Calculate Euclidean distance to another coordinate in meters.
        
        Args:
            other (CoordinateSystem): Another coordinate system instance
            
        Returns:
            float: Distance in meters
        """
        if self._x is None or self._y is None:
            raise ValueError("Current coordinate not initialized")
        if other._x is None or other._y is None:
            raise ValueError("Other coordinate not initialized")
        
        dx = self._x - other._x
        dy = self._y - other._y
        
        return math.sqrt(dx**2 + dy**2)
    
    def bearing_to(self, other: 'CoordinateSystem') -> float:
        """
        Calculate bearing (angle) to another coordinate in degrees.
        
        Args:
            other (CoordinateSystem): Another coordinate system instance
            
        Returns:
            float: Bearing in degrees (0-360, where 0 is North)
        """
        if self._x is None or self._y is None:
            raise ValueError("Current coordinate not initialized")
        if other._x is None or other._y is None:
            raise ValueError("Other coordinate not initialized")
        
        dx = other._x - self._x
        dy = other._y - self._y
        
        # Calculate angle in radians, then convert to degrees
        angle_rad = math.atan2(dx, dy)  # atan2(x, y) for bearing from North
        angle_deg = math.degrees(angle_rad)
        
        # Normalize to 0-360 degrees
        return (angle_deg + 360) % 360
    
    def move(self, dx: float, dy: float) -> 'CoordinateSystem':
        """
        Move coordinate by given offset.
        
        Args:
            dx (float): X offset in meters
            dy (float): Y offset in meters
            
        Returns:
            CoordinateSystem: New coordinate system instance
        """
        if self._x is None or self._y is None:
            raise ValueError("Coordinate not initialized")
        
        new_coord = CoordinateSystem(self.lat_origin, self.lon_origin, self.radius)
        new_coord.from_xy(self._x + dx, self._y + dy)
        
        return new_coord
    
    def copy(self) -> 'CoordinateSystem':
        """
        Create a copy of this coordinate system.
        
        Returns:
            CoordinateSystem: New instance with same coordinates
        """
        new_coord = CoordinateSystem(self.lat_origin, self.lon_origin, self.radius)
        
        if self._x is not None and self._y is not None:
            new_coord.from_xy(self._x, self._y)
        elif self._latitude is not None and self._longitude is not None:
            new_coord.from_latlon(self._latitude, self._longitude)
        
        return new_coord
    
    @staticmethod
    def calculate_distance_from_latlon(
        lat1: float, lon1: float, 
        lat2: float, lon2: float, 
        lat_origin: float = 23.9150, 
        lon_origin: float = 120.6846
    ) -> float:
        """
        Calculate distance between two geographic points specified by latitude/longitude.
        
        Args:
            lat1 (float): Latitude of first point in degrees (e.g., 25.0330 for Taipei)
            lon1 (float): Longitude of first point in degrees (e.g., 121.5654 for Taipei)
            lat2 (float): Latitude of second point in degrees (e.g., 24.1477 for Taichung)
            lon2 (float): Longitude of second point in degrees (e.g., 120.6736 for Taichung)
            lat_origin (float): Origin latitude for coordinate system (default: Taiwan center)
            lon_origin (float): Origin longitude for coordinate system (default: Taiwan center)
            
        Returns:
            float: Distance in meters
            
        Example:
            >>> # Calculate distance between Taipei and Taichung
            >>> distance = CoordinateSystem.calculate_distance_from_latlon(
            ...     25.0330, 121.5654,  # Taipei (lat, lon)
            ...     24.1477, 120.6736   # Taichung (lat, lon)
            ... )
            >>> print(f"Distance: {distance:.2f} meters")
        """
        coord1 = CoordinateSystem(lat_origin, lon_origin)
        coord1.from_latlon(lat1, lon1)
        
        coord2 = CoordinateSystem(lat_origin, lon_origin)
        coord2.from_latlon(lat2, lon2)
        
        return coord1.distance_to(coord2)
    
    @staticmethod
    def calculate_distance_from_xy(
        x1: float, y1: float, 
        x2: float, y2: float
    ) -> float:
        """
        Calculate distance between two points specified by x/y coordinates.
        
        Args:
            x1 (float): X coordinate of first point in meters (relative to origin)
            y1 (float): Y coordinate of first point in meters (relative to origin)
            x2 (float): X coordinate of second point in meters (relative to origin)
            y2 (float): Y coordinate of second point in meters (relative to origin)
            
        Returns:
            float: Distance in meters
            
        Example:
            >>> # Calculate distance between two points in Cartesian coordinates
            >>> distance = CoordinateSystem.calculate_distance_from_xy(
            ...     1000.0, 2000.0,  # Point 1 (x=1km, y=2km from origin)
            ...     3000.0, 4000.0   # Point 2 (x=3km, y=4km from origin)
            ... )
            >>> print(f"Distance: {distance:.2f} meters")
        """
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx**2 + dy**2)
    
    @classmethod
    def create_from_latlon(
        cls, 
        latitude: float, 
        longitude: float, 
        lat_origin: float = 23.9150, 
        lon_origin: float = 120.6846
    ) -> 'CoordinateSystem':
        """
        Create a new CoordinateSystem instance from latitude/longitude.
        
        Args:
            latitude (float): Latitude in degrees (e.g., 25.0330 for Taipei)
            longitude (float): Longitude in degrees (e.g., 121.5654 for Taipei)
            lat_origin (float): Origin latitude for coordinate system
            lon_origin (float): Origin longitude for coordinate system
            
        Returns:
            CoordinateSystem: New coordinate system instance
            
        Example:
            >>> coord = CoordinateSystem.create_from_latlon(25.0330, 121.5654)
            >>> print(coord)
        """
        coord = cls(lat_origin, lon_origin)
        coord.from_latlon(latitude, longitude)
        return coord
    
    @classmethod
    def create_from_xy(
        cls, 
        x: float, 
        y: float, 
        lat_origin: float = 23.9150, 
        lon_origin: float = 120.6846
    ) -> 'CoordinateSystem':
        """
        Create a new CoordinateSystem instance from x/y coordinates.
        
        Args:
            x (float): X coordinate in meters relative to origin
            y (float): Y coordinate in meters relative to origin
            lat_origin (float): Origin latitude for coordinate system
            lon_origin (float): Origin longitude for coordinate system
            
        Returns:
            CoordinateSystem: New coordinate system instance
            
        Example:
            >>> coord = CoordinateSystem.create_from_xy(1000.0, 2000.0)
            >>> print(coord)
        """
        coord = cls(lat_origin, lon_origin)
        coord.from_xy(x, y)
        return coord
    
    def __str__(self) -> str:
        """String representation of the coordinate."""
        if self._latitude is not None and self._longitude is not None:
            return f"CoordinateSystem(lat={self._latitude:.6f}, lon={self._longitude:.6f}, x={self._x:.2f}, y={self._y:.2f})"
        else:
            return "CoordinateSystem(uninitialized)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
    
    def __eq__(self, other: 'CoordinateSystem') -> bool:
        """Check equality with another coordinate system."""
        if not isinstance(other, CoordinateSystem):
            return False
        
        # Compare with small tolerance for floating point precision
        tolerance = 1e-9
        
        if self._latitude is not None and other._latitude is not None:
            return (abs(self._latitude - other._latitude) < tolerance and
                    abs(self._longitude - other._longitude) < tolerance)
        elif self._x is not None and other._x is not None:
            return (abs(self._x - other._x) < tolerance and
                    abs(self._y - other._y) < tolerance)
        
        return False


# # Usage examples:
# if __name__ == "__main__":
#     # Create coordinate system
#     coord1 = CoordinateSystem()
#     coord1.from_latlon(25.0330, 121.5654)  # Taipei
    
#     coord2 = CoordinateSystem()
#     coord2.from_latlon(24.1477, 120.6736)  # Taichung
    
#     # Get coordinates in different formats
#     print(f"Coord1 lat/lon: {coord1.to_latlon()}")
#     print(f"Coord1 x/y: {coord1.to_xy()}")
    
#     # Calculate distance and bearing
#     distance = coord1.distance_to(coord2)
#     bearing = coord1.bearing_to(coord2)
    
#     print(f"Distance: {distance:.2f} meters")
#     print(f"Bearing: {bearing:.2f} degrees")
    
#     # Move coordinate
#     new_coord = coord1.move(1000, 500)  # Move 1km east, 500m north
#     print(f"Moved coordinate: {new_coord}")

if __name__ == "__main__":
    print("=== CoordinateSystem Usage Examples ===\n")
    
    # Method 1: Direct distance calculation from lat/lon (most convenient)
    print("1. Calculate distance directly from latitude/longitude:")
    distance = CoordinateSystem.calculate_distance_from_latlon(
        25.0330, 121.5654,  # Taipei: lat=25.0330°N, lon=121.5654°E
        24.1477, 120.6736   # Taichung: lat=24.1477°N, lon=120.6736°E
    )
    print(f"   Distance between Taipei and Taichung: {distance:.2f} meters")
    print(f"   Distance: {distance/1000:.2f} kilometers\n")
    
    # Method 2: Direct distance calculation from x/y coordinates
    print("2. Calculate distance directly from x/y coordinates:")
    distance_xy = CoordinateSystem.calculate_distance_from_xy(
        1000.0, 2000.0,  # Point 1: 1km east, 2km north from origin
        3000.0, 4000.0   # Point 2: 3km east, 4km north from origin
    )
    print(f"   Distance between two Cartesian points: {distance_xy:.2f} meters\n")
    
    # Method 3: Create coordinate objects (for more complex operations)
    print("3. Create coordinate objects for complex operations:")
    taipei = CoordinateSystem.create_from_latlon(25.0330, 121.5654)
    taichung = CoordinateSystem.create_from_latlon(24.1477, 120.6736)
    
    print(f"   Taipei: {taipei}")
    print(f"   Taichung: {taichung}")
    
    distance_obj = taipei.distance_to(taichung)
    bearing = taipei.bearing_to(taichung)
    
    print(f"   Distance: {distance_obj:.2f} meters")
    print(f"   Bearing from Taipei to Taichung: {bearing:.2f}°")
    
    # Move Taipei 10km north
    moved_taipei = taipei.move(0, 10000)  # 0m east, 10km north
    print(f"   Taipei moved 10km north: {moved_taipei}")