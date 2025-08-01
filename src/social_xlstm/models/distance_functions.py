"""
Distance Functions and Spatial Weighting Module

This module provides vectorized distance calculations and spatial weighting functions
for the Social Pooling mechanism. It integrates with the existing CoordinateSystem
and supports caching for improved performance.

Author: Social-xLSTM Team
Version: 1.0
"""

import math
import torch
import numpy as np
from typing import Union, Dict, Callable, List, Tuple, Optional
from functools import lru_cache
import hashlib
import warnings

# Import existing coordinate system
from social_xlstm.utils.spatial_coords import CoordinateSystem
from social_xlstm.models.social_pooling_config import SocialPoolingConfig

# Type aliases for clarity
ArrayLike = Union[np.ndarray, torch.Tensor, List[List[float]]]
DistanceFunction = Callable[[torch.Tensor], torch.Tensor]
WeightingFunction = Callable[[torch.Tensor, float], torch.Tensor]


# === Core Distance Calculation Functions ===

def compute_euclidean_distance_matrix(coordinates: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distance matrix for projected coordinates.
    
    Args:
        coordinates: Tensor of shape [N, 2] with (x, y) coordinates in meters
        
    Returns:
        Distance matrix of shape [N, N] in meters
        
    Note:
        Uses torch.cdist for optimal performance
    """
    return torch.cdist(coordinates, coordinates, p=2.0)


def compute_manhattan_distance_matrix(coordinates: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Manhattan distance matrix for projected coordinates.
    
    Args:
        coordinates: Tensor of shape [N, 2] with (x, y) coordinates in meters
        
    Returns:
        Distance matrix of shape [N, N] in meters
    """
    return torch.cdist(coordinates, coordinates, p=1.0)


def compute_haversine_distance_matrix(coordinates: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Haversine distance matrix for geographic coordinates.
    
    Args:
        coordinates: Tensor of shape [N, 2] with (lat, lon) in degrees
        
    Returns:
        Distance matrix of shape [N, N] in meters
        
    Note:
        Implements the Haversine formula for great-circle distances
    """
    # Earth radius in meters (WGS84)
    R = 6378137.0
    
    # Convert to radians
    lat_rad = torch.deg2rad(coordinates[:, 0])  # latitude
    lon_rad = torch.deg2rad(coordinates[:, 1])  # longitude
    
    # Expand dimensions for pairwise computation
    lat1 = lat_rad.unsqueeze(1)  # [N, 1]
    lat2 = lat_rad.unsqueeze(0)  # [1, N]
    lon1 = lon_rad.unsqueeze(1)  # [N, 1]
    lon2 = lon_rad.unsqueeze(0)  # [1, N]
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = (torch.sin(dlat / 2) ** 2 + 
         torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
    
    c = 2 * torch.arcsin(torch.sqrt(torch.clamp(a, 0, 1)))
    
    return R * c


# === Spatial Weighting Functions ===

def compute_gaussian_weights(distances: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Compute Gaussian spatial weights from distances.
    
    Formula: w = exp(-d²/(2σ²)) where σ = radius/3
    
    Args:
        distances: Distance matrix of shape [N, N]
        radius: Pooling radius (used to set σ = radius/3)
        
    Returns:
        Weight matrix of shape [N, N] with values in [0, 1]
    """
    sigma = radius / 3.0
    return torch.exp(-distances.pow(2) / (2 * sigma**2))


def compute_exponential_weights(distances: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Compute exponential decay weights from distances.
    
    Formula: w = exp(-d/λ) where λ = radius
    
    Args:
        distances: Distance matrix of shape [N, N]
        radius: Pooling radius (used as decay parameter λ)
        
    Returns:
        Weight matrix of shape [N, N] with values in [0, 1]
    """
    return torch.exp(-distances / radius)


def compute_linear_weights(distances: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Compute linear decay weights from distances.
    
    Formula: w = max(0, 1 - d/R)
    
    Args:
        distances: Distance matrix of shape [N, N]
        radius: Pooling radius R
        
    Returns:
        Weight matrix of shape [N, N] with values in [0, 1]
    """
    return torch.clamp(1.0 - distances / radius, min=0.0)


def compute_inverse_weights(distances: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Compute inverse distance weights.
    
    Formula: w = 1/(1 + d)
    
    Args:
        distances: Distance matrix of shape [N, N] 
        radius: Pooling radius (unused but kept for API consistency)
        
    Returns:
        Weight matrix of shape [N, N] with values in (0, 1]
        
    Note:
        radius parameter is unused but maintained for consistent API
    """
    return 1.0 / (1.0 + distances)


# === Function Registries ===

DISTANCE_FUNCTION_REGISTRY: Dict[str, DistanceFunction] = {
    "euclidean": compute_euclidean_distance_matrix,
    "manhattan": compute_manhattan_distance_matrix,
    "haversine": compute_haversine_distance_matrix,
}

WEIGHTING_FUNCTION_REGISTRY: Dict[str, WeightingFunction] = {
    "gaussian": compute_gaussian_weights,
    "exponential": compute_exponential_weights,
    "linear": compute_linear_weights,
    "inverse": compute_inverse_weights,
}


# === Batch Coordinate Conversion Utilities ===

def convert_coordinates_batch(
    coordinates: ArrayLike,
    source_format: str,
    target_format: str,
    coord_system: Optional[CoordinateSystem] = None
) -> torch.Tensor:
    """
    Convert batch of coordinates between different formats.
    
    Args:
        coordinates: Input coordinates [N, 2]
        source_format: "latlon" or "xy" 
        target_format: "latlon" or "xy"
        coord_system: CoordinateSystem instance (created if None)
        
    Returns:
        Converted coordinates as torch.Tensor [N, 2]
        
    Raises:
        ValueError: If format conversion is not supported
    """
    if source_format == target_format:
        # No conversion needed
        if isinstance(coordinates, torch.Tensor):
            return coordinates
        return torch.tensor(coordinates, dtype=torch.float32)
    
    # Ensure we have a coordinate system
    if coord_system is None:
        coord_system = CoordinateSystem()
    
    # Convert to numpy for processing with CoordinateSystem
    if isinstance(coordinates, torch.Tensor):
        coords_np = coordinates.cpu().numpy()
    else:
        coords_np = np.array(coordinates)
    
    if coords_np.ndim != 2 or coords_np.shape[1] != 2:
        raise ValueError(f"coordinates must be shape [N, 2], got {coords_np.shape}")
    
    converted_coords = []
    
    if source_format == "latlon" and target_format == "xy":
        # Convert latitude/longitude to x/y
        for lat, lon in coords_np:
            temp_coord = CoordinateSystem(
                coord_system.lat_origin, 
                coord_system.lon_origin, 
                coord_system.radius
            )
            temp_coord.from_latlon(lat, lon)
            x, y = temp_coord.to_xy()
            converted_coords.append([x, y])
            
    elif source_format == "xy" and target_format == "latlon":
        # Convert x/y to latitude/longitude
        for x, y in coords_np:
            temp_coord = CoordinateSystem(
                coord_system.lat_origin,
                coord_system.lon_origin, 
                coord_system.radius
            )
            temp_coord.from_xy(x, y)
            lat, lon = temp_coord.to_latlon()
            converted_coords.append([lat, lon])
    else:
        raise ValueError(f"Unsupported conversion: {source_format} -> {target_format}")
    
    return torch.tensor(converted_coords, dtype=torch.float32)


# === Main Spatial Calculator Class ===

class SpatialCalculator:
    """
    High-level interface for Social Pooling spatial calculations.
    
    This class encapsulates distance calculation, weighting, and caching
    functionality with configuration-driven behavior.
    """
    
    def __init__(self, config: SocialPoolingConfig, coord_system: Optional[CoordinateSystem] = None):
        """
        Initialize spatial calculator.
        
        Args:
            config: Social pooling configuration
            coord_system: Coordinate system instance (created if None)
        """
        self.config = config
        self.coord_system = coord_system or CoordinateSystem()
        
        # Cache for distance matrices (keyed by VD ID hash)
        self._distance_cache: Dict[str, torch.Tensor] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate that configuration specifies valid functions."""
        if self.config.distance_metric not in DISTANCE_FUNCTION_REGISTRY:
            raise ValueError(
                f"Unknown distance_metric '{self.config.distance_metric}'. "
                f"Available: {list(DISTANCE_FUNCTION_REGISTRY.keys())}"
            )
        
        if self.config.weighting_function not in WEIGHTING_FUNCTION_REGISTRY:
            raise ValueError(
                f"Unknown weighting_function '{self.config.weighting_function}'. "
                f"Available: {list(WEIGHTING_FUNCTION_REGISTRY.keys())}"
            )
    
    def _get_cache_key(self, vd_ids: List[str]) -> str:
        """Generate cache key from VD IDs."""
        # Sort to ensure consistent key regardless of order
        sorted_ids = sorted(vd_ids)
        key_string = f"{self.config.distance_metric}:" + ":".join(sorted_ids)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _prepare_coordinates(self, coordinates: ArrayLike) -> torch.Tensor:
        """Convert coordinates to appropriate format for distance calculation."""
        # Ensure tensor format
        if not isinstance(coordinates, torch.Tensor):
            coords_tensor = torch.tensor(coordinates, dtype=torch.float32)
        else:
            coords_tensor = coordinates.float()
        
        # Convert coordinates based on distance metric requirements
        if self.config.distance_metric in ["euclidean", "manhattan"]:
            # These require projected coordinates (x, y) in meters
            if self.config.coordinate_system == "geographic":
                # Convert from (lat, lon) to (x, y)
                return convert_coordinates_batch(
                    coords_tensor, "latlon", "xy", self.coord_system
                )
            else:
                # Already in projected coordinates
                return coords_tensor
                
        elif self.config.distance_metric == "haversine":
            # This requires geographic coordinates (lat, lon) in degrees
            if self.config.coordinate_system == "projected":
                # Convert from (x, y) to (lat, lon)
                return convert_coordinates_batch(
                    coords_tensor, "xy", "latlon", self.coord_system
                )
            else:
                # Already in geographic coordinates
                return coords_tensor
        
        return coords_tensor
    
    def compute_distance_matrix(
        self,
        coordinates: ArrayLike,
        vd_ids: List[str]
    ) -> torch.Tensor:
        """
        Compute pairwise distance matrix with caching.
        
        Args:
            coordinates: Coordinates array [N, 2]
            vd_ids: List of VD identifiers for caching
            
        Returns:
            Distance matrix [N, N] in meters
        """
        # Check cache first
        if self.config.enable_caching:
            cache_key = self._get_cache_key(vd_ids)
            if cache_key in self._distance_cache:
                self._cache_hits += 1
                return self._distance_cache[cache_key]
            self._cache_misses += 1
        
        # Prepare coordinates for the specific distance metric
        prepared_coords = self._prepare_coordinates(coordinates)
        
        # Get distance function and compute matrix
        distance_func = DISTANCE_FUNCTION_REGISTRY[self.config.distance_metric]
        distance_matrix = distance_func(prepared_coords)
        
        # Cache the result if caching is enabled
        if self.config.enable_caching:
            cache_key = self._get_cache_key(vd_ids)
            
            # Implement cache size limit
            if len(self._distance_cache) >= self.config.cache_size:
                # Simple FIFO eviction: remove oldest entry
                oldest_key = next(iter(self._distance_cache))
                del self._distance_cache[oldest_key]
            
            self._distance_cache[cache_key] = distance_matrix
        
        return distance_matrix
    
    def compute_spatial_weights(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial weights from distance matrix.
        
        Args:
            distance_matrix: Pairwise distances [N, N]
            
        Returns:
            Spatial weight matrix [N, N]
        """
        weighting_func = WEIGHTING_FUNCTION_REGISTRY[self.config.weighting_function]
        weights = weighting_func(distance_matrix, self.config.pooling_radius)
        
        # Apply radius threshold (set weights to 0 beyond radius)
        if self.config.weighting_function != "inverse":  # inverse doesn't use radius
            mask = distance_matrix <= self.config.pooling_radius
            weights = weights * mask.float()
        
        return weights
    
    def apply_neighbor_limit(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply max_neighbors constraint by keeping only top-k weights per row.
        
        Args:
            weights: Spatial weight matrix [N, N]
            
        Returns:
            Constrained weight matrix [N, N]
        """
        if self.config.max_neighbors <= 0:
            return weights
        
        # For each node, keep only the top-k neighbors (excluding self)
        N = weights.shape[0]
        constrained_weights = torch.zeros_like(weights)
        
        for i in range(N):
            row_weights = weights[i].clone()
            row_weights[i] = 0  # Exclude self-connection
            
            # Get top-k neighbors
            if self.config.max_neighbors < N - 1:
                _, top_indices = torch.topk(row_weights, self.config.max_neighbors)
                mask = torch.zeros(N, dtype=torch.bool)
                mask[top_indices] = True
                constrained_weights[i] = weights[i] * mask.float()
            else:
                # Keep all neighbors if max_neighbors >= available neighbors
                constrained_weights[i] = weights[i]
                constrained_weights[i, i] = 0  # Still exclude self
        
        return constrained_weights
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get caching statistics for performance monitoring.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(total_requests, 1)
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._distance_cache),
            "max_cache_size": self.config.cache_size,
        }
    
    def clear_cache(self):
        """Clear the distance matrix cache."""
        self._distance_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def __str__(self) -> str:
        """String representation of the calculator."""
        return (
            f"SpatialCalculator("
            f"metric={self.config.distance_metric}, "
            f"weighting={self.config.weighting_function}, "
            f"radius={self.config.pooling_radius}m, "
            f"max_neighbors={self.config.max_neighbors})"
        )


# === Utility Functions ===

def validate_coordinates(coordinates: ArrayLike, expected_format: str = "any") -> bool:
    """
    Validate coordinate array format and values.
    
    Args:
        coordinates: Coordinate array to validate
        expected_format: "latlon", "xy", or "any"
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if isinstance(coordinates, torch.Tensor):
            coords = coordinates.cpu().numpy()
        else:
            coords = np.array(coordinates)
        
        if coords.ndim != 2 or coords.shape[1] != 2:
            return False
        
        if expected_format == "latlon":
            # Check latitude/longitude ranges
            lats, lons = coords[:, 0], coords[:, 1]
            if np.any(np.abs(lats) > 90) or np.any(np.abs(lons) > 180):
                return False
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(coords)):
            return False
        
        return True
        
    except Exception:
        return False


def create_spatial_calculator(
    config: SocialPoolingConfig, 
    coord_system: Optional[CoordinateSystem] = None
) -> SpatialCalculator:
    """
    Factory function to create a spatial calculator.
    
    Args:
        config: Social pooling configuration
        coord_system: Optional coordinate system (created if None)
        
    Returns:
        Configured SpatialCalculator instance
    """
    return SpatialCalculator(config, coord_system)


# Export public interface
__all__ = [
    # Core functions
    "compute_euclidean_distance_matrix",
    "compute_manhattan_distance_matrix", 
    "compute_haversine_distance_matrix",
    "compute_gaussian_weights",
    "compute_exponential_weights",
    "compute_linear_weights",
    "compute_inverse_weights",
    
    # Main class
    "SpatialCalculator",
    
    # Utilities
    "convert_coordinates_batch",
    "validate_coordinates",
    "create_spatial_calculator",
    
    # Registries (for testing/debugging)
    "DISTANCE_FUNCTION_REGISTRY",
    "WEIGHTING_FUNCTION_REGISTRY",
]