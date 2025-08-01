"""
Social Pooling Configuration Module

This module provides configuration management for the Social Pooling mechanism
in Social-xLSTM models. It includes comprehensive validation, type safety,
and preset configurations for different traffic scenarios.

Author: Social-xLSTM Team
Version: 1.0
"""

import json
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Literal, ClassVar, Set, Optional, Union
from enum import Enum, auto

# Type definitions for allowed string values
# Based on mathematical-specifications.md and social-pooling-design.md
DistanceMetricType = Literal["euclidean", "manhattan", "haversine"]
WeightingFunctionType = Literal["gaussian", "exponential", "linear", "inverse"]
AggregationMethodType = Literal["weighted_mean", "weighted_sum", "attention"]


class DistanceMetric(Enum):
    """Enumeration of available distance calculation methods."""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    HAVERSINE = "haversine"


class WeightingFunction(Enum):
    """Enumeration of available spatial weighting functions."""
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    INVERSE = "inverse"


class AggregationMethod(Enum):
    """Enumeration of available feature aggregation methods."""
    WEIGHTED_MEAN = "weighted_mean"
    WEIGHTED_SUM = "weighted_sum"
    ATTENTION = "attention"


class SocialPoolingConfigError(Exception):
    """Base exception for Social Pooling configuration errors."""
    pass


class InvalidRadiusError(SocialPoolingConfigError):
    """Raised when pooling radius is invalid."""
    
    def __init__(self, radius: float):
        super().__init__(
            f"Invalid pooling_radius: {radius}. "
            f"Expected value > 0. "
            f"For urban environments, typical values are 500-1000m. "
            f"For highways, typical values are 1500-3000m."
        )


class InvalidNeighborCountError(SocialPoolingConfigError):
    """Raised when max_neighbors count is invalid."""
    
    def __init__(self, count: int):
        super().__init__(
            f"Invalid max_neighbors: {count}. "
            f"Expected value > 0. "
            f"Typical range is 3-20. Higher values increase computational cost."
        )


@dataclass(frozen=True)
class SocialPoolingConfig:
    """
    Immutable configuration for the Social Pooling mechanism.

    This configuration class manages all hyperparameters for the Social Pooling
    layer with comprehensive validation and provides factory methods for common
    traffic scenarios.

    Attributes:
        pooling_radius (float): Spatial radius in meters to consider neighbors.
            Default: 1000.0m (suitable for mixed traffic scenarios)
        max_neighbors (int): Maximum number of neighbors per node.
            Default: 8 (balanced performance/accuracy)
        distance_metric (str): Method for distance calculation.
            Options: "euclidean", "manhattan", "haversine"
        weighting_function (str): Function to convert distance to weight.
            Options: "gaussian", "exponential", "linear", "inverse"
        aggregation_method (str): Method for feature aggregation.
            Options: "weighted_mean", "weighted_sum", "attention"
        coordinate_system (str): Type of coordinate system used.
            Options: "projected", "geographic"
        enable_caching (bool): Whether to cache distance calculations.
            Default: True (recommended for performance)
        cache_size (int): Maximum number of cached distance matrices.
            Default: 100 (adjust based on memory constraints)

    References:
        - Mathematical specifications: docs/reference/mathematical-specifications.md
        - Design document: docs/explanation/social-pooling-design.md
    """
    
    # Class constants for validation
    _ALLOWED_DISTANCE_METRICS: ClassVar[Set[str]] = {
        "euclidean", "manhattan", "haversine"
    }
    _ALLOWED_WEIGHTING_FUNCTIONS: ClassVar[Set[str]] = {
        "gaussian", "exponential", "linear", "inverse"
    }
    _ALLOWED_AGGREGATION_METHODS: ClassVar[Set[str]] = {
        "weighted_mean", "weighted_sum", "attention"
    }
    _ALLOWED_COORDINATE_SYSTEMS: ClassVar[Set[str]] = {
        "projected", "geographic"
    }
    
    # Core Social Pooling parameters
    pooling_radius: float = 1000.0
    max_neighbors: int = 8
    distance_metric: DistanceMetricType = "euclidean"
    weighting_function: WeightingFunctionType = "gaussian"
    aggregation_method: AggregationMethodType = "weighted_mean"
    
    # Integration parameters
    coordinate_system: str = "projected"
    enable_caching: bool = True
    cache_size: int = 100
    
    # Performance monitoring
    enable_profiling: bool = False
    
    def __post_init__(self):
        """Validates configuration parameters after initialization."""
        
        # Critical parameter validation (raises exceptions)
        if self.pooling_radius <= 0:
            raise InvalidRadiusError(self.pooling_radius)
        
        if self.max_neighbors <= 0:
            raise InvalidNeighborCountError(self.max_neighbors)
        
        if self.cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {self.cache_size}")
        
        # String parameter validation
        self._validate_string_parameter(
            "distance_metric", self.distance_metric, self._ALLOWED_DISTANCE_METRICS
        )
        
        self._validate_string_parameter(
            "weighting_function", self.weighting_function, self._ALLOWED_WEIGHTING_FUNCTIONS
        )
        
        self._validate_string_parameter(
            "aggregation_method", self.aggregation_method, self._ALLOWED_AGGREGATION_METHODS
        )
        
        self._validate_string_parameter(
            "coordinate_system", self.coordinate_system, self._ALLOWED_COORDINATE_SYSTEMS
        )
        
        # Performance warnings (non-blocking)
        self._check_performance_warnings()
        self._check_coordinate_system_compatibility()
    
    def _validate_string_parameter(self, param_name: str, value: str, allowed_values: Set[str]):
        """Validates string parameters against allowed values."""
        if value not in allowed_values:
            raise ValueError(
                f"Invalid {param_name} '{value}'. "
                f"Allowed values are: {sorted(list(allowed_values))}"
            )
    
    def _check_performance_warnings(self):
        """Issues warnings for potentially suboptimal parameter combinations."""
        
        # Range warnings based on mathematical-specifications.md
        if not (100 <= self.pooling_radius <= 5000):
            warnings.warn(
                f"pooling_radius ({self.pooling_radius}m) is outside the typical "
                f"range of 100-5000m. This may impact model performance.",
                UserWarning
            )
        
        if not (3 <= self.max_neighbors <= 20):
            warnings.warn(
                f"max_neighbors ({self.max_neighbors}) is outside the typical "
                f"range of 3-20. Higher values increase computational cost.",
                UserWarning
            )
        
        # Performance-specific warnings
        if self.max_neighbors > 15:
            warnings.warn(
                f"max_neighbors ({self.max_neighbors}) > 15 may impact real-time "
                f"performance. Consider reducing for production deployment.",
                UserWarning
            )
    
    def _check_coordinate_system_compatibility(self):
        """Checks coordinate system and distance metric compatibility."""
        if (self.coordinate_system == "geographic" and 
            self.distance_metric != "haversine"):
            warnings.warn(
                f"Using {self.distance_metric} with geographic coordinates. "
                f"Consider using 'haversine' for accurate distance calculation.",
                UserWarning
            )
    
    @classmethod
    def urban_preset(cls, **overrides: Any) -> "SocialPoolingConfig":
        """
        Creates a configuration preset optimized for urban traffic environments.
        
        Urban characteristics:
        - Smaller pooling radius (dense road network)
        - More neighbors (higher detector density)
        - Gaussian weighting (smooth distance decay)
        
        Args:
            **overrides: Keyword arguments to override preset values.
            
        Returns:
            SocialPoolingConfig instance with urban-optimized settings.
            
        Example:
            >>> config = SocialPoolingConfig.urban_preset(max_neighbors=10)
            >>> assert config.pooling_radius == 500.0
        """
        defaults = {
            "pooling_radius": 500.0,
            "max_neighbors": 12,
            "distance_metric": "euclidean",
            "weighting_function": "gaussian",
            "aggregation_method": "weighted_mean",
        }
        defaults.update(overrides)
        return cls(**defaults)
    
    @classmethod
    def highway_preset(cls, **overrides: Any) -> "SocialPoolingConfig":
        """
        Creates a configuration preset optimized for highway traffic environments.
        
        Highway characteristics:
        - Larger pooling radius (sparser detector network)
        - Fewer neighbors (lower detector density)
        - Exponential weighting (faster distance decay)
        
        Args:
            **overrides: Keyword arguments to override preset values.
            
        Returns:
            SocialPoolingConfig instance with highway-optimized settings.
            
        Example:
            >>> config = SocialPoolingConfig.highway_preset(pooling_radius=3000.0)
            >>> assert config.max_neighbors == 5
        """
        defaults = {
            "pooling_radius": 2000.0,
            "max_neighbors": 5,
            "distance_metric": "euclidean",
            "weighting_function": "exponential",
            "aggregation_method": "weighted_mean",
        }
        defaults.update(overrides)
        return cls(**defaults)
    
    @classmethod
    def mixed_preset(cls, **overrides: Any) -> "SocialPoolingConfig":
        """
        Creates a configuration preset for mixed urban/highway environments.
        
        Mixed characteristics:
        - Moderate pooling radius
        - Balanced neighbor count
        - Linear weighting (balanced spatial influence)
        
        Args:
            **overrides: Keyword arguments to override preset values.
            
        Returns:
            SocialPoolingConfig instance with mixed-environment settings.
        """
        defaults = {
            "pooling_radius": 1200.0,
            "max_neighbors": 8,
            "distance_metric": "euclidean",
            "weighting_function": "linear",
            "aggregation_method": "weighted_mean",
        }
        defaults.update(overrides)
        return cls(**defaults)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SocialPoolingConfig":
        """
        Creates configuration from dictionary with backward compatibility.
        
        Args:
            data: Dictionary containing configuration parameters.
            
        Returns:
            SocialPoolingConfig instance.
            
        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        # Handle legacy parameter names for backward compatibility
        if "pooling_range" in data:
            data["pooling_radius"] = data.pop("pooling_range")
        
        if "neighbor_limit" in data:
            data["max_neighbors"] = data.pop("neighbor_limit")
        
        # Filter out unknown parameters to avoid TypeError
        valid_params = {
            field.name for field in cls.__dataclass_fields__.values()
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_params}
        
        return cls(**filtered_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Serializes configuration to JSON string.
        
        Args:
            indent: JSON indentation level. None for compact format.
            
        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
    
    @classmethod
    def from_json(cls, json_str: str) -> "SocialPoolingConfig":
        """
        Creates configuration from JSON string.
        
        Args:
            json_str: JSON string representation.
            
        Returns:
            SocialPoolingConfig instance.
            
        Raises:
            json.JSONDecodeError: If JSON is invalid.
            ValueError: If configuration parameters are invalid.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def validate_for_scenario(self, scenario: str) -> None:
        """
        Validates configuration for specific deployment scenarios.
        
        Args:
            scenario: Deployment scenario ("real_time", "batch", "research").
            
        Raises:
            ValueError: If scenario is unknown.
        """
        valid_scenarios = {"real_time", "batch", "research"}
        if scenario not in valid_scenarios:
            raise ValueError(f"Unknown scenario '{scenario}'. Valid: {valid_scenarios}")
        
        if scenario == "real_time":
            if self.max_neighbors > 10:
                warnings.warn(
                    f"max_neighbors ({self.max_neighbors}) > 10 may impact "
                    f"real-time performance. Consider reducing for production.",
                    UserWarning
                )
            
            if not self.enable_caching:
                warnings.warn(
                    "enable_caching=False may impact real-time performance. "
                    "Consider enabling for production deployment.",
                    UserWarning
                )
        
        elif scenario == "batch":
            if self.cache_size < 50:
                warnings.warn(
                    f"cache_size ({self.cache_size}) < 50 may be too small "
                    f"for efficient batch processing.",
                    UserWarning
                )
    
    def get_memory_estimate(self) -> Dict[str, float]:
        """
        Estimates memory usage for this configuration.
        
        Returns:
            Dictionary with memory estimates in MB for different components.
        """
        # Rough estimates based on typical tensor sizes
        neighbor_matrix_mb = (self.max_neighbors ** 2) * 4 / (1024 ** 2)  # float32
        cache_mb = self.cache_size * neighbor_matrix_mb if self.enable_caching else 0
        
        return {
            "neighbor_matrices": neighbor_matrix_mb,
            "distance_cache": cache_mb,
            "total_estimated": neighbor_matrix_mb + cache_mb,
        }
    
    def __str__(self) -> str:
        """Returns human-readable string representation."""
        return (
            f"SocialPoolingConfig(radius={self.pooling_radius}m, "
            f"neighbors={self.max_neighbors}, "
            f"metric={self.distance_metric}, "
            f"weighting={self.weighting_function})"
        )
    
    def __repr__(self) -> str:
        """Returns detailed string representation for debugging."""
        return (
            f"SocialPoolingConfig("
            f"pooling_radius={self.pooling_radius}, "
            f"max_neighbors={self.max_neighbors}, "
            f"distance_metric='{self.distance_metric}', "
            f"weighting_function='{self.weighting_function}', "
            f"aggregation_method='{self.aggregation_method}', "
            f"coordinate_system='{self.coordinate_system}', "
            f"enable_caching={self.enable_caching}"
            f")"
        )


# Factory function for easier access
def create_social_pooling_config(
    scenario: str = "mixed",
    **overrides: Any
) -> SocialPoolingConfig:
    """
    Factory function to create Social Pooling configuration.
    
    Args:
        scenario: Preset scenario ("urban", "highway", "mixed").
        **overrides: Additional parameter overrides.
        
    Returns:
        SocialPoolingConfig instance.
        
    Raises:
        ValueError: If scenario is unknown.
        
    Example:
        >>> config = create_social_pooling_config("urban", max_neighbors=15)
        >>> print(config.pooling_radius)  # 500.0
    """
    scenarios = {
        "urban": SocialPoolingConfig.urban_preset,
        "highway": SocialPoolingConfig.highway_preset,
        "mixed": SocialPoolingConfig.mixed_preset,
    }
    
    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(scenarios.keys())}")
    
    return scenarios[scenario](**overrides)


# Export public interface
__all__ = [
    "SocialPoolingConfig",
    "SocialPoolingConfigError",
    "InvalidRadiusError", 
    "InvalidNeighborCountError",
    "DistanceMetric",
    "WeightingFunction",
    "AggregationMethod", 
    "create_social_pooling_config",
]