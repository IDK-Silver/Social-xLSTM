"""
Base Feature Extractor Abstract Interface

Defines the contract for extracting features from different dataset formats.
Each dataset type (Taiwan VD, PEMS-BAY, etc.) should implement this interface
to provide standardized feature extraction capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for dataset-specific feature extractors.
    
    Each concrete implementation should handle the specifics of extracting
    standardized traffic features from raw data in different formats.
    """
    
    def __init__(self, dataset_name: str, feature_set: str):
        """
        Initialize the feature extractor.
        
        Args:
            dataset_name: Name of the dataset (e.g., "taiwan_vd", "pems_bay")
            feature_set: Version of feature set (e.g., "traffic_core_v1")
        """
        self.dataset_name = dataset_name
        self.feature_set = feature_set
    
    @abstractmethod
    def extract_features(self, raw_data: Any, feature_names: List[str]) -> List[float]:
        """
        Extract specified features from raw data.
        
        Args:
            raw_data: Raw data in dataset-specific format
            feature_names: List of feature names to extract
            
        Returns:
            List of feature values in the same order as feature_names
            
        Raises:
            ValueError: If any feature_names are not supported
        """
        pass
    
    @abstractmethod
    def validate_feature_names(self, feature_names: List[str]) -> bool:
        """
        Validate that all requested feature names are supported.
        
        Args:
            feature_names: List of feature names to validate
            
        Returns:
            True if all features are supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_features(self) -> List[str]:
        """
        Get list of all features supported by this extractor.
        
        Returns:
            List of supported feature names
        """
        pass
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this extractor's features.
        
        Returns:
            Dictionary with feature metadata
        """
        return {
            "dataset_name": self.dataset_name,
            "feature_set": self.feature_set,
            "supported_features": self.get_supported_features(),
            "extractor_class": self.__class__.__name__
        }
    
    def validate_and_extract(self, raw_data: Any, feature_names: List[str]) -> List[float]:
        """
        Validate feature names and extract features.
        
        Args:
            raw_data: Raw data in dataset-specific format  
            feature_names: List of feature names to extract
            
        Returns:
            List of feature values
            
        Raises:
            ValueError: If any feature names are invalid
        """
        if not self.validate_feature_names(feature_names):
            supported = self.get_supported_features()
            invalid = set(feature_names) - set(supported)
            raise ValueError(
                f"Invalid features for {self.dataset_name}: {invalid}. "
                f"Supported features: {supported}"
            )
        
        return self.extract_features(raw_data, feature_names)
    
    @staticmethod
    def _safe_float_conversion(value: Any, default: float = np.nan) -> float:
        """
        Safely convert value to float, handling common error cases.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            Float value or default
        """
        if value is None:
            return default
        
        try:
            float_val = float(value)
            # Handle common traffic data error codes
            if float_val in [-99, -1, 255] or np.isnan(float_val) or np.isinf(float_val):
                return default
            return float_val
        except (ValueError, TypeError, OverflowError):
            return default
    
    @staticmethod  
    def _is_valid_value(value: Any, min_val: float = 0, max_val: Optional[float] = None) -> bool:
        """
        Check if value is valid (not error code or out of range).
        
        Args:
            value: Value to validate
            min_val: Minimum valid value
            max_val: Maximum valid value (None for no limit)
            
        Returns:
            True if value is valid
        """
        if value is None or np.isnan(value) or np.isinf(value):
            return False
        
        # Common error codes in traffic data
        if value in [-99, -1, 255]:
            return False
            
        if value < min_val:
            return False
            
        if max_val is not None and value > max_val:
            return False
            
        return True