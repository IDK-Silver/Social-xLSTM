"""
Dataset Registry System

Centralizes dataset type information and provides lookup functionality
for dynamic dataset configuration and feature extraction.
"""

from typing import Dict, Any, Type, Optional, List
from dataclasses import dataclass

try:
    from .extractors.base import BaseFeatureExtractor
except ImportError:
    # For standalone testing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from extractors.base import BaseFeatureExtractor


@dataclass
class DatasetRegistryInfo:
    """Information about a registered dataset type."""
    extractor_class: Type[BaseFeatureExtractor]
    feature_set: str
    features: List[str]
    description: str


# Central dataset registry mapping dataset names to their implementations
DATASET_REGISTRY: Dict[str, DatasetRegistryInfo] = {}


def register_dataset(
    name: str,
    extractor_class: Type[BaseFeatureExtractor],
    feature_set: str,
    features: List[str],
    description: str = ""
) -> None:
    """Register a dataset type in the global registry."""
    DATASET_REGISTRY[name] = DatasetRegistryInfo(
        extractor_class=extractor_class,
        feature_set=feature_set,
        features=features,
        description=description
    )


def get_dataset_info(name: str) -> Optional[DatasetRegistryInfo]:
    """Get registry information for a dataset by name."""
    return DATASET_REGISTRY.get(name)


def list_available_datasets() -> Dict[str, str]:
    """List all available dataset types with descriptions."""
    return {name: info.description for name, info in DATASET_REGISTRY.items()}


def create_feature_extractor(dataset_name: str) -> BaseFeatureExtractor:
    """
    Create a feature extractor instance for the specified dataset.
    
    Args:
        dataset_name: Name of the registered dataset
        
    Returns:
        Instance of the appropriate feature extractor
        
    Raises:
        ValueError: If dataset is not registered
    """
    dataset_info = get_dataset_info(dataset_name)
    if not dataset_info:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    
    return dataset_info.extractor_class(
        dataset_name=dataset_name,
        feature_set=dataset_info.feature_set
    )


def validate_dataset_features(dataset_name: str, requested_features: List[str]) -> bool:
    """
    Validate that requested features are supported by the dataset.
    
    Args:
        dataset_name: Name of the registered dataset
        requested_features: List of feature names to validate
        
    Returns:
        True if all features are supported
        
    Raises:
        ValueError: If dataset is not registered
    """
    dataset_info = get_dataset_info(dataset_name)
    if not dataset_info:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    
    supported_features = set(dataset_info.features)
    requested_features_set = set(requested_features)
    
    return requested_features_set.issubset(supported_features)


def get_invalid_features(dataset_name: str, requested_features: List[str]) -> List[str]:
    """
    Get list of invalid features for the dataset.
    
    Args:
        dataset_name: Name of the registered dataset
        requested_features: List of feature names to check
        
    Returns:
        List of invalid feature names
        
    Raises:
        ValueError: If dataset is not registered
    """
    dataset_info = get_dataset_info(dataset_name)
    if not dataset_info:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    
    supported_features = set(dataset_info.features)
    requested_features_set = set(requested_features)
    
    return list(requested_features_set - supported_features)


# Register standard datasets (imported lazily to avoid circular dependencies)
def _register_standard_datasets():
    """Register the standard traffic prediction datasets."""
    
    # Import extractors lazily to avoid circular dependencies
    try:
        from .extractors.taiwan_vd import TaiwanVDExtractor
    except ImportError:
        try:
            # Fallback for different import contexts
            from extractors.taiwan_vd import TaiwanVDExtractor
        except ImportError:
            TaiwanVDExtractor = None
    
    if TaiwanVDExtractor is not None:
        
        # Register Taiwan VD dataset
        register_dataset(
            name="taiwan_vd",
            extractor_class=TaiwanVDExtractor,
            feature_set="traffic_core_v1",
            features=["avg_speed", "total_volume", "avg_occupancy", "speed_std", "lane_count"],
            description="Taiwan VD traffic data with 5 core features"
        )
    
    # Register PEMS Bay dataset
    try:
        from .extractors.pems_bay import PemsBayExtractor
    except ImportError:
        try:
            from extractors.pems_bay import PemsBayExtractor
        except ImportError:
            PemsBayExtractor = None
    
    if PemsBayExtractor is not None:
        register_dataset(
            name="pems_bay",
            extractor_class=PemsBayExtractor,
            feature_set="pems_bay_v1", 
            features=["speed", "lanes", "length", "latitude", "longitude", "direction"],
            description="PEMS Bay Area traffic data with speed and metadata features"
        )


# Auto-register standard datasets when module is imported
_register_standard_datasets()