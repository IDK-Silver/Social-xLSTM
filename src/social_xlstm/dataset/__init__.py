"""Dataset module for Social-xLSTM traffic prediction.

This module provides a comprehensive toolkit for traffic data processing,
organized into focused sub-packages:

- config: Configuration classes for dataset and storage operations
- core: Core dataset functionality (TimeSeries, DataModule, Processor)
- storage: Data storage and persistence (HDF5 operations, features)
- utils: Utility functions (JSON, XML, ZIP processing)
"""

# Configuration classes
from .config import TrafficDatasetConfig, TrafficHDF5Config

# Core functionality
from .core import TrafficDataProcessor, TrafficTimeSeries, TrafficDataModule

# Storage operations
from .storage import (
    TrafficHDF5Converter, TrafficFeatureExtractor, TrafficHDF5Reader,
    create_traffic_hdf5, ensure_traffic_hdf5, TrafficFeature
)

# Utilities
from .utils import VDInfo, VDLiveList

__all__ = [
    # Configuration
    'TrafficDatasetConfig',
    'TrafficHDF5Config',
    
    # Core functionality
    'TrafficDataProcessor',
    'TrafficTimeSeries',
    'TrafficDataModule',
    
    # Storage operations
    'TrafficHDF5Converter',
    'TrafficFeatureExtractor',
    'TrafficHDF5Reader',
    'create_traffic_hdf5',
    'ensure_traffic_hdf5',
    'TrafficFeature',
    
    # Utilities
    'VDInfo',
    'VDLiveList',
]