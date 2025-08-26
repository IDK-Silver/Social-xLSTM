"""
Dataset Feature Extractors

This module provides abstract interfaces and concrete implementations
for extracting features from different types of traffic datasets.
"""

from .base import BaseFeatureExtractor
from .taiwan_vd import TaiwanVDExtractor
from .pems_bay import PemsBayExtractor, PemsBayHDF5Adapter

__all__ = [
    "BaseFeatureExtractor",
    "TaiwanVDExtractor", 
    "PemsBayExtractor",
    "PemsBayHDF5Adapter",
]