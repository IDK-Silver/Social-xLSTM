"""
Dynamic Configuration System

This module provides configuration management for both models and datasets,
including parameter mapping utilities and validation systems.
"""

from .registry import MODEL_REGISTRY, get_model_info
from .manager import DynamicModelConfigManager, load_config_from_paths, load_single_config_file
from .parameter_mapper import ParameterMapper, create_parameter_mapper
from .dataset_config import (
    DatasetConfigManager,
    DatasetConfigInfo,
    ConfigValidationError,
    validate_dataset_config,
    load_and_merge_configs
)

__all__ = [
    "MODEL_REGISTRY",
    "get_model_info", 
    "DynamicModelConfigManager",
    "ParameterMapper",
    "create_parameter_mapper",
    "load_config_from_paths",
    "load_single_config_file",
    # Dataset configuration
    "DatasetConfigManager",
    "DatasetConfigInfo", 
    "ConfigValidationError",
    "validate_dataset_config",
    "load_and_merge_configs",
]