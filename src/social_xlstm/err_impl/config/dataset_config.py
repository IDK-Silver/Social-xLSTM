"""
Dynamic Dataset Configuration System

Provides configuration loading, validation, and management for different datasets.
Integrates with DatasetRegistry to ensure consistency between configurations and implementations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfigInfo:
    """Information about a dataset configuration."""
    name: str
    config_path: Path
    dataset_name: str
    feature_set: str
    feature_names: List[str]
    config_data: Dict[str, Any]


class DatasetConfigManager:
    """Manager for dataset-specific configurations."""
    
    def __init__(self, config_root: Union[str, Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_root: Root directory for configuration files (defaults to cfgs/datasets/)
        """
        if config_root is None:
            # Auto-detect config root relative to this file
            current_file = Path(__file__)
            repo_root = current_file.parent.parent.parent.parent
            config_root = repo_root / "cfgs" / "datasets"
        
        self.config_root = Path(config_root)
        self._config_cache: Dict[str, DatasetConfigInfo] = {}
    
    def load_dataset_config(self, dataset_name: str) -> DatasetConfigInfo:
        """
        Load configuration for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., "taiwan_vd", "pems_bay")
            
        Returns:
            DatasetConfigInfo object with loaded configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if dataset_name in self._config_cache:
            return self._config_cache[dataset_name]
        
        config_path = self.config_root / f"{dataset_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Validate required fields
        self._validate_config_structure(config_data, dataset_name)
        
        config_info = DatasetConfigInfo(
            name=dataset_name,
            config_path=config_path,
            dataset_name=config_data['dataset_name'],
            feature_set=config_data['feature_set'],
            feature_names=config_data['features']['names'],
            config_data=config_data
        )
        
        # Cache the loaded config
        self._config_cache[dataset_name] = config_info
        return config_info
    
    def validate_with_registry(self, dataset_name: str) -> bool:
        """
        Validate that dataset config is consistent with DatasetRegistry.
        
        Args:
            dataset_name: Name of the dataset to validate
            
        Returns:
            True if config is consistent with registry
            
        Raises:
            ValueError: If inconsistencies are found
        """
        try:
            from ..dataset.registry import get_dataset_info, validate_dataset_features
            
            # Load config and registry info
            config_info = self.load_dataset_config(dataset_name)
            registry_info = get_dataset_info(dataset_name)
            
            if not registry_info:
                raise ValueError(f"Dataset '{dataset_name}' not found in registry")
            
            # Check feature set consistency
            if config_info.feature_set != registry_info.feature_set:
                raise ValueError(
                    f"Feature set mismatch for {dataset_name}: "
                    f"config='{config_info.feature_set}', registry='{registry_info.feature_set}'"
                )
            
            # Check feature names consistency
            if not validate_dataset_features(dataset_name, config_info.feature_names):
                from ..dataset.registry import get_invalid_features
                invalid = get_invalid_features(dataset_name, config_info.feature_names)
                raise ValueError(
                    f"Invalid features in config for {dataset_name}: {invalid}. "
                    f"Registry supports: {registry_info.features}"
                )
            
            logger.info(f"✅ Configuration for '{dataset_name}' is consistent with registry")
            return True
            
        except ImportError:
            logger.warning("Registry module not available for validation")
            return True
    
    def get_feature_validation_ranges(self, dataset_name: str, feature_name: str) -> Dict[str, Any]:
        """
        Get validation ranges for a specific feature.
        
        Args:
            dataset_name: Name of the dataset
            feature_name: Name of the feature
            
        Returns:
            Dictionary with validation parameters (min, max, unit, etc.)
        """
        config_info = self.load_dataset_config(dataset_name)
        validation = config_info.config_data.get('features', {}).get('validation', {})
        return validation.get(feature_name, {})
    
    def get_processing_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get processing configuration for a dataset."""
        config_info = self.load_dataset_config(dataset_name)
        return config_info.config_data.get('processing', {})
    
    def get_training_defaults(self, dataset_name: str) -> Dict[str, Any]:
        """Get default training parameters for a dataset."""
        config_info = self.load_dataset_config(dataset_name)
        return config_info.config_data.get('training', {})
    
    def list_available_configs(self) -> List[str]:
        """List all available dataset configurations."""
        if not self.config_root.exists():
            return []
        
        configs = []
        for config_file in self.config_root.glob("*.yaml"):
            if config_file.is_file():
                configs.append(config_file.stem)
        return sorted(configs)
    
    def _validate_config_structure(self, config_data: Dict[str, Any], dataset_name: str) -> None:
        """Validate that config has required structure."""
        required_fields = [
            'dataset_name', 'feature_set', 'features'
        ]
        
        for field in required_fields:
            if field not in config_data:
                raise ValueError(f"Missing required field '{field}' in config for {dataset_name}")
        
        # Validate features structure
        features = config_data['features']
        if 'names' not in features:
            raise ValueError(f"Missing 'features.names' in config for {dataset_name}")
        
        if not isinstance(features['names'], list) or not features['names']:
            raise ValueError(f"'features.names' must be a non-empty list in config for {dataset_name}")


class ConfigValidationError(Exception):
    """Exception raised when dataset configuration validation fails."""
    pass


def validate_dataset_config(dataset_name: str, config_root: Union[str, Path] = None) -> bool:
    """
    Convenience function to validate a dataset configuration.
    
    Args:
        dataset_name: Name of the dataset to validate
        config_root: Optional root directory for configs
        
    Returns:
        True if validation passes
        
    Raises:
        ConfigValidationError: If validation fails
    """
    try:
        manager = DatasetConfigManager(config_root)
        manager.load_dataset_config(dataset_name)
        manager.validate_with_registry(dataset_name)
        return True
    except Exception as e:
        raise ConfigValidationError(f"Config validation failed for {dataset_name}: {e}")


def load_and_merge_configs(dataset_name: str, training_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load dataset config and merge with training configuration.
    
    Args:
        dataset_name: Name of the dataset
        training_config: Optional training configuration to merge
        
    Returns:
        Merged configuration dictionary
    """
    manager = DatasetConfigManager()
    dataset_config = manager.load_dataset_config(dataset_name)
    
    # Start with dataset defaults
    merged_config = {
        'dataset': {
            'name': dataset_config.dataset_name,
            'feature_set': dataset_config.feature_set,
            'features': dataset_config.feature_names,
        },
        'processing': manager.get_processing_config(dataset_name),
        'training': manager.get_training_defaults(dataset_name),
    }
    
    # Merge with provided training config
    if training_config:
        merged_config['training'].update(training_config)
    
    return merged_config