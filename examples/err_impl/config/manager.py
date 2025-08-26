"""
Dynamic Model Configuration Manager

Provides unified configuration loading and merging functionality,
leveraging the existing snakemake_warp.py infrastructure for YAML merging.
"""

import yaml
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from .registry import MODEL_REGISTRY, get_model_info

# Add project root to path for snakemake_warp import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from workflow.snakemake_warp import merge_configs
except ImportError:
    print("Warning: Could not import merge_configs from snakemake_warp")
    merge_configs = None


@dataclass 
class MergedConfig:
    """Container for merged configuration with metadata."""
    model_config: Any           # Instantiated model configuration object
    social_config: Dict[str, Any]  # Social pooling configuration
    vd_config: Dict[str, Any]      # VD mode configuration  
    training_config: Dict[str, Any] # Training configuration
    raw_merged: Dict[str, Any]     # Raw merged dictionary
    model_name: str                # Model type name
    effective_input_size: int      # Calculated effective input size


class DynamicModelConfigManager:
    """
    Manages dynamic model configuration with layered YAML support.
    
    Supports both direct merged config loading and multi-file merging
    using the existing snakemake_warp.py infrastructure.
    """
    
    @classmethod
    def from_merged_config(cls, merged_config_dict: Dict[str, Any]) -> MergedConfig:
        """
        Load configuration from pre-merged configuration dictionary.
        
        Args:
            merged_config_dict: Pre-merged configuration dictionary
            
        Returns:
            MergedConfig object with instantiated model configuration
            
        Raises:
            ValueError: If model name is not found or invalid configuration
        """
        # Extract model configuration
        if "model" not in merged_config_dict:
            raise ValueError("Configuration must contain 'model' section")
        
        model_section = merged_config_dict["model"]
        if "name" not in model_section:
            raise ValueError("Model configuration must contain 'name' field")
        
        model_name = model_section["name"]
        
        # Get model info from registry
        model_info = get_model_info(model_name)
        if not model_info:
            available = list(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
        
        # Extract model-specific parameters
        config_key = model_info.config_key
        if config_key not in model_section:
            raise ValueError(f"Model configuration must contain '{config_key}' section")
        
        model_params = model_section[config_key]
        
        # Instantiate model configuration
        try:
            model_config = model_info.config_class(**model_params)
        except Exception as e:
            raise ValueError(f"Invalid model configuration: {e}")
        
        # Extract other configurations with defaults
        social_config = merged_config_dict.get("social", {"enabled": False})
        vd_config = merged_config_dict.get("vd", {"mode": "single", "count": 1})
        training_config = merged_config_dict.get("training", {})
        
        # Calculate effective input size
        effective_input_size = cls._calculate_effective_input_size(
            model_config, vd_config
        )
        
        # Validate configuration consistency
        cls._validate_configuration(model_config, social_config, vd_config)
        
        return MergedConfig(
            model_config=model_config,
            social_config=social_config,
            vd_config=vd_config,
            training_config=training_config,
            raw_merged=merged_config_dict,
            model_name=model_name,
            effective_input_size=effective_input_size
        )
    
    @classmethod
    def from_yaml_files(cls, yaml_paths: List[Union[str, Path]]) -> MergedConfig:
        """
        Load and merge multiple YAML files using snakemake_warp logic.
        
        Args:
            yaml_paths: List of YAML file paths to merge
            
        Returns:
            MergedConfig object with merged configuration
            
        Raises:
            ValueError: If merging fails or invalid configuration
        """
        if merge_configs is None:
            raise RuntimeError("merge_configs function not available from snakemake_warp")
        
        # Convert paths to strings
        str_paths = [str(p) for p in yaml_paths]
        
        try:
            merged_config = merge_configs(str_paths)
        except Exception as e:
            raise ValueError(f"Failed to merge configuration files: {e}")
        
        return cls.from_merged_config(merged_config)
    
    @staticmethod
    def _calculate_effective_input_size(model_config: Any, vd_config: Dict[str, Any]) -> int:
        """Calculate effective input size based on VD configuration."""
        base_input_size = getattr(model_config, 'input_size', 3)
        vd_count = vd_config.get('count', 1)
        return base_input_size * vd_count
    
    @staticmethod
    def _validate_configuration(
        model_config: Any, 
        social_config: Dict[str, Any], 
        vd_config: Dict[str, Any]
    ) -> None:
        """
        Validate configuration consistency.
        
        Raises:
            ValueError: If configuration is inconsistent
        """
        # Social pooling requires multi-VD mode
        if social_config.get("enabled", False) and vd_config.get("mode") == "single":
            raise ValueError(
                "Social pooling requires multi-VD mode. "
                "Use vd_modes/multi.yaml or disable social pooling."
            )
        
        # VD count consistency
        vd_mode = vd_config.get("mode", "single")
        vd_count = vd_config.get("count", 1)
        
        if vd_mode == "single" and vd_count != 1:
            raise ValueError(f"Single VD mode requires count=1, got {vd_count}")
        
        if vd_mode == "multi" and vd_count <= 1:
            raise ValueError(f"Multi VD mode requires count>1, got {vd_count}")


def load_config_from_paths(config_paths: List[str]) -> MergedConfig:
    """
    Convenience function to load configuration from file paths.
    
    Args:
        config_paths: List of YAML configuration file paths
        
    Returns:
        Merged configuration object
    """
    return DynamicModelConfigManager.from_yaml_files(config_paths)


def load_single_config_file(config_path: str) -> MergedConfig:
    """
    Load configuration from a single pre-merged YAML file.
    
    Args:
        config_path: Path to single YAML configuration file
        
    Returns:
        Merged configuration object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return DynamicModelConfigManager.from_merged_config(config_dict)