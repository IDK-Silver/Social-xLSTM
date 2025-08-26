"""
YAML Utility Functions

Simple utility functions for loading and handling YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Union, Optional, Dict, Any


def deep_merge(base: Dict[Any, Any], update: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        update: Update dictionary
        
    Returns:
        Merged dictionary
        
    Merge rules:
        - Dictionaries: Recursive merge
        - Lists/scalars: Complete replacement
    """
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Dictionaries: recursive merge
            result[key] = deep_merge(result[key], value)
        else:
            # Lists/scalars: complete replacement
            result[key] = value
    
    return result


def load_yaml_file_to_dict(config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load YAML file to dictionary with error handling
    
    Args:
        config_path: Path to YAML configuration file (str or Path object)
        
    Returns:
        dict: Configuration dictionary, or None if loading fails
    """
    # Convert to Path object and check file exists
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Warning: YAML file does not exist: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return config_dict
    except Exception as e:
        print(f"Warning: Failed to load YAML file {config_path}: {e}")
        return None


def load_profile_config(profile_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load profile configuration by merging multiple config files in specified order.
    
    Args:
        profile_path: Path to profile YAML file containing config list and overrides
        
    Returns:
        dict: Merged configuration dictionary, or None if loading fails
        
    Profile YAML format:
        configs:
          - path/to/config1.yaml
          - path/to/config2.yaml
        overrides:
          key1:
            subkey: value
    """
    profile_path = Path(profile_path)
    profile_config = load_yaml_file_to_dict(profile_path)
    
    if not profile_config:
        return None
    
    # Check required structure
    if 'configs' not in profile_config:
        print(f"Warning: Profile {profile_path} missing 'configs' section")
        return None
    
    # Start with empty result
    result = {}
    
    # Merge configurations in order
    for config_path in profile_config['configs']:
        # Resolve relative paths relative to working directory, not profile file
        if not Path(config_path).is_absolute():
            # Use config_path as-is since it's relative to working directory
            resolved_path = Path(config_path)
        else:
            resolved_path = Path(config_path)
        
        config = load_yaml_file_to_dict(resolved_path)
        if config:
            result = deep_merge(result, config)
        else:
            print(f"Warning: Failed to load config: {resolved_path}")
    
    # Apply overrides if present
    if 'overrides' in profile_config:
        result = deep_merge(result, profile_config['overrides'])
    
    return result