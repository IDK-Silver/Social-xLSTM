#!/usr/bin/env python3
"""
Snakemake Wrapper with Configuration Merging

A wrapper to solve Snakemake multiple configuration file timing issues.
Core functionality: First merge multiple config files into a temporary file,
then execute snakemake --replace-workflow-config.

Usage:
    python workflow/snakemake_warp.py --configfile cfg1.yaml --configfile cfg2.yaml target

Merge semantics:
    - Dictionaries: Deep merge (later overwrites earlier)
    - Lists/scalars: Complete replacement
    - File order: Left to right (last one wins)
    - Supports recursive loading via includes field

Features:
    - Flexible multi-config file merging
    - Support for working directory switching
    - Smart temporary file management
"""

import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

import yaml


def deep_merge(base: Dict[Any, Any], update: Dict[Any, Any]) -> Dict[Any, Any]:
    """Deep merge two dictionaries.
    
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


def load_config_with_includes(config_file: str, loaded_files: set = None) -> Dict[Any, Any]:
    """Load configuration file with includes field processing.
    
    Args:
        config_file: Configuration file path
        loaded_files: Set of already loaded files (to avoid circular references)
        
    Returns:
        Configuration dictionary
    """
    if loaded_files is None:
        loaded_files = set()
    
    # Avoid circular references
    if config_file in loaded_files:
        print(f"WARNING: Skipping already loaded config: {config_file}")
        return {}
    
    if not os.path.exists(config_file):
        print(f"ERROR: Config file does not exist: {config_file}")
        sys.exit(1)
    
    loaded_files.add(config_file)
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
    except yaml.YAMLError as e:
        print(f"ERROR: YAML parsing error {config_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Config file reading error {config_file}: {e}")
        sys.exit(1)
    
    # Process includes field
    if 'includes' in config:
        includes = config.pop('includes')
        base_config = {}
        
        # Load all includes first
        for include_file in includes:
            # Handle relative paths
            if not os.path.isabs(include_file):
                include_file = os.path.join(os.path.dirname(config_file), include_file)
            
            print(f"  Loading include: {include_file}")
            include_config = load_config_with_includes(include_file, loaded_files)
            base_config = deep_merge(base_config, include_config)
        
        # Current config overrides includes
        config = deep_merge(base_config, config)
    
    return config


def merge_configs(config_files: List[str]) -> Dict[Any, Any]:
    """Merge multiple configuration files (with includes support).
    
    Args:
        config_files: List of configuration file paths
        
    Returns:
        Merged configuration dictionary
    """
    merged_config = {}
    loaded_files = set()
    
    for config_file in config_files:
        print(f"Processing config: {config_file}")
        config = load_config_with_includes(config_file, loaded_files)
        merged_config = deep_merge(merged_config, config)
        print(f"Merged config: {config_file}")
    
    return merged_config


def create_temp_config(merged_config: Dict[Any, Any]) -> str:
    """Create temporary configuration file.
    
    Args:
        merged_config: Merged configuration
        
    Returns:
        Temporary configuration file path
    """
    temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix='snakemake_merged_')
    
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            yaml.dump(merged_config, f, default_flow_style=False, allow_unicode=True)
        print(f"Generated temporary config: {temp_path}")
        return temp_path
    except Exception as e:
        os.unlink(temp_path)
        print(f"ERROR: Failed to create temporary config: {e}")
        sys.exit(1)


def run_snakemake(temp_config: str, snakemake_args: List[str], work_dir: str = None) -> int:
    """Execute Snakemake.
    
    Args:
        temp_config: Temporary configuration file path
        snakemake_args: Snakemake arguments
        work_dir: Working directory (if specified, execute in that directory)
        
    Returns:
        Exit code
    """
    # Use snakemake from system PATH
    snakemake_cmd = 'snakemake'
    
    # Build Snakemake command
    cmd = [
        snakemake_cmd,
        '--configfile', temp_config,
        '--replace-workflow-config'
    ] + snakemake_args
    
    # Set execution directory
    cwd = work_dir if work_dir else os.getcwd()
    
    print(f"Executing command: {' '.join(cmd)}")
    if work_dir:
        print(f"Working directory: {work_dir}")
    
    try:
        result = subprocess.run(cmd, check=False, cwd=cwd)
        return result.returncode
    except Exception as e:
        print(f"ERROR: Snakemake execution failed: {e}")
        return 1


def parse_args() -> tuple:
    """Parse command line arguments.
    
    Returns:
        (config_files, snakemake_args, work_dir)
    """
    config_files = []
    snakemake_args = []
    work_dir = None
    
    i = 0
    while i < len(sys.argv[1:]):
        arg = sys.argv[i + 1]
        
        if arg == '--configfile':
            # Get configuration file
            if i + 1 < len(sys.argv[1:]):
                config_files.append(sys.argv[i + 2])
                i += 2
            else:
                print("ERROR: --configfile parameter requires a file path")
                sys.exit(1)
        elif arg == '--work-dir':
            # Get working directory
            if i + 1 < len(sys.argv[1:]):
                work_dir = sys.argv[i + 2]
                i += 2
            else:
                print("ERROR: --work-dir parameter requires a directory path")
                sys.exit(1)
        else:
            # Other arguments passed to Snakemake
            snakemake_args.append(arg)
            i += 1
    
    return config_files, snakemake_args, work_dir


def main():
    """Main function."""
    if len(sys.argv) < 2:
        usage_msg = (
            "Usage: python workflow/snakemake_warp.py "
            "[--work-dir /path] --configfile cfg1.yaml "
            "[--configfile cfg2.yaml ...] [snakemake_args...]"
        )
        print(usage_msg)
        sys.exit(1)
    
    # Parse arguments
    config_files, snakemake_args, work_dir = parse_args()
    
    if not config_files:
        print("ERROR: At least one --configfile parameter is required")
        sys.exit(1)
    
    print(f"Config files: {config_files}")
    print(f"Snakemake arguments: {snakemake_args}")
    if work_dir:
        print(f"Working directory: {work_dir}")
    
    # Merge configurations
    merged_config = merge_configs(config_files)
    
    # Create temporary configuration
    temp_config = create_temp_config(merged_config)
    
    try:
        # Execute Snakemake
        exit_code = run_snakemake(temp_config, snakemake_args, work_dir)
        sys.exit(exit_code)
    finally:
        # Clean up temporary files
        if os.path.exists(temp_config):
            os.unlink(temp_config)
            print(f"Cleaned up temporary file: {temp_config}")


if __name__ == '__main__':
    main()