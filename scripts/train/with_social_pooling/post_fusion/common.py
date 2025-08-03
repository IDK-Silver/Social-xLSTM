#!/usr/bin/env python3
"""
Common utilities for Post-Fusion Social Pooling training scripts

⚠️  DEPRECATED: This module uses incorrect centralized architecture ⚠️

This module provides shared utilities specifically for training Social-xLSTM models
using the Post-Fusion strategy. It extends the base training framework to support
Social Pooling integration while maintaining backward compatibility.

DEPRECATION NOTICE:
This module is based on the deprecated centralized Social Pooling architecture that 
fundamentally cannot scale to distributed social traffic scenarios. The Post-Fusion 
utilities create bottlenecks that prevent proper distributed xLSTM implementation.

MIGRATION PATH:
- Use scripts/train/distributed_social_xlstm/common.py instead
- See docs/legacy/explanation/social-pooling-implementation-guide.md
- Historical access: git checkout centralized-legacy-v0.2

Features (DEPRECATED):
- Post-Fusion model creation with SocialTrafficModel wrapper
- Coordinate data loading and validation
- Social Pooling configuration management
- Integration with existing TrafficLSTM/TrafficXLSTM models

Author: Social-xLSTM Project Team
Version: DEPRECATED
"""

import logging
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

# Import base utilities
from scripts.train.without_social_pooling.common import (
    setup_logging, check_conda_environment, create_data_module,
    print_training_start, print_training_complete, get_device_warnings
)

# Import Social-xLSTM components
from social_xlstm.models.social_pooling_config import (
    SocialPoolingConfig, create_social_pooling_config
)
from social_xlstm.models.social_pooling import SocialPooling
from social_xlstm.models.social_traffic_model import SocialTrafficModel, create_social_traffic_model


def add_post_fusion_arguments(parser):
    """
    Add Post-Fusion Social Pooling specific command line arguments.
    
    Args:
        parser: ArgumentParser instance
    """
    # First add base arguments from without_social_pooling
    from scripts.train.without_social_pooling.common import add_common_arguments
    add_common_arguments(parser)
    
    # Override experiment name default for Social Pooling
    for action in parser._actions:
        if action.dest == 'experiment_name':
            action.default = "social_lstm_post_fusion"
    
    # Social Pooling configuration
    social_group = parser.add_argument_group('Social Pooling Configuration')
    
    social_group.add_argument("--scenario", type=str, default="mixed",
                            choices=["urban", "highway", "mixed"],
                            help="Traffic scenario preset (urban: dense/short-range, highway: sparse/long-range)")
    
    social_group.add_argument("--pooling_radius", type=float, default=None,
                            help="Spatial pooling radius in meters (overrides scenario preset)")
    
    social_group.add_argument("--max_neighbors", type=int, default=None,
                            help="Maximum number of spatial neighbors (overrides scenario preset)")
    
    social_group.add_argument("--distance_metric", type=str, default=None,
                            choices=["euclidean", "manhattan", "haversine"],
                            help="Distance calculation method (overrides scenario preset)")
    
    social_group.add_argument("--weighting_function", type=str, default=None,
                            choices=["gaussian", "exponential", "linear", "inverse"],
                            help="Spatial weighting function (overrides scenario preset)")
    
    social_group.add_argument("--aggregation_method", type=str, default=None,
                            choices=["weighted_mean", "weighted_sum", "attention"],
                            help="Feature aggregation method (overrides scenario preset)")
    
    # Coordinate data configuration
    coord_group = parser.add_argument_group('Coordinate Data Configuration')
    
    coord_group.add_argument("--coordinate_data", type=str, required=True,
                           help="Path to VD coordinate data file (JSON format)")
    
    coord_group.add_argument("--coordinate_system", type=str, default="projected",
                           choices=["projected", "geographic"],
                           help="Coordinate system type")
    
    # Performance optimization
    perf_group = parser.add_argument_group('Performance Optimization')
    
    perf_group.add_argument("--enable_caching", action="store_true", default=True,
                          help="Enable distance calculation caching")
    
    perf_group.add_argument("--cache_size", type=int, default=100,
                          help="Distance cache size")
    
    perf_group.add_argument("--enable_profiling", action="store_true",
                          help="Enable performance profiling")


def create_social_pooling_config_from_args(args, logger) -> SocialPoolingConfig:
    """
    Create Social Pooling configuration from command line arguments.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        SocialPoolingConfig instance
    """
    logger.info(f"Creating Social Pooling configuration for scenario: {args.scenario}")
    
    # Start with scenario preset
    overrides = {}
    
    # Apply command line overrides
    if args.pooling_radius is not None:
        overrides['pooling_radius'] = args.pooling_radius
    if args.max_neighbors is not None:
        overrides['max_neighbors'] = args.max_neighbors
    if args.distance_metric is not None:
        overrides['distance_metric'] = args.distance_metric
    if args.weighting_function is not None:
        overrides['weighting_function'] = args.weighting_function
    if args.aggregation_method is not None:
        overrides['aggregation_method'] = args.aggregation_method
    
    # System configuration
    overrides['coordinate_system'] = args.coordinate_system
    overrides['enable_caching'] = args.enable_caching
    overrides['cache_size'] = args.cache_size
    overrides['enable_profiling'] = args.enable_profiling
    
    # Create configuration
    social_config = create_social_pooling_config(args.scenario, **overrides)
    
    logger.info(f"Social Pooling Config: {social_config}")
    
    # Log memory estimates
    memory_est = social_config.get_memory_estimate()
    logger.info(f"Estimated memory usage: {memory_est['total_estimated']:.1f} MB")
    
    return social_config


def load_coordinate_data(coordinate_path: str, logger) -> Dict[str, Tuple[float, float]]:
    """
    Load VD coordinate data from JSON file.
    
    Args:
        coordinate_path: Path to coordinate data file
        logger: Logger instance
        
    Returns:
        Dictionary mapping VD IDs to (x, y) coordinates
        
    Raises:
        FileNotFoundError: If coordinate file doesn't exist
        ValueError: If coordinate data format is invalid
    """
    logger.info(f"Loading coordinate data from: {coordinate_path}")
    
    coord_path = Path(coordinate_path)
    if not coord_path.exists():
        raise FileNotFoundError(
            f"Coordinate data file not found: {coordinate_path}\n"
            f"Please ensure VD coordinate data is available. "
            f"Expected format: {{'vd_id': [x, y], ...}}"
        )
    
    try:
        with open(coord_path, 'r', encoding='utf-8') as f:
            coord_data = json.load(f)
        
        # Convert to expected format
        coordinates = {}
        for vd_id, coords in coord_data.items():
            if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                raise ValueError(f"Invalid coordinate format for VD {vd_id}: {coords}")
            
            try:
                x, y = float(coords[0]), float(coords[1])
                coordinates[str(vd_id)] = (x, y)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid coordinate values for VD {vd_id}: {coords}") from e
        
        logger.info(f"Loaded coordinates for {len(coordinates)} VDs")
        
        # Log coordinate range for validation
        if coordinates:
            x_coords = [coord[0] for coord in coordinates.values()]
            y_coords = [coord[1] for coord in coordinates.values()]
            logger.info(f"Coordinate ranges: X[{min(x_coords):.1f}, {max(x_coords):.1f}], "
                       f"Y[{min(y_coords):.1f}, {max(y_coords):.1f}]")
        
        return coordinates
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in coordinate file: {e}") from e
    except Exception as e:
        raise ValueError(f"Error loading coordinate data: {e}") from e


def create_social_data_module(args, coordinates: Dict[str, Tuple[float, float]], logger):
    """
    Create data module with Social Pooling coordinate support.
    
    Args:
        args: Command line arguments
        coordinates: VD coordinate data
        logger: Logger instance
        
    Returns:
        Enhanced data module with coordinate information
    """
    # Create base data module
    data_module = create_data_module(args, logger)
    
    # Add coordinate information
    data_module.coordinates = coordinates
    
    # Validate that all VDs in dataset have coordinates
    sample_batch = next(iter(data_module.train_dataloader()))
    
    # For single VD mode, ensure selected VD has coordinates
    if hasattr(args, 'select_vd_id') and args.select_vd_id:
        if args.select_vd_id not in coordinates:
            available_vds = list(coordinates.keys())[:10]  # Show first 10 for brevity
            raise ValueError(
                f"Selected VD '{args.select_vd_id}' not found in coordinate data. "
                f"Available VDs: {available_vds}{'...' if len(coordinates) > 10 else ''}"
            )
        logger.info(f"Coordinate validation passed for VD: {args.select_vd_id}")
    
    # For multi-VD mode, check all required VDs
    elif hasattr(args, 'vd_ids') and args.vd_ids:
        required_vds = [vd.strip() for vd in args.vd_ids.split(',')]
        missing_vds = [vd for vd in required_vds if vd not in coordinates]
        if missing_vds:
            raise ValueError(
                f"VDs missing from coordinate data: {missing_vds}. "
                f"Please ensure all required VDs have coordinate information."
            )
        logger.info(f"Coordinate validation passed for {len(required_vds)} VDs")
    
    return data_module


def create_post_fusion_model(args, data_module, social_config: SocialPoolingConfig, 
                           coordinates: Dict[str, Tuple[float, float]], logger) -> SocialTrafficModel:
    """
    Create Post-Fusion Social Traffic model.
    
    Args:
        args: Command line arguments
        data_module: Data module
        social_config: Social Pooling configuration
        coordinates: VD coordinate data
        logger: Logger instance
        
    Returns:
        SocialTrafficModel instance with Post-Fusion architecture
    """
    logger.info(f"Creating Post-Fusion Social-{args.model_type.upper()} model")
    
    # Get feature information
    sample_batch = next(iter(data_module.train_dataloader()))
    input_features = sample_batch['input_seq'].shape[-1]
    
    # Create base model configuration
    if args.model_type == "lstm":
        from social_xlstm.models.lstm import TrafficLSTM
        
        if hasattr(args, 'select_vd_id') and args.select_vd_id:
            # Single VD mode
            base_model = TrafficLSTM.create_single_vd_model(
                input_size=input_features,
                output_size=input_features,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout
            )
        else:
            # Multi-VD mode (if implemented)
            raise NotImplementedError("Multi-VD Post-Fusion training not implemented yet")
            
    elif args.model_type == "xlstm":
        from social_xlstm.models.xlstm import TrafficXLSTM, TrafficXLSTMConfig
        
        if hasattr(args, 'select_vd_id') and args.select_vd_id:
            # Single VD mode
            xlstm_config = TrafficXLSTMConfig(
                input_size=input_features,
                output_size=input_features,
                embedding_dim=getattr(args, 'embedding_dim', args.hidden_size),
                num_blocks=getattr(args, 'num_blocks', 6),
                slstm_at=getattr(args, 'slstm_at', [1, 3]),
                dropout=args.dropout,
                multi_vd_mode=False
            )
            base_model = TrafficXLSTM(xlstm_config)
        else:
            # Multi-VD mode (if implemented)
            raise NotImplementedError("Multi-VD Post-Fusion training not implemented yet")
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Create Social Traffic model using factory
    social_model = create_social_traffic_model(
        base_model=base_model,
        social_config=social_config,
        model_type=f"post_fusion_{args.model_type}",
        scenario=args.scenario
    )
    
    # Log model information
    model_info = social_model.get_model_info()
    logger.info(f"Social-{args.model_type.upper()} Model Information:")
    logger.info(f"  Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"  Base model: {model_info['base_model_type']}")
    logger.info(f"  Fusion strategy: {model_info['fusion_strategy']}")
    logger.info(f"  Social pooling: {model_info['social_pooling_config']}")
    
    return social_model


def validate_post_fusion_setup(args, logger):
    """
    Validate Post-Fusion training setup and environment.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Raises:
        ValueError: If setup validation fails
    """
    logger.info("Validating Post-Fusion training setup...")
    
    # Check required arguments
    if not hasattr(args, 'coordinate_data') or not args.coordinate_data:
        raise ValueError(
            "Post-Fusion training requires --coordinate_data argument. "
            "Please provide path to VD coordinate data file."
        )
    
    # Check coordinate data file exists
    if not Path(args.coordinate_data).exists():
        raise ValueError(f"Coordinate data file not found: {args.coordinate_data}")
    
    # Validate model type support
    if args.model_type not in ["lstm", "xlstm"]:
        raise ValueError(f"Unsupported model type for Post-Fusion: {args.model_type}")
    
    # Check VD selection for single VD mode
    if not (hasattr(args, 'select_vd_id') and args.select_vd_id):
        if not (hasattr(args, 'vd_ids') and args.vd_ids):
            raise ValueError(
                "Post-Fusion training requires VD selection. "
                "Use --select_vd_id for single VD or --vd_ids for multi-VD mode."
            )
    
    logger.info("Post-Fusion setup validation passed")


def print_post_fusion_start(logger, args):
    """
    Print Post-Fusion training start information.
    
    Args:
        logger: Logger instance
        args: Command line arguments
    """
    from datetime import datetime
    
    logger.info("=" * 70)
    logger.info(f"Post-Fusion Social-{args.model_type.upper()} Training Started")
    logger.info(f"Scenario: {args.scenario} | Strategy: Post-Fusion")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # Log configuration
    logger.info(f"Base model: {args.model_type.upper()}")
    logger.info(f"Coordinate data: {args.coordinate_data}")
    logger.info(f"Traffic scenario: {args.scenario}")
    
    if hasattr(args, 'select_vd_id') and args.select_vd_id:
        logger.info(f"Training mode: Single VD ({args.select_vd_id})")
    else:
        logger.info("Training mode: Multi-VD")


def print_post_fusion_complete(logger, trainer):
    """
    Print Post-Fusion training completion information.
    
    Args:
        logger: Logger instance
        trainer: Trainer instance
    """
    from datetime import datetime
    
    logger.info("=" * 70)
    logger.info("Post-Fusion Social-xLSTM Training Completed!")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # Training history
    history = trainer.training_history
    if history['train_loss']:
        logger.info(f"Final train loss: {history['train_loss'][-1]:.6f}")
    if history['val_loss']:
        logger.info(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")
    
    logger.info(f"Total epochs: {len(history['train_loss'])}")
    logger.info(f"Experiment results saved to: {trainer.experiment_dir}")
    
    # Post-Fusion specific information
    logger.info(f"\nPost-Fusion Training Notes:")
    logger.info(f"• Spatial features integrated via Social Pooling")
    logger.info(f"• Gated Fusion strategy used for feature combination")
    logger.info(f"• Compare with baseline models for performance evaluation")
    
    # Saved files
    logger.info(f"\nSaved files:")
    logger.info(f"• Model configuration: {trainer.experiment_dir}/config.json")
    logger.info(f"• Best model weights: {trainer.experiment_dir}/best_model.pt")
    logger.info(f"• Social pooling config: {trainer.experiment_dir}/social_config.json")


# Export for use in training scripts
__all__ = [
    'add_post_fusion_arguments',
    'create_social_pooling_config_from_args',
    'load_coordinate_data',
    'create_social_data_module',
    'create_post_fusion_model',
    'validate_post_fusion_setup',
    'print_post_fusion_start',
    'print_post_fusion_complete'
]


# Test function
if __name__ == "__main__":
    # Test configuration creation
    logger = setup_logging()
    logger.info("Post-Fusion common utilities test successful")
    
    # Test argument parser
    parser = argparse.ArgumentParser()
    add_post_fusion_arguments(parser)
    logger.info("Post-Fusion argument parser test successful")