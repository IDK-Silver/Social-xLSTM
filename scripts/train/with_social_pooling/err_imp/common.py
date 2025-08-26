#!/usr/bin/env python3
"""
Common utilities for training scripts with Social Pooling

This module provides essential shared utilities for training Social-xLSTM models
with spatial interactions and Social Pooling mechanisms.

Enhanced from the base common.py to include Social Pooling specific functionality
and distributed Social-xLSTM model support.

Author: Social-xLSTM Project Team
"""

import logging
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def setup_logging(level=logging.INFO):
    """
    Setup unified logging configuration.
    
    Args:
        level: Log level
    
    Returns:
        logger: Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT
    )
    return logging.getLogger('__main__')


def check_conda_environment(logger):
    """
    Check if running in correct conda environment.
    
    Args:
        logger: Logger instance
    """
    if 'CONDA_DEFAULT_ENV' in os.environ:
        env_name = os.environ['CONDA_DEFAULT_ENV']
        logger.info(f"Detected conda environment: {env_name}")
        if env_name != 'social_xlstm':
            logger.warning(f"Recommend using 'social_xlstm' environment, current: {env_name}")
    else:
        logger.warning("No conda environment detected, recommend: conda activate social_xlstm")


def add_common_arguments(parser):
    """
    Add common command line arguments for Social Pooling training scripts.
    
    Args:
        parser: ArgumentParser instance
    """
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                        help="HDF5 data file path (required)")
    parser.add_argument("--sequence_length", type=int, default=12,
                        help="Time sequence length")
    parser.add_argument("--prediction_length", type=int, default=1,
                        help="Prediction sequence length")
    
    # Model parameters (Social-xLSTM specific)
    parser.add_argument("--model_type", type=str, default="social_xlstm", 
                        choices=["social_xlstm", "xlstm"],
                        help="Model type (social_xlstm: Social-xLSTM, xlstm: Standard xLSTM)")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden layer size")
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="xLSTM embedding dimension")
    parser.add_argument("--num_blocks", type=int, default=6,
                        help="Number of xLSTM blocks")
    parser.add_argument("--slstm_at", type=int, nargs='+', default=[1, 3],
                        help="Positions for sLSTM blocks (e.g., --slstm_at 1 3)")
    parser.add_argument("--context_length", type=int, default=256,
                        help="xLSTM context length")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout ratio")
    
    # Social Pooling parameters
    parser.add_argument("--enable_spatial_pooling", action="store_true",
                        help="Enable spatial social pooling")
    parser.add_argument("--spatial_radius", type=float, default=2.0,
                        help="Spatial pooling radius in meters")
    parser.add_argument("--aggregation_method", type=str, default="weighted_mean",
                        choices=["weighted_mean", "weighted_sum", "attention"],
                        help="Social pooling aggregation method")
    parser.add_argument("--pool_type", type=str, default="weighted_mean",
                        choices=["mean", "max", "weighted_mean"],
                        help="Legacy pool type (for backward compatibility)")
    parser.add_argument("--max_neighbors", type=int, default=8,
                        help="Maximum number of neighbors for social pooling")
    
    # Multi-VD parameters
    parser.add_argument("--selected_vdids", type=str, nargs='*', 
                        help="List of specific VD IDs to use for training")
    parser.add_argument("--num_vds", type=int, default=None,
                        help="Number of VDs to use (legacy parameter)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Weight decay")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--gradient_clip_value", type=float, default=0.5,
                        help="Value for gradient clipping (max norm)")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience epochs for LR scheduler before reduction")
    
    # Infrastructure parameters
    parser.add_argument("--accelerator", type=str, default="auto",
                        choices=["auto", "cpu", "gpu", "mps"],
                        help="Training accelerator")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of devices to use")
    parser.add_argument("--precision", type=str, default="32",
                        choices=["16", "32", "bf16"],
                        help="Training precision")
    parser.add_argument("--enable_gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing for memory efficiency")
    
    # Dataset split parameters
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation data ratio")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Experiment parameters
    parser.add_argument("--experiment_name", type=str, default="social_xlstm_distributed",
                        help="Experiment name")
    parser.add_argument("--save_dir", type=str, default="logs",
                        help="Directory to save logs and checkpoints")
    
    # Development and debugging
    parser.add_argument("--limit_train_batches", type=float, default=1.0,
                        help="Limit training batches (for development)")
    parser.add_argument("--limit_val_batches", type=float, default=1.0,
                        help="Limit validation batches (for development)")
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run one batch for development testing")
    parser.add_argument("--dry_run", action="store_true",
                        help="Set up everything but do not start training")


def resolve_aggregation_method(args) -> str:
    """
    Resolve the aggregation method from arguments using parameter mapping.
    
    Args:
        args: Command line arguments
        
    Returns:
        str: Resolved aggregation method
    """
    try:
        from social_xlstm.config import ParameterMapper
        
        # Use new aggregation_method if explicitly provided
        if hasattr(args, 'aggregation_method') and args.aggregation_method != 'weighted_mean':
            return args.aggregation_method
        
        # Map legacy pool_type to aggregation_method
        mapper = ParameterMapper()
        aggregation_method = mapper.POOL_TYPE_TO_AGGREGATION_METHOD.get(
            args.pool_type, 'weighted_mean'
        )
        
        # Warn if mapping occurs
        if args.pool_type != 'weighted_mean':
            print(f"Warning: Mapping legacy pool_type '{args.pool_type}' to aggregation_method '{aggregation_method}'")
        
        return aggregation_method
    except ImportError:
        # Fallback to legacy pool_type
        return getattr(args, 'aggregation_method', args.pool_type)


def print_training_start(logger, mode='distributed_social'):
    """
    Print training start information.
    
    Args:
        logger: Logger instance
        mode: Training mode
    """
    if mode == 'distributed_social':
        mode_name = "Distributed Social-xLSTM"
    elif mode == 'single_vd_social':
        mode_name = "Single VD Social-xLSTM"
    elif mode == 'multi_vd_social':
        mode_name = "Multi-VD Social-xLSTM"
    else:
        mode_name = "Social-xLSTM"
    
    logger.info("=" * 70)
    logger.info(f"🚀 {mode_name} Training Started (With Social Pooling)")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)


def print_training_complete(logger, trainer, mode='distributed_social'):
    """
    Print training completion information.
    
    Args:
        logger: Logger instance
        trainer: Trainer instance
        mode: Training mode
    """
    if mode == 'distributed_social':
        mode_name = "Distributed Social-xLSTM"
    elif mode == 'single_vd_social':
        mode_name = "Single VD Social-xLSTM"
    elif mode == 'multi_vd_social':
        mode_name = "Multi-VD Social-xLSTM"
    else:
        mode_name = "Social-xLSTM"
    
    logger.info("=" * 70)
    logger.info(f"✅ {mode_name} Training Completed!")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # Get training information
    if hasattr(trainer, 'current_epoch'):
        logger.info(f"Total epochs completed: {trainer.current_epoch + 1}")
    
    if hasattr(trainer, 'callback_metrics'):
        metrics = trainer.callback_metrics
        if 'val_loss' in metrics:
            logger.info(f"Final validation loss: {float(metrics['val_loss']):.6f}")
        if 'train_loss' in metrics:
            logger.info(f"Final training loss: {float(metrics['train_loss']):.6f}")
    
    # Mode-specific information
    if mode == 'distributed_social':
        logger.info("\n🌐 Distributed Social-xLSTM Training Notes:")
        logger.info("• Each VD maintained independent xLSTM processing")
        logger.info("• Spatial social pooling enabled for neighbor interactions")
        logger.info("• Results demonstrate distributed social learning capabilities")
    
    logger.info(f"\n📁 Experiment results saved to: {getattr(trainer, 'log_dir', 'logs/')}")


def create_distributed_data_module(args, logger):
    """
    Create distributed data module for Social-xLSTM training.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        data_module: TrafficDataModule instance
    """
    from social_xlstm.dataset.core.datamodule import TrafficDataModule
    from social_xlstm.dataset.config.base import TrafficDatasetConfig
    import sys
    
    logger.info("Preparing distributed training data...")
    
    # Check if data file exists
    if not Path(args.data_path).exists():
        logger.error(f"Data file not found: {args.data_path}")
        logger.info("Please ensure data preprocessing steps are completed:")
        logger.info("1. conda activate social_xlstm")
        logger.info("2. snakemake --cores 4")
        logger.info("3. Or manually run preprocessing scripts")
        sys.exit(1)
    
    try:
        # Prepare selected VD IDs
        selected_vdids = None
        if hasattr(args, 'selected_vdids') and args.selected_vdids:
            selected_vdids = args.selected_vdids
            logger.info(f"  Selected VD IDs: {selected_vdids}")
        elif hasattr(args, 'num_vds') and args.num_vds:
            # Legacy support: use first N VDs
            from social_xlstm.dataset.storage.h5_reader import TrafficHDF5Reader
            reader = TrafficHDF5Reader(args.data_path)
            metadata = reader.get_metadata()
            available_vds = metadata['vdids']
            selected_vdids = available_vds[:args.num_vds]
            logger.info(f"  Using first {args.num_vds} VDs: {selected_vdids}")
        else:
            logger.info("  Using all available VDs")
        
        # Create dataset configuration
        dataset_config = TrafficDatasetConfig(
            hdf5_path=args.data_path,
            selected_vdids=selected_vdids,
            sequence_length=args.sequence_length,
            prediction_length=getattr(args, 'prediction_length', 1),
            batch_size=args.batch_size,
            train_ratio=getattr(args, 'train_ratio', 0.8),
            val_ratio=getattr(args, 'val_ratio', 0.2),
            test_ratio=1.0 - getattr(args, 'train_ratio', 0.8) - getattr(args, 'val_ratio', 0.2),
            num_workers=getattr(args, 'num_workers', 4),
            pin_memory=True
        )
        
        # Create distributed data module
        # Set distributed batch format for per-VD processing
        dataset_config.batch_format = 'distributed'
        datamodule = TrafficDataModule(dataset_config)
        datamodule.setup()
        
        logger.info("Distributed data loaded successfully:")
        logger.info(f"  Data file: {args.data_path}")
        logger.info(f"  Sequence length: {args.sequence_length}")
        logger.info(f"  Prediction length: {getattr(args, 'prediction_length', 1)}")
        logger.info(f"  Batch size: {args.batch_size}")
        
        return datamodule
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        logger.info("Common solutions:")
        logger.info("1. Check if data file path is correct")
        logger.info("2. Ensure data preprocessing is completed")
        logger.info("3. Check file permissions")
        logger.info("4. Verify selected VD IDs exist in the dataset")
        sys.exit(1)


def create_social_xlstm_model(args, num_features: int, logger):
    """
    Create Social-xLSTM model from arguments.
    
    Args:
        args: Command line arguments
        num_features: Number of input features
        logger: Logger instance
        
    Returns:
        model: DistributedSocialXLSTMModel instance
    """
    from social_xlstm.models.distributed_social_xlstm import DistributedSocialXLSTMModel
    from social_xlstm.models.xlstm import TrafficXLSTMConfig
    
    logger.info("Creating Distributed Social-xLSTM model...")
    
    # Create xLSTM configuration
    xlstm_config = TrafficXLSTMConfig()
    xlstm_config.input_size = num_features
    xlstm_config.hidden_size = args.hidden_size
    xlstm_config.num_blocks = args.num_blocks
    xlstm_config.embedding_dim = args.embedding_dim
    xlstm_config.sequence_length = args.sequence_length
    xlstm_config.prediction_length = getattr(args, 'prediction_length', 1)
    xlstm_config.slstm_at = args.slstm_at
    
    # Validate sLSTM positions
    if any(pos >= xlstm_config.num_blocks for pos in xlstm_config.slstm_at):
        raise ValueError(f"sLSTM positions {xlstm_config.slstm_at} exceed num_blocks={xlstm_config.num_blocks}")
    
    # Resolve aggregation method for Social Pooling
    aggregation_method = resolve_aggregation_method(args)
    
    # Create distributed Social-xLSTM model
    model = DistributedSocialXLSTMModel(
        xlstm_config=xlstm_config,
        num_features=num_features,
        hidden_dim=args.embedding_dim,
        prediction_length=getattr(args, 'prediction_length', 1),
        social_pool_type=aggregation_method,
        learning_rate=args.learning_rate,
        enable_gradient_checkpointing=getattr(args, 'enable_gradient_checkpointing', True),
        enable_spatial_pooling=getattr(args, 'enable_spatial_pooling', False),
        spatial_radius=getattr(args, 'spatial_radius', 2.0)
    )
    
    # Log model information
    model_info = model.get_model_info()
    logger.info(f"Model created successfully:")
    logger.info(f"  Type: Distributed Social-xLSTM")
    logger.info(f"  Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"  Trainable parameters: {model_info['trainable_parameters']:,}")
    logger.info(f"  Hidden dimension: {model_info['hidden_dim']}")
    logger.info(f"  Prediction length: {model_info['prediction_length']}")
    logger.info(f"  Spatial pooling: {args.enable_spatial_pooling if hasattr(args, 'enable_spatial_pooling') else False}")
    if hasattr(args, 'enable_spatial_pooling') and args.enable_spatial_pooling:
        logger.info(f"  Spatial radius: {args.spatial_radius}")
        logger.info(f"  Aggregation method: {aggregation_method}")
    
    return model


def get_social_pooling_warnings(args) -> List[str]:
    """
    Get Social Pooling specific warnings and tips.
    
    Args:
        args: Command line arguments
        
    Returns:
        list: List of warning/tip messages
    """
    warnings = []
    
    # Spatial pooling configuration warnings
    if hasattr(args, 'enable_spatial_pooling') and args.enable_spatial_pooling:
        if args.spatial_radius > 10.0:
            warnings.append("WARNING: Large spatial radius may affect performance and memory usage")
        
        if args.aggregation_method == 'attention':
            warnings.append("INFO: Using attention aggregation - may require more GPU memory")
        
        warnings.extend([
            "TIPS: Social Pooling training tips:",
            "   • Reduce batch_size if encountering OOM with spatial pooling",
            "   • Attention aggregation typically performs better but uses more memory",
            "   • Spatial radius should be appropriate for your geographic scale"
        ])
    else:
        warnings.append("INFO: Spatial pooling disabled - training will be equivalent to independent xLSTM per VD")
    
    # Multi-VD warnings
    if hasattr(args, 'selected_vdids') and args.selected_vdids and len(args.selected_vdids) > 10:
        warnings.append("WARNING: Large number of VDs may significantly increase training time")
    
    return warnings


# Test function
if __name__ == "__main__":
    # Test common functions
    logger = setup_logging()
    logger.info("Social Pooling common utilities module test successful")
    check_conda_environment(logger)
    
    # Test argument parser
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    logger.info("Argument parser test successful")
    
    # Test aggregation method resolution
    class MockArgs:
        aggregation_method = "attention"
        pool_type = "weighted_mean"
    
    args = MockArgs()
    method = resolve_aggregation_method(args)
    logger.info(f"Aggregation method resolution test: {method}")