#!/usr/bin/env python3
"""
Common utilities for training scripts without Social Pooling

This module provides essential shared utilities for training LSTM/xLSTM models
without spatial interactions or Social Pooling mechanisms.

Simplified from the original to remove redundant functionality that's now
handled by the specialized trainer architecture.

Author: Social-xLSTM Project Team
"""

import logging
import os
import argparse
from datetime import datetime
from pathlib import Path

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
    Add common command line arguments for training scripts.
    
    Args:
        parser: ArgumentParser instance
    """
    # Data parameters
    parser.add_argument("--data_path", type=str, default="/tmp/tmpdhc_pz_1.h5",
                        help="HDF5 data file path")
    parser.add_argument("--sequence_length", type=int, default=12,
                        help="Time sequence length")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="lstm", 
                        choices=["lstm", "xlstm"],
                        help="Model type (lstm: Standard LSTM, xlstm: Extended LSTM)")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden layer size (for LSTM) or embedding dim (for xLSTM)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of LSTM layers (ignored for xLSTM)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout ratio")
    
    # xLSTM specific parameters
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="xLSTM embedding dimension (defaults to hidden_size)")
    parser.add_argument("--num_blocks", type=int, default=6,
                        help="Number of xLSTM blocks")
    parser.add_argument("--slstm_at", type=int, nargs='+', default=[1, 3],
                        help="Positions for sLSTM blocks (e.g., --slstm_at 1 3)")
    parser.add_argument("--context_length", type=int, default=256,
                        help="xLSTM context length")
    parser.add_argument("--backend", type=str, default="vanilla",
                        choices=["vanilla", "cuda"],
                        help="xLSTM computation backend")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd"],
                        help="Optimizer type")
    parser.add_argument("--scheduler_type", type=str, default="reduce_on_plateau",
                        choices=["reduce_on_plateau", "step", "cosine"],
                        help="Learning rate scheduler type")
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                        help="Early stopping patience")
    
    # Device parameters
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Training device")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable mixed precision training")
    
    # Experiment parameters
    parser.add_argument("--experiment_name", type=str, default="lstm_without_social_pooling",
                        help="Experiment name")
    parser.add_argument("--save_dir", type=str, default="blob/experiments",
                        help="Experiment results save directory")


def print_training_start(logger, mode='single_vd'):
    """
    Print training start information.
    
    Args:
        logger: Logger instance
        mode: Training mode
    """
    if mode == 'single_vd':
        mode_name = "Single VD"
    elif mode == 'multi_vd':
        mode_name = "Multi-VD"
    elif mode == 'independent_multi_vd':
        mode_name = "Multi-VD Independent"
    else:
        mode_name = "Unknown"
    
    logger.info("=" * 60)
    logger.info(f"{mode_name} LSTM Training Started (Without Social Pooling)")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


def print_training_complete(logger, trainer, mode='single_vd'):
    """
    Print training completion information.
    
    Args:
        logger: Logger instance
        trainer: Trainer instance
        mode: Training mode
    """
    if mode == 'single_vd':
        mode_name = "Single VD"
    elif mode == 'multi_vd':
        mode_name = "Multi-VD"
    elif mode == 'independent_multi_vd':
        mode_name = "Multi-VD Independent"
    else:
        mode_name = "Unknown"
    
    logger.info("=" * 60)
    logger.info(f"{mode_name} Training Completed!")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # Get training history
    history = trainer.training_history
    if history['train_loss']:
        logger.info(f"Final train loss: {history['train_loss'][-1]:.6f}")
    if history['val_loss']:
        logger.info(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")
    
    logger.info(f"Total epochs: {len(history['train_loss'])}")
    logger.info(f"Experiment results saved to: {trainer.experiment_dir}")
    
    # Mode-specific information
    if mode == 'multi_vd':
        logger.info("\nMulti-VD Training Notes:")
        logger.info("• This was independent VD training (no spatial interaction)")
        logger.info("• Results can be used for comparison with Social-xLSTM")
        logger.info("• Model learned individual VD patterns without spatial pooling")
    
    # Saved files
    logger.info(f"\nSaved files:")
    logger.info(f"• Model configuration: {trainer.experiment_dir}/config.json")
    logger.info(f"• Best model weights: {trainer.experiment_dir}/best_model.pt")


def create_data_module(args, logger):
    """
    Create data module for training.
    
    Args:
        args: Command line arguments
        logger: Logger instance
    
    Returns:
        data_module: TrafficDataModule instance
    """
    from social_xlstm.dataset.core.datamodule import TrafficDataModule
    from social_xlstm.dataset.config.base import TrafficDatasetConfig
    import sys
    
    logger.info("Preparing training data...")
    
    # Check if data file exists
    if not Path(args.data_path).exists():
        logger.error(f"Data file not found: {args.data_path}")
        logger.info("Please ensure data preprocessing steps are completed:")
        logger.info("1. conda activate social_xlstm")
        logger.info("2. snakemake --cores 4")
        logger.info("3. Or manually run preprocessing scripts")
        sys.exit(1)
    
    try:
        # Prepare selected VD IDs based on training mode
        selected_vdids = None
        
        # Single VD training
        if hasattr(args, 'select_vd_id') and args.select_vd_id:
            selected_vdids = [args.select_vd_id]
            logger.info(f"  Selected VD ID: {args.select_vd_id}")
        
        # Multi-VD training
        elif hasattr(args, 'vd_ids') and args.vd_ids:
            # Use specific VD IDs if provided
            selected_vdids = [vd.strip() for vd in args.vd_ids.split(',')]
            logger.info(f"  Selected VD IDs: {selected_vdids}")
        elif hasattr(args, 'num_vds') and args.num_vds:
            # Use first N VDs if num_vds is specified
            from social_xlstm.dataset.storage.h5_reader import TrafficHDF5Reader
            reader = TrafficHDF5Reader(args.data_path)
            metadata = reader.get_metadata()
            available_vds = metadata['vdids']
            selected_vdids = available_vds[:args.num_vds]
            logger.info(f"  Using first {args.num_vds} VDs: {selected_vdids}")
        else:
            logger.info("  Using all available VDs")
        
        # Create data configuration
        data_config = TrafficDatasetConfig(
            hdf5_path=args.data_path,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            selected_vdids=selected_vdids,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            num_workers=4
        )
        
        # Create data module
        data_module = TrafficDataModule(data_config)
        data_module.setup()
        
        logger.info("Data loaded successfully:")
        logger.info(f"  Data file: {args.data_path}")
        logger.info(f"  Sequence length: {args.sequence_length}")
        logger.info(f"  Batch size: {args.batch_size}")
        
        return data_module
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        logger.info("Common solutions:")
        logger.info("1. Check if data file path is correct")
        logger.info("2. Ensure data preprocessing is completed")
        logger.info("3. Check file permissions")
        sys.exit(1)


def create_model_for_single_vd(args, data_module, logger):
    """
    Create model for single VD training.
    
    Args:
        args: Command line arguments
        data_module: Data module
        logger: Logger instance
    
    Returns:
        model: TrafficLSTM or TrafficXLSTM model
    """
    # Get feature information
    sample_batch = next(iter(data_module.train_dataloader()))
    actual_features = sample_batch['input_seq'].shape[-1]
    
    logger.info(f"Creating single VD model with {actual_features} features")
    
    if args.model_type == "lstm":
        from social_xlstm.models.lstm import TrafficLSTM
        model = TrafficLSTM.create_single_vd_model(
            input_size=actual_features,
            output_size=actual_features,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    elif args.model_type == "xlstm":
        from social_xlstm.models.xlstm import TrafficXLSTM, TrafficXLSTMConfig
        
        # Create xLSTM configuration
        config = TrafficXLSTMConfig(
            input_size=actual_features,
            output_size=actual_features,
            embedding_dim=getattr(args, 'embedding_dim', args.hidden_size),
            num_blocks=getattr(args, 'num_blocks', 6),
            slstm_at=getattr(args, 'slstm_at', [1, 3]),
            dropout=args.dropout,
            multi_vd_mode=False
        )
        
        model = TrafficXLSTM(config)
        logger.info(f"xLSTM configuration:")
        logger.info(f"  Embedding dim: {config.embedding_dim}")
        logger.info(f"  Num blocks: {config.num_blocks}")
        logger.info(f"  sLSTM positions: {config.slstm_at}")
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Print model info
    model_info = model.get_model_info()
    logger.info(f"Model created successfully:")
    logger.info(f"  Type: {model_info['model_type']} ({args.model_type.upper()})")
    logger.info(f"  Parameters: {model_info['total_parameters']:,}")
    
    if args.model_type == "lstm":
        logger.info(f"  Hidden size: {model_info['config']['hidden_size']}")
        logger.info(f"  Layers: {model_info['config']['num_layers']}")
    elif args.model_type == "xlstm":
        logger.info(f"  Embedding dim: {model_info['embedding_dim']}")
        logger.info(f"  Blocks: {model_info['num_blocks']}")
        logger.info(f"  sLSTM positions: {model_info['slstm_positions']}")
    
    return model


def create_model_for_multi_vd(args, data_module, logger):
    """
    Create model for multi VD training.
    
    Args:
        args: Command line arguments
        data_module: Data module
        logger: Logger instance
    
    Returns:
        model: TrafficLSTM model
    """
    from social_xlstm.models.lstm import TrafficLSTM
    
    # Get feature information
    sample_batch = next(iter(data_module.train_dataloader()))
    input_shape = sample_batch['input_seq'].shape
    
    # Handle both 4D and 3D input formats
    if len(input_shape) == 4:
        # 4D format: [batch, seq, num_vds, features_per_vd]
        batch_size, seq_len, actual_num_vds, features_per_vd = input_shape
        flattened_features = actual_num_vds * features_per_vd
    elif len(input_shape) == 3:
        # 3D format: [batch, seq, flattened_features] (already flattened)
        batch_size, seq_len, flattened_features = input_shape
        features_per_vd = 5  # Standard traffic features
        actual_num_vds = flattened_features // features_per_vd
    else:
        raise ValueError(f"Unexpected input shape: {input_shape}")
    
    logger.info(f"Creating multi VD model with {flattened_features} total features, {actual_num_vds} VDs ({features_per_vd} features per VD)")
    
    if args.model_type == "lstm":
        from social_xlstm.models.lstm import TrafficLSTM
        model = TrafficLSTM.create_multi_vd_model(
            num_vds=actual_num_vds,
            input_size=features_per_vd,  # Per-VD features, not flattened
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    elif args.model_type == "xlstm":
        from social_xlstm.models.xlstm import TrafficXLSTM, TrafficXLSTMConfig
        
        # Create xLSTM configuration for multi-VD
        config = TrafficXLSTMConfig(
            input_size=flattened_features,
            output_size=flattened_features,
            embedding_dim=getattr(args, 'embedding_dim', args.hidden_size),
            num_blocks=getattr(args, 'num_blocks', 6),
            slstm_at=getattr(args, 'slstm_at', [1, 3]),
            dropout=args.dropout,
            multi_vd_mode=True,
            num_vds=actual_num_vds
        )
        
        model = TrafficXLSTM(config)
        logger.info(f"xLSTM configuration:")
        logger.info(f"  Embedding dim: {config.embedding_dim}")
        logger.info(f"  Num blocks: {config.num_blocks}")
        logger.info(f"  sLSTM positions: {config.slstm_at}")
        logger.info(f"  Multi-VD mode: {config.multi_vd_mode}")
        logger.info(f"  Num VDs: {config.num_vds}")
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Print model info
    model_info = model.get_model_info()
    logger.info(f"Model created successfully:")
    logger.info(f"  Type: {model_info['model_type']} ({args.model_type.upper()})")
    logger.info(f"  Parameters: {model_info['total_parameters']:,}")
    logger.info(f"  Hidden size: {args.hidden_size}")
    logger.info(f"  Layers: {args.num_layers}")
    logger.info(f"  VDs: {actual_num_vds}")
    
    return model


def get_device_warnings(device, mode='single_vd'):
    """
    Get device-related warnings.
    
    Args:
        device: Device string ('cpu' or 'cuda')
        mode: Training mode
    
    Returns:
        list: List of warning messages
    """
    warnings = []
    
    if device == "cpu":
        if mode == 'single_vd':
            warnings.append("WARNING: Using CPU - recommend GPU for better training speed")
        else:
            warnings.append("WARNING: Using CPU - multi-VD training is much slower on CPU")
    
    if mode == 'multi_vd' and device == "cuda":
        warnings.extend([
            "TIPS: Multi-VD training memory tips:",
            "   • Reduce batch_size if encountering OOM",
            "   • Enable mixed_precision for better performance",
            "   • Recommend GPU with at least 8GB VRAM"
        ])
    
    return warnings


# Test function
if __name__ == "__main__":
    # Test common functions
    logger = setup_logging()
    logger.info("Common utilities module test successful")
    check_conda_environment(logger)
    
    # Test argument parser
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    logger.info("Argument parser test successful")