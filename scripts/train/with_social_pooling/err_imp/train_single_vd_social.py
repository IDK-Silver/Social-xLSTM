#!/usr/bin/env python3
"""
Single VD Social-xLSTM Training Script

Train Social-xLSTM model on a single VD with spatial social pooling
using simulated neighboring VD data for proof-of-concept validation.

This script is primarily for testing and validation of Social Pooling
mechanisms when only one VD is available.

Usage:
    conda activate social_xlstm
    python scripts/train/with_social_pooling/train_single_vd_social.py \
        --data_path data.h5 \
        --select_vd_id VD-28-0740-000-001 \
        --enable_spatial_pooling \
        --aggregation_method attention

Author: Social-xLSTM Project Team
Date: 2025-08-24
"""

import argparse
import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from common import (
    setup_logging,
    check_conda_environment,
    add_common_arguments,
    print_training_start,
    print_training_complete,
    create_distributed_data_module,
    create_social_xlstm_model,
    get_social_pooling_warnings
)

from social_xlstm.training.recorder import TrainingRecorder


def parse_arguments():
    """Parse command line arguments for single VD Social training."""
    parser = argparse.ArgumentParser(
        description="Single VD Social-xLSTM Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add common arguments
    add_common_arguments(parser)
    
    # Single VD specific parameters
    parser.add_argument("--select_vd_id", type=str, required=True,
                        help="Specific VD ID to train on")
    
    # Override defaults for single VD training
    parser.set_defaults(
        experiment_name="single_vd_social_xlstm",
        batch_size=32,
        epochs=30,
        enable_spatial_pooling=True,  # Enable by default for social training
        spatial_radius=1.0,  # Smaller radius for single VD simulation
        aggregation_method="weighted_mean"
    )
    
    return parser.parse_args()


def create_trainer(args, recorder=None):
    """Create PyTorch Lightning trainer with callbacks and logger."""
    
    # Create experiment directory
    experiment_dir = Path(args.save_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=str(experiment_dir),
        name="tb_logs",
        version=None
    )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_dir / "checkpoints",
        filename=f"single-vd-social-xlstm-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=args.early_stopping_patience,
        verbose=True,
        mode="min"
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Training history callback
    if recorder:
        from social_xlstm.models.distributed_social_xlstm import TrainingHistoryCallback
        history_callback = TrainingHistoryCallback(recorder)
        callbacks.append(history_callback)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_value,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return trainer


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    check_conda_environment(logger)
    
    # Print start information
    print_training_start(logger, mode='single_vd_social')
    
    # Show warnings and tips
    warnings = get_social_pooling_warnings(args)
    if warnings:
        logger.info("\n⚠️  Configuration Notes:")
        for warning in warnings:
            logger.info(f"   {warning}")
    
    # Set deterministic training
    pl.seed_everything(42, workers=True)
    
    # Force selected_vdids to be a list with single VD
    args.selected_vdids = [args.select_vd_id]
    
    logger.info(f"\n📊 Training Configuration:")
    logger.info(f"  Selected VD: {args.select_vd_id}")
    logger.info(f"  Spatial Pooling: {args.enable_spatial_pooling}")
    if args.enable_spatial_pooling:
        logger.info(f"  Spatial Radius: {args.spatial_radius}m")
        logger.info(f"  Aggregation Method: {args.aggregation_method}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    
    # Create dataset configuration and data module
    logger.info("\n📈 Setting up data module...")
    datamodule = create_distributed_data_module(args, logger)
    
    # Get data information for model creation
    data_info = datamodule.get_data_info()
    num_features = data_info['num_features']
    
    logger.info(f"  Number of features: {num_features}")
    logger.info(f"  Available VDs: {len(data_info['vdids'])}")
    
    # Create model
    logger.info("\n🧠 Creating model...")
    model = create_social_xlstm_model(args, num_features, logger)
    
    # Initialize TrainingRecorder for standard format compliance
    logger.info("\n📝 Initializing TrainingRecorder...")
    model_config = {
        'model_type': 'DistributedSocialXLSTM',
        'num_features': num_features,
        'hidden_size': args.hidden_size,
        'num_blocks': args.num_blocks,
        'embedding_dim': args.embedding_dim,
        'slstm_at': args.slstm_at,
        'enable_spatial_pooling': args.enable_spatial_pooling,
        'spatial_radius': args.spatial_radius,
        'aggregation_method': args.aggregation_method,
        'selected_vd': args.select_vd_id
    }
    
    training_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'sequence_length': args.sequence_length,
        'prediction_length': args.prediction_length,
        'accelerator': args.accelerator,
        'precision': args.precision,
        'enable_gradient_checkpointing': args.enable_gradient_checkpointing,
        'training_mode': 'single_vd_social'
    }
    
    recorder = TrainingRecorder(
        experiment_name=args.experiment_name,
        model_config=model_config,
        training_config=training_config
    )
    
    # Create trainer
    logger.info("\n⚙️  Creating trainer...")
    trainer = create_trainer(args, recorder)
    
    # Dry run check
    if args.dry_run:
        logger.info("🏃 Dry run mode - setup completed successfully!")
        logger.info("  Model and data module are ready for training")
        logger.info("  Use without --dry_run to start actual training")
        return
    
    # Start training
    logger.info(f"\n🚀 Starting training for {args.epochs} epochs...")
    logger.info("  Press Ctrl+C to stop training gracefully")
    
    try:
        trainer.fit(model, datamodule)
        
        # Training completed successfully
        print_training_complete(logger, trainer, mode='single_vd_social')
        
        # Save additional outputs for Snakemake compatibility
        experiment_dir = Path(args.save_dir) / args.experiment_name
        
        # Save best model in PyTorch format
        best_model_path = experiment_dir / "best_model.pt"
        torch.save(model.state_dict(), best_model_path)
        
        # Save configuration
        config_path = experiment_dir / "config.json"
        import json
        with open(config_path, 'w') as f:
            json.dump({
                'model_config': model_config,
                'training_config': training_config,
                'args': vars(args)
            }, f, indent=2)
        
        # Save training history
        training_history_path = experiment_dir / "training_history.json"
        training_history = {
            'epochs': trainer.current_epoch + 1,
            'final_epoch': trainer.current_epoch,
            'best_model_path': str(best_model_path),
            'config_path': str(config_path)
        }
        
        # Try to extract more detailed metrics if available
        if hasattr(trainer, 'logged_metrics'):
            training_history['final_metrics'] = {k: float(v) if isinstance(v, torch.Tensor) else v 
                                               for k, v in trainer.logged_metrics.items()}
        
        with open(training_history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        logger.info(f"\n📁 Snakemake outputs created:")
        logger.info(f"  • {best_model_path}")
        logger.info(f"  • {config_path}")
        logger.info(f"  • {training_history_path}")
        
        logger.info(f"\n✅ Single VD Social-xLSTM training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Training interrupted by user")
        logger.info("  Partial results may be saved in checkpoint files")
    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()