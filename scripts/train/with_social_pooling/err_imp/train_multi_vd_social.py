#!/usr/bin/env python3
"""
Multi-VD Social-xLSTM Training Script

Train Social-xLSTM model on multiple VDs with full spatial social pooling
capabilities for comprehensive multi-location traffic prediction.

This script enables the complete Social-xLSTM experience with real
spatial interactions between multiple VD locations.

Usage:
    conda activate social_xlstm
    python scripts/train/with_social_pooling/train_multi_vd_social.py \
        --data_path data.h5 \
        --selected_vdids VD-28-0740-000-001 VD-11-0020-008-001 VD-13-0660-000-002 \
        --enable_spatial_pooling \
        --aggregation_method attention \
        --spatial_radius 2.5

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
    """Parse command line arguments for multi-VD Social training."""
    parser = argparse.ArgumentParser(
        description="Multi-VD Social-xLSTM Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add common arguments
    add_common_arguments(parser)
    
    # Multi-VD specific parameters
    parser.add_argument("--min_vds", type=int, default=3,
                        help="Minimum number of VDs required for training")
    parser.add_argument("--max_vds", type=int, default=10,
                        help="Maximum number of VDs to use for training")
    
    # Override defaults for multi-VD training
    parser.set_defaults(
        experiment_name="multi_vd_social_xlstm",
        batch_size=16,  # Smaller batch for memory efficiency
        epochs=50,
        enable_spatial_pooling=True,  # Enable by default for social training
        spatial_radius=2.0,  # Appropriate for multi-VD scenarios
        aggregation_method="weighted_mean",
        max_neighbors=8
    )
    
    return parser.parse_args()


def validate_vd_selection(args, logger):
    """Validate VD selection for multi-VD training."""
    if hasattr(args, 'selected_vdids') and args.selected_vdids:
        num_vds = len(args.selected_vdids)
        
        if num_vds < args.min_vds:
            logger.error(f"Insufficient VDs: got {num_vds}, need at least {args.min_vds}")
            logger.info("Multi-VD Social training requires multiple locations for spatial interaction")
            sys.exit(1)
        
        if num_vds > args.max_vds:
            logger.warning(f"Large number of VDs ({num_vds}) may impact training performance")
            logger.info("Consider reducing the number of VDs or increasing computational resources")
        
        logger.info(f"✅ VD selection validated: {num_vds} VDs selected")
        return True
    
    elif hasattr(args, 'num_vds') and args.num_vds:
        if args.num_vds < args.min_vds:
            logger.error(f"Insufficient VDs: requested {args.num_vds}, need at least {args.min_vds}")
            sys.exit(1)
        
        logger.info(f"✅ Will use first {args.num_vds} VDs from dataset")
        return True
    
    else:
        logger.warning("No specific VD selection - will use all available VDs")
        logger.info("This may result in very long training times")
        return False


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
        filename=f"multi-vd-social-xlstm-{{epoch:02d}}-{{val_loss:.4f}}",
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
    
    # Create trainer with multi-VD optimized settings
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
        enable_model_summary=True,
        accumulate_grad_batches=2 if args.batch_size < 16 else 1  # Gradient accumulation for small batches
    )
    
    return trainer


def print_multi_vd_summary(args, datamodule, model, logger):
    """Print comprehensive multi-VD training summary."""
    logger.info("\n" + "="*80)
    logger.info("🌐 MULTI-VD SOCIAL-xLSTM TRAINING SUMMARY")
    logger.info("="*80)
    
    # Data information
    data_info = datamodule.get_data_info()
    logger.info(f"\n📈 Dataset Information:")
    logger.info(f"  • Data Path: {args.data_path}")
    logger.info(f"  • Total VDs Available: {len(data_info['vdids'])}")
    logger.info(f"  • Selected VDs: {len(args.selected_vdids) if hasattr(args, 'selected_vdids') and args.selected_vdids else 'All'}")
    logger.info(f"  • Features per VD: {data_info['num_features']}")
    logger.info(f"  • Sequence Length: {args.sequence_length}")
    logger.info(f"  • Batch Size: {args.batch_size}")
    
    # Model information
    model_info = model.get_model_info()
    logger.info(f"\n🧠 Model Architecture:")
    logger.info(f"  • Type: Distributed Social-xLSTM")
    logger.info(f"  • Total Parameters: {model_info['total_parameters']:,}")
    logger.info(f"  • Trainable Parameters: {model_info['trainable_parameters']:,}")
    logger.info(f"  • Hidden Dimension: {model_info['hidden_dim']}")
    logger.info(f"  • xLSTM Blocks: {args.num_blocks}")
    logger.info(f"  • sLSTM Positions: {args.slstm_at}")
    
    # Social Pooling information
    logger.info(f"\n🌐 Social Pooling Configuration:")
    if args.enable_spatial_pooling:
        logger.info(f"  • Status: Enabled ✅")
        logger.info(f"  • Spatial Radius: {args.spatial_radius} km")
        logger.info(f"  • Aggregation Method: {args.aggregation_method}")
        logger.info(f"  • Max Neighbors: {args.max_neighbors}")
        logger.info(f"  • Expected Benefit: Spatial context learning across VDs")
    else:
        logger.info(f"  • Status: Disabled ❌")
        logger.info(f"  • Mode: Independent VD processing (no spatial interaction)")
    
    # Training configuration
    logger.info(f"\n⚙️  Training Configuration:")
    logger.info(f"  • Epochs: {args.epochs}")
    logger.info(f"  • Learning Rate: {args.learning_rate}")
    logger.info(f"  • Early Stopping Patience: {args.early_stopping_patience}")
    logger.info(f"  • Gradient Clipping: {args.gradient_clip_value}")
    logger.info(f"  • Accelerator: {args.accelerator}")
    logger.info(f"  • Precision: {args.precision}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    check_conda_environment(logger)
    
    # Print start information
    print_training_start(logger, mode='multi_vd_social')
    
    # Validate VD selection
    validate_vd_selection(args, logger)
    
    # Show warnings and tips
    warnings = get_social_pooling_warnings(args)
    if warnings:
        logger.info("\n⚠️  Configuration Notes:")
        for warning in warnings:
            logger.info(f"   {warning}")
    
    # Set deterministic training
    pl.seed_everything(42, workers=True)
    
    # Create dataset configuration and data module
    logger.info("\n📈 Setting up distributed data module...")
    datamodule = create_distributed_data_module(args, logger)
    
    # Get data information for model creation
    data_info = datamodule.get_data_info()
    num_features = data_info['num_features']
    
    # Update args with actual VD selection if needed
    if not (hasattr(args, 'selected_vdids') and args.selected_vdids):
        if hasattr(args, 'num_vds') and args.num_vds:
            args.selected_vdids = data_info['vdids'][:args.num_vds]
        else:
            args.selected_vdids = data_info['vdids']
    
    logger.info(f"  Final VD selection: {len(args.selected_vdids)} VDs")
    
    # Create model
    logger.info("\n🧠 Creating distributed Social-xLSTM model...")
    model = create_social_xlstm_model(args, num_features, logger)
    
    # Print comprehensive summary
    print_multi_vd_summary(args, datamodule, model, logger)
    
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
        'num_selected_vds': len(args.selected_vdids),
        'selected_vdids': args.selected_vdids[:5]  # Store first 5 for reference
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
        'training_mode': 'multi_vd_social'
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
        logger.info("\n🏃 Dry run mode - setup completed successfully!")
        logger.info("  Multi-VD model and data module are ready for training")
        logger.info("  Social pooling mechanisms are configured")
        logger.info("  Use without --dry_run to start actual training")
        return
    
    # Start training
    logger.info(f"\n🚀 Starting multi-VD Social-xLSTM training...")
    logger.info(f"  Training {len(args.selected_vdids)} VDs for {args.epochs} epochs")
    logger.info(f"  Expected training time: ~{args.epochs * len(args.selected_vdids) * 0.5:.1f} minutes")
    logger.info("  Press Ctrl+C to stop training gracefully")
    
    try:
        trainer.fit(model, datamodule)
        
        # Training completed successfully
        print_training_complete(logger, trainer, mode='multi_vd_social')
        
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
            'num_vds_trained': len(args.selected_vdids),
            'spatial_pooling_enabled': args.enable_spatial_pooling,
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
        
        logger.info(f"\n✅ Multi-VD Social-xLSTM training completed successfully!")
        logger.info(f"  Trained on {len(args.selected_vdids)} VDs with spatial interaction")
        logger.info(f"  Model learned distributed social patterns across locations")
        
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