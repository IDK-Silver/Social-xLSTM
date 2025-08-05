#!/usr/bin/env python3
"""
Distributed Social-xLSTM Training Script

Train the complete Social-xLSTM model using distributed per-VD xLSTM architecture
with spatial social pooling for multi-VD traffic prediction.

This script implements the final Task 3.3: complete end-to-end Social-xLSTM training
pipeline with distributed architecture, spatial pooling, and backward compatibility.

Usage:
    conda activate social_xlstm
    python scripts/train_distributed_social_xlstm.py
    python scripts/train_distributed_social_xlstm.py --enable_spatial_pooling --spatial_radius 2.5
    python scripts/train_distributed_social_xlstm.py --epochs 50 --batch_size 16

Features:
- Distributed per-VD xLSTM processing
- Spatial social pooling with configurable radius
- Backward compatibility with legacy mode
- PyTorch Lightning integration
- Comprehensive logging and monitoring

Author: Social-xLSTM Project Team
Date: 2025-08-02
"""

import argparse
import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import logging
from pathlib import Path
from typing import Dict, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from social_xlstm.models.xlstm import TrafficXLSTMConfig
from social_xlstm.models.distributed_social_xlstm import DistributedSocialXLSTMModel
from social_xlstm.data.distributed_datamodule import DistributedTrafficDataModule
from social_xlstm.dataset.config.base import TrafficDatasetConfig
from social_xlstm.training.recorder import TrainingRecorder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingHistoryCallback(pl.Callback):
    """
    PyTorch Lightning Callback to record training history
    using the project's standard TrainingRecorder.
    """
    def __init__(self, recorder: TrainingRecorder):
        super().__init__()
        self.recorder = recorder

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record epoch metrics after validation."""
        # Only record on main process to avoid duplicate records
        if not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics
        
        # Extract metrics safely with defaults
        train_loss = float(metrics.get('train_loss', 0.0))
        val_loss = float(metrics.get('val_loss', 0.0)) if 'val_loss' in metrics else None
        
        # Extract additional metrics
        train_metrics = {}
        val_metrics = {}
        
        # Extract MAE, R2, MSE metrics if available
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = float(value)
            if key.startswith('train_'):
                train_metrics[key.replace('train_', '')] = value
            elif key.startswith('val_'):
                val_metrics[key.replace('val_', '')] = value
        
        # Get learning rate from optimizer
        lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else 0.0
        
        # Record the epoch
        self.recorder.log_epoch(
            epoch=trainer.current_epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=lr
        )
        
        logger.debug(f"Recorded epoch {trainer.current_epoch} to TrainingRecorder")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Distributed Social-xLSTM model for traffic prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--data_path', type=str, 
                       default='blob/dataset/pre-processed/h5/traffic_features_dev.h5',
                       help='Path to HDF5 dataset file')
    parser.add_argument('--selected_vdids', type=str, nargs='*', default=None,
                       help='List of VD IDs to use for training (space-separated). If not provided, uses all VDs from HDF5 file.')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--sequence_length', type=int, default=12,
                       help='Input sequence length')
    parser.add_argument('--prediction_length', type=int, default=3,
                       help='Prediction horizon length')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation data ratio')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='xLSTM hidden dimension')
    parser.add_argument('--num_blocks', type=int, default=4,
                       help='Number of xLSTM blocks')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Input embedding dimension')
    parser.add_argument('--slstm_at', type=int, nargs='+', default=[1, 3],
                       help='Positions for sLSTM layers (space-separated list, e.g., 1 3)')
    
    # Social pooling parameters
    parser.add_argument('--enable_spatial_pooling', action='store_true',
                       help='Enable spatial social pooling (default: legacy mode)')
    parser.add_argument('--spatial_radius', type=float, default=2.0,
                       help='Spatial pooling radius in meters')
    parser.add_argument('--pool_type', type=str, default='mean',
                       choices=['mean', 'max', 'weighted_mean'],
                       help='Social pooling aggregation method')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--enable_gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing for memory efficiency')
    
    # Infrastructure parameters
    parser.add_argument('--accelerator', type=str, default='auto',
                       choices=['auto', 'cpu', 'gpu', 'mps'],
                       help='Training accelerator')
    parser.add_argument('--devices', type=int, default=1,
                       help='Number of devices to use')
    parser.add_argument('--precision', type=str, default='32',
                       choices=['16', '32', 'bf16'],
                       help='Training precision')
    
    # Logging and checkpointing
    parser.add_argument('--experiment_name', type=str, default='distributed_social_xlstm',
                       help='Experiment name for logging')
    parser.add_argument('--save_dir', type=str, default='logs',
                       help='Directory to save logs and checkpoints')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--gradient_clip_value', type=float, default=0.5,
                       help='Value for gradient clipping (max norm)')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                       help='Patience epochs for LR scheduler before reduction')
    
    # Development and debugging
    parser.add_argument('--limit_train_batches', type=float, default=1.0,
                       help='Limit training batches (for development)')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                       help='Limit validation batches (for development)')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run one batch for development testing')
    parser.add_argument('--dry_run', action='store_true',
                       help='Set up everything but do not start training')
    
    return parser.parse_args()


def create_xlstm_config(args, num_features: int) -> TrafficXLSTMConfig:
    """Create xLSTM configuration from arguments"""
    config = TrafficXLSTMConfig()
    config.input_size = num_features  # Use actual number of features from dataset
    config.hidden_size = args.hidden_size
    config.num_blocks = args.num_blocks
    config.embedding_dim = args.embedding_dim
    config.sequence_length = args.sequence_length
    config.prediction_length = args.prediction_length
    config.slstm_at = args.slstm_at
    
    # Validate sLSTM positions
    if any(pos >= config.num_blocks for pos in config.slstm_at):
        raise ValueError(f"sLSTM positions {config.slstm_at} exceed num_blocks={config.num_blocks}. "
                        f"Valid positions: 0 to {config.num_blocks - 1}")
    
    logger.info(f"xLSTM Config: hidden_size={config.hidden_size}, "
               f"num_blocks={config.num_blocks}, embedding_dim={config.embedding_dim}, "
               f"slstm_at={config.slstm_at}")
    
    return config


def create_dataset_config(args) -> TrafficDatasetConfig:
    """Create dataset configuration from arguments"""
    config = TrafficDatasetConfig(
        hdf5_path=args.data_path,
        selected_vdids=args.selected_vdids,  # Add VD selection support
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Log VD selection info
    if args.selected_vdids:
        logger.info(f"Dataset Config: hdf5_path={config.hdf5_path}, "
                   f"selected_vdids={args.selected_vdids}, "
                   f"batch_size={config.batch_size}, sequence_length={config.sequence_length}")
    else:
        logger.info(f"Dataset Config: hdf5_path={config.hdf5_path}, "
                   f"using all VDs from HDF5 file, "
                   f"batch_size={config.batch_size}, sequence_length={config.sequence_length}")
    
    return config


def create_model(args, num_features: int) -> DistributedSocialXLSTMModel:
    """Create Distributed Social-xLSTM model"""
    xlstm_config = create_xlstm_config(args, num_features)
    
    model = DistributedSocialXLSTMModel(
        xlstm_config=xlstm_config,
        num_features=num_features,
        hidden_dim=args.embedding_dim,
        prediction_length=args.prediction_length,
        social_pool_type=args.pool_type,
        learning_rate=args.learning_rate,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        enable_spatial_pooling=args.enable_spatial_pooling,
        spatial_radius=args.spatial_radius
    )
    
    # Log model information
    model_info = model.get_model_info()
    logger.info(f"Model created with {model_info['total_parameters']:,} parameters")
    logger.info(f"  Trainable: {model_info['trainable_parameters']:,}")
    logger.info(f"  Spatial pooling: {args.enable_spatial_pooling}")
    if args.enable_spatial_pooling:
        logger.info(f"  Spatial radius: {args.spatial_radius}")
        logger.info(f"  Pool type: {args.pool_type}")
    
    return model


def create_trainer(args, recorder: TrainingRecorder = None) -> pl.Trainer:
    """Create PyTorch Lightning trainer with callbacks and logger"""
    
    # Create experiment directory
    experiment_dir = Path(args.save_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=args.save_dir,
        name=args.experiment_name,
        version=None  # Auto-increment version
    )
    
    # Callbacks
    callbacks = []
    
    # Add TrainingHistoryCallback if recorder is provided
    if recorder is not None:
        history_callback = TrainingHistoryCallback(recorder)
        callbacks.append(history_callback)
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_dir / "checkpoints",
        filename="distributed-social-xlstm-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.early_stopping_patience,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=tb_logger,
        callbacks=callbacks,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_model_summary=True
    )
    
    return trainer


def save_experiment_config(args, experiment_dir: Path):
    """Save experiment configuration for reproducibility"""
    config_dict = vars(args).copy()
    
    # Add system information
    config_dict.update({
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    })
    
    config_file = experiment_dir / "experiment_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Experiment configuration saved to {config_file}")


def create_snakemake_outputs(args, trainer: pl.Trainer, experiment_dir: Path, recorder: TrainingRecorder = None):
    """Create output files in the format expected by Snakemake rules"""
    import shutil
    
    # Define Snakemake output paths (in the experiment base directory)
    base_dir = Path(args.save_dir) / args.experiment_name
    base_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = base_dir / "best_model.pt"
    config_path = base_dir / "config.json"
    training_history_path = base_dir / "training_history.json"
    
    # 1. Copy best model checkpoint as best_model.pt
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        shutil.copy2(trainer.checkpoint_callback.best_model_path, best_model_path)
        logger.info(f"Best model copied to: {best_model_path}")
    else:
        # Create a placeholder if no best model was saved
        torch.save({'message': 'No best model available'}, best_model_path)
        logger.warning(f"No best model found, created placeholder: {best_model_path}")
    
    # 2. Copy experiment config as config.json
    if (experiment_dir / "experiment_config.json").exists():
        shutil.copy2(experiment_dir / "experiment_config.json", config_path)
        logger.info(f"Config copied to: {config_path}")
    
    # 3. Save training history using TrainingRecorder (standard format)
    if recorder is not None:
        # Use TrainingRecorder for standard format compliance
        recorder.save(training_history_path)
        logger.info(f"Training history saved using TrainingRecorder to: {training_history_path}")
    else:
        # Fallback to legacy format (should not happen with new implementation)
        logger.warning("No TrainingRecorder provided, using legacy format")
        training_history = {
            'epochs': trainer.current_epoch + 1,
            'best_val_loss': float(trainer.checkpoint_callback.best_model_score) if trainer.checkpoint_callback else None,
            'best_model_path': str(trainer.checkpoint_callback.best_model_path) if trainer.checkpoint_callback else None,
            'tensorboard_logs': str(trainer.logger.log_dir),
            'training_completed': True
        }
        
        # Try to extract more detailed metrics if available
        if hasattr(trainer, 'logged_metrics'):
            training_history['final_metrics'] = {k: float(v) if isinstance(v, torch.Tensor) else v 
                                               for k, v in trainer.logged_metrics.items()}
        
        with open(training_history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        logger.info(f"Training history saved to: {training_history_path}")
    
    print(f"\\n📁 Snakemake outputs created:")
    print(f"  • {best_model_path}")
    print(f"  • {config_path}")
    print(f"  • {training_history_path}")


def print_training_summary(args, model: DistributedSocialXLSTMModel, datamodule: DistributedTrafficDataModule):
    """Print comprehensive training summary"""
    print("=" * 80)
    print("🚀 DISTRIBUTED SOCIAL-xLSTM TRAINING STARTED")
    print("=" * 80)
    
    # Model information
    model_info = model.get_model_info()
    print(f"📊 Model Architecture:")
    print(f"  • Total Parameters: {model_info['total_parameters']:,}")
    print(f"  • Trainable Parameters: {model_info['trainable_parameters']:,}")
    print(f"  • Hidden Dimension: {model_info['hidden_dim']}")
    print(f"  • Prediction Length: {model_info['prediction_length']}")
    
    # Social pooling information
    print(f"\\n🌐 Social Pooling Configuration:")
    if args.enable_spatial_pooling:
        print(f"  • Mode: Spatial-aware pooling")
        print(f"  • Spatial Radius: {args.spatial_radius} meters")
        print(f"  • Aggregation: {args.pool_type}")
    else:
        print(f"  • Mode: Legacy neighbor-based pooling")
        print(f"  • Aggregation: {args.pool_type}")
    
    # Data information
    data_info = datamodule.get_data_info()
    print(f"\\n📈 Dataset Information:")
    print(f"  • Data Path: {args.data_path}")
    print(f"  • Number of VDs: {len(data_info['vdids'])}")
    print(f"  • Features: {data_info['num_features']}")
    print(f"  • Sequence Length: {args.sequence_length}")
    print(f"  • Batch Size: {args.batch_size}")
    
    # Training configuration
    print(f"\\n⚙️  Training Configuration:")
    print(f"  • Epochs: {args.epochs}")
    print(f"  • Learning Rate: {args.learning_rate}")
    print(f"  • Accelerator: {args.accelerator}")
    print(f"  • Precision: {args.precision}")
    print(f"  • Gradient Checkpointing: {args.enable_gradient_checkpointing}")
    
    print("=" * 80)


def main():
    """Main training function"""
    args = parse_args()
    
    logger.info("Starting Distributed Social-xLSTM training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set up deterministic training
    pl.seed_everything(42, workers=True)
    
    # Create dataset configuration and data module
    logger.info("Setting up data module...")
    dataset_config = create_dataset_config(args)
    datamodule = DistributedTrafficDataModule(dataset_config)
    datamodule.setup()
    
    # Get data information for model creation
    data_info = datamodule.get_data_info()
    num_features = data_info['num_features']
    
    # Create model
    logger.info("Creating model...")
    model = create_model(args, num_features)
    
    # Initialize TrainingRecorder for standard format compliance
    logger.info("Initializing TrainingRecorder...")
    model_config = {
        'model_type': 'DistributedSocialXLSTM',
        'num_features': num_features,
        'hidden_size': args.hidden_size,
        'num_blocks': args.num_blocks,
        'embedding_dim': args.embedding_dim,
        'slstm_at': args.slstm_at,
        'enable_spatial_pooling': args.enable_spatial_pooling,
        'spatial_radius': args.spatial_radius,
        'pool_type': args.pool_type
    }
    
    training_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'sequence_length': args.sequence_length,
        'prediction_length': args.prediction_length,
        'accelerator': args.accelerator,
        'precision': args.precision,
        'enable_gradient_checkpointing': args.enable_gradient_checkpointing
    }
    
    recorder = TrainingRecorder(
        experiment_name=args.experiment_name,
        model_config=model_config,
        training_config=training_config
    )
    
    # Create trainer with TrainingHistoryCallback
    logger.info("Setting up trainer...")
    trainer = create_trainer(args, recorder)
    
    # Save experiment configuration
    experiment_dir = Path(args.save_dir) / args.experiment_name / f"version_{trainer.logger.version}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(args, experiment_dir)
    
    # Print training summary
    print_training_summary(args, model, datamodule)
    
    if args.dry_run:
        logger.info("Dry run completed. Exiting without training.")
        return
    
    # Start training
    try:
        logger.info("Starting training...")
        trainer.fit(model, datamodule)
        
        # Training completed
        print("\\n" + "=" * 80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Print best model information
        if trainer.checkpoint_callback.best_model_path:
            print(f"💾 Best model saved to: {trainer.checkpoint_callback.best_model_path}")
            print(f"📊 Best validation loss: {trainer.checkpoint_callback.best_model_score:.6f}")
        
        # Print logs location
        print(f"📈 TensorBoard logs: {trainer.logger.log_dir}")
        print(f"⚙️  Experiment config: {experiment_dir / 'experiment_config.json'}")
        
        print("\\n🚀 To view training progress:")
        print(f"   tensorboard --logdir {args.save_dir}")
        
        print("=" * 80)
        
        # Create Snakemake-compatible output files
        create_snakemake_outputs(args, trainer, experiment_dir, recorder)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\\n⚠️  Training interrupted. Partial results may be saved.")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()