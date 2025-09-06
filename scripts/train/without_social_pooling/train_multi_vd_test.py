#!/usr/bin/env python3
"""
Multi-VD LSTM Model Training Script (Independent VD Training, Without Social Pooling)

Train multiple independent VD traffic prediction models using unified training system.
This script trains LSTM/xLSTM models without any spatial relationships or Social Pooling.

IMPORTANT: This is NOT Social Pooling - each VD is trained independently.
No spatial relationships or interactions between VDs are considered.
This is purely for establishing baseline performance for comparison.

Usage:
    conda activate social_xlstm
    python scripts/train/without_social_pooling/train_multi_vd.py
    python scripts/train/without_social_pooling/train_multi_vd.py --model_type lstm --num_vds 5 --epochs 100
    python scripts/train/without_social_pooling/train_multi_vd.py --model_type xlstm --vd_ids VD001,VD002,VD003

Author: Social-xLSTM Project Team
"""

import argparse
import sys
import torch

# Import common functions
from common_test import (
    setup_logging, add_common_arguments, print_training_start, print_training_complete,
    create_data_module, create_model_for_multi_vd, get_device_warnings
)

from social_xlstm.training import MultiVDTrainer, MultiVDTrainingConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train multiple independent VD traffic prediction LSTM models (without Social Pooling)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add common arguments
    add_common_arguments(parser)
    
    # Multi-VD specific parameters
    parser.add_argument(
        "--num_vds", 
        type=int, 
        default=5,
        help="Number of VDs to train independently (if not specifying specific VD IDs)"
    )
    parser.add_argument(
        "--vd_ids", 
        type=str, 
        default=None,
        help="Specific VD IDs to train, comma-separated (e.g., VD001,VD002,VD003)"
    )
    # NOTE: spatial_radius removed - not used in independent VD training
    
    # Override defaults for multi-VD mode (independent training)
    parser.set_defaults(
        hidden_size=256,  # Multi-VD suggests larger hidden size
        num_layers=3,     # Multi-VD suggests deeper layers
        dropout=0.3,      # Multi-VD suggests higher dropout
        batch_size=16,    # Multi-VD suggests smaller batch size
        learning_rate=0.0008,  # Multi-VD suggests smaller learning rate
        optimizer="adamw",     # Multi-VD suggests AdamW
        weight_decay=0.01,     # Multi-VD suggests higher weight decay
        early_stopping_patience=20,  # Multi-VD suggests higher patience
        scheduler_type="cosine",      # Multi-VD suggests cosine scheduler
        experiment_name="multi_vd_independent_without_social_pooling"
    )
    
    return parser.parse_args()


# Data module and model creation functions are now in common.py
# VD ID parsing is no longer needed - we use target_vd_index in IndependentMultiVDTrainer


def main():
    """Main function for independent multi-VD LSTM training without Social Pooling"""
    # Setup logging
    logger = setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Print start information
    print_training_start(logger, 'multi_vd')
    
    # Skip conda environment check for flexibility
    
    try:
        # 1. Create data module
        data_module = create_data_module(args, logger)
        
        # 2. Create model (use multi VD model for MultiVDTrainer)
        model = create_model_for_multi_vd(args, data_module, logger)
        
        # 3. Create training configuration
        logger.info("Creating multi-VD training configuration...")
        
        # Device selection
        device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
        
        # Get actual VD count from data
        sample_batch = next(iter(data_module.train_dataloader()))
        actual_num_vds = sample_batch['input_seq'].shape[-2]
        
        training_config = MultiVDTrainingConfig(
            # Basic parameters
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            
            # Optimizer (AdamW is better for multi-VD)
            optimizer_type=args.optimizer if args.optimizer != 'adam' else 'adamw',
            
            # Learning rate scheduling (cosine is better for multi-VD)
            use_scheduler=True,
            scheduler_type=args.scheduler_type if args.scheduler_type != 'reduce_on_plateau' else 'cosine',
            scheduler_patience=15,
            scheduler_factor=0.5,
            
            # Early stopping (more patience for multi-VD)
            early_stopping_patience=args.early_stopping_patience,
            
            # Device and performance
            device=device,
            mixed_precision=args.mixed_precision,
            gradient_clip_value=1.0,
            
            # Experiment management
            experiment_name=args.experiment_name,
            save_dir=args.save_dir,
            
            # Checkpoints (more frequent for multi-VD)
            save_checkpoints=True,
            checkpoint_interval=5,
            save_best_only=True,
            
            # Visualization (disabled for automatic runs)
            plot_training_curves=False,
            plot_predictions=False,
            plot_interval=999999,
            
            # Logging (more frequent for multi-VD)
            log_interval=3,
            
            # Multi VD specific
            num_vds=actual_num_vds,
            vd_aggregation="attention",  # Multi-VD training uses attention across VDs
            prediction_steps=1,
            spatial_features=True  # Enable spatial features for multi-VD training
        )
        
        # Print training configuration
        logger.info(f"Training on {'GPU' if training_config.device == 'cuda' else 'CPU'} for multi-VD")
        logger.info(f"Multi-VD training configuration:")
        logger.info(f"  Device: {training_config.device}")
        logger.info(f"  Mixed precision: {training_config.mixed_precision}")
        logger.info(f"  Optimizer: {training_config.optimizer_type}")
        logger.info(f"  Learning rate: {training_config.learning_rate}")
        logger.info(f"  Weight decay: {training_config.weight_decay}")
        logger.info(f"  Batch size: {training_config.batch_size}")
        logger.info(f"  Scheduler: {training_config.scheduler_type}")
        logger.info(f"  Experiment name: {training_config.experiment_name}")
        logger.info(f"  Number of VDs: {training_config.num_vds}")
        logger.info(f"  VD aggregation: {training_config.vd_aggregation}")
        
        # Device related warnings
        warnings = get_device_warnings(training_config.device, 'multi_vd')
        for warning in warnings:
            logger.info(warning)
        
        # 4. Create trainer (Multi-VD trainer with spatial relationships)
        logger.info("Creating Multi-VD trainer...")
        trainer = MultiVDTrainer(
            model=model,
            config=training_config,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            test_loader=data_module.test_dataloader()
        )
        
        # 5. Start training
        logger.info("Starting multi-VD training...")
        logger.info("NOTE: Multi-VD training typically takes longer")
        logger.info("NOTE: This is multi-VD training with spatial relationships")
        logger.info(f"Experiment directory: {trainer.experiment_dir}")
        
        history = trainer.train()
        
        # 6. Evaluate on test set if available
        if trainer.test_loader:
            try:
                test_metrics = trainer.evaluate_test_set()
                logger.info("Test evaluation completed successfully")
            except Exception as e:
                logger.warning(f"Test evaluation failed: {e}")
        
        # 7. Output results
        print_training_complete(logger, trainer, 'multi_vd')
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Multi-VD training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
