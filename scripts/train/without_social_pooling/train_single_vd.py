#!/usr/bin/env python3
"""
Single VD LSTM Model Training Script (Without Social Pooling)

Train a single VD traffic prediction model using unified training system.
This script trains LSTM/xLSTM models without any spatial relationships or Social Pooling.
Each VD is processed independently for establishing baseline performance.

Usage:
    conda activate social_xlstm
    python scripts/train/without_social_pooling/train_single_vd.py
    python scripts/train/without_social_pooling/train_single_vd.py --model_type lstm --epochs 100
    python scripts/train/without_social_pooling/train_single_vd.py --model_type xlstm --hidden_size 256

Author: Social-xLSTM Project Team
"""

import argparse
import sys
import torch

# Import common functions
from common import (
    setup_logging, add_common_arguments, print_training_start, print_training_complete,
    create_data_module, create_model_for_single_vd, get_device_warnings
)

from social_xlstm.training import SingleVDTrainer, SingleVDTrainingConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train single VD traffic prediction LSTM model (without Social Pooling)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add common arguments
    add_common_arguments(parser)
    
    # Single VD specific parameters (if any)
    # Currently no single VD specific parameters
    
    return parser.parse_args()


# Data module and model creation functions are now in common.py


def main():
    """Main function for single VD LSTM training without Social Pooling"""
    # Setup logging
    logger = setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Print start information
    print_training_start(logger, 'single_vd')
    
    # Skip conda environment check for flexibility
    
    try:
        # 1. Create data module
        data_module = create_data_module(args, logger)
        
        # 2. Create model
        model = create_model_for_single_vd(args, data_module, logger)
        
        # 3. Create training configuration
        logger.info("Creating training configuration...")
        
        # Device selection
        device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
        
        training_config = SingleVDTrainingConfig(
            # Basic parameters
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            
            # Optimizer
            optimizer_type=args.optimizer,
            
            # Learning rate scheduling
            use_scheduler=True,
            scheduler_type=args.scheduler_type,
            scheduler_patience=10,
            scheduler_factor=0.5,
            
            # Early stopping
            early_stopping_patience=args.early_stopping_patience,
            
            # Device and performance
            device=device,
            mixed_precision=args.mixed_precision,
            gradient_clip_value=1.0,
            
            # Experiment management
            experiment_name=args.experiment_name,
            save_dir=args.save_dir,
            
            # Checkpoints
            save_checkpoints=True,
            checkpoint_interval=10,
            save_best_only=True,
            
            # Visualization (disabled for automatic runs)
            plot_training_curves=False,
            plot_predictions=False,
            plot_interval=999999,
            
            # Logging
            log_interval=5,
            
            # Single VD specific
            prediction_steps=1,
            feature_indices=None
        )
        
        # Print training configuration
        logger.info(f"Training on {'GPU' if training_config.device == 'cuda' else 'CPU'}")
        logger.info(f"Training configuration:")
        logger.info(f"  Device: {training_config.device}")
        logger.info(f"  Mixed precision: {training_config.mixed_precision}")
        logger.info(f"  Optimizer: {training_config.optimizer_type}")
        logger.info(f"  Learning rate: {training_config.learning_rate}")
        logger.info(f"  Experiment name: {training_config.experiment_name}")
        logger.info(f"  Prediction steps: {training_config.prediction_steps}")
        
        # 4. Create trainer
        logger.info("Creating Single VD trainer...")
        trainer = SingleVDTrainer(
            model=model,
            config=training_config,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            test_loader=data_module.test_dataloader()
        )
        
        # 5. Print device warnings
        device_warnings = get_device_warnings(training_config.device, 'single_vd')
        for warning in device_warnings:
            logger.info(warning)
        
        # 6. Start training
        logger.info("Starting single VD training...")
        logger.info(f"Experiment directory: {trainer.experiment_dir}")
        
        history = trainer.train()
        
        # 7. Evaluate on test set if available
        if trainer.test_loader:
            try:
                test_metrics = trainer.evaluate_test_set()
                logger.info("Test evaluation completed successfully")
            except Exception as e:
                logger.warning(f"Test evaluation failed: {e}")
        
        # 8. Output results
        print_training_complete(logger, trainer, 'single_vd')
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()