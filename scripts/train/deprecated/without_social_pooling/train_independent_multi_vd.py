#!/usr/bin/env python3
"""
Independent Multi-VD LSTM Model Training Script (Without Social Pooling)

Train independent multi-VD traffic prediction models using unified training system.
This script trains LSTM/xLSTM models for each VD independently without any spatial 
relationships or Social Pooling. Each VD is processed as a separate baseline.

IMPORTANT: This is NOT Social Pooling - each VD is trained completely independently.
No spatial relationships or interactions between VDs are considered.
This is purely for establishing baseline performance for comparison with Social-xLSTM.

Usage:
    conda activate social_xlstm
    python scripts/train/without_social_pooling/train_independent_multi_vd.py
    python scripts/train/without_social_pooling/train_independent_multi_vd.py --model_type lstm --num_vds 5 --epochs 100
    python scripts/train/without_social_pooling/train_independent_multi_vd.py --model_type xlstm --vd_ids VD001,VD002,VD003

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

from social_xlstm.training import IndependentMultiVDTrainer, MultiVDTrainingConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train independent multi-VD traffic prediction LSTM models (without Social Pooling)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add common arguments
    add_common_arguments(parser)
    
    # Independent Multi-VD specific parameters
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
    parser.add_argument(
        "--target_vd_index", 
        type=int, 
        default=0,
        help="Index of the target VD to train (0-based)"
    )
    
    # Override defaults for independent multi-VD mode
    parser.set_defaults(
        hidden_size=256,  # Independent multi-VD suggests larger hidden size
        num_layers=3,     # Independent multi-VD suggests deeper layers
        dropout=0.3,      # Independent multi-VD suggests higher dropout
        batch_size=16,    # Independent multi-VD suggests smaller batch size
        learning_rate=0.0008,  # Independent multi-VD suggests smaller learning rate
        optimizer="adamw",     # Independent multi-VD suggests AdamW
        weight_decay=0.01,     # Independent multi-VD suggests higher weight decay
        early_stopping_patience=20,  # Independent multi-VD suggests higher patience
        scheduler_type="cosine",      # Independent multi-VD suggests cosine scheduler
        experiment_name="independent_multi_vd_without_social_pooling"
    )
    
    return parser.parse_args()


def main():
    """Main function for independent multi-VD LSTM training without Social Pooling"""
    # Setup logging
    logger = setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Print start information
    print_training_start(logger, 'independent_multi_vd')
    
    # Skip conda environment check for flexibility
    
    try:
        # 1. Create data module
        data_module = create_data_module(args, logger)
        
        # 2. Create model (use single VD model for IndependentMultiVDTrainer)
        model = create_model_for_single_vd(args, data_module, logger)
        
        # 3. Create training configuration
        logger.info("Creating independent multi-VD training configuration...")
        
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
            
            # Optimizer (AdamW is better for independent multi-VD)
            optimizer_type=args.optimizer if args.optimizer != 'adam' else 'adamw',
            
            # Learning rate scheduling (cosine is better for independent multi-VD)
            use_scheduler=True,
            scheduler_type=args.scheduler_type if args.scheduler_type != 'reduce_on_plateau' else 'cosine',
            scheduler_patience=15,
            scheduler_factor=0.5,
            
            # Early stopping (more patience for independent multi-VD)
            early_stopping_patience=args.early_stopping_patience,
            
            # Device and performance
            device=device,
            mixed_precision=args.mixed_precision,
            gradient_clip_value=1.0,
            
            # Experiment management
            experiment_name=args.experiment_name,
            save_dir=args.save_dir,
            
            # Checkpoints (more frequent for independent multi-VD)
            save_checkpoints=True,
            checkpoint_interval=5,
            save_best_only=True,
            
            # Visualization (disabled for automatic runs)
            plot_training_curves=False,
            plot_predictions=False,
            plot_interval=999999,
            
            # Logging (more frequent for independent multi-VD)
            log_interval=3,
            
            # Independent Multi VD specific
            num_vds=actual_num_vds,
            vd_aggregation="flatten",  # Independent training uses flatten
            prediction_steps=1,
            spatial_features=False  # No spatial features in independent training
        )
        
        # Print training configuration
        logger.info(f"Training on {'GPU' if training_config.device == 'cuda' else 'CPU'} for independent multi-VD")
        logger.info(f"Independent multi-VD training configuration:")
        logger.info(f"  Device: {training_config.device}")
        logger.info(f"  Mixed precision: {training_config.mixed_precision}")
        logger.info(f"  Optimizer: {training_config.optimizer_type}")
        logger.info(f"  Learning rate: {training_config.learning_rate}")
        logger.info(f"  Weight decay: {training_config.weight_decay}")
        logger.info(f"  Batch size: {training_config.batch_size}")
        logger.info(f"  Scheduler: {training_config.scheduler_type}")
        logger.info(f"  Experiment name: {training_config.experiment_name}")
        logger.info(f"  Number of VDs: {training_config.num_vds}")
        logger.info(f"  Target VD index: {args.target_vd_index}")
        logger.info(f"  VD aggregation: {training_config.vd_aggregation}")
        
        # Device related warnings
        warnings = get_device_warnings(training_config.device, 'independent_multi_vd')
        for warning in warnings:
            logger.info(warning)
        
        # 4. Create trainer (Independent Multi-VD trainer for specified VD)
        logger.info("Creating Independent Multi-VD trainer...")
        trainer = IndependentMultiVDTrainer(
            model=model,
            config=training_config,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            test_loader=data_module.test_dataloader(),
            target_vd_index=args.target_vd_index
        )
        
        # 5. Start training
        logger.info("Starting independent multi-VD training...")
        logger.info("NOTE: Independent multi-VD training typically takes longer")
        logger.info("NOTE: This is independent VD training - no spatial interaction")
        logger.info(f"Training VD at index {args.target_vd_index}")
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
        print_training_complete(logger, trainer, 'independent_multi_vd')
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Independent multi-VD training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()