#!/usr/bin/env python3
"""
Post-Fusion Social Pooling Single VD Training Script

⚠️  DEPRECATED: This script uses incorrect centralized architecture ⚠️

This script implements single VD training with Post-Fusion Social Pooling integration.
It extends the base single VD training to include spatial feature aggregation through
the Social Pooling mechanism using the SocialTrafficModel wrapper.

DEPRECATION NOTICE:
This training script is based on the deprecated centralized Social Pooling architecture
that fundamentally cannot scale to distributed social traffic scenarios. The Post-Fusion 
approach creates bottlenecks that prevent proper distributed xLSTM implementation.

MIGRATION PATH:
- Use scripts/train/distributed_social_xlstm/ instead
- See docs/legacy/explanation/social-pooling-implementation-guide.md
- Historical access: git checkout centralized-legacy-v0.2

Features (DEPRECATED):
- Support for both LSTM and xLSTM base models
- Post-Fusion strategy with Gated Fusion
- Coordinate-driven spatial pooling
- Scenario-based configuration (urban/highway/mixed)
- Performance monitoring and validation

Usage Examples (DEPRECATED):
    # Basic Social-LSTM training
    python train_single_vd.py --model_type lstm --select_vd_id VD-C1T0440-N --coordinate_data data/vd_coords.json --scenario urban --epochs 2

    # Social-xLSTM training
    python train_single_vd.py --model_type xlstm --select_vd_id VD-C1T0440-N --coordinate_data data/vd_coords.json --scenario highway --epochs 2

    # Custom Social Pooling parameters
    python train_single_vd.py --model_type lstm --select_vd_id VD-C1T0440-N --coordinate_data data/vd_coords.json --pooling_radius 1500 --max_neighbors 10 --epochs 2

Author: Social-xLSTM Project Team
Version: DEPRECATED
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import torch
import torch.nn as nn
from social_xlstm.training.without_social_pooling.single_vd_trainer import SingleVDTrainer, SingleVDTrainingConfig

# Import Post-Fusion utilities
from common import (
    setup_logging, check_conda_environment,
    add_post_fusion_arguments, validate_post_fusion_setup,
    create_social_pooling_config_from_args, load_coordinate_data,
    create_social_data_module, create_post_fusion_model,
    print_post_fusion_start, print_post_fusion_complete,
    get_device_warnings
)


class PostFusionSingleVDTrainer(SingleVDTrainer):
    """
    Specialized trainer for Post-Fusion Social Pooling single VD training.
    
    Extends the base SingleVDTrainer to support Social Traffic models with
    coordinate data handling and Social Pooling integration.
    """
    
    def __init__(self, model, config: SingleVDTrainingConfig, train_loader, val_loader=None, test_loader=None,
                 coordinates=None, social_config=None):
        """
        Initialize Post-Fusion trainer.
        
        Args:
            model: SocialTrafficModel instance
            config: Training configuration  
            train_loader: Training data loader
            val_loader: Optional validation data loader
            test_loader: Optional test data loader
            coordinates: VD coordinate data
            social_config: Social Pooling configuration
        """
        super().__init__(model, config, train_loader, val_loader, test_loader)
        self.coordinates = coordinates or {}
        self.social_config = social_config
    
    def _forward_pass_with_post_fusion_social(self, batch):
        """
        Enhanced forward pass with Post-Fusion Social Pooling.
        
        Args:
            batch: Training batch with input sequences
            
        Returns:
            Model output with spatial feature integration
        """
        # Extract input data
        input_seq = batch['input_seq']  # [batch_size, seq_len, features]
        
        # For single VD training, we need to provide coordinate context
        # The model will handle spatial aggregation internally
        vd_id = self.config.select_vd_id
        
        if vd_id and vd_id in self.coordinates:
            # Convert coordinates to tensor format expected by model
            coord_tensor = torch.tensor([self.coordinates[vd_id]], dtype=torch.float32, device=input_seq.device)
            vd_ids = [vd_id]
            
            # Forward pass with Social Pooling
            output = self.model(input_seq, coord_tensor, vd_ids)
        else:
            # Fallback to regular forward pass if coordinates unavailable
            self.logger.warning(f"Coordinates not available for VD {vd_id}, using base model only")
            output = self.model.base_model(input_seq)
        
        return output
    
    def training_step(self, batch, batch_idx):
        """Training step with Post-Fusion Social Pooling."""
        # Use enhanced forward pass
        output = self._forward_pass_with_post_fusion_social(batch)
        
        # Compute loss (same as base trainer)
        target = batch['target_seq']
        loss = self.criterion(output, target)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with Post-Fusion Social Pooling."""
        with torch.no_grad():
            # Use enhanced forward pass
            output = self._forward_pass_with_post_fusion_social(batch)
            
            # Compute loss
            target = batch['target_seq']
            loss = self.criterion(output, target)
            
            return loss
    
    def save_experiment_results(self):
        """Save experiment results including Social Pooling configuration."""
        # Call parent save method
        super().save_experiment_results()
        
        # Save Social Pooling specific configuration
        social_config_path = self.experiment_dir / "social_config.json"
        with open(social_config_path, 'w') as f:
            json.dump(self.social_config.to_dict(), f, indent=2)
        
        # Save coordinate data reference
        coord_info = {
            "coordinate_count": len(self.coordinates),
            "selected_vd": self.config.select_vd_id,
            "coordinate_system": self.social_config.coordinate_system if self.social_config else "unknown",
            "pooling_strategy": "post_fusion"
        }
        
        coord_info_path = self.experiment_dir / "coordinate_info.json"
        with open(coord_info_path, 'w') as f:
            json.dump(coord_info, f, indent=2)
        
        self.logger.info(f"Social Pooling configuration saved to: {social_config_path}")


def parse_arguments():
    """Parse command line arguments for Post-Fusion training."""
    parser = argparse.ArgumentParser(
        description="Post-Fusion Social Pooling Single VD Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add Post-Fusion specific arguments (includes base arguments)
    add_post_fusion_arguments(parser)
    
    # Single VD specific arguments
    parser.add_argument("--select_vd_id", type=str, required=True,
                        help="VD ID to select for single VD training")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Update experiment name to include model type and scenario
    if args.model_type == "xlstm":
        args.experiment_name = f"social_xlstm_post_fusion_{args.scenario}"
    else:
        args.experiment_name = f"social_lstm_post_fusion_{args.scenario}"
    
    return args


def main():
    """Main training function."""
    # Issue deprecation warning
    import warnings
    warnings.warn(
        "This training script uses deprecated centralized Social Pooling architecture. "
        "The Post-Fusion approach fundamentally cannot scale to distributed social traffic scenarios. "
        "Use scripts/train/distributed_social_xlstm/ for correct distributed Social-xLSTM implementation.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    
    # Print training start information
    print_post_fusion_start(logger, args)
    
    # Check conda environment
    check_conda_environment(logger)
    
    # Validate Post-Fusion setup
    try:
        validate_post_fusion_setup(args, logger)
    except ValueError as e:
        logger.error(f"Setup validation failed: {e}")
        sys.exit(1)
    
    # Load coordinate data
    try:
        coordinates = load_coordinate_data(args.coordinate_data, logger)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Coordinate data loading failed: {e}")
        sys.exit(1)
    
    # Create Social Pooling configuration
    try:
        social_config = create_social_pooling_config_from_args(args, logger)
    except Exception as e:
        logger.error(f"Social Pooling configuration failed: {e}")
        sys.exit(1)
    
    # Create data module with coordinate support
    try:
        data_module = create_social_data_module(args, coordinates, logger)
    except Exception as e:
        logger.error(f"Data module creation failed: {e}")
        sys.exit(1)
    
    # Create Post-Fusion Social Traffic model
    try:
        model = create_post_fusion_model(args, data_module, social_config, coordinates, logger)
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        sys.exit(1)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Device warnings
    warnings = get_device_warnings(device, mode='single_vd')
    for warning in warnings:
        logger.warning(warning)
    
    # Move model to device
    model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    # Create training configuration
    config = SingleVDTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler_type,
        early_stopping_patience=args.early_stopping_patience,
        device=device,
        mixed_precision=args.mixed_precision,
        experiment_name=args.experiment_name,
        save_dir=args.save_dir,
        select_vd_id=args.select_vd_id
    )
    
    # Create trainer
    trainer = PostFusionSingleVDTrainer(
        model=model,
        config=config,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        test_loader=data_module.test_dataloader(),
        coordinates=coordinates,
        social_config=social_config
    )
    
    # Start training
    try:
        logger.info("Starting Post-Fusion Social Pooling training...")
        trainer.train()
        
        # Print completion information
        print_post_fusion_complete(logger, trainer)
        
        logger.info("Post-Fusion training completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()