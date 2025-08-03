"""
Multi VD Trainer Implementation

This module provides specialized training for multi VD (Vehicle Detector) models.
It handles spatial-temporal data from multiple VDs and supports different aggregation strategies.

Author: Social-xLSTM Project Team
License: MIT
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

from ..base import BaseTrainer, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class MultiVDTrainingConfig(TrainingConfig):
    """
    Configuration specific to multi VD training.
    
    Extends base configuration with multi VD specific parameters.
    """
    # Multi VD specific parameters
    num_vds: int = 5  # Number of VDs to process
    vd_aggregation: str = "flatten"  # "flatten", "attention", "pooling"
    prediction_steps: int = 1  # Number of time steps to predict
    spatial_features: bool = True  # Whether to use spatial features
    
    # Override defaults optimized for multi VD
    batch_size: int = 16  # Smaller batch size for multi VD
    learning_rate: float = 0.0008  # Lower learning rate
    hidden_size: int = 256  # Larger hidden size
    num_layers: int = 3  # More layers
    dropout: float = 0.3  # Higher dropout
    optimizer_type: str = "adamw"  # AdamW for better regularization
    scheduler_type: str = "cosine"  # Cosine scheduler
    early_stopping_patience: int = 20  # More patience
    
    def __post_init__(self):
        super().__post_init__()
        if self.num_vds < 1:
            raise ValueError("num_vds must be at least 1")
        if self.vd_aggregation not in ["flatten", "attention", "pooling"]:
            raise ValueError("vd_aggregation must be 'flatten', 'attention', or 'pooling'")
        if self.prediction_steps < 1:
            raise ValueError("prediction_steps must be at least 1")


class MultiVDTrainer(BaseTrainer):
    """
    Specialized trainer for multi VD models.
    
    This trainer handles:
    - Multi VD data format (4D tensors)
    - Spatial-temporal relationships
    - Different VD aggregation strategies
    - Multi-step prediction
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: MultiVDTrainingConfig,
                 train_loader,
                 val_loader=None,
                 test_loader=None):
        """
        Initialize Multi VD trainer.
        
        Args:
            model: Model that can handle multi VD input
            config: Multi VD training configuration
            train_loader: Training data loader
            val_loader: Optional validation data loader
            test_loader: Optional test data loader
        """
        super().__init__(model, config, train_loader, val_loader, test_loader)
        
        # Multi VD specific setup
        self.num_vds = config.num_vds
        self.vd_aggregation = config.vd_aggregation
        self.prediction_steps = config.prediction_steps
        self.spatial_features = config.spatial_features
        
        logger.info(f"Multi VD Trainer initialized:")
        logger.info(f"  Number of VDs: {self.num_vds}")
        logger.info(f"  VD aggregation: {self.vd_aggregation}")
        logger.info(f"  Prediction steps: {self.prediction_steps}")
        logger.info(f"  Spatial features: {self.spatial_features}")
    
    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch for multi VD training.
        
        Args:
            batch: Raw batch from DataLoader with format:
                   - 'input_seq': [B, T, N, F] - B=批次, T=時間步, N=VD數量, F=特徵
                   - 'target_seq': [batch_size, pred_len, num_vds, num_features]
        
        Returns:
            Tuple of (inputs, targets) prepared for the model
        """
        # Extract input and target sequences
        inputs = batch['input_seq']  # [B, T, N, F] - B=批次, T=時間步, N=VD數量, F=特徵
        targets = batch['target_seq']  # [B, T, N, F] - B=批次, T=預測長度, N=VD數量, F=特徵
        
        # Select only the required prediction steps first
        targets = targets[:, :self.prediction_steps, :, :]
        
        # Prepare inputs and targets based on aggregation strategy
        if self.vd_aggregation == "flatten":
            # Flatten VD and feature dimensions for both inputs and targets
            batch_size, seq_len, num_vds, num_features = inputs.shape
            inputs = inputs.view(batch_size, seq_len, num_vds * num_features)
            
            # Apply same transformation to targets - key fix!
            target_batch_size, target_seq_len, target_num_vds, target_num_features = targets.shape
            targets = targets.view(target_batch_size, target_seq_len, target_num_vds * target_num_features)
            
        elif self.vd_aggregation == "attention":
            # For attention mechanism, flatten inputs but keep targets 4D initially
            # Then reshape targets to match model output format
            batch_size, seq_len, num_vds, num_features = inputs.shape
            inputs = inputs.view(batch_size, seq_len, num_vds * num_features)
            
            # Reshape targets to match model output (flattened format)
            target_batch_size, target_seq_len, target_num_vds, target_num_features = targets.shape
            targets = targets.view(target_batch_size, target_seq_len, target_num_vds * target_num_features)
            
        elif self.vd_aggregation == "pooling":
            # Average pooling across VDs
            inputs = inputs.mean(dim=2)  # [batch_size, seq_len, num_features]
            targets = targets.mean(dim=2)  # [batch_size, pred_len, num_features]
        
        return inputs, targets
    
    def forward_model(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            inputs: Input tensor (format depends on aggregation strategy)
            
        Returns:
            Model outputs matching target format
        """
        # Direct forward pass - the model should handle the format
        outputs = self.model(inputs)
        
        # Ensure output shape matches expected target shape
        if outputs.dim() == 2:
            # If model outputs [batch_size, features], add time dimension
            outputs = outputs.unsqueeze(1)  # [batch_size, 1, features]
        
        # If model outputs more steps than needed, truncate
        if outputs.shape[1] > self.prediction_steps:
            outputs = outputs[:, :self.prediction_steps, :]
        
        return outputs
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function optimized for multi VD training."""
        # MSE with potential weighting for different VDs
        return nn.MSELoss()
    
    def get_training_info(self) -> Dict:
        """Get training-specific information."""
        return {
            'trainer_type': 'MultiVD',
            'num_vds': self.num_vds,
            'vd_aggregation': self.vd_aggregation,
            'prediction_steps': self.prediction_steps,
            'spatial_features': self.spatial_features,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


class IndependentMultiVDTrainer(BaseTrainer):
    """
    Trainer for independent multi VD training.
    
    This trainer trains separate models for each VD independently,
    useful for baseline comparisons and when spatial interactions are not needed.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: MultiVDTrainingConfig,
                 train_loader,
                 val_loader=None,
                 test_loader=None,
                 target_vd_index: int = 0):
        """
        Initialize Independent Multi VD trainer.
        
        Args:
            model: Model for single VD (will be applied to one VD)
            config: Multi VD training configuration
            train_loader: Training data loader
            val_loader: Optional validation data loader
            test_loader: Optional test data loader
            target_vd_index: Index of the VD to train on
        """
        super().__init__(model, config, train_loader, val_loader, test_loader)
        
        self.target_vd_index = target_vd_index
        self.prediction_steps = config.prediction_steps
        
        logger.info(f"Independent Multi VD Trainer initialized:")
        logger.info(f"  Target VD index: {self.target_vd_index}")
        logger.info(f"  Prediction steps: {self.prediction_steps}")
    
    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch for independent VD training.
        
        Args:
            batch: Raw batch from DataLoader
        
        Returns:
            Tuple of (inputs, targets) for the target VD only
        """
        # Extract input and target sequences
        inputs = batch['input_seq']  # [B, T, N, F] - B=批次, T=時間步, N=VD數量, F=特徵
        targets = batch['target_seq']  # [B, T, N, F] - B=批次, T=預測長度, N=VD數量, F=特徵
        
        # Select only the target VD
        inputs = inputs[:, :, self.target_vd_index, :]  # [batch_size, seq_len, num_features]
        targets = targets[:, :, self.target_vd_index, :]  # [batch_size, pred_len, num_features]
        
        # Select only the required prediction steps
        targets = targets[:, :self.prediction_steps, :]
        
        return inputs, targets
    
    def forward_model(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        outputs = self.model(inputs)
        
        # Ensure output shape matches expected target shape
        if outputs.dim() == 2:
            outputs = outputs.unsqueeze(1)
        
        if outputs.shape[1] > self.prediction_steps:
            outputs = outputs[:, :self.prediction_steps, :]
        
        return outputs
    
    def get_training_info(self) -> Dict:
        """Get training-specific information."""
        return {
            'trainer_type': 'IndependentMultiVD',
            'target_vd_index': self.target_vd_index,
            'prediction_steps': self.prediction_steps,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


# Convenience functions
def create_multi_vd_trainer(model, train_loader, val_loader=None, test_loader=None, **config_kwargs):
    """
    Convenience function to create a multi VD trainer.
    
    Args:
        model: Model for multi VD training
        train_loader: Training data loader
        val_loader: Optional validation data loader
        test_loader: Optional test data loader
        **config_kwargs: Additional configuration parameters
    
    Returns:
        MultiVDTrainer instance
    """
    config = MultiVDTrainingConfig(**config_kwargs)
    return MultiVDTrainer(model, config, train_loader, val_loader, test_loader)


def create_independent_multi_vd_trainer(model, train_loader, target_vd_index=0, 
                                       val_loader=None, test_loader=None, **config_kwargs):
    """
    Convenience function to create an independent multi VD trainer.
    
    Args:
        model: Model for single VD training
        train_loader: Training data loader
        target_vd_index: Index of the VD to train on
        val_loader: Optional validation data loader
        test_loader: Optional test data loader
        **config_kwargs: Additional configuration parameters
    
    Returns:
        IndependentMultiVDTrainer instance
    """
    config = MultiVDTrainingConfig(**config_kwargs)
    return IndependentMultiVDTrainer(model, config, train_loader, val_loader, test_loader, target_vd_index)


# Example usage demonstration
if __name__ == "__main__":
    print("MultiVDTrainer module loaded successfully")
    print("Usage:")
    print("1. MultiVDTrainer: For models that handle spatial relationships")
    print("2. IndependentMultiVDTrainer: For training individual VDs independently")
    print("3. Use create_multi_vd_trainer() or create_independent_multi_vd_trainer()")
    print("4. Call trainer.train() to start training")