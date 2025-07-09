"""
Single VD Trainer Implementation

This module provides specialized training for single VD (Vehicle Detector) models.
It handles the specific data format and training requirements for single VD scenarios.

Author: Social-xLSTM Project Team
License: MIT
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from ..base import BaseTrainer, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class SingleVDTrainingConfig(TrainingConfig):
    """
    Configuration specific to single VD training.
    
    Extends base configuration with single VD specific parameters.
    """
    # Single VD specific parameters
    prediction_steps: int = 1  # Number of time steps to predict
    feature_indices: Optional[list] = None  # Specific features to use (None = all)
    
    # Override defaults optimized for single VD
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    
    def __post_init__(self):
        super().__post_init__()
        if self.prediction_steps < 1:
            raise ValueError("prediction_steps must be at least 1")


class SingleVDTrainer(BaseTrainer):
    """
    Specialized trainer for single VD models.
    
    This trainer handles:
    - Single VD data format (3D tensors)
    - Simple time series prediction
    - Single-step or multi-step prediction
    - Feature selection if needed
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: SingleVDTrainingConfig,
                 train_loader,
                 val_loader=None,
                 test_loader=None):
        """
        Initialize Single VD trainer.
        
        Args:
            model: Model that expects 3D input [batch, seq_len, features]
            config: Single VD training configuration
            train_loader: Training data loader
            val_loader: Optional validation data loader
            test_loader: Optional test data loader
        """
        super().__init__(model, config, train_loader, val_loader, test_loader)
        
        # Single VD specific setup
        self.prediction_steps = config.prediction_steps
        self.feature_indices = config.feature_indices
        
        logger.info(f"Single VD Trainer initialized:")
        logger.info(f"  Prediction steps: {self.prediction_steps}")
        logger.info(f"  Feature selection: {self.feature_indices}")
    
    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch for single VD training.
        
        Args:
            batch: Raw batch from DataLoader with format:
                   - 'input_seq': [batch_size, seq_len, num_vds, num_features]
                   - 'target_seq': [batch_size, pred_len, num_vds, num_features]
        
        Returns:
            Tuple of (inputs, targets) with format:
            - inputs: [batch_size, seq_len, num_features]
            - targets: [batch_size, prediction_steps, num_features]
        """
        # Extract input and target sequences
        inputs = batch['input_seq']  # [batch_size, seq_len, num_vds, num_features]
        targets = batch['target_seq']  # [batch_size, pred_len, num_vds, num_features]
        
        # For single VD, we typically use the first VD or a specific VD
        # Here we use the first VD (index 0)
        inputs = inputs[:, :, 0, :]  # [batch_size, seq_len, num_features]
        targets = targets[:, :, 0, :]  # [batch_size, pred_len, num_features]
        
        # Select only the required prediction steps
        targets = targets[:, :self.prediction_steps, :]  # [batch_size, prediction_steps, num_features]
        
        # Apply feature selection if specified
        if self.feature_indices is not None:
            inputs = inputs[:, :, self.feature_indices]
            targets = targets[:, :, self.feature_indices]
        
        return inputs, targets
    
    def forward_model(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            inputs: Input tensor [batch_size, seq_len, num_features]
            
        Returns:
            Model outputs [batch_size, prediction_steps, num_features]
        """
        # Direct forward pass - the model should handle the format
        outputs = self.model(inputs)
        
        # Ensure output shape matches expected target shape
        if outputs.dim() == 2:
            # If model outputs [batch_size, num_features], add time dimension
            outputs = outputs.unsqueeze(1)  # [batch_size, 1, num_features]
        
        # If model outputs more steps than needed, truncate
        if outputs.shape[1] > self.prediction_steps:
            outputs = outputs[:, :self.prediction_steps, :]
        
        return outputs
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function optimized for single VD training."""
        # MSE is typically good for regression tasks like traffic prediction
        return nn.MSELoss()
    
    def get_training_info(self) -> Dict:
        """Get training-specific information."""
        return {
            'trainer_type': 'SingleVD',
            'prediction_steps': self.prediction_steps,
            'feature_indices': self.feature_indices,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


# Convenience function for creating single VD trainer
def create_single_vd_trainer(model, train_loader, val_loader=None, test_loader=None, **config_kwargs):
    """
    Convenience function to create a single VD trainer.
    
    Args:
        model: Model for single VD training
        train_loader: Training data loader
        val_loader: Optional validation data loader
        test_loader: Optional test data loader
        **config_kwargs: Additional configuration parameters
    
    Returns:
        SingleVDTrainer instance
    """
    config = SingleVDTrainingConfig(**config_kwargs)
    return SingleVDTrainer(model, config, train_loader, val_loader, test_loader)


# Example usage demonstration
if __name__ == "__main__":
    print("SingleVDTrainer module loaded successfully")
    print("Usage:")
    print("1. Create a model that expects 3D input [batch, seq_len, features]")
    print("2. Create data loaders with the standard format")
    print("3. Use create_single_vd_trainer() or SingleVDTrainer() directly")
    print("4. Call trainer.train() to start training")