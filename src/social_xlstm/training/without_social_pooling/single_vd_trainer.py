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
    select_vd_id: Optional[str] = None  # VD ID to select for training (None = first VD)
    
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
        self.select_vd_id = config.select_vd_id
        
        logger.info(f"Single VD Trainer initialized:")
        logger.info(f"  Prediction steps: {self.prediction_steps}")
        logger.info(f"  Feature selection: {self.feature_indices}")
        logger.info(f"  Selected VD ID: {self.select_vd_id}")
    
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
        
        # Determine which VD to use
        vd_idx = 0  # Default to first VD
        
        # If specific VD ID is requested, find its index
        if self.select_vd_id and 'vdids' in batch:
            vdids = batch['vdids']
            
            if isinstance(vdids, list) and len(vdids) > 0:
                # Handle two possible formats:
                # Format 1: [['VD1', 'VD2'], ['VD1', 'VD2']] - correct format
                # Format 2: [['VD1', 'VD1'], ['VD2', 'VD2']] - pytorch collate format
                
                if isinstance(vdids[0], list):
                    first_sample_vdids = vdids[0]
                    
                    # Check if this looks like the correct format (all samples have same VDs)
                    if len(vdids) > 1 and len(vdids[0]) == len(vdids[1]):
                        # Check if it's pytorch collate format by seeing if VDs are repeated across samples
                        is_pytorch_format = all(
                            vdids[i][0] == vdids[i][1] if len(vdids[i]) >= 2 else True
                            for i in range(len(vdids))
                        )
                        
                        if is_pytorch_format and len(vdids) > 1:
                            # Reconstruct the correct VD list: take first element from each sublist
                            sample_vdids = [vdids[i][0] for i in range(len(vdids))]
                            logger.debug(f"Detected pytorch collate format, reconstructed VD list: {sample_vdids}")
                        else:
                            # Use the first sample's VD list
                            sample_vdids = first_sample_vdids
                    else:
                        sample_vdids = first_sample_vdids
                else:
                    sample_vdids = vdids
                
                if isinstance(sample_vdids, list) and self.select_vd_id in sample_vdids:
                    vd_idx = sample_vdids.index(self.select_vd_id)
                    logger.debug(f"Using VD {self.select_vd_id} at VD dimension index {vd_idx}")
                else:
                    logger.warning(f"VD ID {self.select_vd_id} not found in VD list {sample_vdids}, using first VD")
                    vd_idx = 0
            else:
                logger.warning(f"Invalid vdids format: {vdids}, using first VD")
                vd_idx = 0
        
        # Select the specific VD
        inputs = inputs[:, :, vd_idx, :]  # [batch_size, seq_len, num_features]
        targets = targets[:, :, vd_idx, :]  # [batch_size, pred_len, num_features]
        
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