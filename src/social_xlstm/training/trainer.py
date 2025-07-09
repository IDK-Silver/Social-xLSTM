"""
Unified Training System for Traffic LSTM Models

This module provides a comprehensive training system that combines the best practices
from all existing training implementations according to ADR-0002.

Features:
- Professional training pipeline (from loader_lstm.py)
- Comprehensive evaluation (from pure_lstm.py)
- Multi-VD support (from traffic_lstm.py)
- Modern PyTorch practices (from various implementations)

Author: Social-xLSTM Project Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm

from ..models.lstm import TrafficLSTM, TrafficLSTMConfig
from ..evaluation.evaluator import ModelEvaluator
from ..visualization.model import TrafficResultsPlotter

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Comprehensive training configuration.
    
    This replaces scattered configuration across different implementations.
    """
    # Training Parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Optimizer Configuration
    optimizer_type: str = "adam"  # adam, sgd, adamw
    momentum: float = 0.9  # for SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # for Adam/AdamW
    
    # Learning Rate Scheduling
    use_scheduler: bool = True
    scheduler_type: str = "reduce_on_plateau"  # reduce_on_plateau, step, cosine
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_step_size: int = 30
    
    # Training Control
    early_stopping_patience: int = 20
    gradient_clip_value: Optional[float] = 1.0
    
    # Data Configuration
    validation_split: float = 0.2
    test_split: float = 0.2
    shuffle_train: bool = True
    num_workers: int = 4
    
    # Device and Performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False  # Enable for faster training on modern GPUs
    
    # Logging and Checkpoints
    save_checkpoints: bool = True
    checkpoint_interval: int = 10  # Save every N epochs
    save_best_only: bool = True
    log_interval: int = 10  # Log every N batches
    
    # Experiment Tracking
    experiment_name: str = "traffic_lstm_experiment"
    save_dir: str = "experiments"
    
    # Visualization
    plot_training_curves: bool = True
    plot_predictions: bool = True
    plot_interval: int = 20  # Plot every N epochs
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        if not 0 < self.test_split < 1:
            raise ValueError("test_split must be between 0 and 1")
        if self.validation_split + self.test_split >= 1:
            raise ValueError("validation_split + test_split must be < 1")


class Trainer:
    """
    Unified training system combining best practices from all implementations.
    
    Features:
    - Comprehensive training pipeline with validation
    - Advanced evaluation metrics and visualization
    - Flexible model and configuration support
    - Professional logging and checkpointing
    - Multi-VD and single-VD support
    """
    
    def __init__(self, 
                 model: TrafficLSTM,
                 training_config: TrainingConfig,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None):
        """
        Initialize unified trainer.
        
        Args:
            model: TrafficLSTM model to train
            training_config: Training configuration
            train_loader: Training data loader
            val_loader: Optional validation data loader
            test_loader: Optional test data loader
        """
        self.model = model
        self.config = training_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Move model to device
        self.model.to(self.config.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler() if self.config.use_scheduler else None
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Initialize evaluation tools (will be created after training)
        self.evaluator = None
        self.plotter = TrafficResultsPlotter()
        
        # Setup experiment directory
        self.experiment_dir = Path(self.config.save_dir) / self.config.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Mixed precision setup
        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Initialized Trainer with config: {self.config}")
        logger.info(f"Model info: {self.model.get_model_info()}")
        
        # Save configurations
        self._save_configs()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas
            )
        elif self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        if self.config.scheduler_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                verbose=True
            )
        elif self.config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_factor
            )
        elif self.config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
    
    def _save_configs(self):
        """Save model and training configurations."""
        config_data = {
            'model_config': self.model.config.__dict__,
            'training_config': self.config.__dict__,
            'model_info': self.model.get_model_info()
        }
        
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        logger.info(f"Saved configurations to {config_path}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.config.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract data from batch dictionary
            inputs = batch['input_seq']  # [batch_size, seq_len, num_vds, num_features]
            targets = batch['target_seq']  # [batch_size, pred_len, num_vds, num_features]
            
            # Handle single VD vs multi-VD mode
            if not getattr(self.model.config, 'multi_vd_mode', False):
                # Use first VD for single VD mode
                inputs = inputs[:, :, 0, :]  # [batch_size, seq_len, num_features]
                targets = targets[:, :, 0, :]  # [batch_size, pred_len, num_features]
                
                # For now, use only the first timestep of targets for single-step prediction
                # TODO: Implement multi-step prediction properly
                targets = targets[:, 0:1, :]  # [batch_size, 1, num_features]
            else:
                # Multi-VD mode: keep 4D input format, model will flatten internally
                # inputs stays as [batch_size, seq_len, num_vds, num_features]
                # targets should match model output format
                batch_size, seq_len, num_vds, num_features = inputs.shape
                targets = targets.view(batch_size, targets.shape[1], num_vds * num_features)  # [batch_size, pred_len, num_vds * num_features]
                
                # For now, use only the first timestep of targets for single-step prediction
                targets = targets[:, 0:1, :]  # [batch_size, 1, num_vds * num_features]
                
                # Debug logging (only in debug mode)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Multi-VD mode: inputs.shape = {inputs.shape}, targets.shape = {targets.shape}")
                    logger.debug(f"Multi-VD mode: batch_size={batch_size}, seq_len={seq_len}, num_vds={num_vds}, num_features={num_features}")
                    logger.debug(f"Model multi_vd_mode: {getattr(self.model, 'multi_vd_mode', 'NOT SET')}")
                    logger.debug(f"Model config.multi_vd_mode: {getattr(self.model.config, 'multi_vd_mode', 'NOT SET')}")
                    logger.debug(f"Using multi_vd_mode: {getattr(self.model.config, 'multi_vd_mode', False)}")
            
            # Move data to device
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_value:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"About to call model forward with inputs.shape = {inputs.shape}")
                outputs = self.model(inputs)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Model forward completed, outputs.shape = {outputs.shape}")
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_value:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            # Log batch metrics
            if batch_idx % self.config.log_interval == 0:
                logger.debug(f"Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Extract data from batch dictionary
                inputs = batch['input_seq']
                targets = batch['target_seq']
                
                # Handle single VD vs multi-VD mode
                if not getattr(self.model.config, 'multi_vd_mode', False):
                    inputs = inputs[:, :, 0, :]
                    targets = targets[:, :, 0, :]
                    targets = targets[:, 0:1, :]  # Single-step prediction
                else:
                    # Multi-VD mode: keep 4D input format, model will flatten internally
                    batch_size, seq_len, num_vds, num_features = inputs.shape
                    targets = targets.view(batch_size, targets.shape[1], num_vds * num_features)
                    targets = targets[:, 0:1, :]  # Single-step prediction
                
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        if num_batches == 0:
            return {}
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Training history dictionary
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            if 'val_loss' in val_metrics:
                self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Logging
            log_msg = f"Epoch {epoch+1}/{self.config.epochs} - Train Loss: {train_metrics['train_loss']:.6f}"
            if 'val_loss' in val_metrics:
                log_msg += f", Val Loss: {val_metrics['val_loss']:.6f}"
            logger.info(log_msg)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if 'val_loss' in val_metrics:
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step(train_metrics['train_loss'])
                else:
                    self.scheduler.step()
            
            # Early stopping and checkpointing
            if 'val_loss' in val_metrics:
                val_loss = val_metrics['val_loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    
                    if self.config.save_best_only:
                        self.save_checkpoint(is_best=True)
                else:
                    self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Regular checkpointing
            if self.config.save_checkpoints and (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
            
            # Plotting
            if self.config.plot_training_curves and (epoch + 1) % self.config.plot_interval == 0:
                self.plot_training_curves()
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Create evaluator after training
        self.evaluator = ModelEvaluator(
            model=self.model,
            train_losses=self.training_history['train_loss'],
            val_losses=self.training_history['val_loss'],
            config=self.config,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.config.device
        )
        
        # Final evaluation
        if self.test_loader:
            try:
                self.evaluate_test_set()
            except Exception as e:
                logger.warning(f"Test evaluation failed: {e}")
                logger.info("This is normal for very small datasets")
        
        # Final plots
        if self.config.plot_training_curves:
            self.plot_training_curves(save=True)
        
        return self.training_history
    
    def evaluate_test_set(self) -> Dict[str, float]:
        """Comprehensive evaluation on test set."""
        if self.test_loader is None:
            logger.warning("No test loader provided for evaluation")
            return {}
        
        logger.info("Evaluating on test set...")
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Extract data from batch dictionary
                inputs = batch['input_seq']
                targets = batch['target_seq']
                
                # Handle single VD vs multi-VD mode
                if not getattr(self.model.config, 'multi_vd_mode', False):
                    inputs = inputs[:, :, 0, :]
                    targets = targets[:, :, 0, :]
                    targets = targets[:, 0:1, :]  # Single-step prediction
                else:
                    # Multi-VD mode: keep 4D input format, model will flatten internally
                    batch_size, seq_len, num_vds, num_features = inputs.shape
                    targets = targets.view(batch_size, targets.shape[1], num_vds * num_features)
                    targets = targets[:, 0:1, :]  # Single-step prediction
                
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                
                outputs = self.model(inputs)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        if not all_predictions:
            logger.warning("No test data available for evaluation")
            return {}
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Compute comprehensive metrics
        metrics = self.evaluator.evaluate(targets, predictions)
        
        # Log results
        logger.info("Test Set Evaluation Results:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.6f}")
        
        # Save evaluation results
        eval_path = self.experiment_dir / "test_evaluation.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate prediction plots
        if self.config.plot_predictions:
            self.plot_predictions(targets, predictions)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_data = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'model_config': self.model.config.__dict__,
            'training_config': self.config.__dict__
        }
        
        if self.scheduler:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        if not self.config.save_best_only:
            checkpoint_path = self.experiment_dir / f"checkpoint_epoch_{self.epoch+1}.pt"
            torch.save(checkpoint_data, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.experiment_dir / "best_model.pt"
            torch.save(checkpoint_data, best_path)
            logger.info(f"Saved best model checkpoint to {best_path}")
    
    def plot_training_curves(self, save: bool = False):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        axes[0].plot(epochs, self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss']:
            axes[0].plot(epochs, self.training_history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Learning rate curve
        axes[1].plot(epochs, self.training_history['learning_rate'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        
        if save:
            plot_path = self.experiment_dir / "training_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training curves to {plot_path}")
        
        plt.show()
    
    def plot_predictions(self, targets: np.ndarray, predictions: np.ndarray):
        """Plot prediction results."""
        # Use the existing plotter from the evaluation module
        plot_path = self.experiment_dir / "predictions.png"
        self.plotter.plot_predictions(
            y_true=targets,
            y_pred=predictions,
            save_path=str(plot_path)
        )
        logger.info(f"Saved prediction plots to {plot_path}")
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, train_loader: DataLoader,
                       val_loader: Optional[DataLoader] = None,
                       test_loader: Optional[DataLoader] = None) -> 'Trainer':
        """
        Load trainer from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            train_loader: Training data loader
            val_loader: Optional validation data loader
            test_loader: Optional test data loader
        
        Returns:
            Loaded Trainer instance
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Recreate model and configs
        model_config = TrafficLSTMConfig(**checkpoint['model_config'])
        training_config = TrainingConfig(**checkpoint['training_config'])
        
        model = TrafficLSTM(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create trainer
        trainer = cls(model, training_config, train_loader, val_loader, test_loader)
        
        # Restore training state
        trainer.epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint['best_val_loss']
        trainer.training_history = checkpoint['training_history']
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if trainer.scheduler and 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded trainer from {checkpoint_path}")
        return trainer


# Utility functions for training setup

def create_trainer_from_config(model_config: Dict, training_config: Dict,
                              train_loader: DataLoader,
                              val_loader: Optional[DataLoader] = None,
                              test_loader: Optional[DataLoader] = None) -> Trainer:
    """
    Create trainer from configuration dictionaries.
    
    Args:
        model_config: Model configuration dictionary
        training_config: Training configuration dictionary
        train_loader: Training data loader
        val_loader: Optional validation data loader
        test_loader: Optional test data loader
    
    Returns:
        Configured Trainer instance
    """
    # Create model
    lstm_config = TrafficLSTMConfig(**model_config)
    model = TrafficLSTM(lstm_config)
    
    # Create training config
    train_config = TrainingConfig(**training_config)
    
    return Trainer(model, train_config, train_loader, val_loader, test_loader)


# Example usage
if __name__ == "__main__":
    # This would be used with actual data loaders
    print("Unified trainer module loaded successfully")
    print("Use with actual data loaders for training")