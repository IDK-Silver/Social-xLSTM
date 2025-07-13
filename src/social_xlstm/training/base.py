"""
Base Training Interface and Common Components

This module provides the foundation for specialized training implementations,
replacing the overly complex unified Trainer approach.

Author: Social-xLSTM Project Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
from abc import ABC, abstractmethod

from ..evaluation.evaluator import ModelEvaluator
from ..visualization.training_visualizer import TrainingVisualizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Common training configuration for all trainer types.
    
    Each trainer can extend this with specific parameters.
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
    
    # Device and Performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False
    
    # Logging and Checkpoints
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    save_best_only: bool = True
    log_interval: int = 10
    
    # Experiment Tracking
    experiment_name: str = "traffic_lstm_experiment"
    save_dir: str = "blob/experiments"
    
    # Visualization
    plot_training_curves: bool = True
    plot_predictions: bool = True
    plot_interval: int = 20

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.gradient_clip_value is not None and self.gradient_clip_value <= 0:
            raise ValueError("gradient_clip_value must be positive or None")


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    
    This provides common functionality while allowing specialized implementations
    for different training scenarios.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: TrainingConfig,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None):
        """
        Initialize base trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Optional validation data loader
            test_loader: Optional test data loader
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Move model to device
        self.model.to(self.config.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler() if self.config.use_scheduler else None
        
        # Setup loss function
        self.criterion = self._create_loss_function()
        
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
        
        # Evaluation tools
        self.evaluator = None
        self.visualizer = TrainingVisualizer()
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
        
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
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function. Can be overridden by subclasses."""
        return nn.MSELoss()
    
    def _save_configs(self):
        """Save model and training configurations."""
        config_data = {
            'model_config': getattr(self.model, 'config', {}).__dict__ if hasattr(getattr(self.model, 'config', {}), '__dict__') else {},
            'training_config': self.config.__dict__,
            'trainer_type': self.__class__.__name__
        }
        
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        logger.info(f"Saved configurations to {config_path}")
    
    @abstractmethod
    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for training.
        
        This method must be implemented by subclasses to handle their specific
        data format requirements.
        
        Args:
            batch: Raw batch from DataLoader
            
        Returns:
            Tuple of (inputs, targets) ready for model forward pass
        """
        pass
    
    @abstractmethod
    def forward_model(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            inputs: Prepared input tensor
            
        Returns:
            Model outputs
        """
        pass
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.config.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch using subclass implementation
            inputs, targets = self.prepare_batch(batch)
            
            # Move to device
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.forward_model(inputs)
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
                outputs = self.forward_model(inputs)
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
                # Prepare batch using subclass implementation
                inputs, targets = self.prepare_batch(batch)
                
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                
                outputs = self.forward_model(inputs)
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
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save training history at the end of training
        self._save_training_history()
        
        # Force save final model if no best model was saved
        if not (self.experiment_dir / "best_model.pt").exists():
            logger.info("No best model saved, saving final model...")
            self.save_checkpoint(is_best=True)
        
        return self.training_history
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_data = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'trainer_type': self.__class__.__name__
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
    
    def _save_training_history(self):
        """Save training history to JSON file."""
        history_path = self.experiment_dir / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
    
    def evaluate_test_set(self) -> Dict[str, float]:
        """Evaluate on test set if available."""
        if self.test_loader is None:
            logger.warning("No test loader provided for evaluation")
            return {}
        
        logger.info("Evaluating on test set...")
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                inputs, targets = self.prepare_batch(batch)
                
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                
                outputs = self.forward_model(inputs)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        if not all_predictions:
            logger.warning("No test data available for evaluation")
            return {}
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Create evaluator if not exists
        if self.evaluator is None:
            self.evaluator = ModelEvaluator(
                model=self.model,
                train_losses=self.training_history['train_loss'],
                val_losses=self.training_history['val_loss'],
                config=self.config,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.config.device
            )
        
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
        
        return metrics