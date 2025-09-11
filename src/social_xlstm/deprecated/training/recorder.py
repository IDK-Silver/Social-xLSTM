"""
Training Recorder for Social-xLSTM Project

This module provides comprehensive training recording capabilities for machine learning experiments.
The TrainingRecorder class captures detailed training metrics, system information, and analysis tools
to support reproducible research and thorough experiment tracking.

Author: Social-xLSTM Project Team
License: MIT
"""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import subprocess

import numpy as np
import torch
import psutil


@dataclass
class EpochRecord:
    """
    Complete record of a single training epoch.
    
    This dataclass captures all relevant information about a training epoch,
    including losses, metrics, system state, and timing information.
    """
    # Basic epoch information
    epoch: int
    timestamp: datetime
    
    # Loss metrics
    train_loss: float
    val_loss: Optional[float] = None
    
    # Evaluation metrics (MAE, MSE, RMSE, etc.)
    train_metrics: Dict[str, float] = None
    val_metrics: Optional[Dict[str, float]] = None
    
    # Training state
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    memory_usage: Optional[float] = None
    
    # Model state
    gradient_norm: Optional[float] = None
    is_best: bool = False
    
    # Optional detailed data
    sample_predictions: Optional[Dict] = None
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.train_metrics is None:
            self.train_metrics = {}
        if self.val_metrics is None:
            self.val_metrics = {}
        if self.sample_predictions is None:
            self.sample_predictions = {}


class TrainingRecorder:
    """
    Comprehensive training recording system for ML experiments.
    
    This class provides a structured way to record, analyze, and visualize
    training progress. It captures detailed metrics, system information,
    and provides analysis tools for understanding training behavior.
    
    Features:
    - Structured epoch-by-epoch recording
    - Automatic best epoch tracking
    - Training curve visualization
    - Stability analysis
    - JSON serialization for persistence
    - System metadata collection
    
    Example:
        recorder = TrainingRecorder("my_experiment", model_config, training_config)
        
        for epoch in range(epochs):
            # ... training logic ...
            recorder.log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics)
        
        recorder.save("experiment_record.json")
        recorder.plot_training_curves()
    """
    
    def __init__(self, experiment_name: str, model_config: dict, training_config: dict):
        """
        Initialize the training recorder.
        
        Args:
            experiment_name: Name of the experiment
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary
        """
        self.experiment_name = experiment_name
        self.model_config = model_config
        self.training_config = training_config
        self.start_time = datetime.now()
        
        # Core data storage
        self.epochs: List[EpochRecord] = []
        self.best_epoch: Optional[int] = None
        self.best_val_loss: float = float('inf')
        self.early_stopping_info: Dict = {}
        
        # Experiment metadata
        self.experiment_metadata = {
            'git_commit': self._get_git_commit(),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_info': self._get_device_info(),
            'system_info': self._get_system_info()
        }
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                  train_metrics: Optional[Dict] = None, val_metrics: Optional[Dict] = None,
                  learning_rate: float = 0.0, epoch_time: float = 0.0,
                  gradient_norm: Optional[float] = None, **kwargs):
        """
        Record a complete epoch of training.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss value
            val_loss: Validation loss value (optional)
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            learning_rate: Current learning rate
            epoch_time: Time taken for this epoch
            gradient_norm: Gradient norm value
            **kwargs: Additional data to record
        """
        # Check if this is the best epoch
        current_loss = val_loss if val_loss is not None else train_loss
        is_best = current_loss < self.best_val_loss
        
        if is_best:
            self.best_val_loss = current_loss
            self.best_epoch = epoch
        
        # Create epoch record
        record = EpochRecord(
            epoch=epoch,
            timestamp=datetime.now(),
            train_loss=train_loss,
            val_loss=val_loss,
            train_metrics=train_metrics or {},
            val_metrics=val_metrics or {},
            learning_rate=learning_rate,
            epoch_time=epoch_time,
            memory_usage=self._get_memory_usage(),
            gradient_norm=gradient_norm,
            is_best=is_best,
            sample_predictions=kwargs.get('sample_predictions', {})
        )
        
        self.epochs.append(record)
    
    def get_metric_history(self, metric_name: str, split: str = 'train') -> List[float]:
        """
        Get historical values for a specific metric.
        
        Args:
            metric_name: Name of the metric (e.g., 'mae', 'mse')
            split: Data split ('train' or 'val')
            
        Returns:
            List of metric values across epochs
        """
        if split == 'train':
            return [epoch.train_metrics.get(metric_name, 0) for epoch in self.epochs]
        elif split == 'val':
            return [epoch.val_metrics.get(metric_name, 0) for epoch in self.epochs 
                    if epoch.val_metrics]
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")
    
    def get_loss_history(self) -> Tuple[List[float], List[float]]:
        """
        Get training and validation loss history.
        
        Returns:
            Tuple of (train_losses, val_losses)
        """
        train_losses = [epoch.train_loss for epoch in self.epochs]
        val_losses = [epoch.val_loss for epoch in self.epochs if epoch.val_loss is not None]
        return train_losses, val_losses
    
    def get_best_epoch(self) -> Optional[EpochRecord]:
        """
        Get the best epoch record.
        
        Returns:
            EpochRecord of the best epoch or None if no epochs recorded
        """
        return self.epochs[self.best_epoch] if self.best_epoch is not None else None
    
    def get_training_summary(self) -> Dict:
        """
        Get a comprehensive training summary.
        
        Returns:
            Dictionary containing training statistics and analysis
        """
        if not self.epochs:
            return {}
        
        best_epoch = self.get_best_epoch()
        total_time = sum(epoch.epoch_time for epoch in self.epochs)
        
        return {
            'total_epochs': len(self.epochs),
            'total_time': total_time,
            'avg_epoch_time': total_time / len(self.epochs),
            'best_epoch': best_epoch.epoch if best_epoch else None,
            'best_train_loss': best_epoch.train_loss if best_epoch else None,
            'best_val_loss': best_epoch.val_loss if best_epoch else None,
            'final_learning_rate': self.epochs[-1].learning_rate,
            'convergence_info': self._analyze_convergence(),
            'stability_analysis': self.analyze_training_stability()
        }
    
    def export_to_csv(self, path: Union[str, Path]):
        """
        Export training history to CSV format for external analysis.
        
        Args:
            path: Path to save the CSV file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for CSV
        rows = []
        for epoch in self.epochs:
            row = {
                'epoch': epoch.epoch,
                'timestamp': epoch.timestamp.isoformat(),
                'train_loss': epoch.train_loss,
                'val_loss': epoch.val_loss or '',
                'learning_rate': epoch.learning_rate,
                'epoch_time': epoch.epoch_time,
                'memory_usage': epoch.memory_usage or '',
                'gradient_norm': epoch.gradient_norm or '',
                'is_best': epoch.is_best
            }
            
            # Add metrics
            for metric_name, value in epoch.train_metrics.items():
                row[f'train_{metric_name}'] = value
            
            if epoch.val_metrics:
                for metric_name, value in epoch.val_metrics.items():
                    row[f'val_{metric_name}'] = value
            
            rows.append(row)
        
        # Write to CSV
        if rows:
            import csv
            fieldnames = list(rows[0].keys())
            
            with open(path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"Training history exported to CSV: {path}")
    
    def export_to_tensorboard(self, log_dir: Union[str, Path]):
        """
        Export training data for TensorBoard visualization.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            writer = SummaryWriter(log_dir=str(log_dir))
            
            for epoch in self.epochs:
                # Log losses
                writer.add_scalar('Loss/train', epoch.train_loss, epoch.epoch)
                if epoch.val_loss is not None:
                    writer.add_scalar('Loss/val', epoch.val_loss, epoch.epoch)
                
                # Log metrics
                for metric_name, value in epoch.train_metrics.items():
                    writer.add_scalar(f'Metrics/train_{metric_name}', value, epoch.epoch)
                
                if epoch.val_metrics:
                    for metric_name, value in epoch.val_metrics.items():
                        writer.add_scalar(f'Metrics/val_{metric_name}', value, epoch.epoch)
                
                # Log other values
                writer.add_scalar('Learning_Rate', epoch.learning_rate, epoch.epoch)
                if epoch.gradient_norm is not None:
                    writer.add_scalar('Gradient_Norm', epoch.gradient_norm, epoch.epoch)
                if epoch.memory_usage is not None:
                    writer.add_scalar('Memory_Usage_MB', epoch.memory_usage, epoch.epoch)
            
            writer.close()
            print(f"TensorBoard logs exported to: {log_dir}")
            
        except ImportError:
            # Still create directory for testing purposes
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            print("TensorBoard not available. Install with: pip install tensorboard")
    
    def analyze_training_stability(self) -> Dict:
        """
        Analyze training stability and convergence.
        
        Returns:
            Dictionary containing stability metrics
        """
        if len(self.epochs) < 2:
            return {}
        
        train_losses, val_losses = self.get_loss_history()
        
        # Calculate loss trends
        train_trend = np.polyfit(range(len(train_losses)), train_losses, 1)[0]
        val_trend = np.polyfit(range(len(val_losses)), val_losses, 1)[0] if val_losses else 0
        
        # Calculate loss volatility
        train_volatility = np.std(train_losses)
        val_volatility = np.std(val_losses) if val_losses else 0
        
        # Detect overfitting
        overfitting_score = self._detect_overfitting()
        
        return {
            'train_trend': float(train_trend),
            'val_trend': float(val_trend),
            'train_volatility': float(train_volatility),
            'val_volatility': float(val_volatility),
            'overfitting_score': overfitting_score,
            'convergence_status': self._assess_convergence()
        }
    
    def save(self, path: Union[str, Path]):
        """
        Save the complete training record to JSON file.
        
        Args:
            path: Path to save the training record
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        data = {
            'experiment_name': self.experiment_name,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'experiment_metadata': self.experiment_metadata,
            'epochs': [self._serialize_epoch(epoch) for epoch in self.epochs],
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'early_stopping_info': self.early_stopping_info,
            'training_summary': self.get_training_summary(),
            'created_at': self.start_time.isoformat(),
            'saved_at': datetime.now().isoformat()
        }
        
        def json_serializer(obj):
            """Custom JSON serializer to handle special float values."""
            import math
            if isinstance(obj, float):
                if obj == float('inf'):
                    return "Infinity"
                elif obj == float('-inf'):
                    return "-Infinity"
                elif math.isnan(obj):
                    return "NaN"
            return str(obj)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=json_serializer)
        
        print(f"Training record saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainingRecorder':
        """
        Load a training record from JSON file.
        
        Args:
            path: Path to the training record file
            
        Returns:
            TrainingRecorder instance
        """
        path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        recorder = cls(
            experiment_name=data['experiment_name'],
            model_config=data['model_config'],
            training_config=data['training_config']
        )
        
        # Restore metadata
        recorder.experiment_metadata = data['experiment_metadata']
        recorder.best_epoch = data['best_epoch']
        recorder.best_val_loss = data['best_val_loss']
        recorder.early_stopping_info = data['early_stopping_info']
        
        # Restore epoch records
        for epoch_data in data['epochs']:
            recorder.epochs.append(recorder._deserialize_epoch(epoch_data))
        
        print(f"Training record loaded from {path}")
        return recorder
    
    def compare_with(self, other_recorder: 'TrainingRecorder') -> Dict:
        """
        Compare this training record with another.
        
        Args:
            other_recorder: Another TrainingRecorder instance
            
        Returns:
            Dictionary containing comparison results
        """
        self_summary = self.get_training_summary()
        other_summary = other_recorder.get_training_summary()
        
        comparison = {
            'experiment_names': [self.experiment_name, other_recorder.experiment_name],
            'total_epochs': [self_summary.get('total_epochs', 0), other_summary.get('total_epochs', 0)],
            'best_val_loss': [self_summary.get('best_val_loss'), other_summary.get('best_val_loss')],
            'total_time': [self_summary.get('total_time', 0), other_summary.get('total_time', 0)],
            'convergence_comparison': self._compare_convergence(other_recorder)
        }
        
        return comparison
    
    def _serialize_epoch(self, epoch: EpochRecord) -> Dict:
        """Serialize epoch record for JSON storage."""
        data = asdict(epoch)
        data['timestamp'] = epoch.timestamp.isoformat()
        return data
    
    def _deserialize_epoch(self, data: Dict) -> EpochRecord:
        """Deserialize epoch record from JSON data."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return EpochRecord(**data)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
    
    def _get_device_info(self) -> Dict:
        """Get device information."""
        device_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            device_info['cuda_device_name'] = torch.cuda.get_device_name(0)
            device_info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory
        
        return device_info
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'platform': sys.platform,
        }
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return None
    
    def _analyze_convergence(self) -> Dict:
        """Analyze convergence behavior."""
        if len(self.epochs) < 10:
            return {'status': 'insufficient_data'}
        
        train_losses, val_losses = self.get_loss_history()
        
        # Check if loss is decreasing
        recent_window = 5
        recent_train = train_losses[-recent_window:]
        early_train = train_losses[:recent_window]
        
        improvement = np.mean(early_train) - np.mean(recent_train)
        
        return {
            'status': 'converging' if improvement > 0.001 else 'plateaued',
            'improvement': float(improvement),
            'stability': np.std(recent_train)
        }
    
    def _detect_overfitting(self) -> float:
        """Detect overfitting based on train/val loss divergence."""
        train_losses, val_losses = self.get_loss_history()
        
        if len(val_losses) < 5:
            return 0.0
        
        # Calculate divergence in the last portion of training
        window_size = min(10, len(val_losses) // 2)
        recent_train = train_losses[-window_size:]
        recent_val = val_losses[-window_size:]
        
        train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0]
        val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]
        
        # Overfitting score: negative train trend + positive val trend
        overfitting_score = -train_trend + max(0, val_trend)
        return float(overfitting_score)
    
    def _assess_convergence(self) -> str:
        """Assess convergence status."""
        if len(self.epochs) < 5:
            return 'insufficient_data'
        
        train_losses, _ = self.get_loss_history()
        recent_losses = train_losses[-5:]
        
        # Check if loss is stable
        if np.std(recent_losses) < 0.001:
            return 'converged'
        elif np.mean(recent_losses) < train_losses[0] * 0.5:
            return 'converging'
        else:
            return 'not_converged'
    
    def _compare_convergence(self, other: 'TrainingRecorder') -> Dict:
        """Compare convergence with another recorder."""
        self_best = self.get_best_epoch()
        other_best = other.get_best_epoch()
        
        if not self_best or not other_best:
            return {'status': 'insufficient_data'}
        
        return {
            'better_performer': self.experiment_name if self_best.val_loss < other_best.val_loss else other.experiment_name,
            'loss_difference': abs(self_best.val_loss - other_best.val_loss),
            'epoch_difference': abs(self_best.epoch - other_best.epoch)
        }