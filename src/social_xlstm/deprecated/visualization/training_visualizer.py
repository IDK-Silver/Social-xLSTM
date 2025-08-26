"""
DEPRECATED MODULE: Training Visualizer for Social-xLSTM Project

This module is deprecated and will be removed in future versions.

RECOMMENDED ALTERNATIVES:
- For basic visualization: Use `social_xlstm.metrics.plotter.TrainingMetricsPlotter`
- For CLI plotting: Use `scripts/utils/generate_metrics_plots.py`

This module provides comprehensive visualization capabilities for training records.
It works in conjunction with TrainingRecorder to create various plots and analyses.

This module is kept for backward compatibility but is no longer maintained.

Author: Social-xLSTM Project Team
License: MIT
"""

import warnings
warnings.warn(
    "The 'training_visualizer' module is deprecated. Use 'social_xlstm.metrics.plotter' for new implementations.",
    DeprecationWarning,
    stacklevel=2
)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

from ...training.recorder import TrainingRecorder, EpochRecord


class TrainingVisualizer:
    """
    Comprehensive visualization system for training records.
    
    This class provides various plotting functions for analyzing training progress,
    model performance, and experiment comparisons. It's designed to work with
    TrainingRecorder data but is kept separate for modularity.
    
    Features:
    - Basic training curves (loss, metrics, learning rate)
    - Advanced analysis plots (gradient flow, convergence, stability)
    - Comparison plots (multiple experiments)
    - Performance dashboards
    - Statistical analyses
    
    Example:
        recorder = TrainingRecorder.load("experiment.json")
        visualizer = TrainingVisualizer()
        visualizer.plot_training_dashboard(recorder, save_path="dashboard.png")
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', palette: str = 'Set2'):
        """
        Initialize the visualizer with style settings.
        
        Args:
            style: Matplotlib style to use
            palette: Color palette for plots
        """
        self.style = style
        self.palette = palette
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib style settings."""
        plt.style.use(self.style)
        sns.set_palette(self.palette)
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def plot_basic_training_curves(self, recorder: TrainingRecorder, 
                                  save_path: Optional[Union[str, Path]] = None,
                                  figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot basic training curves: loss, metrics, learning rate, and time.
        
        Args:
            recorder: TrainingRecorder instance with training data
            save_path: Optional path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Training Curves - {recorder.experiment_name}', fontsize=16)
        
        if not recorder.epochs:
            print("No epochs recorded.")
            return fig
        
        epochs = range(1, len(recorder.epochs) + 1)
        
        # 1. Loss curves
        self._plot_loss_curves(recorder, axes[0, 0])
        
        # 2. Primary metric curves (MAE)
        self._plot_metric_curves(recorder, axes[0, 1], metric='mae')
        
        # 3. Learning rate
        self._plot_learning_rate(recorder, axes[1, 0])
        
        # 4. Epoch time
        self._plot_epoch_time(recorder, axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_advanced_metrics(self, recorder: TrainingRecorder,
                            save_path: Optional[Union[str, Path]] = None,
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot advanced metrics: all evaluation metrics, gradient norms, memory usage.
        
        Args:
            recorder: TrainingRecorder instance
            save_path: Optional save path
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f'Advanced Metrics - {recorder.experiment_name}', fontsize=16)
        
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # All metrics comparison
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_all_metrics(recorder, ax1)
        
        # Gradient norm
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_gradient_norm(recorder, ax2)
        
        # Memory usage
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_memory_usage(recorder, ax3)
        
        # Convergence analysis
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_convergence_analysis(recorder, ax4)
        
        # Overfitting analysis
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_overfitting_analysis(recorder, ax5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_training_dashboard(self, recorder: TrainingRecorder,
                              save_path: Optional[Union[str, Path]] = None,
                              figsize: Tuple[int, int] = (20, 12)) -> plt.Figure:
        """
        Create a comprehensive training dashboard with multiple plots.
        
        Args:
            recorder: TrainingRecorder instance
            save_path: Optional save path
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f'Training Dashboard - {recorder.experiment_name}', fontsize=18, y=0.98)
        
        # Create grid
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main loss plot (larger)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_loss_curves(recorder, ax1)
        
        # Training summary
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_training_summary(recorder, ax2)
        
        # All metrics
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_all_metrics(recorder, ax3)
        
        # Learning rate
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_learning_rate(recorder, ax4)
        
        # Best epoch highlight
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_best_epoch_details(recorder, ax5)
        
        # Stability metrics
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_stability_metrics(recorder, ax6)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_experiment_comparison(self, recorders: List[TrainingRecorder],
                                 save_path: Optional[Union[str, Path]] = None,
                                 figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Compare multiple experiments side by side.
        
        Args:
            recorders: List of TrainingRecorder instances
            save_path: Optional save path
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Experiment Comparison', fontsize=16)
        
        # Loss comparison
        self._plot_multi_experiment_losses(recorders, axes[0, 0])
        
        # Best performance comparison
        self._plot_performance_comparison(recorders, axes[0, 1])
        
        # Convergence speed comparison
        self._plot_convergence_comparison(recorders, axes[1, 0])
        
        # Training time comparison
        self._plot_time_comparison(recorders, axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_metric_evolution(self, recorder: TrainingRecorder, 
                            metrics: List[str] = ['mae', 'mse', 'rmse', 'r2'],
                            save_path: Optional[Union[str, Path]] = None,
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot the evolution of multiple metrics over training.
        
        Args:
            recorder: TrainingRecorder instance
            metrics: List of metric names to plot
            save_path: Optional save path
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        num_metrics = len(metrics)
        
        # Handle different numbers of metrics
        if num_metrics == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
        elif num_metrics == 2:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, axes = plt.subplots((num_metrics + 1) // 2, 2, figsize=figsize)
            axes = axes.flatten()
        
        fig.suptitle(f'Metric Evolution - {recorder.experiment_name}', fontsize=16)
        
        for idx, metric in enumerate(metrics):
            if idx < len(axes):
                self._plot_metric_curves(recorder, axes[idx], metric=metric)
        
        # Hide unused subplots
        for idx in range(num_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_time_series_comparison(self, predictions: np.ndarray, targets: np.ndarray,
                                   inputs: np.ndarray, feature_names: List[str],
                                   sample_idx: int = 0, feature_idx: int = 0,
                                   save_path: Optional[Union[str, Path]] = None,
                                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot time series comparison between predictions and ground truth.
        
        Args:
            predictions: Model predictions array
            targets: Ground truth targets array
            inputs: Input sequences array
            feature_names: List of feature names
            sample_idx: Index of sample to plot
            feature_idx: Index of feature to plot
            save_path: Optional save path
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract sample data
        input_sample = inputs[sample_idx, :, feature_idx]
        target_sample = targets[sample_idx, :, feature_idx]
        pred_sample = predictions[sample_idx, :, feature_idx]
        
        # Create time axis
        seq_len = len(input_sample)
        pred_len = len(target_sample)
        
        x_input = np.arange(seq_len)
        x_target = np.arange(seq_len, seq_len + pred_len)
        
        # Plot
        ax.plot(x_input, input_sample, 'g-', label='Input History', linewidth=2, marker='o', markersize=4)
        ax.plot(x_target, target_sample, 'b-', label='Ground Truth', linewidth=2, marker='o', markersize=4)
        ax.plot(x_target, pred_sample, 'r--', label='Predicted', linewidth=2, marker='s', markersize=4)
        
        # Add prediction start line
        ax.axvline(x=seq_len-0.5, color='k', linestyle=':', alpha=0.7, label='Prediction Start')
        
        ax.set_title(f'Time Series Prediction: {feature_names[feature_idx]}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Normalized Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_prediction_scatter(self, predictions: np.ndarray, targets: np.ndarray,
                               metrics: Dict[str, float],
                               save_path: Optional[Union[str, Path]] = None,
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot prediction vs ground truth scatter plot.
        
        Args:
            predictions: Model predictions array
            targets: Ground truth targets array
            metrics: Dictionary of evaluation metrics
            save_path: Optional save path
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Flatten arrays
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Create scatter plot
        ax.scatter(target_flat, pred_flat, alpha=0.6, s=10, color='blue')
        
        # Add perfect prediction line
        min_val = min(target_flat.min(), pred_flat.min())
        max_val = max(target_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Add metrics text
        textstr = f"R² = {metrics.get('r2', 0):.3f}\nMAE = {metrics.get('mae', 0):.4f}\nRMSE = {metrics.get('rmse', 0):.4f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax.set_title('Predictions vs Ground Truth')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_feature_performance(self, predictions: np.ndarray, targets: np.ndarray,
                                feature_names: List[str],
                                save_path: Optional[Union[str, Path]] = None,
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot performance comparison across different features.
        
        Args:
            predictions: Model predictions array
            targets: Ground truth targets array
            feature_names: List of feature names
            save_path: Optional save path
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate MAE for each feature
        mae_per_feature = []
        for feat_idx in range(len(feature_names)):
            pred_feat = predictions[:, :, feat_idx].flatten()
            target_feat = targets[:, :, feat_idx].flatten()
            mae = np.mean(np.abs(pred_feat - target_feat))
            mae_per_feature.append(mae)
        
        # Create bar plot
        x_pos = range(len(feature_names))
        bars = ax.bar(x_pos, mae_per_feature, color='lightcoral', edgecolor='darkred', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, mae_per_feature):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_per_feature)*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title('Mean Absolute Error by Feature')
        ax.set_xlabel('Feature')
        ax.set_ylabel('MAE')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_prediction_dashboard(self, predictions: np.ndarray, targets: np.ndarray,
                                 inputs: np.ndarray, feature_names: List[str],
                                 metrics: Dict[str, float],
                                 save_path: Optional[Union[str, Path]] = None,
                                 figsize: Tuple[int, int] = (20, 12)) -> plt.Figure:
        """
        Create a comprehensive prediction evaluation dashboard.
        
        Args:
            predictions: Model predictions array
            targets: Ground truth targets array
            inputs: Input sequences array
            feature_names: List of feature names
            metrics: Dictionary of evaluation metrics
            save_path: Optional save path
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle('Prediction Evaluation Dashboard', fontsize=18, y=0.98)
        
        # Create grid layout
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Time series comparison for first feature
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_single_time_series(predictions, targets, inputs, feature_names, ax1, 0, 0)
        
        # Prediction scatter plot
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_single_prediction_scatter(predictions, targets, metrics, ax2)
        
        # Feature performance comparison
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_single_feature_performance(predictions, targets, feature_names, ax3)
        
        # Multi-feature time series (if multiple features)
        if len(feature_names) > 1:
            ax4 = fig.add_subplot(gs[1, :])
            self._plot_multi_feature_time_series(predictions, targets, inputs, feature_names, ax4)
        else:
            # If only one feature, show different samples
            ax4 = fig.add_subplot(gs[1, :])
            self._plot_multiple_samples(predictions, targets, inputs, feature_names, ax4)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def create_training_report(self, recorder: TrainingRecorder,
                             output_dir: Union[str, Path],
                             include_all: bool = True):
        """
        Generate a complete training report with multiple figures.
        
        Args:
            recorder: TrainingRecorder instance
            output_dir: Directory to save the report
            include_all: Whether to include all possible plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Basic training curves
        self.plot_basic_training_curves(
            recorder, 
            save_path=output_dir / "basic_training_curves.png"
        )
        
        # Training dashboard
        self.plot_training_dashboard(
            recorder,
            save_path=output_dir / "training_dashboard.png"
        )
        
        if include_all:
            # Advanced metrics
            self.plot_advanced_metrics(
                recorder,
                save_path=output_dir / "advanced_metrics.png"
            )
            
            # Metric evolution
            self.plot_metric_evolution(
                recorder,
                save_path=output_dir / "metric_evolution.png"
            )
        
        # Generate summary text report
        self._generate_text_report(recorder, output_dir / "training_summary.txt")
        
        print(f"Training report generated in {output_dir}")
    
    # Private plotting methods
    
    def _plot_loss_curves(self, recorder: TrainingRecorder, ax: plt.Axes):
        """Plot training and validation loss curves."""
        epochs = range(1, len(recorder.epochs) + 1)
        train_losses, val_losses = recorder.get_loss_history()
        
        ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        if val_losses:
            ax.plot(epochs[:len(val_losses)], val_losses, 'r-', label='Val Loss', linewidth=2)
            
            # Mark best epoch
            if recorder.best_epoch is not None:
                best_epoch_data = recorder.epochs[recorder.best_epoch]
                ax.scatter([best_epoch_data.epoch + 1], [best_epoch_data.val_loss],
                          color='green', s=100, marker='*', label='Best Epoch', zorder=5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_metric_curves(self, recorder: TrainingRecorder, ax: plt.Axes, metric: str = 'mae'):
        """Plot specific metric curves."""
        epochs = range(1, len(recorder.epochs) + 1)
        
        train_metric = recorder.get_metric_history(metric, 'train')
        val_metric = recorder.get_metric_history(metric, 'val')
        
        if any(train_metric):
            ax.plot(epochs, train_metric, 'b-', label=f'Train {metric.upper()}', linewidth=2)
        if any(val_metric):
            ax.plot(epochs[:len(val_metric)], val_metric, 'r-', label=f'Val {metric.upper()}', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_learning_rate(self, recorder: TrainingRecorder, ax: plt.Axes):
        """Plot learning rate schedule."""
        epochs = range(1, len(recorder.epochs) + 1)
        lr_history = [epoch.learning_rate for epoch in recorder.epochs]
        
        ax.plot(epochs, lr_history, 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    def _plot_epoch_time(self, recorder: TrainingRecorder, ax: plt.Axes):
        """Plot epoch training time."""
        epochs = range(1, len(recorder.epochs) + 1)
        time_history = [epoch.epoch_time for epoch in recorder.epochs if epoch.epoch_time > 0]
        
        if time_history:
            ax.plot(epochs[:len(time_history)], time_history, 'purple', linewidth=2)
            
            # Add average line
            avg_time = np.mean(time_history)
            ax.axhline(y=avg_time, color='red', linestyle='--', label=f'Average: {avg_time:.2f}s')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Training Time per Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_all_metrics(self, recorder: TrainingRecorder, ax: plt.Axes):
        """Plot all available metrics in one plot."""
        epochs = range(1, len(recorder.epochs) + 1)
        
        # Get all available metrics
        if recorder.epochs and recorder.epochs[0].train_metrics:
            metric_names = list(recorder.epochs[0].train_metrics.keys())
            
            # Create subplots for each metric
            for metric in metric_names:
                train_values = recorder.get_metric_history(metric, 'train')
                val_values = recorder.get_metric_history(metric, 'val')
                
                if any(train_values):
                    ax.plot(epochs, train_values, label=f'Train {metric}', alpha=0.7)
                if any(val_values):
                    ax.plot(epochs[:len(val_values)], val_values, label=f'Val {metric}', 
                           linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title('All Metrics Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_gradient_norm(self, recorder: TrainingRecorder, ax: plt.Axes):
        """Plot gradient norm evolution."""
        epochs = []
        grad_norms = []
        
        for epoch in recorder.epochs:
            if epoch.gradient_norm is not None:
                epochs.append(epoch.epoch + 1)
                grad_norms.append(epoch.gradient_norm)
        
        if grad_norms:
            ax.plot(epochs, grad_norms, 'orange', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm Evolution')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No gradient norm data', ha='center', va='center',
                   transform=ax.transAxes)
    
    def _plot_memory_usage(self, recorder: TrainingRecorder, ax: plt.Axes):
        """Plot memory usage over training."""
        epochs = []
        memory = []
        
        for epoch in recorder.epochs:
            if epoch.memory_usage is not None:
                epochs.append(epoch.epoch + 1)
                memory.append(epoch.memory_usage)
        
        if memory:
            ax.plot(epochs, memory, 'brown', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Memory (MB)')
            ax.set_title('Memory Usage')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No memory usage data', ha='center', va='center',
                   transform=ax.transAxes)
    
    def _plot_convergence_analysis(self, recorder: TrainingRecorder, ax: plt.Axes):
        """Plot convergence analysis."""
        train_losses, _ = recorder.get_loss_history()
        
        if len(train_losses) >= 2:  # Reduced threshold
            # Plot raw data
            ax.plot(range(1, len(train_losses) + 1), train_losses, 'b-', alpha=0.3, label='Raw')
            
            # Calculate moving average if enough data
            if len(train_losses) >= 5:
                window = min(5, len(train_losses) // 2)
                moving_avg = np.convolve(train_losses, np.ones(window)/window, mode='valid')
                epochs = range(window, len(train_losses) + 1)
                ax.plot(epochs, moving_avg, 'b-', linewidth=2, label=f'{window}-epoch MA')
            
            # Add trend line
            z = np.polyfit(range(len(train_losses)), train_losses, 1)
            p = np.poly1d(z)
            ax.plot(range(1, len(train_losses) + 1), p(range(len(train_losses))), 
                   'r--', label=f'Trend: {z[0]:.6f}')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Convergence Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for convergence analysis', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_overfitting_analysis(self, recorder: TrainingRecorder, ax: plt.Axes):
        """Plot overfitting analysis."""
        train_losses, val_losses = recorder.get_loss_history()
        
        if len(val_losses) > 5:
            epochs = range(1, len(val_losses) + 1)
            
            # Calculate loss difference
            loss_diff = [val - train for val, train in zip(val_losses, train_losses[:len(val_losses)])]
            
            ax.plot(epochs, loss_diff, 'purple', linewidth=2)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Color regions
            ax.fill_between(epochs, 0, loss_diff, where=[d > 0 for d in loss_diff],
                          color='red', alpha=0.2, label='Overfitting')
            ax.fill_between(epochs, 0, loss_diff, where=[d <= 0 for d in loss_diff],
                          color='green', alpha=0.2, label='Underfitting')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Val Loss - Train Loss')
            ax.set_title('Overfitting Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_training_summary(self, recorder: TrainingRecorder, ax: plt.Axes):
        """Plot training summary as text."""
        ax.axis('off')
        
        summary = recorder.get_training_summary()
        
        text = f"Training Summary\n" + "="*30 + "\n"
        text += f"Total Epochs: {summary.get('total_epochs', 0)}\n"
        text += f"Total Time: {summary.get('total_time', 0):.2f}s\n"
        text += f"Avg Epoch Time: {summary.get('avg_epoch_time', 0):.2f}s\n"
        text += f"Best Epoch: {summary.get('best_epoch', 'N/A')}\n"
        text += f"Best Train Loss: {summary.get('best_train_loss', 0):.6f}\n"
        text += f"Best Val Loss: {summary.get('best_val_loss', 0):.6f}\n"
        text += f"Final LR: {summary.get('final_learning_rate', 0):.6f}\n"
        text += f"Status: {summary.get('convergence_info', {}).get('status', 'N/A')}"
        
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', fontfamily='monospace')
    
    def _plot_best_epoch_details(self, recorder: TrainingRecorder, ax: plt.Axes):
        """Plot best epoch details."""
        best_epoch = recorder.get_best_epoch()
        
        if best_epoch:
            ax.axis('off')
            
            text = f"Best Epoch Details\n" + "="*30 + "\n"
            text += f"Epoch: {best_epoch.epoch + 1}\n"
            text += f"Train Loss: {best_epoch.train_loss:.6f}\n"
            text += f"Val Loss: {best_epoch.val_loss:.6f}\n"
            
            if best_epoch.train_metrics:
                text += "\nTrain Metrics:\n"
                for metric, value in best_epoch.train_metrics.items():
                    text += f"  {metric}: {value:.6f}\n"
            
            if best_epoch.val_metrics:
                text += "\nVal Metrics:\n"
                for metric, value in best_epoch.val_metrics.items():
                    text += f"  {metric}: {value:.6f}\n"
            
            ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='center', fontfamily='monospace')
    
    def _plot_stability_metrics(self, recorder: TrainingRecorder, ax: plt.Axes):
        """Plot stability metrics."""
        stability = recorder.analyze_training_stability()
        
        ax.axis('off')
        
        text = f"Stability Analysis\n" + "="*30 + "\n"
        text += f"Train Trend: {stability.get('train_trend', 0):.6f}\n"
        text += f"Val Trend: {stability.get('val_trend', 0):.6f}\n"
        text += f"Train Volatility: {stability.get('train_volatility', 0):.6f}\n"
        text += f"Val Volatility: {stability.get('val_volatility', 0):.6f}\n"
        text += f"Overfitting Score: {stability.get('overfitting_score', 0):.4f}\n"
        text += f"Convergence: {stability.get('convergence_status', 'N/A')}"
        
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', fontfamily='monospace')
    
    def _plot_multi_experiment_losses(self, recorders: List[TrainingRecorder], ax: plt.Axes):
        """Plot losses from multiple experiments."""
        for recorder in recorders:
            train_losses, val_losses = recorder.get_loss_history()
            epochs = range(1, len(train_losses) + 1)
            
            ax.plot(epochs, train_losses, label=f'{recorder.experiment_name} (train)', alpha=0.7)
            if val_losses:
                ax.plot(epochs[:len(val_losses)], val_losses, 
                       label=f'{recorder.experiment_name} (val)', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_comparison(self, recorders: List[TrainingRecorder], ax: plt.Axes):
        """Plot performance comparison bar chart."""
        names = []
        best_losses = []
        
        for recorder in recorders:
            names.append(recorder.experiment_name)
            best_epoch = recorder.get_best_epoch()
            best_losses.append(best_epoch.val_loss if best_epoch and best_epoch.val_loss else float('inf'))
        
        bars = ax.bar(names, best_losses)
        
        # Color best performer
        min_idx = np.argmin(best_losses)
        bars[min_idx].set_color('green')
        
        ax.set_ylabel('Best Validation Loss')
        ax.set_title('Performance Comparison')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_convergence_comparison(self, recorders: List[TrainingRecorder], ax: plt.Axes):
        """Plot convergence speed comparison."""
        for recorder in recorders:
            train_losses, _ = recorder.get_loss_history()
            
            # Normalize losses for comparison
            if train_losses:
                normalized_losses = np.array(train_losses) / train_losses[0]
                epochs = range(1, len(normalized_losses) + 1)
                ax.plot(epochs, normalized_losses, label=recorder.experiment_name, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Normalized Loss')
        ax.set_title('Convergence Speed Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_time_comparison(self, recorders: List[TrainingRecorder], ax: plt.Axes):
        """Plot training time comparison."""
        names = []
        total_times = []
        avg_times = []
        
        for recorder in recorders:
            summary = recorder.get_training_summary()
            names.append(recorder.experiment_name)
            total_times.append(summary.get('total_time', 0))
            avg_times.append(summary.get('avg_epoch_time', 0))
        
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, total_times, width, label='Total Time')
        ax.bar(x + width/2, avg_times, width, label='Avg Epoch Time')
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45)
        ax.legend()
    
    def _generate_text_report(self, recorder: TrainingRecorder, output_path: Path):
        """Generate a text summary report."""
        summary = recorder.get_training_summary()
        
        with open(output_path, 'w') as f:
            f.write(f"Training Report for {recorder.experiment_name}\n")
            f.write("="*50 + "\n\n")
            
            f.write("Experiment Metadata:\n")
            f.write(f"- Start Time: {recorder.start_time}\n")
            f.write(f"- Git Commit: {recorder.experiment_metadata.get('git_commit', 'N/A')}\n")
            f.write(f"- Python Version: {recorder.experiment_metadata.get('python_version', 'N/A')}\n")
            f.write(f"- PyTorch Version: {recorder.experiment_metadata.get('pytorch_version', 'N/A')}\n")
            f.write(f"- CUDA Version: {recorder.experiment_metadata.get('cuda_version', 'N/A')}\n")
            f.write("\n")
            
            f.write("Training Summary:\n")
            for key, value in summary.items():
                if isinstance(value, dict):
                    f.write(f"- {key}:\n")
                    for k, v in value.items():
                        f.write(f"  - {k}: {v}\n")
                else:
                    f.write(f"- {key}: {value}\n")
            f.write("\n")
            
            f.write("Model Configuration:\n")
            for key, value in recorder.model_config.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
            
            f.write("Training Configuration:\n")
            for key, value in recorder.training_config.items():
                f.write(f"- {key}: {value}\n")
    
    # Helper methods for prediction visualization
    
    def _plot_single_time_series(self, predictions: np.ndarray, targets: np.ndarray,
                               inputs: np.ndarray, feature_names: List[str],
                               ax: plt.Axes, sample_idx: int = 0, feature_idx: int = 0):
        """Plot single time series comparison."""
        # Extract sample data
        input_sample = inputs[sample_idx, :, feature_idx]
        target_sample = targets[sample_idx, :, feature_idx]
        pred_sample = predictions[sample_idx, :, feature_idx]
        
        # Create time axis
        seq_len = len(input_sample)
        pred_len = len(target_sample)
        
        x_input = np.arange(seq_len)
        x_target = np.arange(seq_len, seq_len + pred_len)
        
        # Plot
        ax.plot(x_input, input_sample, 'g-', label='Input History', linewidth=2, marker='o', markersize=3)
        ax.plot(x_target, target_sample, 'b-', label='Ground Truth', linewidth=2, marker='o', markersize=3)
        ax.plot(x_target, pred_sample, 'r--', label='Predicted', linewidth=2, marker='s', markersize=3)
        
        # Add prediction start line
        ax.axvline(x=seq_len-0.5, color='k', linestyle=':', alpha=0.7, label='Prediction Start')
        
        ax.set_title(f'Time Series: {feature_names[feature_idx]}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_single_prediction_scatter(self, predictions: np.ndarray, targets: np.ndarray,
                                      metrics: Dict[str, float], ax: plt.Axes):
        """Plot single prediction scatter plot."""
        # Flatten arrays
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Create scatter plot
        ax.scatter(target_flat, pred_flat, alpha=0.6, s=8, color='blue')
        
        # Add perfect prediction line
        min_val = min(target_flat.min(), pred_flat.min())
        max_val = max(target_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Add metrics text
        textstr = f"R² = {metrics.get('r2', 0):.3f}\nMAE = {metrics.get('mae', 0):.4f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        ax.set_title('Predictions vs Ground Truth')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_single_feature_performance(self, predictions: np.ndarray, targets: np.ndarray,
                                       feature_names: List[str], ax: plt.Axes):
        """Plot single feature performance bar chart."""
        # Calculate MAE for each feature
        mae_per_feature = []
        for feat_idx in range(len(feature_names)):
            pred_feat = predictions[:, :, feat_idx].flatten()
            target_feat = targets[:, :, feat_idx].flatten()
            mae = np.mean(np.abs(pred_feat - target_feat))
            mae_per_feature.append(mae)
        
        # Create bar plot
        x_pos = range(len(feature_names))
        bars = ax.bar(x_pos, mae_per_feature, color='lightcoral', edgecolor='darkred', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, mae_per_feature):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_per_feature)*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title('MAE by Feature')
        ax.set_xlabel('Feature')
        ax.set_ylabel('MAE')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
    
    def _plot_multi_feature_time_series(self, predictions: np.ndarray, targets: np.ndarray,
                                       inputs: np.ndarray, feature_names: List[str], ax: plt.Axes):
        """Plot multiple features in one time series plot."""
        sample_idx = 0
        seq_len = inputs.shape[1]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(feature_names)))
        
        for feat_idx, (feature_name, color) in enumerate(zip(feature_names, colors)):
            # Extract data
            input_sample = inputs[sample_idx, :, feat_idx]
            target_sample = targets[sample_idx, :, feat_idx]
            pred_sample = predictions[sample_idx, :, feat_idx]
            
            # Create time axis
            x_input = np.arange(seq_len)
            x_target = np.arange(seq_len, seq_len + len(target_sample))
            
            # Plot (only predictions and targets for clarity)
            ax.plot(x_target, target_sample, '-', color=color, 
                   label=f'{feature_name} (GT)', linewidth=2, alpha=0.7)
            ax.plot(x_target, pred_sample, '--', color=color, 
                   label=f'{feature_name} (Pred)', linewidth=2, alpha=0.7)
        
        # Add prediction start line
        ax.axvline(x=seq_len-0.5, color='k', linestyle=':', alpha=0.7, label='Prediction Start')
        
        ax.set_title('Multi-Feature Time Series Comparison')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Normalized Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_multiple_samples(self, predictions: np.ndarray, targets: np.ndarray,
                             inputs: np.ndarray, feature_names: List[str], ax: plt.Axes):
        """Plot multiple samples for single feature."""
        feature_idx = 0
        seq_len = inputs.shape[1]
        num_samples = min(5, predictions.shape[0])  # Show up to 5 samples
        
        colors = plt.cm.Set1(np.linspace(0, 1, num_samples))
        
        for sample_idx in range(num_samples):
            # Extract data
            target_sample = targets[sample_idx, :, feature_idx]
            pred_sample = predictions[sample_idx, :, feature_idx]
            
            # Create time axis
            x_target = np.arange(seq_len, seq_len + len(target_sample))
            
            # Plot
            ax.plot(x_target, target_sample, '-', color=colors[sample_idx], 
                   label=f'Sample {sample_idx+1} (GT)', linewidth=2, alpha=0.7)
            ax.plot(x_target, pred_sample, '--', color=colors[sample_idx], 
                   label=f'Sample {sample_idx+1} (Pred)', linewidth=2, alpha=0.7)
        
        # Add prediction start line
        ax.axvline(x=seq_len-0.5, color='k', linestyle=':', alpha=0.7, label='Prediction Start')
        
        ax.set_title(f'Multiple Samples: {feature_names[feature_idx]}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)