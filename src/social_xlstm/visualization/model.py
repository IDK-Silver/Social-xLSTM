import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

class TrafficResultsPlotter:
    """專門用於繪製交通預測結果的可視化類"""
    
    def __init__(self, style='seaborn-v0_8', figsize=(15, 10)):
        self.style = style
        self.figsize = figsize
        self._setup_style()
    
    def _setup_style(self):
        """設置繪圖樣式"""
        plt.style.use(self.style)
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 16
        })
    
    def plot_training_curves(self, training_history: Dict[str, list], ax=None) -> plt.Axes:
        """繪製訓練曲線"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        epochs = range(1, len(training_history['train_losses']) + 1)
        
        ax.plot(epochs, training_history['train_losses'], 'b-', label='Training Loss', linewidth=2)
        
        if training_history['val_losses'] and any(loss > 0 for loss in training_history['val_losses']):
            ax.plot(epochs, training_history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_title('Training Loss Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_time_series_comparison(self, predictions: np.ndarray, targets: np.ndarray, 
                                  inputs: np.ndarray, feature_names: List[str], 
                                  sample_idx: int = 0, feature_idx: int = 0, ax=None) -> plt.Axes:
        """繪製時間序列對比圖"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
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
        
        return ax
    
    def plot_prediction_scatter(self, predictions: np.ndarray, targets: np.ndarray, 
                               metrics: Dict[str, float], ax=None) -> plt.Axes:
        """繪製預測 vs 實際值散點圖"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
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
        textstr = f"R² = {metrics['r2']:.3f}\nMAE = {metrics['mae']:.4f}\nRMSE = {metrics['rmse']:.4f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax.set_title('Predictions vs Ground Truth')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_feature_performance(self, predictions: np.ndarray, targets: np.ndarray, 
                                feature_names: List[str], ax=None) -> plt.Axes:
        """繪製不同特徵的性能對比"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
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
        
        return ax
    
    def plot_train_val_comparison(self, train_predictions: np.ndarray, train_targets: np.ndarray,
                                 val_predictions: np.ndarray, val_targets: np.ndarray,
                                 train_metrics: Dict[str, float], val_metrics: Dict[str, float],
                                 ax=None) -> plt.Axes:
        """繪製訓練集和驗證集的預測對比散點圖"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Flatten arrays
        train_pred_flat = train_predictions.flatten()
        train_target_flat = train_targets.flatten()
        val_pred_flat = val_predictions.flatten()
        val_target_flat = val_targets.flatten()
        
        # Create scatter plots
        ax.scatter(train_target_flat, train_pred_flat, alpha=0.6, s=8, color='blue', label='Training')
        ax.scatter(val_target_flat, val_pred_flat, alpha=0.6, s=8, color='red', label='Validation')
        
        # Add perfect prediction line
        all_targets = np.concatenate([train_target_flat, val_target_flat])
        all_preds = np.concatenate([train_pred_flat, val_pred_flat])
        min_val = min(all_targets.min(), all_preds.min())
        max_val = max(all_targets.max(), all_preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
        
        # Add metrics text
        textstr = f"Training:\nR² = {train_metrics['r2']:.3f}\nMAE = {train_metrics['mae']:.4f}\n\nValidation:\nR² = {val_metrics['r2']:.3f}\nMAE = {val_metrics['mae']:.4f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        ax.set_title('Training vs Validation: Predictions vs Ground Truth')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def create_evaluation_dashboard(self, evaluation_data: Dict[str, Any], 
                                  save_path: Optional[str] = None) -> plt.Figure:
        """創建完整的評估儀表板，包含訓練和驗證"""
        has_validation = 'val_predictions' in evaluation_data
        
        if has_validation:
            # 如果有驗證集，使用 2x3 布局
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('LSTM Model Evaluation Dashboard (Train + Validation)', fontsize=16)
            
            # Plot 1: Training curves
            self.plot_training_curves(evaluation_data['training_history'], axes[0, 0])
            
            # Plot 2: Training time series
            self.plot_time_series_comparison(
                evaluation_data['train_predictions'],
                evaluation_data['train_targets'],
                evaluation_data['train_inputs'],
                evaluation_data['feature_names'],
                ax=axes[0, 1]
            )
            axes[0, 1].set_title('Training: Time Series Prediction')
            
            # Plot 3: Validation time series
            self.plot_time_series_comparison(
                evaluation_data['val_predictions'],
                evaluation_data['val_targets'],
                evaluation_data['val_inputs'],
                evaluation_data['feature_names'],
                ax=axes[0, 2]
            )
            axes[0, 2].set_title('Validation: Time Series Prediction')
            
            # Plot 4: Train vs Val scatter comparison
            self.plot_train_val_comparison(
                evaluation_data['train_predictions'],
                evaluation_data['train_targets'],
                evaluation_data['val_predictions'],
                evaluation_data['val_targets'],
                evaluation_data['train_metrics'],
                evaluation_data['val_metrics'],
                ax=axes[1, 0]
            )
            
            # Plot 5: Training feature performance
            self.plot_feature_performance(
                evaluation_data['train_predictions'],
                evaluation_data['train_targets'],
                evaluation_data['feature_names'],
                ax=axes[1, 1]
            )
            axes[1, 1].set_title('Training: MAE by Feature')
            
            # Plot 6: Validation feature performance
            self.plot_feature_performance(
                evaluation_data['val_predictions'],
                evaluation_data['val_targets'],
                evaluation_data['feature_names'],
                ax=axes[1, 2]
            )
            axes[1, 2].set_title('Validation: MAE by Feature')
            
        else:
            # 原來的 2x2 布局（僅訓練）
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            fig.suptitle('LSTM Model Evaluation Dashboard (Training Only)', fontsize=16)
            
            # Plot 1: Training curves
            self.plot_training_curves(evaluation_data['training_history'], axes[0, 0])
            
            # Plot 2: Time series comparison
            self.plot_time_series_comparison(
                evaluation_data['predictions'],
                evaluation_data['targets'],
                evaluation_data['inputs'],
                evaluation_data['feature_names'],
                ax=axes[0, 1]
            )
            
            # Plot 3: Prediction scatter
            self.plot_prediction_scatter(
                evaluation_data['predictions'],
                evaluation_data['targets'],
                evaluation_data['metrics'],
                ax=axes[1, 0]
            )
            
            # Plot 4: Feature performance
            self.plot_feature_performance(
                evaluation_data['predictions'],
                evaluation_data['targets'],
                evaluation_data['feature_names'],
                ax=axes[1, 1]
            )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation dashboard saved to: {save_path}")
        
        return fig