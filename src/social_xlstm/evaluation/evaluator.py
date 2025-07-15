import torch
import numpy as np
import h5py
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Encapsulates all model evaluation logic, focusing on data processing and metrics calculation"""
    
    def __init__(self, model, train_losses, val_losses, config, train_loader, val_loader, device, vd_index=0):
        self.model = model
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader  # Added validation loader
        self.device = device
        self.vd_index = vd_index
        
        # Lazy computation attributes
        self._train_predictions = None
        self._train_targets = None
        self._train_inputs = None
        self._val_predictions = None
        self._val_targets = None
        self._val_inputs = None
        self._train_metrics = None
        self._val_metrics = None
    
    def get_train_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get training set prediction results"""
        if self._train_predictions is None:
            self._compute_train_predictions()
        return self._train_predictions, self._train_targets, self._train_inputs
    
    def get_val_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get validation set prediction results"""
        if self._val_predictions is None and self.val_loader is not None:
            self._compute_val_predictions()
        return self._val_predictions, self._val_targets, self._val_inputs
    
    def get_train_metrics(self) -> Dict[str, float]:
        """Get training set evaluation metrics"""
        if self._train_metrics is None:
            predictions, targets, _ = self.get_train_predictions()
            self._train_metrics = self._compute_metrics(predictions, targets)
        return self._train_metrics
    
    def get_val_metrics(self) -> Dict[str, float]:
        """Get validation set evaluation metrics"""
        if self._val_metrics is None and self.val_loader is not None:
            predictions, targets, _ = self.get_val_predictions()
            self._val_metrics = self._compute_metrics(predictions, targets)
        return self._val_metrics
    
    def _compute_train_predictions(self):
        """Compute training set predictions"""
        self._train_predictions, self._train_targets, self._train_inputs = self._compute_predictions_for_loader(self.train_loader)
    
    def _compute_val_predictions(self):
        """Compute validation set predictions"""
        if self.val_loader is not None:
            self._val_predictions, self._val_targets, self._val_inputs = self._compute_predictions_for_loader(self.val_loader)
    
    def _compute_predictions_for_loader(self, data_loader):
        """Generic prediction computation function"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    # Extract data for specific VD
                    data = batch['input_seq'][:, :, self.vd_index, :]
                    target = batch['target_seq'][:, :, self.vd_index, :]
                else:
                    raise ValueError("Expected dictionary format from dataset")
                
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Store results
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_inputs.append(data.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        inputs = np.concatenate(all_inputs, axis=0)
        
        return predictions, targets, inputs
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics"""
        # Flatten for overall metrics
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Remove NaN values first (critical fix for data quality issues)
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        
        if not np.any(valid_mask):
            # All values are NaN - return default metrics
            return {
                'mae': float('nan'),
                'mse': float('nan'),
                'rmse': float('nan'),
                'mape': float('inf'),
                'r2': 0.0
            }
        
        # Use only valid (non-NaN) values
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        # Calculate basic metrics
        mae = np.mean(np.abs(pred_valid - target_valid))
        mse = np.mean((pred_valid - target_valid) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE (avoid division by zero and NaN)
        non_zero_mask = target_valid != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((pred_valid[non_zero_mask] - target_valid[non_zero_mask]) / target_valid[non_zero_mask])) * 100
        else:
            mape = float('inf')  # Use float('inf') instead of string for consistency
        
        # R² (handle zero variance case properly)
        target_mean = np.mean(target_valid)
        ss_res = np.sum((target_valid - pred_valid) ** 2)
        ss_tot = np.sum((target_valid - target_mean) ** 2)
        
        if ss_tot == 0:
            # Zero variance in targets - R² is undefined, return 0
            r2 = 0.0
        elif np.isnan(ss_tot) or np.isnan(ss_res):
            # Safeguard against remaining NaN issues
            r2 = 0.0
        else:
            r2 = 1 - (ss_res / ss_tot)
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        }
    
    def get_evaluation_data(self) -> Dict[str, Any]:
        """Get all evaluation-related data including training and validation"""
        train_pred, train_targ, train_inp = self.get_train_predictions()
        train_metrics = self.get_train_metrics()
        
        eval_data = {
            'train_predictions': train_pred,
            'train_targets': train_targ,
            'train_inputs': train_inp,
            'train_metrics': train_metrics,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            },
            'feature_names': self.config.selected_features,
            'vd_index': self.vd_index,
            'config': self.config
        }
        
        # If validation set exists, add validation data
        if self.val_loader is not None:
            val_pred, val_targ, val_inp = self.get_val_predictions()
            val_metrics = self.get_val_metrics()
            
            eval_data.update({
                'val_predictions': val_pred,
                'val_targets': val_targ,
                'val_inputs': val_inp,
                'val_metrics': val_metrics
            })
        
        return eval_data
    
    def evaluate(self, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance with given targets and predictions.
        
        Args:
            targets: Ground truth values
            predictions: Model predictions
            
        Returns:
            Dictionary containing evaluation metrics
        """
        return self._compute_metrics(predictions, targets)


class DatasetDiagnostics:
    """Diagnostic utilities for comprehensive dataset and model analysis."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_h5_dataset(self, h5_path: str, vd_id: str) -> Dict[str, Any]:
        """Comprehensive analysis of H5 dataset."""
        self.logger.info(f"Analyzing dataset: {h5_path} for VD: {vd_id}")
        
        results = {}
        
        with h5py.File(h5_path, 'r') as h5file:
            if vd_id not in h5file:
                self.logger.error(f"VD {vd_id} not found in H5 file")
                available_vds = list(h5file.keys())
                self.logger.info(f"Available VDs: {available_vds}")
                return {'error': f'VD {vd_id} not found', 'available_vds': available_vds}
            
            vd_group = h5file[vd_id]
            
            # 1. Data completeness analysis
            for feature in vd_group.keys():
                data = vd_group[feature][:]
                total_samples = len(data)
                missing_samples = np.sum(np.isnan(data))
                valid_rate = (total_samples - missing_samples) / total_samples * 100
                
                results[f"{feature}_valid_rate"] = valid_rate
                results[f"{feature}_total_samples"] = total_samples
                results[f"{feature}_missing_samples"] = missing_samples
                
                self.logger.info(f"Feature {feature}: {valid_rate:.2f}% valid ({total_samples-missing_samples}/{total_samples})")
            
            # 2. Data distribution analysis
            for feature in vd_group.keys():
                data = vd_group[feature][:]
                valid_data = data[~np.isnan(data)]
                
                if len(valid_data) > 0:
                    results[f"{feature}_mean"] = np.mean(valid_data)
                    results[f"{feature}_std"] = np.std(valid_data)
                    results[f"{feature}_min"] = np.min(valid_data)
                    results[f"{feature}_max"] = np.max(valid_data)
                    results[f"{feature}_range"] = np.max(valid_data) - np.min(valid_data)
            
            # 3. Temporal patterns
            if 'timestamps' in vd_group:
                timestamps = vd_group['timestamps'][:]
                results['temporal_samples'] = len(timestamps)
                if len(timestamps) > 0:
                    results['first_timestamp'] = timestamps[0]
                    results['last_timestamp'] = timestamps[-1]
            
            # 4. Data variance and stationarity
            for feature in ['avg_speed', 'total_volume', 'avg_occupancy']:
                if feature in vd_group:
                    data = vd_group[feature][:]
                    valid_data = data[~np.isnan(data)]
                    
                    if len(valid_data) > 100:
                        chunk_size = len(valid_data) // 10
                        chunk_vars = []
                        for i in range(0, len(valid_data), chunk_size):
                            chunk = valid_data[i:i+chunk_size]
                            if len(chunk) > 10:
                                chunk_vars.append(np.var(chunk))
                        
                        if chunk_vars:
                            var_stability = np.std(chunk_vars) / np.mean(chunk_vars) if np.mean(chunk_vars) > 0 else float('inf')
                            results[f"{feature}_var_stability"] = var_stability
                            self.logger.info(f"Feature {feature}: variance stability = {var_stability:.3f}")
            
            # 5. Data leakage detection
            for feature in ['avg_speed', 'total_volume']:
                if feature in vd_group:
                    data = vd_group[feature][:]
                    valid_data = data[~np.isnan(data)]
                    
                    if len(valid_data) > 50:
                        repeated_sequences = 0
                        for i in range(len(valid_data) - 10):
                            seq = valid_data[i:i+5]
                            for j in range(i+5, len(valid_data) - 5):
                                if np.allclose(seq, valid_data[j:j+5], rtol=1e-6):
                                    repeated_sequences += 1
                                    break
                        
                        repeat_rate = repeated_sequences / (len(valid_data) - 10) * 100
                        results[f"{feature}_repeat_rate"] = repeat_rate
                        self.logger.info(f"Feature {feature}: {repeat_rate:.2f}% repeated sequences")
        
        return results
    
    def analyze_data_splits(self, h5_path: str, vd_id: str, sequence_length: int = 12, train_ratio: float = 0.8) -> Dict[str, Any]:
        """Analyze train/validation split characteristics."""
        self.logger.info(f"Analyzing data splits for {h5_path}, VD: {vd_id}")
        
        results = {}
        
        with h5py.File(h5_path, 'r') as h5file:
            if vd_id not in h5file:
                return {'error': f'VD {vd_id} not found'}
            
            vd_group = h5file[vd_id]
            
            # Simulate train/val split logic
            total_samples = len(vd_group['avg_speed'][:])
            usable_samples = total_samples - sequence_length
            train_size = int(usable_samples * train_ratio)
            
            results['total_samples'] = total_samples
            results['usable_samples'] = usable_samples
            results['train_size'] = train_size
            results['val_size'] = usable_samples - train_size
            results['train_ratio'] = train_size / usable_samples
            
            self.logger.info(f"Split config - Total: {total_samples}, Train: {train_size}, Val: {usable_samples - train_size}")
            
            # Analyze distribution differences
            for feature in ['avg_speed', 'total_volume', 'avg_occupancy']:
                if feature in vd_group:
                    data = vd_group[feature][:]
                    valid_mask = ~np.isnan(data)
                    
                    train_data = data[valid_mask][:train_size]
                    val_data = data[valid_mask][train_size:]
                    
                    if len(train_data) > 0 and len(val_data) > 0:
                        train_mean = np.mean(train_data)
                        val_mean = np.mean(val_data)
                        train_std = np.std(train_data)
                        val_std = np.std(val_data)
                        
                        mean_diff = abs(train_mean - val_mean) / train_mean * 100 if train_mean != 0 else 0
                        std_diff = abs(train_std - val_std) / train_std * 100 if train_std != 0 else 0
                        
                        results[f"{feature}_train_mean"] = train_mean
                        results[f"{feature}_val_mean"] = val_mean
                        results[f"{feature}_train_std"] = train_std
                        results[f"{feature}_val_std"] = val_std
                        results[f"{feature}_mean_diff"] = mean_diff
                        results[f"{feature}_std_diff"] = std_diff
                        
                        self.logger.info(f"Feature {feature}: mean_diff={mean_diff:.1f}%, std_diff={std_diff:.1f}%")
                        
                        # Flag suspicious differences
                        if mean_diff > 20 or std_diff > 30:
                            self.logger.warning(f"Large distribution difference in {feature}")
        
        return results
    
    def analyze_model_complexity(self, experiments_dir: str = "blob/experiments/default") -> Dict[str, Any]:
        """Analyze model configuration for potential overfitting causes."""
        self.logger.info(f"Analyzing model complexity from {experiments_dir}")
        
        results = {}
        model_paths = [
            f"{experiments_dir}/xlstm/single_vd/training_history.json",
            f"{experiments_dir}/lstm/single_vd/training_history.json",
            f"{experiments_dir}/xlstm/multi_vd/training_history.json",
            f"{experiments_dir}/lstm/multi_vd/training_history.json"
        ]
        
        for model_path in model_paths:
            try:
                with open(model_path, 'r') as f:
                    data = json.load(f)
                
                model_name = f"{model_path.split('/')[-3]}_{model_path.split('/')[-2]}"
                model_results = {}
                
                # Extract model configuration
                training_config = data.get('training_config', {})
                
                # Calculate model parameters (rough estimate)
                if 'xlstm' in model_path:
                    hidden_size = 128
                    num_blocks = 6
                    input_size = 5
                    param_estimate = hidden_size * hidden_size * num_blocks * 4
                else:
                    hidden_size = training_config.get('hidden_size', 128)
                    num_layers = training_config.get('num_layers', 2)
                    input_size = 5
                    param_estimate = (input_size + hidden_size) * hidden_size * 4 * num_layers
                
                model_results['estimated_parameters'] = param_estimate
                model_results['hidden_size'] = hidden_size
                model_results['dropout'] = training_config.get('dropout', 'N/A')
                model_results['weight_decay'] = training_config.get('weight_decay', 'N/A')
                
                # Check training metrics for overfitting signs
                best_epoch = data.get('best_epoch', 0)
                total_epochs = len(data.get('epochs', []))
                
                if total_epochs > 0:
                    final_epoch = data['epochs'][-1]
                    train_loss = final_epoch.get('train_loss', 0)
                    val_loss = final_epoch.get('val_loss', 0)
                    
                    if val_loss > 0 and train_loss > 0:
                        overfitting_ratio = val_loss / train_loss
                        model_results['overfitting_ratio'] = overfitting_ratio
                        model_results['best_epoch'] = best_epoch
                        model_results['total_epochs'] = total_epochs
                        model_results['final_train_loss'] = train_loss
                        model_results['final_val_loss'] = val_loss
                        
                        self.logger.info(f"Model {model_name}: overfitting_ratio={overfitting_ratio:.2f}, best_epoch={best_epoch}/{total_epochs}")
                        
                        if overfitting_ratio > 10:
                            self.logger.warning(f"SEVERE OVERFITTING detected in {model_name}")
                        elif overfitting_ratio > 5:
                            self.logger.warning(f"Moderate overfitting detected in {model_name}")
                
                results[model_name] = model_results
                
            except Exception as e:
                self.logger.error(f"Error analyzing {model_path}: {e}")
                results[model_path] = {'error': str(e)}
        
        return results
    
    def create_diagnostic_plots(self, h5_path: str, vd_id: str, output_dir: str = "blob/debug") -> bool:
        """Create diagnostic plots for data analysis."""
        self.logger.info(f"Creating diagnostic plots for {h5_path}, VD: {vd_id}")
        
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            with h5py.File(h5_path, 'r') as h5file:
                if vd_id not in h5file:
                    self.logger.error(f"VD {vd_id} not found in H5 file")
                    return False
                
                vd_group = h5file[vd_id]
                
                # Plot 1: Data distribution histograms
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                features = ['avg_speed', 'total_volume', 'avg_occupancy', 'speed_std', 'lane_count']
                for i, feature in enumerate(features):
                    if feature in vd_group and i < len(axes):
                        data = vd_group[feature][:]
                        valid_data = data[~np.isnan(data)]
                        
                        if len(valid_data) > 0:
                            axes[i].hist(valid_data, bins=50, alpha=0.7, edgecolor='black')
                            axes[i].set_title(f'{feature} Distribution')
                            axes[i].set_xlabel(feature)
                            axes[i].set_ylabel('Frequency')
                            axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/data_distributions_{vd_id.replace('-', '_')}.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # Plot 2: Time series visualization
                fig, axes = plt.subplots(3, 1, figsize=(15, 12))
                
                timestamps = vd_group['timestamps'][:] if 'timestamps' in vd_group else np.arange(len(vd_group['avg_speed'][:]))
                
                for i, feature in enumerate(['avg_speed', 'total_volume', 'avg_occupancy']):
                    if feature in vd_group:
                        data = vd_group[feature][:]
                        
                        # Sample data for plotting (every 10th point if too many)
                        if len(data) > 1000:
                            indices = np.arange(0, len(data), len(data)//1000)
                            sample_data = data[indices]
                            sample_timestamps = timestamps[indices] if len(timestamps) > 0 else np.arange(len(sample_data))
                        else:
                            sample_data = data
                            sample_timestamps = timestamps if len(timestamps) > 0 else np.arange(len(sample_data))
                        
                        axes[i].plot(sample_timestamps, sample_data, alpha=0.7, linewidth=0.5)
                        axes[i].set_title(f'{feature} Time Series')
                        axes[i].set_ylabel(feature)
                        axes[i].grid(True, alpha=0.3)
                        
                        # Mark missing data
                        missing_mask = np.isnan(sample_data)
                        if np.any(missing_mask):
                            axes[i].scatter(sample_timestamps[missing_mask], 
                                          np.zeros(np.sum(missing_mask)), 
                                          c='red', s=1, alpha=0.5, label='Missing')
                            axes[i].legend()
                
                axes[-1].set_xlabel('Time')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/time_series_{vd_id.replace('-', '_')}.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Diagnostic plots saved to {output_dir}/")
                return True
                
        except Exception as e:
            self.logger.error(f"Error creating diagnostic plots: {e}")
            return False
    
    def comprehensive_diagnosis(self, h5_path: str, vd_id: str, config_path: str = None, 
                              output_dir: str = "blob/debug") -> Dict[str, Any]:
        """Perform comprehensive diagnosis combining all analysis methods."""
        self.logger.info(f"Starting comprehensive diagnosis for {h5_path}, VD: {vd_id}")
        
        diagnosis_results = {
            'h5_path': h5_path,
            'vd_id': vd_id,
            'timestamp': Path(__file__).stat().st_mtime
        }
        
        # 1. Dataset analysis
        dataset_results = self.analyze_h5_dataset(h5_path, vd_id)
        diagnosis_results['dataset_analysis'] = dataset_results
        
        # 2. Data splits analysis
        split_results = self.analyze_data_splits(h5_path, vd_id)
        diagnosis_results['split_analysis'] = split_results
        
        # 3. Model complexity analysis
        model_results = self.analyze_model_complexity()
        diagnosis_results['model_analysis'] = model_results
        
        # 4. Create diagnostic plots
        plots_success = self.create_diagnostic_plots(h5_path, vd_id, output_dir)
        diagnosis_results['plots_created'] = plots_success
        diagnosis_results['plots_directory'] = output_dir
        
        # 5. Generate summary and recommendations
        issues_found = []
        
        # Check data quality issues
        if 'error' not in dataset_results:
            for feature in ['avg_speed', 'total_volume', 'avg_occupancy']:
                valid_rate_key = f"{feature}_valid_rate"
                if valid_rate_key in dataset_results and dataset_results[valid_rate_key] < 70:
                    issues_found.append(f"Low data quality: {feature} only {dataset_results[valid_rate_key]:.1f}% valid")
                
                repeat_rate_key = f"{feature}_repeat_rate"
                if repeat_rate_key in dataset_results and dataset_results[repeat_rate_key] > 5:
                    issues_found.append(f"Suspicious data patterns: {feature} has {dataset_results[repeat_rate_key]:.1f}% repeated sequences")
        
        # Check train/val distribution differences
        if 'error' not in split_results:
            for feature in ['avg_speed', 'total_volume', 'avg_occupancy']:
                mean_diff_key = f"{feature}_mean_diff"
                if mean_diff_key in split_results and split_results[mean_diff_key] > 20:
                    issues_found.append(f"Train/val distribution mismatch: {feature} mean differs by {split_results[mean_diff_key]:.1f}%")
        
        diagnosis_results['issues_found'] = issues_found
        diagnosis_results['recommendations'] = [
            "Reduce model complexity: hidden size from 128 to 64",
            "Increase regularization: dropout from 0.2 to 0.4-0.5",
            "Use early stopping with patience=5",
            "Consider data augmentation or more training data",
            "Use time-based split instead of random split"
        ]
        
        self.logger.info(f"Comprehensive diagnosis complete. Found {len(issues_found)} issues.")
        
        return diagnosis_results