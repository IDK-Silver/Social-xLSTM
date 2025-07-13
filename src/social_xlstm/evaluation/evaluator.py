import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any

class ModelEvaluator:
    """封裝模型評估的所有邏輯，專注於數據處理和指標計算"""
    
    def __init__(self, model, train_losses, val_losses, config, train_loader, val_loader, device, vd_index=0):
        self.model = model
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader  # 新增驗證集 loader
        self.device = device
        self.vd_index = vd_index
        
        # 延遲計算的屬性
        self._train_predictions = None
        self._train_targets = None
        self._train_inputs = None
        self._val_predictions = None
        self._val_targets = None
        self._val_inputs = None
        self._train_metrics = None
        self._val_metrics = None
    
    def get_train_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """獲取訓練集預測結果"""
        if self._train_predictions is None:
            self._compute_train_predictions()
        return self._train_predictions, self._train_targets, self._train_inputs
    
    def get_val_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """獲取驗證集預測結果"""
        if self._val_predictions is None and self.val_loader is not None:
            self._compute_val_predictions()
        return self._val_predictions, self._val_targets, self._val_inputs
    
    def get_train_metrics(self) -> Dict[str, float]:
        """獲取訓練集評估指標"""
        if self._train_metrics is None:
            predictions, targets, _ = self.get_train_predictions()
            self._train_metrics = self._compute_metrics(predictions, targets)
        return self._train_metrics
    
    def get_val_metrics(self) -> Dict[str, float]:
        """獲取驗證集評估指標"""
        if self._val_metrics is None and self.val_loader is not None:
            predictions, targets, _ = self.get_val_predictions()
            self._val_metrics = self._compute_metrics(predictions, targets)
        return self._val_metrics
    
    def _compute_train_predictions(self):
        """計算訓練集預測結果"""
        self._train_predictions, self._train_targets, self._train_inputs = self._compute_predictions_for_loader(self.train_loader)
    
    def _compute_val_predictions(self):
        """計算驗證集預測結果"""
        if self.val_loader is not None:
            self._val_predictions, self._val_targets, self._val_inputs = self._compute_predictions_for_loader(self.val_loader)
    
    def _compute_predictions_for_loader(self, data_loader):
        """通用的預測計算函數"""
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
        """計算評估指標"""
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
        """獲取所有評估相關的數據，包含訓練和驗證"""
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
        
        # 如果有驗證集，加入驗證數據
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