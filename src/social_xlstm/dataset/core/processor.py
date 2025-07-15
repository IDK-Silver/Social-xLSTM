"""Data preprocessing utilities for traffic data."""

import numpy as np
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class TrafficDataProcessor:
    """Data preprocessing utilities for traffic data."""
    
    @staticmethod
    def normalize_features(data: np.ndarray, method: str = 'standard', 
                          scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None,
                          fit_scaler: bool = True) -> Tuple[np.ndarray, Union[StandardScaler, MinMaxScaler]]:
        """Normalize features across time and space dimensions."""
        original_shape = data.shape  # [T, N, F]
        
        # Reshape to [T*N, F] for fitting scaler
        data_reshaped = data.reshape(-1, data.shape[-1])
        
        # Remove NaN values for fitting
        valid_mask = ~np.isnan(data_reshaped).any(axis=1)
        valid_data = data_reshaped[valid_mask]
        
        if scaler is None:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        
        if fit_scaler and len(valid_data) > 0:
            scaler.fit(valid_data)
        elif fit_scaler and len(valid_data) == 0:
            # If no valid data for fitting, create a dummy scaler
            print("Warning: No valid data for fitting scaler. Using dummy scaler.")
            dummy_data = np.zeros((1, data.shape[-1]))
            scaler.fit(dummy_data)
        
        # Transform all data (including NaN)
        normalized_data = np.full_like(data_reshaped, np.nan)
        if len(valid_data) > 0:
            normalized_data[valid_mask] = scaler.transform(valid_data)
        
        # Reshape back to original shape
        normalized_data = normalized_data.reshape(original_shape)
        
        return normalized_data, scaler
    
    @staticmethod
    def handle_missing_values(data: np.ndarray, method: str = 'interpolate') -> np.ndarray:
        """Handle missing values in time series data."""
        if method == 'zero':
            return np.nan_to_num(data, nan=0.0)
        
        elif method == 'forward':
            # Forward fill along time axis
            result = data.copy()
            for vd_idx in range(data.shape[1]):
                for feat_idx in range(data.shape[2]):
                    series = result[:, vd_idx, feat_idx]
                    mask = ~np.isnan(series)
                    if mask.any():
                        # Forward fill
                        valid_indices = np.where(mask)[0]
                        for i in range(len(series)):
                            if np.isnan(series[i]) and len(valid_indices) > 0:
                                # Find the last valid value
                                prev_valid = valid_indices[valid_indices < i]
                                if len(prev_valid) > 0:
                                    series[i] = series[prev_valid[-1]]
            return result
        
        elif method == 'interpolate':
            # Linear interpolation along time axis
            result = data.copy()
            for vd_idx in range(data.shape[1]):
                for feat_idx in range(data.shape[2]):
                    series = result[:, vd_idx, feat_idx]
                    valid_mask = ~np.isnan(series)
                    if valid_mask.sum() > 1:
                        valid_indices = np.where(valid_mask)[0]
                        valid_values = series[valid_mask]
                        # Interpolate
                        interpolated = np.interp(
                            np.arange(len(series)), 
                            valid_indices, 
                            valid_values
                        )
                        result[:, vd_idx, feat_idx] = interpolated
            return result
        
        else:
            raise ValueError(f"Unknown missing value handling method: {method}")
    
    @staticmethod
    def create_time_features(timestamps: List[str]) -> np.ndarray:
        """Create time-based features from timestamps."""
        time_features = []
        
        for ts_str in timestamps:
            # Handle empty or invalid timestamps
            if not ts_str or ts_str.strip() == '':
                # Use NaN features for empty timestamps
                time_features.append([np.nan] * 9)  # 9 features total
                continue
                
            try:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except:
                try:
                    # Fallback parsing
                    dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
                except:
                    # If all parsing fails, use NaN features
                    time_features.append([np.nan] * 9)
                    continue
            
            # Extract time features
            hour = dt.hour / 23.0  # Normalize to [0, 1]
            minute = dt.minute / 59.0
            day_of_week = dt.weekday() / 6.0
            day_of_month = (dt.day - 1) / 30.0
            month = (dt.month - 1) / 11.0
            
            # Cyclical encoding for hour and day of week
            hour_sin = np.sin(2 * np.pi * hour)
            hour_cos = np.cos(2 * np.pi * hour)
            dow_sin = np.sin(2 * np.pi * day_of_week)
            dow_cos = np.cos(2 * np.pi * day_of_week)
            
            time_features.append([
                hour, minute, day_of_week, day_of_month, month,
                hour_sin, hour_cos, dow_sin, dow_cos
            ])
        
        return np.array(time_features, dtype=np.float32)


class TrafficConfigGenerator:
    """Configuration generator for optimized traffic model training."""
    
    @staticmethod
    def create_optimized_configs(stable_h5_path: str, output_dir: str = "cfgs/fixed", 
                               vd_ids: Optional[List[str]] = None) -> List[str]:
        """
        Create optimized model configurations for stable dataset.
        
        Args:
            stable_h5_path: Path to the stabilized H5 dataset
            output_dir: Directory to save configuration files
            vd_ids: List of VD IDs to use (uses defaults if None)
            
        Returns:
            List of paths to saved configuration files
        """
        logger.info(f"Creating optimized configs for {stable_h5_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Default VD IDs if not provided
        if vd_ids is None:
            vd_ids = ['VD-28-0740-000-001', 'VD-11-0020-008-001', 'VD-13-0660-000-002']
        
        # Base configuration template for stable data + overfitting fixes
        base_config = {
            'h5_file_path': stable_h5_path,
            'select_vd_id': vd_ids[0],  # Primary VD with good quality
            'selected_vdids': vd_ids,
            
            # Dataset config optimized for stable data
            'dataset': {
                'sequence_length': 12,
                'prediction_length': 1,
                'train_ratio': 0.8,
                'val_ratio': 0.2,
                'normalize': True,
                'normalization_method': 'standard'
            },
            
            # Training config with strong regularization
            'training': {
                'epochs': 50,           # Reduced from 200
                'batch_size': 16,       # Reduced for small dataset
                'learning_rate': 0.0005, # Reduced learning rate
                'weight_decay': 0.01,   # Increased regularization
                'early_stopping_patience': 8,  # Earlier stopping
                'gradient_clip_value': 0.5,    # Stronger gradient clipping
                'use_scheduler': True,
                'scheduler_patience': 5
            }
        }
        
        # LSTM config (simplified)
        lstm_config = base_config.copy()
        lstm_config.update({
            'model_type': 'lstm',
            'model': {
                'hidden_size': 32,      # Dramatically reduced from 128
                'num_layers': 1,        # Reduced from 2
                'dropout': 0.5,         # Increased from 0.2
                'bidirectional': False
            }
        })
        
        # xLSTM config (simplified)
        xlstm_config = base_config.copy()
        xlstm_config.update({
            'model_type': 'xlstm',
            'model': {
                'embedding_dim': 32,    # Reduced from 128
                'hidden_size': 32,      # Reduced from 128
                'num_blocks': 3,        # Reduced from 6
                'dropout': 0.5,         # Increased regularization
                'slstm_at': [1],        # Simplified structure
            }
        })
        
        # Save configurations
        config_files = {
            'lstm_fixed.yaml': lstm_config,
            'xlstm_fixed.yaml': xlstm_config
        }
        
        saved_configs = []
        for filename, config in config_files.items():
            config_path = output_path / filename
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            saved_configs.append(str(config_path))
            logger.info(f"Saved configuration: {config_path}")
        
        logger.info(f"Created {len(saved_configs)} optimized configurations")
        logger.info("Key optimizations applied:")
        logger.info("  - Dataset: Using stable data (70% of original)")
        logger.info("  - Model size: Reduced hidden_size 128→32")
        logger.info("  - Regularization: Increased dropout 0.2→0.5")
        logger.info("  - Training: Smaller batches, lower LR, early stopping")
        logger.info("  - Complexity: Fewer layers/blocks")
        
        return saved_configs
    
    @staticmethod
    def create_development_configs(h5_path: str, output_dir: str = "cfgs/dev_fixed",
                                 vd_ids: Optional[List[str]] = None) -> List[str]:
        """
        Create development configurations for quick testing.
        
        Args:
            h5_path: Path to the H5 dataset
            output_dir: Directory to save configuration files
            vd_ids: List of VD IDs to use (uses defaults if None)
            
        Returns:
            List of paths to saved configuration files
        """
        logger.info(f"Creating development configs for {h5_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Default VD IDs if not provided
        if vd_ids is None:
            vd_ids = ['VD-28-0740-000-001']
        
        # Base development configuration
        base_config = {
            'h5_file_path': h5_path,
            'select_vd_id': vd_ids[0],
            'selected_vdids': vd_ids,
            
            # Dataset config for fast development
            'dataset': {
                'sequence_length': 6,   # Shorter sequences
                'prediction_length': 1,
                'train_ratio': 0.8,
                'val_ratio': 0.2,
                'normalize': True,
                'normalization_method': 'standard'
            },
            
            # Training config for quick iteration
            'training': {
                'epochs': 5,            # Very few epochs
                'batch_size': 32,       # Larger batches for speed
                'learning_rate': 0.001, # Standard learning rate
                'weight_decay': 0.01,
                'early_stopping_patience': 3,
                'gradient_clip_value': 1.0,
                'use_scheduler': False
            }
        }
        
        # LSTM dev config (minimal)
        lstm_config = base_config.copy()
        lstm_config.update({
            'model_type': 'lstm',
            'model': {
                'hidden_size': 16,      # Very small for development
                'num_layers': 1,
                'dropout': 0.3,
                'bidirectional': False
            }
        })
        
        # xLSTM dev config (minimal)
        xlstm_config = base_config.copy()
        xlstm_config.update({
            'model_type': 'xlstm',
            'model': {
                'embedding_dim': 16,    # Very small for development
                'hidden_size': 16,
                'num_blocks': 2,        # Minimal blocks
                'dropout': 0.3,
                'slstm_at': [1],
            }
        })
        
        # Save configurations
        config_files = {
            'lstm_dev.yaml': lstm_config,
            'xlstm_dev.yaml': xlstm_config
        }
        
        saved_configs = []
        for filename, config in config_files.items():
            config_path = output_path / filename
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            saved_configs.append(str(config_path))
            logger.info(f"Saved development configuration: {config_path}")
        
        logger.info(f"Created {len(saved_configs)} development configurations")
        
        return saved_configs
    
    @staticmethod
    def create_production_configs(h5_path: str, output_dir: str = "cfgs/production",
                                vd_ids: Optional[List[str]] = None) -> List[str]:
        """
        Create production configurations for final experiments.
        
        Args:
            h5_path: Path to the H5 dataset
            output_dir: Directory to save configuration files
            vd_ids: List of VD IDs to use (uses defaults if None)
            
        Returns:
            List of paths to saved configuration files
        """
        logger.info(f"Creating production configs for {h5_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Default VD IDs if not provided
        if vd_ids is None:
            vd_ids = ['VD-28-0740-000-001', 'VD-11-0020-008-001', 'VD-13-0660-000-002']
        
        # Base production configuration
        base_config = {
            'h5_file_path': h5_path,
            'select_vd_id': vd_ids[0],
            'selected_vdids': vd_ids,
            
            # Dataset config for production
            'dataset': {
                'sequence_length': 12,
                'prediction_length': 1,
                'train_ratio': 0.8,
                'val_ratio': 0.2,
                'normalize': True,
                'normalization_method': 'standard'
            },
            
            # Training config for production
            'training': {
                'epochs': 100,          # Full training
                'batch_size': 64,       # Optimal batch size
                'learning_rate': 0.001, # Standard learning rate
                'weight_decay': 0.001,  # Moderate regularization
                'early_stopping_patience': 10,
                'gradient_clip_value': 1.0,
                'use_scheduler': True,
                'scheduler_patience': 7
            }
        }
        
        # LSTM production config
        lstm_config = base_config.copy()
        lstm_config.update({
            'model_type': 'lstm',
            'model': {
                'hidden_size': 128,     # Full model size
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': False
            }
        })
        
        # xLSTM production config
        xlstm_config = base_config.copy()
        xlstm_config.update({
            'model_type': 'xlstm',
            'model': {
                'embedding_dim': 128,   # Full model size
                'hidden_size': 128,
                'num_blocks': 6,        # Full blocks
                'dropout': 0.2,
                'slstm_at': [1, 3, 5],  # Mixed architecture
            }
        })
        
        # Save configurations
        config_files = {
            'lstm_production.yaml': lstm_config,
            'xlstm_production.yaml': xlstm_config
        }
        
        saved_configs = []
        for filename, config in config_files.items():
            config_path = output_path / filename
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            saved_configs.append(str(config_path))
            logger.info(f"Saved production configuration: {config_path}")
        
        logger.info(f"Created {len(saved_configs)} production configurations")
        
        return saved_configs