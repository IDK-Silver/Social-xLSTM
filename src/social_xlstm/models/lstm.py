"""
Unified Traffic LSTM Implementation

This module provides a unified, production-ready LSTM implementation for traffic prediction,
combining the best features from all existing implementations according to ADR-0002.

Design Principles:
- Clean architecture (based on pure/traffic_lstm.py)
- Multi-VD support (from traffic_lstm.py)
- Professional configuration (from pure_lstm.py)
- Extensible for future xLSTM integration

Author: Social-xLSTM Project Team
License: MIT
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrafficLSTMConfig:
    """
    Configuration class for Traffic LSTM model.
    
    This replaces hardcoded parameters and provides a clean configuration interface.
    """
    # Model Architecture
    input_size: int = 3  # [volume, speed, occupancy]
    hidden_size: int = 128
    num_layers: int = 2
    output_size: int = 3  # Same as input features
    sequence_length: int = 12  # Input sequence length
    prediction_length: int = 1  # Number of future timesteps to predict
    
    # Regularization
    dropout: float = 0.2
    
    # Input/Output Configuration
    batch_first: bool = True
    bidirectional: bool = False
    
    # Multi-VD Configuration
    multi_vd_mode: bool = False  # Single VD by default, can enable multi-VD
    num_vds: Optional[int] = None  # Required when multi_vd_mode=True
    
    # Training Configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.multi_vd_mode and self.num_vds is None:
            raise ValueError("num_vds must be specified when multi_vd_mode=True")
        
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")


class TrafficLSTM(nn.Module):
    """
    Unified Traffic LSTM Model
    
    This implementation combines the best features from all existing LSTM implementations:
    - Clean architecture from pure/traffic_lstm.py
    - Multi-VD support from traffic_lstm.py  
    - Professional configuration management
    - Extensible design for future xLSTM integration
    
    Supports both single VD and multi-VD modes:
    - Single VD: Input shape [batch_size, seq_len, num_features]
    - Multi VD: Input shape [B, T, N, F] - B=批次, T=時間步, N=VD數量, F=特徵
    """
    
    def __init__(self, config: TrafficLSTMConfig):
        super(TrafficLSTM, self).__init__()
        self.config = config
        
        # Core LSTM layer
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=config.batch_first,
            bidirectional=config.bidirectional
        )
        
        # Calculate LSTM output size (considering bidirectional)
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        
        # Output projection layers (enhanced from pure implementation)
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.output_size)
        )
        
        # Multi-VD processing layer (if enabled)
        if config.multi_vd_mode:
            self.vd_aggregation = nn.Linear(config.num_vds * config.output_size, config.output_size)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized TrafficLSTM with config: {config}")
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (LSTM best practice)
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
               - Single VD mode: [batch_size, seq_len, num_features]
               - Multi VD mode: [B, T, N, F] - B=批次, T=時間步, N=VD數量, F=特徵
            hidden: Optional initial hidden state
        
        Returns:
            Output tensor: [batch_size, prediction_length, output_size]
        """
        batch_size = x.size(0)
        
        if self.config.multi_vd_mode:
            # Handle multi-VD input - accept both 4D and 3D (pre-flattened) formats
            if x.dim() == 4:
                # 4D input: [B, T, N, F] - B=批次, T=時間步, N=VD數量, F=特徵 (needs flattening)
                seq_len, num_vds, num_features = x.size(1), x.size(2), x.size(3)
                x = x.view(batch_size, seq_len, num_vds * num_features)
                logger.debug(f"Flattened 4D input to 3D: {num_vds} VDs x {num_features} features")
                
            elif x.dim() == 3:
                # 3D input: [batch_size, seq_len, flattened_features] - already flattened
                seq_len, flattened_features = x.size(1), x.size(2)
                logger.debug(f"Using pre-flattened 3D input: {flattened_features} features")
                
            else:
                raise ValueError(f"Multi-VD mode expects 4D or 3D input, got {x.dim()}D")
            
            # Validate input size matches model expectations
            expected_size = self.lstm.input_size
            actual_size = x.size(-1)
            if expected_size != actual_size:
                logger.warning(f"Input size mismatch: expected {expected_size}, got {actual_size}")
        
        else:
            # Handle single VD input: [batch_size, seq_len, num_features]
            if x.dim() != 3:
                raise ValueError(f"Single-VD mode expects 3D input, got {x.dim()}D")
        
        # LSTM forward pass
        lstm_output, hidden = self.lstm(x, hidden)
        
        # Use last timestep output for prediction
        last_output = lstm_output[:, -1, :]  # [batch_size, hidden_size]
        
        # Apply output projection
        prediction = self.output_projection(last_output)  # [batch_size, output_size]
        
        # Handle multi-VD aggregation if needed
        if self.config.multi_vd_mode and hasattr(self, 'vd_aggregation'):
            # If prediction has VD dimension, aggregate across VDs
            if prediction.size(-1) == self.config.num_vds * self.config.output_size:
                prediction = self.vd_aggregation(prediction)
        
        # Expand to prediction_length if needed
        if self.config.prediction_length > 1:
            prediction = prediction.unsqueeze(1).repeat(1, self.config.prediction_length, 1)
        else:
            prediction = prediction.unsqueeze(1)  # [batch_size, 1, output_size]
        
        return prediction
    
    def predict_multi_step(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Multi-step prediction using autoregressive approach.
        
        This method is preserved from the pure implementation for advanced use cases.
        
        Args:
            x: Input sequence [batch_size, seq_len, features]
            steps: Number of future timesteps to predict
        
        Returns:
            Predictions [batch_size, steps, output_size]
        """
        self.eval()
        with torch.no_grad():
            predictions = []
            current_input = x.clone()
            
            for _ in range(steps):
                # Predict next timestep
                pred = self.forward(current_input)  # [batch_size, 1, output_size]
                predictions.append(pred)
                
                # Update input for next prediction (sliding window)
                next_input = torch.cat([current_input[:, 1:, :], pred], dim=1)
                current_input = next_input
            
            return torch.cat(predictions, dim=1)  # [batch_size, steps, output_size]
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model metadata and statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'TrafficLSTM',
            'config': self.config.__dict__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # Assuming float32
            'device': next(self.parameters()).device.type,
            'multi_vd_mode': self.config.multi_vd_mode,
            'architecture_summary': {
                'lstm_layers': self.config.num_layers,
                'hidden_size': self.config.hidden_size,
                'bidirectional': self.config.bidirectional,
                'dropout': self.config.dropout,
                'output_projection_layers': len(self.output_projection)
            }
        }
    
    def to_device(self, device: str):
        """Move model to specified device and update config."""
        self.to(device)
        self.config.device = device
        logger.info(f"Moved model to device: {device}")
    
    @classmethod
    def create_single_vd_model(cls, input_size: int = 3, hidden_size: int = 128, 
                              num_layers: int = 2, **kwargs) -> 'TrafficLSTM':
        """
        Factory method to create a single VD model with common defaults.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            **kwargs: Additional configuration parameters
        
        Returns:
            Configured TrafficLSTM model for single VD prediction
        """
        config = TrafficLSTMConfig(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            multi_vd_mode=False,
            **kwargs
        )
        return cls(config)
    
    @classmethod
    def create_multi_vd_model(cls, num_vds: int, input_size: int = 3, 
                             hidden_size: int = 128, num_layers: int = 2, **kwargs) -> 'TrafficLSTM':
        """
        Factory method to create a multi-VD model.
        
        Args:
            num_vds: Number of VDs to process simultaneously
            input_size: Number of input features per VD
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            **kwargs: Additional configuration parameters
        
        Returns:
            Configured TrafficLSTM model for multi-VD prediction
        """
        config = TrafficLSTMConfig(
            input_size=input_size * num_vds,  # Adjust for multi-VD
            output_size=input_size * num_vds,  # Output should match flattened target format
            hidden_size=hidden_size,
            num_layers=num_layers,
            multi_vd_mode=True,
            num_vds=num_vds,
            **kwargs
        )
        return cls(config)

    @staticmethod
    def parse_multi_vd_output(flat_output: torch.Tensor, 
                             num_vds: int, 
                             num_features: int) -> torch.Tensor:
        """
        Parse flattened multi-VD output back to structured format.
        
        This method converts the flattened output from multi-VD models back to
        a structured 4D tensor where each VD's predictions are separated.
        
        Args:
            flat_output: Flattened model output [batch_size, seq_len, num_vds * num_features]
            num_vds: Number of VDs in the output
            num_features: Number of features per VD
            
        Returns:
            torch.Tensor: Structured output [batch_size, seq_len, num_vds, num_features]
            
        Example:
            >>> flat_output = torch.randn(4, 1, 15)  # 3 VDs × 5 features
            >>> structured = TrafficLSTM.parse_multi_vd_output(flat_output, 3, 5)
            >>> print(structured.shape)  # torch.Size([4, 1, 3, 5])
        """
        batch_size, seq_len, total_features = flat_output.shape
        
        # Validate input dimensions
        expected_features = num_vds * num_features
        if total_features != expected_features:
            raise ValueError(
                f"Feature dimension mismatch: got {total_features}, "
                f"expected {expected_features} (num_vds={num_vds} × num_features={num_features})"
            )
        
        # Reshape to structured format
        structured = flat_output.view(batch_size, seq_len, num_vds, num_features)
        
        logger.debug(f"Parsed multi-VD output: {flat_output.shape} → {structured.shape}")
        return structured

    @staticmethod  
    def extract_vd_prediction(structured_output: torch.Tensor, 
                             vd_index: int) -> torch.Tensor:
        """
        Extract specific VD prediction from structured multi-VD output.
        
        Args:
            structured_output: Structured output [batch_size, seq_len, num_vds, num_features]
            vd_index: Index of the VD to extract (0-based)
            
        Returns:
            torch.Tensor: VD prediction [batch_size, seq_len, num_features]
            
        Example:
            >>> structured = torch.randn(4, 1, 3, 5)  # 3 VDs
            >>> vd_001 = TrafficLSTM.extract_vd_prediction(structured, 1)
            >>> print(vd_001.shape)  # torch.Size([4, 1, 5])
        """
        _, _, num_vds, _ = structured_output.shape
        
        # Validate VD index
        if vd_index < 0 or vd_index >= num_vds:
            raise IndexError(
                f"VD index {vd_index} out of range [0, {num_vds})"
            )
        
        # Extract specific VD
        vd_prediction = structured_output[:, :, vd_index, :]
        
        logger.debug(f"Extracted VD_{vd_index:03d} prediction: {vd_prediction.shape}")
        return vd_prediction


# Utility functions for model creation and management

def create_model_from_config(config_dict: Dict) -> TrafficLSTM:
    """
    Create model from configuration dictionary.
    
    Args:
        config_dict: Dictionary containing model configuration
    
    Returns:
        Configured TrafficLSTM model
    """
    config = TrafficLSTMConfig(**config_dict)
    return TrafficLSTM(config)


def load_model(checkpoint_path: str, config: Optional[TrafficLSTMConfig] = None) -> TrafficLSTM:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Optional configuration (will be loaded from checkpoint if not provided)
    
    Returns:
        Loaded TrafficLSTM model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if config is None:
        config = TrafficLSTMConfig(**checkpoint['config'])
    
    model = TrafficLSTM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def save_model(model: TrafficLSTM, checkpoint_path: str, 
               additional_info: Optional[Dict] = None):
    """
    Save model checkpoint.
    
    Args:
        model: TrafficLSTM model to save
        checkpoint_path: Path to save checkpoint
        additional_info: Optional additional information to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.config.__dict__,
        'model_info': model.get_model_info()
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved model to {checkpoint_path}")


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Single VD model
    print("Creating single VD model...")
    single_vd_model = TrafficLSTM.create_single_vd_model(
        input_size=3,
        hidden_size=64,
        num_layers=2
    )
    print(f"Single VD model info: {single_vd_model.get_model_info()}")
    
    # Test single VD forward pass
    batch_size, seq_len, features = 32, 12, 3
    x_single = torch.randn(batch_size, seq_len, features)
    output_single = single_vd_model(x_single)
    print(f"Single VD output shape: {output_single.shape}")
    
    # Example 2: Multi-VD model
    print("\nCreating multi VD model...")
    multi_vd_model = TrafficLSTM.create_multi_vd_model(
        num_vds=5,
        input_size=3,
        hidden_size=64,
        num_layers=2
    )
    print(f"Multi VD model info: {multi_vd_model.get_model_info()}")
    
    # Test multi VD forward pass
    num_vds = 5
    x_multi = torch.randn(batch_size, seq_len, num_vds, features)
    output_multi = multi_vd_model(x_multi)
    print(f"Multi VD output shape: {output_multi.shape}")
    
    # Example 3: Multi-step prediction
    print("\nTesting multi-step prediction...")
    predictions = single_vd_model.predict_multi_step(x_single, steps=5)
    print(f"Multi-step prediction shape: {predictions.shape}")