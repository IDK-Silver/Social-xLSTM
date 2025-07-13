"""
Traffic xLSTM Implementation

This module provides a production-ready xLSTM implementation for traffic prediction,
implementing extended LSTM with sLSTM and mLSTM blocks according to ADR-0501.

🎯 主要功能:
- 混合 sLSTM + mLSTM 架構用於交通預測
- 6 個 xLSTM 區塊，sLSTM 在位置 [1, 3]
- 654,883 個參數，支援 GPU 加速
- 完整的配置管理和錯誤處理

📊 預期改善:
- 解決傳統 LSTM 的過擬合問題（負 R² 值）
- 更好的長期時間依賴建模
- 為 Social Pooling 整合做準備

🚀 快速使用:
```python
from social_xlstm.models import TrafficXLSTM, TrafficXLSTMConfig

config = TrafficXLSTMConfig()
model = TrafficXLSTM(config)

# 輸入: (batch, seq_len, features) = (4, 12, 3)
x = torch.randn(4, 12, 3)
output = model(x)  # 輸出: (4, 1, 3)
```

📚 完整文檔: docs/guides/xlstm_usage_guide.md

Design Principles:
- Independent from TrafficLSTM (ADR-0501)
- Clean configuration with dataclass
- Integration with xlstm library
- Support for hybrid sLSTM/mLSTM architecture

Author: Social-xLSTM Project Team
License: MIT
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import logging

# Import xlstm library components
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig

logger = logging.getLogger(__name__)


@dataclass
class TrafficXLSTMConfig:
    """
    Configuration class for Traffic xLSTM model.
    
    Based on ADR-0501 decisions for xLSTM integration strategy.
    """
    # Model Architecture
    input_size: int = 3  # [volume, speed, occupancy]
    embedding_dim: int = 128
    hidden_size: int = 128
    num_blocks: int = 6
    output_size: int = 3  # Same as input features
    sequence_length: int = 12  # Input sequence length
    prediction_length: int = 1  # Number of future timesteps to predict
    
    # xLSTM Block Configuration (ADR-0501)
    slstm_at: List[int] = field(default_factory=lambda: [1, 3])  # sLSTM at positions [1, 3]
    slstm_backend: str = "vanilla"  # Use vanilla backend for stability
    mlstm_backend: str = "vanilla"  # Use vanilla backend for stability
    
    # Context and Memory
    context_length: int = 256
    
    # Regularization
    dropout: float = 0.1
    
    # Input/Output Configuration
    batch_first: bool = True
    
    # Multi-VD Configuration
    multi_vd_mode: bool = False  # Single VD by default
    num_vds: Optional[int] = None  # Required when multi_vd_mode=True
    
    # Training Configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.multi_vd_mode and self.num_vds is None:
            raise ValueError("num_vds must be specified when multi_vd_mode=True")
        
        # Validate sLSTM positions
        if any(pos >= self.num_blocks for pos in self.slstm_at):
            raise ValueError(f"sLSTM positions {self.slstm_at} exceed num_blocks={self.num_blocks}")
        
        logger.info(f"TrafficXLSTM Config: {self.num_blocks} blocks, "
                   f"sLSTM at {self.slstm_at}, "
                   f"embedding_dim={self.embedding_dim}")


class TrafficXLSTM(nn.Module):
    """
    Traffic prediction model using extended LSTM (xLSTM) architecture.
    
    This implementation follows ADR-0501 decisions:
    - Independent from TrafficLSTM 
    - Uses xlstm library's xLSTMBlockStack
    - Supports hybrid sLSTM/mLSTM configuration
    - Designed for traffic time series prediction
    """
    
    def __init__(self, config: TrafficXLSTMConfig):
        super(TrafficXLSTM, self).__init__()
        self.config = config
        
        # Input embedding layer
        self.input_embedding = nn.Linear(config.input_size, config.embedding_dim)
        
        # xLSTM Block Stack Configuration
        mlstm_layer_config = mLSTMLayerConfig(
            embedding_dim=config.embedding_dim,
            context_length=config.context_length,
        )
        mlstm_config = mLSTMBlockConfig(mlstm=mlstm_layer_config)
        
        slstm_layer_config = sLSTMLayerConfig(
            backend=config.slstm_backend,
            embedding_dim=config.embedding_dim,
        )
        slstm_config = sLSTMBlockConfig(slstm=slstm_layer_config)
        
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mlstm_config,
            slstm_block=slstm_config,
            context_length=config.context_length,
            num_blocks=config.num_blocks,
            embedding_dim=config.embedding_dim,
            slstm_at=config.slstm_at,
        )
        
        # Initialize xLSTM Block Stack
        self.xlstm_stack = xLSTMBlockStack(xlstm_config)
        
        # Output projection layer
        self.output_projection = nn.Linear(config.embedding_dim, config.output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        logger.info(f"TrafficXLSTM initialized with {config.num_blocks} blocks")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the xLSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, prediction_length, output_size)
        """
        # Input validation and Multi-VD handling
        batch_size = x.size(0)
        
        if self.config.multi_vd_mode:
            # Handle multi-VD input - accept both 4D and 3D (pre-flattened) formats
            if x.dim() == 4:
                # 4D input: [batch_size, seq_len, num_vds, num_features] - needs flattening
                seq_len, num_vds, num_features = x.size(1), x.size(2), x.size(3)
                x = x.view(batch_size, seq_len, num_vds * num_features)
                logging.getLogger(__name__).debug(f"Flattened 4D input to 3D: {num_vds} VDs x {num_features} features")
                
            elif x.dim() == 3:
                # 3D input: [batch_size, seq_len, flattened_features] - already flattened
                seq_len, flattened_features = x.size(1), x.size(2)
                logging.getLogger(__name__).debug(f"Using pre-flattened 3D input: {flattened_features} features")
                
            else:
                raise ValueError(f"Multi-VD mode expects 4D or 3D input, got {x.dim()}D")
        else:
            # Single VD mode - expect 3D input
            if x.dim() != 3:
                raise ValueError(f"Single VD mode expects 3D input (batch, seq, features), got {x.dim()}D")
        
        batch_size, seq_len, input_size = x.shape
        
        if input_size != self.config.input_size:
            raise ValueError(f"Expected input_size={self.config.input_size}, got {input_size}")
        
        # Input embedding
        embedded = self.input_embedding(x)  # (batch, seq, embedding_dim)
        embedded = self.dropout(embedded)
        
        # xLSTM processing
        xlstm_output = self.xlstm_stack(embedded)  # (batch, seq, embedding_dim)
        
        # Take the last timestep for prediction
        last_hidden = xlstm_output[:, -1, :]  # (batch, embedding_dim)
        
        # Output projection
        output = self.output_projection(last_hidden)  # (batch, output_size)
        
        # Reshape to match expected output format
        output = output.unsqueeze(1)  # (batch, 1, output_size)
        
        return output
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "TrafficXLSTM",
            "num_blocks": self.config.num_blocks,
            "embedding_dim": self.config.embedding_dim,
            "slstm_positions": self.config.slstm_at,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "multi_vd_mode": self.config.multi_vd_mode,
            "device": self.config.device
        }
    
    def to_device(self, device: Optional[str] = None):
        """Move model to specified device."""
        target_device = device or self.config.device
        self.to(target_device)
        self.config.device = target_device
        logger.info(f"TrafficXLSTM moved to device: {target_device}")
        return self