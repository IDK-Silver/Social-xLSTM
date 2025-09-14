"""
Distributed Social xLSTM Configuration System

This module provides strict configuration management for DistributedSocialXLSTMModel.
Designed for spatial-only social pooling with explicit parameter requirements.

Author: Social-xLSTM Team
"""

from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Dict, Any
from .xlstm import TrafficXLSTMConfig

# Valid pool types based on actual XLSTMSocialPoolingLayer implementation
# Verified from src/social_xlstm/pooling/xlstm_pooling.py line 185
ALLOWED_POOL_TYPES: Tuple[str, ...] = ("mean", "max", "weighted_mean")


@dataclass
class OptimizerConfig:
    """
    Configuration for optimizer.
    """
    name: str = "Adam"              # Optimizer type: Adam, AdamW, SGD
    lr: float = 0.001              # Learning rate
    weight_decay: float = 0.0      # Weight decay
    betas: Tuple[float, float] = (0.9, 0.999)  # Adam betas
    eps: float = 1e-08            # Adam epsilon
    momentum: float = 0.9         # SGD momentum


@dataclass
class SocialPoolingConfig:
    """
    Configuration for social pooling.

    Legacy (distance-based) and ST-Attention variants are supported.
    """
    enabled: bool                              # Whether to enable social pooling

    # Variant type: 'legacy' (distance-based), 'st_attention'
    type: str = 'legacy'

    # Legacy options (kept for backward compatibility)
    radius: float = 100.0                      # Spatial radius in meters
    pool_type: str = 'mean'                    # Aggregation: mean|max|weighted_mean

    # ST-Attention options
    knn_k: int = 16                            # kNN neighbors per VD
    time_window: int = 4                       # Timesteps for neighbor temporal repr
    heads: int = 4                             # Attention heads
    learnable_tau: bool = True                 # Learnable temperature for distance kernel
    tau_init: float = 1.0                      # Initial temperature
    dropout: float = 0.1                       # Attention dropout
    use_radius_mask: bool = False              # Optionally mask neighbors beyond radius
    


@dataclass  
class DistributedSocialXLSTMConfig:
    """
    Complete configuration for DistributedSocialXLSTMModel.
    
    Parameters are collected from different config sections:
    - xlstm: from models/xlstm.yaml
    - num_features: derived from xlstm.input_size
    - prediction_length: from datasets/*.yaml
    - learning_rate: from training/*.yaml  
    - social_pooling: from social_pooling/*.yaml
    - optimizer: from training/*.yaml or profile overrides
    """
    xlstm: TrafficXLSTMConfig
    num_features: int
    prediction_length: int
    learning_rate: float
    enable_gradient_checkpointing: bool
    social_pooling: SocialPoolingConfig
    optimizer: Optional[OptimizerConfig] = None
    


