"""
Distributed Social xLSTM Configuration System

This module provides strict configuration management for DistributedSocialXLSTMModel.
Designed for spatial-only social pooling with explicit parameter requirements.

Author: Social-xLSTM Team
"""

from dataclasses import dataclass
from typing import Literal, Tuple
from .xlstm import TrafficXLSTMConfig

# Valid pool types based on actual XLSTMSocialPoolingLayer implementation
# Verified from src/social_xlstm/pooling/xlstm_pooling.py line 185
ALLOWED_POOL_TYPES: Tuple[str, ...] = ("mean", "max", "weighted_mean")


@dataclass
class SocialPoolingConfig:
    """
    Configuration for spatial-only social pooling.
    
    Only contains social pooling specific parameters.
    Other parameters are derived from xlstm/dataset/training configs.
    """
    enabled: bool                    # Whether to enable social pooling
    radius: float                   # Spatial radius in meters
    pool_type: str                  # Aggregation method: mean, max, weighted_mean
    


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
    """
    xlstm: TrafficXLSTMConfig
    num_features: int
    prediction_length: int
    learning_rate: float
    enable_gradient_checkpointing: bool
    social_pooling: SocialPoolingConfig
    



