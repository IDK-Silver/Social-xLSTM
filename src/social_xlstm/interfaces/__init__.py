"""
Social-xLSTM Interface Definitions

This package contains the core interfaces and abstract base classes for the
distributed Social-xLSTM architecture.
"""

from .types import (
    AgentId, SceneId, TimeSteps, HiddenDim,
    HiddenStates, Positions, PooledTensor,
    SocialPoolingInterface
)
from .config import (
    DistanceConfig, SocialPoolingConfig, XLSTMConfig, ModelConfig
)
from .base_social_pooling import BaseSocialPooling
from .vd_manager import VDXLSTMManager
from .distributed_model import DistributedSocialXLSTMModel

__all__ = [
    "AgentId", "SceneId", "TimeSteps", "HiddenDim",
    "HiddenStates", "Positions", "PooledTensor",
    "SocialPoolingInterface",
    "DistanceConfig", "SocialPoolingConfig", "XLSTMConfig", "ModelConfig",
    "BaseSocialPooling", "VDXLSTMManager", "DistributedSocialXLSTMModel"
]