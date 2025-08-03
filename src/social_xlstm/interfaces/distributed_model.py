from __future__ import annotations
from typing import Mapping, Any
import torch
from torch import nn, Tensor

from .config import ModelConfig
from .vd_manager import VDXLSTMManager
from .base_social_pooling import BaseSocialPooling
from .types import HiddenStates, Positions, PooledTensor

class DistributedSocialXLSTMModel(nn.Module):
    """
    High-level façade that wires:
        VDXLSTMManager  +  Social-Pooling  +  downstream head(s)
    """

    def __init__(
        self,
        pooling: BaseSocialPooling,
        *,
        config: ModelConfig
    ):
        super().__init__()
        self.cfg = config
        self.pooling = pooling
        self.manager = VDXLSTMManager(config.xlstm, device=config.device)

        # Example prediction head – purely illustrative
        self.decoder = nn.Linear(
            in_features=pooling.output_size,
            out_features=config.xlstm.hidden_size
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        *,
        hidden_states: HiddenStates,     # vd-id → [T, H]
        positions: Positions             # [N, T, 2]
    ) -> Tensor:
        """
        1. Perform social pooling
        2. Run pooled features through a decoding head
        """
        pooled: PooledTensor = self.pooling(
            agents=hidden_states,
            neighbors=hidden_states,      # simplified for illustration
            positions=positions,
            scene=None
        )
        return self.decoder(pooled)       # [N, T, H]