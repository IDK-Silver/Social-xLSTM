from __future__ import annotations
from typing import Dict, Iterable, Mapping
import torch
from torch import nn, Tensor
from collections import defaultdict

from .types import AgentId, HiddenStates
from .config import XLSTMConfig

class VDXLSTMManager:
    """
    Lazily instantiates and caches one xLSTM per Virtual Driver (VD).
    Handles:
        • hidden-state book-keeping
        • batch processing across heterogeneous numbers of agents
        • memory recycling of inactive VDs
    """

    def __init__(self, config: XLSTMConfig, *, device: str = "cpu"):
        self.cfg = config
        self.device = torch.device(device)
        self._vd_registry: Dict[AgentId, nn.Module] = {}
        self._last_used: Dict[AgentId, int] = defaultdict(int)
        self._step: int = 0

    # ------------------------------------------------------------
    def get(self, vd_id: AgentId) -> nn.Module:
        """
        Retrieve xLSTM instance for vd_id, creating on-demand.
        """
        if vd_id not in self._vd_registry:
            self._vd_registry[vd_id] = self._build_xlstm().to(self.device)
        self._last_used[vd_id] = self._step
        return self._vd_registry[vd_id]

    def step(self) -> None:
        """Call at every global simulation step to age instances."""
        self._step += 1

    def prune(self, max_idle_steps: int = 100) -> None:
        """Remove xLSTM modules that have been idle for >max_idle_steps."""
        to_drop = [
            k for k, last in self._last_used.items()
            if self._step - last > max_idle_steps
        ]
        for k in to_drop:
            del self._vd_registry[k]
            del self._last_used[k]

    # ------------------------------------------------------------
    def _build_xlstm(self) -> nn.Module:
        """Factory for a single xLSTM – decoupled for testing overrides."""
        return torch.nn.LSTM(
            input_size=self.cfg.hidden_size,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
            batch_first=True
        )