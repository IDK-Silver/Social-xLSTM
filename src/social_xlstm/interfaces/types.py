from __future__ import annotations
from typing import (
    Any, Dict, Iterable, Mapping, MutableMapping, Optional,
    Protocol, Sequence, Tuple, TypeVar, runtime_checkable
)
import torch
from torch import Tensor

# ---------- high-level aliases ----------
AgentId     = str                       # canonical identifier inside one batch
SceneId     = str                       # identifier of the physical scene
TimeSteps   = int
HiddenDim   = int

# (N = num agents, T = timesteps, H = hidden size, H' = pooled size)
HiddenStates = Mapping[AgentId, Tensor]   # shape: [T, H]  – *per agent*
Positions    = Tensor                     # shape: [N, T, 2]
PooledTensor = Tensor                     # shape: [N, T, H']

@runtime_checkable
class SocialPoolingInterface(Protocol):
    """
    Stateless callable used by the training step.
    The concrete pooling strategy can be swapped out at runtime.
    """
    def __call__(
        self, *,
        hidden_states: HiddenStates,
        positions: Positions,
        radius: float = 2.0,
        **kwargs: Any
    ) -> PooledTensor: ...