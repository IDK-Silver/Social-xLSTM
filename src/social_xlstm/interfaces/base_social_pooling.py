from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional
from dataclasses import dataclass
import torch
from torch import nn, Tensor

from .types import HiddenStates, Positions, PooledTensor

@dataclass
class SocialPoolingConfig:
    """Configuration for social pooling without pydantic dependency."""
    distance_name: str = "euclidean"
    distance_p: float = 2.0
    
    @property 
    def distance(self):
        """Mock distance config for compatibility."""
        return type('DistanceConfig', (), {
            'name': self.distance_name,
            'p': self.distance_p
        })()

class BaseSocialPooling(nn.Module, ABC):
    """
    Abstract base-class for all hidden-state social-pooling variants.

    Forward signature aligns with the training step:
        agents    : Hidden states for *central* agents in this mini-batch
        neighbors : Hidden states for *neighboring* agents
        scene     : Optional scene context (map, semantic layers …)
    """

    def __init__(self, output_size: int, *, config: SocialPoolingConfig):
        super().__init__()
        self.output_size = output_size
        self.config      = config

    # -------- helpers that can be overridden if necessary --------
    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Batched distance between two point sets p and q (…, 2).
        Can be overridden to implement custom metrics.
        """
        if self.config.distance.name == "manhattan":
            return torch.cdist(p, q, p=1)
        elif self.config.distance.name == "cosine":
            # Cosine distance = 1 - cosine similarity
            sim = torch.nn.functional.cosine_similarity(
                p.unsqueeze(-2), q.unsqueeze(-3), dim=-1
            )
            return 1 - sim
        # default: Euclidean (p=2)
        return torch.cdist(p, q, p=self.config.distance.p)

    # ---------------- main contract ----------------
    def forward(
        self,
        *,
        agents:    HiddenStates,
        neighbors: HiddenStates,
        positions: Positions,
        scene: Optional[Any] = None,
    ) -> PooledTensor:
        # lightweight runtime validation
        self._check_shapes(agents, neighbors, positions)
        return self._pool(agents=agents, neighbors=neighbors,
                          positions=positions, scene=scene)

    @abstractmethod
    def _pool(
        self,
        *,
        agents:    HiddenStates,
        neighbors: HiddenStates,
        positions: Positions,
        scene: Optional[Any] = None,
    ) -> PooledTensor:
        """Sub-classes implement the actual pooling algorithm."""
        ...

    # ------------------------------------------------------------
    def _check_shapes(
        self,
        agents: HiddenStates,
        neighbors: HiddenStates,
        positions: Positions
    ) -> None:
        # basic shape sanity – can be extended
        if not agents:
            raise ValueError("`agents` mapping cannot be empty.")
        t, h = next(iter(agents.values())).shape
        if positions.shape[1] != t:
            raise ValueError("positions.shape[1] must equal T of hidden state.")