from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, Type, Union
from dataclasses import dataclass, field
import yaml
import torch

@dataclass
class DistanceConfig:
    """Configuration for the distance function used inside pooling."""
    name: Literal["euclidean", "manhattan", "cosine"] = "euclidean"
    p: int = 2

@dataclass
class SocialPoolingConfig:
    strategy: Literal["grid", "knn", "attention"] = "grid"
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    radius: float = 2.0

@dataclass
class XLSTMConfig:
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.0

@dataclass
class ModelConfig:
    pooling: SocialPoolingConfig = field(default_factory=SocialPoolingConfig)
    xlstm: XLSTMConfig = field(default_factory=XLSTMConfig)
    device: str = "cpu"

    def __post_init__(self):
        """Validate device after initialization."""
        if self.device != "cpu" and not torch.cuda.is_available():
            raise ValueError(f"CUDA not available – requested device='{self.device}'")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfig":
        """Load config from YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        # Convert dataclass to dict for serialization
        import dataclasses
        data = dataclasses.asdict(self)
        Path(path).write_text(yaml.safe_dump(data))