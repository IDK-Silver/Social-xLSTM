"""Configuration classes for traffic dataset."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class TrafficDatasetConfig:
    """Configuration for traffic dataset."""
    hdf5_path: Path
    sequence_length: int = 60           # Input sequence length (minutes)
    prediction_length: int = 15         # Prediction sequence length (minutes)
    selected_vdids: Optional[List[str]] = None
    selected_features: Optional[List[str]] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    normalize: bool = True
    normalization_method: str = 'standard'  # 'standard', 'minmax'
    fill_missing: str = 'interpolate'   # 'zero', 'forward', 'interpolate'
    stride: int = 1                     # Sliding window stride
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        self.hdf5_path = Path(self.hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")
        
        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")


@dataclass
class TrafficHDF5Config:
    """Configuration for traffic data HDF5 conversion."""
    source_dir: Path
    output_path: Path
    selected_vdids: Optional[List[str]] = None
    feature_names: List[str] = field(default_factory=lambda: [
        'avg_speed', 'total_volume', 'avg_occupancy', 'speed_std', 'lane_count'
    ])
    time_range: Optional[Tuple[str, str]] = None
    compression: str = 'gzip'
    compression_opts: int = 6
    overwrite: bool = False  # Add overwrite flag
    check_consistency: bool = True  # Add consistency check flag
    
    def __post_init__(self):
        self.source_dir = Path(self.source_dir)
        self.output_path = Path(self.output_path)
        
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")