"""Configuration classes for traffic dataset."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class TrafficDatasetConfig:
    """Configuration for traffic dataset.
    
    Batch format semantics:
    - 'centralized': Standard format [B, T, N, F] tensor batches
    - 'distributed': Per-VD format {"VD_001": [B, T, F], ...} dictionary batches
    
    Note: batch_size represents per-process batch size in both modes.
    Effective total batch size = batch_size * world_size in distributed training.
    """
    hdf5_path: Path
    sequence_length: int                         # Input sequence length (minutes)
    prediction_length: int                       # Prediction sequence length (minutes) 
    train_ratio: float                          # Training data ratio
    val_ratio: float                            # Validation data ratio
    test_ratio: float                           # Test data ratio
    normalize: bool                             # Whether to normalize data
    normalization_method: str                  # Normalization method
    fill_missing: str                          # Missing value handling
    stride: int                                # Sliding window stride
    batch_size: int                            # DataLoader batch size
    num_workers: int                           # DataLoader workers
    pin_memory: bool                           # DataLoader pin memory
    selected_vdids: Optional[List[str]] = None   # VD IDs to use (None = use all)
    selected_features: Optional[List[str]] = None  # Features to use (None = use all)
    batch_format: str = 'centralized'          # Batch format: 'centralized' | 'distributed'
    
    # Allowed batch formats
    ALLOWED_BATCH_FORMATS = ('centralized', 'distributed')
    
    def __post_init__(self):
        self.hdf5_path = Path(self.hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")
        
        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")
        
        # Validate and normalize batch_format
        self.batch_format = str(self.batch_format).strip().lower()
        if self.batch_format not in self.ALLOWED_BATCH_FORMATS:
            raise ValueError(f"batch_format must be one of {self.ALLOWED_BATCH_FORMATS}, got {self.batch_format}")
    
    @property
    def is_distributed(self) -> bool:
        """True if batch format is distributed."""
        return self.batch_format == 'distributed'


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
    
    # Performance options
    show_progress: bool = True  # Show progress bar during conversion
    verbose_warnings: bool = False  # Show detailed warnings (can be noisy)
    max_missing_report: int = 10  # Maximum number of missing VDIDs to report per timestep
    
    def __post_init__(self):
        self.source_dir = Path(self.source_dir)
        self.output_path = Path(self.output_path)
        
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")