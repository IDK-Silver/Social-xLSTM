# Configuration Reference

Complete reference for all configuration options and parameters in Social-xLSTM.

## Dataset Configuration

### TrafficDatasetConfig
```python
@dataclass
class TrafficDatasetConfig:
    # Time series parameters
    sequence_length: int = 12           # Input sequence length
    prediction_length: int = 1          # Prediction horizon
    
    # Data splitting
    train_ratio: float = 0.7           # Training split ratio
    val_ratio: float = 0.2             # Validation split ratio  
    test_ratio: float = 0.1            # Test split ratio (calculated)
    
    # Preprocessing options
    normalize_features: bool = True     # Enable feature normalization
    handle_missing_values: bool = True  # Handle NaN values
    create_time_features: bool = True   # Extract temporal features
    
    # Filtering options
    filter_invalid_vds: bool = True     # Remove VDs with too much missing data
    min_valid_ratio: float = 0.6       # Minimum valid data ratio per VD
    
    # Feature selection
    selected_features: Optional[List[str]] = None  # Use all features if None
    feature_names: List[str] = field(default_factory=lambda: [
        'avg_speed', 'total_volume', 'avg_occupancy', 'speed_std', 'lane_count'
    ])
```

### TrafficHDF5Config  
```python
@dataclass
class TrafficHDF5Config:
    # File processing
    batch_size: int = 1000             # Processing batch size
    overwrite: bool = False            # Overwrite existing files
    
    # Compression settings
    compression: str = "gzip"          # Compression algorithm
    compression_opts: int = 9          # Compression level (0-9)
    
    # Chunking for large files
    enable_chunking: bool = True       # Enable HDF5 chunking
    chunk_size: Tuple[int, ...] = (100, 1, 5)  # Chunk dimensions
    
    # Data validation
    validate_data: bool = True         # Perform data validation
    max_nan_ratio: float = 0.5         # Maximum allowed NaN ratio
```

## Model Configuration

### TrafficLSTMConfig
```python
@dataclass  
class TrafficLSTMConfig:
    # Architecture parameters
    input_size: int = 5                # Number of input features
    hidden_size: int = 128             # Hidden layer dimension
    num_layers: int = 2                # Number of LSTM layers
    dropout: float = 0.2               # Dropout rate
    
    # Model behavior
    bidirectional: bool = False        # Use bidirectional LSTM
    batch_first: bool = True           # Input format [batch, seq, features]
    
    # Multi-VD specific
    multi_vd_mode: bool = False        # Enable multi-VD processing
    num_vds: Optional[int] = None      # Number of VDs (for multi-VD mode)
    aggregation_method: str = "flatten"  # "flatten", "attention", "pooling"
    
    # Output configuration
    output_size: Optional[int] = None  # Output dimension (auto if None)
    prediction_length: int = 1         # Prediction horizon
```

### TrafficXLSTMConfig
```python
@dataclass
class TrafficXLSTMConfig:
    # Input/Output dimensions
    input_size: int = 5                # Input feature dimension
    output_size: int = 5               # Output dimension
    
    # xLSTM architecture
    embedding_dim: int = 128           # Embedding dimension
    num_blocks: int = 6                # Number of xLSTM blocks
    slstm_at: List[int] = field(default_factory=lambda: [1, 3])  # sLSTM positions
    
    # Memory configuration
    context_length: int = 256          # Context window length
    
    # Regularization
    dropout: float = 0.1               # Dropout rate
    layer_norm: bool = True            # Enable layer normalization
    
    # Training parameters
    vocab_size: Optional[int] = None   # For tokenized inputs (if applicable)
```

### SocialPoolingConfig
```python
@dataclass
class SocialPoolingConfig:
    # Spatial parameters
    pooling_radius: float = 1000.0     # Pooling radius in meters
    max_neighbors: int = 10            # Maximum neighbors to consider
    min_neighbors: int = 1             # Minimum neighbors required
    
    # Distance calculation
    distance_metric: str = "euclidean" # "euclidean", "manhattan", "haversine"
    
    # Weighting functions
    weighting_function: str = "gaussian"  # "gaussian", "exponential", "linear", "inverse"
    
    # Aggregation methods
    aggregation_method: str = "weighted_mean"  # "weighted_mean", "weighted_sum", "attention"
    
    # Performance optimization
    enable_caching: bool = True        # Cache distance calculations
    cache_coordinates: bool = True     # Cache coordinate transformations
    use_sparse_computation: bool = False  # Use sparse operations
    
    # Normalization
    normalize_weights: bool = True     # Normalize spatial weights
    normalize_coordinates: bool = False # Normalize coordinate inputs
    
    # Special options
    include_self: bool = True          # Include self node in pooling
    exclude_self: bool = False         # Explicitly exclude self node
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.pooling_radius <= 0:
            raise ValueError("pooling_radius must be positive")
        if self.max_neighbors < self.min_neighbors:
            raise ValueError("max_neighbors must be >= min_neighbors")
        if self.include_self and self.exclude_self:
            raise ValueError("Cannot both include and exclude self")
        return True
```

## Training Configuration

### TrainingConfig
```python
@dataclass
class TrainingConfig:
    # Basic training parameters
    epochs: int = 100                  # Number of training epochs
    batch_size: int = 32               # Training batch size
    learning_rate: float = 0.001       # Initial learning rate
    
    # Optimizer settings
    optimizer_type: str = "adam"       # "adam", "adamw", "sgd", "rmsprop"
    weight_decay: float = 0.0          # L2 regularization strength
    momentum: float = 0.9              # SGD momentum (if using SGD)
    
    # Learning rate scheduling
    scheduler_type: Optional[str] = None  # "step", "exponential", "reduce_on_plateau"
    scheduler_step_size: int = 30      # Steps for StepLR
    scheduler_gamma: float = 0.1       # LR decay factor
    scheduler_patience: int = 10       # Patience for ReduceLROnPlateau
    scheduler_factor: float = 0.5      # Factor for ReduceLROnPlateau
    
    # Regularization
    dropout: Optional[float] = None    # Override model dropout
    gradient_clip_value: Optional[float] = None  # Gradient clipping threshold
    gradient_clip_norm: Optional[float] = None   # Gradient norm clipping
    
    # Early stopping
    early_stopping_patience: int = 20  # Early stopping patience
    early_stopping_min_delta: float = 1e-4  # Minimum improvement threshold
    early_stopping_metric: str = "val_loss"  # Metric to monitor
    
    # Mixed precision training
    mixed_precision: bool = False      # Enable automatic mixed precision
    
    # Logging and checkpointing
    log_interval: int = 10             # Logging frequency (epochs)
    save_interval: int = 50            # Checkpoint saving frequency
    save_best_only: bool = True        # Save only best model
    
    # Experiment tracking
    experiment_name: str = "experiment"  # Experiment identifier
    tags: List[str] = field(default_factory=list)  # Experiment tags
    notes: str = ""                    # Additional notes
    
    # Hardware settings
    device: str = "cuda"               # "cuda", "cpu", "auto"
    num_workers: int = 4               # DataLoader workers
    pin_memory: bool = True            # Pin memory for GPU
    
    # Reproducibility
    seed: Optional[int] = None         # Random seed
    deterministic: bool = False        # Deterministic training
```

## Evaluation Configuration

### EvaluationConfig
```python
@dataclass
class EvaluationConfig:
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "mae", "mse", "rmse", "mape", "r2"
    ])
    
    # Evaluation settings
    batch_size: int = 64               # Evaluation batch size
    device: str = "cuda"               # Device for evaluation
    
    # Output options
    save_predictions: bool = False     # Save prediction results
    save_metrics: bool = True          # Save metric results
    
    # Visualization
    plot_predictions: bool = False     # Generate prediction plots
    plot_metrics: bool = True          # Generate metric plots
    
    # Statistical analysis
    confidence_intervals: bool = False  # Compute confidence intervals
    bootstrap_samples: int = 1000      # Bootstrap samples for CI
```

## Scenario-specific Configurations

### Urban Environment (High Density)
```python
urban_config = SocialPoolingConfig(
    pooling_radius=500.0,              # Smaller radius
    max_neighbors=12,                  # More neighbors
    weighting_function="gaussian",
    distance_metric="euclidean",
    enable_caching=True
)

urban_training = TrainingConfig(
    batch_size=32,                     # Standard batch size
    learning_rate=0.001,
    early_stopping_patience=25,
    gradient_clip_value=1.0
)
```

### Highway Environment (Sparse)
```python
highway_config = SocialPoolingConfig(
    pooling_radius=2000.0,             # Larger radius
    max_neighbors=5,                   # Fewer neighbors
    weighting_function="exponential",
    distance_metric="euclidean",
    use_sparse_computation=True
)

highway_training = TrainingConfig(
    batch_size=64,                     # Larger batch size
    learning_rate=0.0008,              # Slightly lower LR
    scheduler_type="reduce_on_plateau",
    scheduler_patience=15
)
```

### Development/Debug Configuration
```python
debug_config = SocialPoolingConfig(
    pooling_radius=800.0,
    max_neighbors=3,                   # Few neighbors for easier debugging
    weighting_function="linear",       # Simple weighting
    enable_caching=False,              # Disable caching for debugging
    include_self=True
)

debug_training = TrainingConfig(
    epochs=10,                         # Short training
    batch_size=8,                      # Small batches
    learning_rate=0.01,                # Higher LR for quick convergence
    log_interval=1,                    # Frequent logging
    save_interval=5
)
```

### Production Configuration
```python
production_config = SocialPoolingConfig(
    pooling_radius=1200.0,
    max_neighbors=8,
    weighting_function="gaussian",
    enable_caching=True,
    use_sparse_computation=True,       # Optimize for large scale
    normalize_weights=True
)

production_training = TrainingConfig(
    epochs=200,
    batch_size=32,
    learning_rate=0.0005,              # Conservative LR
    optimizer_type="adamw",
    weight_decay=0.01,
    scheduler_type="reduce_on_plateau",
    early_stopping_patience=30,
    mixed_precision=True,              # Memory optimization
    gradient_clip_value=1.0
)
```

## Environment Configuration

### Development Environment
```yaml
# cfgs/snakemake/dev.yaml
dataset:
  data_subset: "small"                 # Use subset of data
  max_vds: 5                          # Limit VD count
  max_hours: 1                        # 1 hour of data

training:
  epochs: 2                           # Quick training
  batch_size: 16
  early_stopping_patience: 5

output:
  base_dir: "blob/experiments/dev/"   # Development output
```

### Production Environment  
```yaml
# cfgs/snakemake/default.yaml
dataset:
  data_subset: "full"                 # Complete dataset
  max_vds: null                       # No VD limit
  max_hours: null                     # All available data

training:
  epochs: 100                         # Full training
  batch_size: 32
  early_stopping_patience: 20

output:
  base_dir: "blob/experiments/"       # Production output
```

## Validation Rules

### Configuration Validation
```python
def validate_training_config(config: TrainingConfig) -> List[str]:
    """Validate training configuration and return warnings"""
    warnings = []
    
    if config.learning_rate > 0.1:
        warnings.append("Learning rate seems high")
    
    if config.batch_size > 128:
        warnings.append("Large batch size may require LR adjustment")
        
    if config.gradient_clip_value and config.gradient_clip_value < 0.1:
        warnings.append("Very aggressive gradient clipping")
        
    return warnings

def validate_social_pooling_config(config: SocialPoolingConfig) -> List[str]:
    """Validate social pooling configuration"""
    warnings = []
    
    if config.pooling_radius > 5000:
        warnings.append("Very large pooling radius may be computationally expensive")
        
    if config.max_neighbors > 20:
        warnings.append("High neighbor count increases computation")
        
    if not config.enable_caching and config.max_neighbors > 10:
        warnings.append("Consider enabling caching for performance")
        
    return warnings
```

## Configuration Loading

### From File
```python
import yaml
from pathlib import Path

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_configs_from_file(config_path: str) -> Tuple[TrainingConfig, SocialPoolingConfig]:
    """Create configuration objects from file"""
    config_dict = load_config(config_path)
    
    training_config = TrainingConfig(**config_dict.get('training', {}))
    social_config = SocialPoolingConfig(**config_dict.get('social_pooling', {}))
    
    return training_config, social_config
```

### From Environment Variables
```python
import os

def load_config_from_env() -> TrainingConfig:
    """Load configuration from environment variables"""
    return TrainingConfig(
        epochs=int(os.getenv('EPOCHS', 100)),
        batch_size=int(os.getenv('BATCH_SIZE', 32)),
        learning_rate=float(os.getenv('LEARNING_RATE', 0.001)),
        device=os.getenv('DEVICE', 'cuda'),
        experiment_name=os.getenv('EXPERIMENT_NAME', 'default')
    )
```

## Configuration Best Practices

### Parameter Selection Guidelines
1. **Start Conservative**: Begin with smaller models and shorter training
2. **Scale Gradually**: Increase complexity after establishing baselines
3. **Monitor Resources**: Track memory usage and computation time
4. **Validate Changes**: Compare against previous configurations
5. **Document Decisions**: Record reasoning for parameter choices

### Common Pitfalls
- **Learning Rate Too High**: Causes training instability
- **Batch Size Too Large**: May require memory optimization or LR adjustment
- **Insufficient Regularization**: Leads to overfitting
- **Excessive Neighbors**: Increases computational overhead unnecessarily
- **Cache Disabled**: Reduces performance for repeated computations