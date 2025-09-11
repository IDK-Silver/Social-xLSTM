# API Reference

Complete API reference for Social-xLSTM components and modules.

## Core Package Structure (`src/social_xlstm/`)

### Dataset Module (`dataset/`)

The dataset module provides comprehensive data handling capabilities through a structured sub-package architecture:

#### Configuration Classes (`config/`)

**TrafficDatasetConfig**
```python
@dataclass
class TrafficDatasetConfig:
    sequence_length: int = 12
    prediction_length: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    normalize_features: bool = True
```

**TrafficHDF5Config**
```python
@dataclass  
class TrafficHDF5Config:
    batch_size: int = 1000
    compression: str = "gzip"
    compression_opts: int = 9
```

#### Core Data Processing (`core/`)

**TrafficDataProcessor**
```python
class TrafficDataProcessor:
    def __init__(self, config: TrafficDatasetConfig)
    
    def process(self, raw_data: np.ndarray) -> np.ndarray:
        """Process raw traffic data with normalization and missing value handling"""
        
    def normalize_features(self, features: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
        """Normalize features using StandardScaler"""
        
    def create_time_features(self, timestamps: List[str]) -> np.ndarray:
        """Extract temporal features from timestamps"""
```

**TrafficTimeSeries**
```python
class TrafficTimeSeries(Dataset):
    def __init__(self, 
                 data_path: str,
                 config: TrafficDatasetConfig,
                 split: str = "train")
                 
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]
```

**TrafficDataModule**
```python
class TrafficDataModule(LightningDataModule):
    def __init__(self,
                 data_path: str, 
                 config: TrafficDatasetConfig,
                 batch_size: int = 32)
                 
    def setup(self, stage: Optional[str] = None)
    def train_dataloader(self) -> DataLoader
    def val_dataloader(self) -> DataLoader  
    def test_dataloader(self) -> DataLoader
```

#### Storage & Persistence (`storage/`)

**TrafficHDF5Converter**
```python
class TrafficHDF5Converter:
    def __init__(self, config: TrafficHDF5Config)
    
    def convert_json_to_h5(self, 
                          source_dir: str,
                          output_path: str,
                          selected_vdids: Optional[List[str]] = None)
```

**TrafficHDF5Reader**
```python
class TrafficHDF5Reader:
    def __init__(self, file_path: str)
    
    def read_features(self, 
                     start_idx: Optional[int] = None,
                     end_idx: Optional[int] = None) -> np.ndarray
                     
    def read_metadata(self) -> Dict[str, Any]
    def get_vd_list(self) -> List[str]
```

**TrafficFeature**
```python
@dataclass
class TrafficFeature:
    avg_speed: float
    total_volume: float
    avg_occupancy: float
    speed_std: float
    lane_count: int
```

### Models Module (`models/`)

**TrafficLSTM**
```python
class TrafficLSTM(nn.Module):
    def __init__(self, config: TrafficLSTMConfig)
    
    @classmethod
    def create_single_vd_model(cls,
                              input_size: int = 5,
                              hidden_size: int = 128,
                              num_layers: int = 2,
                              dropout: float = 0.2) -> 'TrafficLSTM'
                              
    @classmethod  
    def create_multi_vd_model(cls,
                             num_vds: int,
                             input_size: int = 5,
                             hidden_size: int = 256,
                             num_layers: int = 2,
                             dropout: float = 0.3) -> 'TrafficLSTM'
                             
    @staticmethod
    def parse_multi_vd_output(flat_output: torch.Tensor,
                             num_vds: int,
                             num_features: int) -> torch.Tensor
                             
    @staticmethod
    def extract_vd_prediction(structured_output: torch.Tensor,
                             vd_index: int) -> torch.Tensor
```

**TrafficXLSTM**
```python
class TrafficXLSTM(nn.Module):
    def __init__(self, config: TrafficXLSTMConfig)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Returns model information including parameter count"""
        
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

### Social Pooling Module (`models/social_pooling`)

**SocialPooling**
```python
class SocialPooling(nn.Module):
    def __init__(self, config: SocialPoolingConfig)
    
    def forward(self, 
               features: torch.Tensor,
               coordinates: torch.Tensor, 
               vd_ids: List[str],
               vd_mask: Optional[torch.Tensor] = None) -> torch.Tensor
               
    def calculate_distances(self, coordinates: torch.Tensor) -> torch.Tensor
    def compute_weights(self, distances: torch.Tensor) -> torch.Tensor
    def find_neighbors(self, distances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

**SocialPoolingConfig**
```python
@dataclass
class SocialPoolingConfig:
    pooling_radius: float = 1000.0
    max_neighbors: int = 10
    distance_metric: str = "euclidean"  # "euclidean", "manhattan", "haversine"
    weighting_function: str = "gaussian"  # "gaussian", "exponential", "linear"
    aggregation_method: str = "weighted_mean"
    enable_caching: bool = True
    normalize_weights: bool = True
    include_self: bool = True
```

**Current Architecture**
```python
# Use DistributedSocialXLSTMModel directly
from social_xlstm.models.distributed_social_xlstm import DistributedSocialXLSTMModel

model = DistributedSocialXLSTMModel(
    xlstm_config=xlstm_config,
    num_features=num_features,
    social_pool_type="weighted_mean"  # or "weighted_sum", "attention"
)
```

### Evaluation Module (`evaluation/`)

**ModelEvaluator**
```python
class ModelEvaluator:
    def __init__(self)
    
    def evaluate(self, 
                model: nn.Module,
                data_loader: DataLoader,
                device: str = "cuda") -> Dict[str, float]:
        """Returns metrics: MAE, MSE, RMSE, MAPE, R²"""
        
    def compute_mae(self, predictions: torch.Tensor, targets: torch.Tensor) -> float
    def compute_mse(self, predictions: torch.Tensor, targets: torch.Tensor) -> float  
    def compute_rmse(self, predictions: torch.Tensor, targets: torch.Tensor) -> float
    def compute_mape(self, predictions: torch.Tensor, targets: torch.Tensor) -> float
    def compute_r2(self, predictions: torch.Tensor, targets: torch.Tensor) -> float
```

### Training Module (`training/`)

**Trainer**
```python
class Trainer:
    def __init__(self,
                model: nn.Module,
                training_config: TrainingConfig,
                train_loader: DataLoader,
                val_loader: DataLoader,
                test_loader: Optional[DataLoader] = None)
                
    def train(self) -> TrainingRecorder
    def train_epoch(self) -> Dict[str, float]
    def validate_epoch(self) -> Dict[str, float]
    def test(self) -> Dict[str, float]
```

**TrainingConfig**
```python
@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer_type: str = "adam"  # "adam", "adamw", "sgd"
    scheduler_type: Optional[str] = None
    weight_decay: float = 0.0
    gradient_clip_value: Optional[float] = None
    early_stopping_patience: int = 20
    mixed_precision: bool = False
    experiment_name: str = "experiment"
```

**TrainingRecorder**
```python
class TrainingRecorder:
    def __init__(self,
                experiment_name: str,
                model_config: Dict[str, Any],
                training_config: Dict[str, Any])
                
    def log_epoch(self,
                 epoch: int,
                 train_loss: float,
                 val_loss: Optional[float],
                 train_metrics: Dict[str, float],
                 val_metrics: Dict[str, float],
                 learning_rate: float,
                 epoch_time: float,
                 **kwargs)
                 
    def save(self, file_path: str)
    @classmethod
    def load(cls, file_path: str) -> 'TrainingRecorder'
    
    def get_best_epoch(self) -> EpochRecord
    def get_training_summary(self) -> Dict[str, Any]
    def get_metric_history(self, metric_name: str, split: str) -> List[float]
    def analyze_training_stability(self) -> Dict[str, float]
```

### Utilities Module (`utils/`)

**Coordinate Conversion**
```python
def convert_coords(coordinates: np.ndarray, 
                  from_system: str, 
                  to_system: str) -> np.ndarray
```

**Spatial Coordinates**
```python
class SpatialCoords:
    def __init__(self, coordinates: np.ndarray)
    
    def compute_distances(self, method: str = "euclidean") -> np.ndarray
    def find_neighbors(self, radius: float) -> Dict[int, List[int]]
```

## Usage Examples

### Basic Model Training
```python
from social_xlstm.models.lstm import TrafficLSTM
from social_xlstm.dataset import TrafficDataModule
from social_xlstm.training import Trainer, TrainingConfig

# Setup data
data_module = TrafficDataModule("data.h5", batch_size=32)
data_module.setup()

# Create model
model = TrafficLSTM.create_single_vd_model(hidden_size=128)

# Train
config = TrainingConfig(epochs=50, learning_rate=0.001)
trainer = Trainer(model, config, data_module.train_dataloader(), 
                 data_module.val_dataloader())
history = trainer.train()
```

### Social Pooling Integration
```python
from social_xlstm.models.social_pooling import SocialPooling, SocialPoolingConfig, create_social_traffic_model
from social_xlstm.models.lstm import TrafficLSTMConfig

# Configure components
lstm_config = TrafficLSTMConfig(hidden_size=64)
social_config = SocialPoolingConfig(pooling_radius=1000.0, max_neighbors=8)

# Create distributed Social-xLSTM model
from social_xlstm.models.distributed_social_xlstm import DistributedSocialXLSTMModel
from social_xlstm.models.xlstm import TrafficXLSTMConfig

xlstm_config = TrafficXLSTMConfig(input_size=5, hidden_size=64)

model = DistributedSocialXLSTMModel(
    xlstm_config=xlstm_config,
    num_features=5,
    enable_spatial_pooling=True,
    social_pool_type="weighted_mean"
)
```

### Multi-VD Output Parsing
```python
# Model prediction
flat_output = model(inputs)  # [batch_size, 1, num_vds * num_features]

# Parse to structured format
structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds=3, num_features=5)

# Extract specific VD predictions
vd_001 = TrafficLSTM.extract_vd_prediction(structured, vd_index=1)
```

## Error Handling

### Common Exceptions
```python
class SocialPoolingError(Exception):
    """Base class for Social Pooling errors"""
    
class InvalidCoordinatesError(SocialPoolingError):
    """Invalid coordinate data"""
    
class InsufficientNeighborsError(SocialPoolingError):
    """Not enough neighbors found"""
    
class ConfigurationError(SocialPoolingError):
    """Invalid configuration parameters"""
```

### Safe Operations
```python
try:
    pooled_features = social_pooling(features, coordinates, vd_ids)
except InvalidCoordinatesError:
    # Handle coordinate issues
    pass
except InsufficientNeighborsError:
    # Adjust pooling_radius or min_neighbors
    pass
```

## Performance Considerations

### Memory Usage
- Distance matrix: O(N²) where N = number of VDs
- Feature tensors: O(B×S×F) where B=batch_size, S=sequence_length, F=features
- Enable caching to reduce computation at cost of memory

### Optimization Tips
```python
# Reduce computational complexity
config = SocialPoolingConfig(
    max_neighbors=5,           # Limit neighbors
    enable_caching=True,       # Cache distance calculations
    pooling_radius=800.0       # Reasonable radius
)

# For large deployments
config = SocialPoolingConfig(
    use_sparse_computation=True,  # Sparse calculations
    batch_processing=True         # Process in batches
)
```