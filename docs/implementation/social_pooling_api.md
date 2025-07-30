# Social Pooling API Reference

## Overview

Social Pooling API 提供座標驅動的空間聚合機制，用於 Social-xLSTM 交通預測系統。此 API 實現了基於地理座標的鄰居發現和特徵聚合功能。

## Prerequisites

- Python 3.11+
- PyTorch 2.0+
- 已安裝 `social_xlstm` 套件
- 基本了解 PyTorch tensor 操作

## Core Classes

### SocialPooling

空間池化的核心實現類別，提供座標驅動的特徵聚合功能。

```python
from social_xlstm.models.social_pooling import SocialPooling, SocialPoolingConfig
import torch

class SocialPooling(nn.Module):
    """
    座標驅動的社交池化模組
    
    基於地理座標計算空間權重，聚合鄰近 VD (Vehicle Detector) 的交通特徵。
    支援多種距離計算方法和權重函數。
    
    Attributes:
        config (SocialPoolingConfig): 池化配置參數
        _distance_cache (Optional[torch.Tensor]): 距離矩陣快取
        _coordinate_cache (Optional[torch.Tensor]): 座標快取
    """
```

#### Constructor

```python
def __init__(self, config: SocialPoolingConfig)
```

**Parameters:**
- `config` (SocialPoolingConfig): 社交池化配置物件

**Description:**
初始化 Social Pooling 模組，設定距離計算方法、權重函數和池化參數。

**Example:**
```python
from social_xlstm.models.social_pooling import SocialPooling, SocialPoolingConfig

config = SocialPoolingConfig(
    pooling_radius=1000.0,  # 1km radius
    max_neighbors=10,
    weighting_function="gaussian",
    distance_metric="euclidean"
)
social_pooling = SocialPooling(config)
```

#### Forward Method

```python
def forward(self, 
           features: torch.Tensor, 
           coordinates: torch.Tensor, 
           vd_ids: List[str],
           vd_mask: Optional[torch.Tensor] = None) -> torch.Tensor
```

**Parameters:**
- `features` (torch.Tensor): 形狀為 `[batch_size, seq_len, feature_dim]` 的輸入特徵
- `coordinates` (torch.Tensor): 形狀為 `[num_vds, 2]` 的 VD 座標 (x, y)
- `vd_ids` (List[str]): VD 識別碼列表，長度為 `num_vds`
- `vd_mask` (Optional[torch.Tensor]): 形狀為 `[num_vds]` 的可用性遮罩

**Returns:**
- `torch.Tensor`: 形狀為 `[batch_size, seq_len, feature_dim]` 的池化後特徵

**Description:**
執行空間特徵聚合。對每個 VD 位置，基於座標距離找到鄰居，計算空間權重，並聚合鄰近 VD 的特徵。

**Algorithm:**
1. 計算所有 VD 間的距離矩陣
2. 基於距離和池化半徑找到每個 VD 的鄰居
3. 計算空間權重（高斯、指數衰減等）
4. 執行權重特徵聚合
5. 返回增強的特徵表示

**Computational Complexity:**
- 時間複雜度: O(N² + B×S×F×K)，其中 N=VD數量，B=batch_size，S=seq_len，F=feature_dim，K=平均鄰居數
- 空間複雜度: O(N² + B×S×F)

**Example:**
```python
import torch

# 示例數據
batch_size, seq_len, feature_dim = 32, 12, 3
num_vds = 50

features = torch.randn(batch_size, seq_len, feature_dim)
coordinates = torch.randn(num_vds, 2) * 1000  # 座標範圍 ±1000m
vd_ids = [f"VD_{i:03d}" for i in range(num_vds)]

# 執行社交池化
pooled_features = social_pooling(features, coordinates, vd_ids)
print(f"Original shape: {features.shape}")
print(f"Pooled shape: {pooled_features.shape}")  # 應該相同
```

#### Distance Calculation Methods

```python
def calculate_distances(self, coordinates: torch.Tensor) -> torch.Tensor
```

**Parameters:**
- `coordinates` (torch.Tensor): 形狀為 `[num_vds, 2]` 的 VD 座標

**Returns:**
- `torch.Tensor`: 形狀為 `[num_vds, num_vds]` 的距離矩陣

**Available Distance Metrics:**
- `"euclidean"`: 歐幾里得距離 `sqrt((x1-x2)² + (y1-y2)²)`
- `"manhattan"`: 曼哈頓距離 `|x1-x2| + |y1-y2|`
- `"haversine"`: 球面距離（適用於經緯度座標）

**Example:**
```python
coordinates = torch.tensor([[0, 0], [100, 0], [0, 100]], dtype=torch.float32)
distances = social_pooling.calculate_distances(coordinates)
print(distances)
# tensor([[  0.0000, 100.0000, 100.0000],
#         [100.0000,   0.0000, 141.4214],
#         [100.0000, 141.4214,   0.0000]])
```

#### Weight Computation

```python
def compute_weights(self, distances: torch.Tensor) -> torch.Tensor
```

**Parameters:**
- `distances` (torch.Tensor): 形狀為 `[num_vds, num_vds]` 的距離矩陣

**Returns:**
- `torch.Tensor`: 形狀為 `[num_vds, num_vds]` 的權重矩陣

**Available Weighting Functions:**
- `"gaussian"`: 高斯核函數 `exp(-d² / (2σ²))`，其中 σ = pooling_radius / 3
- `"exponential"`: 指數衰減 `exp(-d / λ)`，其中 λ = pooling_radius
- `"inverse"`: 反距離權重 `1 / (1 + d)`
- `"linear"`: 線性衰減 `max(0, 1 - d/radius)`

**Example:**
```python
distances = torch.tensor([[0, 500, 1000], [500, 0, 500], [1000, 500, 0]], dtype=torch.float32)
weights = social_pooling.compute_weights(distances)
print(f"Weights shape: {weights.shape}")
print(f"Max weight: {weights.max():.4f}")  # 應該是 1.0 (自己到自己)
```

#### Feature Aggregation

```python
def aggregate_features(self, 
                      features: torch.Tensor, 
                      weights: torch.Tensor,
                      neighbor_indices: torch.Tensor) -> torch.Tensor
```

**Parameters:**
- `features` (torch.Tensor): 形狀為 `[batch_size, seq_len, feature_dim]` 的輸入特徵
- `weights` (torch.Tensor): 形狀為 `[num_vds, max_neighbors]` 的權重矩陣
- `neighbor_indices` (torch.Tensor): 形狀為 `[num_vds, max_neighbors]` 的鄰居索引

**Returns:**
- `torch.Tensor`: 聚合後的特徵，形狀與輸入相同

**Aggregation Methods:**
- `"weighted_mean"`: 權重平均 `Σ(w_i × f_i) / Σ(w_i)`
- `"weighted_sum"`: 權重總和 `Σ(w_i × f_i)`
- `"attention"`: 注意力機制聚合（進階功能）

#### Neighbor Discovery

```python
def find_neighbors(self, 
                  distances: torch.Tensor, 
                  max_neighbors: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]
```

**Parameters:**
- `distances` (torch.Tensor): 距離矩陣
- `max_neighbors` (Optional[int]): 最大鄰居數量，預設使用配置值

**Returns:**
- `Tuple[torch.Tensor, torch.Tensor]`: (neighbor_indices, neighbor_distances)

**Description:**
基於距離和池化半徑找到每個 VD 的鄰居。返回鄰居索引和對應距離。

### SocialPoolingConfig

```python
@dataclass
class SocialPoolingConfig:
    """
    Social Pooling 配置類別
    
    包含所有池化相關的參數設定，支援不同的距離計算方法和權重函數。
    """
    
    # 空間配置
    pooling_radius: float = 1000.0          # 池化半徑（公尺）
    max_neighbors: int = 10                 # 最大鄰居數量
    
    # 計算方法
    distance_metric: str = "euclidean"      # 距離計算方法
    weighting_function: str = "gaussian"    # 權重函數類型
    aggregation_method: str = "weighted_mean"  # 聚合方法
    
    # 性能優化
    enable_caching: bool = True             # 啟用距離矩陣快取
    cache_coordinates: bool = True          # 快取座標信息
    use_sparse_computation: bool = False    # 使用稀疏計算（大規模部署）
    
    # 特殊功能
    normalize_weights: bool = True          # 標準化權重
    include_self: bool = True              # 包含自身節點
    min_neighbors: int = 1                 # 最小鄰居數量
    
    def validate(self) -> bool:
        """驗證配置參數的有效性"""
        if self.pooling_radius <= 0:
            raise ValueError("pooling_radius must be positive")
        if self.max_neighbors < self.min_neighbors:
            raise ValueError("max_neighbors must be >= min_neighbors")
        return True
```

**Configuration Examples:**

```python
# 基本配置
basic_config = SocialPoolingConfig()

# 高精度配置（小半徑，多鄰居）
precision_config = SocialPoolingConfig(
    pooling_radius=500.0,
    max_neighbors=15,
    weighting_function="gaussian"
)

# 效能優化配置（大半徑，少鄰居）
performance_config = SocialPoolingConfig(
    pooling_radius=2000.0,
    max_neighbors=5,
    enable_caching=True,
    use_sparse_computation=True
)

# 地理座標配置（使用 Haversine 距離）
geo_config = SocialPoolingConfig(
    distance_metric="haversine",
    pooling_radius=1000.0,
    weighting_function="exponential"
)
```

## Integration with TrafficLSTM

### Factory Function

```python
def create_social_traffic_model(
    base_model_type: str,           # "lstm" or "xlstm"
    strategy: str,                  # "post_fusion" or "internal_injection"
    base_config: Union[TrafficLSTMConfig, TrafficXLSTMConfig],
    social_config: SocialPoolingConfig
) -> SocialTrafficModel
```

**Description:**
工廠函數，創建帶有 Social Pooling 的交通預測模型。

**Example:**
```python
from social_xlstm.models.lstm import TrafficLSTMConfig
from social_xlstm.models.social_pooling import SocialPoolingConfig, create_social_traffic_model

# 配置設定
lstm_config = TrafficLSTMConfig(hidden_size=64, num_layers=2)
social_config = SocialPoolingConfig(pooling_radius=800.0, max_neighbors=8)

# 創建模型
model = create_social_traffic_model(
    base_model_type="lstm",
    strategy="post_fusion",
    base_config=lstm_config,
    social_config=social_config
)

print(f"Model type: {type(model)}")
print(f"Strategy: {model.strategy}")
```

## Error Handling

### Common Exceptions

```python
class SocialPoolingError(Exception):
    """Social Pooling 相關錯誤的基類"""
    pass

class InvalidCoordinatesError(SocialPoolingError):
    """座標數據無效錯誤"""
    pass

class InsufficientNeighborsError(SocialPoolingError):
    """鄰居數量不足錯誤"""
    pass

class ConfigurationError(SocialPoolingError):
    """配置參數錯誤"""
    pass
```

### Error Handling Examples

```python
try:
    # 可能引發錯誤的操作
    pooled_features = social_pooling(features, coordinates, vd_ids)
except InvalidCoordinatesError as e:
    print(f"座標數據錯誤: {e}")
    # 處理座標問題
except InsufficientNeighborsError as e:
    print(f"鄰居不足: {e}")
    # 降低 min_neighbors 或增加 pooling_radius
except ConfigurationError as e:
    print(f"配置錯誤: {e}")
    # 檢查並修正配置參數
```

## Performance Considerations

### Memory Usage

- **距離矩陣**: O(N²) 記憶體，其中 N 為 VD 數量
- **特徵張量**: O(B×S×F) 記憶體
- **快取優化**: 啟用快取可減少重複計算，但增加記憶體使用

### Optimization Tips

```python
# 1. 使用快取減少重複計算
config = SocialPoolingConfig(enable_caching=True)

# 2. 限制鄰居數量控制計算複雜度
config = SocialPoolingConfig(max_neighbors=5)

# 3. 適當設定池化半徑
config = SocialPoolingConfig(pooling_radius=800.0)  # 平衡精度和效能

# 4. 大規模部署使用稀疏計算
config = SocialPoolingConfig(use_sparse_computation=True)
```

### Benchmarking

```python
import time
import torch

def benchmark_social_pooling(social_pooling, features, coordinates, vd_ids, num_runs=100):
    """效能基準測試"""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.time()
    for _ in range(num_runs):
        _ = social_pooling(features, coordinates, vd_ids)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"平均執行時間: {avg_time*1000:.2f} ms")
    return avg_time
```

## See Also

- [Social Pooling Integration Guide](../guides/social_pooling_integration_guide.md)
- [Configuration Reference](social_pooling_config.md)
- [Performance Optimization Guide](../technical/social_pooling_optimization.md)
- [ADR-0100: Social Pooling vs Graph Networks](../adr/0100-social-pooling-vs-graph-networks.md)
- [ADR-0600: Social Pooling Integration Strategy](../adr/0600-social-pooling-integration-strategy.md)