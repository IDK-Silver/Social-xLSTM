# ADR-0700: Social Pooling 統一架構設計

**狀態**: Accepted  
**日期**: 2025-07-25  
**決策者**: Technical Team  
**技術故事**: Social Pooling 核心實現策略

## 背景與問題陳述

經過前期規劃，發現現有文檔存在架構設計不一致的問題：

1. **文檔矛盾**：`docs/architecture/social_xlstm_design.md` 與 `docs/technical/comparisons/` 的設計理念不統一
2. **文檔冗餘**：過多理論文檔但缺乏具體實施指導
3. **復用不足**：未充分利用現有的 TrafficLSTM (477行) 和 spatial_coords (437行) 穩定代碼

需要制定一個統一、實用的 Social Pooling 架構設計方案。

## 決策驅動因素

- **代碼復用**：最大化利用現有穩定組件
- **最小侵入**：不破壞現有架構和訓練系統
- **實施效率**：避免重複造輪子，專注核心功能
- **擴展性**：支持後續 xLSTM 和 Internal Gate Injection 整合
- **維護性**：清晰的模組化設計，易於調試和優化

## 考慮的選項

### 選項 1: 獨立 VD 模型架構
- **優點**: 符合原始 Social LSTM 理念
- **缺點**: 與現有 TrafficLSTM 架構衝突，開發量大

### 選項 2: 深度修改現有模型
- **優點**: 性能潛力高
- **缺點**: 風險大，破壞現有穩定代碼

### 選項 3: 包裝器模式 + Post-Fusion 策略
- **優點**: 最大復用，最小風險，漸進實現
- **缺點**: 初期性能可能不如深度整合

## 決策結果

**選擇**: 包裝器模式 + Post-Fusion 策略 (選項 3)

**理由**:
1. **風險控制**: 不修改現有穩定代碼
2. **開發效率**: 復用 900+ 行現有代碼，新增僅 360 行
3. **漸進實現**: 可分階段驗證，後續擴展 Internal Gate Injection
4. **維護性**: 模組化設計，清晰的責任分離

## 實施細節

### 核心架構設計

#### 1. 檔案結構
```
src/social_xlstm/models/
├── lstm.py                 # 保持不變 (477 lines)
├── social_pooling.py       # 新增核心模組 (~200 lines)
├── social_traffic.py       # 新增包裝器 (~150 lines)
└── __init__.py            # 更新導入 (~10 lines)

src/social_xlstm/utils/
└── spatial_coords.py      # 保持不變 (437 lines)
```

#### 2. 核心組件設計

**SocialPoolingConfig**:
```python
@dataclass
class SocialPoolingConfig:
    pooling_radius: float = 1000.0
    max_neighbors: int = 8
    distance_metric: str = "euclidean"
    weighting_function: str = "gaussian"
    aggregation_method: str = "weighted_mean"
    
    def validate(self) -> bool:
        # 配置驗證邏輯
        return True
```

**SocialPooling 核心類**:
```python
class SocialPooling(nn.Module):
    def __init__(self, config: SocialPoolingConfig):
        super().__init__()
        self.config = config
        # 復用現有座標系統
        from social_xlstm.utils.spatial_coords import CoordinateSystem
        self.coord_system = CoordinateSystem()
        
    def forward(self, features, coordinates, vd_ids) -> torch.Tensor:
        # 核心空間聚合邏輯
        pass
        
    def find_neighbors(self, coordinates, target_idx) -> List[int]:
        # 基於 spatial_coords 的鄰居發現
        pass
        
    def calculate_weights(self, distances) -> torch.Tensor:
        # 權重計算（高斯、指數等）
        pass
```

**SocialTrafficModel 包裝器**:
```python
class SocialTrafficModel(nn.Module):
    def __init__(self, base_model: TrafficLSTM, social_pooling: SocialPooling, strategy="post_fusion"):
        super().__init__()
        self.base_model = base_model      # 復用現有 TrafficLSTM
        self.social_pooling = social_pooling
        self.strategy = strategy
        
        if strategy == "post_fusion":
            self.fusion_layer = nn.Linear(...)
    
    def forward(self, x, coordinates, vd_ids) -> torch.Tensor:
        if self.strategy == "post_fusion":
            return self._post_fusion_forward(x, coordinates, vd_ids)
        elif self.strategy == "internal_injection":
            return self._internal_injection_forward(x, coordinates, vd_ids)
        
    def _post_fusion_forward(self, x, coordinates, vd_ids):
        # 1. 基礎模型處理
        base_output = self.base_model(x)
        # 2. Social Pooling
        social_features = self.social_pooling(x, coordinates, vd_ids)
        # 3. 特徵融合
        fused = torch.cat([base_output, social_features], dim=-1)
        return self.fusion_layer(fused)
```

**工廠函數簡化創建**:
```python
def create_social_traffic_model(
    base_config: TrafficLSTMConfig,
    social_config: SocialPoolingConfig,
    strategy: str = "post_fusion"
) -> SocialTrafficModel:
    base_model = TrafficLSTM(base_config)
    social_pooling = SocialPooling(social_config)
    return SocialTrafficModel(base_model, social_pooling, strategy)
```

#### 3. 數據流設計

**輸入格式**:
```python
def forward(self, x: torch.Tensor, coordinates: torch.Tensor, vd_ids: List[str]):
    # x: [batch_size, seq_len, feature_dim]  - 現有格式
    # coordinates: [num_vds, 2]  - 新增座標信息
    # vd_ids: List[str]  - VD 標識符
```

**與現有訓練系統整合**:
```python
# 最小變化
# 舊: model = TrafficLSTM(config)
# 新: model = create_social_traffic_model(lstm_config, social_config)
```

### 實施階段

#### 階段 1: 核心 Social Pooling 實現
- 實現 SocialPoolingConfig 和驗證邏輯
- 基於 spatial_coords 實現距離計算和權重函數
- 實現特徵聚合機制

#### 階段 2: 包裝器整合
- 實現 SocialTrafficModel 包裝器
- 實現 Post-Fusion 策略
- 創建工廠函數和模組導入更新

#### 階段 3: 整合測試  
- 單元測試和整合測試
- 與現有訓練腳本整合驗證
- 性能基準測試和調優

#### 階段 4: 未來擴展 (可選)
- xLSTM 模型支援
- Internal Gate Injection 策略實現

### 技術規格

**復用策略**:
- **距離計算**: 直接使用 `spatial_coords.py` 的 `CoordinateSystem` 類
- **模型基礎**: 基於現有 `TrafficLSTM` 架構
- **配置系統**: 遵循現有 `TrafficLSTMConfig` 模式
- **訓練接口**: 保持與現有訓練腳本兼容

**關鍵接口**:
```python
# 核心接口
social_pooling = SocialPooling(config)
social_features = social_pooling(features, coordinates, vd_ids)

# 整合接口
social_model = create_social_traffic_model(lstm_config, social_config)
output = social_model(x, coordinates, vd_ids)
```

## 後果

### 正面後果
- **代碼復用**: 900+ 行現有代碼直接利用，新增僅 360 行
- **風險控制**: 不修改現有穩定架構
- **維護性**: 清晰的模組化設計，職責分離
- **擴展性**: 預留 xLSTM 和 Internal Gate Injection 接口
- **兼容性**: 與現有訓練系統零衝突整合

### 負面後果
- **性能天花板**: Post-Fusion 策略的表達能力可能有限
- **額外複雜度**: 新增座標參數傳遞邏輯
- **依賴關係**: 對 spatial_coords 的格式依賴

### 風險與緩解措施
- **風險1**: 座標數據格式不匹配 / **緩解**: 實施前檢查現有 VD 數據格式
- **風險2**: 記憶體使用過高 / **緩解**: 提供 max_neighbors 參數限制複雜度
- **風險3**: 訓練不收斂 / **緩解**: 先確保基準 LSTM 訓練正常，漸進增加複雜度

## 成功指標

### 技術指標
- 代碼通過所有單元測試
- 記憶體使用增長 < 50%
- 訓練時間增長 < 30%

### 性能指標
- MAE/RMSE 比基準 LSTM 改善 > 5%
- 模型穩定收斂，無過擬合現象
- 支持多 VD 場景的空間交互建模

### 整合指標
- 與現有訓練腳本零衝突整合
- 配置系統向後兼容
- API 接口簡潔易用

## 相關決策

- [ADR-0100: Social Pooling vs Graph Networks](0100-social-pooling-vs-graph-networks.md)
- [ADR-0101: xLSTM vs Traditional LSTM](0101-xlstm-vs-traditional-lstm.md)
- [ADR-0200: Coordinate System Selection](0200-coordinate-system-selection.md)
- [ADR-0600: Social Pooling Integration Strategy](0600-social-pooling-integration-strategy.md)

## 註記

### 立即可執行的第一步
1. **檢查座標格式**: 確認現有 VD 數據的座標格式與 spatial_coords 兼容性
2. **創建核心配置**: 實現 SocialPoolingConfig 數據結構
3. **距離計算復用**: 基於 spatial_coords 實現距離矩陣計算

### 第一個檔案框架
```python
# src/social_xlstm/models/social_pooling.py
from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
from social_xlstm.utils.spatial_coords import CoordinateSystem

@dataclass
class SocialPoolingConfig:
    pooling_radius: float = 1000.0
    max_neighbors: int = 8
    distance_metric: str = "euclidean"
    weighting_function: str = "gaussian"
    aggregation_method: str = "weighted_mean"
    
    def validate(self) -> bool:
        if self.pooling_radius <= 0:
            raise ValueError("pooling_radius must be positive")
        if self.max_neighbors < 1:
            raise ValueError("max_neighbors must be >= 1")
        return True

class SocialPooling(nn.Module):
    def __init__(self, config: SocialPoolingConfig):
        super().__init__()
        self.config = config
        config.validate()
        self.coord_system = CoordinateSystem()
        
    def forward(self, features: torch.Tensor, coordinates: torch.Tensor, vd_ids: List[str]) -> torch.Tensor:
        # 核心實現將在此處
        pass
```

### 文檔整理
此 ADR 統一了之前分散的設計文檔，後續將：
1. 淘汰衝突的架構文檔
2. 專注於核心實現
3. 避免創建冗餘的理論文檔