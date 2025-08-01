# Social Pooling 設計說明

## 概述

Social Pooling 是 Social-xLSTM 的核心創新機制，實現座標驅動的空間聚合。本文檔整合了所有相關設計決策，提供統一且可操作的實現指南。

## 核心原理

### 基本概念

Social Pooling 基於以下核心原則：

1. **座標驅動的空間聚合**: 使用 VD（Vehicle Detector）地理座標自動發現鄰居
2. **無拓撲依賴**: 不需要預定義的圖結構，純粹基於座標的空間關係學習
3. **可解釋性**: 明確的空間權重計算，基於物理距離的直觀理解

### 技術原理

```python
def social_pooling(node_features, coordinates, radius):
    """
    座標驅動的社交池化
    
    Args:
        node_features: [num_nodes, hidden_size] 節點特徵
        coordinates: [num_nodes, 2] 節點座標 (x, y)
        radius: float 空間半徑
    
    Returns:
        pooled_features: [num_nodes, hidden_size] 池化後特徵
    """
    # 1. 計算距離矩陣
    distances = compute_distance_matrix(coordinates)
    
    # 2. 生成空間權重
    spatial_weights = gaussian_kernel(distances, radius)
    
    # 3. 加權聚合
    pooled_features = weighted_aggregation(node_features, spatial_weights)
    
    return pooled_features
```

## 設計決策

### 為什麼選擇 Social Pooling

**技術優勢**:
- 適應真實世界的不完整資訊（台灣公路總局數據缺乏完整拓撲）
- 感測器分佈不規則時仍能有效工作
- 易於添加新的空間節點，不需要重新設計圖結構

**研究創新性**:
- 填補無拓撲交通預測的空白
- 結合 Social LSTM 的空間互動思想
- 為相關研究提供新思路

**實用價值**:
- 降低系統部署複雜度
- 提高模型的泛化能力
- 支援動態的空間配置

### 為什麼拒絕 Graph Networks

**數據限制**:
- 缺乏完整的道路拓撲資訊
- 人工構建鄰接矩陣容易出錯
- 靜態圖結構不適應動態交通

**計算複雜度**:
- 圖卷積需要複雜的矩陣運算
- 梯度傳播通過圖結構複雜
- 模型解釋性差

## 實現架構

### 兩種整合策略

Social-xLSTM 提供兩種 Social Pooling 整合策略，各有不同的技術特點和適用場景：

#### 1. Post-Fusion 策略（優先實現）

**技術原理**：在基礎模型（LSTM/xLSTM）輸出後進行 Social Pooling 聚合

**數據流**：`VD 輸入 → 基礎模型 → 個體特徵 → Social Pooling → 空間特徵 → 特徵融合 → 預測輸出`

**核心優勢**：
- 模組化設計，組件分離清晰
- 實現簡單，調試容易
- 與現有模型兼容性高
- 訓練穩定性好

#### 2. Internal Gate Injection (IGI) 策略（未來擴展）

**技術原理**：直接在 xLSTM 門控計算中注入 Social 信息

**數據流**：`VD 輸入 + 鄰居空間特徵 → 增強的 xLSTM 門控 → 預測輸出`

**核心優勢**：
- 深度整合，理論表達能力更強
- 即時響應，空間信息實時影響記憶機制
- 細粒度控制，精確調節各個門控
- 更強的空間-時間耦合建模

### 當前實施決策

經過技術評估，確定**優先實現 Post-Fusion 策略**：

#### 核心組件

**1. SocialPoolingConfig**
```python
@dataclass
class SocialPoolingConfig:
    pooling_radius: float = 1000.0      # 池化半徑（公尺）
    max_neighbors: int = 8              # 最大鄰居數
    distance_metric: str = "euclidean"  # 距離計算方法
    weighting_function: str = "gaussian" # 權重函數
    aggregation_method: str = "weighted_mean" # 聚合方法
```

**2. SocialPooling 核心類**
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
```

**3. SocialTrafficModel 包裝器**
```python
class SocialTrafficModel(nn.Module):
    def __init__(self, base_model: TrafficLSTM, social_pooling: SocialPooling):
        super().__init__()
        self.base_model = base_model      # 復用現有 TrafficLSTM
        self.social_pooling = social_pooling
        self.fusion_layer = nn.Linear(...)
    
    def forward(self, x, coordinates, vd_ids) -> torch.Tensor:
        # 1. 基礎模型處理
        base_output = self.base_model(x)
        # 2. Social Pooling
        social_features = self.social_pooling(x, coordinates, vd_ids)
        # 3. 特徵融合
        fused = torch.cat([base_output, social_features], dim=-1)
        return self.fusion_layer(fused)
```

### 數據流設計

**輸入格式**:
```python
def forward(self, x: torch.Tensor, coordinates: torch.Tensor, vd_ids: List[str]):
    # x: [batch_size, seq_len, feature_dim]  - 現有格式
    # coordinates: [num_vds, 2]  - 新增座標信息
    # vd_ids: List[str]  - VD 標識符
```

**與現有系統整合**:
```python
# 最小變化
# 舊: model = TrafficLSTM(config)
# 新: model = create_social_traffic_model(lstm_config, social_config)
```

## 實施階段

### 階段 1: 核心 Social Pooling 實現
- 實現 SocialPoolingConfig 和驗證邏輯
- 基於 spatial_coords 實現距離計算和權重函數
- 實現特徵聚合機制

### 階段 2: 包裝器整合
- 實現 SocialTrafficModel 包裝器
- 實現 Post-Fusion 策略
- 創建工廠函數

### 階段 3: 整合測試
- 單元測試和整合測試
- 與現有訓練腳本整合驗證
- 性能基準測試和調優

### 階段 4: IGI 策略實現（第二階段）
- Internal Gate Injection 策略完整實現
- sLSTM 和 mLSTM 的門控增強
- 兩種策略的性能比較和分析
- 深度空間-時間耦合建模驗證

## 技術規格

### 復用策略
- **距離計算**: 直接使用 `spatial_coords.py` 的 `CoordinateSystem` 類
- **模型基礎**: 基於現有 `TrafficLSTM` 架構
- **配置系統**: 遵循現有 `TrafficLSTMConfig` 模式
- **訓練接口**: 保持與現有訓練腳本兼容

### 性能指標
- MAE/RMSE 比基準 LSTM 改善 > 5%
- 記憶體使用增長 < 50%
- 訓練時間增長 < 30%
- 模型穩定收斂，無過擬合現象

## 使用示例

### 基本使用
```python
# 1. 配置
social_config = SocialPoolingConfig(
    pooling_radius=1000.0,
    max_neighbors=8,
    weighting_function="gaussian"
)

# 2. 創建模型
social_model = create_social_traffic_model(
    base_config=lstm_config,
    social_config=social_config
)

# 3. 訓練和預測
output = social_model(x, coordinates, vd_ids)
```

### 配置模式

**城市環境配置**:
```python
urban_config = SocialPoolingConfig(
    pooling_radius=500.0,       # 較小半徑
    max_neighbors=12,           # 更多鄰居
    weighting_function="gaussian"
)
```

**高速公路配置**:
```python
highway_config = SocialPoolingConfig(
    pooling_radius=2000.0,      # 較大半徑
    max_neighbors=5,            # 較少鄰居
    weighting_function="exponential"
)
```

## 策略技術比較

### 深度比較表

| 技術方面 | Post-Fusion | Internal Gate Injection |
|----------|-------------|------------------------|
| **整合時機** | 基礎模型輸出後 | 門控計算過程中 |
| **實現複雜度** | 簡單（模組化） | 複雜（深度整合） |
| **計算開銷** | 較低 | 較高 |
| **訓練穩定性** | 高（組件獨立） | 中等（需要調參） |
| **空間響應** | 延遲 1 時間步 | 即時響應 |
| **記憶影響** | 輸出層融合 | 門控層調節 |
| **調試難度** | 容易（組件分離） | 困難（內部整合） |
| **擴展性** | 高（任何基礎模型） | 中等（需要模型適配） |
| **表達能力** | 中等 | 理論上更高 |
| **適用場景** | 一般應用、快速原型 | 高性能要求、研究探索 |

### 選擇指導原則

**選擇 Post-Fusion 當**：
- 開發速度是優先考慮
- 模型可解釋性重要
- 計算資源有限
- 需要穩定可靠的性能
- 團隊經驗水平不一

**選擇 IGI 當**：
- 追求最大性能表現
- 需要深度空間-時間耦合
- 有充足的計算資源
- 進行前沿研究探索
- 團隊有深度學習專業知識

## 完整技術規格

### 詳細技術文檔

**Post-Fusion 策略**：
- 完整規格：[../../reference/post-fusion-specification.md](../../reference/post-fusion-specification.md)
- 數學公式、實現架構、性能分析

**Internal Gate Injection 策略**：
- 完整規格：[../../reference/internal-gate-injection-specification.md](../../reference/internal-gate-injection-specification.md)
- sLSTM/mLSTM 門控增強、複雜度分析、深度整合設計

**基礎數學規格**：
- 數學基礎：[../../reference/mathematical-specifications.md](../../reference/mathematical-specifications.md)
- Social Pooling 核心公式、xLSTM 架構、評估指標

## 相關技術決策

本設計基於以下關鍵決策：
- **ADR-0100**: Social Pooling vs Graph Networks - 選擇 Social Pooling 方法
- **ADR-0200**: 座標系統選擇 - 支援 Social Pooling 的座標處理
- **ADR-0700**: 統一架構設計 - 確定包裝器模式和 Post-Fusion 策略優先

## 風險與緩解

### 技術風險
- **風險**: 座標數據格式不匹配
- **緩解**: 實施前檢查現有 VD 數據格式

- **風險**: 記憶體使用過高
- **緩解**: 提供 max_neighbors 參數限制複雜度

- **風險**: 訓練不收斂
- **緩解**: 先確保基準 LSTM 訓練正常，漸進增加複雜度

### 實現風險
- **風險**: 缺乏現成的庫支援
- **緩解**: 基於現有穩定組件（spatial_coords）進行實現

- **風險**: 新方法效果未知
- **緩解**: 建立完整的基準比較和性能評估機制

## 下一步行動

### 立即可執行
1. 檢查座標格式：確認現有 VD 數據與 spatial_coords 的兼容性
2. 創建核心配置：實現 SocialPoolingConfig 數據結構
3. 距離計算復用：基於 spatial_coords 實現距離矩陣計算

### 開發路線圖
- **Week 1-2**: 核心 Social Pooling 模組實現
- **Week 3**: SocialTrafficModel 包裝器實現
- **Week 4**: 整合測試和性能調優
- **Week 5**: 文檔更新和部署準備

---

**版本**: 1.0 (統一版本)  
**狀態**: 確定實施  
**最後更新**: 2025-01-15

本文檔整合了所有 Social Pooling 相關的設計決策和技術方案，提供統一且可操作的實現指南。