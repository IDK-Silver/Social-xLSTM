# ADR-0600: Social Pooling Integration Strategy

**狀態**: Accepted  
**日期**: 2025-07-14  
**決策者**: Technical Team  
**技術故事**: Social-xLSTM 核心架構設計

## 背景與問題陳述

在 Social-xLSTM 架構中，Social Pooling 機制需要與 xLSTM (sLSTM + mLSTM) 進行整合以實現空間-時間序列預測。目前有兩種主要的整合策略需要評估：

1. **Post-Fusion**: 在 xLSTM 模型輸出後進行 Social Pooling 聚合
2. **Internal Gate Injection (IGI)**: 直接在 xLSTM 門控計算過程中注入 Social 信息

兩種方法各有技術優勢，需要確定實施策略和對應的軟體架構設計。

## 決策驅動因素

- **計算效率**: 不同方法的計算複雜度和資源消耗
- **模型表達能力**: 空間-時間信息融合的深度和效果
- **實施複雜度**: 開發和維護的技術難度
- **擴展性**: 支援不同場景和參數配置的靈活性
- **研究價值**: 兩種方法的學術貢獻和比較研究價值

## 考慮的選項

### 選項 1: Post-Fusion 單一實現

- **優點**:
  - 實施簡單，模組化清晰
  - xLSTM 和 Social Pooling 獨立開發
  - 計算邏輯直觀易懂
  - 調試和優化相對容易
- **缺點**:
  - 空間信息融合較淺層
  - 可能錯失深度整合的模型表達優勢
  - 研究貢獻相對有限

### 選項 2: Internal Gate Injection 單一實現

- **優點**:
  - 深度整合，理論表達能力更強
  - 創新性較高，研究價值大
  - 空間信息在序列建模過程中持續影響
- **缺點**:
  - 實施複雜度很高
  - 調試困難，bug 定位不易
  - 與標準 xLSTM 兼容性問題

### 選項 3: 雙重實現策略

- **優點**:
  - 提供完整的比較研究基礎
  - 適應不同使用場景需求
  - 最大化研究和實用價值
  - 為後續論文發表提供豐富實驗對比
- **缺點**:
  - 開發工作量增加一倍
  - 維護成本較高
  - 需要設計統一的接口架構

## 決策結果

**選擇**: 雙重實現策略 (選項 3)

**理由**: 
1. **研究完整性**: 兩種方法的對比研究具有重要學術價值
2. **實用性**: 不同場景下可選擇最適合的實現方式
3. **技術探索**: 深入理解 Social Pooling 與序列建模的不同整合模式
4. **期末報告**: 為完整的實驗對比和性能分析提供基礎

## 實施細節 (重新設計)

### ⚠️ 架構問題與重新設計

**原設計問題分析**:
1. **破壞現有結構**: 提議創建新的 Social-xLSTM 類別會與現有 `TrafficXLSTM` 產生概念衝突
2. **忽略 LSTM 需求**: 用戶明確需要測試 LSTM + Social Pooling，原設計主要關注 xLSTM
3. **複雜化依賴**: 多層類別架構會讓簡潔的現有結構變複雜

**現有架構優勢**:
- 簡潔清晰: `TrafficLSTM` + `TrafficXLSTM`
- 接口統一: 相似的 config, forward, get_model_info 接口
- Multi-VD 支援: 兩個模型都已支援多VD模式
- 擴展性設計: LSTM 註解中提到 "Extensible for future xLSTM integration"

### 📋 新架構設計: 組合模式 (Non-Breaking)

#### 1. 保持現有模型完全不變
```python
# 現有模型保持不變
TrafficLSTM         # 繼續用於基準測試
TrafficXLSTM        # 繼續用於 xLSTM 基準測試
```

#### 2. 獨立 Social Pooling 模組
```python
from dataclasses import dataclass
from typing import Tuple
import torch.nn as nn

@dataclass
class SocialPoolingConfig:
    """Social Pooling 配置"""
    grid_size: Tuple[int, int] = (10, 10)
    radius: float = 1000.0  # meters
    aggregation_method: str = "weighted_mean"  # "mean", "weighted_mean", "attention"
    enable_social_features: bool = True
    social_embedding_dim: int = 64
    
class SocialPooling(nn.Module):
    """獨立的 Social Pooling 模組"""
    def __init__(self, config: SocialPoolingConfig):
        super().__init__()
        self.config = config
        # Social pooling implementation
    
    def forward(self, hidden_states: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """計算空間社會特徵"""
        pass
```

#### 3. 組合包裝器支援兩種策略
```python
from typing import Union
import torch
import torch.nn as nn

class SocialTrafficModel(nn.Module):
    """Social 增強的交通預測模型包裝器"""
    
    def __init__(self, 
                 base_model: Union[TrafficLSTM, TrafficXLSTM],
                 social_pooling: SocialPooling,
                 strategy: str = "post_fusion"):
        super().__init__()
        self.base_model = base_model
        self.social_pooling = social_pooling
        self.strategy = strategy  # "post_fusion" or "internal_injection"
        
        # 根據策略初始化不同組件
        if strategy == "post_fusion":
            self._init_post_fusion()
        elif strategy == "internal_injection":
            self._init_internal_injection()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _init_post_fusion(self):
        """初始化 Post-Fusion 策略組件"""
        # 在基礎模型輸出後融合 social features
        self.fusion_layer = nn.Linear(
            self.base_model.config.output_size + self.social_pooling.config.social_embedding_dim,
            self.base_model.config.output_size
        )
    
    def _init_internal_injection(self):
        """初始化 Internal Gate Injection 策略組件"""
        # 修改基礎模型的內部計算，注入 social features
        if isinstance(self.base_model, TrafficLSTM):
            self._modify_lstm_gates()
        elif isinstance(self.base_model, TrafficXLSTM):
            self._modify_xlstm_blocks()
    
    def forward(self, x: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """統一的前向傳播接口"""
        if self.strategy == "post_fusion":
            return self._forward_post_fusion(x, coordinates)
        else:
            return self._forward_internal_injection(x, coordinates)
```

#### 4. 工廠函數簡化創建
```python
from typing import Union
from social_xlstm.models.lstm import TrafficLSTM, TrafficLSTMConfig
from social_xlstm.models.xlstm import TrafficXLSTM, TrafficXLSTMConfig

def create_social_traffic_model(
    base_model_type: str,  # "lstm" or "xlstm"
    strategy: str,         # "post_fusion" or "internal_injection"
    base_config: Union[TrafficLSTMConfig, TrafficXLSTMConfig],
    social_config: SocialPoolingConfig
) -> SocialTrafficModel:
    """工廠函數創建 Social 增強模型"""
    
    # 創建基礎模型
    if base_model_type == "lstm":
        base_model = TrafficLSTM(base_config)
    elif base_model_type == "xlstm":
        base_model = TrafficXLSTM(base_config)
    else:
        raise ValueError(f"Unknown base_model_type: {base_model_type}")
    
    # 創建 social pooling
    social_pooling = SocialPooling(social_config)
    
    # 創建組合模型
    return SocialTrafficModel(base_model, social_pooling, strategy)
```

### 📁 檔案結構 (基於現有組件的模組化設計)

#### ✅ **可直接復用的現有組件**
```
src/social_xlstm/utils/
├── spatial_coords.py             # ✅ 完整的座標系統 (437 lines)
│                                 #    - CoordinateSystem 類別
│                                 #    - 距離計算、方位角、座標轉換
│                                 #    - 墨卡托投影、工廠方法
└── graph.py                      # ✅ VD 座標可視化工具

src/social_xlstm/training/
└── with_social_pooling/          # ✅ 預留目錄，有清楚的 TODO 規劃
    └── __init__.py
```

#### 🆕 **需要新增的模組化結構**
```
src/social_xlstm/models/
├── __init__.py                   # 添加新的 exports
├── lstm.py                       # 不變
├── xlstm.py                      # 不變
└── social/                       # 新增模組化目錄
    ├── __init__.py
    ├── pooling/                  # Social Pooling 核心
    │   ├── __init__.py
    │   ├── config.py             # SocialPoolingConfig
    │   ├── base.py               # SocialPooling 基類
    │   ├── grid.py               # 網格構建 (復用 spatial_coords)
    │   ├── distance.py           # 距離權重 (復用 spatial_coords)
    │   └── aggregation.py        # 特徵聚合機制
    ├── strategies/               # 整合策略
    │   ├── __init__.py
    │   ├── base.py               # 策略基類
    │   ├── post_fusion.py        # Post-Fusion 實現
    │   └── internal_injection.py # IGI 實現
    ├── wrappers/                 # 模型包裝器
    │   ├── __init__.py
    │   ├── social_model.py       # SocialTrafficModel 主類
    │   ├── lstm_injection.py     # LSTM 門控修改邏輯
    │   └── xlstm_injection.py    # xLSTM 塊修改邏輯
    └── factory.py                # 工廠函數 create_social_traffic_model
```

#### 🔄 **復用策略**
```python
# grid.py - 復用現有座標系統
from typing import Tuple, List
from social_xlstm.utils.spatial_coords import CoordinateSystem

class SpatialGrid:
    def __init__(self, grid_size: Tuple[int, int], bounds: Tuple[float, float, float, float]):
        self.coord_system = CoordinateSystem()  # 復用現有實現
        
    def build_grid(self, vd_coordinates: List[Tuple[float, float]]):
        # 直接使用 CoordinateSystem.calculate_distance_from_latlon
        # 避免重複實現距離計算
        pass

# distance.py - 復用距離計算
import torch
from social_xlstm.utils.spatial_coords import CoordinateSystem

def calculate_spatial_weights(coordinates: torch.Tensor, radius: float):
    # 使用現有的 CoordinateSystem.calculate_distance_from_xy
    # 避免重新實現距離計算邏輯
    pass
```

### 🧪 支援的測試組合
```python
# 1. LSTM + Post-Fusion Social Pooling
lstm_social_post = create_social_traffic_model("lstm", "post_fusion", lstm_config, social_config)

# 2. LSTM + Internal Gate Injection  
lstm_social_igi = create_social_traffic_model("lstm", "internal_injection", lstm_config, social_config)

# 3. xLSTM + Post-Fusion Social Pooling
xlstm_social_post = create_social_traffic_model("xlstm", "post_fusion", xlstm_config, social_config)

# 4. xLSTM + Internal Gate Injection
xlstm_social_igi = create_social_traffic_model("xlstm", "internal_injection", xlstm_config, social_config)

# 5. 原始基準模型 (不變)
lstm_baseline = TrafficLSTM(lstm_config)
xlstm_baseline = TrafficXLSTM(xlstm_config)
```

## 後果

### 正面後果 (重新設計後)

- **保護現有投資**: TrafficLSTM 和 TrafficXLSTM 完全不受影響
- **支援完整比較**: 6種模型組合 (2基礎模型 × 2策略 + 2基準)
- **最小化風險**: 組合模式降低代碼耦合
- **靈活擴展**: 未來可輕易添加新的 Social Pooling 策略
- **測試需求滿足**: 同時支援 LSTM 和 xLSTM 的 Social 版本

### 負面後果 (風險降低)

- **開發時間適中**: 約增加 30% (相比原 50%)
- **維護複雜度可控**: 組合模式比繼承模式更易維護
- **測試範圍擴大**: 需要測試 6 種模型組合

### 風險與緩解措施 (更新)

- **風險1**: Internal Injection 實現複雜 / **緩解**: 先實現 Post-Fusion，IGI 階段性實現
- **風險2**: 組合模式性能開銷 / **緩解**: 基準測試驗證，必要時優化
- **風險3**: 接口複雜化 / **緩解**: 工廠函數提供簡潔的創建接口

## 相關決策

- [ADR-0100: Social Pooling vs Graph Networks](0100-social-pooling-vs-graph-networks.md)
- [ADR-0101: xLSTM vs Traditional LSTM](0101-xlstm-vs-traditional-lstm.md)
- [ADR-0200: Coordinate System Selection](0200-coordinate-system-selection.md)

## 註記

### 開發優先級 (更新計劃)
1. **Phase 1**: 實現 SocialPooling 獨立模組 (1-2 週)
2. **Phase 2**: 實現 Post-Fusion 策略包裝器 (1-2 週)
3. **Phase 3**: 實現 Internal Gate Injection 策略 (2-3 週)
4. **Phase 4**: 完整測試和效能比較 (1 週)

### 實現路徑 (基於現有組件)
```python
# Week 1: 復用現有組件，實現 Social Pooling 核心
# ✅ 直接復用 utils/spatial_coords.py (437 lines)
# 🆕 實現 social/pooling/ 模組 (約200 lines，大幅減少)
#     - config.py: SocialPoolingConfig
#     - base.py: SocialPooling 基類
#     - grid.py: 網格構建 (復用 CoordinateSystem)
#     - aggregation.py: 特徵聚合

# Week 2: Post-Fusion 策略實現  
# 🆕 social/strategies/post_fusion.py (約100 lines)
# 🆕 social/wrappers/social_model.py (基礎版本)
# 🆕 social/factory.py (工廠函數)

# Week 3-4: Internal Gate Injection 策略
# 🆕 social/strategies/internal_injection.py (約150 lines)
# 🆕 social/wrappers/lstm_injection.py (約100 lines)
# 🆕 social/wrappers/xlstm_injection.py (約100 lines)

# Week 5: 整合測試和 training/ 模組實現
# 🔄 更新 training/with_social_pooling/ (復用現有架構)
# 🧪 完整的 6 模型比較測試
# 📊 效能基準和論文實驗準備
```

### 📊 **開發工作量評估 (大幅減少)**
- **原評估**: ~800-900 lines 新代碼
- **基於復用**: ~550-650 lines 新代碼 (**減少約 30%**)
- **關鍵節省**: 座標系統 (437 lines) 完全復用
- **額外優勢**: 已測試穩定的座標計算邏輯

### 架構設計原則
- **Non-Breaking**: 現有代碼完全不受影響
- **組合優於繼承**: 使用包裝器而非擴展現有類別
- **統一接口**: 工廠函數提供一致的創建體驗
- **漸進實現**: 可分階段實現不同策略

### 測試策略確認
✅ **支援的測試組合**:
1. `TrafficLSTM` (基準)
2. `TrafficXLSTM` (基準)  
3. `LSTM + Post-Fusion Social Pooling`
4. `LSTM + Internal Gate Injection`
5. `xLSTM + Post-Fusion Social Pooling`
6. `xLSTM + Internal Gate Injection`

這個重新設計的方案完全滿足了用戶的需求：保護現有結構、支援 LSTM 測試、提供兩種 Social Pooling 策略。