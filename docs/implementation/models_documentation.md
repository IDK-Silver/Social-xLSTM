# Models 模組文檔

## 模組概述

`social_xlstm.models` 模組實現了專案的核心模型架構，包含統一的 LSTM 實現、Social Pooling 機制和完整的 Social-xLSTM 模型。此模組是專案的核心創新部分，實現了基於座標的空間互動學習。

## 核心組件

### 1. TrafficLSTM (`lstm.py`)

**功能**: 統一的交通 LSTM 實現，支援單VD和多VD模式

**設計理念**: 
- 基於 ADR-0002 統一方案，整合了5個重複實現的精華
- 乾淨的架構設計，專業的配置管理
- 可擴展設計，為未來 xLSTM 整合做準備

**核心類別**:
- `TrafficLSTMConfig`: 配置類，管理所有模型參數
- `TrafficLSTM`: 主要的 LSTM 模型實現

**主要功能**:
- 支援單VD模式: `[batch_size, seq_len, features]`
- 支援多VD模式: `[batch_size, seq_len, num_vds, features]`
- 多步預測功能
- 工廠方法快速創建模型
- 完整的模型信息和檢查點管理

**使用場景**:
```python
from social_xlstm.models.lstm import TrafficLSTM

# 單VD模型
single_model = TrafficLSTM.create_single_vd_model(
    input_size=3,
    hidden_size=128,
    num_layers=2
)

# 多VD模型
multi_model = TrafficLSTM.create_multi_vd_model(
    num_vds=5,
    input_size=3,
    hidden_size=128
)
```

**設計評估**:
- ✅ **優點**: 統一設計，消除了重複代碼
- ✅ **優點**: 清晰的配置管理和工廠方法
- ✅ **優點**: 良好的錯誤處理和驗證
- ✅ **優點**: 完整的文檔和示例
- ⚠️ **改進建議**: 多VD模式的輸入尺寸處理可以更靈活

### 2. SocialPoolingLayer (`social_pooling.py`)

**功能**: 核心的 Social Pooling 實現，基於座標的空間關係學習

**設計理念**: 
- 實現 ADR-0100 決策的座標驅動社交池化
- 無需預定義拓撲結構，純粹基於地理位置
- 可擴展的網格劃分機制

**核心類別**:
- `SocialPoolingConfig`: 社交池化配置
- `SocialPoolingLayer`: 主要的社交池化實現

**關鍵創新**:
- **座標驅動**: 使用 GPS 座標自動發現鄰居
- **網格劃分**: 8×8 網格進行空間特徵聚合
- **距離加權**: 基於歐幾里得距離的權重計算
- **多種池化方法**: 支援 mean、max、attention 等

**核心算法流程**:
1. 計算相對座標（目標VD為中心）
2. 將鄰居分配到網格單元
3. 計算距離權重
4. 在每個網格單元內聚合特徵

**使用場景**:
```python
from social_xlstm.models.social_pooling import SocialPoolingLayer, SocialPoolingConfig

# 配置社交池化
config = SocialPoolingConfig(
    grid_size=(8, 8),
    spatial_radius=25000.0,  # 25km
    input_feature_dim=3,
    output_feature_dim=64,
    pooling_method="mean"
)

# 創建層
social_pooling = SocialPoolingLayer(config)

# 使用
pooled_features = social_pooling(
    target_coords,      # [batch_size, 2]
    neighbor_features,  # [batch_size, seq_len, num_neighbors, feature_dim]
    neighbor_coords     # [batch_size, num_neighbors, 2]
)
```

**設計評估**:
- ✅ **優點**: 創新的座標驅動方法
- ✅ **優點**: 不依賴道路拓撲結構
- ✅ **優點**: 可擴展的網格設計
- ✅ **優點**: 整合座標系統模組
- ⚠️ **性能考量**: 座標轉換在每次前向傳播中執行，可能影響效率
- ⚠️ **改進建議**: 可以預計算相對座標以提升性能

### 3. SocialXLSTM (`social_xlstm.py`)

**功能**: 完整的 Social-xLSTM 模型，結合個別VD的xLSTM和社交池化

**設計理念**: 
- 遵循 Social LSTM 原始概念：每個VD有自己的模型
- 社交池化用於隱藏狀態的空間互動
- 分散式預測：每個VD預測自己的未來

**核心類別**:
- `SocialXLSTMConfig`: 整體模型配置
- `VDxLSTM`: 單個VD的xLSTM實現
- `SocialXLSTM`: 完整的社交xLSTM模型

**架構設計**:
1. **個別模型**: 每個VD擁有獨立的xLSTM模型
2. **社交互動**: 隱藏狀態通過社交池化共享空間信息
3. **狀態更新**: 結合原始和社交上下文更新隱藏狀態
4. **分散預測**: 每個VD基於更新後的隱藏狀態進行預測

**關鍵特性**:
- 支援任意數量的VD
- 自動空間鄰居發現
- 多步序列預測
- 豐富的模型信息

**使用場景**:
```python
from social_xlstm.models.social_xlstm import SocialXLSTM

# 定義VD列表和座標
vd_ids = ["VD_001", "VD_002", "VD_003"]
vd_coordinates = {
    "VD_001": (23.8, 120.7),
    "VD_002": (23.82, 120.72),
    "VD_003": (23.85, 120.75)
}

# 創建模型
model = SocialXLSTM.create_from_vd_list(
    vd_ids=vd_ids,
    hidden_size=128,
    num_layers=2
)

# 訓練數據
vd_data = {
    vd_id: torch.randn(batch_size, seq_len, input_size)
    for vd_id in vd_ids
}

# 預測
predictions = model(vd_data, vd_coordinates)
```

**設計評估**:
- ✅ **優點**: 正確實現了 Social LSTM 概念
- ✅ **優點**: 分散式架構適合交通場景
- ✅ **優點**: 靈活的VD管理和座標處理
- ✅ **優點**: 完整的多步預測支援
- ⚠️ **xLSTM 整合**: 目前使用標準LSTM，待整合真正的xLSTM
- ⚠️ **效率考量**: 每個VD獨立模型可能參數較多

## 模組架構分析

### 設計模式
1. **配置驅動**: 所有模型使用配置類管理參數
2. **工廠方法**: 提供便捷的模型創建方法
3. **模組化設計**: 清晰的組件分離和介面定義
4. **可擴展性**: 為未來功能預留擴展空間

### 創新點
1. **座標驅動社交池化**: 無需拓撲結構的空間學習
2. **統一LSTM架構**: 消除重複代碼的標準化實現
3. **分散式預測**: 每個VD獨立預測適合交通場景
4. **空間權重學習**: 自動學習空間影響關係

### 技術債務分析

#### 1. xLSTM 整合待完成
- **現狀**: 目前使用標準LSTM作為佔位符
- **計劃**: 整合真正的xLSTM實現（sLSTM + mLSTM）
- **優先級**: 高（ADR-0101 決策）

#### 2. 性能優化機會
- **問題**: 座標轉換在每次前向傳播中執行
- **解決**: 可以預計算相對座標矩陣
- **優先級**: 中

#### 3. 批次處理改進
- **問題**: 社交池化中的迴圈處理可能影響效率
- **解決**: 向量化操作和更高效的張量操作
- **優先級**: 中

## 整體評估

### 優點
1. **創新設計**: 座標驅動的社交池化是核心創新
2. **架構清晰**: 模組化設計便於維護和擴展
3. **文檔完整**: 豐富的註解和使用示例
4. **配置管理**: 統一的配置系統便於實驗
5. **ADR 實施**: 正確實施了核心技術決策

### 需要改進的問題

#### 1. xLSTM 整合 (P0)
- **狀態**: 待實施
- **ADR**: ADR-0101 已批准
- **時程**: 第二週優先任務

#### 2. 性能優化 (P1)
- **問題**: 座標轉換和批次處理效率
- **解決**: 向量化操作和預計算
- **時程**: 第三週性能調優

#### 3. 測試覆蓋率 (P2)
- **現狀**: 主要是示例代碼
- **需要**: 完整的單元測試和整合測試
- **時程**: 第三週測試提升

## 使用建議

### 開發階段建議
1. **原型開發**: 使用現有LSTM實現進行快速原型
2. **空間調參**: 調整 `spatial_radius` 和 `grid_size` 參數
3. **性能監控**: 關注座標轉換的計算開銷
4. **xLSTM 準備**: 準備整合真正的xLSTM實現

### 生產使用建議
1. **預計算優化**: 對固定VD配置預計算相對座標
2. **批次大小調優**: 根據GPU記憶體調整批次大小
3. **監控指標**: 監控社交池化的空間聚合效果
4. **模型檢查點**: 定期保存模型狀態便於恢復

### 典型工作流程
```python
# 1. 準備VD資料和座標
vd_ids = ["VD_001", "VD_002", "VD_003"]
vd_coordinates = {...}  # GPS座標

# 2. 創建模型
model = SocialXLSTM.create_from_vd_list(vd_ids, hidden_size=128)

# 3. 訓練循環
for epoch in range(num_epochs):
    for batch in dataloader:
        vd_data = prepare_vd_data(batch)
        predictions = model(vd_data, vd_coordinates)
        loss = compute_loss(predictions, targets)
        loss.backward()
        optimizer.step()

# 4. 多步預測
future_predictions = model.predict_sequence(vd_data, vd_coordinates, steps=5)
```

## 總結

Models 模組成功實現了專案的核心創新：座標驅動的社交池化和分散式交通預測。統一的LSTM架構解決了代碼重複問題，為xLSTM整合提供了良好基礎。下一步重點是整合真正的xLSTM實現，並進行性能優化和測試覆蓋率提升。