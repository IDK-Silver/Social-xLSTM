# 核心技術決策

本文檔整合了 Social-xLSTM 專案的關鍵架構和技術決策，為開發者提供清晰的技術方向指引。

## 決策概覽

Social-xLSTM 專案的核心創新圍繞兩個關鍵選擇：
1. **Social Pooling** - 無拓撲依賴的空間建模方法
2. **xLSTM** - 擴展記憶容量的時序建模架構

## 核心技術決策

### 1. 空間建模方法：Social Pooling vs Graph Networks

**決策**: 選擇 Social Pooling 作為空間互動建模方法

**背景**:
傳統交通預測依賴預定義的道路拓撲圖結構，但在實際應用中，完整的道路拓撲資訊往往不可用或不準確。台灣公路總局的交通資料缺乏完整拓撲資訊，感測器分佈不規則。

**核心原理**:
```python
def social_pooling(node_features, coordinates, radius):
    # 1. 計算距離矩陣（基於座標）
    distances = compute_distance_matrix(coordinates)
    
    # 2. 生成空間權重（高斯核函數）
    spatial_weights = gaussian_kernel(distances, radius)
    
    # 3. 加權聚合鄰居特徵
    pooled_features = weighted_aggregation(node_features, spatial_weights)
    
    return pooled_features
```

**優勢**:
- **無拓撲依賴**: 純粹基於座標的空間關係學習
- **適應不規則分佈**: 感測器位置不規則時仍能有效工作
- **動態擴展**: 易於添加新的空間節點
- **可解釋性**: 基於物理距離的直觀理解

**技術基礎**: 座標驅動的空間聚合，使用歐幾里得距離和高斯權重函數

---

### 2. 時序模型選擇：xLSTM vs Traditional LSTM

**決策**: 選擇 xLSTM (Extended LSTM) 作為核心時序模型

**背景**:
傳統 LSTM 在時間序列預測中表現穩定，但記憶容量有限，難以處理長序列依賴。xLSTM 提供更強的建模能力和更大的記憶容量。

**架構設計**:
```python
class SocialXLSTM(nn.Module):
    def __init__(self, config):
        # sLSTM 處理單變量時間序列
        self.temporal_encoder = sLSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers
        )
        
        # mLSTM 處理多變量空間特徵
        self.spatial_encoder = mLSTM(
            input_size=config.spatial_size,
            hidden_size=config.hidden_size
        )
```

**優勢**:
- **混合架構**: sLSTM + mLSTM 分別處理時序和空間特徵
- **指數門控**: 提供更大的記憶容量
- **長期依賴**: 更好的長序列建模能力
- **研究創新**: 結合最新的 LSTM 擴展技術

**技術基礎**: xlstm 庫提供的 sLSTM (scalar LSTM) 和 mLSTM (matrix LSTM)

---

### 3. 座標系統選擇：墨卡托投影 vs UTM

**決策**: 使用現有的墨卡托投影座標系統

**背景**:
Social Pooling 需要準確的平面座標系統進行距離計算和網格劃分。專案中已有完整的座標處理實現。

**技術規格**:
- **投影方式**: 墨卡托投影，以台灣南投為原點
- **實現位置**: `src/social_xlstm/utils/spatial_coords.py`
- **功能**: 雙向轉換、距離計算、方位角計算
- **優勢**: 已測試穩定，立即可用

**選擇理由**:
- **完整實現**: 437 行代碼，功能全面
- **適合台灣**: 以南投為原點，適合台灣地區
- **即用性**: 無需額外開發，降低實施風險
- **精度足夠**: 對於交通預測應用精度充足

---

### 4. 實施架構：包裝器模式 vs 深度整合

**決策**: 採用包裝器模式 + Post-Fusion 策略

**背景**:
需要將 Social Pooling 與現有的 LSTM/xLSTM 模型整合，考慮實施複雜度和風險控制。

**架構設計**:
```python
class SocialTrafficModel(nn.Module):
    def __init__(self, base_model: TrafficLSTM, social_pooling: SocialPooling):
        super().__init__()
        self.base_model = base_model      # 復用現有模型
        self.social_pooling = social_pooling
        self.fusion_layer = nn.Linear(...)
    
    def forward(self, x, coordinates, vd_ids):
        # 1. 基礎模型處理
        base_output = self.base_model(x)
        
        # 2. Social Pooling
        social_features = self.social_pooling(x, coordinates, vd_ids)
        
        # 3. 特徵融合
        fused = torch.cat([base_output, social_features], dim=-1)
        return self.fusion_layer(fused)
```

**優勢**:
- **風險控制**: 不修改現有穩定代碼
- **代碼復用**: 最大化利用現有 900+ 行穩定實現
- **漸進實現**: 可分階段驗證和優化
- **維護性**: 清晰的模組化設計

---

### 5. 評估策略：雙資料集架構

**決策**: 支援 METR-LA 與台灣交通資料的統一評估

**背景**:
為了驗證方法的通用性和進行國際比較，需要支援多種交通資料集。

**架構優勢**:
- **國際比較**: METR-LA 是標準基準資料集
- **本土驗證**: 台灣資料證明實際應用價值
- **方法驗證**: 證明無拓撲方法的通用性
- **研究價值**: 提供完整的評估基礎

## 實施優先級

### P0: 核心功能實現
1. **Social Pooling 算法** - 基於現有座標系統實現
2. **包裝器模式整合** - 與 TrafficLSTM 整合
3. **基本測試驗證** - 確保功能正確性

### P1: 擴展功能
1. **xLSTM 模型整合** - 替換 LSTM 為 xLSTM
2. **多 VD 支援** - 支援多車道檢測器場景
3. **性能優化** - 記憶體和計算效率優化

### P2: 驗證評估
1. **基準比較** - 與傳統方法比較
2. **多資料集驗證** - METR-LA 和台灣資料
3. **研究分析** - 詳細的效果分析報告

## 技術基礎設施

### 已就緒的組件
- ✅ **座標系統**: `spatial_coords.py` (437 行，功能完整)
- ✅ **統一 LSTM**: `lstm.py` (477 行，穩定)
- ✅ **評估框架**: `evaluator.py` (完善)
- ✅ **訓練系統**: `trainer.py` (統一)
- ✅ **數據管線**: 完整的 XML → JSON → HDF5 處理

### 待實現的組件
- 🚧 **Social Pooling**: 核心空間聚合算法
- 🚧 **xLSTM 整合**: sLSTM + mLSTM 混合架構
- 🚧 **Social-xLSTM**: 完整的整合模型

## 風險評估與緩解

### 技術風險
1. **新方法不確定性**: Social Pooling 效果未知
   - **緩解**: 建立完整的基準比較和評估機制

2. **計算複雜度**: 距離計算可能昂貴
   - **緩解**: 提供 max_neighbors 參數限制，使用快取機制

3. **記憶體使用**: xLSTM 記憶體需求較高
   - **緩解**: 批次大小調優，梯度累積策略

### 實施風險
1. **座標數據相容性**: VD 座標格式不匹配
   - **緩解**: 實施前驗證現有數據格式

2. **模型訓練穩定性**: 新架構可能不收斂
   - **緩解**: 先確保基準模型正常，漸進增加複雜度

## 成功指標

### 技術指標
- **準確性**: MAE/RMSE 比基準 LSTM 改善 > 5%
- **效率**: 記憶體使用增長 < 50%，訓練時間增長 < 30%
- **穩定性**: 模型穩定收斂，無過擬合現象

### 研究指標
- **創新性**: 無拓撲交通預測新方法
- **通用性**: 支援多種資料集和場景
- **實用性**: 適應真實世界的不完整資訊

## 相關文獻與依據

- **Social LSTM**: Alahi, A., et al. (2016). Social LSTM: Human trajectory prediction in crowded spaces.
- **xLSTM**: Beck, M., et al. (2024). xLSTM: Extended Long Short-Term Memory.
- **Traffic Prediction**: Li, Y., et al. (2018). Diffusion convolutional recurrent neural network.

---

**決策版本**: 1.0  
**最後更新**: 2025-01-15  
**狀態**: 已確認實施

*本文檔整合了專案的核心技術決策，為開發提供統一的技術方向指引。詳細的決策過程和背景資料保存在原始 ADR 文檔中。*