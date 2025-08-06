# ADR-001: Distance-Based Social Pooling Implementation

## Status
✅ **Accepted** - 已實現並運行中

## Context

原始 Social LSTM 論文 (Alahi et al., 2016) 使用基於網格的空間離散化進行社交池化。對於交通預測應用，我們需要在網格基礎和距離基礎方法之間做出選擇。

### 背景問題
1. **空間表示選擇**：如何最有效地建模 VD (Vehicle Detector) 之間的空間關係？
2. **計算效率**：不同方法的計算複雜度和性能影響
3. **交通場景適配性**：哪種方法更適合真實世界的交通流預測？

## Decision

**實施距離基礎的連續社交池化，而非網格基礎的離散化方法。**

### 核心實現
```python
# 距離基礎聚合 (我們的實現)
distance = torch.norm(target_pos - neighbor_pos, p=2, dim=-1)
within_radius = distance <= radius
weights = 1.0 / (distance + eps)  # 反距離權重
social_context = weighted_aggregation(neighbor_hidden_states, weights)
```

## Rationale

### 支持距離基礎方法的理由

1. **交通特性對齊**
   - 交通流在空間中連續流動，無自然網格結構
   - VD 感測器位於不規則地理位置
   - 高速公路/城市網絡的拓撲不遵循規整網格

2. **現代最佳實踐**
   - 當代軌跡預測模型普遍使用連續空間表示
   - Social-GAN, Trajectron++, Sophie 等先進模型均採用距離基礎方法
   - 避免網格邊界處的離散化人工痕跡

3. **計算優勢**
   - 稀疏交通場景下更高效 (大多數實際情況)
   - 無需處理稀疏張量操作
   - 自然支援可變交互範圍

4. **超參數直觀性**
   - 交互半徑 R 比網格解析度 N_o 更直觀
   - 直接對應物理距離概念
   - 便於基於領域知識調整

5. **表示質量**
   - 保持完整空間解析度
   - 訓練穩定性更好的平滑梯度
   - 距離權重反映自然影響衰減

## Consequences

### 正面影響 ✅

- **更好的交通適配性**：連續表示更符合交通流的物理特性
- **計算效率提升**：稀疏場景下避免大量零元素運算
- **參數調整簡化**：半徑比網格大小更容易理解和設定
- **梯度品質改善**：連續函數提供更穩定的訓練過程
- **擴展性增強**：支援任意交互距離，無解析度限制

### 負面影響 ⚠️

- **與原始論文偏離**：研究比較時需要特別說明
- **不同的超參數空間**：半徑調整 vs 網格大小調整的經驗不同
- **文檔需求增加**：需要清楚解釋架構選擇給期望網格方法的研究者
- **複雜度變化**：從 O(N_o²) 變為 O(N²) 或 O(N×k) with 半徑截斷

### 風險緩解

- **文檔化**：詳細記錄差異和原理 (本 ADR + 架構文檔)
- **基準測試**：與網格方法進行性能比較驗證
- **參數指導**：提供半徑設定的最佳實踐建議
- **可配置性**：保留未來增加網格選項的可能性

## Implementation Details

### 核心演算法位置
- **實現檔案**: `src/social_xlstm/pooling/xlstm_pooling.py`
- **主要函數**: `xlstm_hidden_states_aggregation()`
- **類別包裝**: `XLSTMSocialPoolingLayer`

### 關鍵配置參數
```python
{
    "radius": 2.0,              # 空間交互半徑 (公尺)
    "pool_type": "mean",        # 聚合方式: mean/max/weighted_mean
    "learnable_radius": False   # 是否將半徑設為可學習參數
}
```

### 數學規格
```python
# 距離計算
d_ij = ||pos_i - pos_j||_2

# 鄰居選擇
N_i = {j : d_ij ≤ R, j ≠ i}

# 權重計算 (加權平均模式)
w_ij = 1 / (d_ij + ε)

# 社交上下文聚合
social_context_i = Σ_{j∈N_i} w_ij * h_j / Σ_{j∈N_i} w_ij
```

## Performance Considerations

### 計算複雜度
- **理論複雜度**: O(N²) 對所有 VD 對
- **實際複雜度**: O(N×k) 其中 k 是平均鄰居數 (≪ N)
- **最佳化策略**: 半徑截斷大幅減少實際計算量

### 記憶體使用
- **距離矩陣**: O(N²) 但可動態計算避免存儲
- **隱狀態存儲**: O(N×H) 與網格方法相同
- **鄰居索引**: O(N×k) 稀疏結構

## Future Considerations

### 可能的擴展
1. **可學習半徑**: 將半徑設為模型參數，自動學習最佳交互範圍
2. **多尺度池化**: 同時使用多個半徑進行分層交互建模
3. **注意力機制**: 將距離權重替換為學習的注意力權重
4. **稀疏最佳化**: 使用 KD-Tree 等資料結構加速鄰居搜索

### 研究方向
- **實證比較**: 系統性比較距離 vs 網格方法在不同交通場景的表現
- **敏感度分析**: 半徑參數對模型性能的影響分析
- **可解釋性**: 分析學習到的空間交互模式

## References

### 相關文檔
- [架構設計文檔](../architecture/social_pooling.md#11-implementation-approach-distance-based-vs-grid-based-social-pooling)
- [數學規格文檔](../technical/mathematical-specifications.md)
- [原始論文整理](../papers/social-lstm-2016.md)

### 核心論文
- Alahi, A., et al. (2016). Social LSTM: Human Trajectory Prediction in Crowded Spaces. CVPR.
- Beck, M., et al. (2024). xLSTM: Extended Long Short-Term Memory. NeurIPS.

### 現代參考
- Gupta, A., et al. (2018). Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks. CVPR.
- Salzmann, T., et al. (2020). Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data. ECCV.

---

**決策日期**: 2025-08-03  
**決策者**: Social-xLSTM 開發團隊  
**下次審查**: 初步實證結果可用時  
**相關 ADR**: 無 (首個架構決策記錄)