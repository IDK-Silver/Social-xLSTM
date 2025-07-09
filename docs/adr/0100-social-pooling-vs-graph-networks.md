# ADR-0100: Social Pooling vs Graph Networks

**Status**: Accepted  
**Date**: 2025-01-08  
**Updated**: 2025-01-09  

## Context

Social-xLSTM 專案需要處理空間交通預測中的節點互動問題。傳統的時空圖神經網路（ST-GNN）依賴預定義的圖結構（鄰接矩陣），但在實際交通場景中，完整的道路拓撲資訊往往不可用或不準確。

### 現有方法的限制

1. **Graph Neural Networks (GNNs)**:
   - 需要預定義的圖結構
   - 依賴人工設計的鄰接關係
   - 難以處理動態的空間關係
   - 對拓撲錯誤敏感

2. **Attention Mechanisms**:
   - 計算複雜度高 O(N²)
   - 缺乏明確的空間歸納偏差
   - 難以解釋空間關係

3. **Convolutional Networks**:
   - 假設規則的空間結構
   - 不適用於不規則分佈的感測器

## Decision

**選擇 Social Pooling 方法**，基於以下核心原則：

### Social Pooling 設計決策

1. **座標驅動的空間聚合**：
   - 使用感測器地理座標自動發現鄰居
   - 基於歐幾里得距離的空間權重
   - 自適應的空間半徑

2. **無拓撲依賴**：
   - 不需要預定義的圖結構
   - 純粹基於座標的空間關係學習
   - 適應不規則節點分佈

3. **可解釋性**：
   - 明確的空間權重計算
   - 可視化的影響範圍
   - 基於物理距離的直觀理解

### 技術實現策略

```python
# 核心 Social Pooling 概念
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

## Rationale

### 為什麼選擇 Social Pooling

1. **研究創新性**：
   - 填補無拓撲交通預測的空白
   - 結合 Social LSTM 的空間互動思想
   - 適應真實世界的不完整資訊

2. **實用性**：
   - 台灣公路總局數據沒有完整拓撲
   - 感測器分佈不規則
   - 需要自適應的空間學習

3. **可擴展性**：
   - 易於添加新的空間節點
   - 不需要重新設計圖結構
   - 支援動態的空間配置

### 為什麼拒絕 Graph Networks

1. **數據限制**：
   - 缺乏完整的道路拓撲資訊
   - 人工構建鄰接矩陣容易出錯
   - 靜態圖結構不適應動態交通

2. **計算複雜度**：
   - 圖卷積需要複雜的矩陣運算
   - 梯度傳播通過圖結構複雜
   - 模型解釋性差

3. **研究重複性**：
   - 已有大量 ST-GNN 研究
   - 創新價值有限
   - 不符合專案創新目標

## Consequences

### 正面影響

1. **技術創新**：
   - 開創無拓撲交通預測新方向
   - 結合座標資訊的空間學習
   - 為相關研究提供新思路

2. **實用價值**：
   - 適應真實世界的不完整資訊
   - 降低系統部署複雜度
   - 提高模型的泛化能力

3. **研究價值**：
   - 填補研究空白
   - 具有發表潛力
   - 符合學術創新要求

### 負面影響

1. **實現複雜度**：
   - 需要自行實現 Social Pooling
   - 缺乏現成的庫支援
   - 調試和優化困難

2. **不確定性**：
   - 新方法的效果未知
   - 可能不如成熟的 GNN 方法
   - 需要大量實驗驗證

3. **計算開銷**：
   - 距離計算可能較昂貴
   - 需要優化空間查詢
   - 可能需要空間索引

## Implementation Notes

### 優先級
- **P0**: 基本 Social Pooling 實現
- **P1**: 距離計算優化
- **P2**: 空間權重學習
- **P3**: 可視化和分析工具

### 相關 ADR
- ADR-0200: 座標系統選擇（支援 Social Pooling）
- ADR-0101: xLSTM 整合（提供特徵處理）
- ADR-0300: 開發優先級（架構清理後實現）

### 實現狀態
- [ ] 座標系統整合 (依賴 ADR-0200)
- [ ] 基本 Social Pooling 算法
- [ ] 空間權重計算
- [ ] 與 LSTM 的整合
- [ ] 性能優化
- [ ] 實驗驗證

## References

1. Alahi, A., et al. (2016). Social LSTM: Human trajectory prediction in crowded spaces.
2. Li, Y., et al. (2018). Diffusion convolutional recurrent neural network.
3. Wu, Z., et al. (2020). A comprehensive survey on graph neural networks.
4. 專案座標系統文檔: `src/social_xlstm/utils/spatial_coords.py`