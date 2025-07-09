# ADR-0101: xLSTM vs Traditional LSTM

**Status**: Accepted  
**Date**: 2025-01-08  
**Updated**: 2025-01-09  

## Context

Social-xLSTM 專案需要選擇適合的時序模型作為核心架構。傳統 LSTM 在時間序列預測中表現穩定，但新興的 xLSTM (Extended LSTM) 提供了更強的建模能力和更大的記憶容量。

### 技術背景

1. **Traditional LSTM**:
   - 成熟穩定的時序模型
   - 廣泛的社群支援和文獻
   - 計算效率高，容易優化
   - 但記憶容量有限，難以處理長序列

2. **xLSTM (Extended LSTM)**:
   - 包含 sLSTM (scalar LSTM) 和 mLSTM (matrix LSTM)
   - 指數門控機制提供更大記憶容量
   - 更好的長期依賴建模
   - 但計算複雜度更高，較新的技術

## Decision

**選擇 xLSTM 作為核心時序模型**，基於以下考量：

### 混合 xLSTM 架構

```python
class SocialXLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用 sLSTM 處理單變量時間序列
        self.temporal_encoder = sLSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers
        )
        
        # 使用 mLSTM 處理多變量空間特徵
        self.spatial_encoder = mLSTM(
            input_size=config.spatial_size,
            hidden_size=config.hidden_size,
            num_layers=config.spatial_layers
        )
        
        # Social Pooling 整合
        self.social_pooling = SocialPooling(
            hidden_size=config.hidden_size,
            spatial_radius=config.spatial_radius
        )
```

### 技術選擇理由

1. **sLSTM 用於時間序列**：
   - 標量門控適合單變量時間序列
   - 指數門控提供更好的長期記憶
   - 適合交通數據的時間模式

2. **mLSTM 用於空間特徵**：
   - 矩陣門控適合多變量空間特徵
   - 更大的記憶容量處理複雜空間關係
   - 支援 Social Pooling 的空間聚合

3. **混合架構優勢**：
   - 時空特徵的專門化處理
   - 更好的模型表達能力
   - 符合 Social-xLSTM 的創新定位

## Rationale

### 為什麼選擇 xLSTM

1. **技術創新性**：
   - xLSTM 是 2024 年的新技術
   - 結合 xLSTM 和 Social Pooling 具有創新性
   - 為研究貢獻提供技術亮點

2. **建模能力**：
   - 指數門控機制提供更大記憶容量
   - 更好的長期依賴建模
   - 適合複雜的交通時空模式

3. **研究價值**：
   - 探索新技術在交通預測中的應用
   - 為 xLSTM 提供實際應用案例
   - 增強論文的技術貢獻

### 為什麼不選擇 Traditional LSTM

1. **創新性不足**：
   - 傳統 LSTM 已被廣泛研究
   - 缺乏技術創新點
   - 不符合專案創新目標

2. **能力限制**：
   - 記憶容量有限
   - 難以處理長期依賴
   - 可能無法充分利用 Social Pooling

3. **研究影響**：
   - 技術貢獻有限
   - 發表價值較低
   - 缺乏新穎性

## Consequences

### 正面影響

1. **技術優勢**：
   - 更強的時序建模能力
   - 更大的記憶容量
   - 更好的長期依賴處理

2. **研究價值**：
   - 技術創新性高
   - 學術貢獻明顯
   - 發表潛力大

3. **實用價值**：
   - 可能獲得更好的預測效果
   - 適合複雜的交通場景
   - 為未來研究奠定基礎

### 負面影響

1. **實現複雜度**：
   - 需要整合 xlstm 庫
   - 調試和優化困難
   - 缺乏充分的文檔

2. **計算開銷**：
   - 比傳統 LSTM 計算量大
   - 需要更多 GPU 記憶體
   - 訓練時間可能更長

3. **穩定性風險**：
   - 新技術穩定性未知
   - 可能存在未發現的問題
   - 調試困難

4. **學習成本**：
   - 團隊需要學習新技術
   - 缺乏現成的最佳實踐
   - 可能影響開發進度

## Implementation Strategy

### 分階段實現

1. **Phase 1: 基礎整合**
   - 整合 xlstm 庫
   - 實現基本的 sLSTM 和 mLSTM
   - 確保與現有架構兼容

2. **Phase 2: 混合架構**
   - 實現 sLSTM + mLSTM 混合
   - 整合 Social Pooling
   - 優化計算效率

3. **Phase 3: 性能優化**
   - 調優超參數
   - 優化記憶體使用
   - 提升訓練穩定性

### 風險緩解

1. **技術風險**：
   - 保持 Traditional LSTM 作為備選
   - 實現 A/B 測試框架
   - 建立性能基準

2. **時間風險**：
   - 設定實現時間上限
   - 準備回退到 LSTM 的方案
   - 並行開發兩種方案

3. **效果風險**：
   - 建立全面的評估指標
   - 對比多種基準模型
   - 進行統計顯著性測試

## Implementation Notes

### 技術依賴
- **xlstm 庫**: 提供 sLSTM 和 mLSTM 實現
- **PyTorch**: 深度學習框架
- **CUDA**: GPU 加速支援

### 性能考量
- **記憶體使用**: mLSTM 需要更多記憶體
- **計算效率**: 可能需要混合精度訓練
- **批次大小**: 可能需要調整以適應 GPU 記憶體

### 相關 ADR
- ADR-0100: Social Pooling 實現（提供空間特徵）
- ADR-0200: 座標系統（支援空間計算）
- ADR-0002: LSTM 統一（整合不同 LSTM 實現）

### 實現狀態
- [x] xlstm 庫整合 (已在 environment.yaml 中)
- [ ] 基本 sLSTM 實現
- [ ] 基本 mLSTM 實現
- [ ] 混合架構設計
- [ ] Social Pooling 整合
- [ ] 性能優化
- [ ] 效果評估

## References

1. Beck, M., et al. (2024). xLSTM: Extended Long Short-Term Memory.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
3. xlstm 庫文檔: https://github.com/NX-AI/xlstm
4. 專案 LSTM 實現: `src/social_xlstm/models/lstm.py`