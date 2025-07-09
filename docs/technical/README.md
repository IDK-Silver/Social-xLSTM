# Technical Documentation

本目錄包含 Social-xLSTM 專案的技術文檔，涵蓋理論基礎、算法分析和數學公式。

## 📚 文檔清單

### 🔬 核心技術分析
- **[`social_lstm_analysis.md`](social_lstm_analysis.md)** - **Social LSTM 原始論文深度分析**
  - 基於 Alahi et al. (CVPR 2016) 的詳細解讀
  - 架構設計原理與實現細節
  - 數學公式推導與實驗結果分析
  - 與我們專案的關聯性分析

### 📐 數學公式定義
- **[`mathematical_formulation.tex`](mathematical_formulation.tex)** - Social-xLSTM 完整數學公式定義
  - Social Pooling 機制數學描述
  - xLSTM (sLSTM/mLSTM) 公式推導
  - 混合架構與複雜度分析

### 🗺️ 座標系統
- **座標處理實現** - 詳見 `src/social_xlstm/utils/spatial_coords.py`
  - 墨卡托投影轉換
  - 空間距離計算
  - GPS 座標處理

## 🎯 閱讀建議

### 理解 Social LSTM 機制
1. 先閱讀 [`social_lstm_analysis.md`](social_lstm_analysis.md) 理解原始論文
2. 對照 [`mathematical_formulation.tex`](mathematical_formulation.tex) 中的數學公式
3. 參考 [`../architecture/social_lstm_correct_understanding.md`](../architecture/social_lstm_correct_understanding.md) 了解正確實現

### 實現技術細節
1. 查看 `src/social_xlstm/utils/spatial_coords.py` 了解空間處理
2. 結合 [`../implementation/modules.md`](../implementation/modules.md) 了解具體實現
3. 參考 [`../guides/lstm_usage_guide.md`](../guides/lstm_usage_guide.md) 學習使用方法

## 🔗 相關文檔

- **架構設計**: [`../architecture/`](../architecture/) - 系統架構設計
- **實現文檔**: [`../implementation/`](../implementation/) - 具體實現細節
- **使用指南**: [`../guides/`](../guides/) - 使用方法與範例

## 📝 更新記錄

- **2025-01-08**: 新增 `social_lstm_analysis.md` - Social LSTM 原始論文分析
- **2025-01-08**: 完善文檔結構和交叉引用

---

**注意**: 這些技術文檔需要一定的深度學習和數學基礎。建議從 [`social_lstm_analysis.md`](social_lstm_analysis.md) 開始，逐步深入理解。