# Social-xLSTM 文檔系統

**版本**: 2.0  
**最後更新**: 2025-08-01  
**狀態**: 基於快速入門指南重建，確保準確性

## 📚 文檔導覽

### 🚀 [快速開始](quickstart/)
**5分鐘快速上手 Social-xLSTM**
- [Social Pooling 快速入門](quickstart/social-pooling-quickstart.md) - 核心概念和實現
- [專案快速設置](quickstart/project-setup.md) - 環境配置和依賴安裝
- [第一個模型](quickstart/first-model.md) - 建立和訓練你的第一個模型

### 📖 [使用指南](guides/)
**詳細的功能使用指南**
- [完整訓練指南](guides/training-guide.md) - 從數據到模型的完整流程
- [模型配置指南](guides/model-configuration.md) - LSTM vs xLSTM 配置
- [Social Pooling 進階用法](guides/social-pooling-advanced.md) - 高級配置和調優
- [評估和可視化](guides/evaluation-visualization.md) - 模型性能分析

### 📋 [技術參考](reference/)
**完整的 API 和技術規範**
- [API 參考](reference/api-reference.md) - 完整的類和函數文檔
- [配置參考](reference/configuration-reference.md) - 所有配置選項詳解
- [數據格式規範](reference/data-formats.md) - 輸入輸出格式定義
- [性能基準](reference/benchmarks.md) - 模型性能對比

### 🔧 [技術細節](technical/)
**深入的技術實現細節**
- [架構設計原理](technical/architecture-design.md) - 系統架構和設計決策
- [Social Pooling 算法](technical/social-pooling-algorithm.md) - 數學原理和實現
- [xLSTM vs LSTM 對比](technical/xlstm-lstm-comparison.md) - 技術差異分析
- [座標系統實現](technical/coordinate-system.md) - 空間計算和投影

## 🎯 學習路徑推薦

### 新用戶（第一次使用）
1. [Social Pooling 快速入門](quickstart/social-pooling-quickstart.md)
2. [專案快速設置](quickstart/project-setup.md)
3. [第一個模型](quickstart/first-model.md)

### 開發者（需要集成）
1. [完整訓練指南](guides/training-guide.md)
2. [模型配置指南](guides/model-configuration.md)
3. [API 參考](reference/api-reference.md)

### 研究者（需要深入理解）
1. [架構設計原理](technical/architecture-design.md)
2. [Social Pooling 算法](technical/social-pooling-algorithm.md)
3. [xLSTM vs LSTM 對比](technical/xlstm-lstm-comparison.md)

## 🆕 版本 2.0 更新重點

### 基於快速入門指南重建
- ✅ **準確性驗證**: 所有文檔與實際程式碼保持一致
- ✅ **分散式架構**: 正確描述每個 VD 獨立 recurrent core 的架構
- ✅ **xLSTM 整合**: 完整覆蓋 sLSTM + mLSTM 混合架構
- ✅ **實際可用**: 所有程式碼範例都經過驗證

### 結構簡化
- **4個主要目錄**: quickstart, guides, reference, technical
- **清晰學習路徑**: 從入門到精通的漸進式引導
- **內容整合**: 消除重複，統一術語

### 用戶導向設計
- **任務導向**: 基於用戶實際需求組織內容
- **範例豐富**: 每個概念都有完整的程式碼範例
- **即時可用**: 複製貼上即可運行的程式碼

## 🔄 從舊版遷移

如果你正在使用舊版文檔系統（`docs/`），請參考以下對應關係：

```
舊版 docs/                          → 新版 docs_rebuild/
├── getting-started/                → quickstart/
├── explanation/ + how-to/          → guides/
├── reference/                      → reference/
├── technical/                      → technical/
└── 分散的配置文檔                   → guides/model-configuration.md
```

## 📞 支援和貢獻

### 問題回報
- 文檔錯誤：請在 GitHub Issues 中標記 `documentation`
- 程式碼問題：請在 GitHub Issues 中標記 `bug`
- 功能請求：請在 GitHub Issues 中標記 `enhancement`

### 文檔維護原則
1. **Single Source of Truth**: 快速入門指南是所有文檔的基準
2. **Code-First**: 程式碼變更時，文檔必須同步更新
3. **Example-Driven**: 每個概念都必須有可運行的範例
4. **User-Focused**: 優先解決用戶實際問題

---

**注意**: 舊版 `docs/` 目錄仍可參考，但建議使用新版 `docs_rebuild/` 系統以獲得最準確和最新的資訊。