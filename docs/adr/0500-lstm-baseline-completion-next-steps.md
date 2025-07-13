# ADR-0500: LSTM 基準系統完成與下一步規劃

**狀態**: 批准  
**日期**: 2025-07-13  
**決策者**: 開發團隊  

## 📋 背景

經過完整的 LSTM 基準系統建立，我們已成功實現並驗證了三種不同的 LSTM 架構：
- Single VD LSTM (226,309 參數)
- Multi-VD LSTM (1,433,537 參數)  
- Independent Multi-VD LSTM (1,421,317 參數)

所有模型都配備完整的訓練流程、評估指標和視覺化系統。現在需要明確下一階段的開發重點。

## 🎯 已完成成果

### 1. 完整的 LSTM 基準系統
- ✅ 三種 LSTM 架構實現並完成訓練
- ✅ 統一的訓練系統 (src/social_xlstm/training/)
- ✅ 完整的評估框架 (src/social_xlstm/evaluation/)
- ✅ 自動化 Snakemake 工作流程

### 2. 完整的視覺化與分析系統
- ✅ training_curves.png - 基本訓練曲線
- ✅ metric_evolution.png - 評估指標演變
- ✅ advanced_metrics.png - 進階分析圖表
- ✅ understanding_training_metrics.md - 指標解讀文檔

### 3. 關鍵發現
從訓練結果分析：
- **Single VD**: 嚴重過擬合 (R² 從 0.93 降到 -6)
- **Multi-VD**: 複雜度增加，性能略有改善
- **Independent Multi-VD**: 相似表現但架構更簡潔

## 🚀 決策：下一階段重點

根據 ADR-0100 (Social Pooling) 和 ADR-0101 (xLSTM) 的技術決策，以及當前已建立的完整 LSTM 基準，決定優先順序如下：

### P0 - 核心 Social-xLSTM 實現 (接下來 2 週)

#### 1. Social Pooling 算法實現
**基於 ADR-0100 決策**：
- 實現座標驅動的空間聚合機制
- 利用現有的 spatial_coords.py 座標系統
- 設計網格劃分和鄰近 VD 聚合算法
- 建立 Social Pooling 層的單元測試

#### 2. xLSTM 架構整合
**基於 ADR-0101 決策**：
- 實現 sLSTM (scalar memory + exponential gating)
- 實現 mLSTM (matrix memory)
- 建立 Hybrid 架構 (sLSTM + mLSTM)
- 與現有 TrafficLSTM 保持兼容性

#### 3. Social-xLSTM 模型骨架
- 創建 SocialTrafficXLSTM 類
- 整合 Social Pooling + xLSTM
- 設計前向傳播流程
- 建立完整的模型初始化

### P1 - 實驗驗證與比較 (第 3 週)

#### 1. Social-xLSTM vs LSTM 基準比較
- 使用相同的數據集和評估指標
- 對比三種 LSTM 基準與 Social-xLSTM
- 分析空間聚合的效果
- 評估 xLSTM 的性能提升

#### 2. 超參數調優
- Social Pooling 網格大小調優
- xLSTM 隱藏層大小調優
- 學習率和正規化參數調優

### P2 - 品質提升與文檔化 (第 4 週)

#### 1. 測試覆蓋率提升
- Social Pooling 單元測試
- xLSTM 架構測試
- 整合測試和性能基準

#### 2. 期末報告準備
- 實驗結果分析和比較
- 技術文檔更新
- 論文撰寫準備

## 🛠️ 技術實施策略

### 1. 漸進式開發
- 先實現 Social Pooling，再整合 xLSTM
- 每個組件完成後立即進行單元測試
- 保持與現有 LSTM 系統的兼容性

### 2. 基於現有基礎
- 複用統一的 TrafficLSTM 架構
- 利用現有的訓練和評估框架
- 擴展現有的 Snakemake 工作流程

### 3. 持續驗證
- 每個階段完成後與 LSTM 基準比較
- 使用相同的視覺化和評估系統
- 確保性能提升可衡量

## 📊 成功指標

### 第一週結束目標
- [ ] Social Pooling 算法實現完成
- [ ] 基本的 xLSTM (sLSTM) 實現
- [ ] Social-xLSTM 模型能夠初始化和載入

### 第二週結束目標
- [ ] 完整的 Social-xLSTM 實現
- [ ] 能夠在小數據集上訓練
- [ ] 初步性能驗證完成

### 第三週結束目標
- [ ] Social-xLSTM vs LSTM 完整比較
- [ ] 所有實驗結果文檔化
- [ ] 測試覆蓋率 > 70%

## 🔗 相關 ADR

- ADR-0100: Social Pooling vs Graph Networks
- ADR-0101: xLSTM vs Traditional LSTM  
- ADR-0200: 座標系統選擇
- ADR-0300: 下一階段開發優先級

## 💡 風險與緩解

### 風險 1: xLSTM 實現複雜度
**緩解**: 先實現 sLSTM，再逐步添加 mLSTM 和 Hybrid 架構

### 風險 2: Social Pooling 空間聚合效果不明顯
**緩解**: 設計多種網格大小和聚合策略進行對比

### 風險 3: 整合複雜度過高
**緩解**: 保持模組化設計，確保各組件可獨立測試

## 📝 結論

LSTM 基準系統的完整建立為 Social-xLSTM 研究提供了堅實的基礎。下一階段將專注於核心的 Social Pooling 和 xLSTM 實現，最終目標是建立完整的 Social-xLSTM 模型並與 LSTM 基準進行系統性比較。

**批准日期**: 2025-07-13  
**實施開始**: 立即開始