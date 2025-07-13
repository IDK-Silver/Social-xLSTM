# Social-xLSTM 文檔導覽

本目錄包含 Social-xLSTM 專案的完整文檔，按主題分類組織。

## 📁 目錄結構

### overview（專案概覽）
- **`project_overview.md`** - 專案目的、研究問題、核心創新與技術架構概述

### technical（技術文檔）
- **`mathematical_formulation.tex`** - Social-xLSTM 完整數學公式定義
  - Social Pooling 機制數學描述
  - xLSTM (sLSTM/mLSTM) 公式推導
  - 混合架構與複雜度分析
- **`social_lstm_analysis.md`** - **Social LSTM 原始論文深度分析**
  - 原始論文 (Alahi et al., CVPR 2016) 詳細解讀
  - 架構設計原理與實現細節
  - 數學公式推導與實驗結果分析
- **`output_formats_and_parsing.md`** - **模型輸出格式與解析技術文檔** ⭐ **NEW!**
  - Single VD vs Multi-VD 輸出格式說明
  - 扁平化輸出的技術原理與設計權衡
  - TrafficLSTM 解析方法使用指南
  - 錯誤處理與性能優化建議

### architecture（架構設計）
- **`social_xlstm_design.md`** - Social xLSTM 架構設計文檔
- **`social_lstm_correct_understanding.md`** - Social LSTM 正確理解與實現

### implementation（實現文檔）
- **`modules.md`** - 各模組功能詳細說明
  - 核心套件結構
  - 數據處理管線
  - 模型架構實現
  - 測試框架

### guides（使用指南）
- **`lstm_usage_guide.md`** - **LSTM 使用指南**
  - 基本的模型使用和評估
  - 模型創建和參數配置
- **`trainer_usage_guide.md`** - **統一訓練系統使用指南**
  - 完整的訓練系統使用方法
  - 進階配置和超參數調優
  - 實際範例和常見問題解答

### examples（實作範例）
- **`multi_vd_output_parsing_examples.md`** - **Multi-VD 輸出解析實作範例** ⭐ **NEW!**
  - 基本解析和VD提取使用方法
  - 實際應用場景：模型評估、實時預測、批次處理
  - 進階應用：特徵分析、空間相關性分析
  - 錯誤處理與性能優化範例

### 📊 資料格式與品質（**NEW!**）
- **`data_format.md`** - **交通資料格式完整說明**
  - JSON 資料結構詳解 (VDList/VDLiveList)
  - 資料路徑解析 (LinkFlows → Lanes)
  - 錯誤碼定義與處理 (-99, 255 等)
  - 特徵定義與計算方式
- **`data_quality.md`** - **資料品質檢查指南**
  - 品質標準與指標
  - 常見問題與解決方案
  - 自動化檢查工具使用
  - 品質改善建議

### reports（報告文檔）
- **`project_status.md`** - 專案狀態報告（健康狀況、進度、風險評估）
- **`project_changelog.md`** - 專案變更記錄
- **`國立臺南大學資訊工程學系_期中進度報告書.pdf`** - 期中進度報告

### adr（架構決策記錄）
- **Architecture Decision Records (ADR)** - 記錄重要的技術決策和設計選擇
- 包含決策背景、考慮選項、選擇理由和後果分析

## 🔗 快速導航

### 新手入門
1. 首先閱讀 → [`overview/project_overview.md`](overview/project_overview.md)
2. **了解資料** → [`data_format.md`](data_format.md) ← **建議先看**
3. 了解實現 → [`implementation/modules.md`](implementation/modules.md)
4. 開始訓練 → [`guides/trainer_usage_guide.md`](guides/trainer_usage_guide.md)

### 研究開發
1. 數學基礎 → [`technical/mathematical_formulation.tex`](technical/mathematical_formulation.tex)
2. **Social LSTM 分析** → [`technical/social_lstm_analysis.md`](technical/social_lstm_analysis.md)
3. 架構決策 → [`adr/README.md`](adr/README.md)
4. 專案狀況 → [`reports/project_status.md`](reports/project_status.md)

### 學術參考
1. 期中報告 → [`reports/國立臺南大學資訊工程學系_期中進度報告書.pdf`](reports/國立臺南大學資訊工程學系_期中進度報告書.pdf)
2. 技術細節 → [`technical/mathematical_formulation.tex`](technical/mathematical_formulation.tex)

## 📝 文檔維護

- 所有 Markdown 文檔使用繁體中文撰寫
- LaTeX 文檔包含完整的數學公式與英文術語
- 文檔間的交叉引用使用相對路徑
- 定期更新以反映最新的專案進展

## 🔄 文檔更新流程

1. **概覽更新**：重大變更時更新 `overview/`
2. **技術更新**：模型改進時更新 `technical/`
3. **實現更新**：代碼變更時更新 `implementation/`
4. **報告新增**：里程碑完成時新增到 `reports/`