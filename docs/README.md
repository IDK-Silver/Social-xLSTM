# Social-xLSTM 文檔導覽

本目錄包含 Social-xLSTM 專案的完整文檔，已重新組織為更清晰的結構。

## 📁 目錄結構

### 🎯 overview（專案概覽）
- **[`project_overview.md`](overview/project_overview.md)** - 專案目的、研究問題、核心創新與技術架構概述
- **[`key_decisions.md`](overview/key_decisions.md)** - 重要決策記錄

### 🔬 technical（技術文檔）
**重組為子分類結構** ⭐ **NEW!**

#### 數學公式 (`formulas/`)
- **[`mathematical_formulation.tex`](technical/formulas/mathematical_formulation.tex)** - Social-xLSTM 完整數學公式定義
- **[`social_xlstm_comprehensive_report.tex`](technical/formulas/social_xlstm_comprehensive_report.tex)** - 完整技術報告
- **[`social_xlstm_comprehensive_report.pdf`](technical/formulas/social_xlstm_comprehensive_report.pdf)** - 技術報告PDF版本

#### 技術分析 (`analysis/`)
- **[`social_lstm_analysis.md`](technical/analysis/social_lstm_analysis.md)** - Social LSTM 原始論文深度分析
- **[`h5_structure_analysis.md`](technical/analysis/h5_structure_analysis.md)** - HDF5 數據結構分析
- **[`output_formats_and_parsing.md`](technical/analysis/output_formats_and_parsing.md)** - 輸出格式和解析分析

#### 方法比較 (`comparisons/`)
- **[`social_xlstm_implementation_comparison.md`](technical/comparisons/social_xlstm_implementation_comparison.md)** - Social-xLSTM 實現方法比較 ⭐ **NEW!**
- **[`Social-xLSTM-Internal-Gate-Injection.tex`](technical/comparisons/Social-xLSTM-Internal-Gate-Injection.tex)** - 內部門控注入方法技術規格

#### 設計文檔 (`design/`)
- **[`snakefile_xlstm_design.md`](technical/design/snakefile_xlstm_design.md)** - Snakefile 設計文檔
- **[`vd_batch_processing.md`](technical/design/vd_batch_processing.md)** - VD 批處理設計

#### 問題記錄 (`issues/`)
- **[`design_issues_refactoring.md`](technical/issues/design_issues_refactoring.md)** - 設計問題和重構需求
- **[`known_errors.md`](technical/issues/known_errors.md)** - 已知錯誤和解決方案

#### 數據格式 (`data/`)
- **[`data_format.md`](technical/data/data_format.md)** - 交通資料格式完整說明
- **[`data_quality.md`](technical/data/data_quality.md)** - 資料品質檢查指南

### 🏗️ architecture（架構設計）
- **[`social_xlstm_design.md`](architecture/social_xlstm_design.md)** - Social xLSTM 架構設計文檔
- **[`social_lstm_correct_understanding.md`](architecture/social_lstm_correct_understanding.md)** - Social LSTM 正確理解與實現

### 💻 implementation（實現文檔）
**合併 modules/ 內容** ⭐ **UPDATED!**
- **[`modules.md`](implementation/modules.md)** - 各模組功能詳細說明
- **[`comprehensive_documentation.md`](implementation/comprehensive_documentation.md)** - 綜合文檔
- **[`dataset_documentation.md`](implementation/dataset_documentation.md)** - 數據集文檔
- **[`models_documentation.md`](implementation/models_documentation.md)** - 模型文檔

### 📖 guides（使用指南）
- **[`lstm_usage_guide.md`](guides/lstm_usage_guide.md)** - LSTM 使用指南
- **[`trainer_usage_guide.md`](guides/trainer_usage_guide.md)** - 統一訓練系統使用指南
- **[`training_recorder_guide.md`](guides/training_recorder_guide.md)** - 訓練記錄器指南
- **[`training_scripts_guide.md`](guides/training_scripts_guide.md)** - 訓練腳本指南
- **[`understanding_training_metrics.md`](guides/understanding_training_metrics.md)** - 訓練指標理解指南
- **[`xlstm_usage_guide.md`](guides/xlstm_usage_guide.md)** - xLSTM 使用指南
- **[`troubleshooting_workflow.md`](guides/troubleshooting_workflow.md)** - 問題排除工作流程

### 📋 examples（實作範例）
- **[`multi_vd_output_parsing_examples.md`](examples/multi_vd_output_parsing_examples.md)** - Multi-VD 輸出解析實作範例

### 📊 reports（報告文檔）
- **[`project_status.md`](reports/project_status.md)** - 專案狀態報告
- **[`project_changelog.md`](reports/project_changelog.md)** - 專案變更記錄
- **[`next_session_tasks.md`](reports/next_session_tasks.md)** - 下次會議任務 ⭐ **MOVED!**
- **[`adr_system_update.md`](reports/adr_system_update.md)** - ADR 系統更新報告
- **[`lstm_baseline_completion_summary.md`](reports/lstm_baseline_completion_summary.md)** - LSTM 基線完成摘要
- **[`testing_refactoring_summary.md`](reports/testing_refactoring_summary.md)** - 測試重構摘要
- **[`國立臺南大學資訊工程學系_期中進度報告書.pdf`](reports/國立臺南大學資訊工程學系_期中進度報告書.pdf)** - 期中進度報告

### 🏛️ adr（架構決策記錄）
- **[`README.md`](adr/README.md)** - ADR 系統說明
- **Architecture Decision Records (ADR)** - 記錄重要的技術決策和設計選擇
- 包含決策背景、考慮選項、選擇理由和後果分析

## 🔗 快速導航

### 🚀 新手入門
1. **專案概覽** → [`overview/project_overview.md`](overview/project_overview.md)
2. **快速開始** → [`QUICK_START.md`](QUICK_START.md)
3. **數據格式** → [`technical/data/data_format.md`](technical/data/data_format.md)
4. **基本使用** → [`guides/lstm_usage_guide.md`](guides/lstm_usage_guide.md)

### 🔬 研究開發
1. **數學基礎** → [`technical/formulas/mathematical_formulation.tex`](technical/formulas/mathematical_formulation.tex)
2. **Social LSTM 分析** → [`technical/analysis/social_lstm_analysis.md`](technical/analysis/social_lstm_analysis.md)
3. **方法比較** → [`technical/comparisons/social_xlstm_implementation_comparison.md`](technical/comparisons/social_xlstm_implementation_comparison.md)
4. **架構決策** → [`adr/README.md`](adr/README.md)

### 📐 技術實現
1. **模組說明** → [`implementation/modules.md`](implementation/modules.md)
2. **訓練系統** → [`guides/trainer_usage_guide.md`](guides/trainer_usage_guide.md)
3. **問題排除** → [`guides/troubleshooting_workflow.md`](guides/troubleshooting_workflow.md)
4. **已知問題** → [`technical/issues/known_errors.md`](technical/issues/known_errors.md)

### 📚 學術參考
1. **期中報告** → [`reports/國立臺南大學資訊工程學系_期中進度報告書.pdf`](reports/國立臺南大學資訊工程學系_期中進度報告書.pdf)
2. **技術報告** → [`technical/formulas/social_xlstm_comprehensive_report.pdf`](technical/formulas/social_xlstm_comprehensive_report.pdf)
3. **專案狀態** → [`reports/project_status.md`](reports/project_status.md)

## 🔍 快速查找表

| 需求 | 推薦文檔 |
|------|----------|
| 理解 Social LSTM 原理 | [`technical/analysis/social_lstm_analysis.md`](technical/analysis/social_lstm_analysis.md) |
| 選擇實現方法 | [`technical/comparisons/social_xlstm_implementation_comparison.md`](technical/comparisons/social_xlstm_implementation_comparison.md) |
| 數學公式參考 | [`technical/formulas/mathematical_formulation.tex`](technical/formulas/mathematical_formulation.tex) |
| 數據格式規範 | [`technical/data/data_format.md`](technical/data/data_format.md) |
| 訓練系統使用 | [`guides/trainer_usage_guide.md`](guides/trainer_usage_guide.md) |
| 問題排除 | [`guides/troubleshooting_workflow.md`](guides/troubleshooting_workflow.md) |
| 已知問題查詢 | [`technical/issues/known_errors.md`](technical/issues/known_errors.md) |
| 架構決策背景 | [`adr/README.md`](adr/README.md) |

## 📝 文檔維護

### 文檔規範
- 所有 Markdown 文檔使用繁體中文撰寫
- LaTeX 文檔包含完整的數學公式與英文術語
- 文檔間的交叉引用使用相對路徑
- 定期更新以反映最新的專案進展

### 更新流程
1. **概覽更新**：重大變更時更新 `overview/`
2. **技術更新**：模型改進時更新 `technical/`
3. **實現更新**：代碼變更時更新 `implementation/`
4. **報告新增**：里程碑完成時新增到 `reports/`

## 🔄 最近更新

### 2025-01-14 重大重組
- ✅ **技術文檔重組**：將 `technical/` 分為 6 個子分類
- ✅ **清理編譯產物**：移除 .aux、.log 等臨時文件
- ✅ **合併重複目錄**：將 `modules/` 合併到 `implementation/`
- ✅ **文檔重新分類**：移動文檔到更合適的位置
- ✅ **新增方法比較**：詳細的 Social-xLSTM 實現方法比較文檔
- ✅ **更新導航系統**：提供更清晰的文檔查找路徑

### 文檔結構優化
- 📁 **子分類結構**：technical/ 現在有 formulas/, analysis/, comparisons/ 等子目錄
- 🔗 **改善導航**：快速查找表和分類導航
- 📋 **待辦管理**：移動到根級別便於追蹤

---

**注意**: 文檔結構已於 2025-01-14 重新組織。舊的文檔路徑可能已改變，請參考新的目錄結構或使用快速查找表。