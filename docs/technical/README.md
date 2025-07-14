# Technical Documentation

本目錄包含 Social-xLSTM 專案的技術文檔，現已重組為更清晰的結構。

## 📚 文檔結構

### 🔬 數學公式 (`formulas/`)
- **[`mathematical_formulation.tex`](formulas/mathematical_formulation.tex)** - Social-xLSTM 完整數學公式定義
- **[`social_xlstm_comprehensive_report.tex`](formulas/social_xlstm_comprehensive_report.tex)** - 完整技術報告
- **[`social_xlstm_comprehensive_report.pdf`](formulas/social_xlstm_comprehensive_report.pdf)** - 技術報告PDF版本

### 📊 技術分析 (`analysis/`)
- **[`social_lstm_analysis.md`](analysis/social_lstm_analysis.md)** - Social LSTM 原始論文深度分析
- **[`h5_structure_analysis.md`](analysis/h5_structure_analysis.md)** - HDF5 數據結構分析
- **[`output_formats_and_parsing.md`](analysis/output_formats_and_parsing.md)** - 輸出格式和解析分析

### 🔄 方法比較 (`comparisons/`)
- **[`social_xlstm_implementation_comparison.md`](comparisons/social_xlstm_implementation_comparison.md)** - Social-xLSTM 實現方法比較
- **[`Social-xLSTM-Internal-Gate-Injection.tex`](comparisons/Social-xLSTM-Internal-Gate-Injection.tex)** - 內部門控注入方法技術規格

### 🏗️ 設計文檔 (`design/`)
- **[`snakefile_xlstm_design.md`](design/snakefile_xlstm_design.md)** - Snakefile 設計文檔
- **[`vd_batch_processing.md`](design/vd_batch_processing.md)** - VD 批處理設計

### ⚠️ 問題和錯誤 (`issues/`)
- **[`design_issues_refactoring.md`](issues/design_issues_refactoring.md)** - 設計問題和重構需求
- **[`known_errors.md`](issues/known_errors.md)** - 已知錯誤和解決方案

### 📋 數據格式 (`data/`)
- **[`data_format.md`](data/data_format.md)** - 數據格式規範
- **[`data_quality.md`](data/data_quality.md)** - 數據質量評估

## 🎯 閱讀建議

### 理解 Social LSTM 機制
1. 先閱讀 [`analysis/social_lstm_analysis.md`](analysis/social_lstm_analysis.md) 理解原始論文
2. 對照 [`formulas/mathematical_formulation.tex`](formulas/mathematical_formulation.tex) 中的數學公式
3. 參考 [`../architecture/social_lstm_correct_understanding.md`](../architecture/social_lstm_correct_understanding.md) 了解正確實現

### 選擇實現方法
1. 閱讀 [`comparisons/social_xlstm_implementation_comparison.md`](comparisons/social_xlstm_implementation_comparison.md) 了解兩種實現方法
2. 根據決策樹選擇適合的實現方式
3. 參考分階段實施策略進行開發

### 技術實現細節
1. 查看 `src/social_xlstm/utils/spatial_coords.py` 了解空間處理
2. 結合 [`../implementation/modules.md`](../implementation/modules.md) 了解具體實現
3. 參考 [`../guides/lstm_usage_guide.md`](../guides/lstm_usage_guide.md) 學習使用方法

### 數據處理
1. 閱讀 [`data/data_format.md`](data/data_format.md) 了解數據格式
2. 查看 [`analysis/h5_structure_analysis.md`](analysis/h5_structure_analysis.md) 理解存儲結構
3. 參考 [`data/data_quality.md`](data/data_quality.md) 進行質量評估

## 🔗 相關文檔

- **架構設計**: [`../architecture/`](../architecture/) - 系統架構設計
- **實現文檔**: [`../implementation/`](../implementation/) - 具體實現細節
- **使用指南**: [`../guides/`](../guides/) - 使用方法與範例
- **ADR 記錄**: [`../adr/`](../adr/) - 架構決策記錄

## 📝 更新記錄

- **2025-01-14**: 重組技術文檔目錄結構，建立子分類
- **2025-01-14**: 新增 `social_xlstm_implementation_comparison.md` - Social-xLSTM 實現方法比較分析
- **2025-01-14**: 清理編譯產物，移動數據格式文檔
- **2025-01-08**: 新增 `social_lstm_analysis.md` - Social LSTM 原始論文分析
- **2025-01-08**: 完善文檔結構和交叉引用

## 🔍 快速查找

| 需求 | 推薦文檔 |
|------|----------|
| 理解 Social LSTM 原理 | [`analysis/social_lstm_analysis.md`](analysis/social_lstm_analysis.md) |
| 選擇實現方法 | [`comparisons/social_xlstm_implementation_comparison.md`](comparisons/social_xlstm_implementation_comparison.md) |
| 數學公式參考 | [`formulas/mathematical_formulation.tex`](formulas/mathematical_formulation.tex) |
| 數據格式規範 | [`data/data_format.md`](data/data_format.md) |
| 已知問題查詢 | [`issues/known_errors.md`](issues/known_errors.md) |
| 設計決策背景 | [`design/`](design/) 目錄 |

---

**注意**: 這些技術文檔需要一定的深度學習和數學基礎。建議從 [`analysis/social_lstm_analysis.md`](analysis/social_lstm_analysis.md) 開始，逐步深入理解。

**結構重組**: 本目錄已於 2025-01-14 重新組織，提供更清晰的分類結構。舊的文檔路徑可能已改變，請參考新的目錄結構。