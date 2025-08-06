# Social-xLSTM 文檔

本專案文檔採用三層結構，讓 LLM 和開發者都能快速找到所需資訊。

## 📚 文檔結構

### [guides/](guides/) - 使用指南（如何做）
教學和操作說明，幫助你快速上手和使用系統。

- **[quickstart/](guides/quickstart/)** - 快速開始指南
  - [project-setup.md](guides/quickstart/project-setup.md) - 專案設置
  - [first-model.md](guides/quickstart/first-model.md) - 第一個模型
  - [social-pooling-quickstart.md](guides/quickstart/social-pooling-quickstart.md) - Social Pooling 快速入門
  
- **訓練指南**
  - [training-without-sp.md](guides/training-without-sp.md) - 無 Social Pooling 訓練
  - [training-with-sp.md](guides/training-with-sp.md) - 含 Social Pooling 訓練
  
- **工具使用**
  - [utils-guide.md](guides/utils-guide.md) - 工具使用指南

### [concepts/](concepts/) - 概念說明（為什麼）
架構設計、理論基礎和決策記錄。

- **[architecture/](concepts/architecture/)** - 系統架構
  - [data_pipeline.md](concepts/architecture/data_pipeline.md) - 數據管線架構
  - [social_pooling.md](concepts/architecture/social_pooling.md) - Social Pooling 架構
  
- **[papers/](concepts/papers/)** - 相關論文
  - [xlstm-2024.md](concepts/papers/xlstm-2024.md) - xLSTM 論文解析
  - [social-lstm-2016.md](concepts/papers/social-lstm-2016.md) - Social LSTM 論文解析
  
- **[decisions/](concepts/decisions/)** - 架構決策記錄
  - [adr-001-distance-based-social-pooling.md](concepts/decisions/adr-001-distance-based-social-pooling.md)
  
- **技術規範**
  - [mathematical-specifications.md](concepts/mathematical-specifications.md) - 數學規範
  - [datamodule-comparison.md](concepts/datamodule-comparison.md) - DataModule 比較
  - [data-quality-remediation-plan.md](concepts/data-quality-remediation-plan.md) - 數據品質修復計劃

### [reference/](reference/) - API 參考（是什麼）
詳細的 API 文檔和配置參考。

- **API 文檔**
  - [api-reference.md](reference/api-reference.md) - API 參考手冊
  
- **配置指南**
  - [configuration-guide.md](reference/configuration-guide.md) - 配置指南
  - [configuration-reference.md](reference/configuration-reference.md) - 配置參考
  - [data-formats.md](reference/data-formats.md) - 數據格式說明
  
- **工具參考**
  - [tools-overview.md](reference/tools-overview.md) - 工具概覽
  - [analysis-tools.md](reference/analysis-tools.md) - 分析工具
  - [validation-tools.md](reference/validation-tools.md) - 驗證工具
  - [testing-guide.md](reference/testing-guide.md) - 測試指南

## 🔍 快速導航

| 我想要... | 去這裡 |
|-----------|--------|
| 快速開始使用 | [guides/quickstart/](guides/quickstart/) |
| 了解系統架構 | [concepts/architecture/](concepts/architecture/) |
| 查找 API 文檔 | [reference/api-reference.md](reference/api-reference.md) |
| 閱讀相關論文 | [concepts/papers/](concepts/papers/) |
| 設置訓練流程 | [guides/training-*.md](guides/) |
| 理解設計決策 | [concepts/decisions/](concepts/decisions/) |

## 📝 其他文檔

- [PROJECT_STATUS.md](PROJECT_STATUS.md) - 專案狀態追蹤
- `../reports/` - 分析報告（專案根目錄）
- `../maintenance/` - 維護文檔（專案根目錄）