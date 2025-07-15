# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🌐 Language Processing Guidelines

**重要**: 為了最佳的技術準確性，請遵循以下語言處理原則：

1. **理解階段**: 完全理解中文輸入的語境和需求
2. **思考階段**: 用英文進行技術思考、規劃和實作決策
   - 技術術語在英文中更精確
   - 程式碼概念和架構決策用英文思考
   - 與現有英文文檔和程式碼保持一致
3. **回報階段**: 用中文回報結果和說明
   - 保持用戶的語言偏好
   - 技術術語可以保留英文並附中文說明
   - 程式碼和命令保持原文

**範例流程**:
```
用戶中文輸入 → 英文技術思考 → 中文結果回報
"實現LSTM模型" → "implement LSTM model with PyTorch" → "已實現LSTM模型，使用PyTorch框架"
```

這樣可以確保技術準確性的同時保持良好的溝通體驗。

## 🚀 快速開始

**新的 Claude Code 會話**: 請先閱讀 [快速入門指南](docs/QUICK_START.md) 或執行:
```bash
python scripts/utils/claude_init.py --quick
```

這將在 5 分鐘內讓你了解專案狀態並開始工作。

## Project Overview

Social-xLSTM is a traffic prediction system using extended LSTM (xLSTM) models for analyzing spatial-temporal traffic data. The project implements a complete ML pipeline from data preprocessing to model evaluation, with support for both xLSTM and traditional LSTM models.

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment (REQUIRED)
conda env create -f environment.yaml
conda activate social_xlstm

# Install package in development mode (REQUIRED)
pip install -e .
```

**⚠️ 重要**: 所有開發工作都必須在 conda 環境中進行

## 🤖 Claude Code 初始化檢查清單

**重要**: 每次新的 Claude Code 會話開始時，請按順序執行以下檢查：

### 1. 專案狀態快速檢查
```bash
# 檢查專案整體狀態
cat docs/reports/project_status.md

# 檢查當前 ADR 狀態
cat docs/adr/README.md

# 檢查當前待辦事項
cat docs/todo.md
```

### 2. 核心技術決策了解
```bash
# 閱讀核心技術決策
cat docs/adr/0100-social-pooling-vs-graph-networks.md
cat docs/adr/0101-xlstm-vs-traditional-lstm.md
cat docs/adr/0300-next-development-priorities.md
```

### 3. 實施狀態檢查
```bash
# 檢查已完成的架構清理
cat docs/adr/0002-lstm-implementation-unification.md
cat docs/adr/0400-training-script-refactoring.md

# 檢查座標系統實施
cat docs/adr/0200-coordinate-system-selection.md
```

### 4. 程式碼結構理解
```bash
# 查看統一的 LSTM 實現
cat src/social_xlstm/models/lstm.py

# 查看座標系統實現
cat src/social_xlstm/utils/spatial_coords.py

# 查看訓練系統
cat src/social_xlstm/training/trainer.py
```

### 5. 當前開發重點確認
基於 ADR-0100 和 ADR-0101 決策：
- **下一步**: 實現 Social Pooling 算法
- **技術基礎**: 座標系統 + 統一 LSTM 已就緒
- **目標**: 結合 Social Pooling 和 xLSTM 的完整模型

**⚠️ 建議**: 在開始任何開發工作前，請完成上述檢查以確保了解專案當前狀態。

## 🤝 協作原則

### 技術決策流程
- **ADR 優先**: 重大架構變更請先參考相關 ADR 文檔並進行討論
- **充分討論**: 所有技術決策都應考慮對專案長期架構和維護的影響
- **建設性回饋**: 提出替代方案時，解釋技術優勢和與現有系統的整合方式
- **最佳實踐**: 優先採用官方文檔、標準庫和主流社區認可的實施方法

### 程式碼品質標準
- **模塊化設計**: 遵循現有的模組架構和分層設計原則
- **可讀性**: 保持程式碼清晰易懂，變數和函數命名具有描述性
- **註解語言**: 所有程式碼註解使用英文
- **架構一致性**: 新功能必須與現有架構模式保持一致
- **避免技術債**: 拒絕非標準實作方法和臨時解決方案

### 長期視角與專案願景
- **當前重點**: Social Pooling 和 xLSTM 整合（參考 ADR-0100, ADR-0101）
- **結構穩定**: 確保新功能與現有架構一致，避免破壞長期穩定性
- **可維護性**: 考慮代碼的未來擴展性和維護成本
- **技術債管理**: 及時識別和解決技術債，避免累積影響專案發展

### 🚀 快速初始化腳本
```bash
# 使用自動化腳本快速了解專案狀態
python scripts/utils/claude_init.py          # 完整模式
python scripts/utils/claude_init.py --quick  # 快速模式
```

此腳本會自動：
- 📊 顯示專案狀態概覽
- 🏛️ 展示 ADR 決策狀態
- 🎯 說明當前開發重點
- 💻 檢查關鍵檔案存在性
- ⚡ 提供快速開發命令

### Data Processing Pipeline
```bash
# 🚨 重要：開發階段請統一使用開發配置
# Use development configuration (RECOMMENDED for development)
snakemake --configfile cfgs/snakemake/dev.yaml --cores 4

# 生產環境使用預設配置 (僅用於正式實驗)
# Production configuration (for final experiments only)
snakemake --cores 4

# Run individual preprocessing steps
python scripts/dataset/pre-process/list_all_zips.py --input_folder_list <folders> --output_file_path <output>
python scripts/dataset/pre-process/unzip_and_to_json.py --input_zip_list_path <input> --output_folder_path <output> --status_file <status>
python scripts/dataset/pre-process/create_h5_file.py --source_dir <dir> --output_path <path> [--selected_vdids <ids>]
```

### Model Training
```bash
# 無 Social Pooling 的模型訓練（必須在 conda 環境中）
conda activate social_xlstm

# 單VD 訓練（無 Social Pooling）
python scripts/train/without_social_pooling/train_single_vd.py

# 多VD 訓練（獨立VD處理，無 Social Pooling）
python scripts/train/without_social_pooling/train_multi_vd.py

# 🚨 開發階段使用 Snakemake + 開發配置（強烈推薦）
snakemake --configfile=cfgs/snakemake/dev.yaml train_single_vd_without_social_pooling
snakemake --configfile=cfgs/snakemake/dev.yaml train_multi_vd_without_social_pooling

# 生產環境使用預設配置（僅用於正式實驗）
snakemake train_single_vd_without_social_pooling
snakemake train_multi_vd_without_social_pooling

# 訓練參數配置：
# - 開發: cfgs/snakemake/dev.yaml (快速測試，小數據)
# - 生產: cfgs/snakemake/default.yaml (完整實驗)
# 開發輸出: blob/experiments/dev/ | 生產輸出: blob/experiments/
```

### Testing
```bash
# Run all tests
pytest

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest test/test_social_xlstm/dataset/test_json_utils.py
```

### Visualization
```bash
# Plot VD (Vehicle Detector) coordinates
python scripts/utils/plot_vd_point.py --VDListJson <json_file_path>

# Generate all training plots (recommended for development)
python scripts/utils/run_all_plots.py --config dev --timeout 120

# Generate all training plots (production)
python scripts/utils/run_all_plots.py --config default

# Generate plots with specific options
python scripts/utils/run_all_plots.py --config cfgs/snakemake/dev.yaml --cores 2 --sequential
```

## Code Architecture

### Core Package Structure (`src/social_xlstm/`)

**Dataset Module** (`dataset/`) - 重構為結構化子包：
- `config/` - 配置管理
  - `base.py` - TrafficDatasetConfig (資料集配置), TrafficHDF5Config (HDF5轉換配置)
- `core/` - 核心數據操作
  - `processor.py` - TrafficDataProcessor (數據前處理: 歸一化、缺失值處理)
  - `timeseries.py` - TrafficTimeSeries (PyTorch時間序列數據集)
  - `datamodule.py` - TrafficDataModule (PyTorch Lightning數據模組)
- `storage/` - 存儲與持久化
  - `h5_converter.py` - TrafficHDF5Converter (JSON到HDF5轉換), TrafficFeatureExtractor (特徵提取)
  - `h5_reader.py` - TrafficHDF5Reader (HDF5讀取), create_traffic_hdf5, ensure_traffic_hdf5 (工具函數)
  - `feature.py` - TrafficFeature dataclass (交通特徵數據結構)
- `utils/` - 工具函數
  - `json_utils.py` - VDInfo, VDLiveList (JSON數據結構), 車輛檢測器數據處理
  - `xml_utils.py` - VDList_xml_to_Json (XML轉JSON工具)
  - `zip_utils.py` - 壓縮檔案處理 (ZIP/7z格式支援)

**Models Module** (`models/`):
- `lstm.py` - 統一的 LSTM 實現 (TrafficLSTM class) - 支援單VD和多VD模式
- `social_pooling.py` - Social pooling mechanism implementation (待實現)
- `social_xlstm.py` - Social-xLSTM model combining LSTM and Social Pooling (待實現)

**Evaluation Module** (`evaluation/`):
- `evaluator.py` - ModelEvaluator class for computing metrics (MAE, MSE, RMSE, MAPE, R²)

**Utils Module** (`utils/`):
- `convert_coords.py` - Coordinate system conversions
- `graph.py` - Graph processing utilities
- `spatial_coords.py` - Spatial coordinate handling

**Visualization Module** (`visualization/`):
- `model.py` - Model visualization functions

### Data Flow Architecture

1. **Raw Data Ingestion**: ZIP archives containing traffic XML data
2. **Preprocessing Pipeline**: 
   - Extract archives → Convert XML to JSON → Create HDF5 datasets
   - Managed by Snakemake workflow with logging
3. **Dataset Loading**: 
   - TrafficTimeSeries class handles time series windowing (core/timeseries.py)
   - TrafficDataModule provides PyTorch Lightning integration (core/datamodule.py)
   - TrafficDataProcessor handles normalization and missing value processing (core/processor.py)
   - Built-in normalization and missing value handling
4. **Model Training**: 
   - Support for both xLSTM and LSTM architectures
   - GPU acceleration with CUDA support
5. **Evaluation**: 
   - Comprehensive metrics calculation
   - Visualization utilities for results

### Key Design Patterns

**Configuration-Driven Development**:
- `TrafficDatasetConfig` dataclass for dataset parameters
- `config.yaml` for pipeline configuration
- Centralized parameter management

**Data Processing Pipeline**:
- Three-stage preprocessing: list → extract → convert (utils/ 工具)
- HDF5 for efficient storage of large time series data (storage/ 模組)
- TrafficDataProcessor for normalization and missing value handling (core/processor.py)
- Robust error handling and logging

**Model Architecture**:
- `Traffic_xLSTM` wraps the xlstm library's xLSTMBlockStack
- Clean separation between model definition and training logic
- Support for spatial-temporal traffic prediction

## Important Notes

### Directory Structure
- **blob/** - 所有資料處理和實驗輸出的統一目錄
  - `dataset/` - 資料集相關檔案
  - `experiments/` - 模型訓練結果和實驗輸出
  - `status/` - 處理狀態追蹤檔案
- **logs/** - 所有日誌檔案
- **src/** - 原始程式碼 (核心套件)
- **scripts/** - 主要執行腳本 (核心工作流程)
- **tools/** - 開發者工具 (配置、分析、診斷、驗證)
- **tests/** - 測試套件 (單元測試、整合測試、功能測試)
- **notebooks/** - 探索性分析 (本地開發，已在 .gitignore 中)

### Package Structure
- The project uses src/ layout with `social_xlstm` as the main package
- **重要**: 使用 `pip install -e .` 安裝後，直接使用 `social_xlstm.module` 導入
- **不要使用**: `sys.path.insert()` 或相對路徑導入
- **正確導入**: 
  - `from social_xlstm.models.lstm import TrafficLSTM`
  - `from social_xlstm.dataset import TrafficDatasetConfig, TrafficTimeSeries, TrafficDataModule`
  - `from social_xlstm.dataset import TrafficDataProcessor, TrafficHDF5Reader, TrafficFeature`
  - `from social_xlstm.dataset import VDInfo, VDLiveList`
  - `from social_xlstm.dataset.core import TrafficDataProcessor, TrafficTimeSeries, TrafficDataModule`
  - `from social_xlstm.dataset.storage import TrafficHDF5Converter, TrafficHDF5Reader, TrafficFeature`
  - `from social_xlstm.dataset.config import TrafficDatasetConfig, TrafficHDF5Config`
  - `from social_xlstm.dataset.utils import VDInfo, VDLiveList`

### Dependencies
- Requires Python 3.11 with CUDA 12.4 support
- Key libraries: PyTorch, PyTorch Lightning, xlstm, h5py, scikit-learn
- Uses conda for environment management (not pip requirements)

### Data Format
- Input: Traffic data in XML format within ZIP archives
- Processing: JSON intermediate format for flexibility
- Storage: HDF5 for efficient time series data access
- Features: Standard traffic metrics (speed, volume, occupancy, etc.)

### Testing Strategy
- pytest with parallel execution support
- Separate test directories for different components
- GPU functionality tests included

### Workflow Management
- Snakemake handles the complete data processing pipeline
- All operations logged to `logs/` directory
- Configuration-driven approach for reproducibility

## 重要文檔參考

**所有專案文檔都在 `docs/` 目錄下，已完成分類整理**

### 核心技術文檔
- **數學公式定義**: `docs/technical/mathematical_formulation.tex` - 完整的 Social-xLSTM 數學定義
- **座標系統**: `src/social_xlstm/utils/spatial_coords.py` - 完整的座標處理系統（已實現）
- **Social LSTM 分析**: `docs/technical/social_lstm_analysis.md` - 原始論文深度分析
- **專案概述**: `docs/overview/project_overview.md` - 研究目標和創新點
- **ADR 系統**: `docs/adr/` - 架構決策記錄
- **🚨 設計問題記錄**: `docs/technical/design_issues_refactoring.md` - 需要重構的設計問題
- **🚨 已知錯誤記錄**: `docs/technical/known_errors.md` - 多VD訓練潛在錯誤與解決方案

### 當前開發狀態
- **專案狀態**: `docs/reports/project_status.md` - 統一的狀態報告（健康狀況、進度、風險評估）
- **待辦事項**: `docs/todo.md` - 完整的任務追蹤清單
- **專案變更**: `docs/reports/project_changelog.md` - 重要變更記錄
- **關鍵決策**: `docs/overview/key_decisions.md` - 重要決策紀錄

### 使用指南
- **LSTM 使用**: `docs/guides/lstm_usage_guide.md` - 基本的 LSTM 使用指南
- **訓練系統**: `docs/guides/trainer_usage_guide.md` - 統一訓練系統完整使用指南
- **訓練腳本**: `docs/guides/training_scripts_guide.md` - 訓練腳本使用指南
- **模組說明**: `docs/implementation/modules.md` - 各模組功能詳細說明
- **文檔導覽**: `docs/README.md` - 完整的文檔導覽系統

### 當前技術挑戰（按優先級排序）
1. ✅ **架構清理**: 5 個重複LSTM實現統一（ADR-0002 已完成）
2. ✅ **訓練腳本重構**: 減少代碼重複（ADR-0400 已完成）
3. 🚧 **專案重組**: sandbox/ 目錄清理和結構重組（進行中）
4. 📋 **Social Pooling**: 核心算法實現（ADR-0100 已決策，待開發）
5. 📋 **xLSTM 整合**: sLSTM + mLSTM 混合架構（ADR-0101 已決策，待開發）

**✅ 架構清理進展**: 根據 ADR-0300 決策，主要架構清理工作已完成，現可進行核心功能開發。

### 當前優先級任務（核心功能開發）
- **P0**: **Social Pooling 實現**（ADR-0100 座標驅動空間聚合）
- **P1**: **xLSTM 整合**（ADR-0101 sLSTM + mLSTM 混合架構）
- **P2**: **Social-xLSTM 模型**（結合 Social Pooling 和 xLSTM）
- **P3**: **實驗驗證**（效果評估、基準比較）
- **P4**: **期末報告準備**（實驗結果、文檔整理）

**📋 技術基礎已就緒**:
- ✅ 統一的 LSTM 實現 (src/social_xlstm/models/lstm.py)
- ✅ 座標系統支援 (src/social_xlstm/utils/spatial_coords.py)
- ✅ 評估框架 (src/social_xlstm/evaluation/evaluator.py)
- ✅ 訓練系統 (src/social_xlstm/training/trainer.py)

## 架構決策記錄 (ADR)

**重要**: 在進行任何重大架構或技術決策前，請先檢查 `docs/adr/` 目錄中的相關 ADR 文檔。

### 當前 ADR 狀態
- **ADR-0001**: 專案架構清理決策 (部分完成)
- **ADR-0002**: LSTM 實現統一方案 (部分完成) - 5個重複實現已統一為1個
- **ADR-0100**: Social Pooling vs Graph Networks (已批准) - 選擇 Social Pooling 方法
- **ADR-0101**: xLSTM vs Traditional LSTM (已批准) - 選擇 xLSTM 混合架構
- **ADR-0200**: 座標系統選擇 (已實施) - 確認使用現有 spatial_coords.py
- **ADR-0300**: 下一階段開發優先級 (已批准) - Social Pooling 優先
- **ADR-0400**: 訓練腳本重構策略 (已實施) - 減少48%代碼重複

### ADR 查閱指南
1. **開發工作開始前**: 必須先查閱 ADR-0300 確認當前優先級
2. **技術選擇決策**: 參考 ADR-0100 (Social Pooling) 和 ADR-0101 (xLSTM) 的技術選擇
3. **新功能開發前**: 檢查是否有相關的已批准 ADR
4. **座標系統使用**: 參考 ADR-0200 的座標系統決策

**✅ 已完成的架構清理**:
1. ✅ ADR-0002 LSTM 統一方案 (5個實現統一為1個)
2. ✅ ADR-0400 訓練腳本重構 (減少48%代碼重複)
3. 🚧 專案架構重組 (進行中)

**📋 當前開發重點**:
根據 ADR-0100 和 ADR-0101 決策，下一步應該實現：
1. Social Pooling 算法 (基於座標的空間聚合)
2. xLSTM 整合 (sLSTM + mLSTM 混合架構)
3. Social-xLSTM 模型 (結合 Social Pooling 和 xLSTM)

### 使用方式
```bash
# 查看所有 ADR
ls docs/adr/

# 閱讀特定 ADR
cat docs/adr/0002-lstm-implementation-unification.md
```

## ⚙️ 配置管理原則

**🚨 重要：開發階段統一使用 `cfgs/snakemake/dev.yaml`**

### 1. **配置選擇指導**
- **開發/測試階段**: 統一使用 `cfgs/snakemake/dev.yaml`
  - 快速訓練 (2 epochs)
  - 小數據集 (約1小時數據)
  - 輸出到 `blob/experiments/dev/`
- **生產/正式實驗**: 使用 `cfgs/snakemake/default.yaml`
  - 完整訓練 (5+ epochs)  
  - 完整數據集
  - 輸出到 `blob/experiments/`

### 2. **配置同步原則**
**重要**: 新增配置選項時，**必須同時更新兩個配置檔案**：
- `cfgs/snakemake/dev.yaml` (開發配置)
- `cfgs/snakemake/default.yaml` (生產配置)

### 3. **配置修改工作流程**
```bash
# 1. 修改開發配置
vim cfgs/snakemake/dev.yaml

# 2. 同步修改生產配置 (相同結構，不同參數值)
vim cfgs/snakemake/default.yaml

# 3. 更新配置文檔
vim cfgs/README.md

# 4. 測試兩種配置都能正常工作
snakemake --configfile cfgs/snakemake/dev.yaml --dry-run
snakemake --configfile cfgs/snakemake/default.yaml --dry-run
```

### 4. **當前開發標準命令**
```bash
# 訓練 (開發標準)
snakemake --configfile=cfgs/snakemake/dev.yaml train_single_vd_without_social_pooling

# 數據處理 (開發標準) 
snakemake --configfile=cfgs/snakemake/dev.yaml create_h5_file
```

## 🛠️ 開發者工具使用指南

**重要**: 專案已重新組織，將開發者工具從 `scripts/` 移動到 `tools/` 目錄，以便更好地區分核心工作流程和開發者工具。

### Tools 目錄結構
```
tools/
├── config/         # 配置生成和管理工具
├── analysis/       # 數據分析工具
├── diagnostics/    # 診斷和檢查工具
└── validation/     # 驗證工具
```

### 工具使用範例

#### 配置工具
```bash
# 生成最佳化配置
python tools/config/config_generator.py --type optimized --h5_path stable_dataset.h5

# 生成開發配置
python tools/config/config_generator.py --type development --h5_path dataset.h5
```

#### 分析工具
```bash
# 數據品質分析
python tools/analysis/data_quality_analysis.py --input data.h5

# 時間模式分析
python tools/analysis/temporal_pattern_analysis.py --data_path data.h5

# HDF5 數據分析
python tools/analysis/h5_data_analysis.py --file_path data.h5
```

#### 診斷工具
```bash
# HDF5 結構檢查
python tools/diagnostics/h5_structure_inspector.py --input data.h5

# 數據穩定性檢查
python tools/diagnostics/data_stability_tools.py --check stability

# 系統診斷
python tools/diagnostics/diagnostic_tools.py --system_check
```

#### 驗證工具
```bash
# 時間分割驗證
python tools/validation/temporal_split_validation.py --data_path data.h5

# 訓練驗證
python tools/validation/training_validation.py --model_path model.pt

# 過度擬合驗證
python tools/validation/overfitting_validation.py --experiment_path experiments/
```

### 工具開發指南
- **新增工具**: 根據功能分類加入相應的子目錄
- **自包含**: 每個工具都應該可以獨立執行
- **文檔**: 每個工具都應該有 `--help` 選項
- **錯誤處理**: 提供清晰的錯誤訊息

詳細說明請參考 [tools/README.md](tools/README.md)

## 🔧 檔案修改原則

**重要**: 在改進或修改程式碼時，請遵循以下原則：

### 1. **優先修改現有檔案**
- 如果要改進的檔案已存在於 Git 紀錄中，**直接修改該檔案**
- **不要創建新的** "improved"、"new"、"v2" 或類似版本的檔案
- 避免檔案碎片化和版本混亂

### 2. **修改前檢查 Git 狀態**
```bash
# 檢查檔案是否已被修改
git status --porcelain filename.py

# 如果檔案有未提交的變更，詢問使用者
```

### 3. **詢問使用者確認**
如果檔案已被修改過（通過 `git status` 檢查），先詢問使用者：
- 說明將要進行的修改內容
- 詢問是否要繼續修改
- 提供選項：覆蓋、合併、或取消

### 4. **範例對比**
```bash
# ❌ 錯誤做法 - 創建新檔案
create_h5_file_improved.py      # 當 create_h5_file.py 已存在
train_model_v2.py              # 當 train_model.py 已存在
config_new.yaml               # 當 config.yaml 已存在

# ✅ 正確做法 - 直接修改現有檔案
create_h5_file.py             # 直接改進現有實現
train_model.py               # 直接升級現有功能
config.yaml                  # 直接更新配置
```

### 5. **例外情況**
只有在以下情況才創建新檔案：
- **使用者明確要求**創建新檔案
- **需要保留原始版本**作為備份或參考
- **進行 A/B 測試**需要兩個版本並存
- **實驗性功能**需要獨立測試

### 6. **Git 工作流程**
```bash
# 修改前檢查
git status path/to/file.py

# 如果有未提交變更，先了解內容
git diff path/to/file.py

# 進行修改後，檢查差異
git diff path/to/file.py

# 確認修改正確後提交
git add path/to/file.py
git commit -m "Improve file.py: add feature X"
```

這個原則確保：
- **代碼庫整潔**：避免重複和混亂的檔案
- **版本控制清晰**：修改歷史易於追蹤
- **團隊協作順暢**：避免檔案衝突和混淆

## 🧪 開發方法論

### TDD (Test-Driven Development) 原則

**適用於核心功能開發**（如 Social Pooling、xLSTM 整合）：

#### TDD 循環
```
Red (紅) → Green (綠) → Refactor (重構)
```

1. **Red**: 寫一個失敗的測試
2. **Green**: 實現最少代碼讓測試通過
3. **Refactor**: 在測試通過的基礎上改進代碼結構

#### 實施指導
- **測試優先**：先寫測試再寫實現
- **小步迭代**：每次只實現一個小功能
- **有意義的測試名稱**：描述行為而非實現（如 `test_social_pooling_aggregates_nearby_vds`）
- **最少實現**：只寫通過測試所需的最少代碼

### Tidy First 重構原則

**分離兩種變更類型**：

#### 1. 結構性變更（Structural Changes）
- 重命名變數、函數、類別
- 提取方法、移動代碼
- 調整代碼組織結構
- **不改變行為**，只改變代碼結構

#### 2. 行為性變更（Behavioral Changes）
- 新增功能
- 修改現有功能邏輯
- 修復 bug
- **改變系統行為**

#### 最佳實踐
```bash
# 先進行結構性變更
git commit -m "Refactor: extract social pooling calculation method"

# 再進行行為性變更
git commit -m "Add: implement weighted distance calculation in social pooling"
```

### 程式碼品質標準

#### 核心原則
- **消除重複**：DRY (Don't Repeat Yourself)
- **表達意圖**：清楚的命名和結構
- **單一職責**：每個函數/類別只做一件事
- **顯式依賴**：明確表示依賴關係
- **最簡解決方案**：能運作的最簡單實現

#### 重構指導
- **只在測試通過時重構**（Green 階段）
- **一次只做一種重構**
- **每次重構後運行測試**
- **優先移除重複和提高清晰度**

### 提交規範

#### 提交條件
✅ **只在以下情況提交**：
- 所有測試通過
- 無編譯/linter 警告
- 變更代表單一邏輯單元
- 提交訊息清楚說明變更類型

#### 提交訊息格式
```bash
# 結構性變更
git commit -m "Refactor: extract coordinate transformation utility"

# 行為性變更
git commit -m "Add: implement Social Pooling spatial aggregation"

# 測試
git commit -m "Test: add unit tests for Social Pooling algorithm"
```

### 開發工作流程

#### 新功能開發
1. **寫失敗測試**：定義小部分功能的測試
2. **最少實現**：寫最少代碼讓測試通過
3. **運行測試**：確認綠燈
4. **結構改進**：必要時進行 Tidy First 重構
5. **提交結構變更**：單獨提交重構
6. **下一個測試**：為下一個功能增量寫測試
7. **重複循環**：直到功能完成
8. **提交行為變更**：單獨提交功能實現

#### 實際應用建議

**高嚴格度適用場景**：
- Social Pooling 算法實現
- xLSTM 模型整合
- 核心數據處理邏輯
- 評估指標計算

**彈性處理場景**：
- 配置文件調整
- 快速原型驗證
- 文檔更新
- 實驗腳本

**記住**：方法論是工具，要根據專案實際需求靈活應用，確保效率和品質的平衡。