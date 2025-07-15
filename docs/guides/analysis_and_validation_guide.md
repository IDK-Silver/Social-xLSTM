# Analysis and Validation Tools Guide

## 概述

本指南詳細說明 Social-xLSTM 專案中的分析和驗證工具的使用方法。這些工具在 ADR-0500 重新組織後，按功能分類到不同目錄中。

## 目錄結構

```
scripts/
├── analysis/          # 數據分析工具
├── validation/        # 驗證測試工具
└── utils/            # 通用工具
```

## 分析工具 (`scripts/analysis/`)

### 1. 時間模式分析 (`temporal_pattern_analysis.py`)

**用途**：深度分析交通數據中的時間模式，調查為什麼時間切分不能改善分佈一致性。

**主要功能**：
- 分析數據隨時間的完整性
- 檢測數值分佈隨時間的漂移
- 系統性模式檢測
- 數據生成問題調查

**使用方法**：
```bash
# 基本使用
python scripts/analysis/temporal_pattern_analysis.py

# 默認分析的數據集
# blob/dataset/pre-processed/h5/traffic_features_default.h5
```

**輸出**：
- 控制台報告：數據完整性、分佈漂移、系統性模式
- 可視化圖表：`blob/debug/temporal_completeness_analysis.png`
- 分佈分析：`blob/debug/temporal_distribution_drift.png`

**重要參數**：
- `chunk_size = 500`：時間塊大小（約8小時數據）
- `period_size = total_samples // 3`：時期劃分大小
- `sequence_length = 10`：重複序列檢測長度

### 2. HDF5 數據分析 (`h5_data_analysis.py`)

**用途**：針對 HDF5 格式的交通數據進行深度分析。

**主要功能**：
- 實際 H5 數據格式分析
- 每個 VD（車輛檢測器）的數據品質分析
- 訓練/驗證集分佈比較
- 數據洩漏檢測

**使用方法**：
```bash
python scripts/analysis/h5_data_analysis.py
```

**分析內容**：
1. **數據結構檢查**：
   - 數據形狀：`(4267, 3, 5)` - 時間樣本 x VD數量 x 特徵數
   - 特徵名稱：`avg_speed`, `total_volume`, `avg_occupancy` 等
   - VD 識別碼和時間戳

2. **數據品質分析**：
   - 有效數據率（排除 NaN, 0, 無限值）
   - 統計特徵（平均值、標準差、範圍）
   - 異常值檢測
   - 重複值檢測

3. **訓練/驗證集比較**：
   - 80/20 切分分析
   - 分佈差異計算
   - 數據洩漏檢測

**輸出**：
- 品質報告：控制台詳細分析
- 可視化：`blob/debug/data_quality_analysis.png`

### 3. 數據品質分析 (`data_quality_analysis.py`)

**用途**：數據清理和時間品質分析，提供數據健康狀況評估。

**主要功能**：
- 時間數據品質驗證
- 綜合數據健康檢查
- 品質評估報告
- 數據品質問題識別和修復建議

**使用方法**：
```bash
python scripts/analysis/data_quality_analysis.py
```

**分析維度**：
1. **時間一致性**：時間戳的連續性和完整性
2. **數值合理性**：交通參數的合理範圍檢查
3. **缺失數據分析**：缺失模式和影響評估
4. **異常檢測**：統計異常和系統性錯誤

## 驗證工具 (`scripts/validation/`)

### 1. 訓練驗證 (`training_validation.py`)

**用途**：最小化訓練測試，快速驗證過擬合修復效果。

**主要功能**：
- 快速訓練會話（8-10 個 epoch）
- 過擬合行為測試
- 訓練曲線可視化
- 模型配置變更的快速反饋

**使用方法**：
```bash
python scripts/validation/training_validation.py
```

**配置要求**：
- 配置文件：`cfgs/fixed/lstm_fixed.yaml`
- 數據集：由配置文件指定的 HDF5 文件

**輸出**：
- 訓練指標：最終訓練/驗證損失比例
- 可視化：`blob/debug/minimal_training_test.png`
- 結果文件：`blob/debug/minimal_training_results.json`

**品質評估標準**：
- 優秀：過擬合比例 < 3
- 良好：過擬合比例 < 8
- 中等：過擬合比例 < 20
- 差：過擬合比例 > 20

### 2. 過擬合驗證 (`overfitting_validation.py`)

**用途**：綜合過擬合修復效果測試，與原始問題進行對比。

**主要功能**：
- 運行快速訓練比較
- 分析訓練結果檢查過擬合改善
- 創建前後對比圖表
- 提供詳細的效果評估

**使用方法**：
```bash
python scripts/validation/overfitting_validation.py
```

**測試配置**：
- 主要測試：`cfgs/fixed/lstm_fixed.yaml`
- 可選測試：`cfgs/fixed/xlstm_fixed.yaml`

**比較基準**：
- 原始 LSTM：訓練/驗證比例 113.55
- 原始 xLSTM：訓練/驗證比例 98.98
- 目標：比例 < 5（優秀）或 < 10（可接受）

**輸出**：
- 對比圖表：`blob/debug/overfitting_fix_test.png`
- 詳細評估報告：控制台輸出
- 改善建議：基於結果的下一步建議

### 3. 時間切分驗證 (`temporal_split_validation.py`)

**用途**：驗證時間數據切分策略的有效性。

**主要功能**：
- 測試時間切分功能
- 比較時間 vs 隨機切分方法
- 驗證切分品質和分佈一致性
- 創建切分方法對比可視化

**使用方法**：
```bash
python scripts/validation/temporal_split_validation.py
```

**分析流程**：
1. **原始隨機切分測試**：
   - 簡單 80/20 比例切分
   - 分佈差異分析
   - 品質問題識別

2. **時間切分測試**：
   - 使用 `TemporalSplitter` 進行切分
   - 序列長度和預測長度配置
   - 間隔設置防止時間洩漏

3. **方法比較**：
   - 分佈差異改善度量
   - 品質檢查通過率
   - 改善因子計算

**輸出**：
- 對比圖表：`blob/debug/splitting_comparison.png`
- 驗證結果：`blob/debug/temporal_split_validation/`
- 改善度量：控制台詳細報告

## 通用工具 (`scripts/utils/`)

### HDF5 結構檢查器 (`h5_structure_inspector.py`)

**用途**：快速檢查 HDF5 文件結構和內容。

**主要功能**：
- 文件結構檢查
- 數據集形狀和類型顯示
- 群組和元數據檢查
- 調試數據載入問題

**使用方法**：
```bash
python scripts/utils/h5_structure_inspector.py
```

**輸出示例**：
```
🔍 Checking H5 file structure: blob/dataset/pre-processed/h5/traffic_features_default.h5

📁 Root level keys: ['data', 'metadata']

🏗️ Full structure:
📄 Dataset: data/features, shape: (4267, 3, 5), dtype: float64
📄 Dataset: metadata/feature_names, shape: (5,), dtype: |S20
📄 Dataset: metadata/timestamps, shape: (4267,), dtype: |S20
📄 Dataset: metadata/vdids, shape: (3,), dtype: |S20
```

## 最佳實踐

### 1. 數據分析工作流程

```bash
# 1. 首先檢查數據結構
python scripts/utils/h5_structure_inspector.py

# 2. 進行綜合數據分析
python scripts/analysis/h5_data_analysis.py

# 3. 如果發現時間問題，進行深度時間分析
python scripts/analysis/temporal_pattern_analysis.py

# 4. 評估數據品質
python scripts/analysis/data_quality_analysis.py
```

### 2. 模型驗證工作流程

```bash
# 1. 快速訓練測試
python scripts/validation/training_validation.py

# 2. 如果需要，進行過擬合綜合測試
python scripts/validation/overfitting_validation.py

# 3. 驗證數據切分策略
python scripts/validation/temporal_split_validation.py
```

### 3. 問題診斷工作流程

```bash
# 1. 數據問題診斷
python scripts/analysis/data_quality_analysis.py

# 2. 時間相關問題診斷
python scripts/analysis/temporal_pattern_analysis.py

# 3. 訓練問題診斷
python scripts/validation/training_validation.py
```

## 配置和依賴

### 環境要求
- Python 3.11+
- Conda 環境：`conda activate social_xlstm`
- 安裝包：`pip install -e .`

### 數據要求
- HDF5 數據集：`blob/dataset/pre-processed/h5/traffic_features_default.h5`
- 配置文件：`cfgs/fixed/` 目錄中的 YAML 文件

### 輸出目錄
- 調試輸出：`blob/debug/`
- 分析結果：`blob/analysis/`
- 實驗結果：`blob/experiments/dev/`

## 整合與 API

### 與主代碼庫整合
這些工具與主代碼庫中的整合功能互補：

- **數據穩定性**：`src/social_xlstm/dataset/storage/h5_converter.py`
- **診斷系統**：`src/social_xlstm/evaluation/evaluator.py`
- **配置生成**：`src/social_xlstm/dataset/core/processor.py`

### API 調用示例
```python
# 使用整合的數據穩定性功能
from social_xlstm.dataset.storage.h5_converter import TrafficFeatureExtractor

# 驗證數據品質
is_good, metrics = TrafficFeatureExtractor.validate_dataset_quality(
    "blob/dataset/pre-processed/h5/traffic_features_default.h5"
)

# 使用診斷系統
from social_xlstm.evaluation.evaluator import DatasetDiagnostics

diagnostics = DatasetDiagnostics()
result = diagnostics.analyze_h5_dataset(
    h5_path="blob/dataset/pre-processed/h5/traffic_features_default.h5",
    output_dir="blob/debug"
)
```

## 故障排除

### 常見問題
1. **導入錯誤**：確保已安裝 `social_xlstm` 包
2. **文件路徑錯誤**：檢查 HDF5 文件是否存在
3. **權限問題**：確保輸出目錄有寫入權限
4. **CUDA 錯誤**：檢查 GPU 可用性

### 調試技巧
1. 使用 `--help` 參數查看詳細用法
2. 檢查日誌輸出查找錯誤信息
3. 使用較小的數據集進行測試
4. 確保配置文件格式正確

## 相關文檔

- [ADR-0500: Scripts Directory Reorganization](../adr/0500-scripts-directory-reorganization.md)
- [Training System Guide](trainer_usage_guide.md)
- [LSTM Usage Guide](lstm_usage_guide.md)
- [Data Quality Documentation](../technical/data/data_quality.md)
- [Project Status](../reports/project_status.md)

---

最後更新：2025-07-15