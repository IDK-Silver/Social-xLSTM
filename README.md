# Social-xLSTM

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/Tests-189%2F189%20Passing-green.svg)](tests/)
[![Code Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen.svg)](tests/)

> 🚀 **基於 Social Pooling 與 xLSTM 的無拓撲依賴交通流量預測系統**

Social-xLSTM 結合座標驅動的社會聚合機制與擴展長短期記憶網路，實現在缺乏完整道路拓撲資訊的情況下進行高精度交通預測。

## 🚀 快速開始

### 環境設置

```bash
# 創建 conda 環境
conda env create -f environment.yaml
conda activate social_xlstm

# 安裝項目
pip install -e .
```

### PEMS-BAY 數據集訓練

使用 PEMS-BAY 數據集進行 Social-xLSTM 訓練：

```bash
# 使用預設輸出位置
python scripts/train/with_social_pooling/train_multi_vd.py \
  --config cfgs/profiles/pems_bay_dev.yaml

# 指定自定義輸出位置
python scripts/train/with_social_pooling/train_multi_vd.py \
  --config cfgs/profiles/pems_bay_dev.yaml \
  --output_dir blob/experiments/soical_pooling/xlstm/
```

**輸出文件結構**：
- Lightning 日誌: `{output_dir}/lightning_logs/version_X/`
- 指標文件: `{output_dir}/metrics/metrics.csv`
- 模型檢查點: `{output_dir}/lightning_logs/version_X/checkpoints/`

### Taiwan VD 數據集訓練

```bash
# 使用預設輸出位置
python scripts/train/with_social_pooling/train_multi_vd.py \
  --config cfgs/profiles/taiwan_vd_dev.yaml

# 指定自定義輸出位置
python scripts/train/with_social_pooling/train_multi_vd.py \
  --config cfgs/profiles/taiwan_vd_dev.yaml \
  --output_dir /path/to/my/experiments
```

## 📊 數據集支持

### PEMS-BAY
- **特徵數量**: 6個 (avg_speed, lanes, length, latitude, longitude, direction)
- **數據位置**: `blob/dataset/processed/pems_bay.h5`
- **批次大小**: 16 (針對較大特徵集優化)

### Taiwan VD
- **特徵數量**: 3個 (avg_speed, total_volume, avg_occupancy) 
- **數據位置**: `blob/dataset/processed/taiwan_vd.h5`
- **批次大小**: 8 (預設配置)

## 🔪 HDF5 時間分割工具

為了快速驗證訓練優化效果，項目提供通用的 HDF5 時間分割腳本，可從完整數據集創建小型測試數據集。

### 基本使用

```bash
# 創建 150 個時間步的測試數據集 (確保驗證集有足夠樣本)
python scripts/utils/h5_time_slice.py \
  --input blob/dataset/processed/pems_bay.h5 \
  --output blob/dataset/processed/pems_bay_fast_test.h5 \
  --start-index 0 --length 150 \
  --progress --atomic
```

### 進階選項

```bash
# 使用時間範圍切分（需要時間戳）
python scripts/utils/h5_time_slice.py \
  --input blob/dataset/processed/pems_bay.h5 \
  --output blob/dataset/processed/pems_bay_custom.h5 \
  --start-time "2017-01-01 00:00:00" \
  --end-time "2017-01-02 00:00:00" \
  --progress --atomic

# 使用自定義時間戳數據集路徑
python scripts/utils/h5_time_slice.py \
  --input your_data.h5 \
  --output your_test_data.h5 \
  --timestamp-dset "metadata/custom_timestamps" \
  --start-index 0 --length 50
```

### 快速測試工作流

項目提供兩種快速測試方案，適用於不同的優化需求：

#### 🚀 超快速測試（10-VD，20秒完成）

適用於算法邏輯驗證和快速調試：

```bash
# 1. 創建測試數據集（150 個時間步確保所有數據分割都有樣本）
python scripts/utils/h5_time_slice.py \
  --input blob/dataset/processed/pems_bay.h5 \
  --output blob/dataset/processed/pems_bay_fast_test.h5 \
  --start-index 0 --length 150 --progress --atomic

# 2. 使用 10-VD 超快速測試 Profile（約 20 秒完成）
python scripts/train/with_social_pooling/train_multi_vd.py \
  --config cfgs/profiles/pems_bay_10vd_fast.yaml \
  --output_dir blob/experiments/ultra_fast_10vd

# 3. 生成指標圖表
python scripts/utils/generate_metrics_plots.py \
  --csv_path blob/experiments/ultra_fast_10vd/metrics/metrics.csv
```

**特色**：
- ⚡ **97% 記憶體減少** - 325 個 VD → 10 個代表性高質量 VD
- 🚀 **6-8倍速度提升** - 從 2-3 分鐘縮短到 20 秒
- 🎯 **智能 VD 選擇** - 基於數據質量和地理分布的代表性採樣

#### 🔧 標準快速測試（全 VD，2-3分鐘完成）

適用於完整功能驗證：

```bash
# 使用標準快速測試 Profile 進行訓練（約 2-3 分鐘完成）
python scripts/train/with_social_pooling/train_multi_vd.py \
  --config cfgs/profiles/pems_bay_fast_test.yaml \
  --output_dir blob/experiments/fast_test

# 比較結果並迭代優化
python scripts/utils/generate_metrics_plots.py \
  --experiment_dir blob/experiments/fast_test/metrics
```

### 支持功能

- ✅ **索引範圍切分** - 指定起始索引和長度
- ✅ **時間範圍切分** - 使用時間戳進行精確切分  
- ✅ **元數據保持** - 完整保留原始文件的元數據和屬性
- ✅ **自動塊調整** - 智能調整 HDF5 塊大小以適應新維度
- ✅ **進度顯示** - 實時顯示切分進度
- ✅ **原子操作** - 使用臨時文件確保操作安全性
- ✅ **格式檢測** - 自動檢測時間戳格式（字符串/數字）

### ⚠️ 重要注意事項

**時間步數選擇**：確保切分後的數據集有足夠的樣本用於訓練、驗證和測試分割。

- **最小需求**：`(sequence_length + prediction_length) × 3` ≈ 45 個時間步
- **推薦大小**：150+ 個時間步，確保每個分割都有足夠的樣本
- **PEMS-BAY 配置**：sequence_length=12, prediction_length=3，所以需要 15 個時間步創建 1 個樣本

**GPU 記憶體優化**：
- 保持 `batch_size: 16` 以避免 CUDA OOM（針對 325 個 VD 的 PEMS-BAY）
- 較小的批次大小可避免創建過多 xLSTM 實例導致的記憶體問題

## 📈 指標記錄與可視化

項目內建輕量級指標記錄系統，自動記錄 MAE、MSE、RMSE、R² 四個核心指標。

**特色功能**：
- ✅ **數據持久化** - 支持後續重新繪圖，無需重新訓練
- ✅ **分散式安全** - 支持 DDP 分散式訓練
- ✅ **Lightning 整合** - 無縫整合 PyTorch Lightning 框架
- ✅ **輕量設計** - 遵循 YAGNI 原則，避免過度設計

### 生成訓練圖表

```bash
# 從實驗目錄生成圖表
python scripts/utils/generate_metrics_plots.py \
  --experiment_dir ./lightning_logs/version_0

# 直接從 CSV 生成圖表
python scripts/utils/generate_metrics_plots.py \
  --csv_path ./path/to/metrics.csv --output_dir ./plots
```

### 輸出文件
- `metrics.csv` - 詳細的 epoch 級指標數據
- `metrics_summary.json` - 訓練摘要和最終指標
- `plots/` - 自動生成的可視化圖表

## 🔧 配置系統

### Profile-based 配置

使用 `cfgs/profiles/` 中的預設配置快速開始：

#### 🏭 生產環境配置
- `pems_bay_dev.yaml` - PEMS-BAY 完整開發配置 (325 VDs)
- `taiwan_vd_dev.yaml` - Taiwan VD 完整開發配置

#### ⚡ 快速測試配置  
- `pems_bay_fast_test.yaml` - 快速測試配置 (325 VDs, 150 時間步, ~2-3 分鐘)
- `pems_bay_10vd_fast.yaml` - **超快速測試配置** (10 VDs, 150 時間步, ~20 秒) ⭐

#### 配置選擇指南
| 用途 | Profile | VDs | 時間 | 適用場景 |
|------|---------|-----|------|----------|
| 生產訓練 | `pems_bay_dev.yaml` | 325 | 15-30 分鐘 | 完整模型訓練、論文結果 |
| 功能驗證 | `pems_bay_fast_test.yaml` | 325 | 2-3 分鐘 | 完整流程測試、參數調優 |
| 算法調試 | `pems_bay_10vd_fast.yaml` | 10 | 20 秒 | 快速驗證、代碼調試 ⚡ |

### 自定義配置

配置系統支持模組化 YAML 合併，可參考 `cfgs/` 目錄中的範例配置。

## 🗂️ 項目結構

```
Social-xLSTM/
├── src/social_xlstm/           # 核心源代碼
│   ├── models/                 # 模型實現（xLSTM, Social Pooling）
│   ├── dataset/                # 數據處理和載入
│   ├── metrics/                # 輕量級指標記錄系統
│   ├── training/               # 訓練框架
│   └── deprecated/             # 已廢棄的複雜系統
├── scripts/                    # 訓練和工具腳本
│   ├── train/with_social_pooling/
│   └── utils/                  # 可視化和分析工具
├── cfgs/                       # 配置文件
│   └── profiles/               # 數據集特定配置
├── blob/dataset/               # 數據存儲（HDF5 格式）
└── docs/                       # 文檔
```

## 📚 文檔

- [快速開始指南](docs/guides/quickstart/) - 15分鐘建立第一個模型
- [訓練指南](docs/guides/training-with-sp.md) - 詳細訓練流程  
- [配置指南](docs/reference/configuration-guide.md) - 配置系統說明
- [API 參考](docs/reference/api-reference.md) - 完整 API 文檔

## 🚧 系統要求

- Python 3.11+
- PyTorch 2.0+
- CUDA 12.4 (GPU 訓練)
- 16GB+ RAM (推薦)

## 📄 許可證

MIT License - 詳見 [LICENSE](LICENSE) 文件

---

**基於 YAGNI 原則的現代化架構** | **支持 PEMS-BAY 和 Taiwan VD 數據集** | **輕量級指標系統**
