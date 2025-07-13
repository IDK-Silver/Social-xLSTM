# Configuration Files 🔧

此目錄包含 Social-xLSTM 專案的所有配置檔案，按類型和用途分類組織。

## 目錄結構

```
cfgs/
├── README.md           # 配置說明文檔
└── snakemake/         # Snakemake 工作流程配置
    ├── default.yaml   # 預設/生產環境配置
    └── dev.yaml       # 開發環境配置
```

## Snakemake 配置

### 預設配置 (`snakemake/default.yaml`)
- **用途**: 生產環境、正式實驗
- **特色**: 
  - 完整資料集處理
  - 正式實驗參數
  - 完整的模型訓練配置
  - 標準輸出目錄

### 開發配置 (`snakemake/dev.yaml`)
- **用途**: 開發測試、快速驗證
- **特色**:
  - 小型資料集 (快速測試)
  - 短訓練週期 (2 epochs)
  - 開發專用目錄 (`blob/experiments/dev/`)
  - 快速反饋循環

### 生產配置 (`snakemake/default.yaml`)
- **用途**: 正式實驗、完整訓練
- **特色**:
  - 完整資料集
  - 完整訓練週期 (5+ epochs)  
  - 生產專用目錄 (`blob/experiments/default/`)
  - 正式實驗結果

## 使用方式

### 使用生產配置
```bash
# 默認使用 cfgs/snakemake/default.yaml
snakemake train_single_vd_without_social_pooling --cores=4

# 或明確指定
snakemake --config configfile=cfgs/snakemake/default.yaml train_single_vd_without_social_pooling --cores=4
```

### 使用開發配置
```bash
# 使用開發助手腳本 (推薦)
python run_dev.py train_single_vd_without_social_pooling --cores=1

# 或直接使用 Snakemake
snakemake --config configfile=cfgs/snakemake/dev.yaml train_single_vd_without_social_pooling --cores=1
```

## 配置結構

兩個配置檔案都包含以下主要區塊：

### 1. Storage Configuration
```yaml
storage:
  cold_storage:
    raw_zip:
      folders: [...] # 原始資料路徑
```

### 2. Dataset Configuration
```yaml
dataset:
  pre-processed:
    h5:
      file: "..."           # H5 檔案路徑
      selected_vdids: [...] # 選定的 VD IDs
      time_range: "..."     # 時間範圍
```

### 3. Training Configuration
```yaml
training:
  single_vd:
    epochs: ...           # 訓練輪數
    batch_size: ...       # 批次大小
    select_vd_id: "..."   # 指定 VD ID
  multi_vd:
    num_vds: ...          # VD 數量
```

## 配置最佳實踐

1. **開發時使用 `snakemake/dev.yaml`**
   - 快速測試和驗證
   - 小資料集，快速反饋

2. **生產時使用 `snakemake/default.yaml`**
   - 完整實驗
   - 正式結果產出

3. **新增配置選項時**
   - 同時更新兩個配置檔案
   - 保持結構一致性
   - 在此 README 中記錄說明

## 配置參數說明

### 開發配置特色 (`snakemake/dev.yaml`)
- `epochs: 2` - 快速訓練
- `batch_size: 4` - 小批次大小
- `time_range: "2025-03-18_00-49-00,2025-03-18_02-00-00"` - 約1小時資料
- `selected_vdids: 3個` - 限制 VD 數量
- 所有輸出到 `blob/experiments/dev/` 和 `logs/dev/`

### 生產配置特色 (`snakemake/default.yaml`)
- `epochs: 5` - 完整訓練
- `batch_size: 1` - 標準批次大小
- `time_range: null` - 使用完整資料集
- `selected_vdids: null` - 使用所有 VD
- 所有輸出到 `blob/experiments/default/` 和 `logs/default/`

## 相關文件
- [CLAUDE.md](/CLAUDE.md) - 專案開發指南
- [Quick Start Guide](/docs/QUICK_START.md) - 快速入門
- [Training Scripts Guide](/docs/guides/training_scripts_guide.md) - 訓練腳本使用