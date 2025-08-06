# Configuration Files 🔧

此目錄包含 Social-xLSTM 專案的所有配置檔案，簡化為兩個主要配置文件。

## 目錄結構

```
cfgs/
├── README.md           # 配置說明文檔
└── snakemake/         # Snakemake 工作流程配置
    ├── default.yaml   # 預設/生產環境配置
    └── dev.yaml       # 開發環境配置
```

## 配置文件說明

### 預設配置 (`snakemake/default.yaml`)
- **用途**: 生產環境、正式實驗
- **特色**: 
  - 完整資料集處理
  - 正式實驗參數 (200 epochs)
  - 完整的模型訓練配置
  - 標準輸出目錄 (`blob/experiments/default/`)
  - 優化的訓練參數 (包含 early stopping、gradient clipping 等)

### 開發配置 (`snakemake/dev.yaml`)
- **用途**: 開發測試、快速驗證
- **特色**:
  - 小型資料集 (快速測試)
  - 短訓練週期 (LSTM: 50 epochs, xLSTM: 10 epochs)
  - 開發專用目錄 (`blob/experiments/dev/`)
  - 快速反饋循環
  - 完整的優化參數 (與生產環境相同的訓練邏輯)

## 整合的配置參數

兩個配置文件現已整合以下優化參數：

### 數據集配置
- **穩定數據集支援**: 包含 `h5_stable` 配置用於過擬合驗證
- **標準化**: 支援 standard normalization
- **數據分割**: 80/20 train/validation split
- **序列長度**: 開發環境使用較短序列，生產環境使用較長序列

### 訓練優化參數
- **Early Stopping**: `early_stopping_patience: 8`
- **Gradient Clipping**: `gradient_clip_value: 0.5`
- **Learning Rate Scheduler**: `use_scheduler: true`, `scheduler_patience: 5`
- **優化器參數**: `learning_rate: 0.0005`, `weight_decay: 0.01`
- **正則化**: 適當的 dropout 設定

### 模型配置
- **LSTM**: `hidden_size: 32`, `num_layers: 1`, `dropout: 0.5`
- **xLSTM**: 開發環境使用較小模型，生產環境使用較大模型
- **支援模型**: 同時支援 LSTM 和 xLSTM 架構

## 使用方式

### 開發環境 (推薦)
```bash
# 使用開發配置進行快速測試
snakemake --configfile cfgs/snakemake/dev.yaml train_single_vd_without_social_pooling --cores=4

# 數據處理
snakemake --configfile cfgs/snakemake/dev.yaml create_h5_file --cores=4
```

### 生產環境
```bash
# 使用生產配置進行完整實驗
snakemake --configfile cfgs/snakemake/default.yaml train_single_vd_without_social_pooling --cores=4

# 或使用預設配置 (自動使用 default.yaml)
snakemake train_single_vd_without_social_pooling --cores=4
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
      file: "..."           # 主要 H5 檔案路徑
      selected_vdids: [...] # 選定的 VD IDs
      time_range: "..."     # 時間範圍
    h5_stable:
      file: "..."           # 穩定版 H5 檔案路徑 (用於過擬合驗證)
```

### 3. Training Configuration
```yaml
training_lstm:
  single_vd:
    epochs: ...           # 訓練輪數
    batch_size: ...       # 批次大小
    select_vd_id: "..."   # 指定 VD ID
    # 完整的優化參數
    early_stopping_patience: 8
    gradient_clip_value: 0.5
    use_scheduler: true
    # 數據集參數
    prediction_length: 1
    train_ratio: 0.8
    val_ratio: 0.2
    normalize: true
    normalization_method: standard
```

## 配置最佳實踐

1. **開發時使用 `dev.yaml`**
   - 快速測試和驗證
   - 小資料集，快速反饋
   - 完整的優化參數

2. **生產時使用 `default.yaml`**
   - 完整實驗
   - 正式結果產出
   - 相同的優化參數但更長訓練時間

3. **配置統一性**
   - 兩個配置使用相同的優化參數
   - 只有訓練時間和資料集大小不同
   - 確保開發和生產環境的一致性

## 配置參數對比

| 參數 | 開發環境 | 生產環境 |
|------|----------|----------|
| **訓練輪數** | LSTM: 50, xLSTM: 10 | LSTM: 200, xLSTM: 200 |
| **批次大小** | 16 (LSTM), 4 (xLSTM) | 32 (LSTM), 32 (xLSTM) |
| **序列長度** | 5 | 20 |
| **時間範圍** | 1個月 | 3天 |
| **VD 數量** | 3 | 50 |
| **輸出目錄** | `blob/experiments/dev/` | `blob/experiments/default/` |
| **日誌目錄** | `logs/dev/` | `logs/default/` |
| **優化參數** | ✅ 完整 | ✅ 完整 |

## 相關文件
- [CLAUDE.md](/CLAUDE.md) - 專案開發指南
- [Quick Start Guide](/docs/QUICK_START.md) - 快速入門
- [Training Scripts Guide](/docs/guides/training_scripts_guide.md) - 訓練腳本使用
- [ADR-0500: Scripts Directory Reorganization](/docs/adr/0500-scripts-directory-reorganization.md) - 腳本重組決策