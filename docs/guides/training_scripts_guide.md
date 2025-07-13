# 訓練腳本使用說明

本指南說明 Social-xLSTM 專案的訓練腳本使用方法。專案採用專業化訓練架構，為不同的訓練場景提供專門的腳本。

## 📁 文件結構

```
scripts/train/
├── without_social_pooling/      # 無社交池化訓練腳本
│   ├── README.md               # 腳本說明
│   ├── train_single_vd.py      # 單VD訓練腳本
│   ├── train_multi_vd.py       # 多VD訓練腳本 (空間關係)
│   ├── train_independent_multi_vd.py  # 獨立多VD訓練腳本
│   └── common.py               # 共用函數
└── with_social_pooling/         # 社交池化訓練腳本 (開發中)
    └── README.md               # 待實現
```

## 🚀 快速開始

### 環境準備

```bash
# 激活 conda 環境（必須）
conda activate social_xlstm
```

### 1. 測試腳本是否正常

```bash
# 快速測試（推薦）
python scripts/train/test_training_scripts.py --quick

# 完整測試
python scripts/train/test_training_scripts.py --full
```

### 2. 單VD模型訓練

```bash
# 基本使用
python scripts/train/without_social_pooling/train_single_vd.py

# 自定義參數
python scripts/train/without_social_pooling/train_single_vd.py \
  --epochs 100 \
  --batch_size 64 \
  --hidden_size 256 \
  --experiment_name "my_single_vd_experiment"

# 指定數據路徑
python scripts/train/without_social_pooling/train_single_vd.py \
  --data_path blob/dataset/pre-processed/h5/traffic_features.h5
```

### 3. 多VD模型訓練

```bash
# 基本使用（空間關係）
python scripts/train/without_social_pooling/train_multi_vd.py

# 自定義參數
python scripts/train/without_social_pooling/train_multi_vd.py \
  --num_vds 5 \
  --epochs 100 \
  --batch_size 16 \
  --hidden_size 256 \
  --experiment_name "my_multi_vd_experiment"

# 指定特定VD
python scripts/train/without_social_pooling/train_multi_vd.py \
  --vd_ids "VD001,VD002,VD003,VD004,VD005"

# 啟用混合精度
python scripts/train/without_social_pooling/train_multi_vd.py \
  --mixed_precision
```

### 4. 獨立多VD訓練

```bash
# 基本使用（基準比較）
python scripts/train/without_social_pooling/train_independent_multi_vd.py

# 自定義參數
python scripts/train/without_social_pooling/train_independent_multi_vd.py \
  --num_vds 5 \
  --epochs 100 \
  --batch_size 32 \
  --hidden_size 128 \
  --experiment_name "my_independent_multi_vd_experiment"
```

### 5. 使用 Snakemake 執行訓練

```bash
# 使用預設配置
snakemake --cores 1

# 使用開發配置
snakemake --configfile cfgs/snakemake/dev.yaml --cores 1

# 單VD訓練
snakemake --configfile cfgs/snakemake/dev.yaml train_single_vd_without_social_pooling --cores 1

# 多VD訓練
snakemake --configfile cfgs/snakemake/dev.yaml train_multi_vd_without_social_pooling --cores 1

# 獨立多VD訓練
snakemake --configfile cfgs/snakemake/dev.yaml train_independent_multi_vd_without_social_pooling --cores 1

# 並行執行所有訓練
snakemake --configfile cfgs/snakemake/dev.yaml train_single_vd_without_social_pooling train_multi_vd_without_social_pooling train_independent_multi_vd_without_social_pooling --cores 3

# 強制重新執行（測試用）
snakemake --configfile cfgs/snakemake/dev.yaml --forceall --cores 1
```

## 📋 腳本詳細說明

### train_single_vd.py - 單VD訓練腳本

**功能**: 訓練單個VD的LSTM模型

**主要參數**:
- `--data_path`: 數據文件路徑
- `--epochs`: 訓練輪數 (預設: 50)
- `--batch_size`: 批次大小 (預設: 32)
- `--hidden_size`: 隱藏層大小 (預設: 128)
- `--num_layers`: LSTM層數 (預設: 2)
- `--learning_rate`: 學習率 (預設: 0.001)
- `--experiment_name`: 實驗名稱

**使用場景**:
- 單點交通預測
- 建立基準模型
- 算法驗證

### train_multi_vd.py - 多VD訓練腳本（空間關係）

**功能**: 訓練多個VD的LSTM模型，使用MultiVDTrainer處理空間關係

**主要參數**:
- `--num_vds`: VD數量 (預設: 5)
- `--vd_ids`: 指定VD IDs
- `--spatial_radius`: 空間半徑 (預設: 25000)
- `--batch_size`: 批次大小 (預設: 16，建議較小)
- `--hidden_size`: 隱藏層大小 (預設: 256，建議較大)
- `--mixed_precision`: 啟用混合精度

**使用場景**:
- 區域交通預測
- 學習空間關係
- Social Pooling基礎

### train_independent_multi_vd.py - 獨立多VD訓練腳本

**功能**: 訓練多個VD但使用獨立訓練策略，用於基準比較

**主要參數**:
- `--num_vds`: VD數量 (預設: 5)
- `--vd_ids`: 指定VD IDs
- `--batch_size`: 批次大小 (預設: 32)
- `--hidden_size`: 隱藏層大小 (預設: 128)
- `--experiment_name`: 實驗名稱

**使用場景**:
- 基準比較
- 獨立VD性能評估
- 對比空間關係的效果

### common.py - 共用函數模組

**功能**: 提供訓練腳本的共用函數和配置

**主要功能**:
- 配置管理和解析
- 模型創建和初始化
- 數據載入和處理
- 實驗目錄管理
- 日誌設置

**使用方式**:
```python
# 在訓練腳本中導入
from common import setup_experiment, create_model, load_data
```

## 📊 輸出結果

訓練完成後，結果保存在 `blob/experiments/實驗名稱/` 目錄：

```
blob/experiments/my_experiment/
├── config.json              # 配置文件
├── training_record.json     # 完整訓練記錄
├── training_curves.png      # 訓練曲線
├── predictions.png          # 預測結果
├── test_evaluation.json     # 測試評估
├── best_model.pt           # 最佳模型
├── checkpoint_epoch_*.pt   # 檢查點文件
└── logs/                   # 訓練日誌
    ├── training.log
    └── error.log
```

## 🔧 常見問題

### Q1: 數據文件不存在
**錯誤**: `數據文件不存在: blob/dataset/pre-processed/h5/traffic_features.h5`

**解決方案**:
```bash
# 激活環境並執行數據預處理
conda activate social_xlstm
snakemake --configfile cfgs/snakemake/dev.yaml --cores 4

# 或手動執行
python scripts/dataset/pre-process/create_h5_file.py \
  --source_dir blob/dataset/pre-processed/unzip_to_json \
  --output_path blob/dataset/pre-processed/h5/traffic_features_dev.h5
```

### Q2: 記憶體不足
**錯誤**: `CUDA out of memory`

**解決方案**:
```bash
# 單VD訓練：降低批次大小
python scripts/train/without_social_pooling/train_single_vd.py --batch_size 16

# 多VD訓練：降低批次大小並啟用混合精度
python scripts/train/without_social_pooling/train_multi_vd.py --batch_size 8 --mixed_precision

# 獨立多VD訓練：降低批次大小
python scripts/train/without_social_pooling/train_independent_multi_vd.py --batch_size 16
```

### Q3: 訓練速度太慢
**解決方案**:
```bash
# 啟用混合精度
python scripts/train/without_social_pooling/train_single_vd.py --mixed_precision

# 使用更大的批次大小
python scripts/train/without_social_pooling/train_single_vd.py --batch_size 64
```

### Q4: 模型不收斂
**解決方案**:
```bash
# 降低學習率
python scripts/train/without_social_pooling/train_single_vd.py --learning_rate 0.0001

# 增加早停耐心
python scripts/train/without_social_pooling/train_single_vd.py --early_stopping_patience 25
```

### Q5: Snakemake 訓練失敗
**錯誤**: `Snakemake 執行訓練規則失敗`

**解決方案**:
```bash
# 檢查配置文件
cat cfgs/snakemake/dev.yaml

# 單獨測試訓練腳本
python scripts/train/without_social_pooling/train_single_vd.py --epochs 2

# 查看 Snakemake 日誌
snakemake --configfile cfgs/snakemake/dev.yaml train_single_vd_without_social_pooling --cores 1 --verbose

# 測試乾運行
snakemake --configfile cfgs/snakemake/dev.yaml --dry-run --cores 1
```

### Q6: Conda 環境問題
**錯誤**: `未檢測到 conda 環境`

**解決方案**:
```bash
# 確認並激活環境
conda env list
conda activate social_xlstm

# 驗證環境
python -c "import torch; print('PyTorch:', torch.__version__)"

# 如果環境不存在，重新創建
conda env create -f environment.yaml
```

### Q7: 訓練器不匹配錯誤
**錯誤**: `訓練器與模型配置不匹配`

**解決方案**:
```bash
# 確認使用正確的訓練器
# 單VD: SingleVDTrainer
# 多VD（空間關係）: MultiVDTrainer  
# 獨立多VD: IndependentMultiVDTrainer

# 查看訓練器使用情況
grep -n "Trainer" scripts/train/without_social_pooling/train_*.py
```

## 🔗 相關文檔

- [統一訓練系統使用指南](../../docs/guides/trainer_usage_guide.md)
- [LSTM 使用指南](../../docs/guides/lstm_usage_guide.md)
- [模組功能說明](../../docs/implementation/modules.md)

## 📞 技術支援

如果遇到問題：

1. 先確認 conda 環境已正確激活
2. 運行測試腳本確認環境正常
3. 查看相關文檔
4. 檢查日誌輸出
5. 提交 GitHub Issue

---

**提醒**: 
- 首次使用建議先運行測試腳本
- 多VD訓練建議使用GPU
- 實驗結果會自動保存，請確保磁碟空間充足
- 所有腳本都要求在 conda 環境中運行